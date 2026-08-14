"""SHORT-trade post-entry trend-persistence audit (research/analysis only).

Answers one question only: what market-state characteristics -- measured
from already-existing OHLCV data at a fixed checkpoint ladder
(4h/12h/24h/48h/3d/7d/14d/21d/30d/45d) -- distinguish the 3 previously
identified persistent SHORT winners (structurally valid for 40-60 days)
from the 5 ordinary SHORT winners and 13 SHORT losers? This module never
runs a backtest, never touches `TrendFollowCore.py`, config, or the pair
whitelist, never redefines the persistent/ordinary grouping, and never
searches for or proposes a new threshold.

This is a pure-function, Freqtrade-independent research layer over an
already-persisted trade list plus already-existing OHLCV data.

Design decisions
-----------------
* **EMA200/ADX14/Donchian20 reuse the same independent pandas
  reimplementation already established in three prior audits**
  (`hermes.short_persistence_audit`, `hermes.short_runner_lifecycle_audit`,
  `hermes.short_ema_exit_attribution_audit`) -- same periods, same
  conventions, not a fourth separately-written version.
* **The checkpoint ladder is fixed and literal** (`CHECKPOINTS`), taken
  directly from this block's own required list -- never tuned, added to,
  or removed from based on results.
* **A checkpoint reached after the trade's own exit is flagged
  `closed_before_checkpoint=True`**, and its metrics are computed from
  the trade's actual (shorter) window -- never fabricated by
  extrapolating past the real exit. This is the same discipline already
  used in `hermes.short_runner_lifecycle_audit`.
* **Trend-persistence run-length statistics (`TrendPersistenceStats`) are
  new to this module** -- percentage of candles since entry that were
  SHORT-aligned / below EMA200 / ADX>25, plus the longest consecutive run
  of each -- computed over the candles from entry through the checkpoint
  (inclusive), never using a candle after the checkpoint or after the
  trade's own exit.
* **`ADX_ENTRY_THRESHOLD = 25.0` is the strategy's existing, already-coded
  threshold** (`TrendFollowCore.ADX_THRESHOLD`), reused verbatim -- not
  re-derived or re-tuned by this module.
* **Persistent/ordinary/loser classification reuses the exact
  `(pair, entry_time)` matching already established in
  `hermes.short_ema_exit_attribution_audit`** -- never re-ranked or
  re-derived here.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Sequence

import pandas as pd

from hermes.trade_report import Trade

LONG = "LONG"
SHORT = "SHORT"

EMA_PERIOD = 200
ADX_PERIOD = 14
DONCHIAN_PERIOD = 20
ADX_ENTRY_THRESHOLD = 25.0  # TrendFollowCore.ADX_THRESHOLD, reused verbatim

THIN_SAMPLE_THRESHOLD = 5

PERSISTENT_KEYS: frozenset[tuple[str, str]] = frozenset({
    ("ETH/USDC:USDC", "2026-01-20 16:00:00+00:00"),
    ("ETH/USDC:USDC", "2026-05-15 16:00:00+00:00"),
    ("BTC/USDC:USDC", "2026-01-20 08:00:00+00:00"),
})

# Fixed, literal checkpoint ladder (minutes from entry) -- from the
# block's own required list. Never searched or extended based on results.
CHECKPOINTS: dict[str, float] = {
    "4h": 240.0,
    "12h": 720.0,
    "24h": 1440.0,
    "48h": 2880.0,
    "3d": 4320.0,
    "7d": 10080.0,
    "14d": 20160.0,
    "21d": 30240.0,
    "30d": 43200.0,
    "45d": 64800.0,
}


def classify_group(trade: Trade) -> str:
    """`"PERSISTENT"` / `"ORDINARY"` / `"LOSER"` for a SHORT trade, by
    exact `(pair, entry_time)` match against `PERSISTENT_KEYS` for
    winners, and `is_winner` otherwise. Never applied to a non-SHORT
    trade's classification by this function's callers."""
    if trade.is_winner is False:
        return "LOSER"
    key = (trade.pair, str(trade.entry_time))
    return "PERSISTENT" if key in PERSISTENT_KEYS else "ORDINARY"


def ema(series: pd.Series, period: int = EMA_PERIOD) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move, down_move = high.diff(), -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def donchian_prev_bounds(df: pd.DataFrame, period: int = DONCHIAN_PERIOD) -> tuple[pd.Series, pd.Series]:
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    return prev_high.rolling(window=period).max(), prev_low.rolling(window=period).min()


# ---------------------------------------------------------------------------
# Full trade reconstruction (entry through exit, one row per candle)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradeCandle:
    date: pd.Timestamp
    close: float
    ema200: float | None
    ema_distance_pct: float | None
    adx: float | None
    donchian_breakout_pct: float | None
    realized_vol_pct: float | None
    price_change_pct: float
    cumulative_mfe_pct: float
    cumulative_mae_pct: float
    structurally_aligned: bool | None
    adx_above_threshold: bool | None
    days_since_entry: float


def reconstruct_full_trade(
    ohlcv: pd.DataFrame | None, entry_time, exit_time, entry_price: float | None, direction: str | None,
) -> list[TradeCandle]:
    """Every candle from `entry_time` through `exit_time` inclusive
    (never later), each carrying EMA200/ADX14/Donchian20 computed from
    all history up to and including that candle (never a later one), a
    trailing realized-volatility figure (std of returns over the last up
    to 20 candles ending at this candle, within the trade window only),
    cumulative MFE/MAE, and the SHORT/LONG-aligned + ADX>threshold flags.
    Returns an empty list if inputs are missing or the window can't be
    sliced."""
    if ohlcv is None or ohlcv.empty or entry_price is None or direction not in (LONG, SHORT):
        return []
    try:
        entry_ts = pd.Timestamp(entry_time)
        exit_ts = pd.Timestamp(exit_time) if exit_time is not None else None
    except (ValueError, TypeError):
        return []
    entry_ts = entry_ts.tz_localize("UTC") if entry_ts.tzinfo is None else entry_ts.tz_convert("UTC")
    if exit_ts is not None:
        exit_ts = exit_ts.tz_localize("UTC") if exit_ts.tzinfo is None else exit_ts.tz_convert("UTC")
        if exit_ts < entry_ts:
            return []

    full_history = ohlcv.sort_values("date").reset_index(drop=True)
    mask = full_history["date"] >= entry_ts
    if exit_ts is not None:
        mask &= full_history["date"] <= exit_ts
    window = full_history.loc[mask]
    if window.empty:
        return []

    records: list[TradeCandle] = []
    cum_mfe, cum_mae = 0.0, 0.0
    trade_closes: list[float] = []
    for _, row in window.iterrows():
        up_to = full_history.loc[full_history["date"] <= row["date"]]
        ema200_series = ema(up_to["close"], EMA_PERIOD)
        adx_series = adx(up_to, ADX_PERIOD)
        upper, lower = donchian_prev_bounds(up_to, DONCHIAN_PERIOD)
        ema_value = ema200_series.iloc[-1]
        adx_value = adx_series.iloc[-1]
        upper_value, lower_value = upper.iloc[-1], lower.iloc[-1]

        close, high, low = float(row["close"]), float(row["high"]), float(row["low"])
        trade_closes.append(close)

        if direction == SHORT:
            favorable = (entry_price - low) / entry_price * 100.0
            adverse = (high - entry_price) / entry_price * 100.0
        else:
            favorable = (high - entry_price) / entry_price * 100.0
            adverse = (entry_price - low) / entry_price * 100.0
        cum_mfe = max(cum_mfe, favorable)
        cum_mae = max(cum_mae, adverse)

        ema_distance = None
        aligned = None
        if pd.notna(ema_value) and ema_value:
            ema_distance = 100.0 * (close - ema_value) / ema_value
            aligned = bool(close < ema_value) if direction == SHORT else bool(close > ema_value)

        breakout = None
        if direction == SHORT and pd.notna(lower_value) and lower_value:
            breakout = 100.0 * (lower_value - close) / lower_value
        elif direction == LONG and pd.notna(upper_value) and upper_value:
            breakout = 100.0 * (close - upper_value) / upper_value

        recent_closes = trade_closes[-21:]
        realized_vol = None
        if len(recent_closes) >= 3:
            s = pd.Series(recent_closes)
            returns = s.pct_change().dropna()
            if len(returns) >= 2:
                realized_vol = float(returns.std()) * 100.0

        price_change_pct = 100.0 * (close - entry_price) / entry_price
        days_since_entry = (row["date"] - entry_ts).total_seconds() / 86400.0

        records.append(TradeCandle(
            date=row["date"], close=close,
            ema200=(float(ema_value) if pd.notna(ema_value) else None),
            ema_distance_pct=ema_distance,
            adx=(float(adx_value) if pd.notna(adx_value) else None),
            donchian_breakout_pct=breakout,
            realized_vol_pct=realized_vol,
            price_change_pct=price_change_pct,
            cumulative_mfe_pct=cum_mfe,
            cumulative_mae_pct=cum_mae,
            structurally_aligned=aligned,
            adx_above_threshold=(bool(adx_value > ADX_ENTRY_THRESHOLD) if pd.notna(adx_value) else None),
            days_since_entry=days_since_entry,
        ))
    return records


# ---------------------------------------------------------------------------
# Checkpoint slicing + trend-persistence run-length stats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointSnapshot:
    checkpoint_label: str
    closed_before_checkpoint: bool
    n_candles_in_subset: int
    mean_ema_distance_pct: float | None
    mean_adx: float | None
    mean_donchian_breakout_pct: float | None
    mean_realized_vol_pct: float | None
    price_change_pct: float | None
    mfe_pct: float | None
    mae_pct: float | None
    pct_structurally_aligned: float | None
    pct_adx_above_threshold: float | None


@dataclass(frozen=True)
class TrendPersistenceStats:
    pct_aligned: float | None
    pct_below_ema: float | None
    pct_adx_above_threshold: float | None
    longest_run_aligned: int
    longest_run_below_ema: int
    longest_run_adx_above_threshold: int


def _longest_run(flags: Sequence[bool | None]) -> int:
    """Longest consecutive run of `True` in `flags`, treating `None` as a
    break (never counted as `True`)."""
    longest = current = 0
    for f in flags:
        if f is True:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _pct_true(flags: Sequence[bool | None]) -> float | None:
    resolved = [f for f in flags if f is not None]
    return (100.0 * sum(1 for f in resolved if f) / len(resolved)) if resolved else None


def compute_trend_persistence(candles: Sequence[TradeCandle], direction: str) -> TrendPersistenceStats:
    """Run-length/percentage trend-persistence stats over `candles`
    (already sliced to entry-through-checkpoint or entry-through-exit).
    `below_ema` is direction-aware: for SHORT it's `close < ema200`
    (already captured in `structurally_aligned` for SHORT trades), for
    LONG it's `close > ema200`. This module is used only for SHORT trades
    by its callers, but the function itself is direction-agnostic."""
    aligned_flags = [c.structurally_aligned for c in candles]
    adx_flags = [c.adx_above_threshold for c in candles]
    return TrendPersistenceStats(
        pct_aligned=_pct_true(aligned_flags),
        pct_below_ema=_pct_true(aligned_flags),  # same flag for SHORT: aligned == below EMA200
        pct_adx_above_threshold=_pct_true(adx_flags),
        longest_run_aligned=_longest_run(aligned_flags),
        longest_run_below_ema=_longest_run(aligned_flags),
        longest_run_adx_above_threshold=_longest_run(adx_flags),
    )


def checkpoint_snapshot(
    full_trade_candles: Sequence[TradeCandle], checkpoint_label: str, checkpoint_minutes: float
) -> CheckpointSnapshot:
    """The metrics table row for one checkpoint, sliced from an
    already-reconstructed full-trade candle list (`reconstruct_full_trade`).
    `closed_before_checkpoint=True` when the trade's own last candle is
    before the checkpoint horizon -- the subset then simply contains every
    candle the trade actually had, never a fabricated extension."""
    if not full_trade_candles:
        return CheckpointSnapshot(
            checkpoint_label=checkpoint_label, closed_before_checkpoint=False, n_candles_in_subset=0,
            mean_ema_distance_pct=None, mean_adx=None, mean_donchian_breakout_pct=None,
            mean_realized_vol_pct=None, price_change_pct=None, mfe_pct=None, mae_pct=None,
            pct_structurally_aligned=None, pct_adx_above_threshold=None,
        )
    checkpoint_days = checkpoint_minutes / 1440.0
    subset = [c for c in full_trade_candles if c.days_since_entry <= checkpoint_days]
    if not subset:
        subset = [full_trade_candles[0]]
    closed_before = full_trade_candles[-1].days_since_entry < checkpoint_days

    def _mean_or_none(values):
        resolved = [v for v in values if v is not None]
        return mean(resolved) if resolved else None

    last = subset[-1]
    return CheckpointSnapshot(
        checkpoint_label=checkpoint_label,
        closed_before_checkpoint=closed_before,
        n_candles_in_subset=len(subset),
        mean_ema_distance_pct=_mean_or_none([c.ema_distance_pct for c in subset]),
        mean_adx=_mean_or_none([c.adx for c in subset]),
        mean_donchian_breakout_pct=_mean_or_none([c.donchian_breakout_pct for c in subset]),
        mean_realized_vol_pct=_mean_or_none([c.realized_vol_pct for c in subset]),
        price_change_pct=last.price_change_pct,
        mfe_pct=last.cumulative_mfe_pct,
        mae_pct=last.cumulative_mae_pct,
        pct_structurally_aligned=_pct_true([c.structurally_aligned for c in subset]),
        pct_adx_above_threshold=_pct_true([c.adx_above_threshold for c in subset]),
    )


def all_checkpoint_snapshots(
    full_trade_candles: Sequence[TradeCandle], checkpoints: dict[str, float] = CHECKPOINTS
) -> dict[str, CheckpointSnapshot]:
    """`checkpoint_snapshot` for every entry in `checkpoints`."""
    return {label: checkpoint_snapshot(full_trade_candles, label, minutes) for label, minutes in checkpoints.items()}


# ---------------------------------------------------------------------------
# Group-level aggregation across trades at one checkpoint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupCheckpointAggregate:
    checkpoint_label: str
    n_total: int
    n_reached: int
    is_thin_sample: bool
    mean_ema_distance_pct: float | None
    mean_adx: float | None
    mean_donchian_breakout_pct: float | None
    mean_realized_vol_pct: float | None
    mean_mfe_pct: float | None
    mean_mae_pct: float | None
    pct_structurally_aligned: float | None
    pct_adx_above_threshold: float | None


def aggregate_group_checkpoint(
    per_trade_snapshots: Sequence[dict[str, CheckpointSnapshot]], checkpoint_label: str
) -> GroupCheckpointAggregate:
    """Group-level aggregate for `checkpoint_label` across
    `per_trade_snapshots` (one `all_checkpoint_snapshots` result per
    trade). A trade with zero candles in its subset at this checkpoint
    contributes to `n_total` but not `n_reached` or any mean."""
    entries = [s[checkpoint_label] for s in per_trade_snapshots if checkpoint_label in s]
    reached = [e for e in entries if e.n_candles_in_subset > 0]

    def _mean_or_none(values):
        resolved = [v for v in values if v is not None]
        return mean(resolved) if resolved else None

    def _pct_or_none(values):
        resolved = [v for v in values if v is not None]
        return mean(resolved) if resolved else None

    return GroupCheckpointAggregate(
        checkpoint_label=checkpoint_label,
        n_total=len(entries),
        n_reached=len(reached),
        is_thin_sample=(len(reached) < THIN_SAMPLE_THRESHOLD),
        mean_ema_distance_pct=_mean_or_none([e.mean_ema_distance_pct for e in reached]),
        mean_adx=_mean_or_none([e.mean_adx for e in reached]),
        mean_donchian_breakout_pct=_mean_or_none([e.mean_donchian_breakout_pct for e in reached]),
        mean_realized_vol_pct=_mean_or_none([e.mean_realized_vol_pct for e in reached]),
        mean_mfe_pct=_mean_or_none([e.mfe_pct for e in reached]),
        mean_mae_pct=_mean_or_none([e.mae_pct for e in reached]),
        pct_structurally_aligned=_pct_or_none([e.pct_structurally_aligned for e in reached]),
        pct_adx_above_threshold=_pct_or_none([e.pct_adx_above_threshold for e in reached]),
    )
