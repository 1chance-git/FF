"""SHORT-trade post-entry lifecycle audit (research/analysis only).

Answers one question only: once a SHORT trade enters, do the previously
identified persistent/extreme SHORT winners already look measurably
different from ordinary SHORT winners and SHORT losers within the first
several candles/days after entry -- before the eventual large move
becomes obvious -- or does the difference remain indistinguishable that
early?

This module never runs a backtest, never touches `TrendFollowCore.py`,
config, or the pair whitelist, and never searches for or tunes a new
"persistent winner predictor" threshold. It reconstructs a fixed set of
post-entry observation checkpoints (4h/12h/24h/48h/3d/7d) from
already-existing OHLCV data for every SHORT trade in an already-persisted
trade list, using only candles up to and including each checkpoint (or up
to the trade's own exit, if it closed earlier) -- never a candle after the
checkpoint being measured, and never a candle after the trade's own exit.

Design decisions
-----------------
* **Checkpoints are fixed, not searched.** `CHECKPOINTS` is a documented,
  literal constant (4h/12h/24h/48h/3d/7d in minutes) taken directly from
  the block's own required list -- this module never tunes, adds, or
  removes a checkpoint based on what makes any trade look better or
  worse.
* **A checkpoint reached after the trade already exited is flagged
  `trade_closed_before_checkpoint=True`, and every metric for it is
  computed from the trade's actual (shorter) window, never fabricated
  by extrapolating past the real exit.** A trade that closed at 6h has
  no real "48h checkpoint" to report data for beyond its own exit
  candle -- reporting the 6h-window value under the 48h label would be
  misleading, so the flag makes that explicit rather than silently
  presenting a truncated window as if it were the full checkpoint.
* **MFE/MAE at a checkpoint reuses the same LONG/SHORT high/low
  convention as `hermes.mfe_mae_forensics`** (SHORT: favorable = low,
  adverse = high) -- not a different convention invented for this
  narrower, per-checkpoint use.
* **EMA200/ADX14/Donchian20 reuse the same independent pandas
  reimplementation already used in `hermes.short_persistence_audit`**
  (see that module's docstring for why: no freqtrade/talib dependency
  in the testable core, same periods as `TrendFollowCore.py`) -- not a
  third, separately-written version of the same three indicators.
* **The "meaningful adverse excursion" flag reuses the same fixed 1%
  threshold already established in `hermes.mfe_mae_forensics`**
  (`MEANINGFUL_EXCURSION_THRESHOLD_PCT`), not a new or re-tuned cutoff.
* **Grouping trades into persistent/ordinary/loser is done by the
  caller, from the fixed, already-established trade identities (the
  three named persistent winners), never re-derived by this module from
  a ranking or threshold** -- `checkpoint_series_for_trades` takes
  whatever `Trade` list the caller passes in per group.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Sequence

import pandas as pd

from hermes.trade_report import Trade

LONG = "LONG"
SHORT = "SHORT"

MEANINGFUL_EXCURSION_THRESHOLD_PCT = 1.0  # reused from hermes.mfe_mae_forensics
THIN_SAMPLE_THRESHOLD = 5

EMA_PERIOD = 200
ADX_PERIOD = 14
DONCHIAN_PERIOD = 20

# Fixed, literal checkpoints from the block's own required list -- minutes
# from entry. Never searched, added, or removed based on results.
CHECKPOINTS: dict[str, float] = {
    "4h": 240.0,
    "12h": 720.0,
    "24h": 1440.0,
    "48h": 2880.0,
    "3d": 4320.0,
    "7d": 10080.0,
}


# ---------------------------------------------------------------------------
# Indicator reimplementations (see module docstring: no freqtrade/talib
# dependency; same periods/conventions as hermes.short_persistence_audit)
# ---------------------------------------------------------------------------


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
# Checkpoint reconstruction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointMetrics:
    checkpoint_label: str
    checkpoint_minutes: float
    trade_closed_before_checkpoint: bool
    n_candles_in_window: int
    mfe_pct: float | None
    mae_pct: float | None
    price_change_pct: float | None
    ema200_distance_pct: float | None
    adx: float | None
    donchian_breakout_pct: float | None
    realized_vol_pct: float | None
    still_favorable_side: bool | None
    meaningful_adverse_seen: bool | None
    trend_aligned: bool | None


def _to_utc_ts(value) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def compute_checkpoint_metrics(
    ohlcv: pd.DataFrame | None,
    entry_time,
    exit_time,
    entry_price: float | None,
    direction: str | None,
    checkpoint_label: str,
    checkpoint_minutes: float,
) -> CheckpointMetrics:
    """Every requested metric for one trade at one fixed checkpoint,
    using only candles from `entry_time` through
    `min(entry_time + checkpoint_minutes, exit_time)` inclusive -- never a
    candle after the checkpoint, and never a candle after the trade's own
    exit. Returns an all-`None` result (with `n_candles_in_window=0`) if
    `ohlcv`/`entry_time`/`entry_price`/`direction` is missing or the
    window can't be sliced."""
    empty = CheckpointMetrics(
        checkpoint_label=checkpoint_label, checkpoint_minutes=checkpoint_minutes,
        trade_closed_before_checkpoint=False, n_candles_in_window=0,
        mfe_pct=None, mae_pct=None, price_change_pct=None, ema200_distance_pct=None,
        adx=None, donchian_breakout_pct=None, realized_vol_pct=None,
        still_favorable_side=None, meaningful_adverse_seen=None, trend_aligned=None,
    )
    if ohlcv is None or ohlcv.empty or direction not in (LONG, SHORT) or entry_price is None:
        return empty

    entry_ts = _to_utc_ts(entry_time)
    exit_ts = _to_utc_ts(exit_time)
    if entry_ts is None:
        return empty

    checkpoint_ts = entry_ts + pd.Timedelta(minutes=checkpoint_minutes)
    trade_closed_before_checkpoint = exit_ts is not None and exit_ts < checkpoint_ts
    window_end_ts = min(checkpoint_ts, exit_ts) if exit_ts is not None else checkpoint_ts

    window = ohlcv.loc[(ohlcv["date"] >= entry_ts) & (ohlcv["date"] <= window_end_ts)]
    window = window.sort_values("date").reset_index(drop=True)
    if window.empty:
        return CheckpointMetrics(
            checkpoint_label=checkpoint_label, checkpoint_minutes=checkpoint_minutes,
            trade_closed_before_checkpoint=trade_closed_before_checkpoint, n_candles_in_window=0,
            mfe_pct=None, mae_pct=None, price_change_pct=None, ema200_distance_pct=None,
            adx=None, donchian_breakout_pct=None, realized_vol_pct=None,
            still_favorable_side=None, meaningful_adverse_seen=None, trend_aligned=None,
        )

    if direction == LONG:
        favorable = (window["high"] - entry_price) / entry_price
        adverse = (entry_price - window["low"]) / entry_price
    else:
        favorable = (entry_price - window["low"]) / entry_price
        adverse = (window["high"] - entry_price) / entry_price
    mfe_pct = float(favorable.max()) * 100.0
    mae_pct = float(adverse.max()) * 100.0

    last_close = float(window["close"].iloc[-1])
    price_change_pct = 100.0 * (last_close - entry_price) / entry_price

    # Indicators computed from the full history up to the window's own
    # end (not just the trade's own window) so EMA200/ADX/Donchian are
    # computed the same way they would be live -- using all prior history,
    # not re-warming-up from the trade's entry candle.
    up_to_window_end = ohlcv.loc[ohlcv["date"] <= window_end_ts].sort_values("date").reset_index(drop=True)
    ema200 = ema(up_to_window_end["close"], EMA_PERIOD)
    adx14 = adx(up_to_window_end, ADX_PERIOD)
    upper, lower = donchian_prev_bounds(up_to_window_end, DONCHIAN_PERIOD)
    last_ema = ema200.iloc[-1]
    last_adx = adx14.iloc[-1]
    last_upper, last_lower = upper.iloc[-1], lower.iloc[-1]

    ema_distance = None
    if pd.notna(last_ema) and last_ema:
        ema_distance = 100.0 * (last_close - last_ema) / last_ema

    breakout_pct = None
    if direction == LONG and pd.notna(last_upper) and last_upper:
        breakout_pct = 100.0 * (last_close - last_upper) / last_upper
    elif direction == SHORT and pd.notna(last_lower) and last_lower:
        breakout_pct = 100.0 * (last_lower - last_close) / last_lower

    realized_vol = None
    if len(window) >= 3:
        returns = window["close"].pct_change().dropna()
        if len(returns) >= 2:
            realized_vol = float(returns.std()) * 100.0

    if direction == LONG:
        still_favorable = last_close > entry_price
        trend_aligned = pd.notna(last_ema) and last_close > last_ema
    else:
        still_favorable = last_close < entry_price
        trend_aligned = pd.notna(last_ema) and last_close < last_ema

    meaningful_adverse = mae_pct >= MEANINGFUL_EXCURSION_THRESHOLD_PCT

    return CheckpointMetrics(
        checkpoint_label=checkpoint_label,
        checkpoint_minutes=checkpoint_minutes,
        trade_closed_before_checkpoint=trade_closed_before_checkpoint,
        n_candles_in_window=len(window),
        mfe_pct=mfe_pct,
        mae_pct=mae_pct,
        price_change_pct=price_change_pct,
        ema200_distance_pct=(float(ema_distance) if ema_distance is not None else None),
        adx=(float(last_adx) if pd.notna(last_adx) else None),
        donchian_breakout_pct=(float(breakout_pct) if breakout_pct is not None else None),
        realized_vol_pct=realized_vol,
        still_favorable_side=bool(still_favorable),
        meaningful_adverse_seen=bool(meaningful_adverse),
        trend_aligned=bool(trend_aligned),
    )


def compute_checkpoint_series(
    ohlcv: pd.DataFrame | None, trade: Trade, checkpoints: dict[str, float] = CHECKPOINTS
) -> dict[str, CheckpointMetrics]:
    """`{checkpoint_label: CheckpointMetrics}` for every entry in
    `checkpoints`, for one trade."""
    return {
        label: compute_checkpoint_metrics(
            ohlcv, trade.entry_time, trade.exit_time, trade.entry_price, trade.direction, label, minutes
        )
        for label, minutes in checkpoints.items()
    }


# ---------------------------------------------------------------------------
# Group aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointAggregate:
    checkpoint_label: str
    n_trades_reached: int
    n_trades_total: int
    is_thin_sample: bool
    mean_mfe_pct: float | None
    median_mfe_pct: float | None
    mean_mae_pct: float | None
    median_mae_pct: float | None
    mean_price_change_pct: float | None
    mean_ema200_distance_pct: float | None
    mean_adx: float | None
    mean_donchian_breakout_pct: float | None
    mean_realized_vol_pct: float | None
    pct_still_favorable_side: float | None
    pct_meaningful_adverse_seen: float | None
    pct_trend_aligned: float | None


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    resolved = [v for v in values if v is not None]
    return mean(resolved) if resolved else None


def _median_or_none(values: Sequence[float | None]) -> float | None:
    resolved = [v for v in values if v is not None]
    return median(resolved) if resolved else None


def _pct_true(values: Sequence[bool | None]) -> float | None:
    resolved = [v for v in values if v is not None]
    return (100.0 * sum(1 for v in resolved if v) / len(resolved)) if resolved else None


def aggregate_checkpoint(
    per_trade_series: Sequence[dict[str, CheckpointMetrics]], checkpoint_label: str
) -> CheckpointAggregate:
    """Group-level aggregate for `checkpoint_label` across
    `per_trade_series` (one `compute_checkpoint_series` result per trade
    in the group). A trade whose window at this checkpoint had no
    resolvable data (`n_candles_in_window == 0`) contributes to
    `n_trades_total` but not `n_trades_reached` or any mean/median."""
    entries = [series[checkpoint_label] for series in per_trade_series if checkpoint_label in series]
    reached = [e for e in entries if e.n_candles_in_window > 0]

    return CheckpointAggregate(
        checkpoint_label=checkpoint_label,
        n_trades_reached=len(reached),
        n_trades_total=len(entries),
        is_thin_sample=(len(reached) < THIN_SAMPLE_THRESHOLD),
        mean_mfe_pct=_mean_or_none([e.mfe_pct for e in reached]),
        median_mfe_pct=_median_or_none([e.mfe_pct for e in reached]),
        mean_mae_pct=_mean_or_none([e.mae_pct for e in reached]),
        median_mae_pct=_median_or_none([e.mae_pct for e in reached]),
        mean_price_change_pct=_mean_or_none([e.price_change_pct for e in reached]),
        mean_ema200_distance_pct=_mean_or_none([e.ema200_distance_pct for e in reached]),
        mean_adx=_mean_or_none([e.adx for e in reached]),
        mean_donchian_breakout_pct=_mean_or_none([e.donchian_breakout_pct for e in reached]),
        mean_realized_vol_pct=_mean_or_none([e.realized_vol_pct for e in reached]),
        pct_still_favorable_side=_pct_true([e.still_favorable_side for e in reached]),
        pct_meaningful_adverse_seen=_pct_true([e.meaningful_adverse_seen for e in reached]),
        pct_trend_aligned=_pct_true([e.trend_aligned for e in reached]),
    )


def aggregate_all_checkpoints(
    per_trade_series: Sequence[dict[str, CheckpointMetrics]], checkpoints: dict[str, float] = CHECKPOINTS
) -> dict[str, CheckpointAggregate]:
    """`aggregate_checkpoint` for every checkpoint label in `checkpoints`."""
    return {label: aggregate_checkpoint(per_trade_series, label) for label in checkpoints}
