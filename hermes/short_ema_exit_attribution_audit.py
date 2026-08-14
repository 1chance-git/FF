"""SHORT-winner EMA200 structural-exit attribution audit (research only).

Answers one question only: for the 8 SHORT winners in the extended
BTC+ETH backtest, is the difference between the 3 previously identified
persistent runners and the 5 ordinary winners explained by *when* and
*how* `TrendFollowCore`'s EMA200 structural exit condition
(`close > ema200` invalidates a SHORT) was first reached -- reconstructed
candle-by-candle from entry to exit, using only candles up to and
including the candle being evaluated (no lookahead).

This module never runs a backtest, never touches `TrendFollowCore.py`,
config, or the pair whitelist, and never tunes the EMA period or
threshold. It is a pure-function, Freqtrade-independent research layer
over an already-persisted trade list plus already-existing OHLCV data.

Design decisions
-----------------
* **EMA200 reconstruction reuses the same independent pandas
  reimplementation already established in `hermes.short_persistence_audit`
  and `hermes.short_runner_lifecycle_audit`** (`ewm(span=200,
  adjust=False)`, matching `TrendFollowCore.compute_indicators`'s
  `talib.EMA` period) -- not a third, separately-written version.
* **The structural exit rule reconstructed here is exactly
  `TrendFollowCore.compute_exit_signals`'s SHORT condition: `close >
  ema200`, evaluated only on candles with a non-NaN EMA200** -- read
  directly from that file's logic (documented in this module's
  docstring), not guessed independently.
* **"First invalidation" is found by scanning candles in chronological
  order and stopping at the first one where the condition is `True`,
  using only that candle and everything before it** -- `find_first_invalidation`
  never looks ahead to confirm a candle "actually" invalidated the trade
  by checking what happens afterward, and never uses a candle after the
  trade's own persisted exit time.
* **Persistent/ordinary classification is by exact `(pair, entry_time)`
  identity against the three already-established persistent trades**,
  the same matching approach already used in
  `hermes.short_runner_lifecycle_audit` -- never re-derived by ranking or
  threshold in this module.
* **A trade with no invalidation candle found before its own exit gets
  `first_invalidation_time=None`, never a fabricated timestamp** -- this
  is itself a reportable finding (the EMA200 condition was never true
  before the trade's actual exit), not an error.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Sequence

import pandas as pd

from hermes.trade_report import Trade

SHORT = "SHORT"
LONG = "LONG"

EMA_PERIOD = 200
THIN_SAMPLE_THRESHOLD = 5

# The three previously established persistent SHORT winners, identified
# by exact (pair, entry_time) -- see hermes.short_runner_lifecycle_audit
# for the same matching approach.
PERSISTENT_KEYS: frozenset[tuple[str, str]] = frozenset({
    ("ETH/USDC:USDC", "2026-01-20 16:00:00+00:00"),
    ("ETH/USDC:USDC", "2026-05-15 16:00:00+00:00"),
    ("BTC/USDC:USDC", "2026-01-20 08:00:00+00:00"),
})


def ema(series: pd.Series, period: int = EMA_PERIOD) -> pd.Series:
    """Exponential moving average matching `TrendFollowCore.py`'s
    `talib.EMA(close, timeperiod=200)` (see module docstring)."""
    return series.ewm(span=period, adjust=False).mean()


def classify_persistent_or_ordinary(trade: Trade, persistent_keys: frozenset = PERSISTENT_KEYS) -> str:
    """`"PERSISTENT"` if `(trade.pair, str(trade.entry_time))` matches one
    of `persistent_keys` exactly, else `"ORDINARY"`. Never applied to a
    non-SHORT or non-winning trade by this function itself -- callers
    filter first."""
    key = (trade.pair, str(trade.entry_time))
    return "PERSISTENT" if key in persistent_keys else "ORDINARY"


# ---------------------------------------------------------------------------
# Candle-by-candle reconstruction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandleRecord:
    date: pd.Timestamp
    close: float
    ema200: float | None
    distance_pct: float | None
    structurally_aligned: bool | None
    exit_condition_true: bool | None
    cumulative_mfe_pct: float
    cumulative_mae_pct: float
    days_since_entry: float


def reconstruct_trade_candles(
    ohlcv: pd.DataFrame | None, entry_time, exit_time, entry_price: float | None, direction: str | None,
) -> list[CandleRecord]:
    """Every candle from `entry_time` through `exit_time` inclusive
    (never a candle after the trade's own exit), each carrying EMA200
    computed from all history up to and including that candle (never a
    later candle), plus cumulative MFE/MAE up to and including that
    candle and days elapsed since entry.

    For SHORT (the only direction this audit's exit rule applies to),
    `structurally_aligned` is `close < ema200` and `exit_condition_true`
    is `close > ema200` -- the exact mirror of
    `TrendFollowCore.compute_exit_signals`'s `exit_short` condition.
    Returns an empty list if `ohlcv`/`entry_time`/`entry_price` is
    missing or the window can't be sliced."""
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
    window_mask = full_history["date"] >= entry_ts
    if exit_ts is not None:
        window_mask &= full_history["date"] <= exit_ts
    window = full_history.loc[window_mask]
    if window.empty:
        return []

    records: list[CandleRecord] = []
    cumulative_mfe = 0.0
    cumulative_mae = 0.0
    for _, row in window.iterrows():
        # EMA computed from all history up to and including this candle
        # -- never a later candle -- matching how the live strategy would
        # have seen it.
        up_to_here = full_history.loc[full_history["date"] <= row["date"]]
        ema200_series = ema(up_to_here["close"], EMA_PERIOD)
        ema_value = ema200_series.iloc[-1]

        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])

        if direction == SHORT:
            favorable = (entry_price - low) / entry_price * 100.0
            adverse = (high - entry_price) / entry_price * 100.0
        else:
            favorable = (high - entry_price) / entry_price * 100.0
            adverse = (entry_price - low) / entry_price * 100.0
        cumulative_mfe = max(cumulative_mfe, favorable)
        cumulative_mae = max(cumulative_mae, adverse)

        distance_pct = None
        aligned = None
        exit_true = None
        if pd.notna(ema_value) and ema_value:
            distance_pct = 100.0 * (close - ema_value) / ema_value
            if direction == SHORT:
                aligned = bool(close < ema_value)
                exit_true = bool(close > ema_value)
            else:
                aligned = bool(close > ema_value)
                exit_true = bool(close < ema_value)

        days_since_entry = (row["date"] - entry_ts).total_seconds() / 86400.0

        records.append(CandleRecord(
            date=row["date"], close=close,
            ema200=(float(ema_value) if pd.notna(ema_value) else None),
            distance_pct=distance_pct,
            structurally_aligned=aligned,
            exit_condition_true=exit_true,
            cumulative_mfe_pct=cumulative_mfe,
            cumulative_mae_pct=cumulative_mae,
            days_since_entry=days_since_entry,
        ))
    return records


def find_first_invalidation(candles: Sequence[CandleRecord]) -> CandleRecord | None:
    """The first `CandleRecord` (in chronological order, as given) whose
    `exit_condition_true` is `True` -- `None` if the condition was never
    true across every candle provided. Never inspects any candle after
    the first `True` one found; never uses information beyond `candles`
    itself."""
    index, record = find_first_invalidation_index(candles)
    return record


def find_first_invalidation_index(candles: Sequence[CandleRecord]) -> tuple[int | None, CandleRecord | None]:
    """Same as `find_first_invalidation`, but also returns the candle's
    0-based position in `candles` (or `(None, None)` if never true) --
    used internally to look up the immediately-preceding candle without
    relying on dataclass equality/`.index()`."""
    for i, record in enumerate(candles):
        if record.exit_condition_true is True:
            return i, record
    return None, None


# ---------------------------------------------------------------------------
# Per-trade diagnosis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradeExitDiagnosis:
    pair: str
    entry_time: str | None
    group: str
    final_profit_pct: float | None
    duration_minutes: float | None
    exit_time: str | None
    exit_reason: str | None
    n_candles_reconstructed: int
    first_invalidation_time: pd.Timestamp | None
    hours_invalidation_to_exit: float | None
    distance_pct_before_invalidation: float | None
    distance_pct_at_invalidation: float | None
    mfe_before_invalidation_pct: float | None
    mfe_after_invalidation_pct: float | None


def diagnose_trade(trade: Trade, candles: Sequence[CandleRecord]) -> TradeExitDiagnosis:
    """The full per-trade diagnosis record for `trade`, given its
    already-reconstructed `candles` (see `reconstruct_trade_candles`)."""
    group = classify_persistent_or_ordinary(trade) if trade.direction == SHORT else "N/A"

    if not candles:
        return TradeExitDiagnosis(
            pair=trade.pair, entry_time=trade.entry_time, group=group,
            final_profit_pct=trade.profit_pct, duration_minutes=trade.duration_minutes,
            exit_time=trade.exit_time, exit_reason=trade.exit_reason,
            n_candles_reconstructed=0, first_invalidation_time=None,
            hours_invalidation_to_exit=None, distance_pct_before_invalidation=None,
            distance_pct_at_invalidation=None, mfe_before_invalidation_pct=None,
            mfe_after_invalidation_pct=None,
        )

    idx, first_invalidation = find_first_invalidation_index(candles)

    hours_to_exit = None
    distance_before = None
    distance_at = None
    mfe_before = None
    mfe_after = None

    if first_invalidation is not None:
        last_candle = candles[-1]
        hours_to_exit = (last_candle.date - first_invalidation.date).total_seconds() / 3600.0
        distance_at = first_invalidation.distance_pct
        mfe_before = first_invalidation.cumulative_mfe_pct
        mfe_after = last_candle.cumulative_mfe_pct

        if idx is not None and idx > 0:
            distance_before = candles[idx - 1].distance_pct

    return TradeExitDiagnosis(
        pair=trade.pair, entry_time=trade.entry_time, group=group,
        final_profit_pct=trade.profit_pct, duration_minutes=trade.duration_minutes,
        exit_time=trade.exit_time, exit_reason=trade.exit_reason,
        n_candles_reconstructed=len(candles),
        first_invalidation_time=(first_invalidation.date if first_invalidation else None),
        hours_invalidation_to_exit=hours_to_exit,
        distance_pct_before_invalidation=distance_before,
        distance_pct_at_invalidation=distance_at,
        mfe_before_invalidation_pct=mfe_before,
        mfe_after_invalidation_pct=mfe_after,
    )


# ---------------------------------------------------------------------------
# Group aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupAggregate:
    n: int
    is_thin_sample: bool
    n_with_invalidation_found: int
    mean_hours_invalidation_to_exit: float | None
    median_hours_invalidation_to_exit: float | None
    mean_distance_at_invalidation: float | None
    mean_mfe_before_invalidation: float | None
    mean_mfe_after_invalidation: float | None


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    resolved = [v for v in values if v is not None]
    return mean(resolved) if resolved else None


def _median_or_none(values: Sequence[float | None]) -> float | None:
    resolved = [v for v in values if v is not None]
    return median(resolved) if resolved else None


def aggregate_group(diagnoses: Sequence[TradeExitDiagnosis]) -> GroupAggregate:
    """Group-level aggregate across `diagnoses` (one group: PERSISTENT or
    ORDINARY). `THIN_SAMPLE_THRESHOLD`-based flag applied to `n`, per
    this research program's established discipline."""
    with_invalidation = [d for d in diagnoses if d.first_invalidation_time is not None]
    return GroupAggregate(
        n=len(diagnoses),
        is_thin_sample=(len(diagnoses) < THIN_SAMPLE_THRESHOLD),
        n_with_invalidation_found=len(with_invalidation),
        mean_hours_invalidation_to_exit=_mean_or_none([d.hours_invalidation_to_exit for d in with_invalidation]),
        median_hours_invalidation_to_exit=_median_or_none([d.hours_invalidation_to_exit for d in with_invalidation]),
        mean_distance_at_invalidation=_mean_or_none([d.distance_pct_at_invalidation for d in with_invalidation]),
        mean_mfe_before_invalidation=_mean_or_none([d.mfe_before_invalidation_pct for d in with_invalidation]),
        mean_mfe_after_invalidation=_mean_or_none([d.mfe_after_invalidation_pct for d in with_invalidation]),
    )
