"""MFE/MAE trade-development forensics (research/analysis only).

Answers one question only: for the frozen BTC+ETH baseline, does LONG's
observed underperformance (see `hermes.long_short_directional_audit`)
look like an ENTRY QUALITY problem -- LONG trades rarely move favorably
after entry at all -- or a TRADE DEVELOPMENT / STOP INTERACTION problem
-- LONG trades often move favorably, then reverse into the stop. MFE
(maximum favorable excursion) and MAE (maximum adverse excursion),
reconstructed candle-by-candle from the execution candle through the
exit candle, are the only way to distinguish these two explanations;
the trade export alone (entry price, exit price, exit reason) cannot.

This module never launches a backtest, never touches
`TrendFollowCore.py`, config, the pair whitelist, or any strategy
parameter, and never fabricates a candle that isn't in the supplied
OHLCV data -- a trade whose candle window can't be sliced (missing
data, timestamps not found) gets an explicit unresolved result, never
an approximation.

Design decisions
-----------------
* **OHLC convention, stated explicitly rather than assumed:** for LONG,
  favorable excursion is measured off each candle's `high` (the best
  price the position could have been worth) and adverse excursion off
  each candle's `low` (the worst); for SHORT, this is reversed (`low`
  is favorable, `high` is adverse), since price falling is favorable to
  a short. `close` is never used for MFE/MAE -- a candle's high/low
  captures the true extremes within that period, `close` would
  understate both.
* **The candle window is `[execution_candle, exit_candle]` inclusive,
  never earlier and never later.** The execution candle is the trade's
  own `entry_time` (Freqtrade's fill time, not the signal candle 4h
  earlier -- the prior signal-forensics work's timing rule). Excursion
  is measured across the whole window including the exit candle itself,
  since the exit candle's high/low can still register a favorable or
  adverse extreme reached before the exit price was actually hit within
  that same candle.
* **Intrabar order within a single candle is fundamentally unknowable
  from OHLC data alone.** If a trade's maximum-favorable and
  maximum-adverse extremes both fall on the same candle,
  `MfeMaeResult.same_candle_ambiguous` is set `True` -- this is reported
  as a limitation of the underlying data, never resolved by guessing
  whether the high or the low came first within that candle.
* **The "meaningful movement" threshold used for trade-development
  classification (`classify_trade_development`) is a fixed, documented
  labeling constant (`MEANINGFUL_EXCURSION_THRESHOLD_PCT = 1.0`), not a
  tuned or optimized parameter.** It exists only to bucket trades into
  the four descriptive categories the spec asks for (A/B/C/D); changing
  it would relabel trades for this report, not change any strategy
  behavior, since nothing here feeds back into `TrendFollowCore.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Any, Sequence

import pandas as pd

LONG = "LONG"
SHORT = "SHORT"

# Fixed labeling threshold for `classify_trade_development` -- see module
# docstring. Not a strategy parameter; never optimized or searched.
MEANINGFUL_EXCURSION_THRESHOLD_PCT = 1.0

CATEGORY_IMMEDIATE_FAILURE = "A_IMMEDIATE_FAILURE"
CATEGORY_FAVORABLE_THEN_REVERSAL = "B_FAVORABLE_THEN_REVERSAL"
CATEGORY_SUCCESSFUL_TREND = "C_SUCCESSFUL_TREND"
CATEGORY_SMALL_MOVEMENT = "D_SMALL_MOVEMENT"
CATEGORY_UNRESOLVED = "U_UNRESOLVED"


# ---------------------------------------------------------------------------
# Candle window slicing (execution candle through exit candle, inclusive)
# ---------------------------------------------------------------------------


def slice_trade_window(
    ohlcv_df: pd.DataFrame, entry_time: Any, exit_time: Any
) -> pd.DataFrame | None:
    """Rows of `ohlcv_df` with `date` in `[entry_time, exit_time]` inclusive,
    sorted ascending, index reset to a 0-based position from the execution
    candle. Returns `None` (not an empty DataFrame) if either timestamp is
    unparseable, if `exit_time < entry_time`, or if the slice contains zero
    rows -- every one of these is a reportable data gap, never silently
    treated as "no movement"."""
    if entry_time is None or exit_time is None:
        return None
    try:
        entry_ts = pd.Timestamp(entry_time)
        exit_ts = pd.Timestamp(exit_time)
    except (ValueError, TypeError):
        return None
    entry_ts = entry_ts.tz_localize("UTC") if entry_ts.tzinfo is None else entry_ts.tz_convert("UTC")
    exit_ts = exit_ts.tz_localize("UTC") if exit_ts.tzinfo is None else exit_ts.tz_convert("UTC")
    if exit_ts < entry_ts:
        return None

    window = ohlcv_df.loc[(ohlcv_df["date"] >= entry_ts) & (ohlcv_df["date"] <= exit_ts)]
    if window.empty:
        return None
    return window.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# MFE/MAE computation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MfeMaeResult:
    direction: str | None
    mfe_pct: float | None
    mae_pct: float | None
    mfe_candle_index: int | None
    mfe_candle_time: str | None
    mae_candle_index: int | None
    mae_candle_time: str | None
    n_candles: int
    same_candle_ambiguous: bool
    unresolved_reason: str | None

    @property
    def is_resolved(self) -> bool:
        return self.unresolved_reason is None


def _unresolved(direction: str | None, reason: str) -> MfeMaeResult:
    return MfeMaeResult(
        direction=direction, mfe_pct=None, mae_pct=None,
        mfe_candle_index=None, mfe_candle_time=None,
        mae_candle_index=None, mae_candle_time=None,
        n_candles=0, same_candle_ambiguous=False, unresolved_reason=reason,
    )


def compute_mfe_mae(
    direction: str | None, entry_price: float | None, candles: pd.DataFrame | None
) -> MfeMaeResult:
    """MFE/MAE over `candles` (already sliced to `[execution, exit]`
    inclusive by `slice_trade_window`, 0-based positional index from the
    execution candle). See the module docstring for the OHLC convention.
    Returns an unresolved result (never a fabricated 0.0) if `direction`
    is neither `"LONG"` nor `"SHORT"`, `entry_price` is missing, or
    `candles` is `None`/empty."""
    if direction not in (LONG, SHORT):
        return _unresolved(direction, "unknown_direction")
    if entry_price is None:
        return _unresolved(direction, "missing_entry_price")
    if candles is None or candles.empty:
        return _unresolved(direction, "no_candle_window")
    for col in ("date", "high", "low"):
        if col not in candles.columns:
            return _unresolved(direction, f"missing_column:{col}")

    if direction == LONG:
        favorable = (candles["high"] - entry_price) / entry_price
        adverse = (entry_price - candles["low"]) / entry_price
    else:
        favorable = (entry_price - candles["low"]) / entry_price
        adverse = (candles["high"] - entry_price) / entry_price

    mfe_idx = favorable.idxmax()
    mae_idx = adverse.idxmax()

    return MfeMaeResult(
        direction=direction,
        mfe_pct=float(favorable.loc[mfe_idx]) * 100.0,
        mae_pct=float(adverse.loc[mae_idx]) * 100.0,
        mfe_candle_index=int(mfe_idx),
        mfe_candle_time=str(candles["date"].loc[mfe_idx]),
        mae_candle_index=int(mae_idx),
        mae_candle_time=str(candles["date"].loc[mae_idx]),
        n_candles=len(candles),
        same_candle_ambiguous=(mfe_idx == mae_idx),
        unresolved_reason=None,
    )


def minutes_from_entry(candle_index: int | None, timeframe_minutes: float) -> float | None:
    """Convert a 0-based candle index (from `MfeMaeResult`) into minutes
    from the execution candle, given the OHLCV timeframe's minute length
    (e.g. 240.0 for "4h"). `candle_index=0` (the execution candle itself)
    is 0 minutes, never `None` or a fabricated non-zero value."""
    if candle_index is None:
        return None
    return candle_index * timeframe_minutes


# ---------------------------------------------------------------------------
# Trade-development classification (Question 6)
# ---------------------------------------------------------------------------


def classify_trade_development(
    result: MfeMaeResult,
    is_winner: bool | None,
    *,
    threshold_pct: float = MEANINGFUL_EXCURSION_THRESHOLD_PCT,
) -> str:
    """A/B/C/D per the spec's TRADE-DEVELOPMENT CLASSIFICATION, using the
    fixed `threshold_pct` labeling constant (see module docstring) to
    decide what counts as "meaningful" excursion. `U_UNRESOLVED` when
    `result` is itself unresolved or `is_winner` is unknown and the MFE
    excursion was meaningful (the only case where the category genuinely
    depends on knowing the outcome)."""
    if not result.is_resolved:
        return CATEGORY_UNRESOLVED
    mfe_meaningful = result.mfe_pct >= threshold_pct
    mae_meaningful = result.mae_pct >= threshold_pct

    if not mfe_meaningful and not mae_meaningful:
        return CATEGORY_SMALL_MOVEMENT
    if not mfe_meaningful and mae_meaningful:
        return CATEGORY_IMMEDIATE_FAILURE
    # mfe_meaningful is True from here on.
    if is_winner is True:
        return CATEGORY_SUCCESSFUL_TREND
    if is_winner is False:
        return CATEGORY_FAVORABLE_THEN_REVERSAL
    return CATEGORY_UNRESOLVED


# ---------------------------------------------------------------------------
# Aggregate summaries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MfeMaeSummary:
    n: int
    mean_mfe_pct: float | None
    median_mfe_pct: float | None
    mean_mae_pct: float | None
    median_mae_pct: float | None


def summarize_mfe_mae(results: Sequence[MfeMaeResult]) -> MfeMaeSummary:
    """Mean/median MFE and MAE across `results`, excluding unresolved
    entries (never treating "unresolved" as 0.0)."""
    resolved = [r for r in results if r.is_resolved]
    mfes = [r.mfe_pct for r in resolved]
    maes = [r.mae_pct for r in resolved]
    return MfeMaeSummary(
        n=len(resolved),
        mean_mfe_pct=(mean(mfes) if mfes else None),
        median_mfe_pct=(median(mfes) if mfes else None),
        mean_mae_pct=(mean(maes) if maes else None),
        median_mae_pct=(median(maes) if maes else None),
    )


def mfe_threshold_breakdown(results: Sequence[MfeMaeResult], thresholds_pct: Sequence[float]) -> dict[float, int]:
    """For each threshold in `thresholds_pct`, the count of resolved
    `results` whose `mfe_pct` reached or exceeded it -- Question 3's
    "did stopped trades ever exceed +1%/+2%/+3%/+5%" breakdown."""
    resolved = [r for r in results if r.is_resolved]
    return {t: sum(1 for r in resolved if r.mfe_pct >= t) for t in thresholds_pct}
