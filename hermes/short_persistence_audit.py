"""SHORT-trade persistence attribution audit (research/analysis only).

Answers one question only: why does a minority of SHORT trades in the
extended BTC+ETH sample persist much longer and develop much larger
favorable excursions than ordinary SHORT trades -- and is that
distinguishable at entry, or does it emerge only after entry?

This module never runs a backtest, never touches `TrendFollowCore.py`,
config, or the pair whitelist, and never searches for or recommends a
threshold. It is a pure-function, Freqtrade-independent research layer
over already-persisted trade records plus already-existing OHLCV data.

Design decisions
-----------------
* **No Freqtrade/talib dependency in the testable core.** `TrendFollowCore.py`
  computes EMA200/ADX14/Donchian20 via `talib`, which this module's unit
  tests cannot assume is installed. `ema`, `adx`, and `donchian_prev_bounds`
  here are independent, transparent pandas/pure-Python reimplementations of
  the *same* indicator definitions (same periods, same "previous N
  completed candles" Donchian convention as `TrendFollowCore.compute_indicators`)
  -- for descriptive/forensic use only. They are not claimed to be
  numerically bit-identical to `talib`'s implementation (talib's ADX in
  particular has its own internal smoothing/convergence behavior); they
  exist to answer "was this trade's entry condition unusual," not to
  reproduce the strategy's exact live signal.
* **"Persistent SHORT" is defined by reusing the already-established
  finding from the prior SHORT-edge attribution audit, not a newly
  invented cutoff.** That audit found the top 3 SHORT winners (by
  `profit_pct`) accounted for the overwhelming majority of SHORT's
  aggregate P/L (+74.72% falling to +3.68% once those three are
  removed). `identify_persistent_winners` generalizes that already-used
  "top N by profit_pct" ranking (the same primitive as
  `hermes.short_edge_attribution_audit.remove_top_n_winners`, viewed from
  the kept side rather than the removed side) with a documented default
  of `n=3` -- not a parameter search, not an optimization.
* **Entry-only metrics and post-entry metrics are kept in clearly
  separate dataclasses (`EntryConditionMetrics` vs. post-entry MFE/MAE),
  never merged into one ambiguous "trade characteristics" blob** -- this
  is what Requirement F asks for: never letting a post-entry observation
  get mislabeled as something visible at entry.
* **MFE/MAE reuses `hermes.mfe_mae_forensics` verbatim** (same
  execution-candle-through-exit-candle-inclusive window, same
  high/low-based LONG/SHORT convention, same same-candle-ambiguity flag)
  -- not reimplemented a second, possibly inconsistent way.
* **Every subgroup function reports its own `n`; nothing here decides
  "thin sample" on the caller's behalf** -- the fixed `THIN_SAMPLE_THRESHOLD`
  constant is exposed so a report-writing caller can apply the same
  labeling rule used elsewhere in this research program (`<5` trades),
  but this module never suppresses or skips computing a statistic because
  a group is small.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Sequence

import pandas as pd

from hermes.mfe_mae_forensics import MfeMaeResult, compute_mfe_mae, slice_trade_window
from hermes.trade_report import Trade

LONG = "LONG"
SHORT = "SHORT"

THIN_SAMPLE_THRESHOLD = 5

# Same indicator periods as TrendFollowCore.py -- reused, not re-chosen.
EMA_PERIOD = 200
ADX_PERIOD = 14
DONCHIAN_PERIOD = 20

DEFAULT_PERSISTENT_N = 3  # see module docstring: reuses the prior audit's finding


# ---------------------------------------------------------------------------
# A. Reconciliation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReconciliationResult:
    n: int
    n_long: int
    n_short: int
    n_btc: int
    n_eth: int
    earliest_entry: str | None
    latest_exit: str | None
    matches_expected: bool


def reconcile(
    trades: Sequence[Trade],
    *,
    expected_n: int = 39,
    expected_long: int = 18,
    expected_short: int = 21,
    expected_btc: int = 19,
    expected_eth: int = 20,
    btc_pair: str = "BTC/USDC:USDC",
    eth_pair: str = "ETH/USDC:USDC",
) -> ReconciliationResult:
    """Whether `trades` matches the established 39/18/21/19/20 extended
    BTC+ETH sample. Never raises -- `matches_expected` is `False` on any
    mismatch, leaving the caller to decide whether to stop."""
    n = len(trades)
    n_long = sum(1 for t in trades if t.direction == LONG)
    n_short = sum(1 for t in trades if t.direction == SHORT)
    n_btc = sum(1 for t in trades if t.pair == btc_pair)
    n_eth = sum(1 for t in trades if t.pair == eth_pair)
    entries = [t.entry_time for t in trades if t.entry_time]
    exits = [t.exit_time for t in trades if t.exit_time]

    matches = (
        n == expected_n and n_long == expected_long and n_short == expected_short
        and n_btc == expected_btc and n_eth == expected_eth
    )
    return ReconciliationResult(
        n=n, n_long=n_long, n_short=n_short, n_btc=n_btc, n_eth=n_eth,
        earliest_entry=(min(entries) if entries else None),
        latest_exit=(max(exits) if exits else None),
        matches_expected=matches,
    )


# ---------------------------------------------------------------------------
# B. Persistent SHORT identification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShortWinnerRecord:
    trade: Trade
    duration_minutes: float | None
    mfe_pct: float | None
    mae_pct: float | None
    profit_pct: float | None


def list_short_winners(trades: Sequence[Trade], joined_mfe_mae: dict[int, MfeMaeResult]) -> list[ShortWinnerRecord]:
    """Every SHORT winner in `trades`, with its duration/MFE/MAE/P&L.
    `joined_mfe_mae` maps `id(trade)` to its already-computed `MfeMaeResult`
    (see `attach_mfe_mae_by_id`) -- this function never computes MFE/MAE
    itself, only assembles the record."""
    records = []
    for t in trades:
        if t.direction != SHORT or t.is_winner is not True:
            continue
        result = joined_mfe_mae.get(id(t))
        records.append(ShortWinnerRecord(
            trade=t,
            duration_minutes=t.duration_minutes,
            mfe_pct=(result.mfe_pct if result and result.is_resolved else None),
            mae_pct=(result.mae_pct if result and result.is_resolved else None),
            profit_pct=t.profit_pct,
        ))
    return records


def identify_persistent_winners(
    short_winners: Sequence[ShortWinnerRecord], n: int = DEFAULT_PERSISTENT_N
) -> tuple[list[ShortWinnerRecord], list[ShortWinnerRecord]]:
    """`(persistent, ordinary)`: the `n` largest-`profit_pct` SHORT winners
    vs. the rest -- reusing the prior audit's already-established top-N
    finding (see module docstring), not a newly searched cutoff. Winners
    with a `None` profit_pct are never candidates for the persistent
    group and are always placed in `ordinary`."""
    resolved = [w for w in short_winners if w.profit_pct is not None]
    ranked = sorted(resolved, key=lambda w: w.profit_pct, reverse=True)
    persistent_ids = set(id(w) for w in ranked[:n])
    persistent = [w for w in short_winners if id(w) in persistent_ids]
    ordinary = [w for w in short_winners if id(w) not in persistent_ids]
    return persistent, ordinary


# ---------------------------------------------------------------------------
# Entry-condition metrics (pure reimplementation of TrendFollowCore's
# indicator definitions; see module docstring for why this isn't just an
# import of that file)
# ---------------------------------------------------------------------------


def ema(series: pd.Series, period: int = EMA_PERIOD) -> pd.Series:
    """Exponential moving average, `alpha = 2 / (period + 1)`, matching
    the standard EMA definition `TrendFollowCore.py` uses via `talib.EMA`."""
    return series.ewm(span=period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.Series:
    """Wilder's Average Directional Index over `high`/`low`/`close`,
    matching the period `TrendFollowCore.py` uses via `talib.ADX`. An
    independent pandas reimplementation (see module docstring) -- not
    claimed bit-identical to `talib`'s output, only directionally
    equivalent for descriptive comparison."""
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low, (high - prev_close).abs(), (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def donchian_prev_bounds(df: pd.DataFrame, period: int = DONCHIAN_PERIOD) -> tuple[pd.Series, pd.Series]:
    """`(upper, lower)`: rolling max of `high` / min of `low` over the
    *previous* `period` completed candles -- `high`/`low` shifted forward
    one bar before the rolling window, identical convention to
    `TrendFollowCore.compute_indicators`'s Donchian columns (the current
    candle never contributes to its own breakout threshold)."""
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    return prev_high.rolling(window=period).max(), prev_low.rolling(window=period).min()


@dataclass(frozen=True)
class EntryConditionMetrics:
    ema200_distance_pct: float | None
    adx_at_entry: float | None
    donchian_breakout_pct: float | None
    realized_vol_before_entry_pct: float | None


def compute_entry_condition_metrics(
    ohlcv: pd.DataFrame, entry_time, entry_price: float | None, direction: str | None,
    *, lookback_candles: int = 20,
) -> EntryConditionMetrics:
    """Entry-only metrics computed from candles strictly at-or-before
    `entry_time` (the indicator/breakout columns) plus a fixed
    `lookback_candles`-candle pre-entry window (for realized volatility)
    -- never using any candle after `entry_time`, so nothing here can leak
    post-entry information into an "entry condition."""
    if ohlcv is None or ohlcv.empty or entry_time is None:
        return EntryConditionMetrics(None, None, None, None)
    try:
        entry_ts = pd.Timestamp(entry_time)
    except (ValueError, TypeError):
        return EntryConditionMetrics(None, None, None, None)
    entry_ts = entry_ts.tz_localize("UTC") if entry_ts.tzinfo is None else entry_ts.tz_convert("UTC")

    up_to_entry = ohlcv.loc[ohlcv["date"] <= entry_ts].sort_values("date").reset_index(drop=True)
    if up_to_entry.empty:
        return EntryConditionMetrics(None, None, None, None)

    ema200 = ema(up_to_entry["close"], EMA_PERIOD)
    adx14 = adx(up_to_entry, ADX_PERIOD)
    upper, lower = donchian_prev_bounds(up_to_entry, DONCHIAN_PERIOD)

    last_ema = ema200.iloc[-1]
    last_adx = adx14.iloc[-1]
    last_upper, last_lower = upper.iloc[-1], lower.iloc[-1]
    last_close = up_to_entry["close"].iloc[-1]

    ema_distance = None
    if entry_price is not None and pd.notna(last_ema) and last_ema != 0:
        ema_distance = 100.0 * (entry_price - last_ema) / last_ema

    breakout_pct = None
    if direction == LONG and pd.notna(last_upper) and last_upper != 0:
        breakout_pct = 100.0 * (last_close - last_upper) / last_upper
    elif direction == SHORT and pd.notna(last_lower) and last_lower != 0:
        breakout_pct = 100.0 * (last_lower - last_close) / last_lower

    pre_window = up_to_entry.tail(lookback_candles)
    realized_vol = None
    if len(pre_window) >= 2:
        returns = pre_window["close"].pct_change().dropna()
        if len(returns) >= 2:
            realized_vol = float(returns.std()) * 100.0

    return EntryConditionMetrics(
        ema200_distance_pct=(float(ema_distance) if ema_distance is not None and pd.notna(ema_distance) else None),
        adx_at_entry=(float(last_adx) if pd.notna(last_adx) else None),
        donchian_breakout_pct=(float(breakout_pct) if breakout_pct is not None and pd.notna(breakout_pct) else None),
        realized_vol_before_entry_pct=realized_vol,
    )


@dataclass(frozen=True)
class PostEntryMetrics:
    realized_vol_after_entry_pct: float | None
    price_expansion_pct: float | None


def compute_post_entry_metrics(window: pd.DataFrame | None) -> PostEntryMetrics:
    """Post-entry-only metrics computed from the trade's own
    execution-through-exit candle window (`window`, e.g. from
    `hermes.mfe_mae_forensics.slice_trade_window`) -- realized volatility
    of closes across the window, and price expansion (the window's total
    high-to-low range as a percent of the first candle's close). Never
    labeled an "entry predictor" -- these only exist once the trade is
    already open."""
    if window is None or window.empty or len(window) < 2:
        return PostEntryMetrics(None, None)
    returns = window["close"].pct_change().dropna()
    realized_vol = float(returns.std()) * 100.0 if len(returns) >= 2 else None

    first_close = window["close"].iloc[0]
    expansion = None
    if first_close:
        expansion = 100.0 * (window["high"].max() - window["low"].min()) / first_close

    return PostEntryMetrics(
        realized_vol_after_entry_pct=realized_vol,
        price_expansion_pct=(float(expansion) if expansion is not None else None),
    )


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricAggregate:
    n: int
    mean_value: float | None
    median_value: float | None
    is_thin_sample: bool


def aggregate_metric(values: Sequence[float | None]) -> MetricAggregate:
    """Mean/median of the non-`None` entries of `values`, with a
    `THIN_SAMPLE_THRESHOLD`-based flag -- never silently drops the flag
    even when `n` is large enough to look reassuring at a glance."""
    resolved = [v for v in values if v is not None]
    return MetricAggregate(
        n=len(resolved),
        mean_value=(mean(resolved) if resolved else None),
        median_value=(median(resolved) if resolved else None),
        is_thin_sample=(len(resolved) < THIN_SAMPLE_THRESHOLD),
    )


def group_by_quarter(trades: Sequence[Trade], n_quarters: int = 4) -> list[list[Trade]]:
    """Same mechanical chronological-quarter segmentation as
    `hermes.regime_robustness_audit.chronological_quarters` (by entry
    order, not calendar date) -- reused here under this module's own name
    so this module has no import-time dependency on that one, since both
    already exist independently in this research program."""
    ordered = sorted((t for t in trades if t.entry_time is not None), key=lambda t: t.entry_time)
    n = len(ordered)
    if n == 0:
        return [[] for _ in range(n_quarters)]
    base_size, remainder = divmod(n, n_quarters)
    quarters: list[list[Trade]] = []
    start = 0
    for i in range(n_quarters):
        size = base_size + (1 if i < remainder else 0)
        quarters.append(ordered[start:start + size])
        start += size
    return quarters
