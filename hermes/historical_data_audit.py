"""Historical OHLCV data-availability audit (research/analysis only).

Answers one question only: given the OHLCV candles actually available to
this project for BTC/USDC:USDC and ETH/USDC:USDC, how far back can the
existing TrendFollowCore strategy be reliably evaluated? This module never
runs a backtest, never touches the strategy, config, or pair whitelist, and
never recommends a date range -- it only inspects candle data already
present on disk (or supplied as a DataFrame) and reports what is there:
coverage, gaps, duplicates, timeframe consistency, and the longest window
where both BTC and ETH have valid, gap-free candles.

Design decisions
-----------------
* **A "gap" is any place where consecutive candle timestamps are farther
  apart than one timeframe step.** This module never fabricates a candle to
  fill a gap and never assumes a gap represents "no movement" -- a missing
  period is reported as a missing period, nothing more.
* **A "duplicate" is more than one row sharing the same `date`.** Detected
  and counted, never silently deduplicated by this module (the caller's
  loader may already dedupe; this module reports what it's given).
* **"Shared coverage" and "longest continuous shared window"** are computed
  only from each series' own gap list -- no interpolation, no merging of
  the two DataFrames into a single reindexed series, so the result can't
  invent a candle time that isn't actually present in both.
* **No performance interpretation.** This module never touches
  `profit_pct`, entry/exit prices, or backtest results -- it is a pure data
  quality report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class Gap:
    after: pd.Timestamp
    before: pd.Timestamp
    missing_candles: int


@dataclass(frozen=True)
class CoverageReport:
    n_rows: int
    earliest: pd.Timestamp | None
    latest: pd.Timestamp | None
    n_duplicates: int
    n_gaps: int
    gaps: tuple[Gap, ...]
    timeframe_consistent: bool
    inconsistent_intervals: tuple[float, ...]


@dataclass(frozen=True)
class ContinuousWindow:
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    n_candles: int


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Sorted-by-date view with duplicates (same `date`) kept but counted
    separately by the caller -- this only sorts, it never drops rows."""
    return df.sort_values("date").reset_index(drop=True)


def count_duplicates(df: pd.DataFrame) -> int:
    """Number of rows whose `date` value repeats at least once elsewhere in
    `df` (i.e. `duplicated(keep=False)` count minus one occurrence per
    group -- the number of *extra* rows beyond the first for each
    timestamp)."""
    if df.empty:
        return 0
    return int(df["date"].duplicated(keep="first").sum())


def find_gaps(df: pd.DataFrame, timeframe_minutes: float) -> list[Gap]:
    """Every place where two chronologically consecutive candles (after
    sorting, deduplicating by `date`) are more than one `timeframe_minutes`
    step apart. Returns an empty list for empty/single-row input -- never
    fabricates a gap where there's insufficient data to detect one."""
    if df.empty or timeframe_minutes <= 0:
        return []
    ordered = _normalize(df).drop_duplicates(subset="date", keep="first")
    if len(ordered) < 2:
        return []
    step = pd.Timedelta(minutes=timeframe_minutes)
    dates = ordered["date"]
    deltas = dates.diff().iloc[1:]
    gaps: list[Gap] = []
    for idx, delta in deltas.items():
        if delta > step:
            missing = int(round(delta / step)) - 1
            gaps.append(
                Gap(after=dates.loc[idx - 1], before=dates.loc[idx], missing_candles=missing)
            )
    return gaps


def check_timeframe_consistency(
    df: pd.DataFrame, timeframe_minutes: float
) -> tuple[bool, tuple[float, ...]]:
    """Whether every consecutive-candle interval (ignoring gaps, i.e. only
    intervals that equal a whole number of `timeframe_minutes` steps) is
    actually a multiple of `timeframe_minutes` -- catches candles at the
    wrong cadence (e.g. a stray 5m candle mixed into a 4h series), not just
    missing candles. Returns the distinct offending interval lengths (in
    minutes) found, if any."""
    if df.empty or timeframe_minutes <= 0:
        return True, ()
    ordered = _normalize(df).drop_duplicates(subset="date", keep="first")
    if len(ordered) < 2:
        return True, ()
    step = pd.Timedelta(minutes=timeframe_minutes)
    deltas = ordered["date"].diff().dropna()
    bad = sorted({
        round(d / pd.Timedelta(minutes=1), 4)
        for d in deltas
        if (d % step) != pd.Timedelta(0)
    })
    return (len(bad) == 0), tuple(bad)


def analyze_coverage(df: pd.DataFrame, timeframe_minutes: float) -> CoverageReport:
    """Full single-series coverage report: row count, earliest/latest
    candle, duplicate count, gap list, and timeframe-consistency check."""
    if df.empty:
        return CoverageReport(
            n_rows=0, earliest=None, latest=None, n_duplicates=0,
            n_gaps=0, gaps=(), timeframe_consistent=True, inconsistent_intervals=(),
        )
    ordered = _normalize(df)
    gaps = find_gaps(df, timeframe_minutes)
    consistent, bad_intervals = check_timeframe_consistency(df, timeframe_minutes)
    return CoverageReport(
        n_rows=len(ordered),
        earliest=ordered["date"].iloc[0],
        latest=ordered["date"].iloc[-1],
        n_duplicates=count_duplicates(df),
        n_gaps=len(gaps),
        gaps=tuple(gaps),
        timeframe_consistent=consistent,
        inconsistent_intervals=bad_intervals,
    )


def shared_coverage(
    btc_report: CoverageReport, eth_report: CoverageReport
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """The overlap of the two series' [earliest, latest] ranges -- the
    widest window in which BOTH assets have at least one candle at each
    end. Returns `(None, None)` if either report has no data or the ranges
    don't overlap at all."""
    if btc_report.earliest is None or eth_report.earliest is None:
        return None, None
    start = max(btc_report.earliest, eth_report.earliest)
    end = min(btc_report.latest, eth_report.latest)
    if start > end:
        return None, None
    return start, end


def _gap_free_segments(report: CoverageReport) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Contiguous [start, end] segments between `report.earliest` and
    `report.latest`, split at every recorded gap -- built purely from the
    gap list already computed by `analyze_coverage`, no re-scan of raw
    data."""
    if report.earliest is None:
        return []
    bounds = [report.earliest]
    for gap in report.gaps:
        bounds.append(gap.after)
        bounds.append(gap.before)
    bounds.append(report.latest)
    segments = []
    for i in range(0, len(bounds), 2):
        segments.append((bounds[i], bounds[i + 1]))
    return segments


def longest_continuous_shared_window(
    btc_report: CoverageReport, eth_report: CoverageReport, timeframe_minutes: float
) -> ContinuousWindow:
    """The longest [start, end] window in which BOTH BTC and ETH have
    gap-free candle coverage -- the intersection of each asset's own
    gap-free segments, keeping the widest overlapping pair. Purely derived
    from each report's own `gaps` list; never assumes continuity across a
    detected gap in either series."""
    btc_segments = _gap_free_segments(btc_report)
    eth_segments = _gap_free_segments(eth_report)
    if not btc_segments or not eth_segments:
        return ContinuousWindow(start=None, end=None, n_candles=0)

    best: tuple[pd.Timestamp, pd.Timestamp] | None = None
    for b_start, b_end in btc_segments:
        for e_start, e_end in eth_segments:
            start = max(b_start, e_start)
            end = min(b_end, e_end)
            if start > end:
                continue
            if best is None or (end - start) > (best[1] - best[0]):
                best = (start, end)

    if best is None:
        return ContinuousWindow(start=None, end=None, n_candles=0)
    start, end = best
    step = pd.Timedelta(minutes=timeframe_minutes)
    n_candles = int((end - start) / step) + 1 if timeframe_minutes > 0 else 0
    return ContinuousWindow(start=start, end=end, n_candles=n_candles)


@dataclass(frozen=True)
class BaselineCoverageCheck:
    baseline_start: pd.Timestamp
    baseline_end: pd.Timestamp
    fully_covered: bool
    missing_before: pd.Timedelta | None
    missing_after: pd.Timedelta | None


def check_baseline_coverage(
    shared_start: pd.Timestamp | None,
    shared_end: pd.Timestamp | None,
    baseline_start: pd.Timestamp,
    baseline_end: pd.Timestamp,
) -> BaselineCoverageCheck:
    """Whether the existing frozen baseline period `[baseline_start,
    baseline_end]` lies entirely within `[shared_start, shared_end]`.
    Reports the exact shortfall on either end when it doesn't -- never a
    bare True/False without the gap size."""
    if shared_start is None or shared_end is None:
        return BaselineCoverageCheck(
            baseline_start=baseline_start, baseline_end=baseline_end,
            fully_covered=False, missing_before=None, missing_after=None,
        )
    missing_before = (shared_start - baseline_start) if shared_start > baseline_start else pd.Timedelta(0)
    missing_after = (baseline_end - shared_end) if baseline_end > shared_end else pd.Timedelta(0)
    fully_covered = missing_before <= pd.Timedelta(0) and missing_after <= pd.Timedelta(0)
    return BaselineCoverageCheck(
        baseline_start=baseline_start, baseline_end=baseline_end,
        fully_covered=fully_covered,
        missing_before=(missing_before if missing_before > pd.Timedelta(0) else None),
        missing_after=(missing_after if missing_after > pd.Timedelta(0) else None),
    )
