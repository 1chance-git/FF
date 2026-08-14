"""MFE-to-realized-P/L capture audit (research/analysis only).

Answers one question only: when TrendFollowCore captures a favorable
excursion (MFE), how much of it ends up as realized final P/L? This
module contains no trading logic, no network calls, and never mutates
strategy/config/pair-list state. It is a pure-function computation layer
over already-reconstructed trade records -- it does not itself load
OHLCV data or reconstruct MFE from candles (that reconstruction reuses
the existing `hermes.short_runner_divergence_audit.reconstruct_full_trade`
pattern at the call site, exactly as every prior forensic module in this
research program has done); this module starts from a `CaptureRecord`
already carrying `max_mfe_pct` and (optionally) a checkpoint trajectory,
and only computes ratios, aggregates, and groupings over those inputs.

Design decisions
-----------------
* **Capture ratio is defined once, explicitly**: `final_profit_pct /
  max_mfe_pct`, where `max_mfe_pct` is the trade's own maximum observed
  favorable excursion across its *entire* entry-to-exit (or
  entry-to-censoring) window -- never a value observed after the trade
  actually closed, and never a checkpoint value silently substituted for
  the true maximum. Callers are responsible for only ever passing an
  `max_mfe_pct` that respects this temporal ordering; this module does
  not itself validate timestamps (it has none to check) but every field
  name says explicitly which MFE definition it is.
* **Zero/near-zero MFE is handled explicitly, never via silent division.**
  `compute_capture_ratio` returns `None` (not `inf`, not `0.0`, not a
  clamped value) whenever `max_mfe_pct` is `None` or below
  `MIN_MEANINGFUL_MFE_PCT` -- a trade that never moved favorably has no
  meaningful "how much of the excursion was captured" answer, and this
  module never invents one.
* **Censored trades are flagged, never silently included as ordinary
  observations.** `CENSORED_EXIT_REASONS = {"force_exit"}` (Freqtrade's
  own end-of-backtest-window termination reason) -- a `CaptureRecord`
  whose `exit_reason` is in this set is marked `is_censored=True` by
  `classify_censoring`, and every aggregate function in this module
  accepts an `include_censored` flag (default `False`) so censored
  trades are excluded from capture-ratio statistics by default, never
  silently blended in. This module never widens the definition of
  "censored" beyond `force_exit` on its own judgment -- a caller
  reporting a different reason as censored must say so explicitly.
* **No threshold optimization, no new indicator.** EMA-distance-vs-capture
  interaction (report section F) is answered via `bucket_by_ema_distance`,
  a fixed-count quantile split of whatever records are passed in -- never
  a searched-for cutoff, never persisted as a rule.
* **Median over mean where a summary is genuinely needed**, because
  capture ratios are typically fat-tailed (a handful of stop-loss losers
  can produce large-magnitude negative ratios); both are always reported
  side by side rather than one being silently preferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Sequence

LONG = "LONG"
SHORT = "SHORT"

# Freqtrade's own end-of-backtest-window termination reason. Never
# extended by this module's own judgment -- a caller must explicitly
# widen this set (or pass a record's own is_censored override) to treat
# any other exit_reason as censored.
CENSORED_EXIT_REASONS: frozenset[str] = frozenset({"force_exit"})

# A trade whose max observed MFE is at or below this magnitude has no
# meaningful "fraction captured" -- not an invented cutoff for a trading
# decision, just the floor below which final_profit_pct / max_mfe_pct is
# numerically meaningless (division by near-zero).
MIN_MEANINGFUL_MFE_PCT = 0.01


@dataclass(frozen=True)
class CaptureRecord:
    pair: str
    direction: str  # "LONG" or "SHORT"
    entry_time: str
    exit_time: str | None
    duration_days: float | None
    pct_structurally_aligned: float | None
    final_profit_pct: float | None
    max_mfe_pct: float | None
    mfe_checkpoints: dict[str, float | None]  # e.g. {"4h": ..., "7d": ..., "14d": ..., "21d": ..., "30d": ...}
    exit_reason: str | None
    mean_ema_distance_pct: float | None = None  # optional, for section F only


@dataclass(frozen=True)
class CaptureResult:
    record: CaptureRecord
    is_censored: bool
    capture_ratio: float | None  # None when max_mfe_pct is missing/near-zero


def classify_censoring(record: CaptureRecord) -> bool:
    """`True` only if `record.exit_reason` is in `CENSORED_EXIT_REASONS`.
    Never inferred from duration or candle count -- the exit reason is
    the single source of truth for censoring."""
    return record.exit_reason in CENSORED_EXIT_REASONS


def compute_capture_ratio(record: CaptureRecord) -> float | None:
    """`final_profit_pct / max_mfe_pct`, or `None` when either value is
    missing or `max_mfe_pct` is at/below `MIN_MEANINGFUL_MFE_PCT` (never
    a division by near-zero, never an invented fallback value)."""
    if record.final_profit_pct is None or record.max_mfe_pct is None:
        return None
    if record.max_mfe_pct <= MIN_MEANINGFUL_MFE_PCT:
        return None
    return record.final_profit_pct / record.max_mfe_pct


def compute_capture_result(record: CaptureRecord) -> CaptureResult:
    return CaptureResult(
        record=record,
        is_censored=classify_censoring(record),
        capture_ratio=compute_capture_ratio(record),
    )


def compute_capture_results(records: Sequence[CaptureRecord]) -> list[CaptureResult]:
    return [compute_capture_result(r) for r in records]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaptureAggregate:
    n_total: int
    n_eligible: int  # has a defined capture_ratio
    n_censored_excluded: int
    n_zero_mfe_excluded: int
    median_mfe_pct: float | None
    median_final_profit_pct: float | None
    median_capture_ratio: float | None
    mean_capture_ratio: float | None
    min_capture_ratio: float | None
    max_capture_ratio: float | None


def _median_or_none(values: Sequence[float]) -> float | None:
    resolved = [v for v in values if v is not None]
    return median(resolved) if resolved else None


def _mean_or_none(values: Sequence[float]) -> float | None:
    resolved = [v for v in values if v is not None]
    return mean(resolved) if resolved else None


def aggregate_capture(
    results: Sequence[CaptureResult], include_censored: bool = False
) -> CaptureAggregate:
    """Aggregate capture statistics over `results`. Censored trades are
    excluded by default (`include_censored=False`) -- pass `True` only
    when the caller explicitly wants censored trades blended in, and even
    then `classify_censoring`'s flag on each result stays visible to the
    caller. Trades with an undefined `capture_ratio` (missing or
    near-zero MFE) are always excluded from the ratio statistics, whether
    or not they're censored, and counted separately."""
    n_total = len(results)
    scoped = [r for r in results if include_censored or not r.is_censored]
    n_censored_excluded = n_total - len(scoped)
    eligible = [r for r in scoped if r.capture_ratio is not None]
    n_zero_mfe_excluded = len(scoped) - len(eligible)

    ratios = [r.capture_ratio for r in eligible]
    return CaptureAggregate(
        n_total=n_total,
        n_eligible=len(eligible),
        n_censored_excluded=n_censored_excluded,
        n_zero_mfe_excluded=n_zero_mfe_excluded,
        median_mfe_pct=_median_or_none([r.record.max_mfe_pct for r in scoped]),
        median_final_profit_pct=_median_or_none([r.record.final_profit_pct for r in scoped]),
        median_capture_ratio=_median_or_none(ratios),
        mean_capture_ratio=_mean_or_none(ratios),
        min_capture_ratio=(min(ratios) if ratios else None),
        max_capture_ratio=(max(ratios) if ratios else None),
    )


def aggregate_by_direction(
    results: Sequence[CaptureResult], include_censored: bool = False
) -> dict[str, CaptureAggregate]:
    """`aggregate_capture` split by `record.direction` ("LONG"/"SHORT")."""
    out: dict[str, CaptureAggregate] = {}
    for direction in (LONG, SHORT):
        subset = [r for r in results if r.record.direction == direction]
        out[direction] = aggregate_capture(subset, include_censored=include_censored)
    return out


def aggregate_by_pair(
    results: Sequence[CaptureResult], include_censored: bool = False
) -> dict[str, CaptureAggregate]:
    """`aggregate_capture` split by `record.pair`."""
    pairs = sorted({r.record.pair for r in results})
    return {
        pair: aggregate_capture([r for r in results if r.record.pair == pair], include_censored=include_censored)
        for pair in pairs
    }


def aggregate_by_exit_reason(
    results: Sequence[CaptureResult],
) -> dict[str, CaptureAggregate]:
    """`aggregate_capture` split by `record.exit_reason`, always including
    every reason's own trades regardless of censoring (censoring itself
    IS the split here, so `include_censored=True` for every subgroup --
    a `force_exit` bucket with `include_censored=False` would report zero
    trades, which would be misleading rather than informative)."""
    reasons = sorted({r.record.exit_reason for r in results if r.record.exit_reason is not None})
    return {
        reason: aggregate_capture([r for r in results if r.record.exit_reason == reason], include_censored=True)
        for reason in reasons
    }


# ---------------------------------------------------------------------------
# MFE trajectory pattern classification (descriptive only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectoryPattern:
    label: str  # "early_plateau" / "steady_increase" / "late_acceleration" / "giveback" / "insufficient_data"
    checkpoints_used: tuple[str, ...]


def classify_trajectory_pattern(
    checkpoints: dict[str, float | None], order: Sequence[str] = ("4h", "7d", "14d", "21d", "30d")
) -> TrajectoryPattern:
    """Purely descriptive classification of an MFE trajectory shape,
    using only the checkpoints actually present (never fabricates a
    missing observation). Requires at least 3 non-`None` checkpoints to
    classify anything beyond `"insufficient_data"`.

    - `"giveback"`: MFE is non-monotonic in a way that decreases at any
      step by more than a trivial rounding amount. Since MFE is defined
      as a running maximum in every reconstruction this module consumes,
      a real decrease here means the record itself was built from a
      *point-in-time* (not cumulative) MFE reading -- flagged rather than
      silently reinterpreted.
    - `"early_plateau"`: the last available reading is within 5% of the
      first non-zero reading (little growth after the earliest point).
    - `"late_acceleration"`: more than half of the total observed MFE
      growth happens in the final available step.
    - `"steady_increase"`: none of the above -- growth spread across
      multiple steps.
    """
    present = [(label, checkpoints[label]) for label in order if checkpoints.get(label) is not None]
    if len(present) < 3:
        return TrajectoryPattern(label="insufficient_data", checkpoints_used=tuple(l for l, _ in present))

    values = [v for _, v in present]
    labels_used = tuple(l for l, _ in present)

    for a, b in zip(values, values[1:]):
        if b < a - 1e-9:
            return TrajectoryPattern(label="giveback", checkpoints_used=labels_used)

    total_growth = values[-1] - values[0]
    if total_growth <= 1e-9:
        return TrajectoryPattern(label="early_plateau", checkpoints_used=labels_used)

    first_step_growth = values[1] - values[0]
    final_step_growth = values[-1] - values[-2]

    if first_step_growth > 0.5 * total_growth:
        return TrajectoryPattern(label="early_plateau", checkpoints_used=labels_used)
    if final_step_growth > 0.5 * total_growth:
        return TrajectoryPattern(label="late_acceleration", checkpoints_used=labels_used)

    return TrajectoryPattern(label="steady_increase", checkpoints_used=labels_used)


# ---------------------------------------------------------------------------
# EMA-distance interaction (observational only, no threshold search)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmaDistanceBucket:
    bucket_label: str  # "low" / "mid" / "high" (tercile by |ema_distance|)
    n: int
    median_mfe_pct: float | None
    median_final_profit_pct: float | None
    median_capture_ratio: float | None
    median_duration_days: float | None


def bucket_by_ema_distance(results: Sequence[CaptureResult]) -> list[EmaDistanceBucket]:
    """Splits `results` with a non-`None` `mean_ema_distance_pct` into
    three equal-count buckets ("low"/"mid"/"high") by `abs(mean_ema_
    distance_pct)`, purely for observing whether MFE/capture/duration
    differ across the existing distribution -- never a threshold search,
    never a proposed cutoff. Returns an empty list if fewer than 3
    records carry a usable EMA-distance value (a tercile split is
    meaningless below that)."""
    usable = [r for r in results if r.record.mean_ema_distance_pct is not None]
    if len(usable) < 3:
        return []
    ordered = sorted(usable, key=lambda r: abs(r.record.mean_ema_distance_pct))
    n = len(ordered)
    third = n // 3
    groups = {
        "low": ordered[:third] if third > 0 else ordered[:1],
        "mid": ordered[third: n - third] if third > 0 else ordered[1:-1] if n > 2 else [],
        "high": ordered[n - third:] if third > 0 else ordered[-1:],
    }
    buckets = []
    for label in ("low", "mid", "high"):
        group = groups[label]
        buckets.append(EmaDistanceBucket(
            bucket_label=label, n=len(group),
            median_mfe_pct=_median_or_none([r.record.max_mfe_pct for r in group]),
            median_final_profit_pct=_median_or_none([r.record.final_profit_pct for r in group]),
            median_capture_ratio=_median_or_none([r.capture_ratio for r in group]),
            median_duration_days=_median_or_none([r.record.duration_days for r in group]),
        ))
    return buckets
