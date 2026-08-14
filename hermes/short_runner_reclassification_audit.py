"""SHORT-winner structural-persistence reclassification audit
(research/analysis only, DIAGNOSIS ONLY).

Answers one question only: "Which SHORT winners are structurally
persistent based on how long their favorable trend structure survives,
independent of final P/L ranking?" This module never renames or
reclassifies the existing PERSISTENT/ORDINARY groups -- it only compares
their fixed P/L-based labels (reused verbatim) against structural
behavior reconstructed from already-existing OHLCV data. It never runs a
backtest, never touches `TrendFollowCore.py`, config, or the pair
whitelist, and never searches for or proposes a new trading rule or
optimized threshold.

Design decisions
-----------------
* **Every reconstruction primitive here is reused, not reinvented.**
  Full-trade candle reconstruction, checkpoint slicing, and the
  PERSISTENT/ORDINARY P/L-based identity match come from
  `hermes.short_runner_divergence_audit` (itself inherited from
  `hermes.short_trend_persistence_audit`). First-EMA200-invalidation
  detection comes from `hermes.short_ema_exit_attribution_audit`
  (`reconstruct_trade_candles` / `find_first_invalidation`), which
  already implements the exact `close > ema200` mirror of
  `TrendFollowCore.compute_exit_signals`'s SHORT exit condition. Nothing
  in this module recomputes EMA200/ADX/Donchian independently.
* **Only the 8 SHORT winners are in scope** -- SHORT losers are excluded,
  matching this block's stated question.
* **The P/L-based label (`pl_label`) is `classify_group`'s existing
  output, reused verbatim** -- this module never re-derives, re-ranks, or
  overwrites PERSISTENT/ORDINARY. A separate, purely descriptive
  `structurally_persistent` flag is computed from structural
  measurements only (duration, % aligned, longest aligned run, first
  invalidation timing, MFE trajectory) and is reported *alongside* the
  P/L label for comparison -- never used to relabel a trade's official
  group.
* **No new threshold is invented.** `find_largest_gap` describes,
  descriptively, where the single largest gap sits in a sorted list of
  values already present in the 8-trade table (e.g. duration in days) --
  it never proposes that gap as a cutoff rule, and the report is
  required to say explicitly if no such gap is large/defensible enough
  to act as a natural dividing line.
* **MFE-trajectory and EMA-distance-trajectory checkpoints reuse a small
  subset of the already-established checkpoint ladder**
  (`hermes.short_runner_divergence_audit.CHECKPOINTS`) -- 4h, 7d, 14d,
  21d, 30d -- never a newly invented ladder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from hermes.short_ema_exit_attribution_audit import (
    find_first_invalidation,
    reconstruct_trade_candles,
)
from hermes.short_runner_divergence_audit import (
    CHECKPOINTS as FULL_CHECKPOINTS,
    LONG,
    PERSISTENT_KEYS,
    SHORT,
    all_checkpoint_snapshots,
    classify_group,
    reconstruct_full_trade,
)
from hermes.trade_report import Trade

# A small subset of the already-established checkpoint ladder -- never a
# newly invented one -- used only to sample the EMA-distance and MFE
# trajectories at a few readable points.
TRAJECTORY_CHECKPOINTS: dict[str, float] = {
    label: FULL_CHECKPOINTS[label] for label in ("4h", "7d", "14d", "21d", "30d")
}
TRAJECTORY_ORDER: tuple[str, ...] = tuple(TRAJECTORY_CHECKPOINTS.keys())


def list_short_winners(all_trades: Sequence[Trade]) -> list[Trade]:
    """The 8 SHORT winners, in the order they appear in `all_trades` --
    never re-sorted by P/L."""
    return [t for t in all_trades if t.direction == SHORT and t.is_winner is True]


# ---------------------------------------------------------------------------
# Per-trade structural record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuralTradeRecord:
    pair: str
    entry_time: str | None
    pl_label: str  # "PERSISTENT" or "ORDINARY" -- classify_group's existing output, reused verbatim
    final_profit_pct: float | None
    duration_minutes: float | None
    duration_days: float | None
    pct_structurally_aligned: float | None
    longest_aligned_run_candles: int
    ema_distance_trajectory: dict[str, float | None]  # checkpoint label -> mean EMA200 distance %
    first_invalidation_time: object | None  # pandas.Timestamp | None
    hours_entry_to_invalidation: float | None
    mfe_trajectory: dict[str, float | None]  # checkpoint label -> cumulative MFE %


def build_structural_record(trade: Trade, ohlcv) -> StructuralTradeRecord:
    """Reconstructs the 6 measurements named in this block for a single
    SHORT winner, using only already-existing reconstruction primitives.
    Returns a record with `None`/empty fields (never fabricated values)
    when `ohlcv` is unavailable."""
    pl_label = classify_group(trade)
    duration_days = (trade.duration_minutes / 1440.0) if trade.duration_minutes is not None else None

    full_candles = reconstruct_full_trade(ohlcv, trade.entry_time, trade.exit_time, trade.entry_price, trade.direction)
    aligned_flags = [c.structurally_aligned for c in full_candles]
    resolved_aligned = [f for f in aligned_flags if f is not None]
    pct_aligned = (100.0 * sum(1 for f in resolved_aligned if f) / len(resolved_aligned)) if resolved_aligned else None

    longest = current = 0
    for f in aligned_flags:
        if f is True:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    snapshots = all_checkpoint_snapshots(full_candles, checkpoints=TRAJECTORY_CHECKPOINTS)
    ema_trajectory = {label: snapshots[label].mean_ema_distance_pct for label in TRAJECTORY_ORDER}
    mfe_trajectory = {label: snapshots[label].mfe_pct for label in TRAJECTORY_ORDER}

    invalidation_candles = reconstruct_trade_candles(ohlcv, trade.entry_time, trade.exit_time, trade.entry_price, trade.direction)
    first_invalidation = find_first_invalidation(invalidation_candles)
    first_invalidation_time = first_invalidation.date if first_invalidation is not None else None
    hours_to_invalidation = None
    if first_invalidation is not None:
        hours_to_invalidation = first_invalidation.days_since_entry * 24.0

    return StructuralTradeRecord(
        pair=trade.pair, entry_time=str(trade.entry_time), pl_label=pl_label,
        final_profit_pct=trade.profit_pct, duration_minutes=trade.duration_minutes, duration_days=duration_days,
        pct_structurally_aligned=pct_aligned, longest_aligned_run_candles=longest,
        ema_distance_trajectory=ema_trajectory, first_invalidation_time=first_invalidation_time,
        hours_entry_to_invalidation=hours_to_invalidation, mfe_trajectory=mfe_trajectory,
    )


def build_structural_table(short_winners: Sequence[Trade], ohlcv_by_pair: dict[str, object]) -> list[StructuralTradeRecord]:
    """`build_structural_record` for every trade in `short_winners`, in
    the order given -- never re-sorted."""
    return [build_structural_record(t, ohlcv_by_pair.get(t.pair)) for t in short_winners]


# ---------------------------------------------------------------------------
# Structural-persistence flag (descriptive only -- never a relabel)
# ---------------------------------------------------------------------------

# The single existing checkpoint reused as the descriptive "still alive
# at 30d, still favorable" reading -- not a new indicator, just reading
# the already-computed 30d snapshot's own `closed_before_checkpoint`
# equivalent (a trade whose real duration is shorter than 30 days simply
# has its trajectory dict frozen at its own last real value, per
# `all_checkpoint_snapshots`'s existing convention).
LONG_DURATION_LABEL = "30d"


def is_structurally_persistent(record: StructuralTradeRecord, all_records: Sequence[StructuralTradeRecord]) -> bool:
    """A trade is flagged structurally persistent, purely descriptively,
    when its trade duration is at or above the MEDIAN duration of all 8
    SHORT winners in `all_records` AND its 30d MFE reading is at or above
    that same median. This reuses only values already present in the
    8-trade table (duration, MFE) and the ordinary statistical median of
    the existing sample -- it is not an invented magnitude cutoff, and it
    is never written back onto `record.pl_label`."""
    durations = sorted(r.duration_days for r in all_records if r.duration_days is not None)
    mfes = sorted(r.mfe_trajectory.get(LONG_DURATION_LABEL) for r in all_records if r.mfe_trajectory.get(LONG_DURATION_LABEL) is not None)
    if not durations or not mfes or record.duration_days is None:
        return False
    median_duration = durations[len(durations) // 2] if len(durations) % 2 == 1 else (durations[len(durations) // 2 - 1] + durations[len(durations) // 2]) / 2.0
    median_mfe = mfes[len(mfes) // 2] if len(mfes) % 2 == 1 else (mfes[len(mfes) // 2 - 1] + mfes[len(mfes) // 2]) / 2.0
    record_mfe = record.mfe_trajectory.get(LONG_DURATION_LABEL)
    if record_mfe is None:
        return False
    return record.duration_days >= median_duration and record_mfe >= median_mfe


# ---------------------------------------------------------------------------
# Natural-gap description (never adopted as a threshold)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GapDescription:
    gap_index: int  # position in the sorted list where the largest gap starts (between index and index+1)
    gap_size: float
    value_before: float
    value_after: float
    is_dominant: bool  # True only if this gap is more than double every other gap in the same sorted list


def find_largest_gap(sorted_values: Sequence[float]) -> GapDescription | None:
    """Purely descriptive: the single largest consecutive gap in an
    already-sorted list of values (e.g. duration_days across the 8 SHORT
    winners). `is_dominant` is `True` only when that gap is more than
    double every other consecutive gap in the same list -- a description
    of whether the data itself suggests one obvious natural split, not an
    invented rule. Returns `None` for fewer than 2 values."""
    n = len(sorted_values)
    if n < 2:
        return None
    gaps = [(i, sorted_values[i + 1] - sorted_values[i]) for i in range(n - 1)]
    gap_index, gap_size = max(gaps, key=lambda pair: pair[1])
    other_gaps = [g for i, g in gaps if i != gap_index]
    is_dominant = bool(other_gaps) and all(gap_size > 2 * g for g in other_gaps) if other_gaps else True
    return GapDescription(
        gap_index=gap_index, gap_size=gap_size,
        value_before=sorted_values[gap_index], value_after=sorted_values[gap_index + 1],
        is_dominant=is_dominant,
    )
