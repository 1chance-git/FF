"""Focused unit tests for `hermes.ema_ceiling_temporal_audit` (research-only)."""

from __future__ import annotations

from dataclasses import dataclass

from hermes.ema_ceiling_temporal_audit import (
    FAILED,
    INSUFFICIENT,
    MIXED,
    ROBUST,
    chronological_split,
    classify_removal_mode,
    classify_temporal_robustness,
)


@dataclass(frozen=True)
class _T:
    entry_time: str
    profit_pct: float = 0.0


# ---------------------------------------------------------------------------
# chronological_split
# ---------------------------------------------------------------------------


def test_chronological_split_orders_by_entry_time_not_input_order():
    trades = [_T("2026-03-01"), _T("2026-01-01"), _T("2026-02-01"), _T("2026-04-01")]
    split = chronological_split(trades)
    assert [t.entry_time for t in split.early] == ["2026-01-01", "2026-02-01"]
    assert [t.entry_time for t in split.late] == ["2026-03-01", "2026-04-01"]


def test_chronological_split_odd_count_late_gets_extra():
    trades = [_T(f"2026-01-{d:02d}") for d in range(1, 6)]  # 5 trades
    split = chronological_split(trades)
    assert len(split.early) == 2
    assert len(split.late) == 3


def test_chronological_split_reports_boundaries():
    trades = [_T("2026-01-01"), _T("2026-06-01")]
    split = chronological_split(trades)
    assert split.early_start == "2026-01-01"
    assert split.early_end == "2026-01-01"
    assert split.late_start == "2026-06-01"
    assert split.late_end == "2026-06-01"


def test_chronological_split_empty():
    split = chronological_split([])
    assert split.early == ()
    assert split.late == ()
    assert split.early_start is None


# ---------------------------------------------------------------------------
# classify_temporal_robustness
# ---------------------------------------------------------------------------


def test_classify_robust_when_improved_in_both_periods():
    verdict = classify_temporal_robustness(
        early_baseline_trades=10,
        late_baseline_trades=10,
        early_baseline_metric=1.0,
        early_variant_metric=1.5,
        late_baseline_metric=1.0,
        late_variant_metric=2.0,
    )
    assert verdict == ROBUST


def test_classify_failed_when_worse_in_both_periods():
    verdict = classify_temporal_robustness(
        early_baseline_trades=10,
        late_baseline_trades=10,
        early_baseline_metric=1.5,
        early_variant_metric=1.0,
        late_baseline_metric=2.0,
        late_variant_metric=1.0,
    )
    assert verdict == FAILED


def test_classify_mixed_when_only_one_period_improves():
    verdict = classify_temporal_robustness(
        early_baseline_trades=10,
        late_baseline_trades=10,
        early_baseline_metric=1.0,
        early_variant_metric=2.0,
        late_baseline_metric=2.0,
        late_variant_metric=1.0,
    )
    assert verdict == MIXED


def test_classify_insufficient_when_too_few_trades():
    verdict = classify_temporal_robustness(
        early_baseline_trades=2,
        late_baseline_trades=10,
        early_baseline_metric=1.0,
        early_variant_metric=2.0,
        late_baseline_metric=1.0,
        late_variant_metric=2.0,
    )
    assert verdict == INSUFFICIENT


def test_classify_insufficient_when_metric_undefined():
    verdict = classify_temporal_robustness(
        early_baseline_trades=10,
        late_baseline_trades=10,
        early_baseline_metric=None,
        early_variant_metric=2.0,
        late_baseline_metric=1.0,
        late_variant_metric=2.0,
    )
    assert verdict == INSUFFICIENT


def test_classify_respects_custom_min_trades():
    verdict = classify_temporal_robustness(
        early_baseline_trades=3,
        late_baseline_trades=3,
        early_baseline_metric=1.0,
        early_variant_metric=2.0,
        late_baseline_metric=1.0,
        late_variant_metric=2.0,
        min_trades=2,
    )
    assert verdict == ROBUST


# ---------------------------------------------------------------------------
# classify_removal_mode
# ---------------------------------------------------------------------------


def test_classify_removal_mode_primarily_losers():
    label = classify_removal_mode(
        baseline_losers=31, baseline_winners=8, removed_losers=15, removed_winners=1
    )
    assert label == "A_PRIMARILY_LOSERS"


def test_classify_removal_mode_primarily_winners():
    label = classify_removal_mode(
        baseline_losers=31, baseline_winners=8, removed_losers=1, removed_winners=5
    )
    assert label == "B_PRIMARILY_WINNERS"


def test_classify_removal_mode_proportional():
    label = classify_removal_mode(
        baseline_losers=31, baseline_winners=8, removed_losers=10, removed_winners=3
    )
    assert label == "C_BOTH_PROPORTIONALLY"


def test_classify_removal_mode_insufficient_when_bucket_empty():
    label = classify_removal_mode(
        baseline_losers=31, baseline_winners=0, removed_losers=10, removed_winners=0
    )
    assert label == "INSUFFICIENT"
