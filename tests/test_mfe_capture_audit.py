"""Focused unit tests for `hermes.mfe_capture_audit` (research-only)."""

from __future__ import annotations

import pytest

from hermes.mfe_capture_audit import (
    CENSORED_EXIT_REASONS,
    MIN_MEANINGFUL_MFE_PCT,
    CaptureRecord,
    aggregate_by_direction,
    aggregate_by_exit_reason,
    aggregate_by_pair,
    aggregate_capture,
    bucket_by_ema_distance,
    classify_censoring,
    classify_trajectory_pattern,
    compute_capture_ratio,
    compute_capture_result,
    compute_capture_results,
)


def _record(
    pair="BTC/USDC:USDC", direction="SHORT", entry_time="2026-01-01T00:00:00Z", exit_time="2026-01-10T00:00:00Z",
    duration_days=9.0, pct_structurally_aligned=99.0, final_profit_pct=10.0, max_mfe_pct=20.0,
    mfe_checkpoints=None, exit_reason="exit_signal", mean_ema_distance_pct=None,
):
    return CaptureRecord(
        pair=pair, direction=direction, entry_time=entry_time, exit_time=exit_time, duration_days=duration_days,
        pct_structurally_aligned=pct_structurally_aligned, final_profit_pct=final_profit_pct, max_mfe_pct=max_mfe_pct,
        mfe_checkpoints=mfe_checkpoints or {}, exit_reason=exit_reason, mean_ema_distance_pct=mean_ema_distance_pct,
    )


# ---------------------------------------------------------------------------
# censoring classification
# ---------------------------------------------------------------------------


def test_censored_exit_reasons_is_force_exit_only():
    assert CENSORED_EXIT_REASONS == frozenset({"force_exit"})


def test_classify_censoring_true_for_force_exit():
    assert classify_censoring(_record(exit_reason="force_exit")) is True


def test_classify_censoring_false_for_exit_signal_and_stop_loss():
    assert classify_censoring(_record(exit_reason="exit_signal")) is False
    assert classify_censoring(_record(exit_reason="stop_loss")) is False


def test_classify_censoring_false_for_none_reason():
    assert classify_censoring(_record(exit_reason=None)) is False


# ---------------------------------------------------------------------------
# capture ratio: explicit None handling, never division-by-near-zero
# ---------------------------------------------------------------------------


def test_compute_capture_ratio_normal_case():
    r = _record(final_profit_pct=10.0, max_mfe_pct=20.0)
    assert compute_capture_ratio(r) == pytest.approx(0.5)


def test_compute_capture_ratio_none_when_mfe_missing():
    r = _record(final_profit_pct=10.0, max_mfe_pct=None)
    assert compute_capture_ratio(r) is None


def test_compute_capture_ratio_none_when_profit_missing():
    r = _record(final_profit_pct=None, max_mfe_pct=20.0)
    assert compute_capture_ratio(r) is None


def test_compute_capture_ratio_none_when_mfe_near_zero():
    r = _record(final_profit_pct=1.0, max_mfe_pct=MIN_MEANINGFUL_MFE_PCT)
    assert compute_capture_ratio(r) is None
    r2 = _record(final_profit_pct=1.0, max_mfe_pct=0.0)
    assert compute_capture_ratio(r2) is None


def test_compute_capture_ratio_negative_profit_allowed():
    r = _record(final_profit_pct=-5.0, max_mfe_pct=10.0)
    assert compute_capture_ratio(r) == pytest.approx(-0.5)


def test_compute_capture_result_bundles_censoring_and_ratio():
    r = compute_capture_result(_record(exit_reason="force_exit", final_profit_pct=0.03, max_mfe_pct=0.61))
    assert r.is_censored is True
    assert r.capture_ratio == pytest.approx(0.03 / 0.61)


def test_compute_capture_results_preserves_order():
    records = [_record(pair="BTC/USDC:USDC"), _record(pair="ETH/USDC:USDC")]
    results = compute_capture_results(records)
    assert [res.record.pair for res in results] == ["BTC/USDC:USDC", "ETH/USDC:USDC"]


# ---------------------------------------------------------------------------
# aggregation: censored excluded by default, zero-MFE excluded always
# ---------------------------------------------------------------------------


def test_aggregate_capture_excludes_censored_by_default():
    results = compute_capture_results([
        _record(final_profit_pct=5.0, max_mfe_pct=10.0, exit_reason="exit_signal"),
        _record(final_profit_pct=0.03, max_mfe_pct=0.61, exit_reason="force_exit"),
    ])
    agg = aggregate_capture(results)
    assert agg.n_total == 2
    assert agg.n_censored_excluded == 1
    assert agg.n_eligible == 1
    assert agg.median_capture_ratio == pytest.approx(0.5)


def test_aggregate_capture_include_censored_true():
    results = compute_capture_results([
        _record(final_profit_pct=5.0, max_mfe_pct=10.0, exit_reason="exit_signal"),
        _record(final_profit_pct=1.0, max_mfe_pct=2.0, exit_reason="force_exit"),
    ])
    agg = aggregate_capture(results, include_censored=True)
    assert agg.n_censored_excluded == 0
    assert agg.n_eligible == 2


def test_aggregate_capture_excludes_zero_mfe_from_ratios_but_counts_it():
    results = compute_capture_results([
        _record(final_profit_pct=5.0, max_mfe_pct=10.0),
        _record(final_profit_pct=0.0, max_mfe_pct=0.0),
    ])
    agg = aggregate_capture(results)
    assert agg.n_total == 2
    assert agg.n_zero_mfe_excluded == 1
    assert agg.n_eligible == 1


def test_aggregate_capture_empty_input():
    agg = aggregate_capture([])
    assert agg.n_total == 0
    assert agg.median_capture_ratio is None
    assert agg.mean_capture_ratio is None
    assert agg.min_capture_ratio is None
    assert agg.max_capture_ratio is None


def test_aggregate_capture_reports_min_max():
    results = compute_capture_results([
        _record(final_profit_pct=8.0, max_mfe_pct=10.0),
        _record(final_profit_pct=-4.0, max_mfe_pct=10.0),
        _record(final_profit_pct=2.0, max_mfe_pct=10.0),
    ])
    agg = aggregate_capture(results)
    assert agg.min_capture_ratio == pytest.approx(-0.4)
    assert agg.max_capture_ratio == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# aggregate_by_direction / aggregate_by_pair / aggregate_by_exit_reason
# ---------------------------------------------------------------------------


def test_aggregate_by_direction_splits_long_short():
    results = compute_capture_results([
        _record(direction="LONG", final_profit_pct=5.0, max_mfe_pct=10.0),
        _record(direction="SHORT", final_profit_pct=8.0, max_mfe_pct=10.0),
        _record(direction="SHORT", final_profit_pct=6.0, max_mfe_pct=10.0),
    ])
    by_dir = aggregate_by_direction(results)
    assert by_dir["LONG"].n_total == 1
    assert by_dir["SHORT"].n_total == 2


def test_aggregate_by_pair_splits_by_pair():
    results = compute_capture_results([
        _record(pair="BTC/USDC:USDC"), _record(pair="BTC/USDC:USDC"), _record(pair="ETH/USDC:USDC"),
    ])
    by_pair = aggregate_by_pair(results)
    assert by_pair["BTC/USDC:USDC"].n_total == 2
    assert by_pair["ETH/USDC:USDC"].n_total == 1


def test_aggregate_by_exit_reason_includes_force_exit_bucket_with_its_own_trades():
    results = compute_capture_results([
        _record(exit_reason="exit_signal", final_profit_pct=5.0, max_mfe_pct=10.0),
        _record(exit_reason="stop_loss", final_profit_pct=-5.0, max_mfe_pct=1.0),
        _record(exit_reason="force_exit", final_profit_pct=0.03, max_mfe_pct=0.61),
    ])
    by_reason = aggregate_by_exit_reason(results)
    assert set(by_reason.keys()) == {"exit_signal", "stop_loss", "force_exit"}
    # force_exit bucket must show its own trade, not be zeroed out by default censoring exclusion
    assert by_reason["force_exit"].n_total == 1
    assert by_reason["force_exit"].n_eligible == 1


# ---------------------------------------------------------------------------
# trajectory pattern classification
# ---------------------------------------------------------------------------


def test_trajectory_pattern_insufficient_data():
    pattern = classify_trajectory_pattern({"4h": 1.0, "7d": None, "14d": None, "21d": None, "30d": None})
    assert pattern.label == "insufficient_data"


def test_trajectory_pattern_early_plateau():
    pattern = classify_trajectory_pattern({"4h": 2.0, "7d": 25.0, "14d": 25.5, "21d": 25.8, "30d": 25.89})
    assert pattern.label == "early_plateau"


def test_trajectory_pattern_steady_increase():
    pattern = classify_trajectory_pattern({"4h": 1.0, "7d": 5.0, "14d": 10.0, "21d": 15.0, "30d": 20.0})
    assert pattern.label == "steady_increase"


def test_trajectory_pattern_late_acceleration():
    pattern = classify_trajectory_pattern({"4h": 1.0, "7d": 1.5, "14d": 2.0, "21d": 2.5, "30d": 30.0})
    assert pattern.label == "late_acceleration"


def test_trajectory_pattern_giveback_flagged():
    pattern = classify_trajectory_pattern({"4h": 5.0, "7d": 20.0, "14d": 10.0, "21d": 15.0, "30d": 18.0})
    assert pattern.label == "giveback"


def test_trajectory_pattern_only_uses_present_checkpoints():
    pattern = classify_trajectory_pattern({"4h": 1.0, "7d": None, "14d": 5.0, "21d": None, "30d": 10.0})
    assert pattern.checkpoints_used == ("4h", "14d", "30d")


# ---------------------------------------------------------------------------
# EMA-distance bucketing: observational only
# ---------------------------------------------------------------------------


def test_bucket_by_ema_distance_too_few_records():
    results = compute_capture_results([
        _record(mean_ema_distance_pct=-2.0), _record(mean_ema_distance_pct=-4.0),
    ])
    assert bucket_by_ema_distance(results) == []


def test_bucket_by_ema_distance_splits_into_three_groups():
    records = [
        _record(mean_ema_distance_pct=-1.0, max_mfe_pct=5.0, final_profit_pct=1.0, duration_days=3.0),
        _record(mean_ema_distance_pct=-2.0, max_mfe_pct=6.0, final_profit_pct=2.0, duration_days=4.0),
        _record(mean_ema_distance_pct=-3.0, max_mfe_pct=7.0, final_profit_pct=3.0, duration_days=5.0),
        _record(mean_ema_distance_pct=-8.0, max_mfe_pct=15.0, final_profit_pct=8.0, duration_days=20.0),
        _record(mean_ema_distance_pct=-9.0, max_mfe_pct=16.0, final_profit_pct=9.0, duration_days=25.0),
        _record(mean_ema_distance_pct=-10.0, max_mfe_pct=17.0, final_profit_pct=10.0, duration_days=30.0),
    ]
    results = compute_capture_results(records)
    buckets = bucket_by_ema_distance(results)
    assert [b.bucket_label for b in buckets] == ["low", "mid", "high"]
    assert sum(b.n for b in buckets) == 6
    # the "high" |ema_distance| bucket should show larger median duration
    # in this constructed example -- purely observational check of the
    # grouping mechanics, not a claim about real data.
    low = next(b for b in buckets if b.bucket_label == "low")
    high = next(b for b in buckets if b.bucket_label == "high")
    assert high.median_duration_days > low.median_duration_days


def test_bucket_by_ema_distance_skips_records_without_ema_distance():
    results = compute_capture_results([
        _record(mean_ema_distance_pct=-1.0), _record(mean_ema_distance_pct=None),
        _record(mean_ema_distance_pct=-2.0), _record(mean_ema_distance_pct=-3.0),
    ])
    buckets = bucket_by_ema_distance(results)
    assert sum(b.n for b in buckets) == 3


# ---------------------------------------------------------------------------
# real-shaped regression case: the BTC 2026-08-11 censored 2-candle trade
# ---------------------------------------------------------------------------


def test_btc_20260811_censored_trade_excluded_from_default_aggregate():
    censored = _record(
        pair="BTC/USDC:USDC", direction="SHORT", duration_days=0.17, pct_structurally_aligned=100.0,
        final_profit_pct=0.03116626076627989, max_mfe_pct=0.6133522057088936, exit_reason="force_exit",
    )
    normal = _record(
        pair="BTC/USDC:USDC", direction="SHORT", duration_days=60.33, pct_structurally_aligned=99.72,
        final_profit_pct=18.51, max_mfe_pct=27.49, exit_reason="exit_signal",
    )
    results = compute_capture_results([censored, normal])
    assert results[0].is_censored is True
    agg = aggregate_capture(results)
    assert agg.n_total == 2
    assert agg.n_censored_excluded == 1
    assert agg.n_eligible == 1
