"""Focused unit tests for `hermes.historical_data_audit` (research-only)."""

from __future__ import annotations

import pandas as pd
import pytest

from hermes.historical_data_audit import (
    analyze_coverage,
    check_baseline_coverage,
    check_timeframe_consistency,
    count_duplicates,
    find_gaps,
    longest_continuous_shared_window,
    shared_coverage,
)

TF = 240.0  # 4h in minutes


def _candles(dates):
    return pd.DataFrame({"date": [pd.Timestamp(d, tz="UTC") for d in dates], "close": [1.0] * len(dates)})


# ---------------------------------------------------------------------------
# count_duplicates
# ---------------------------------------------------------------------------


def test_count_duplicates_none():
    df = _candles(["2026-01-01T00:00", "2026-01-01T04:00"])
    assert count_duplicates(df) == 0


def test_count_duplicates_counts_extras():
    df = _candles(["2026-01-01T00:00", "2026-01-01T00:00", "2026-01-01T04:00", "2026-01-01T04:00"])
    assert count_duplicates(df) == 2


def test_count_duplicates_empty():
    assert count_duplicates(pd.DataFrame(columns=["date"])) == 0


# ---------------------------------------------------------------------------
# find_gaps
# ---------------------------------------------------------------------------


def test_find_gaps_none_when_consecutive():
    df = _candles(["2026-01-01T00:00", "2026-01-01T04:00", "2026-01-01T08:00"])
    assert find_gaps(df, TF) == []


def test_find_gaps_detects_single_gap():
    df = _candles(["2026-01-01T00:00", "2026-01-01T04:00", "2026-01-02T00:00"])
    gaps = find_gaps(df, TF)
    assert len(gaps) == 1
    assert gaps[0].after == pd.Timestamp("2026-01-01T04:00", tz="UTC")
    assert gaps[0].before == pd.Timestamp("2026-01-02T00:00", tz="UTC")
    assert gaps[0].missing_candles == 4  # 08:00,12:00,16:00,20:00 missing


def test_find_gaps_empty_and_single_row():
    assert find_gaps(pd.DataFrame(columns=["date"]), TF) == []
    assert find_gaps(_candles(["2026-01-01T00:00"]), TF) == []


def test_find_gaps_ignores_duplicates_not_gaps():
    df = _candles(["2026-01-01T00:00", "2026-01-01T00:00", "2026-01-01T04:00"])
    assert find_gaps(df, TF) == []


# ---------------------------------------------------------------------------
# check_timeframe_consistency
# ---------------------------------------------------------------------------


def test_timeframe_consistency_true_for_uniform_4h():
    df = _candles(["2026-01-01T00:00", "2026-01-01T04:00", "2026-01-01T08:00"])
    consistent, bad = check_timeframe_consistency(df, TF)
    assert consistent is True
    assert bad == ()


def test_timeframe_consistency_flags_offbeat_candle():
    df = _candles(["2026-01-01T00:00", "2026-01-01T00:05", "2026-01-01T04:05"])
    consistent, bad = check_timeframe_consistency(df, TF)
    assert consistent is False
    assert 5.0 in bad


def test_timeframe_consistency_ignores_whole_multiple_gaps():
    # an 8h gap (2x the 4h step) is a gap, not an inconsistency
    df = _candles(["2026-01-01T00:00", "2026-01-01T08:00"])
    consistent, _ = check_timeframe_consistency(df, TF)
    assert consistent is True


# ---------------------------------------------------------------------------
# analyze_coverage
# ---------------------------------------------------------------------------


def test_analyze_coverage_empty_df():
    report = analyze_coverage(pd.DataFrame(columns=["date"]), TF)
    assert report.n_rows == 0
    assert report.earliest is None
    assert report.latest is None
    assert report.n_gaps == 0


def test_analyze_coverage_basic_fields():
    df = _candles(["2026-01-01T00:00", "2026-01-01T00:00", "2026-01-01T04:00", "2026-01-02T00:00"])
    report = analyze_coverage(df, TF)
    assert report.n_rows == 4
    assert report.earliest == pd.Timestamp("2026-01-01T00:00", tz="UTC")
    assert report.latest == pd.Timestamp("2026-01-02T00:00", tz="UTC")
    assert report.n_duplicates == 1
    assert report.n_gaps == 1
    assert report.timeframe_consistent is True


# ---------------------------------------------------------------------------
# shared_coverage
# ---------------------------------------------------------------------------


def test_shared_coverage_intersection():
    btc = analyze_coverage(_candles(["2026-01-01T00:00", "2026-03-01T00:00"]), TF)
    eth = analyze_coverage(_candles(["2026-02-01T00:00", "2026-04-01T00:00"]), TF)
    start, end = shared_coverage(btc, eth)
    assert start == pd.Timestamp("2026-02-01T00:00", tz="UTC")
    assert end == pd.Timestamp("2026-03-01T00:00", tz="UTC")


def test_shared_coverage_none_when_no_overlap():
    btc = analyze_coverage(_candles(["2026-01-01T00:00", "2026-01-02T00:00"]), TF)
    eth = analyze_coverage(_candles(["2026-06-01T00:00", "2026-06-02T00:00"]), TF)
    assert shared_coverage(btc, eth) == (None, None)


def test_shared_coverage_none_when_either_empty():
    btc = analyze_coverage(pd.DataFrame(columns=["date"]), TF)
    eth = analyze_coverage(_candles(["2026-01-01T00:00"]), TF)
    assert shared_coverage(btc, eth) == (None, None)


# ---------------------------------------------------------------------------
# longest_continuous_shared_window
# ---------------------------------------------------------------------------


def test_longest_continuous_shared_window_no_gaps():
    btc = analyze_coverage(_candles(["2026-01-01T00:00", "2026-01-01T04:00", "2026-01-01T08:00"]), TF)
    eth = analyze_coverage(_candles(["2026-01-01T00:00", "2026-01-01T04:00", "2026-01-01T08:00"]), TF)
    window = longest_continuous_shared_window(btc, eth, TF)
    assert window.start == pd.Timestamp("2026-01-01T00:00", tz="UTC")
    assert window.end == pd.Timestamp("2026-01-01T08:00", tz="UTC")
    assert window.n_candles == 3


def test_longest_continuous_shared_window_respects_gap_in_one_series():
    btc = analyze_coverage(
        _candles(["2026-01-01T00:00", "2026-01-01T04:00", "2026-01-02T00:00", "2026-01-02T04:00"]), TF
    )
    eth = analyze_coverage(
        _candles(["2026-01-01T00:00", "2026-01-01T04:00", "2026-01-01T08:00", "2026-01-01T12:00"]), TF
    )
    window = longest_continuous_shared_window(btc, eth, TF)
    # BTC has a gap after 04:00 on day 1; the longest shared gap-free overlap
    # is bounded by that gap even though ETH keeps going gap-free.
    assert window.start == pd.Timestamp("2026-01-01T00:00", tz="UTC")
    assert window.end == pd.Timestamp("2026-01-01T04:00", tz="UTC")
    assert window.n_candles == 2


def test_longest_continuous_shared_window_none_when_no_data():
    btc = analyze_coverage(pd.DataFrame(columns=["date"]), TF)
    eth = analyze_coverage(pd.DataFrame(columns=["date"]), TF)
    window = longest_continuous_shared_window(btc, eth, TF)
    assert window.start is None
    assert window.n_candles == 0


# ---------------------------------------------------------------------------
# check_baseline_coverage
# ---------------------------------------------------------------------------


def test_baseline_coverage_fully_covered():
    result = check_baseline_coverage(
        shared_start=pd.Timestamp("2026-01-01", tz="UTC"),
        shared_end=pd.Timestamp("2026-12-01", tz="UTC"),
        baseline_start=pd.Timestamp("2026-03-01", tz="UTC"),
        baseline_end=pd.Timestamp("2026-06-01", tz="UTC"),
    )
    assert result.fully_covered is True
    assert result.missing_before is None
    assert result.missing_after is None


def test_baseline_coverage_missing_before_and_after():
    result = check_baseline_coverage(
        shared_start=pd.Timestamp("2026-03-01", tz="UTC"),
        shared_end=pd.Timestamp("2026-05-01", tz="UTC"),
        baseline_start=pd.Timestamp("2026-01-01", tz="UTC"),
        baseline_end=pd.Timestamp("2026-06-01", tz="UTC"),
    )
    assert result.fully_covered is False
    assert result.missing_before == pd.Timedelta(days=59)
    assert result.missing_after == pd.Timedelta(days=31)


def test_baseline_coverage_none_shared_is_not_covered():
    result = check_baseline_coverage(
        shared_start=None, shared_end=None,
        baseline_start=pd.Timestamp("2026-01-01", tz="UTC"),
        baseline_end=pd.Timestamp("2026-02-01", tz="UTC"),
    )
    assert result.fully_covered is False
