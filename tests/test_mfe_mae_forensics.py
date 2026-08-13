"""Focused unit tests for `hermes.mfe_mae_forensics` (research-only)."""

from __future__ import annotations

import pandas as pd
import pytest

from hermes.mfe_mae_forensics import (
    CATEGORY_FAVORABLE_THEN_REVERSAL,
    CATEGORY_IMMEDIATE_FAILURE,
    CATEGORY_SMALL_MOVEMENT,
    CATEGORY_SUCCESSFUL_TREND,
    CATEGORY_UNRESOLVED,
    LONG,
    SHORT,
    MfeMaeResult,
    classify_trade_development,
    compute_mfe_mae,
    mfe_threshold_breakdown,
    minutes_from_entry,
    slice_trade_window,
    summarize_mfe_mae,
)


def _candles(rows):
    """rows: list of (date_str, open, high, low, close)."""
    return pd.DataFrame(
        [{"date": pd.Timestamp(d, tz="UTC"), "open": o, "high": h, "low": l, "close": c} for d, o, h, l, c in rows]
    )


# ---------------------------------------------------------------------------
# slice_trade_window
# ---------------------------------------------------------------------------


def test_slice_trade_window_inclusive_of_both_endpoints():
    df = _candles([
        ("2026-01-01T00:00", 100, 105, 95, 102),
        ("2026-01-01T04:00", 102, 110, 100, 108),
        ("2026-01-01T08:00", 108, 112, 106, 110),
        ("2026-01-01T12:00", 110, 115, 108, 112),
    ])
    window = slice_trade_window(df, "2026-01-01T04:00", "2026-01-01T08:00")
    assert len(window) == 2
    assert window["date"].iloc[0] == pd.Timestamp("2026-01-01T04:00", tz="UTC")
    assert window["date"].iloc[-1] == pd.Timestamp("2026-01-01T08:00", tz="UTC")


def test_slice_trade_window_excludes_post_exit_candles():
    df = _candles([
        ("2026-01-01T00:00", 100, 105, 95, 102),
        ("2026-01-01T04:00", 102, 110, 100, 108),
        ("2026-01-01T08:00", 108, 112, 106, 110),
    ])
    window = slice_trade_window(df, "2026-01-01T00:00", "2026-01-01T04:00")
    assert list(window["date"]) == [pd.Timestamp("2026-01-01T00:00", tz="UTC"), pd.Timestamp("2026-01-01T04:00", tz="UTC")]
    assert pd.Timestamp("2026-01-01T08:00", tz="UTC") not in list(window["date"])


def test_slice_trade_window_none_on_empty_result():
    df = _candles([("2026-01-01T00:00", 100, 105, 95, 102)])
    assert slice_trade_window(df, "2026-06-01T00:00", "2026-06-02T00:00") is None


def test_slice_trade_window_none_on_missing_timestamps():
    df = _candles([("2026-01-01T00:00", 100, 105, 95, 102)])
    assert slice_trade_window(df, None, "2026-01-01T00:00") is None
    assert slice_trade_window(df, "2026-01-01T00:00", None) is None


def test_slice_trade_window_none_when_exit_before_entry():
    df = _candles([
        ("2026-01-01T00:00", 100, 105, 95, 102),
        ("2026-01-01T04:00", 102, 110, 100, 108),
    ])
    assert slice_trade_window(df, "2026-01-01T04:00", "2026-01-01T00:00") is None


# ---------------------------------------------------------------------------
# compute_mfe_mae -- LONG
# ---------------------------------------------------------------------------


def test_long_mfe_uses_high_and_mae_uses_low():
    candles = _candles([
        ("2026-01-01T00:00", 100, 103, 98, 101),   # entry candle
        ("2026-01-01T04:00", 101, 108, 96, 104),   # highest high=108, lowest low=96
        ("2026-01-01T08:00", 104, 106, 99, 100),
    ])
    result = compute_mfe_mae(LONG, 100.0, candles)
    assert result.mfe_pct == pytest.approx(8.0)  # (108-100)/100
    assert result.mae_pct == pytest.approx(4.0)  # (100-96)/100
    assert result.mfe_candle_index == 1
    assert result.mae_candle_index == 1
    assert result.same_candle_ambiguous is True
    assert result.n_candles == 3


def test_long_mfe_mae_different_candles_not_ambiguous():
    candles = _candles([
        ("2026-01-01T00:00", 100, 103, 98, 101),
        ("2026-01-01T04:00", 101, 110, 100, 105),  # MFE here
        ("2026-01-01T08:00", 105, 106, 90, 95),    # MAE here
    ])
    result = compute_mfe_mae(LONG, 100.0, candles)
    assert result.mfe_candle_index == 1
    assert result.mae_candle_index == 2
    assert result.same_candle_ambiguous is False


# ---------------------------------------------------------------------------
# compute_mfe_mae -- SHORT
# ---------------------------------------------------------------------------


def test_short_mfe_uses_low_and_mae_uses_high():
    candles = _candles([
        ("2026-01-01T00:00", 100, 102, 97, 99),
        ("2026-01-01T04:00", 99, 105, 90, 95),  # lowest low=90 (favorable for short), highest high=105 (adverse)
    ])
    result = compute_mfe_mae(SHORT, 100.0, candles)
    assert result.mfe_pct == pytest.approx(10.0)  # (100-90)/100
    assert result.mae_pct == pytest.approx(5.0)   # (105-100)/100


# ---------------------------------------------------------------------------
# compute_mfe_mae -- unresolved / missing data
# ---------------------------------------------------------------------------


def test_compute_mfe_mae_unresolved_on_unknown_direction():
    candles = _candles([("2026-01-01T00:00", 100, 103, 98, 101)])
    result = compute_mfe_mae("FLAT", 100.0, candles)
    assert result.is_resolved is False
    assert result.unresolved_reason == "unknown_direction"
    assert result.mfe_pct is None


def test_compute_mfe_mae_unresolved_on_missing_entry_price():
    candles = _candles([("2026-01-01T00:00", 100, 103, 98, 101)])
    result = compute_mfe_mae(LONG, None, candles)
    assert result.is_resolved is False
    assert result.unresolved_reason == "missing_entry_price"


def test_compute_mfe_mae_unresolved_on_none_candles():
    result = compute_mfe_mae(LONG, 100.0, None)
    assert result.is_resolved is False
    assert result.unresolved_reason == "no_candle_window"


def test_compute_mfe_mae_unresolved_on_empty_candles():
    result = compute_mfe_mae(LONG, 100.0, pd.DataFrame(columns=["date", "high", "low"]))
    assert result.is_resolved is False
    assert result.unresolved_reason == "no_candle_window"


def test_compute_mfe_mae_never_fabricates_zero_on_missing_data():
    result = compute_mfe_mae(LONG, 100.0, None)
    assert result.mfe_pct is None
    assert result.mae_pct is None
    assert result.mfe_pct != 0.0


# ---------------------------------------------------------------------------
# minutes_from_entry
# ---------------------------------------------------------------------------


def test_minutes_from_entry_execution_candle_is_zero():
    assert minutes_from_entry(0, 240.0) == 0.0


def test_minutes_from_entry_scales_by_timeframe():
    assert minutes_from_entry(3, 240.0) == 720.0


def test_minutes_from_entry_none_index():
    assert minutes_from_entry(None, 240.0) is None


# ---------------------------------------------------------------------------
# classify_trade_development
# ---------------------------------------------------------------------------


def _result(mfe, mae):
    return MfeMaeResult(
        direction=LONG, mfe_pct=mfe, mae_pct=mae,
        mfe_candle_index=0, mfe_candle_time="t", mae_candle_index=0, mae_candle_time="t",
        n_candles=1, same_candle_ambiguous=False, unresolved_reason=None,
    )


def test_classify_small_movement():
    assert classify_trade_development(_result(0.3, 0.2), is_winner=False) == CATEGORY_SMALL_MOVEMENT


def test_classify_immediate_failure():
    assert classify_trade_development(_result(0.2, 5.0), is_winner=False) == CATEGORY_IMMEDIATE_FAILURE


def test_classify_favorable_then_reversal():
    assert classify_trade_development(_result(3.0, 5.0), is_winner=False) == CATEGORY_FAVORABLE_THEN_REVERSAL


def test_classify_successful_trend():
    assert classify_trade_development(_result(3.0, 0.5), is_winner=True) == CATEGORY_SUCCESSFUL_TREND


def test_classify_unresolved_on_unknown_outcome_with_meaningful_mfe():
    assert classify_trade_development(_result(3.0, 0.5), is_winner=None) == CATEGORY_UNRESOLVED


def test_classify_unresolved_when_result_itself_unresolved():
    unresolved = MfeMaeResult(
        direction=LONG, mfe_pct=None, mae_pct=None,
        mfe_candle_index=None, mfe_candle_time=None, mae_candle_index=None, mae_candle_time=None,
        n_candles=0, same_candle_ambiguous=False, unresolved_reason="no_candle_window",
    )
    assert classify_trade_development(unresolved, is_winner=True) == CATEGORY_UNRESOLVED


def test_classify_respects_custom_threshold():
    # mfe=1.5 is meaningful at default 1.0 threshold but not at 2.0
    assert classify_trade_development(_result(1.5, 0.5), is_winner=True) == CATEGORY_SUCCESSFUL_TREND
    assert classify_trade_development(_result(1.5, 0.5), is_winner=True, threshold_pct=2.0) == CATEGORY_SMALL_MOVEMENT


# ---------------------------------------------------------------------------
# summarize_mfe_mae / mfe_threshold_breakdown
# ---------------------------------------------------------------------------


def test_summarize_mfe_mae_excludes_unresolved():
    results = [
        _result(2.0, 1.0),
        _result(4.0, 3.0),
        MfeMaeResult(LONG, None, None, None, None, None, None, 0, False, "no_candle_window"),
    ]
    summary = summarize_mfe_mae(results)
    assert summary.n == 2
    assert summary.mean_mfe_pct == pytest.approx(3.0)
    assert summary.median_mfe_pct == pytest.approx(3.0)
    assert summary.mean_mae_pct == pytest.approx(2.0)


def test_summarize_mfe_mae_all_unresolved_returns_none():
    results = [MfeMaeResult(LONG, None, None, None, None, None, None, 0, False, "no_candle_window")]
    summary = summarize_mfe_mae(results)
    assert summary.n == 0
    assert summary.mean_mfe_pct is None


def test_mfe_threshold_breakdown_counts_correctly():
    results = [_result(0.5, 0.1), _result(1.5, 0.1), _result(3.5, 0.1), _result(6.0, 0.1)]
    breakdown = mfe_threshold_breakdown(results, [1.0, 2.0, 3.0, 5.0])
    assert breakdown == {1.0: 3, 2.0: 2, 3.0: 2, 5.0: 1}
