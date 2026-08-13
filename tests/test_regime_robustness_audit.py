"""Focused unit tests for `hermes.regime_robustness_audit` (research-only)."""

from __future__ import annotations

from hermes.regime_robustness_audit import (
    THIN_SAMPLE_THRESHOLD,
    build_all_quarter_reports,
    build_quarter_report,
    chronological_quarters,
    direction_breakdown,
    outlier_removal_pnl,
    stop_loss_rates,
)
from hermes.trade_report import Trade


def _trade(
    entry_time, pair="BTC/USDC:USDC", direction="LONG", profit_pct=1.0, profit_abs=10.0,
    exit_reason="exit_signal",
):
    return Trade(
        pair=pair, direction=direction, entry_time=entry_time, exit_time=entry_time,
        entry_price=100.0, exit_price=101.0, enter_tag=None, exit_reason=exit_reason,
        profit_abs=profit_abs, profit_pct=profit_pct, duration_minutes=240.0, is_open=False,
    )


# ---------------------------------------------------------------------------
# chronological_quarters
# ---------------------------------------------------------------------------


def test_chronological_quarters_even_split():
    trades = [_trade(f"2026-01-{i:02d}T00:00:00Z") for i in range(1, 9)]  # 8 trades
    quarters = chronological_quarters(trades, 4)
    assert [len(q) for q in quarters] == [2, 2, 2, 2]
    assert quarters[0][0].entry_time == "2026-01-01T00:00:00Z"
    assert quarters[3][-1].entry_time == "2026-01-08T00:00:00Z"


def test_chronological_quarters_uneven_remainder_goes_to_earliest():
    trades = [_trade(f"2026-01-{i:02d}T00:00:00Z") for i in range(1, 10)]  # 9 trades / 4
    quarters = chronological_quarters(trades, 4)
    assert [len(q) for q in quarters] == [3, 2, 2, 2]


def test_chronological_quarters_sorts_by_entry_time_regardless_of_input_order():
    trades = [
        _trade("2026-01-03T00:00:00Z"),
        _trade("2026-01-01T00:00:00Z"),
        _trade("2026-01-02T00:00:00Z"),
        _trade("2026-01-04T00:00:00Z"),
    ]
    quarters = chronological_quarters(trades, 4)
    assert [q[0].entry_time for q in quarters] == [
        "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z",
        "2026-01-03T00:00:00Z", "2026-01-04T00:00:00Z",
    ]


def test_chronological_quarters_excludes_none_entry_time():
    trades = [_trade("2026-01-01T00:00:00Z"), _trade(None), _trade("2026-01-02T00:00:00Z")]
    quarters = chronological_quarters(trades, 2)
    assert sum(len(q) for q in quarters) == 2


def test_chronological_quarters_empty_input():
    quarters = chronological_quarters([], 4)
    assert quarters == [[], [], [], []]


# ---------------------------------------------------------------------------
# outlier_removal_pnl
# ---------------------------------------------------------------------------


def test_outlier_removal_pnl_basic():
    trades = [
        _trade("2026-01-01T00:00:00Z", profit_pct=10.0),
        _trade("2026-01-02T00:00:00Z", profit_pct=5.0),
        _trade("2026-01-03T00:00:00Z", profit_pct=-2.0),
    ]
    result = outlier_removal_pnl(trades)
    assert result.largest_winner_pct == 10.0
    assert result.second_largest_winner_pct == 5.0
    assert result.total_pnl_pct == 13.0
    assert result.total_pnl_excl_top1_pct == 3.0
    assert result.total_pnl_excl_top2_pct == -2.0


def test_outlier_removal_pnl_fewer_than_two_winners():
    trades = [_trade("2026-01-01T00:00:00Z", profit_pct=3.0), _trade("2026-01-02T00:00:00Z", profit_pct=-1.0)]
    result = outlier_removal_pnl(trades)
    assert result.largest_winner_pct == 3.0
    assert result.second_largest_winner_pct is None
    assert result.total_pnl_excl_top2_pct == result.total_pnl_excl_top1_pct


def test_outlier_removal_pnl_empty():
    result = outlier_removal_pnl([])
    assert result.total_pnl_pct is None
    assert result.largest_winner_pct is None


# ---------------------------------------------------------------------------
# direction_breakdown
# ---------------------------------------------------------------------------


def test_direction_breakdown_basic():
    trades = [
        _trade("2026-01-01T00:00:00Z", direction="LONG", profit_pct=2.0, exit_reason="stop_loss"),
        _trade("2026-01-02T00:00:00Z", direction="LONG", profit_pct=-1.0, exit_reason="exit_signal"),
        _trade("2026-01-03T00:00:00Z", direction="SHORT", profit_pct=4.0, exit_reason="exit_signal"),
    ]
    long_result = direction_breakdown(trades, "LONG")
    assert long_result.n == 2
    assert long_result.total_profit_pct == 1.0
    assert long_result.stop_loss_count == 1

    short_result = direction_breakdown(trades, "SHORT")
    assert short_result.n == 1
    assert short_result.stop_loss_count == 0


def test_direction_breakdown_empty_direction():
    trades = [_trade("2026-01-01T00:00:00Z", direction="LONG")]
    result = direction_breakdown(trades, "SHORT")
    assert result.n == 0
    assert result.total_profit_pct is None


# ---------------------------------------------------------------------------
# stop_loss_rates
# ---------------------------------------------------------------------------


def test_stop_loss_rates_basic():
    trades = [
        _trade("2026-01-01T00:00:00Z", direction="LONG", exit_reason="stop_loss"),
        _trade("2026-01-02T00:00:00Z", direction="LONG", exit_reason="exit_signal"),
        _trade("2026-01-03T00:00:00Z", direction="SHORT", exit_reason="stop_loss"),
        _trade("2026-01-04T00:00:00Z", direction="SHORT", exit_reason="stop_loss"),
    ]
    result = stop_loss_rates(trades)
    assert result.overall_rate_pct == 75.0
    assert result.long_rate_pct == 50.0
    assert result.short_rate_pct == 100.0


def test_stop_loss_rates_none_for_empty_direction():
    trades = [_trade("2026-01-01T00:00:00Z", direction="LONG", exit_reason="stop_loss")]
    result = stop_loss_rates(trades)
    assert result.short_rate_pct is None


def test_stop_loss_rates_empty_trades():
    result = stop_loss_rates([])
    assert result.overall_rate_pct is None
    assert result.long_rate_pct is None
    assert result.short_rate_pct is None


# ---------------------------------------------------------------------------
# build_quarter_report / build_all_quarter_reports
# ---------------------------------------------------------------------------


def test_build_quarter_report_thin_sample_flag():
    few_trades = [_trade(f"2026-01-{i:02d}T00:00:00Z") for i in range(1, 4)]  # 3 trades
    report = build_quarter_report(1, few_trades)
    assert report.is_thin_sample is True
    assert report.n_trades == 3

    many_trades = [_trade(f"2026-01-{i:02d}T00:00:00Z") for i in range(1, 8)]  # 7 trades
    report2 = build_quarter_report(2, many_trades)
    assert report2.is_thin_sample is False
    assert THIN_SAMPLE_THRESHOLD == 5


def test_build_quarter_report_computes_max_winner_and_loser():
    trades = [
        _trade("2026-01-01T00:00:00Z", profit_pct=8.0),
        _trade("2026-01-02T00:00:00Z", profit_pct=-3.0),
        _trade("2026-01-03T00:00:00Z", profit_pct=2.0),
    ]
    report = build_quarter_report(1, trades)
    assert report.max_winner_pct == 8.0
    assert report.max_loser_pct == -3.0
    assert report.period_start == "2026-01-01T00:00:00Z"
    assert report.period_end == "2026-01-03T00:00:00Z"


def test_build_quarter_report_empty_quarter():
    report = build_quarter_report(1, [])
    assert report.n_trades == 0
    assert report.is_thin_sample is True
    assert report.max_winner_pct is None
    assert report.period_start is None


def test_build_all_quarter_reports_count_and_order():
    trades = [_trade(f"2026-01-{i:02d}T00:00:00Z") for i in range(1, 13)]  # 12 trades
    reports = build_all_quarter_reports(trades, 4)
    assert len(reports) == 4
    assert [r.index for r in reports] == [1, 2, 3, 4]
    assert sum(r.n_trades for r in reports) == 12


def test_build_all_quarter_reports_directional_split_within_quarter():
    trades = [
        _trade("2026-01-01T00:00:00Z", direction="LONG", profit_pct=1.0),
        _trade("2026-01-02T00:00:00Z", direction="SHORT", profit_pct=3.0),
    ]
    reports = build_all_quarter_reports(trades, 1)
    assert reports[0].long.n == 1
    assert reports[0].short.n == 1
    assert reports[0].summary.n == 2
