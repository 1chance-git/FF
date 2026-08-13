"""Focused unit tests for `hermes.extended_baseline_report` (research-only)."""

from __future__ import annotations

from hermes.extended_baseline_report import (
    compare_summaries,
    compute_summary_stats,
    date_range,
    split_by_direction,
    split_by_pair,
)
from hermes.trade_report import Trade


def _trade(
    pair="BTC/USDC:USDC", direction="LONG", entry_time="2026-01-01T00:00:00Z",
    exit_time="2026-01-02T00:00:00Z", profit_pct=1.0, profit_abs=10.0,
    exit_reason="exit_signal",
):
    return Trade(
        pair=pair, direction=direction, entry_time=entry_time, exit_time=exit_time,
        entry_price=100.0, exit_price=101.0, enter_tag=None, exit_reason=exit_reason,
        profit_abs=profit_abs, profit_pct=profit_pct, duration_minutes=1440.0, is_open=False,
    )


# ---------------------------------------------------------------------------
# split_by_direction / split_by_pair
# ---------------------------------------------------------------------------


def test_split_by_direction():
    trades = [_trade(direction="LONG"), _trade(direction="SHORT"), _trade(direction="LONG")]
    assert len(split_by_direction(trades, "LONG")) == 2
    assert len(split_by_direction(trades, "SHORT")) == 1


def test_split_by_pair():
    trades = [_trade(pair="BTC/USDC:USDC"), _trade(pair="ETH/USDC:USDC")]
    assert len(split_by_pair(trades, "BTC/USDC:USDC")) == 1


# ---------------------------------------------------------------------------
# compute_summary_stats
# ---------------------------------------------------------------------------


def test_compute_summary_stats_basic():
    trades = [
        _trade(profit_pct=2.0, profit_abs=20.0, exit_reason="exit_signal"),
        _trade(profit_pct=-1.0, profit_abs=-10.0, exit_reason="stop_loss"),
        _trade(profit_pct=3.0, profit_abs=30.0, exit_reason="exit_signal"),
    ]
    stats = compute_summary_stats(trades)
    assert stats.n == 3
    assert stats.winners == 2
    assert stats.losers == 1
    assert stats.win_rate_pct == pytest_approx(66.6667)
    assert stats.total_profit_pct == pytest_approx(4.0)
    assert stats.avg_profit_pct == pytest_approx(4.0 / 3)
    assert stats.median_profit_pct == 2.0
    assert stats.profit_factor == pytest_approx(5.0)  # gross_profit=5.0 / abs(gross_loss)=1.0
    assert stats.stop_loss_count == 1
    assert stats.exit_signal_count == 2


def test_compute_summary_stats_empty():
    stats = compute_summary_stats([])
    assert stats.n == 0
    assert stats.winners == 0
    assert stats.win_rate_pct is None
    assert stats.total_profit_pct is None
    assert stats.profit_factor is None


def test_compute_summary_stats_no_losers_profit_factor_none():
    trades = [_trade(profit_pct=1.0, profit_abs=10.0)]
    stats = compute_summary_stats(trades)
    assert stats.profit_factor is None


def test_compute_summary_stats_excludes_open_trades_from_pnl_but_counts_n():
    open_trade = Trade(
        pair="BTC/USDC:USDC", direction="LONG", entry_time="2026-01-01T00:00:00Z",
        exit_time=None, entry_price=100.0, exit_price=None, enter_tag=None,
        exit_reason=None, profit_abs=None, profit_pct=None, duration_minutes=None, is_open=True,
    )
    trades = [_trade(profit_pct=1.0, profit_abs=10.0), open_trade]
    stats = compute_summary_stats(trades)
    assert stats.n == 2
    assert stats.winners == 1
    assert stats.total_profit_pct == pytest_approx(1.0)


# ---------------------------------------------------------------------------
# date_range
# ---------------------------------------------------------------------------


def test_date_range_basic():
    trades = [
        _trade(entry_time="2026-02-01T00:00:00Z", exit_time="2026-02-02T00:00:00Z"),
        _trade(entry_time="2026-01-01T00:00:00Z", exit_time="2026-03-01T00:00:00Z"),
    ]
    earliest, latest = date_range(trades)
    assert earliest == "2026-01-01T00:00:00Z"
    assert latest == "2026-03-01T00:00:00Z"


def test_date_range_empty():
    assert date_range([]) == (None, None)


# ---------------------------------------------------------------------------
# compare_summaries
# ---------------------------------------------------------------------------


def test_compare_summaries_deltas():
    extended = compute_summary_stats([
        _trade(profit_pct=2.0, profit_abs=20.0), _trade(profit_pct=2.0, profit_abs=20.0),
    ])
    baseline = compute_summary_stats([_trade(profit_pct=1.0, profit_abs=10.0)])
    comparison = compare_summaries(extended, baseline)
    assert comparison.n_delta == 1
    assert comparison.total_profit_pct_delta == pytest_approx(3.0)
    assert comparison.stop_loss_count_delta == 0


def test_compare_summaries_none_when_either_side_undefined():
    extended = compute_summary_stats([_trade(profit_pct=1.0, profit_abs=10.0)])  # no losers -> profit_factor None
    baseline = compute_summary_stats([
        _trade(profit_pct=1.0, profit_abs=10.0), _trade(profit_pct=-1.0, profit_abs=-10.0, exit_reason="stop_loss"),
    ])
    comparison = compare_summaries(extended, baseline)
    assert comparison.profit_factor_delta is None


def pytest_approx(value, rel=1e-3):
    import pytest
    return pytest.approx(value, rel=rel)
