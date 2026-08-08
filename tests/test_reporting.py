"""Unit tests for optimize.reporting.

Cross-checks compute_performance_report's output directly against the
underlying freqtrade.data.metrics calls it wraps, confirming it's a
thin, correct composition rather than a parallel reimplementation that
could silently drift from what those functions actually compute.
"""

from __future__ import annotations

import pandas as pd
import pytest

pytestmark = pytest.mark.unit

from freqtrade.data.metrics import (
    calculate_calmar,
    calculate_expectancy,
    calculate_max_drawdown,
    calculate_sharpe,
    calculate_sortino,
    calculate_sqn,
)

from optimize.reporting import (
    ReportingError,
    compute_performance_report,
    render_performance_report,
)


def make_trades() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    return pd.DataFrame(
        {
            "pair": ["ETH/USDC:USDC"] * 10 + ["BTC/USDC:USDC"] * 10,
            "close_date": dates,
            "profit_abs": [10, -5, 8, 12, -3, 6, 9, -8, 4, 11, -2, 7, 5, -4, 9, 3, -6, 8, 2, 10],
        }
    )


def test_compute_performance_report_rejects_empty_trades() -> None:
    with pytest.raises(ReportingError, match="zero trades"):
        compute_performance_report(pd.DataFrame(columns=["pair", "close_date", "profit_abs"]), 1000.0)


def test_total_trades_and_win_rate() -> None:
    trades = make_trades()
    report = compute_performance_report(trades, starting_balance=1000.0)

    assert report.total_trades == 20
    expected_win_rate = (trades["profit_abs"] > 0).mean()
    assert report.win_rate == pytest.approx(expected_win_rate)


def test_total_profit() -> None:
    trades = make_trades()
    report = compute_performance_report(trades, starting_balance=1000.0)

    assert report.total_profit_abs == pytest.approx(trades["profit_abs"].sum())
    assert report.total_profit_pct == pytest.approx(trades["profit_abs"].sum() / 1000.0)


def test_sharpe_matches_direct_freqtrade_call() -> None:
    trades = make_trades()
    report = compute_performance_report(trades, starting_balance=1000.0)

    expected = calculate_sharpe(
        trades, trades["close_date"].min(), trades["close_date"].max(), 1000.0
    )
    assert report.sharpe == pytest.approx(expected)


def test_sortino_matches_direct_freqtrade_call() -> None:
    trades = make_trades()
    report = compute_performance_report(trades, starting_balance=1000.0)

    expected = calculate_sortino(
        trades, trades["close_date"].min(), trades["close_date"].max(), 1000.0
    )
    assert report.sortino == pytest.approx(expected)


def test_calmar_matches_direct_freqtrade_call() -> None:
    trades = make_trades()
    report = compute_performance_report(trades, starting_balance=1000.0)

    expected = calculate_calmar(
        trades, trades["close_date"].min(), trades["close_date"].max(), 1000.0
    )
    assert report.calmar == pytest.approx(expected)


def test_max_drawdown_matches_direct_freqtrade_call() -> None:
    trades = make_trades()
    report = compute_performance_report(trades, starting_balance=1000.0)

    expected = calculate_max_drawdown(trades, starting_balance=1000.0, relative=True)
    assert report.max_drawdown_pct == pytest.approx(float(expected.relative_account_drawdown))
    assert report.max_drawdown_abs == pytest.approx(float(expected.drawdown_abs))


def test_expectancy_matches_direct_freqtrade_call() -> None:
    trades = make_trades()
    report = compute_performance_report(trades, starting_balance=1000.0)

    expected_expectancy, expected_ratio = calculate_expectancy(trades)
    assert report.expectancy == pytest.approx(float(expected_expectancy))
    assert report.expectancy_ratio == pytest.approx(float(expected_ratio))


def test_sqn_matches_direct_freqtrade_call() -> None:
    trades = make_trades()
    report = compute_performance_report(trades, starting_balance=1000.0)

    expected = calculate_sqn(trades, 1000.0)
    assert report.sqn == pytest.approx(float(expected))


def test_per_pair_profit_breakdown() -> None:
    trades = make_trades()
    report = compute_performance_report(trades, starting_balance=1000.0)

    eth_profit = trades[trades["pair"] == "ETH/USDC:USDC"]["profit_abs"].sum()
    btc_profit = trades[trades["pair"] == "BTC/USDC:USDC"]["profit_abs"].sum()

    assert report.per_pair_profit["ETH/USDC:USDC"] == pytest.approx(eth_profit)
    assert report.per_pair_profit["BTC/USDC:USDC"] == pytest.approx(btc_profit)


def test_custom_date_bounds_are_respected() -> None:
    trades = make_trades()
    narrow_min = trades["close_date"].iloc[5]
    narrow_max = trades["close_date"].iloc[15]

    report_full = compute_performance_report(trades, starting_balance=1000.0)
    report_narrow = compute_performance_report(
        trades, starting_balance=1000.0, min_date=narrow_min, max_date=narrow_max
    )

    # Sharpe depends on the date range even though the same trades are passed,
    # since it affects the annualization period.
    assert report_full.sharpe != report_narrow.sharpe


def test_render_performance_report_produces_a_table() -> None:
    from rich.table import Table

    trades = make_trades()
    report = compute_performance_report(trades, starting_balance=1000.0)

    table = render_performance_report(report, title="My Report")
    assert isinstance(table, Table)
    assert table.title == "My Report"
