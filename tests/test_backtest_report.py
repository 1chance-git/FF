"""Tests for hermes/backtest_report.py and the `hermes backtest-report` CLI.

Every test here is read-only with respect to backtesting: **no test in
this module runs a Freqtrade backtest, launches a subprocess, or touches
an exchange.** Realistic fixtures are produced by calling Freqtrade's
own *formatting* functions (`text_table_bt_results` /
`text_table_add_metrics`) with synthetic stats dicts — pure rendering,
no engine involved. That choice is deliberate: pinning the parser to
the installed Freqtrade's real output means a future Freqtrade upgrade
that changes the rendering fails these tests loudly, instead of
silently degrading the report.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest
from click.testing import CliRunner

from hermes.backtest_report import (
    BacktestReport,
    PairResult,
    extract_stdout,
    find_table,
    parse_backtest_stdout,
    parse_rich_tables,
)
from hermes.cli import cli, render_backtest_report
from hermes.memory import BacktestResult as MemoryBacktestResult
from hermes.memory import MemoryStore

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixture generation via Freqtrade's own formatters (no backtest executed)
# ---------------------------------------------------------------------------


def _pair_rows() -> list[dict]:
    return [
        {
            "key": "BTC/USDC:USDC", "trades": 12, "profit_mean_pct": 0.51,
            "profit_total_abs": 61.2345, "profit_total_pct": 6.12,
            "duration_avg": "1 day, 4:00:00", "wins": 7, "draws": 0, "losses": 5,
        },
        {
            "key": "ETH/USDC:USDC", "trades": 9, "profit_mean_pct": -0.22,
            "profit_total_abs": -19.5, "profit_total_pct": -1.95,
            "duration_avg": "0 days 16:00:00", "wins": 3, "draws": 0, "losses": 6,
        },
        {
            "key": "SOL/USDC:USDC", "trades": 0, "profit_mean_pct": 0.0,
            "profit_total_abs": 0.0, "profit_total_pct": 0.0,
            "duration_avg": "0:00:00", "wins": 0, "draws": 0, "losses": 0,
        },
        {
            "key": "TOTAL", "trades": 21, "profit_mean_pct": 0.20,
            "profit_total_abs": 41.7345, "profit_total_pct": 4.17,
            "duration_avg": "1 day, 0:00:00", "wins": 10, "draws": 0, "losses": 11,
        },
    ]


def _summary_stats() -> dict:
    return {
        "stake_currency": "USDC",
        "trades": [
            {"profit_ratio": 0.05, "pair": "BTC/USDC:USDC"},
            {"profit_ratio": -0.02, "pair": "ETH/USDC:USDC"},
        ],
        "trade_count_long": 13, "trade_count_short": 8,
        "profit_total_long": 0.031, "profit_total_short": -0.009,
        "profit_total_long_abs": 31.0, "profit_total_short_abs": -9.0,
        "max_relative_drawdown": 0.0812, "max_drawdown_account": 0.0745,
        "max_drawdown_abs": 74.5, "drawdown_duration": "5 days 08:00:00",
        "max_drawdown_high": 120.0, "max_drawdown_low": 45.5,
        "drawdown_start": "2026-03-02 12:00:00", "drawdown_end": "2026-03-07 20:00:00",
        "backtest_start": "2026-01-15 00:00:00", "backtest_end": "2026-08-11 00:00:00",
        "max_open_trades": 3, "total_trades": 21, "trades_per_day": 0.1,
        "starting_balance": 1000.0, "final_balance": 1041.73,
        "profit_total_abs": 41.73, "profit_total": 0.0417,
        "cagr": 0.0731, "sharpe": 0.84, "sortino": 1.12, "calmar": 0.55,
        "sqn": 0.42, "p_value": 0.3312, "profit_factor": 1.23,
        "expectancy": 1.98, "expectancy_ratio": 0.11, "winrate": 0.476,
        "trading_mode": "futures", "margin_mode": "isolated",
        "csum_min": 980.0, "csum_max": 1120.0,
        "avg_stake_amount": 330.0, "total_volume": 6930.0,
        "best_pair": {"key": "BTC/USDC:USDC", "profit_total_abs": 61.23, "profit_total": 0.0612},
        "worst_pair": {"key": "ETH/USDC:USDC", "profit_total_abs": -19.5, "profit_total": -0.0195},
        "winner_holding_avg": "1 day, 6:00:00", "loser_holding_avg": "0 days 14:00:00",
        "holding_avg": "1 day, 0:00:00", "backtest_days": 209, "market_change": 0.15,
        "backtest_best_day_abs": 25.0, "backtest_worst_day_abs": -30.0,
        "backtest_best_day": 0.025, "backtest_worst_day": -0.03,
        "winning_days": 40, "draw_days": 100, "losing_days": 69,
    }


def freqtrade_rendered_stdout() -> str:
    """Realistic backtest stdout, rendered by Freqtrade's own formatters.

    Pure formatting: no backtest, no subprocess, no exchange access.
    """
    from freqtrade.optimize.optimize_reports.bt_output import (
        text_table_add_metrics,
        text_table_bt_results,
    )

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        text_table_bt_results(_pair_rows(), "USDC", "BACKTESTING REPORT")
        text_table_add_metrics(_summary_stats())
    return buffer.getvalue()


@pytest.fixture(scope="module")
def rendered_stdout() -> str:
    return freqtrade_rendered_stdout()


@pytest.fixture(scope="module")
def parsed(rendered_stdout: str) -> BacktestReport:
    return parse_backtest_stdout(rendered_stdout)


# ---------------------------------------------------------------------------
# Generic rich-table parsing
# ---------------------------------------------------------------------------


class TestParseRichTables:
    def test_finds_both_freqtrade_tables(self, rendered_stdout: str) -> None:
        tables = parse_rich_tables(rendered_stdout)
        titles = [table.title for table in tables]
        assert "BACKTESTING REPORT" in titles
        assert "SUMMARY METRICS" in titles

    def test_headers_are_recovered(self, rendered_stdout: str) -> None:
        table = find_table(parse_rich_tables(rendered_stdout), "BACKTESTING REPORT")
        assert table is not None
        assert table.headers[0] == "Pair"
        assert table.headers[1] == "Trades"

    def test_row_cells_are_stripped(self, rendered_stdout: str) -> None:
        table = find_table(parse_rich_tables(rendered_stdout), "BACKTESTING REPORT")
        assert table is not None
        first_cells = [row[0] for row in table.rows]
        assert "BTC/USDC:USDC" in first_cells  # no surrounding padding

    def test_find_table_is_case_insensitive(self, rendered_stdout: str) -> None:
        tables = parse_rich_tables(rendered_stdout)
        assert find_table(tables, "backtesting report") is not None

    def test_find_table_returns_none_for_unknown_title(self, rendered_stdout: str) -> None:
        assert find_table(parse_rich_tables(rendered_stdout), "NO SUCH TABLE") is None

    def test_text_without_tables_yields_nothing(self) -> None:
        assert parse_rich_tables("just some log lines\nand another\n") == []

    def test_empty_text_yields_nothing(self) -> None:
        assert parse_rich_tables("") == []


# ---------------------------------------------------------------------------
# Per-pair extraction
# ---------------------------------------------------------------------------


class TestPairResults:
    def test_total_row_is_separated_from_pair_rows(self, parsed: BacktestReport) -> None:
        assert [p.pair for p in parsed.pair_results] == [
            "BTC/USDC:USDC",
            "ETH/USDC:USDC",
            "SOL/USDC:USDC",
        ]
        assert parsed.total is not None
        assert parsed.total.pair == "TOTAL"

    def test_pair_numbers_match_what_freqtrade_printed(self, parsed: BacktestReport) -> None:
        btc = parsed.pair_results[0]
        assert btc.trades == 12
        assert btc.wins == 7
        assert btc.draws == 0
        assert btc.losses == 5
        assert btc.win_rate_pct == pytest.approx(58.3)
        assert btc.profit_total_abs == pytest.approx(61.234)
        assert btc.avg_duration == "1 day, 4:00:00"

    def test_negative_profit_is_parsed_with_sign(self, parsed: BacktestReport) -> None:
        eth = parsed.pair_results[1]
        assert eth.profit_total_abs == pytest.approx(-19.5)
        assert eth.profit_total_pct == pytest.approx(-1.95)

    def test_zero_trade_pair_is_kept_not_dropped(self, parsed: BacktestReport) -> None:
        sol = parsed.pair_results[2]
        assert sol.pair == "SOL/USDC:USDC"
        assert sol.trades == 0
        assert sol.wins == 0 and sol.losses == 0

    def test_trades_for_pair_lookup(self, parsed: BacktestReport) -> None:
        assert parsed.trades_for_pair("BTC/USDC:USDC") == 12
        assert parsed.trades_for_pair("SOL/USDC:USDC") == 0

    def test_trades_for_unknown_pair_is_none_not_zero(self, parsed: BacktestReport) -> None:
        """A pair that was never traded must be distinguishable from one
        that isn't in the report at all."""
        assert parsed.trades_for_pair("XRP/USDC:USDC") is None


# ---------------------------------------------------------------------------
# Summary metrics + normalized accessors
# ---------------------------------------------------------------------------


class TestSummaryMetrics:
    def test_totals_come_from_the_total_row(self, parsed: BacktestReport) -> None:
        assert parsed.total_trades == 21
        assert parsed.wins == 10
        assert parsed.losses == 11
        assert parsed.draws == 0
        assert parsed.win_rate_pct == pytest.approx(47.6)

    def test_profit_and_factor(self, parsed: BacktestReport) -> None:
        assert parsed.profit_total_abs == pytest.approx(41.734)
        assert parsed.profit_total_pct == pytest.approx(4.17)
        assert parsed.profit_factor == pytest.approx(1.23)

    def test_drawdown_is_reported_verbatim(self, parsed: BacktestReport) -> None:
        assert parsed.max_drawdown == "74.5 USDC (7.45%)"

    def test_long_short_split(self, parsed: BacktestReport) -> None:
        assert parsed.long_trades == 13
        assert parsed.short_trades == 8

    def test_balances(self, parsed: BacktestReport) -> None:
        assert parsed.starting_balance == "1000 USDC"
        assert parsed.final_balance == "1041.73 USDC"

    def test_avg_trade_duration(self, parsed: BacktestReport) -> None:
        assert parsed.avg_trade_duration == "1 day, 0:00:00"

    def test_na_metric_is_reported_as_missing_not_as_the_string_na(
        self, parsed: BacktestReport
    ) -> None:
        """Freqtrade prints 'N/A' for metrics it didn't compute; those must
        surface as None so the report never presents them as values."""
        assert parsed.metric("Max Consecutive Wins / Loss") is None

    def test_unknown_metric_is_none(self, parsed: BacktestReport) -> None:
        assert parsed.metric("Totally Invented Metric") is None


class TestMissingDataIsNotInvented:
    def test_empty_stdout_parses_to_an_empty_report(self) -> None:
        report = parse_backtest_stdout("")
        assert report.parsed_anything is False
        assert report.total_trades is None
        assert report.profit_factor is None
        assert report.max_drawdown is None
        assert report.pair_results == ()

    def test_stdout_without_tables_parses_to_an_empty_report(self) -> None:
        report = parse_backtest_stdout("ERROR - No data found. Terminating.\n")
        assert report.parsed_anything is False
        assert report.total_trades is None

    def test_accessors_are_none_without_a_total_row(self) -> None:
        report = BacktestReport(
            pair_results=(
                PairResult(
                    pair="BTC/USDC:USDC", trades=3, avg_profit_pct=None,
                    profit_total_abs=None, profit_total_pct=None,
                    avg_duration="", wins=1, draws=0, losses=2, win_rate_pct=None,
                ),
            )
        )
        assert report.wins is None
        assert report.losses is None
        assert report.win_rate_pct is None
        assert report.avg_trade_duration is None


class TestExtractStdout:
    def test_reads_the_key_the_launcher_writes(self) -> None:
        assert extract_stdout({"stdout": "results"}) == "results"

    def test_missing_key_is_none(self) -> None:
        assert extract_stdout({"exit_code": 0}) is None

    def test_empty_string_is_none(self) -> None:
        assert extract_stdout({"stdout": ""}) is None

    def test_non_string_is_none(self) -> None:
        assert extract_stdout({"stdout": 42}) is None

    def test_empty_or_missing_metrics_is_none(self) -> None:
        assert extract_stdout({}) is None
        assert extract_stdout(None) is None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _record(metrics: dict, strategy: str = "TrendFollowCore") -> MemoryBacktestResult:
    return MemoryBacktestResult(
        strategy=strategy, timerange="20260115-20260811", metrics=metrics
    )


class TestRenderBacktestReport:
    def test_renders_the_real_numbers(self, rendered_stdout: str, parsed: BacktestReport) -> None:
        text = render_backtest_report(
            _record({"exit_code": 0, "succeeded": True, "timeframe": "4h",
                     "stdout": rendered_stdout}),
            parsed,
        )
        assert "[HERMES][BACKTEST REPORT]" in text
        assert "STRATEGY: TrendFollowCore" in text
        assert "TIMERANGE: 20260115-20260811" in text
        assert "TRADES: 21" in text
        assert "WIN RATE: 47.6%" in text
        assert "PROFIT FACTOR: 1.23" in text
        assert "MAX DRAWDOWN: 74.5 USDC (7.45%)" in text
        assert "LONG TRADES: 13" in text
        assert "SHORT TRADES: 8" in text
        assert "BTC/USDC:USDC: 12 trades" in text
        assert "STATUS: OBSERVATION ONLY" in text

    def test_renders_missing_fields_as_na(self) -> None:
        """A report that has *some* metrics but not others must render the
        absent ones as N/A — never as 0, blank, or a guessed value."""
        report = BacktestReport(summary_metrics={"Profit factor": "N/A"})
        assert report.parsed_anything is True  # guard: takes the full render path

        text = render_backtest_report(
            _record({"exit_code": 0, "succeeded": True, "stdout": "x"}), report
        )
        assert "PROFIT FACTOR: N/A" in text
        assert "MAX DRAWDOWN: N/A" in text
        assert "WIN RATE: N/A" in text
        assert "TRADES: N/A" in text

    def test_empty_report_says_so_rather_than_printing_zeroes(self) -> None:
        text = render_backtest_report(
            _record({"exit_code": 2, "succeeded": False, "stdout": "boom"}),
            BacktestReport(),
        )
        assert "STATUS: NO RESULTS RECORDED" in text
        assert "EXIT CODE: 2" in text
        assert "TRADES: 0" not in text  # must not fabricate a zero


# ---------------------------------------------------------------------------
# CLI (read-only; never launches a backtest)
# ---------------------------------------------------------------------------


def _seed_store(tmp_path: Path, metrics: dict, strategy: str = "TrendFollowCore") -> Path:
    user_data = tmp_path / "user_data"
    user_data.mkdir()
    store = MemoryStore(user_data / "hermes_memory.sqlite3")
    store.record_backtest_result(
        MemoryBacktestResult(
            strategy=strategy, timerange="20260115-20260811", metrics=metrics
        )
    )
    return user_data


class TestBacktestReportCommand:
    def test_command_is_registered(self) -> None:
        result = CliRunner().invoke(cli, ["--help"])
        assert "backtest-report" in result.output

    def test_reports_a_recorded_backtest(self, tmp_path: Path, rendered_stdout: str) -> None:
        user_data = _seed_store(
            tmp_path,
            {"exit_code": 0, "succeeded": True, "timeframe": "4h", "stdout": rendered_stdout},
        )
        result = CliRunner().invoke(
            cli, ["backtest-report", "--user-data-dir", str(user_data)]
        )
        assert result.exit_code == 0, result.output
        assert "TRADES: 21" in result.output
        assert "WIN RATE: 47.6%" in result.output
        assert "BTC/USDC:USDC: 12 trades" in result.output

    def test_reports_most_recent_when_several_exist(
        self, tmp_path: Path, rendered_stdout: str
    ) -> None:
        user_data = _seed_store(tmp_path, {"exit_code": 2, "succeeded": False, "stdout": "old"})
        store = MemoryStore(user_data / "hermes_memory.sqlite3")
        store.record_backtest_result(
            MemoryBacktestResult(
                strategy="TrendFollowCore",
                timerange="20260115-20260811",
                metrics={"exit_code": 0, "succeeded": True, "stdout": rendered_stdout},
            )
        )
        result = CliRunner().invoke(
            cli, ["backtest-report", "--user-data-dir", str(user_data)]
        )
        assert result.exit_code == 0, result.output
        assert "TRADES: 21" in result.output

    def test_strategy_filter(self, tmp_path: Path, rendered_stdout: str) -> None:
        user_data = _seed_store(
            tmp_path,
            {"exit_code": 0, "succeeded": True, "stdout": rendered_stdout},
            strategy="TrendFollowCore",
        )
        result = CliRunner().invoke(
            cli,
            ["backtest-report", "--user-data-dir", str(user_data), "--strategy", "StatArbSwing"],
        )
        assert result.exit_code == 1
        assert "No backtest results have been recorded" in result.output

    def test_no_records_reports_clearly(self, tmp_path: Path) -> None:
        user_data = tmp_path / "user_data"
        user_data.mkdir()
        MemoryStore(user_data / "hermes_memory.sqlite3")  # creates schema, no rows
        result = CliRunner().invoke(
            cli, ["backtest-report", "--user-data-dir", str(user_data)]
        )
        assert result.exit_code == 1
        assert "No backtest results have been recorded" in result.output

    def test_record_without_stdout_reports_clearly(self, tmp_path: Path) -> None:
        user_data = _seed_store(tmp_path, {"exit_code": 0, "succeeded": True})
        result = CliRunner().invoke(
            cli, ["backtest-report", "--user-data-dir", str(user_data)]
        )
        assert result.exit_code == 1
        assert "no captured output" in result.output

    def test_missing_project_directory_reports_clearly(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(
            cli, ["backtest-report", "--user-data-dir", str(tmp_path / "nope")]
        )
        assert result.exit_code == 1
        assert "Hermes project directory not detected" in result.output
