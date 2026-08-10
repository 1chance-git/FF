"""Unit tests for hermes.cli.

Uses click's CliRunner (the standard, mature testing utility bundled
with click itself) rather than shelling out to a real `hermes` process.
Commands that would otherwise hit a real bot's REST API or spawn a real
`freqtrade` subprocess are exercised against fakes injected via
monkeypatching the small surface area (`FtRestClient`,
`BacktestLauncher.run`) those commands call.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner

from hermes.backtest import BacktestResult
from hermes.cli import cli
from hermes.health import CheckResult, HealthReport, HealthStatus
from hermes.memory import MemoryStore, TradeRecord

pytestmark = pytest.mark.unit


def test_cli_help() -> None:
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "health" in result.output
    assert "backtest" in result.output
    assert "start" in result.output


def test_status_command_reports_not_running(tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text("{}")
    pid_file = tmp_path / "bot.pid"

    result = CliRunner().invoke(
        cli,
        [
            "status",
            "-c",
            str(config_file),
            "--strategy",
            "StatArbSwing",
            "--pid-file",
            str(pid_file),
        ],
    )

    assert result.exit_code == 1
    assert "running" in result.output.lower()


def test_health_command_reports_healthy(monkeypatch) -> None:
    def fake_ft_rest_client(*args, **kwargs):
        class Client:
            def ping(self):
                return {"status": "pong"}

            def health(self):
                return {}

            def version(self):
                return {"version": "2026.7"}

            def sysinfo(self):
                return {"cpu_pct": [1.0], "ram_pct": 1.0}

        return Client()

    import freqtrade_client

    monkeypatch.setattr(freqtrade_client, "FtRestClient", fake_ft_rest_client)

    result = CliRunner().invoke(cli, ["health"])

    assert result.exit_code == 0
    assert "healthy" in result.output.lower()


def test_health_command_reads_credentials_from_environment(monkeypatch) -> None:
    """--username/--password must be settable via HERMES_API_USERNAME/PASSWORD.

    Avoids operators having to pass secrets as bare CLI arguments, which
    are visible in shell history and `ps` output on shared systems.
    """
    captured = {}

    def fake_ft_rest_client(api_url, username, password):
        captured["username"] = username
        captured["password"] = password

        class Client:
            def ping(self):
                return {"status": "pong"}

            def health(self):
                return {}

            def version(self):
                return {"version": "2026.7"}

            def sysinfo(self):
                return {"cpu_pct": [1.0], "ram_pct": 1.0}

        return Client()

    import freqtrade_client

    monkeypatch.setattr(freqtrade_client, "FtRestClient", fake_ft_rest_client)
    monkeypatch.setenv("HERMES_API_USERNAME", "env-user")
    monkeypatch.setenv("HERMES_API_PASSWORD", "env-pass")

    result = CliRunner().invoke(cli, ["health"])

    assert result.exit_code == 0
    assert captured["username"] == "env-user"
    assert captured["password"] == "env-pass"


def test_health_command_reports_unhealthy_with_nonzero_exit(monkeypatch) -> None:
    def fake_ft_rest_client(*args, **kwargs):
        class Client:
            def ping(self):
                raise ConnectionError("refused")

            def health(self):
                raise ConnectionError("refused")

            def version(self):
                raise ConnectionError("refused")

            def sysinfo(self):
                raise ConnectionError("refused")

        return Client()

    import freqtrade_client

    monkeypatch.setattr(freqtrade_client, "FtRestClient", fake_ft_rest_client)

    result = CliRunner().invoke(cli, ["health"])

    assert result.exit_code == 1
    assert "unhealthy" in result.output.lower()


def test_backtest_command_reports_success(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text("{}")

    def fake_run(self, config, timeout_seconds=None):
        return BacktestResult(
            command=("freqtrade", "backtesting"),
            exit_code=0,
            stdout="ok",
            stderr="",
            duration_seconds=1.23,
        )

    from hermes.backtest import BacktestLauncher

    monkeypatch.setattr(BacktestLauncher, "run", fake_run)

    result = CliRunner().invoke(
        cli,
        [
            "backtest",
            "-c",
            str(config_file),
            "--strategy",
            "StatArbSwing",
            "--user-data-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "succeeded" in result.output.lower()


def test_backtest_command_reports_failure(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text("{}")

    def fake_run(self, config, timeout_seconds=None):
        return BacktestResult(
            command=("freqtrade", "backtesting"),
            exit_code=2,
            stdout="",
            stderr="something went wrong",
            duration_seconds=0.5,
        )

    from hermes.backtest import BacktestLauncher

    monkeypatch.setattr(BacktestLauncher, "run", fake_run)

    result = CliRunner().invoke(
        cli,
        [
            "backtest",
            "-c",
            str(config_file),
            "--strategy",
            "StatArbSwing",
            "--user-data-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 2
    assert "failed" in result.output.lower()


def test_backtest_command_persists_result_via_memory_store(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text("{}")

    def fake_run(self, config, timeout_seconds=None):
        assert self.memory_store is not None
        return BacktestResult(
            command=("freqtrade", "backtesting"),
            exit_code=0,
            stdout="ok",
            stderr="",
            duration_seconds=1.23,
        )

    from hermes.backtest import BacktestLauncher

    monkeypatch.setattr(BacktestLauncher, "run", fake_run)

    result = CliRunner().invoke(
        cli,
        [
            "backtest",
            "-c",
            str(config_file),
            "--strategy",
            "StatArbSwing",
            "--user-data-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "hermes_memory.sqlite3").exists()


def test_verbose_and_json_log_file_options_accepted(tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text("{}")
    log_file = tmp_path / "hermes.log"

    result = CliRunner().invoke(
        cli,
        [
            "--json-log-file",
            str(log_file),
            "-v",
            "status",
            "-c",
            str(config_file),
            "--strategy",
            "StatArbSwing",
            "--pid-file",
            str(tmp_path / "bot.pid"),
        ],
    )

    assert result.exit_code == 1  # not running, but the command itself ran fine
    assert log_file.exists()


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def _seed_trades(db_path: Path, n: int = 20) -> None:
    """Write `n` completed trades directly to a fresh MemoryStore at `db_path`."""
    store = MemoryStore(db_path)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    pnls = [10.0, -5.0, 15.0, -3.0, 8.0]
    for i in range(n):
        pnl = pnls[i % len(pnls)]
        store.record_trade(
            TradeRecord(
                pair="BTC/USDC:USDC",
                side="long",
                entry_time=base + timedelta(hours=i),
                exit_time=base + timedelta(hours=i + 1),
                entry_price=100.0,
                exit_price=100.0 + pnl,
                pnl=pnl,
                pnl_pct=pnl / 100,
                fees=0.5,
                entry_zscore=2.0 + i * 0.05,
                exit_zscore=0.1,
                hedge_ratio=1.3,
                holding_time_seconds=3600.0,
                exit_reason="exit_signal",
                regime="mean_reverting",
            )
        )


def test_analyze_command_help() -> None:
    """`python -m hermes --help` and `... analyze --help` both explain the command."""
    top_level = CliRunner().invoke(cli, ["--help"])
    assert top_level.exit_code == 0
    assert "analyze" in top_level.output

    command_help = CliRunner().invoke(cli, ["analyze", "--help"])
    assert command_help.exit_code == 0
    assert "Analyze recorded trade history" in command_help.output


def test_analyze_command_with_valid_database(tmp_path: Path) -> None:
    _seed_trades(tmp_path / "hermes_memory.sqlite3", n=20)

    result = CliRunner().invoke(cli, ["analyze", "--user-data-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "[HERMES][ANALYSIS]" in result.output
    assert "TRADES: 20" in result.output
    assert "WIN RATE:" in result.output
    assert "EXPECTANCY:" in result.output
    assert "MAX DRAWDOWN:" in result.output
    assert "STATUS: OBSERVATION ONLY" in result.output


def test_analyze_command_with_empty_database(tmp_path: Path) -> None:
    """A database that exists but has zero recorded trades (schema only)."""
    MemoryStore(tmp_path / "hermes_memory.sqlite3")  # create schema, no trades

    result = CliRunner().invoke(cli, ["analyze", "--user-data-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "[HERMES][ANALYSIS]" in result.output
    assert "Insufficient historical data for meaningful analysis." in result.output


def test_analyze_command_with_insufficient_data_reports_no_error(tmp_path: Path) -> None:
    """A database that doesn't exist yet at all: not an error, just no history yet."""
    result = CliRunner().invoke(cli, ["analyze", "--user-data-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "Insufficient historical data for meaningful analysis." in result.output
    assert (tmp_path / "hermes_memory.sqlite3").exists()  # a fresh, empty store was created


def test_analyze_command_with_missing_project_directory(tmp_path: Path) -> None:
    """The --user-data-dir itself doesn't exist -- the "wrong directory" case."""
    missing = tmp_path / "does_not_exist"

    result = CliRunner().invoke(cli, ["analyze", "--user-data-dir", str(missing)])

    assert result.exit_code == 1
    assert "[HERMES][ERROR]" in result.output
    assert "Hermes project directory not detected" in result.output
    assert "Run this command from the project's root folder" in result.output
    assert not missing.exists()  # nothing was created for a directory that isn't real


def test_analyze_command_with_malformed_database(tmp_path: Path) -> None:
    """A file exists at the expected path but isn't a valid SQLite database."""
    db_path = tmp_path / "hermes_memory.sqlite3"
    db_path.write_text("this is not a sqlite database")

    result = CliRunner().invoke(cli, ["analyze", "--user-data-dir", str(tmp_path)])

    assert result.exit_code == 1
    assert "[HERMES][ERROR]" in result.output
    assert "Could not read the Hermes trading history database" in result.output


def test_analyze_command_never_writes_trade_history(tmp_path: Path) -> None:
    """Read-only guarantee: running analyze must not add, remove, or change any rows."""
    db_path = tmp_path / "hermes_memory.sqlite3"
    _seed_trades(db_path, n=5)
    before = MemoryStore(db_path).get_trades()

    CliRunner().invoke(cli, ["analyze", "--user-data-dir", str(tmp_path)])

    after = MemoryStore(db_path).get_trades()
    assert before == after


def test_render_analysis_report_matches_expected_shape() -> None:
    from hermes.analyzer import AnalysisReport, Finding
    from hermes.cli import render_analysis_report

    report = AnalysisReport(
        trade_count=147,
        win_rate=0.612,
        average_pnl=1.0,
        expectancy=0.34,
        profit_factor=1.5,
        max_drawdown=8.7,
        average_holding_time_seconds=3600.0,
        total_fees=10.0,
        total_funding=0.0,
        fees_and_funding_drag_pct=1.0,
        largest_losses=[],
        max_consecutive_losses=2,
        findings=[
            Finding(
                observation="Larger Z-score entries have historically produced better expectancy.",
                hypothesis="A stricter entry threshold may be worth testing.",
            )
        ],
    )

    output = render_analysis_report(report)

    assert "TRADES: 147" in output
    assert "WIN RATE: 61.2%" in output
    assert "EXPECTANCY: +0.34" in output
    assert "MAX DRAWDOWN: 8.70" in output
    assert "[OBSERVATION]" in output
    assert "Larger Z-score entries have historically produced better expectancy." in output
    assert "[HYPOTHESIS]" in output
    assert "A stricter entry threshold may be worth testing." in output
    assert output.strip().endswith("STATUS: OBSERVATION ONLY")
