"""Unit tests for hermes.cli.

Uses click's CliRunner (the standard, mature testing utility bundled
with click itself) rather than shelling out to a real `hermes` process.
Commands that would otherwise hit a real bot's REST API or spawn a real
`freqtrade` subprocess are exercised against fakes injected via
monkeypatching the small surface area (`FtRestClient`,
`BacktestLauncher.run`) those commands call.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from hermes.backtest import BacktestResult
from hermes.cli import cli
from hermes.health import CheckResult, HealthReport, HealthStatus

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
        cli, ["backtest", "-c", str(config_file), "--strategy", "StatArbSwing"]
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
        cli, ["backtest", "-c", str(config_file), "--strategy", "StatArbSwing"]
    )

    assert result.exit_code == 2
    assert "failed" in result.output.lower()


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
