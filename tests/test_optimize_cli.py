"""Unit tests for optimize.cli."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from click.testing import CliRunner

from optimize.cli import cli
from optimize.hyperopt_launcher import HyperoptResult


def test_cli_help() -> None:
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "hyperopt" in result.output
    assert "report" in result.output


def test_hyperopt_command_reports_success(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text("{}")

    def fake_run(self, config, timeout_seconds=None):
        return HyperoptResult(
            command=("freqtrade", "hyperopt"),
            exit_code=0,
            stdout="Best epoch found",
            stderr="",
            duration_seconds=2.5,
        )

    from optimize.hyperopt_launcher import HyperoptLauncher

    monkeypatch.setattr(HyperoptLauncher, "run", fake_run)

    result = CliRunner().invoke(
        cli, ["hyperopt", "-c", str(config_file), "--strategy", "StatArbSwing", "--epochs", "5"]
    )

    assert result.exit_code == 0
    assert "succeeded" in result.output.lower()


def test_hyperopt_command_reports_failure(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text("{}")

    def fake_run(self, config, timeout_seconds=None):
        return HyperoptResult(
            command=("freqtrade", "hyperopt"),
            exit_code=1,
            stdout="",
            stderr="something went wrong",
            duration_seconds=1.0,
        )

    from optimize.hyperopt_launcher import HyperoptLauncher

    monkeypatch.setattr(HyperoptLauncher, "run", fake_run)

    result = CliRunner().invoke(
        cli, ["hyperopt", "-c", str(config_file), "--strategy", "StatArbSwing"]
    )

    assert result.exit_code == 1
    assert "failed" in result.output.lower()


def test_report_command_renders_from_trades_file(tmp_path: Path) -> None:
    trades = [
        {"pair": "ETH/USDC:USDC", "close_date": "2024-01-01", "profit_abs": 10.0},
        {"pair": "ETH/USDC:USDC", "close_date": "2024-01-02", "profit_abs": -5.0},
        {"pair": "BTC/USDC:USDC", "close_date": "2024-01-03", "profit_abs": 8.0},
        {"pair": "BTC/USDC:USDC", "close_date": "2024-01-04", "profit_abs": 12.0},
        {"pair": "ETH/USDC:USDC", "close_date": "2024-01-05", "profit_abs": -3.0},
        {"pair": "ETH/USDC:USDC", "close_date": "2024-01-06", "profit_abs": 6.0},
        {"pair": "BTC/USDC:USDC", "close_date": "2024-01-07", "profit_abs": 9.0},
        {"pair": "BTC/USDC:USDC", "close_date": "2024-01-08", "profit_abs": -8.0},
        {"pair": "ETH/USDC:USDC", "close_date": "2024-01-09", "profit_abs": 4.0},
        {"pair": "BTC/USDC:USDC", "close_date": "2024-01-10", "profit_abs": 11.0},
    ]
    trades_file = tmp_path / "trades.json"
    trades_file.write_text(json.dumps(trades))

    result = CliRunner().invoke(
        cli,
        [
            "report",
            "--trades-file",
            str(trades_file),
            "--starting-balance",
            "1000",
            "--title",
            "CLI Test Report",
        ],
    )

    assert result.exit_code == 0
    assert "Sharpe" in result.output
    assert "Total trades" in result.output
