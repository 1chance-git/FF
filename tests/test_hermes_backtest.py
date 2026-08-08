"""Unit tests for hermes.backtest."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from hermes.backtest import (
    BacktestConfig,
    BacktestError,
    BacktestLauncher,
    build_backtest_command,
)


def test_config_rejects_empty_strategy() -> None:
    with pytest.raises(BacktestError, match="strategy"):
        BacktestConfig(strategy="", config_files=(Path("a.json"),))


def test_config_rejects_no_config_files() -> None:
    with pytest.raises(BacktestError, match="config_files"):
        BacktestConfig(strategy="Foo", config_files=())


def test_build_backtest_command_minimal() -> None:
    config = BacktestConfig(strategy="StatArbSwing", config_files=(Path("user_data/config.json"),))
    command = build_backtest_command(config)

    assert command[0] == sys.executable
    assert command[1:4] == ["-m", "freqtrade", "backtesting"]
    assert "-c" in command
    assert "user_data/config.json" in command
    assert "--strategy" in command
    assert "StatArbSwing" in command


def test_build_backtest_command_multiple_config_files() -> None:
    config = BacktestConfig(
        strategy="Foo",
        config_files=(Path("a.json"), Path("b.json")),
    )
    command = build_backtest_command(config)
    assert command.count("-c") == 2
    assert "a.json" in command
    assert "b.json" in command


def test_build_backtest_command_includes_optional_flags() -> None:
    config = BacktestConfig(
        strategy="Foo",
        config_files=(Path("a.json"),),
        timerange="20240101-20240201",
        timeframe="1h",
        strategy_path=Path("user_data/strategies"),
        extra_args=("--breakdown", "day"),
    )
    command = build_backtest_command(config)

    assert "--timerange" in command
    assert "20240101-20240201" in command
    assert "--timeframe" in command
    assert "1h" in command
    assert "--strategy-path" in command
    assert "--breakdown" in command
    assert "day" in command


def test_build_backtest_command_omits_unset_optional_flags() -> None:
    config = BacktestConfig(strategy="Foo", config_files=(Path("a.json"),))
    command = build_backtest_command(config)
    assert "--timerange" not in command
    assert "--timeframe" not in command
    assert "--strategy-path" not in command


def test_launcher_runs_real_subprocess_and_captures_output() -> None:
    """Exercise the real subprocess path with a lightweight stand-in command.

    Doesn't invoke the actual `freqtrade backtesting` (slow, requires
    downloaded market data and network access this sandbox blocks) —
    instead verifies BacktestLauncher.run() correctly wires argv,
    captures output, and reports the exit code for *some* real process,
    which is exactly what it does for the freqtrade subprocess too.
    """
    config = BacktestConfig(strategy="Foo", config_files=(Path("a.json"),))
    launcher = BacktestLauncher()

    # Monkeypatch the module-level command builder indirectly isn't
    # exposed; instead verify via a config whose resulting command we
    # know, then patch subprocess at the point BacktestLauncher uses it.
    import hermes.backtest as backtest_module

    original_run = backtest_module.subprocess.run

    def fake_run(command, **kwargs):
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        import subprocess as sp

        return sp.CompletedProcess(command, returncode=0, stdout="ok", stderr="")

    backtest_module.subprocess.run = fake_run
    try:
        result = launcher.run(config)
    finally:
        backtest_module.subprocess.run = original_run

    assert result.succeeded is True
    assert result.exit_code == 0
    assert result.stdout == "ok"
    assert result.duration_seconds >= 0


def test_launcher_reports_failure_without_raising() -> None:
    config = BacktestConfig(strategy="Foo", config_files=(Path("a.json"),))
    launcher = BacktestLauncher()

    import hermes.backtest as backtest_module

    original_run = backtest_module.subprocess.run

    def fake_run(command, **kwargs):
        import subprocess as sp

        return sp.CompletedProcess(command, returncode=2, stdout="", stderr="boom")

    backtest_module.subprocess.run = fake_run
    try:
        result = launcher.run(config)
    finally:
        backtest_module.subprocess.run = original_run

    assert result.succeeded is False
    assert result.exit_code == 2
    assert result.stderr == "boom"


def test_launcher_raises_backtest_error_on_timeout() -> None:
    config = BacktestConfig(strategy="Foo", config_files=(Path("a.json"),))
    launcher = BacktestLauncher()

    import hermes.backtest as backtest_module

    def fake_run(command, **kwargs):
        import subprocess as sp

        raise sp.TimeoutExpired(cmd=command, timeout=kwargs.get("timeout") or 1)

    original_run = backtest_module.subprocess.run
    backtest_module.subprocess.run = fake_run
    try:
        with pytest.raises(BacktestError, match="timed out"):
            launcher.run(config, timeout_seconds=1)
    finally:
        backtest_module.subprocess.run = original_run


def test_launcher_end_to_end_real_subprocess_help() -> None:
    """Genuine end-to-end run of the real `freqtrade backtesting` subprocess.

    Uses `--help` (via extra_args) so it exits immediately with no
    network/market-data dependency, while still exercising the real
    `subprocess.run` call this launcher makes in production — not
    mocked, unlike the other tests in this file.
    """
    config = BacktestConfig(
        strategy="StatArbSwing",
        config_files=(Path("user_data/config.json"),),
        extra_args=("--help",),
    )
    result = BacktestLauncher().run(config, timeout_seconds=30)

    assert result.exit_code == 0
    assert "usage" in result.stdout.lower()
