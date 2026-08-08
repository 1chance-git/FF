"""Unit tests for optimize.hyperopt_launcher.

Mirrors tests/test_hermes_backtest.py's approach: mocks subprocess.run
for most cases (fast, deterministic), plus one genuine end-to-end real
subprocess run using --help to avoid network/market-data dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from optimize.hyperopt_launcher import (
    HyperoptConfig,
    HyperoptError,
    HyperoptLauncher,
    build_hyperopt_command,
)


def test_config_rejects_empty_strategy() -> None:
    with pytest.raises(HyperoptError, match="strategy"):
        HyperoptConfig(strategy="", config_files=(Path("a.json"),))


def test_config_rejects_no_config_files() -> None:
    with pytest.raises(HyperoptError, match="config_files"):
        HyperoptConfig(strategy="Foo", config_files=())


def test_config_rejects_non_positive_epochs() -> None:
    with pytest.raises(HyperoptError, match="epochs"):
        HyperoptConfig(strategy="Foo", config_files=(Path("a.json"),), epochs=0)


def test_config_rejects_empty_spaces() -> None:
    with pytest.raises(HyperoptError, match="spaces"):
        HyperoptConfig(strategy="Foo", config_files=(Path("a.json"),), spaces=())


def test_build_hyperopt_command_minimal() -> None:
    config = HyperoptConfig(strategy="StatArbSwing", config_files=(Path("user_data/config.json"),))
    command = build_hyperopt_command(config)

    assert command[0] == sys.executable
    assert command[1:4] == ["-m", "freqtrade", "hyperopt"]
    assert "--strategy" in command
    assert "StatArbSwing" in command
    assert "--epochs" in command
    assert "100" in command
    assert "--spaces" in command
    assert "buy" in command and "sell" in command
    assert "--hyperopt-loss" in command
    assert "StatArbHyperOptLoss" in command
    assert "--hyperopt-path" in command


def test_build_hyperopt_command_default_hyperopt_path_points_at_this_package() -> None:
    config = HyperoptConfig(strategy="Foo", config_files=(Path("a.json"),))
    command = build_hyperopt_command(config)
    path_index = command.index("--hyperopt-path") + 1
    assert command[path_index].endswith("optimize")


def test_build_hyperopt_command_includes_optional_flags() -> None:
    config = HyperoptConfig(
        strategy="Foo",
        config_files=(Path("a.json"),),
        epochs=50,
        spaces=("buy",),
        timerange="20240101-20240201",
        timeframe="1h",
        strategy_path=Path("user_data/strategies"),
        jobs=4,
        random_state=42,
        min_trades=10,
        extra_args=("--print-json",),
    )
    command = build_hyperopt_command(config)

    assert "--timerange" in command and "20240101-20240201" in command
    assert "--timeframe" in command and "1h" in command
    assert "--strategy-path" in command
    assert "-j" in command and "4" in command
    assert "--random-state" in command and "42" in command
    assert "--min-trades" in command and "10" in command
    assert "--print-json" in command


def test_launcher_reports_success_via_mocked_subprocess() -> None:
    config = HyperoptConfig(strategy="Foo", config_files=(Path("a.json"),))
    launcher = HyperoptLauncher()

    import optimize.hyperopt_launcher as launcher_module

    original_run = launcher_module.subprocess.run

    def fake_run(command, **kwargs):
        import subprocess as sp

        return sp.CompletedProcess(command, returncode=0, stdout="Best result", stderr="")

    launcher_module.subprocess.run = fake_run
    try:
        result = launcher.run(config)
    finally:
        launcher_module.subprocess.run = original_run

    assert result.succeeded is True
    assert result.stdout == "Best result"


def test_launcher_reports_failure_without_raising() -> None:
    config = HyperoptConfig(strategy="Foo", config_files=(Path("a.json"),))
    launcher = HyperoptLauncher()

    import optimize.hyperopt_launcher as launcher_module

    original_run = launcher_module.subprocess.run

    def fake_run(command, **kwargs):
        import subprocess as sp

        return sp.CompletedProcess(command, returncode=1, stdout="", stderr="epoch failed")

    launcher_module.subprocess.run = fake_run
    try:
        result = launcher.run(config)
    finally:
        launcher_module.subprocess.run = original_run

    assert result.succeeded is False
    assert result.exit_code == 1


def test_launcher_raises_hyperopt_error_on_timeout() -> None:
    config = HyperoptConfig(strategy="Foo", config_files=(Path("a.json"),))
    launcher = HyperoptLauncher()

    import optimize.hyperopt_launcher as launcher_module

    def fake_run(command, **kwargs):
        import subprocess as sp

        raise sp.TimeoutExpired(cmd=command, timeout=kwargs.get("timeout") or 1)

    original_run = launcher_module.subprocess.run
    launcher_module.subprocess.run = fake_run
    try:
        with pytest.raises(HyperoptError, match="timed out"):
            launcher.run(config, timeout_seconds=1)
    finally:
        launcher_module.subprocess.run = original_run


def test_launcher_end_to_end_real_subprocess_help() -> None:
    """Genuine end-to-end run of the real `freqtrade hyperopt` subprocess, not mocked."""
    config = HyperoptConfig(
        strategy="StatArbSwing",
        config_files=(Path("user_data/config.json"),),
        extra_args=("--help",),
    )
    result = HyperoptLauncher().run(config, timeout_seconds=30)

    assert result.exit_code == 0
    assert "usage" in result.stdout.lower()
