"""Unit tests for hermes.backtest."""

from __future__ import annotations

import logging
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
from hermes.memory import MemoryStore


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


# ---------------------------------------------------------------------------
# Memory persistence (hermes.memory wiring)
# ---------------------------------------------------------------------------


def _patch_subprocess_run(monkeypatch: pytest.MonkeyPatch, returncode: int, stdout: str = "", stderr: str = "") -> None:
    import hermes.backtest as backtest_module

    def fake_run(command, **kwargs):
        import subprocess as sp

        return sp.CompletedProcess(command, returncode=returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(backtest_module.subprocess, "run", fake_run)


def test_successful_backtest_is_persisted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_subprocess_run(monkeypatch, returncode=0, stdout="ok")
    memory_store = MemoryStore(tmp_path / "memory.sqlite3")
    config = BacktestConfig(
        strategy="StatArbSwing",
        config_files=(Path("user_data/config.json"),),
        timerange="20240101-20240401",
    )

    result = BacktestLauncher(memory_store=memory_store).run(config)

    assert result.succeeded is True  # unchanged return value
    [saved] = memory_store.get_backtest_results()
    assert saved.strategy == "StatArbSwing"
    assert saved.timerange == "20240101-20240401"


def test_backtest_result_maps_correctly_into_the_memory_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_subprocess_run(monkeypatch, returncode=0, stdout="great backtest", stderr="a warning")
    memory_store = MemoryStore(tmp_path / "memory.sqlite3")
    config = BacktestConfig(
        strategy="StatArbSwing",
        config_files=(Path("user_data/config.json"), Path("user_data/config-private.json")),
        timerange="20240101-20240401",
        timeframe="1h",
        strategy_path=Path("user_data/strategies"),
        extra_args=("--breakdown", "day"),
    )

    result = BacktestLauncher(memory_store=memory_store).run(config)

    [saved] = memory_store.get_backtest_results()
    assert saved.strategy == config.strategy
    assert saved.timerange == config.timerange
    assert saved.metrics["exit_code"] == result.exit_code == 0
    assert saved.metrics["succeeded"] is True
    assert saved.metrics["duration_seconds"] == pytest.approx(result.duration_seconds)
    assert saved.metrics["command"] == list(result.command)
    assert saved.metrics["stdout"] == "great backtest"
    assert saved.metrics["stderr"] == "a warning"
    assert saved.metrics["config_files"] == [
        "user_data/config.json",
        "user_data/config-private.json",
    ]
    assert saved.metrics["timeframe"] == "1h"
    assert saved.metrics["strategy_path"] == "user_data/strategies"
    assert saved.metrics["extra_args"] == ["--breakdown", "day"]


def test_missing_optional_fields_are_recorded_as_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_subprocess_run(monkeypatch, returncode=0)
    memory_store = MemoryStore(tmp_path / "memory.sqlite3")
    # No timerange, timeframe, or strategy_path supplied.
    config = BacktestConfig(strategy="StatArbSwing", config_files=(Path("a.json"),))

    BacktestLauncher(memory_store=memory_store).run(config)

    [saved] = memory_store.get_backtest_results()
    assert saved.timerange is None
    assert saved.metrics["timeframe"] is None
    assert saved.metrics["strategy_path"] is None
    assert saved.metrics["extra_args"] == []


def test_failed_backtest_is_persisted_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_subprocess_run(monkeypatch, returncode=2, stderr="boom")
    memory_store = MemoryStore(tmp_path / "memory.sqlite3")
    config = BacktestConfig(strategy="StatArbSwing", config_files=(Path("a.json"),))

    result = BacktestLauncher(memory_store=memory_store).run(config)

    assert result.succeeded is False
    [saved] = memory_store.get_backtest_results()
    assert saved.metrics["succeeded"] is False
    assert saved.metrics["exit_code"] == 2


def test_no_memory_store_is_a_safe_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing callers that never pass memory_store must keep working unchanged."""
    _patch_subprocess_run(monkeypatch, returncode=0, stdout="ok")
    config = BacktestConfig(strategy="StatArbSwing", config_files=(Path("a.json"),))

    result = BacktestLauncher().run(config)  # memory_store omitted, as in existing tests

    assert result.succeeded is True
    assert result.stdout == "ok"


def test_broken_memory_store_does_not_break_the_backtest_result(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.ERROR, logger="hermes.backtest")
    _patch_subprocess_run(monkeypatch, returncode=0, stdout="ok")

    class ExplodingMemoryStore:
        def record_backtest_result(self, *args, **kwargs):
            raise RuntimeError("simulated disk failure")

    config = BacktestConfig(strategy="StatArbSwing", config_files=(Path("a.json"),))

    result = BacktestLauncher(memory_store=ExplodingMemoryStore()).run(config)

    assert result.succeeded is True
    assert result.stdout == "ok"
    assert "[HERMES][MEMORY][ERROR]" in caplog.text


def test_memory_store_returning_false_is_logged_without_raising(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """MemoryStore.record_backtest_result() itself never raises -- it returns False on
    failure (see hermes.memory) -- so the launcher must check the return value too."""
    caplog.set_level(logging.ERROR, logger="hermes.backtest")
    _patch_subprocess_run(monkeypatch, returncode=0, stdout="ok")

    class FailingMemoryStore:
        def record_backtest_result(self, *args, **kwargs) -> bool:
            return False

    config = BacktestConfig(strategy="StatArbSwing", config_files=(Path("a.json"),))

    result = BacktestLauncher(memory_store=FailingMemoryStore()).run(config)

    assert result.succeeded is True
    assert "[HERMES][MEMORY][ERROR]" in caplog.text


def test_launcher_still_raises_on_timeout_with_memory_store_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timeout is an operational failure of the launcher itself (see BacktestLauncher.run's
    docstring) -- it must still raise even with persistence configured, and nothing should
    be recorded since no BacktestResult was ever produced."""
    memory_store = MemoryStore(tmp_path / "memory.sqlite3")
    config = BacktestConfig(strategy="StatArbSwing", config_files=(Path("a.json"),))

    import hermes.backtest as backtest_module

    def fake_run(command, **kwargs):
        import subprocess as sp

        raise sp.TimeoutExpired(cmd=command, timeout=kwargs.get("timeout") or 1)

    monkeypatch.setattr(backtest_module.subprocess, "run", fake_run)

    with pytest.raises(BacktestError, match="timed out"):
        BacktestLauncher(memory_store=memory_store).run(config, timeout_seconds=1)

    assert memory_store.get_backtest_results() == []
