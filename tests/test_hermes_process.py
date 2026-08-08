"""Unit tests for hermes.process.

Uses a lightweight `python -c "time.sleep(...)"` stand-in process
(injected via BotProcessManager's command_builder/process_matcher hooks)
instead of the real `freqtrade trade` subprocess, so these tests are
fast and have no network/exchange dependency, while still exercising
real subprocess start/stop/signal-handling end to end — not mocked.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from hermes.process import (
    BotProcessManager,
    ProcessConfig,
    ProcessError,
    RestartSupervisor,
    RestartSupervisorConfig,
    build_trade_command,
)

# A process that ignores nothing special, sleeps, and honors SIGTERM
# (the default action for SIGTERM is termination, so plain sleep works).
_SLEEPER_SCRIPT = "import time; time.sleep(30)"


def _fake_command_builder(script: str = _SLEEPER_SCRIPT):
    return lambda config: [sys.executable, "-c", script]


def _accept_all(cmdline: str) -> bool:
    return True


def make_manager(tmp_path: Path, script: str = _SLEEPER_SCRIPT, **config_kwargs) -> BotProcessManager:
    config = ProcessConfig(
        config_files=(tmp_path / "config.json",),
        strategy="StatArbSwing",
        pid_file=tmp_path / "bot.pid",
        stop_grace_period_seconds=config_kwargs.pop("stop_grace_period_seconds", 5.0),
        **config_kwargs,
    )
    return BotProcessManager(
        config, command_builder=_fake_command_builder(script), process_matcher=_accept_all
    )


# ---------------------------------------------------------------------------
# ProcessConfig validation
# ---------------------------------------------------------------------------


def test_process_config_rejects_no_config_files(tmp_path: Path) -> None:
    with pytest.raises(ProcessError, match="config_files"):
        ProcessConfig(config_files=(), strategy="Foo", pid_file=tmp_path / "bot.pid")


def test_process_config_rejects_empty_strategy(tmp_path: Path) -> None:
    with pytest.raises(ProcessError, match="strategy"):
        ProcessConfig(config_files=(tmp_path / "a.json",), strategy="", pid_file=tmp_path / "bot.pid")


def test_process_config_rejects_non_positive_grace_period(tmp_path: Path) -> None:
    with pytest.raises(ProcessError, match="stop_grace_period_seconds"):
        ProcessConfig(
            config_files=(tmp_path / "a.json",),
            strategy="Foo",
            pid_file=tmp_path / "bot.pid",
            stop_grace_period_seconds=0,
        )


# ---------------------------------------------------------------------------
# build_trade_command
# ---------------------------------------------------------------------------


def test_build_trade_command(tmp_path: Path) -> None:
    config = ProcessConfig(
        config_files=(tmp_path / "config.json",), strategy="StatArbSwing", pid_file=tmp_path / "bot.pid"
    )
    command = build_trade_command(config)
    assert command[0] == sys.executable
    assert command[1:3] == ["-m", "freqtrade"]
    assert "trade" in command
    assert "--strategy" in command
    assert "StatArbSwing" in command


# ---------------------------------------------------------------------------
# BotProcessManager: real subprocess lifecycle
# ---------------------------------------------------------------------------


def test_start_writes_pid_file_and_reports_running(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    try:
        pid = manager.start()
        assert manager.config.pid_file.exists()
        assert int(manager.config.pid_file.read_text()) == pid
        assert manager.is_running() is True
    finally:
        manager.stop()


def test_start_is_noop_if_already_running(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    try:
        pid1 = manager.start()
        pid2 = manager.start()
        assert pid1 == pid2
    finally:
        manager.stop()


def test_stop_terminates_process_and_removes_pid_file(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    manager.start()

    stopped = manager.stop()

    assert stopped is True
    assert manager.is_running() is False
    assert not manager.config.pid_file.exists()


def test_stop_is_noop_when_nothing_running(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    assert manager.stop() is False


def test_stop_escalates_to_sigkill_when_process_ignores_sigterm(tmp_path: Path) -> None:
    ignore_sigterm_script = (
        "import signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(30)"
    )
    manager = make_manager(tmp_path, script=ignore_sigterm_script, stop_grace_period_seconds=1.0)
    manager.start()

    start_time = time.monotonic()
    stopped = manager.stop()
    elapsed = time.monotonic() - start_time

    assert stopped is True
    assert manager.is_running() is False
    # Should have waited roughly the grace period before escalating, not
    # hung indefinitely.
    assert elapsed < 10.0


def test_restart_stops_old_and_starts_new_process(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    try:
        pid1 = manager.start()
        pid2 = manager.restart()
        assert pid1 != pid2
        assert manager.is_running() is True
    finally:
        manager.stop()


def test_status_reports_not_running_when_no_pid_file(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    status = manager.status()
    assert status == {"running": False, "pid": None}


def test_status_reports_running_with_metrics(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    try:
        pid = manager.start()
        status = manager.status()
        assert status["running"] is True
        assert status["pid"] == pid
        assert "cpu_percent" in status
        assert "memory_mb" in status
        assert status["memory_mb"] > 0
        assert status["uptime_seconds"] >= 0
    finally:
        manager.stop()


def test_stale_pid_file_pointing_at_dead_process_is_ignored(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    manager.config.pid_file.parent.mkdir(parents=True, exist_ok=True)
    # A PID essentially guaranteed not to exist.
    manager.config.pid_file.write_text("999999", encoding="utf-8")

    assert manager.is_running() is False
    assert not manager.config.pid_file.exists()


def test_pid_file_for_unrelated_process_is_not_matched(tmp_path: Path) -> None:
    config = ProcessConfig(
        config_files=(tmp_path / "config.json",),
        strategy="StatArbSwing",
        pid_file=tmp_path / "bot.pid",
    )
    # process_matcher that never matches -> even the current (real) test
    # process's own PID should not be considered "our bot".
    manager = BotProcessManager(
        config, command_builder=_fake_command_builder(), process_matcher=lambda cmdline: False
    )
    import os

    manager.config.pid_file.parent.mkdir(parents=True, exist_ok=True)
    manager.config.pid_file.write_text(str(os.getpid()), encoding="utf-8")

    assert manager.is_running() is False


# ---------------------------------------------------------------------------
# RestartSupervisor
# ---------------------------------------------------------------------------


def test_restart_supervisor_config_rejects_invalid_values() -> None:
    with pytest.raises(ProcessError, match="max_restarts"):
        RestartSupervisorConfig(max_restarts=0)
    with pytest.raises(ProcessError, match="initial_backoff_seconds"):
        RestartSupervisorConfig(initial_backoff_seconds=0)
    with pytest.raises(ProcessError, match="max_backoff_seconds"):
        RestartSupervisorConfig(initial_backoff_seconds=10, max_backoff_seconds=5)


def test_restart_supervisor_backoff_schedule_doubles_and_caps() -> None:
    config = RestartSupervisorConfig(
        max_restarts=5, initial_backoff_seconds=1.0, max_backoff_seconds=6.0
    )
    assert config.backoff_schedule() == [1.0, 2.0, 4.0, 6.0, 6.0]


def test_restart_supervisor_succeeds_on_first_attempt(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    try:
        manager.start()
        sleeps: list[float] = []
        supervisor = RestartSupervisor(
            manager,
            RestartSupervisorConfig(max_restarts=3, initial_backoff_seconds=0.01, max_backoff_seconds=0.02),
            sleep=sleeps.append,
        )

        pid = supervisor.handle_crash()

        assert manager.is_running() is True
        assert sleeps == [0.01]  # only one attempt needed
        assert pid > 0
    finally:
        manager.stop()


def test_restart_supervisor_exhausts_retries_and_raises(tmp_path: Path) -> None:
    config = ProcessConfig(
        config_files=(tmp_path / "config.json",), strategy="Foo", pid_file=tmp_path / "bot.pid"
    )

    def always_failing_builder(_config):
        raise OSError("no such executable")

    class AlwaysFailingManager(BotProcessManager):
        def restart(self) -> int:
            raise ProcessError("boom")

    manager = AlwaysFailingManager(config, command_builder=_fake_command_builder())
    sleeps: list[float] = []
    supervisor = RestartSupervisor(
        manager,
        RestartSupervisorConfig(max_restarts=3, initial_backoff_seconds=0.01, max_backoff_seconds=0.02),
        sleep=sleeps.append,
    )

    with pytest.raises(ProcessError, match="failed to restart after 3 attempts"):
        supervisor.handle_crash()

    assert len(sleeps) == 3
