"""Unit tests for hermes.supervisor.

Uses lightweight `python -c "..."` stand-in processes (injected via
`Supervisor`'s `command_builder` hook) instead of the real `freqtrade
trade` subprocess, so these tests are fast and have no exchange
dependency, while still exercising real subprocess start/monitor/crash/
restart/signal handling end to end — not mocked.
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from hermes.memory import MemoryStore
from hermes.process import ProcessConfig, ProcessError
from hermes.supervisor import (
    Supervisor,
    SupervisorConfig,
    install_signal_handlers,
)

SLEEP_LONG = [sys.executable, "-c", "import time; time.sleep(60)"]
EXIT_IMMEDIATELY = [sys.executable, "-c", "pass"]
PRINT_THEN_EXIT = [
    sys.executable,
    "-c",
    "import sys; print('hello-stdout'); print('hello-warn', file=sys.stderr)",
]


def _process_config(tmp_path: Path) -> ProcessConfig:
    return ProcessConfig(
        config_files=(tmp_path / "config.json",),
        strategy="StatArbSwing",
        pid_file=tmp_path / "hermes_supervisor.pid",
    )


def _fast_config(tmp_path: Path, **overrides: object) -> SupervisorConfig:
    defaults = dict(
        process_config=_process_config(tmp_path),
        health_interval_seconds=0.2,
        max_restarts=2,
        initial_backoff_seconds=0.01,
        max_backoff_seconds=0.01,
    )
    defaults.update(overrides)
    return SupervisorConfig(**defaults)


def _run_in_thread(supervisor: Supervisor) -> threading.Thread:
    thread = threading.Thread(target=supervisor.run, daemon=True)
    thread.start()
    return thread


# -- SupervisorConfig ------------------------------------------------


def test_config_rejects_non_positive_health_interval(tmp_path: Path) -> None:
    with pytest.raises(ProcessError):
        _fast_config(tmp_path, health_interval_seconds=0)


def test_config_rejects_zero_max_restarts(tmp_path: Path) -> None:
    with pytest.raises(ProcessError):
        _fast_config(tmp_path, max_restarts=0)


def test_config_rejects_max_backoff_below_initial(tmp_path: Path) -> None:
    with pytest.raises(ProcessError):
        _fast_config(tmp_path, initial_backoff_seconds=5.0, max_backoff_seconds=1.0)


def test_backoff_schedule_doubles_and_caps() -> None:
    config = SupervisorConfig(
        process_config=ProcessConfig(
            config_files=(Path("x"),), strategy="S", pid_file=Path("p")
        ),
        max_restarts=4,
        initial_backoff_seconds=2.0,
        max_backoff_seconds=5.0,
    )
    assert config.backoff_schedule() == [2.0, 4.0, 5.0, 5.0]


# -- start / health / stop -------------------------------------------


def test_start_logs_start_event(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="hermes.supervisor")
    config = _fast_config(tmp_path)
    supervisor = Supervisor(config, command_builder=lambda _: SLEEP_LONG)

    thread = _run_in_thread(supervisor)
    time.sleep(0.1)
    supervisor.request_stop()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert "[HERMES][START]" in caplog.text


def test_health_heartbeat_logged_while_alive(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="hermes.supervisor")
    config = _fast_config(tmp_path, health_interval_seconds=0.2)
    supervisor = Supervisor(config, command_builder=lambda _: SLEEP_LONG)

    thread = _run_in_thread(supervisor)
    time.sleep(0.3)
    supervisor.request_stop()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert "[HERMES][HEALTH]" in caplog.text


def test_graceful_stop_logs_stop_event_not_crash(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="hermes.supervisor")
    config = _fast_config(tmp_path)
    supervisor = Supervisor(config, command_builder=lambda _: SLEEP_LONG)

    thread = _run_in_thread(supervisor)
    time.sleep(0.1)
    supervisor.request_stop()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert "[HERMES][STOP]" in caplog.text
    assert "[HERMES][CRASH]" not in caplog.text


def test_request_stop_before_start_is_a_safe_noop(tmp_path: Path) -> None:
    config = _fast_config(tmp_path)
    supervisor = Supervisor(config, command_builder=lambda _: SLEEP_LONG)
    supervisor.request_stop()  # must not raise even though nothing has started


# -- crash / restart ---------------------------------------------------


def test_unexpected_exit_logs_crash_and_attempts_restart(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="hermes.supervisor")
    config = _fast_config(tmp_path, health_interval_seconds=0.2, max_restarts=2)
    supervisor = Supervisor(
        config, command_builder=lambda _: EXIT_IMMEDIATELY, sleep=lambda _: None
    )

    with pytest.raises(ProcessError):
        supervisor.run()

    assert "[HERMES][CRASH]" in caplog.text
    assert "[HERMES][RESTART]" in caplog.text


def test_restart_recovers_once_process_stabilizes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="hermes.supervisor")
    config = _fast_config(tmp_path, health_interval_seconds=0.2, max_restarts=3)

    calls = {"count": 0}

    def command_builder(_config: ProcessConfig) -> list[str]:
        calls["count"] += 1
        return EXIT_IMMEDIATELY if calls["count"] == 1 else SLEEP_LONG

    supervisor = Supervisor(config, command_builder=command_builder, sleep=lambda _: None)

    thread = _run_in_thread(supervisor)
    time.sleep(0.4)
    supervisor.request_stop()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert "[HERMES][CRASH]" in caplog.text
    assert "Restart attempt 1 succeeded" in caplog.text


def test_restarts_exhausted_raises_process_error(tmp_path: Path) -> None:
    config = _fast_config(tmp_path, health_interval_seconds=0.2, max_restarts=2)
    supervisor = Supervisor(
        config, command_builder=lambda _: EXIT_IMMEDIATELY, sleep=lambda _: None
    )

    with pytest.raises(ProcessError, match="failed to restart after 2 attempts"):
        supervisor.run()


def test_restart_failing_to_launch_is_logged_and_retried(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="hermes.supervisor")
    config = _fast_config(tmp_path, health_interval_seconds=0.2, max_restarts=2)

    calls = {"count": 0}

    def command_builder(_config: ProcessConfig) -> list[str]:
        calls["count"] += 1
        if calls["count"] == 2:
            # The second launch (first restart attempt) fails outright.
            return ["/no/such/executable-hermes-test"]
        return EXIT_IMMEDIATELY

    supervisor = Supervisor(config, command_builder=command_builder, sleep=lambda _: None)

    with pytest.raises(ProcessError):
        supervisor.run()

    assert "failed to launch" in caplog.text


# -- output capture ----------------------------------------------------


def test_stdout_and_stderr_are_captured_and_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="hermes.supervisor")
    config = _fast_config(tmp_path, health_interval_seconds=0.2, max_restarts=1)
    supervisor = Supervisor(
        config, command_builder=lambda _: PRINT_THEN_EXIT, sleep=lambda _: None
    )

    with pytest.raises(ProcessError):
        supervisor.run()
    time.sleep(0.2)  # let the reader threads catch up with the exited process

    assert "hello-stdout" in caplog.text
    assert "hello-warn" in caplog.text


# -- signal handling -----------------------------------------------------


def test_install_signal_handlers_triggers_request_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fast_config(tmp_path)
    supervisor = Supervisor(config, command_builder=lambda _: SLEEP_LONG)

    previous_term = signal.signal(signal.SIGTERM, signal.SIG_DFL)
    try:
        install_signal_handlers(supervisor)
        thread = _run_in_thread(supervisor)
        time.sleep(0.1)

        assert not supervisor._stop_requested.is_set()
        signal.raise_signal(signal.SIGTERM)
        thread.join(timeout=5)

        assert not thread.is_alive()
        assert supervisor._stop_requested.is_set()
    finally:
        signal.signal(signal.SIGTERM, previous_term)


# -- memory wiring -------------------------------------------------------


def test_start_and_graceful_stop_are_recorded_to_memory(tmp_path: Path) -> None:
    memory_store = MemoryStore(tmp_path / "memory.sqlite3")
    config = _fast_config(tmp_path)
    supervisor = Supervisor(config, command_builder=lambda _: SLEEP_LONG, memory_store=memory_store)

    thread = _run_in_thread(supervisor)
    time.sleep(0.1)
    supervisor.request_stop()
    thread.join(timeout=5)

    assert not thread.is_alive()
    events = memory_store.get_process_events()
    event_types = [e.event_type for e in events]
    assert event_types == ["start", "stop"]
    assert events[0].pid is not None


def test_crash_and_restart_are_recorded_to_memory(tmp_path: Path) -> None:
    memory_store = MemoryStore(tmp_path / "memory.sqlite3")
    config = _fast_config(tmp_path, health_interval_seconds=0.2, max_restarts=3)

    calls = {"count": 0}

    def command_builder(_config: ProcessConfig) -> list[str]:
        calls["count"] += 1
        return EXIT_IMMEDIATELY if calls["count"] == 1 else SLEEP_LONG

    supervisor = Supervisor(
        config, command_builder=command_builder, sleep=lambda _: None, memory_store=memory_store
    )

    thread = _run_in_thread(supervisor)
    time.sleep(0.4)
    supervisor.request_stop()
    thread.join(timeout=5)

    assert not thread.is_alive()
    events = memory_store.get_process_events()
    event_types = [e.event_type for e in events]
    assert event_types == ["start", "crash", "restart", "start", "restart", "stop"]
    # The "restart attempt succeeded" event carries the recovered pid.
    restart_events = [e for e in events if e.event_type == "restart"]
    assert all(e.message is not None for e in restart_events)


def test_restarts_exhausted_records_a_critical_error(tmp_path: Path) -> None:
    memory_store = MemoryStore(tmp_path / "memory.sqlite3")
    config = _fast_config(tmp_path, health_interval_seconds=0.2, max_restarts=2)
    supervisor = Supervisor(
        config,
        command_builder=lambda _: EXIT_IMMEDIATELY,
        sleep=lambda _: None,
        memory_store=memory_store,
    )

    with pytest.raises(ProcessError):
        supervisor.run()

    [error] = memory_store.get_errors()
    assert error.source == "hermes.supervisor"
    assert error.severity == "critical"
    assert "failed to restart" in error.message


def test_no_memory_store_is_a_safe_default(tmp_path: Path) -> None:
    config = _fast_config(tmp_path)
    supervisor = Supervisor(config, command_builder=lambda _: SLEEP_LONG)  # memory_store omitted

    thread = _run_in_thread(supervisor)
    time.sleep(0.1)
    supervisor.request_stop()
    thread.join(timeout=5)

    assert not thread.is_alive()  # must not raise for lacking a memory_store


def test_broken_memory_store_does_not_interrupt_supervision(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.ERROR, logger="hermes.supervisor")

    class ExplodingMemoryStore:
        def record_process_event(self, *args, **kwargs):
            raise RuntimeError("simulated disk failure")

        def record_error(self, *args, **kwargs):
            raise RuntimeError("simulated disk failure")

    config = _fast_config(tmp_path)
    supervisor = Supervisor(
        config, command_builder=lambda _: SLEEP_LONG, memory_store=ExplodingMemoryStore()
    )

    thread = _run_in_thread(supervisor)
    time.sleep(0.1)
    supervisor.request_stop()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert "[HERMES][MEMORY][ERROR] Failed to persist start event" in caplog.text
