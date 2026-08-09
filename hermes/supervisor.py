"""Process supervisor for a Freqtrade bot: start, monitor, capture output, restart on crash.

Hermes is an external sidecar — this module starts and watches the
Freqtrade process from outside, but never touches trading logic or
strategy code. It is deliberately minimal: a subprocess, two reader
threads for its stdout/stderr, and a poll loop. No process pool, no
IPC beyond what `subprocess` already gives us, no external services.

Design decisions
-----------------
* **`subprocess.Popen` directly, no `psutil`.** `hermes.process` already
  uses `psutil` for the operator-facing start/stop/restart CLI, where
  identifying a bot process by PID-file + command-line match matters
  (an operator might run `hermes status` from a fresh process and need
  to rediscover a still-running bot). This module owns the process for
  its own lifetime instead — it starts it and holds the `Popen` handle
  directly — so a lighter, dependency-free approach is enough here and
  keeps this module runnable with nothing beyond the standard library.
* **Command construction is reused, not duplicated.** `build_trade_command`
  and `ProcessConfig` from `hermes.process` already build and validate
  the `freqtrade trade` argv; this module composes them rather than
  re-implementing config-file/strategy handling.
* **Output is streamed, not buffered.** Two daemon threads read the
  child's stdout/stderr line-by-line and forward each line to the
  logger, rather than capturing to a string (unbounded memory) or
  leaving the pipes unread, which would eventually fill the OS pipe
  buffer and deadlock the child.
* **A crash is "the process exited and nobody asked it to."** There is
  no special-casing of exit codes: a live-trading bot process is not
  expected to exit at all during normal operation, so *any* exit that
  wasn't preceded by a call to `request_stop()` is treated as a crash
  and triggers the restart path, exit code 0 included.
* **Restart attempts are bounded by actually surviving, not just
  launching.** A naive "retry N times, success = the subprocess call
  didn't raise" would let a process that starts fine and immediately
  crashes again restart forever, since each individual `Popen()` call
  "succeeds" — it just doesn't stay running. Instead, each restart
  attempt must survive one full health-check interval before it counts
  as a successful recovery; otherwise it consumes one of
  `max_restarts` attempts, so a persistently crashing bot fails loudly
  (raises `ProcessError`) instead of spinning forever.
* **Signal handling lives outside `run()`.** `signal.signal()` may only
  be called from the main thread of the main interpreter — calling it
  inside `run()` would make the supervisor untestable from a background
  thread. `install_signal_handlers()` is a separate, explicit step the
  process entrypoint calls once before `run()`; `request_stop()` itself
  has no such restriction and is what the installed handlers (or a
  test, or any other caller) actually invoke.
* **Memory recording is optional and never allowed to affect supervision.**
  `Supervisor` takes an optional `hermes.memory.MemoryStore` and records
  start/crash/stop events (and a final error if restarts are exhausted)
  to it — but `MemoryStore` is injected, not constructed here, so a
  caller that doesn't want persistence can simply omit it, and every
  recording call is wrapped so a database failure is logged and
  swallowed rather than interrupting supervision (the same principle
  `StatArbSwing.confirm_trade_entry`/`confirm_trade_exit` apply to
  trade recording).
"""

from __future__ import annotations

import logging
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from hermes.memory import ErrorEvent, MemoryStore, ProcessEvent
from hermes.process import ProcessConfig, ProcessError, build_trade_command

logger = logging.getLogger(__name__)

TAG_START = "START"
TAG_HEALTH = "HEALTH"
TAG_CRASH = "CRASH"
TAG_RESTART = "RESTART"
TAG_STOP = "STOP"


def _log(tag: str, message: str, *, level: int = logging.INFO) -> None:
    """Emit a structured ``[HERMES][TAG] message`` log line."""
    logger.log(level, "[HERMES][%s] %s", tag, message)


@dataclass(frozen=True)
class SupervisorConfig:
    """Configuration for :class:`Supervisor`.

    Parameters
    ----------
    process_config:
        Freqtrade launch configuration (config files, strategy, ...),
        reused from `hermes.process.ProcessConfig`.
    health_interval_seconds:
        How often to log a `[HERMES][HEALTH]` heartbeat while the bot is
        alive; also the minimum time a restarted process must stay up
        to count as a successful recovery.
    max_restarts:
        Maximum consecutive restart attempts after a crash before the
        supervisor gives up and raises `ProcessError`.
    initial_backoff_seconds:
        Delay before the first restart attempt after a crash.
    max_backoff_seconds:
        Backoff doubles after each failed attempt, capped at this value.
    """

    process_config: ProcessConfig
    health_interval_seconds: float = 30.0
    max_restarts: int = 4
    initial_backoff_seconds: float = 2.0
    max_backoff_seconds: float = 16.0

    def __post_init__(self) -> None:
        if self.health_interval_seconds <= 0:
            raise ProcessError("health_interval_seconds must be positive")
        if self.max_restarts < 1:
            raise ProcessError("max_restarts must be >= 1")
        if self.initial_backoff_seconds <= 0:
            raise ProcessError("initial_backoff_seconds must be positive")
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ProcessError("max_backoff_seconds must be >= initial_backoff_seconds")

    def backoff_schedule(self) -> list[float]:
        """The sequence of backoff delays (seconds), one per restart attempt."""
        schedule = []
        delay = self.initial_backoff_seconds
        for _ in range(self.max_restarts):
            schedule.append(min(delay, self.max_backoff_seconds))
            delay *= 2
        return schedule


def _stream_output(pipe: IO[str], log_fn: Callable[[str], None]) -> None:
    """Forward each line from `pipe` to `log_fn` until the pipe closes."""
    try:
        for line in iter(pipe.readline, ""):
            stripped = line.rstrip("\n")
            if stripped:
                log_fn(stripped)
    finally:
        pipe.close()


class Supervisor:
    """Starts, watches, and restarts a Freqtrade bot process.

    Usage::

        config = SupervisorConfig(process_config=ProcessConfig(...))
        supervisor = Supervisor(config)
        install_signal_handlers(supervisor)  # main thread only
        supervisor.run()  # blocks until stopped or restarts are exhausted
    """

    def __init__(
        self,
        config: SupervisorConfig,
        command_builder: Callable[[ProcessConfig], list[str]] = build_trade_command,
        sleep: Callable[[float], None] = time.sleep,
        memory_store: MemoryStore | None = None,
    ) -> None:
        """Initialize the supervisor.

        Parameters
        ----------
        config:
            Supervisor configuration.
        command_builder:
            Function producing the argv to launch. Defaults to
            `build_trade_command`; overridable in tests to launch a
            lightweight stand-in process instead of the real bot.
        sleep:
            Sleep function, overridable in tests so restart backoff
            doesn't block real time.
        memory_store:
            Optional `hermes.memory.MemoryStore` to record start/crash/
            stop events to, alongside the structured log lines. Omit to
            run with logging only.
        """
        self.config = config
        self._command_builder = command_builder
        self._sleep = sleep
        self.memory_store = memory_store
        self._stop_requested = threading.Event()
        self._process: subprocess.Popen | None = None

    def run(self) -> None:
        """Run the supervise loop until stopped or restarts are exhausted.

        Blocks the calling thread. Call `request_stop()` from another
        thread or a signal handler to end it gracefully.

        Raises
        ------
        ProcessError
            If the bot crashes and all configured restart attempts fail
            to produce a process that stays up for at least one health
            interval.
        """
        self._start()
        while not self._stop_requested.is_set():
            exit_code = self._wait_with_heartbeat()
            if exit_code is None:
                continue  # heartbeat fired; still alive

            if self._stop_requested.is_set():
                _log(TAG_STOP, f"Bot exited (pid={self._process.pid}, code={exit_code})")
                self._remember_event(TAG_STOP.lower(), pid=self._process.pid, message=f"exit code {exit_code}")
                return

            _log(
                TAG_CRASH,
                f"Bot exited unexpectedly (pid={self._process.pid}, code={exit_code})",
                level=logging.ERROR,
            )
            self._remember_event(TAG_CRASH.lower(), pid=self._process.pid, message=f"exit code {exit_code}")
            self._restart_with_backoff()

    def request_stop(self) -> None:
        """Signal the run loop to stop and gracefully terminate the bot.

        Safe to call from a signal handler or another thread.
        """
        self._stop_requested.set()
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()

    # -- internals ---------------------------------------------------

    def _remember_event(self, event_type: str, *, pid: int | None = None, message: str | None = None) -> None:
        """Best-effort record of a process event; never raises."""
        if self.memory_store is None:
            return
        try:
            self.memory_store.record_process_event(
                ProcessEvent(event_type=event_type, pid=pid, message=message)
            )
        except Exception:
            logger.exception("Failed to record %s event to Hermes memory (supervision unaffected)", event_type)

    def _remember_error(self, message: str) -> None:
        """Best-effort record of a fatal supervisor error; never raises."""
        if self.memory_store is None:
            return
        try:
            self.memory_store.record_error(
                ErrorEvent(source="hermes.supervisor", message=message, severity="critical")
            )
        except Exception:
            logger.exception("Failed to record error to Hermes memory (supervision unaffected)")

    def _start(self) -> None:
        command = self._command_builder(self.config.process_config)
        _log(TAG_START, f"Starting bot: {' '.join(command)}")
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise ProcessError(f"Failed to launch bot process: {exc}") from exc

        self._process = process
        self._start_output_threads(process)
        _log(TAG_START, f"Bot started (pid={process.pid})")
        self._remember_event(TAG_START.lower(), pid=process.pid)

    def _start_output_threads(self, process: subprocess.Popen) -> None:
        threading.Thread(
            target=_stream_output,
            args=(process.stdout, lambda line: logger.info("[bot] %s", line)),
            daemon=True,
        ).start()
        threading.Thread(
            target=_stream_output,
            args=(process.stderr, lambda line: logger.warning("[bot] %s", line)),
            daemon=True,
        ).start()

    def _wait_with_heartbeat(self) -> int | None:
        """Wait up to one health interval; return the exit code if the bot exited, else None."""
        assert self._process is not None
        try:
            exit_code = self._process.wait(timeout=self.config.health_interval_seconds)
        except subprocess.TimeoutExpired:
            _log(TAG_HEALTH, f"Bot alive (pid={self._process.pid})")
            return None
        return exit_code

    def _restart_with_backoff(self) -> None:
        """Retry starting the bot with backoff until one attempt survives a health interval.

        Raises
        ------
        ProcessError
            If every attempt in `config.backoff_schedule()` either fails
            to launch or crashes again before surviving one health
            interval.
        """
        backoff_schedule = self.config.backoff_schedule()

        for attempt, delay in enumerate(backoff_schedule, start=1):
            if self._stop_requested.is_set():
                return

            _log(TAG_RESTART, f"Restart attempt {attempt}/{self.config.max_restarts} in {delay:.1f}s")
            self._sleep(delay)
            if self._stop_requested.is_set():
                return

            try:
                self._start()
            except ProcessError as exc:
                _log(TAG_RESTART, f"Restart attempt {attempt} failed to launch: {exc}", level=logging.ERROR)
                continue

            exit_code = self._wait_with_heartbeat()
            if exit_code is None:
                _log(TAG_RESTART, f"Restart attempt {attempt} succeeded (pid={self._process.pid})")
                return

            if self._stop_requested.is_set():
                _log(TAG_STOP, f"Bot exited (pid={self._process.pid}, code={exit_code})")
                self._remember_event(TAG_STOP.lower(), pid=self._process.pid, message=f"exit code {exit_code}")
                return

            _log(
                TAG_CRASH,
                f"Restart attempt {attempt} crashed again (pid={self._process.pid}, code={exit_code})",
                level=logging.ERROR,
            )
            self._remember_event(TAG_CRASH.lower(), pid=self._process.pid, message=f"exit code {exit_code}")

        message = f"Bot failed to restart after {self.config.max_restarts} attempts"
        self._remember_error(message)
        raise ProcessError(message)


def install_signal_handlers(supervisor: Supervisor) -> None:
    """Install `SIGTERM`/`SIGINT` handlers that trigger `supervisor.request_stop()`.

    Must be called from the main thread of the main interpreter — a
    Python/OS restriction on `signal.signal()` — typically once at
    process startup, before calling `supervisor.run()`.
    """

    def _handle(signum: int, frame: object) -> None:
        _log(TAG_STOP, f"Received signal {signal.Signals(signum).name}; shutting down")
        supervisor.request_stop()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Hermes process supervisor.")
    parser.add_argument("-c", "--config", action="append", required=True, dest="config_files")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--strategy-path", default=None)
    parser.add_argument("--pid-file", default="user_data/hermes_supervisor.pid")
    parser.add_argument("--user-data-dir", default="user_data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    process_config = ProcessConfig(
        config_files=tuple(Path(p) for p in args.config_files),
        strategy=args.strategy,
        pid_file=Path(args.pid_file),
        strategy_path=Path(args.strategy_path) if args.strategy_path else None,
    )

    memory_store = None
    try:
        memory_store = MemoryStore(Path(args.user_data_dir) / "hermes_memory.sqlite3")
    except Exception:
        logger.exception(
            "Failed to initialize Hermes memory store; process events will not be recorded"
        )

    supervisor = Supervisor(SupervisorConfig(process_config=process_config), memory_store=memory_store)
    install_signal_handlers(supervisor)
    supervisor.run()
