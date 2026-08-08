"""Process lifecycle management and restart support for a Freqtrade bot process.

Manages the bot as an external OS process (`freqtrade trade`), the same
way `hermes.backtest` manages backtesting: launched via `subprocess`,
never imported into and run in-process. Liveness and graceful shutdown
are implemented on top of `psutil`, a mature, cross-platform process
management library, rather than hand-rolled `os.kill(pid, 0)` probing
and raw signal handling.

Design decisions
-----------------
* **A PID file is the single source of truth for "is a bot running".**
  It is written on start and removed on a clean stop, so a fresh
  `BotProcessManager` (e.g. after Hermes itself restarts) can rediscover
  a still-running bot rather than losing track of it. `is_running` also
  confirms the PID file's process is actually the process Hermes
  started (matching command line), so a stale PID file pointing at an
  unrelated, since-reused PID is correctly reported as not running
  rather than producing a false positive.
* **Graceful shutdown, then escalate.** `stop()` sends `SIGTERM` and
  waits up to a configurable grace period for the process to exit on
  its own — Freqtrade handles `SIGTERM` by finishing its current loop
  iteration and shutting down cleanly (closing open orders' state,
  flushing the database) — before sending `SIGKILL` as a last resort.
  Killing immediately would risk leaving the bot's SQLite database or an
  in-flight order in an inconsistent state.
* **`restart()` composes `stop()` and `start()` rather than duplicating
  their logic**, and `RestartSupervisor` composes `restart()` with a
  bounded, exponentially-backed-off retry loop (2s, 4s, 8s, ... capped)
  for unattended crash recovery — the same backoff shape used elsewhere
  in this project's tooling conventions, not an ad hoc retry scheme.
  The retry count is bounded (`max_restarts`) so a persistently crashing
  bot fails loudly instead of spinning forever.
"""

from __future__ import annotations

import logging
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import psutil

logger = logging.getLogger(__name__)


class ProcessError(Exception):
    """Raised for process management failures (e.g. failing to start or stop)."""


@dataclass(frozen=True)
class ProcessConfig:
    """Configuration for :class:`BotProcessManager`.

    Parameters
    ----------
    config_files:
        Freqtrade config file paths, passed as repeated ``-c`` flags.
    strategy:
        Strategy class name to run.
    pid_file:
        Path to the PID file used to track the managed process.
    strategy_path:
        Optional directory to search for the strategy.
    extra_args:
        Additional raw CLI arguments appended verbatim.
    stop_grace_period_seconds:
        How long to wait after ``SIGTERM`` before escalating to
        ``SIGKILL``.
    """

    config_files: tuple[Path, ...]
    strategy: str
    pid_file: Path
    strategy_path: Path | None = None
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    stop_grace_period_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not self.config_files:
            raise ProcessError("config_files must contain at least one path")
        if not self.strategy:
            raise ProcessError("strategy must be a non-empty string")
        if self.stop_grace_period_seconds <= 0:
            raise ProcessError("stop_grace_period_seconds must be positive")


def build_trade_command(config: ProcessConfig) -> list[str]:
    """Build the `freqtrade trade` argv for ``config``. Pure, no I/O.

    Parameters
    ----------
    config:
        Process configuration.

    Returns
    -------
    list[str]
        Full argv for launching the bot.
    """
    command = [sys.executable, "-m", "freqtrade", "trade"]
    for config_file in config.config_files:
        command += ["-c", str(config_file)]
    command += ["--strategy", config.strategy]
    if config.strategy_path is not None:
        command += ["--strategy-path", str(config.strategy_path)]
    command += list(config.extra_args)
    return command


class BotProcessManager:
    """Starts, stops, and restarts a Freqtrade bot process, tracked via a PID file."""

    def __init__(
        self,
        config: ProcessConfig,
        command_builder: Callable[[ProcessConfig], list[str]] = build_trade_command,
        process_matcher: Callable[[str], bool] = lambda cmdline: "freqtrade" in cmdline,
    ) -> None:
        """Initialize the manager.

        Parameters
        ----------
        config:
            Process configuration.
        command_builder:
            Function producing the argv to launch. Defaults to
            :func:`build_trade_command`; overridable in tests to launch
            a lightweight stand-in process instead of the real bot.
        process_matcher:
            Predicate over a candidate process' joined command line,
            used to confirm a PID file's process is actually the bot
            (or its test stand-in) rather than an unrelated process that
            reused the same PID. Defaults to checking for
            ``"freqtrade"``; overridable in tests alongside
            ``command_builder``.
        """
        self.config = config
        self._command_builder = command_builder
        self._process_matcher = process_matcher

    def start(self) -> int:
        """Start the bot process if not already running.

        Returns
        -------
        int
            PID of the running process (newly started, or already
            running).

        Raises
        ------
        ProcessError
            If the process fails to launch.
        """
        existing_pid = self._read_running_pid()
        if existing_pid is not None:
            logger.info("Bot already running (pid=%d); start() is a no-op", existing_pid)
            return existing_pid

        command = self._command_builder(self.config)
        logger.info("Starting bot: %s", " ".join(command))
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            raise ProcessError(f"Failed to launch bot process: {exc}") from exc

        self.config.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.config.pid_file.write_text(str(process.pid), encoding="utf-8")
        logger.info("Bot started (pid=%d)", process.pid)
        return process.pid

    def stop(self) -> bool:
        """Stop the running bot process, if any.

        Sends ``SIGTERM`` and waits up to
        ``config.stop_grace_period_seconds`` for the process to exit,
        then escalates to ``SIGKILL`` if it hasn't.

        Returns
        -------
        bool
            ``True`` if a running process was found and stopped,
            ``False`` if nothing was running.
        """
        pid = self._read_running_pid()
        if pid is None:
            logger.info("No running bot found; stop() is a no-op")
            self._remove_pid_file()
            return False

        logger.info("Stopping bot (pid=%d): sending SIGTERM", pid)
        try:
            process = psutil.Process(pid)
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=self.config.stop_grace_period_seconds)
        except psutil.NoSuchProcess:
            pass
        except psutil.TimeoutExpired:
            logger.warning(
                "Bot (pid=%d) did not exit within %.0fs; sending SIGKILL",
                pid,
                self.config.stop_grace_period_seconds,
            )
            try:
                psutil.Process(pid).kill()
            except psutil.NoSuchProcess:
                pass

        self._remove_pid_file()
        logger.info("Bot stopped (pid=%d)", pid)
        return True

    def restart(self) -> int:
        """Stop the bot (if running) and start it again.

        Returns
        -------
        int
            PID of the newly started process.
        """
        logger.info("Restarting bot")
        self.stop()
        return self.start()

    def is_running(self) -> bool:
        """Whether the managed bot process is currently running."""
        return self._read_running_pid() is not None

    def status(self) -> dict[str, object]:
        """Return a small status summary suitable for CLI/health-check display."""
        pid = self._read_running_pid()
        if pid is None:
            return {"running": False, "pid": None}
        try:
            process = psutil.Process(pid)
            return {
                "running": True,
                "pid": pid,
                "cpu_percent": process.cpu_percent(interval=0.1),
                "memory_mb": process.memory_info().rss / (1024 * 1024),
                "uptime_seconds": time.time() - process.create_time(),
            }
        except psutil.NoSuchProcess:
            return {"running": False, "pid": None}

    def _read_running_pid(self) -> int | None:
        """Read the PID file and confirm the process it names is actually our bot."""
        if not self.config.pid_file.exists():
            return None

        try:
            pid = int(self.config.pid_file.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            return None

        if not psutil.pid_exists(pid):
            self._remove_pid_file()
            return None

        try:
            process = psutil.Process(pid)
            cmdline = " ".join(process.cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            self._remove_pid_file()
            return None

        if not self._process_matcher(cmdline):
            # The PID was recycled by an unrelated process; the bot isn't running.
            self._remove_pid_file()
            return None

        return pid

    def _remove_pid_file(self) -> None:
        self.config.pid_file.unlink(missing_ok=True)


@dataclass(frozen=True)
class RestartSupervisorConfig:
    """Configuration for :class:`RestartSupervisor`.

    Parameters
    ----------
    max_restarts:
        Maximum number of consecutive restart attempts before giving up.
    initial_backoff_seconds:
        Delay before the first restart attempt.
    max_backoff_seconds:
        Backoff is doubled after each attempt, capped at this value.
    """

    max_restarts: int = 4
    initial_backoff_seconds: float = 2.0
    max_backoff_seconds: float = 16.0

    def __post_init__(self) -> None:
        if self.max_restarts < 1:
            raise ProcessError("max_restarts must be >= 1")
        if self.initial_backoff_seconds <= 0:
            raise ProcessError("initial_backoff_seconds must be positive")
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ProcessError("max_backoff_seconds must be >= initial_backoff_seconds")

    def backoff_schedule(self) -> list[float]:
        """The sequence of backoff delays (seconds) for each restart attempt."""
        schedule = []
        delay = self.initial_backoff_seconds
        for _ in range(self.max_restarts):
            schedule.append(min(delay, self.max_backoff_seconds))
            delay *= 2
        return schedule


class RestartSupervisor:
    """Restarts a bot process on crash with a bounded, exponentially-backed-off retry loop.

    Unlike :meth:`BotProcessManager.restart`, which is a single
    unconditional stop-then-start, this supervisor is meant to be driven
    by an external crash detector (e.g. an operator or a monitoring
    loop noticing ``BotProcessManager.is_running()`` has gone
    unexpectedly false) and handles the "keep retrying, but not
    forever" policy around a single restart attempt.
    """

    def __init__(
        self,
        manager: BotProcessManager,
        config: RestartSupervisorConfig | None = None,
        sleep: callable = time.sleep,
    ) -> None:
        """Initialize the supervisor.

        Parameters
        ----------
        manager:
            The process manager to restart.
        config:
            Restart policy configuration.
        sleep:
            Sleep function to call between attempts. Overridable for
            tests so they don't have to wait in real time.
        """
        self.manager = manager
        self.config = config or RestartSupervisorConfig()
        self._sleep = sleep

    def handle_crash(self) -> int:
        """Attempt to restart the bot, retrying with backoff on failure.

        Returns
        -------
        int
            PID of the successfully restarted process.

        Raises
        ------
        ProcessError
            If all ``config.max_restarts`` attempts fail.
        """
        backoff_schedule = self.config.backoff_schedule()
        last_error: Exception | None = None

        for attempt, delay in enumerate(backoff_schedule, start=1):
            logger.warning(
                "Restart attempt %d/%d in %.1fs", attempt, self.config.max_restarts, delay
            )
            self._sleep(delay)
            try:
                pid = self.manager.restart()
                logger.info("Restart attempt %d succeeded (pid=%d)", attempt, pid)
                return pid
            except ProcessError as exc:
                last_error = exc
                logger.error("Restart attempt %d failed: %s", attempt, exc)

        raise ProcessError(
            f"Bot failed to restart after {self.config.max_restarts} attempts"
        ) from last_error
