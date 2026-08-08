"""Backtest launcher: builds and runs `freqtrade backtesting` as a subprocess.

Hermes never re-implements or reaches into Freqtrade's backtesting
engine — it treats the installed `freqtrade` CLI as the single source of
truth for what a backtest is and launches it as an external process.
This keeps Hermes correctly decoupled: Freqtrade version upgrades that
add/change backtesting flags don't require touching this module, only
the (thin, tested) argument list this module builds.

Design decisions
-----------------
* **Command construction and command execution are two different,
  independently testable functions.** `build_backtest_command` is a pure
  function (given a `BacktestConfig`, returns the exact argv list) that
  can be asserted against in a unit test with no subprocess involved.
  `BacktestLauncher.run` is the thin, I/O-touching wrapper around
  `subprocess.run` that actually launches it — mirroring the same
  separation of pure logic from I/O used throughout this project (e.g.
  `stat_arb.data.market_data.MarketDataLoader` vs. its pure
  `clean_and_fill`/`validate_ohlcv` helpers).
* **`sys.executable -m freqtrade` instead of a bare `"freqtrade"`
  argv[0].** Invoking the CLI via the current Python interpreter's `-m`
  flag guarantees the backtest runs inside the same virtual environment
  Hermes itself is running in, regardless of what (if anything) is on
  `$PATH` — the same robust pattern used for subprocess-launching
  Python tools generally.
* **The launcher returns a structured `BacktestResult` rather than
  raising on a non-zero exit code.** A failed backtest (bad strategy,
  missing data, invalid timerange) is an expected outcome an operator
  needs to see the stdout/stderr for, not an exceptional condition that
  should unwind the caller's stack. `BacktestResult.succeeded` lets a
  caller branch on outcome without needing a try/except.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


class BacktestError(Exception):
    """Raised for invalid backtest configuration (not for a failed backtest run)."""


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for a single backtest run.

    Parameters
    ----------
    strategy:
        Strategy class name to backtest, e.g. ``"StatArbSwing"``.
    config_files:
        One or more Freqtrade config file paths, passed as repeated
        ``-c`` flags (later files override earlier ones, same as the
        Freqtrade CLI itself).
    timerange:
        Optional Freqtrade timerange string, e.g. ``"20240101-20240401"``.
    timeframe:
        Optional timeframe override, e.g. ``"1h"``.
    strategy_path:
        Optional directory to search for the strategy, if not already
        covered by ``config_files``.
    extra_args:
        Additional raw CLI arguments appended verbatim (escape hatch for
        flags this dataclass doesn't model explicitly, e.g.
        ``["--breakdown", "day"]``).
    """

    strategy: str
    config_files: tuple[Path, ...]
    timerange: str | None = None
    timeframe: str | None = None
    strategy_path: Path | None = None
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.strategy:
            raise BacktestError("strategy must be a non-empty string")
        if not self.config_files:
            raise BacktestError("config_files must contain at least one path")


@dataclass(frozen=True)
class BacktestResult:
    """Outcome of running a backtest subprocess.

    Attributes
    ----------
    command:
        The exact argv list that was executed.
    exit_code:
        Process exit code (``0`` on success).
    stdout:
        Captured standard output.
    stderr:
        Captured standard error.
    duration_seconds:
        Wall-clock time the subprocess took to run.
    """

    command: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float

    @property
    def succeeded(self) -> bool:
        """``True`` if the backtest process exited with code 0."""
        return self.exit_code == 0


def build_backtest_command(config: BacktestConfig) -> list[str]:
    """Build the `freqtrade backtesting` argv for ``config``.

    Pure function: no subprocess, no I/O. Fully deterministic given its
    input, which is what makes it independently unit-testable.

    Parameters
    ----------
    config:
        Backtest configuration.

    Returns
    -------
    list[str]
        Full argv, e.g.
        ``[sys.executable, "-m", "freqtrade", "backtesting", "-c", "...", "--strategy", "..."]``.
    """
    command = [sys.executable, "-m", "freqtrade", "backtesting"]

    for config_file in config.config_files:
        command += ["-c", str(config_file)]

    command += ["--strategy", config.strategy]

    if config.strategy_path is not None:
        command += ["--strategy-path", str(config.strategy_path)]
    if config.timerange is not None:
        command += ["--timerange", config.timerange]
    if config.timeframe is not None:
        command += ["--timeframe", config.timeframe]

    command += list(config.extra_args)
    return command


class BacktestLauncher:
    """Launches `freqtrade backtesting` as a subprocess and reports the outcome."""

    def run(self, config: BacktestConfig, timeout_seconds: float | None = None) -> BacktestResult:
        """Run a backtest and return its result.

        Parameters
        ----------
        config:
            Backtest configuration.
        timeout_seconds:
            Optional wall-clock timeout. On expiry, the subprocess is
            killed and a :class:`BacktestError` is raised (a timeout is
            an operational failure of the launcher itself, distinct from
            a backtest that ran to completion and simply failed).

        Returns
        -------
        BacktestResult

        Raises
        ------
        BacktestError
            If the process times out.
        """
        command = build_backtest_command(config)
        logger.info("Launching backtest: %s", " ".join(command))

        start = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - start
            logger.error("Backtest timed out after %.1fs: %s", duration, " ".join(command))
            raise BacktestError(
                f"Backtest timed out after {timeout_seconds}s"
            ) from exc
        duration = time.monotonic() - start

        result = BacktestResult(
            command=tuple(command),
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_seconds=duration,
        )
        if result.succeeded:
            logger.info("Backtest completed successfully in %.1fs", duration)
        else:
            logger.warning(
                "Backtest exited with code %d after %.1fs", result.exit_code, duration
            )
        return result
