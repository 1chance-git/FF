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
* **Persistence to Hermes memory is optional and strictly best-effort.**
  `BacktestLauncher` takes an optional `hermes.memory.MemoryStore` and,
  after every run, records the result via
  `memory_store.record_backtest_result()` — but this never changes what
  `run()` returns, and never raises: a database failure is logged as
  `[HERMES][MEMORY][ERROR]` and otherwise ignored. This module's own
  `BacktestResult` (the subprocess outcome: command/exit
  code/stdout/stderr/duration) is distinct from
  `hermes.memory.BacktestResult` (the persisted row: strategy/timerange/
  a JSON `metrics` blob) — rather than growing the memory schema with
  columns mirroring this one, the whole subprocess outcome plus the
  config identifiers not already covered by `strategy`/`timerange`
  (config files, timeframe, strategy path) are packed into `metrics`,
  since `metrics` already exists precisely to hold run-shaped data that
  varies by caller instead of forcing a schema migration per new field.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from hermes.export_paths import ExportPathError, prepare_export_directory
from hermes.memory import BacktestResult as MemoryBacktestResult
from hermes.memory import MemoryStore

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
    export_type:
        Optional Freqtrade ``--export`` artifact type to request: one of
        ``"trades"``, ``"signals"``, or ``"none"`` -- the installed
        Freqtrade version's ``backtesting`` subcommand accepts exactly
        one of these (confirmed via ``freqtrade backtesting --help``; it
        is not a repeatable/multi-value flag). Passing this without
        ``export_directory`` still lets Freqtrade choose its own
        (unvalidated, likely-ephemeral) default location -- pair it with
        ``export_directory`` to guarantee the artifact lands somewhere
        proven persistent.
    export_directory:
        Optional directory Freqtrade should write the requested export
        artifact to (passed as ``--export-directory``, the installed
        version's non-deprecated flag for this -- ``--export-filename``/
        ``--backtest-filename`` is deprecated and names a *file*, not a
        directory). Purely observational: this instructs Freqtrade to
        *additionally* persist data it already computes internally, and
        changes nothing about which trades the strategy takes or how
        they're evaluated. :func:`hermes.export_paths.prepare_export_directory`
        validates this is actually inside proven-persistent storage
        before the backtest is launched -- see :meth:`BacktestLauncher.run`.
    """

    strategy: str
    config_files: tuple[Path, ...]
    timerange: str | None = None
    timeframe: str | None = None
    strategy_path: Path | None = None
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    export_type: str | None = None
    export_directory: Path | None = None

    _VALID_EXPORT_TYPES = ("none", "trades", "signals")

    def __post_init__(self) -> None:
        if not self.strategy:
            raise BacktestError("strategy must be a non-empty string")
        if not self.config_files:
            raise BacktestError("config_files must contain at least one path")
        if self.export_type is not None and self.export_type not in self._VALID_EXPORT_TYPES:
            raise BacktestError(
                f"export_type must be one of {self._VALID_EXPORT_TYPES}, got: {self.export_type!r}"
            )
        if self.export_directory is not None and self.export_type is None:
            raise BacktestError(
                "export_directory was given but export_type is unset; "
                "specify what to export (e.g. 'trades')"
            )


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

    if config.export_type is not None:
        command += ["--export", config.export_type]
    if config.export_directory is not None:
        command += ["--export-directory", str(config.export_directory)]

    command += list(config.extra_args)
    return command


class BacktestLauncher:
    """Launches `freqtrade backtesting` as a subprocess and reports the outcome."""

    def __init__(
        self,
        memory_store: MemoryStore | None = None,
        *,
        export_persistent_root: Path | None = None,
    ) -> None:
        """Initialize the launcher.

        Parameters
        ----------
        memory_store:
            Optional `hermes.memory.MemoryStore` to persist each
            completed run's result to. Omit to run with no persistence
            (e.g. existing callers, or tests that don't care about it).
        export_persistent_root:
            The proven-persistent directory any `BacktestConfig.export_directory`
            must live inside (see `hermes.export_paths`). Required only
            if a config passed to :meth:`run` sets `export_directory`;
            omitted entirely, this launcher behaves exactly as it did
            before trade-export support existed.
        """
        self.memory_store = memory_store
        self.export_persistent_root = export_persistent_root

    def _preflight_export_directory(self, export_directory: Path) -> Path:
        """Validate `export_directory` before any subprocess is launched.

        Raises `BacktestError` (never launching a backtest) if the
        directory can't be proven both inside persistent storage and
        writable -- fail-safe, per this launcher's contract of never
        raising *from* a completed backtest but always raising on its
        own invalid configuration.
        """
        if self.export_persistent_root is None:
            raise BacktestError(
                "config.export_directory was set, but this BacktestLauncher "
                "was constructed without export_persistent_root; refusing "
                "to guess where persistent storage is"
            )
        try:
            return prepare_export_directory(
                export_directory, persistent_root=self.export_persistent_root
            )
        except ExportPathError as exc:
            raise BacktestError(f"invalid export directory: {exc}") from exc

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
        if config.export_directory is not None:
            self._preflight_export_directory(config.export_directory)

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

        self._record_result(config, result)
        return result

    def _record_result(self, config: BacktestConfig, result: BacktestResult) -> None:
        """Best-effort persistence of `result` to Hermes memory; never raises.

        Persists the strategy and timerange as their own columns
        (`hermes.memory.BacktestResult` already has them) and packs
        everything else this launcher knows about the run — the
        subprocess outcome plus the configuration identifiers not
        already covered — into `metrics`, so a database failure or an
        absent `memory_store` never affects the value `run()` returns.
        """
        if self.memory_store is None:
            return

        metrics = {
            "exit_code": result.exit_code,
            "succeeded": result.succeeded,
            "duration_seconds": result.duration_seconds,
            "command": list(result.command),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "config_files": [str(path) for path in config.config_files],
            "timeframe": config.timeframe,
            "strategy_path": str(config.strategy_path) if config.strategy_path else None,
            "extra_args": list(config.extra_args),
            "export_type": config.export_type,
            "export_directory": str(config.export_directory) if config.export_directory else None,
        }

        try:
            recorded = self.memory_store.record_backtest_result(
                MemoryBacktestResult(
                    strategy=config.strategy,
                    timerange=config.timerange,
                    metrics=metrics,
                )
            )
            if not recorded:
                logger.error(
                    "[HERMES][MEMORY][ERROR] Failed to persist backtest result "
                    "(backtest result is unaffected)"
                )
        except Exception as exc:
            logger.error(
                "[HERMES][MEMORY][ERROR] Failed to persist backtest result "
                "(backtest result is unaffected): %s",
                exc,
                exc_info=True,
            )
