"""Hyperopt launcher: builds and runs `freqtrade hyperopt` as a subprocess.

Mirrors `hermes.backtest.BacktestLauncher`'s design exactly: Freqtrade's
own hyperopt engine (Optuna-based Bayesian parameter search under the
hood) is treated as an external process to launch, never reimplemented
or imported in-process. Reusing that design here — rather than inventing
a different pattern for "the optimizer" — keeps the two launchers
interchangeable in tooling (e.g. `hermes` or CI scripts) that only cares
about "build a command, run it, get a structured result back."

Design decisions
-----------------
* **Command construction and execution are separate, independently
  testable functions**, exactly as in `hermes.backtest`:
  `build_hyperopt_command` is pure (config in, argv out); `HyperoptLauncher.run`
  is the thin subprocess wrapper.
* **`--hyperopt-path` always points at this package's directory by
  default.** `optimize/hyperopt_loss.py` defines `StatArbHyperOptLoss`;
  Freqtrade only finds custom loss functions on its hyperopt lookup
  path, so the launcher wires that path automatically rather than
  requiring every caller to remember the flag.
* **Freqtrade's hyperopt is used for the entry/exit threshold space it
  was configured for in `StatArbSwing.py`** (`entry_zscore_param`,
  `exit_zscore_param`, both in the standard `buy`/`sell` spaces) — no
  indicator-space parameters, so hyperopt runs at Freqtrade's normal
  (fast) per-epoch cost rather than the slower indicator-recalculation
  path.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

#: This package's directory, passed as Freqtrade's --hyperopt-path so it
#: can find StatArbHyperOptLoss.
_DEFAULT_HYPEROPT_PATH = Path(__file__).resolve().parent


class HyperoptError(Exception):
    """Raised for invalid hyperopt configuration (not for a failed hyperopt run)."""


@dataclass(frozen=True)
class HyperoptConfig:
    """Configuration for a single `freqtrade hyperopt` run.

    Parameters
    ----------
    strategy:
        Strategy class name to optimize, e.g. ``"StatArbSwing"``.
    config_files:
        One or more Freqtrade config file paths, passed as repeated
        ``-c`` flags.
    epochs:
        Number of hyperopt epochs to run.
    spaces:
        Parameter spaces to optimize, e.g. ``("buy", "sell")`` — matches
        `entry_zscore_param`'s ``space="buy"`` / `exit_zscore_param`'s
        ``space="sell"`` in `StatArbSwing.py`.
    hyperopt_loss:
        Loss function class name. Defaults to this package's
        `StatArbHyperOptLoss`.
    hyperopt_path:
        Directory Freqtrade searches for the loss function. Defaults to
        this package's own directory.
    timerange:
        Optional Freqtrade timerange string to restrict optimization to
        (the in-sample/training window — see `optimize.walk_forward`
        for using this to avoid overfitting to the full dataset).
    timeframe:
        Optional timeframe override.
    strategy_path:
        Optional directory to search for the strategy.
    jobs:
        Number of parallel jobs. ``None`` uses Freqtrade's default.
    random_state:
        Optional random seed, for reproducible searches.
    min_trades:
        Optional minimum trade count filter passed to Freqtrade
        (separate from, and in addition to, the loss function's own
        `MIN_TRADES_FOR_VALID_LOSS` gate).
    extra_args:
        Additional raw CLI arguments appended verbatim.
    """

    strategy: str
    config_files: tuple[Path, ...]
    epochs: int = 100
    spaces: tuple[str, ...] = ("buy", "sell")
    hyperopt_loss: str = "StatArbHyperOptLoss"
    hyperopt_path: Path = _DEFAULT_HYPEROPT_PATH
    timerange: str | None = None
    timeframe: str | None = None
    strategy_path: Path | None = None
    jobs: int | None = None
    random_state: int | None = None
    min_trades: int | None = None
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.strategy:
            raise HyperoptError("strategy must be a non-empty string")
        if not self.config_files:
            raise HyperoptError("config_files must contain at least one path")
        if self.epochs < 1:
            raise HyperoptError("epochs must be >= 1")
        if not self.spaces:
            raise HyperoptError("spaces must contain at least one space name")


@dataclass(frozen=True)
class HyperoptResult:
    """Outcome of running a hyperopt subprocess.

    Attributes
    ----------
    command:
        The exact argv list that was executed.
    exit_code:
        Process exit code (``0`` on success).
    stdout, stderr:
        Captured output.
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
        """``True`` if the hyperopt process exited with code 0."""
        return self.exit_code == 0


def build_hyperopt_command(config: HyperoptConfig) -> list[str]:
    """Build the `freqtrade hyperopt` argv for ``config``. Pure, no I/O.

    Parameters
    ----------
    config:
        Hyperopt configuration.

    Returns
    -------
    list[str]
        Full argv.
    """
    command = [sys.executable, "-m", "freqtrade", "hyperopt"]

    for config_file in config.config_files:
        command += ["-c", str(config_file)]

    command += ["--strategy", config.strategy]
    command += ["--epochs", str(config.epochs)]
    command += ["--spaces", *config.spaces]
    command += ["--hyperopt-loss", config.hyperopt_loss]
    command += ["--hyperopt-path", str(config.hyperopt_path)]

    if config.strategy_path is not None:
        command += ["--strategy-path", str(config.strategy_path)]
    if config.timerange is not None:
        command += ["--timerange", config.timerange]
    if config.timeframe is not None:
        command += ["--timeframe", config.timeframe]
    if config.jobs is not None:
        command += ["-j", str(config.jobs)]
    if config.random_state is not None:
        command += ["--random-state", str(config.random_state)]
    if config.min_trades is not None:
        command += ["--min-trades", str(config.min_trades)]

    command += list(config.extra_args)
    return command


class HyperoptLauncher:
    """Launches `freqtrade hyperopt` as a subprocess and reports the outcome."""

    def run(self, config: HyperoptConfig, timeout_seconds: float | None = None) -> HyperoptResult:
        """Run a hyperopt search and return its result.

        Parameters
        ----------
        config:
            Hyperopt configuration.
        timeout_seconds:
            Optional wall-clock timeout. On expiry, the subprocess is
            killed and a :class:`HyperoptError` is raised.

        Returns
        -------
        HyperoptResult

        Raises
        ------
        HyperoptError
            If the process times out.
        """
        command = build_hyperopt_command(config)
        logger.info("Launching hyperopt: %s", " ".join(command))

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
            logger.error("Hyperopt timed out after %.1fs: %s", duration, " ".join(command))
            raise HyperoptError(f"Hyperopt timed out after {timeout_seconds}s") from exc
        duration = time.monotonic() - start

        result = HyperoptResult(
            command=tuple(command),
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_seconds=duration,
        )
        if result.succeeded:
            logger.info("Hyperopt completed successfully in %.1fs", duration)
        else:
            logger.warning("Hyperopt exited with code %d after %.1fs", result.exit_code, duration)
        return result
