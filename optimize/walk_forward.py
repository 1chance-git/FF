"""Walk-forward testing: rolling train/test windows to validate out-of-sample robustness.

Freqtrade's `backtesting`/`hyperopt` commands operate on a single,
fixed date range — optimizing parameters on the *same* data a strategy
is then judged on is the textbook overfitting setup. Walk-forward
testing addresses this directly: repeatedly optimize on an in-sample
("train") window, then evaluate the optimized parameters on the
immediately following, never-optimized-on out-of-sample ("test")
window, then roll both windows forward and repeat. Freqtrade has no
built-in walk-forward automation, so this module provides it, built on
top of the launchers already in this package.

Design decisions
-----------------
* **Window generation is a pure function** (:func:`generate_windows`),
  independently testable with no optimizer, backtester, or dates
  touching real data — exactly the same pure-computation-vs-I/O split
  used throughout this project.
* **The optimize and evaluate steps are injected callables, not fixed
  to `optimize.hyperopt_launcher`/`hermes.backtest`.** `WalkForwardRunner`
  takes an `optimizer: Callable[[WalkForwardWindow], ParameterCombination]`
  and an `evaluator: Callable[[WalkForwardWindow, ParameterCombination],
  float]`. In production these would wrap `HyperoptLauncher` (or
  `grid_search`) and `hermes.backtest.BacktestLauncher` respectively, but
  injecting them means `WalkForwardRunner`'s orchestration logic — window
  sequencing, result aggregation, in-sample-vs-out-of-sample comparison —
  is fully unit-testable with fast synthetic stand-ins, independent of
  whether a real backtest can run in the current environment (this
  sandbox's network restriction, in particular).
* **Every window's result is kept, not just an aggregate.** A single
  averaged out-of-sample score would hide exactly the failure mode walk-
  forward testing exists to catch: a parameter set that performs great
  in some periods and terribles in others (regime-dependent overfitting)
  averages out to a mediocre-looking but misleading number if only the
  mean is reported.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta

logger = logging.getLogger(__name__)

#: A candidate parameter combination, matching optimize.grid_search's type.
ParameterCombination = dict[str, object]


class WalkForwardError(Exception):
    """Raised for invalid walk-forward configuration."""


@dataclass(frozen=True)
class WalkForwardWindow:
    """One train/test window pair.

    Attributes
    ----------
    train_start, train_end:
        In-sample (optimization) window bounds, inclusive.
    test_start, test_end:
        Out-of-sample (evaluation) window bounds, inclusive. Always
        immediately follows ``train_end`` (no gap, no overlap).
    """

    train_start: date
    train_end: date
    test_start: date
    test_end: date

    def train_timerange(self) -> str:
        """Freqtrade `--timerange`-formatted string for the train window."""
        return f"{self.train_start:%Y%m%d}-{self.train_end:%Y%m%d}"

    def test_timerange(self) -> str:
        """Freqtrade `--timerange`-formatted string for the test window."""
        return f"{self.test_start:%Y%m%d}-{self.test_end:%Y%m%d}"


def generate_windows(
    start: date,
    end: date,
    train_period: timedelta,
    test_period: timedelta,
    step: timedelta | None = None,
) -> list[WalkForwardWindow]:
    """Generate rolling train/test windows spanning ``[start, end]``.

    Each window's test period immediately follows its train period; the
    next window's train period begins ``step`` after the current one's
    (``step`` defaults to ``test_period``, giving contiguous,
    non-overlapping test windows that together cover the full range).

    Parameters
    ----------
    start, end:
        Overall date range to cover.
    train_period:
        Length of each window's in-sample period.
    test_period:
        Length of each window's out-of-sample period.
    step:
        How far to advance between windows. Defaults to ``test_period``
        (contiguous test windows). A smaller step produces overlapping
        test windows (more windows, more compute, smoother coverage); a
        larger step skips data between windows.

    Returns
    -------
    list[WalkForwardWindow]
        Windows in chronological order. Empty if ``start + train_period
        + test_period > end`` (the range is too short for even one
        window).

    Raises
    ------
    WalkForwardError
        If ``end <= start``, or ``train_period``/``test_period`` is not
        positive.
    """
    if end <= start:
        raise WalkForwardError("end must be after start")
    if train_period <= timedelta(0):
        raise WalkForwardError("train_period must be positive")
    if test_period <= timedelta(0):
        raise WalkForwardError("test_period must be positive")
    step = step if step is not None else test_period
    if step <= timedelta(0):
        raise WalkForwardError("step must be positive")

    windows = []
    train_start = start
    while True:
        train_end = train_start + train_period
        test_start = train_end
        test_end = test_start + test_period
        if test_end > end:
            break
        windows.append(
            WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        train_start = train_start + step

    logger.info(
        "Generated %d walk-forward window(s) covering %s to %s", len(windows), start, end
    )
    return windows


@dataclass(frozen=True)
class WindowResult:
    """Result of optimizing and evaluating one walk-forward window.

    Attributes
    ----------
    window:
        The window this result is for.
    params:
        The parameter combination selected by the optimizer on the
        train period.
    out_of_sample_score:
        The evaluator's score for ``params`` on the (never-optimized-on)
        test period.
    """

    window: WalkForwardWindow
    params: ParameterCombination
    out_of_sample_score: float


@dataclass(frozen=True)
class WalkForwardReport:
    """Aggregate result of a full walk-forward run.

    Attributes
    ----------
    results:
        Per-window results, in chronological order.
    """

    results: tuple[WindowResult, ...]

    @property
    def out_of_sample_scores(self) -> tuple[float, ...]:
        """Every window's out-of-sample score, in chronological order."""
        return tuple(r.out_of_sample_score for r in self.results)

    @property
    def mean_out_of_sample_score(self) -> float:
        """Mean out-of-sample score across all windows."""
        scores = self.out_of_sample_scores
        return sum(scores) / len(scores) if scores else float("nan")

    @property
    def worst_window(self) -> WindowResult:
        """The window with the lowest out-of-sample score.

        The single most important number for judging robustness — see
        module docstring on why the mean alone can mislead.
        """
        return min(self.results, key=lambda r: r.out_of_sample_score)

    @property
    def fraction_profitable_windows(self) -> float:
        """Fraction of windows with a positive out-of-sample score."""
        if not self.results:
            return float("nan")
        profitable = sum(1 for r in self.results if r.out_of_sample_score > 0)
        return profitable / len(self.results)


class WalkForwardRunner:
    """Orchestrates optimize-on-train / evaluate-on-test across a window sequence.

    Usage::

        def optimizer(window: WalkForwardWindow) -> dict:
            result = HyperoptLauncher().run(HyperoptConfig(
                ..., timerange=window.train_timerange()
            ))
            return parse_best_params(result)  # caller-defined

        def evaluator(window: WalkForwardWindow, params: dict) -> float:
            result = BacktestLauncher().run(BacktestConfig(
                ..., timerange=window.test_timerange()
            ))
            return parse_sharpe(result)  # caller-defined

        runner = WalkForwardRunner(optimizer, evaluator)
        report = runner.run(windows)
    """

    def __init__(
        self,
        optimizer: Callable[[WalkForwardWindow], ParameterCombination],
        evaluator: Callable[[WalkForwardWindow, ParameterCombination], float],
    ) -> None:
        """Initialize the runner.

        Parameters
        ----------
        optimizer:
            Given a window, returns the best parameter combination found
            on its train period (e.g. wrapping `HyperoptLauncher` or
            `optimize.grid_search.grid_search`).
        evaluator:
            Given a window and a parameter combination, returns a score
            for that combination evaluated on the window's test period
            (e.g. wrapping `hermes.backtest.BacktestLauncher`).
        """
        self.optimizer = optimizer
        self.evaluator = evaluator

    def run(self, windows: list[WalkForwardWindow]) -> WalkForwardReport:
        """Run the optimize/evaluate cycle across every window.

        Parameters
        ----------
        windows:
            Windows to process, in any order (results preserve the
            input order).

        Returns
        -------
        WalkForwardReport

        Raises
        ------
        WalkForwardError
            If ``windows`` is empty.
        """
        if not windows:
            raise WalkForwardError("windows must not be empty")

        results = []
        for i, window in enumerate(windows, start=1):
            logger.info(
                "Walk-forward window %d/%d: train=%s test=%s",
                i,
                len(windows),
                window.train_timerange(),
                window.test_timerange(),
            )
            params = self.optimizer(window)
            score = self.evaluator(window, params)
            results.append(WindowResult(window=window, params=params, out_of_sample_score=score))
            logger.info("Window %d/%d out-of-sample score: %.6f", i, len(windows), score)

        report = WalkForwardReport(results=tuple(results))
        logger.info(
            "Walk-forward complete: %d window(s), mean OOS score=%.6f, "
            "%.0f%% profitable windows",
            len(results),
            report.mean_out_of_sample_score,
            report.fraction_profitable_windows * 100,
        )
        return report
