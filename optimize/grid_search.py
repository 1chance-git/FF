"""Grid and random parameter search: a lightweight, Freqtrade-independent alternative to hyperopt.

`optimize.hyperopt_launcher` wires up Freqtrade's own Bayesian
(Optuna-based) hyperopt engine — the right tool for a real, thorough
search over the strategy's parameter space. This module is deliberately
different and complementary: a simple, deterministic sweep (exhaustive
grid, or bounded random sampling) over an arbitrary parameter space and
an arbitrary objective function, with **no dependency on Freqtrade at
all**. Uses cases this covers that a full hyperopt run doesn't fit well:

* A quick, coarse sweep over one or two parameters where the overhead of
  spinning up Freqtrade's optimizer isn't worth it.
* Searching parameters that aren't Freqtrade `IStrategy` hyperopt
  parameters at all — e.g. sweeping `stat_arb.risk.risk.RiskEngineConfig`
  fields directly against a custom objective, which Freqtrade's
  strategy-parameter-based hyperopt has no way to express.
* Fast, fully offline testing/CI of the search *algorithm* itself
  (see this module's own test suite), independent of whether a real
  backtest can run in the current environment.

Design decisions
-----------------
* **The objective function is injected, not embedded.** Neither
  `grid_search` nor `random_search` know anything about Freqtrade,
  backtesting, or trading — they call whatever ``objective(params) ->
  float`` a caller provides. Wiring an objective that runs a real
  backtest (e.g. via `hermes.backtest.BacktestLauncher` or the
  in-process `Backtesting` technique `tests/conftest.py` establishes) is
  the caller's job. This is the same dependency-injection pattern used
  throughout this project (`stat_arb.risk.risk.HealthChecker` and
  friends) — it keeps the search algorithm itself trivially unit
  testable against a synthetic objective (e.g. a quadratic bowl with a
  known minimum) without needing any trading infrastructure at all.
* **Grid search uses `itertools.product`, random search uses the
  standard-library `random` module.** Neither the combinatorics nor the
  sampling here are numerically subtle enough to warrant a dependency —
  unlike the actual statistics elsewhere in this project (regression,
  cointegration), which do lean on `statsmodels`/`scipy` specifically
  because those computations are easy to get subtly wrong by hand.
* **Both searches return every trial's result, not just the best.**
  `SearchResult.trials` preserves the full sweep (parameters -> score,
  in evaluation order) so a caller can inspect the whole landscape —
  e.g. for `optimize.reporting` to plot or tabulate it — not just the
  single best point.
"""

from __future__ import annotations

import itertools
import logging
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: A candidate parameter combination: parameter name -> value.
ParameterCombination = dict[str, Any]

#: Callable that scores one parameter combination. Higher is better,
#: by convention (matching e.g. Sharpe ratio) — callers wanting to
#: minimize a loss should negate it before returning.
Objective = Callable[[ParameterCombination], float]


class SearchError(Exception):
    """Raised for invalid search space configuration."""


@dataclass(frozen=True)
class ParameterSpec:
    """One parameter's set of candidate values for grid search, or bounds for random search.

    Parameters
    ----------
    name:
        Parameter name (must match what the objective function expects
        as a dict key).
    values:
        Discrete candidate values, used directly by :func:`grid_search`.
        For :func:`random_search`, used only if ``low``/``high`` are not
        given (uniform choice among ``values``).
    low, high:
        Continuous bounds for :func:`random_search`'s uniform sampling.
        Ignored by :func:`grid_search`, which always uses ``values``.
    """

    name: str
    values: tuple[Any, ...] = ()
    low: float | None = None
    high: float | None = None

    def __post_init__(self) -> None:
        has_values = len(self.values) > 0
        has_bounds = self.low is not None and self.high is not None
        if not has_values and not has_bounds:
            raise SearchError(
                f"ParameterSpec {self.name!r} needs either 'values' or both 'low'/'high'"
            )
        if has_bounds and self.low >= self.high:
            raise SearchError(f"ParameterSpec {self.name!r}: low must be < high")


@dataclass(frozen=True)
class Trial:
    """One evaluated parameter combination.

    Attributes
    ----------
    params:
        The parameter combination evaluated.
    score:
        The objective function's return value for ``params``.
    """

    params: ParameterCombination
    score: float


@dataclass(frozen=True)
class SearchResult:
    """Outcome of a grid or random search.

    Attributes
    ----------
    trials:
        Every evaluated combination, in evaluation order.
    best:
        The trial with the highest score (ties broken by evaluation
        order — the first one found).
    """

    trials: tuple[Trial, ...]

    @property
    def best(self) -> Trial:
        """The highest-scoring trial."""
        return max(self.trials, key=lambda trial: trial.score)

    def sorted_by_score(self, descending: bool = True) -> tuple[Trial, ...]:
        """All trials sorted by score."""
        return tuple(sorted(self.trials, key=lambda trial: trial.score, reverse=descending))


def grid_search(
    parameters: Sequence[ParameterSpec], objective: Objective
) -> SearchResult:
    """Exhaustively evaluate every combination of ``parameters``' discrete values.

    Parameters
    ----------
    parameters:
        Parameter specs; each must provide ``values`` (a grid search
        can't use ``low``/``high`` bounds — there's no discretization
        policy that wouldn't be arbitrary, so it's rejected explicitly
        rather than guessed).
    objective:
        Scoring function, higher is better.

    Returns
    -------
    SearchResult

    Raises
    ------
    SearchError
        If ``parameters`` is empty, or any spec has no discrete
        ``values``.
    """
    if not parameters:
        raise SearchError("grid_search requires at least one ParameterSpec")
    for spec in parameters:
        if not spec.values:
            raise SearchError(
                f"grid_search requires discrete 'values' for every parameter; "
                f"{spec.name!r} only has low/high bounds"
            )

    names = [spec.name for spec in parameters]
    value_lists = [spec.values for spec in parameters]

    trials = []
    for combination in itertools.product(*value_lists):
        params = dict(zip(names, combination, strict=True))
        score = objective(params)
        trials.append(Trial(params=params, score=score))

    logger.info("Grid search complete: %d combination(s) evaluated", len(trials))
    return SearchResult(trials=tuple(trials))


def random_search(
    parameters: Sequence[ParameterSpec],
    objective: Objective,
    n_trials: int,
    seed: int | None = None,
) -> SearchResult:
    """Randomly sample ``n_trials`` parameter combinations and evaluate each.

    Parameters
    ----------
    parameters:
        Parameter specs; each may provide continuous ``low``/``high``
        bounds (sampled uniformly) or discrete ``values`` (sampled with
        equal probability).
    objective:
        Scoring function, higher is better.
    n_trials:
        Number of combinations to sample and evaluate.
    seed:
        Optional seed for reproducible sampling.

    Returns
    -------
    SearchResult

    Raises
    ------
    SearchError
        If ``parameters`` is empty or ``n_trials`` is not positive.
    """
    if not parameters:
        raise SearchError("random_search requires at least one ParameterSpec")
    if n_trials < 1:
        raise SearchError("n_trials must be >= 1")

    rng = random.Random(seed)
    trials = []
    for _ in range(n_trials):
        params: ParameterCombination = {}
        for spec in parameters:
            if spec.low is not None and spec.high is not None:
                params[spec.name] = rng.uniform(spec.low, spec.high)
            else:
                params[spec.name] = rng.choice(spec.values)
        score = objective(params)
        trials.append(Trial(params=params, score=score))

    logger.info("Random search complete: %d trial(s) evaluated (seed=%s)", len(trials), seed)
    return SearchResult(trials=tuple(trials))
