"""Unit tests for optimize.grid_search.

Uses a synthetic quadratic-bowl objective with a known analytic minimum
(x=3, y=1) so the search algorithms' correctness can be verified
directly, independent of any trading/backtesting infrastructure.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from optimize.grid_search import (
    ParameterSpec,
    SearchError,
    grid_search,
    random_search,
)


def bowl_objective(params: dict) -> float:
    """Negative squared distance from (3, 1); maximized at exactly that point."""
    return -((params["x"] - 3) ** 2 + (params["y"] - 1) ** 2)


# ---------------------------------------------------------------------------
# ParameterSpec validation
# ---------------------------------------------------------------------------


def test_parameter_spec_requires_values_or_bounds() -> None:
    with pytest.raises(SearchError, match="needs either"):
        ParameterSpec("x")


def test_parameter_spec_rejects_invalid_bounds() -> None:
    with pytest.raises(SearchError, match="low must be < high"):
        ParameterSpec("x", low=5, high=1)


def test_parameter_spec_accepts_values() -> None:
    spec = ParameterSpec("x", values=(1, 2, 3))
    assert spec.values == (1, 2, 3)


def test_parameter_spec_accepts_bounds() -> None:
    spec = ParameterSpec("x", low=0.0, high=1.0)
    assert spec.low == 0.0 and spec.high == 1.0


# ---------------------------------------------------------------------------
# grid_search
# ---------------------------------------------------------------------------


def test_grid_search_finds_the_exact_optimum_on_the_grid() -> None:
    specs = [
        ParameterSpec("x", values=(0, 1, 2, 3, 4, 5)),
        ParameterSpec("y", values=(-1, 0, 1, 2)),
    ]
    result = grid_search(specs, bowl_objective)

    assert result.best.params == {"x": 3, "y": 1}
    assert result.best.score == 0


def test_grid_search_evaluates_every_combination() -> None:
    specs = [ParameterSpec("x", values=(1, 2, 3)), ParameterSpec("y", values=(10, 20))]
    result = grid_search(specs, bowl_objective)
    assert len(result.trials) == 3 * 2


def test_grid_search_rejects_empty_parameters() -> None:
    with pytest.raises(SearchError, match="at least one"):
        grid_search([], bowl_objective)


def test_grid_search_rejects_bounds_only_spec() -> None:
    specs = [ParameterSpec("x", low=0.0, high=1.0)]
    with pytest.raises(SearchError, match="discrete 'values'"):
        grid_search(specs, bowl_objective)


def test_grid_search_sorted_by_score() -> None:
    specs = [ParameterSpec("x", values=(0, 3, 6)), ParameterSpec("y", values=(1,))]
    result = grid_search(specs, bowl_objective)
    sorted_trials = result.sorted_by_score()
    scores = [t.score for t in sorted_trials]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# random_search
# ---------------------------------------------------------------------------


def test_random_search_converges_near_the_optimum() -> None:
    specs = [ParameterSpec("x", low=-5.0, high=10.0), ParameterSpec("y", low=-5.0, high=5.0)]
    result = random_search(specs, bowl_objective, n_trials=500, seed=1)

    assert result.best.params["x"] == pytest.approx(3.0, abs=0.5)
    assert result.best.params["y"] == pytest.approx(1.0, abs=0.5)


def test_random_search_is_deterministic_given_a_seed() -> None:
    specs = [ParameterSpec("x", low=-5.0, high=10.0), ParameterSpec("y", low=-5.0, high=5.0)]
    result_a = random_search(specs, bowl_objective, n_trials=50, seed=7)
    result_b = random_search(specs, bowl_objective, n_trials=50, seed=7)

    assert [t.params for t in result_a.trials] == [t.params for t in result_b.trials]
    assert [t.score for t in result_a.trials] == [t.score for t in result_b.trials]


def test_random_search_different_seeds_differ() -> None:
    specs = [ParameterSpec("x", low=-5.0, high=10.0)]
    result_a = random_search(specs, lambda p: p["x"], n_trials=20, seed=1)
    result_b = random_search(specs, lambda p: p["x"], n_trials=20, seed=2)

    assert [t.params for t in result_a.trials] != [t.params for t in result_b.trials]


def test_random_search_evaluates_exactly_n_trials() -> None:
    specs = [ParameterSpec("x", low=0.0, high=1.0)]
    result = random_search(specs, lambda p: p["x"], n_trials=17, seed=0)
    assert len(result.trials) == 17


def test_random_search_supports_discrete_values() -> None:
    specs = [ParameterSpec("x", values=("a", "b", "c"))]
    result = random_search(specs, lambda p: {"a": 1, "b": 2, "c": 3}[p["x"]], n_trials=30, seed=0)
    assert all(t.params["x"] in ("a", "b", "c") for t in result.trials)


def test_random_search_rejects_empty_parameters() -> None:
    with pytest.raises(SearchError, match="at least one"):
        random_search([], bowl_objective, n_trials=10)


def test_random_search_rejects_non_positive_n_trials() -> None:
    specs = [ParameterSpec("x", low=0.0, high=1.0)]
    with pytest.raises(SearchError, match="n_trials must be"):
        random_search(specs, bowl_objective, n_trials=0)


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


def test_search_result_best_breaks_ties_by_evaluation_order() -> None:
    specs = [ParameterSpec("x", values=(1, 2))]
    # Both x=1 and x=2 score identically -> `best` should be the first evaluated.
    result = grid_search(specs, lambda p: 0.0)
    assert result.best.params == {"x": 1}
