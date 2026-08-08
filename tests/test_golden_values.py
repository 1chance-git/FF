"""Regression tests: fixed inputs, pinned ("golden") expected outputs.

Unlike the correctness tests elsewhere in the suite (which check
properties like "the estimated hedge ratio is close to the true beta"),
these tests pin the *exact* numeric output of each module for a fixed,
seeded synthetic dataset. Their purpose is different: catching
unintended behavior drift. If a future refactor changes the rolling
window's edge handling, the cointegration test's default trend term, the
risk engine's regime threshold, or any other implementation detail in a
way that shifts these numbers, one of these tests will fail — even
though the change might still produce "reasonable-looking" statistics
that a property-based correctness test wouldn't catch.

Golden values here were generated once by running each function against
the fixed-seed dataset below and recording its actual output (see the
generation commands in each test's docstring) — they are not
independently derived expected values, and are not meant to assert
statistical correctness (that's what the rest of the suite is for).
If a change to any of these values is intentional, regenerate and
update the pinned constant; don't loosen the tolerance to make an
unexplained drift pass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stat_arb.risk.risk import (
    PositionSide,
    PositionSizingConfig,
    RegimeConfig,
    TrendFilterConfig,
    calculate_position_size,
    compute_stop_loss_price,
    compute_trend_filter,
    detect_regime,
)
from stat_arb.signal.cointegration import (
    CointegrationConfig,
    CointegrationEngine,
    test_cointegration as run_cointegration_test,
)
from stat_arb.signal.regression import RollingRegressionConfig, RollingRegressionEngine

pytestmark = pytest.mark.regression


def _make_golden_dataset() -> tuple[pd.Series, pd.Series]:
    """The fixed-seed (y, x) pair every golden value below was generated from.

    y = 1.8 * x + 3 + N(0, 0.3), x a random walk — seed=123, n=200,
    5-minute bars starting 2024-01-01 UTC. Deliberately identical
    across all tests in this module so the whole pipeline (regression ->
    cointegration -> regime/trend) is exercised on one shared dataset.
    """
    idx = pd.date_range("2024-01-01", periods=200, freq="5min", tz="UTC")
    rng = np.random.default_rng(123)
    x = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)), index=idx)
    y = pd.Series(1.8 * x.to_numpy() + 3 + rng.normal(0, 0.3, 200), index=idx)
    return y, x


def test_golden_rolling_hedge_ratio() -> None:
    """Generated via:

        RollingRegressionEngine(RollingRegressionConfig(window=30)).fit(y, x).hedge_ratio
    """
    y, x = _make_golden_dataset()
    result = RollingRegressionEngine(RollingRegressionConfig(window=30)).fit(y, x)

    assert result.hedge_ratio.iloc[-1] == pytest.approx(1.846089405611601, rel=1e-9)
    assert result.hedge_ratio.iloc[50] == pytest.approx(1.7884621805306424, rel=1e-9)
    assert result.n_unstable == 29  # 30-bar warmup - 1


def test_golden_cointegration_test() -> None:
    """Generated via: test_cointegration(y, x, significance_level=0.05)."""
    y, x = _make_golden_dataset()
    result = run_cointegration_test(y, x, significance_level=0.05)

    assert result.test_statistic == pytest.approx(-13.671805065359685, rel=1e-9)
    assert result.p_value == pytest.approx(1.4131231459713937e-24, rel=1e-6, abs=1e-30)
    assert result.is_cointegrated is True


def test_golden_spread_and_zscore() -> None:
    """Generated via CointegrationEngine on the golden dataset's hedge ratio.

    Uses the rolling hedge ratio from test_golden_rolling_hedge_ratio's
    dataset (not the true beta of 1.8), matching how the assembled
    strategy actually computes the spread.
    """
    y, x = _make_golden_dataset()
    regression_result = RollingRegressionEngine(RollingRegressionConfig(window=30)).fit(y, x)
    valid_index = regression_result.hedge_ratio.dropna().index

    engine = CointegrationEngine(CointegrationConfig(spread_window=30, require_cointegration=False))
    spread_result = engine.compute(
        y.loc[valid_index], x.loc[valid_index], regression_result.hedge_ratio.loc[valid_index]
    )

    assert spread_result.zscore.iloc[-1] == pytest.approx(-1.2119459048920371, rel=1e-9)
    assert spread_result.spread.iloc[-1] == pytest.approx(-2.1604412137296833, rel=1e-9)


def test_golden_regime_and_trend_filter() -> None:
    """Generated via detect_regime / compute_trend_filter on spread = y - 1.8*x."""
    y, x = _make_golden_dataset()
    spread = y - 1.8 * x

    regime = detect_regime(spread, RegimeConfig(window=30))
    trend = compute_trend_filter(spread, TrendFilterConfig(window=20))

    assert regime.iloc[-1] == "mean_reverting"
    assert bool(trend.is_trending.iloc[-1]) is False
    assert trend.t_stat.iloc[-1] == pytest.approx(0.8153383283724742, rel=1e-9)


def test_golden_position_sizing() -> None:
    """Generated via calculate_position_size(10_000, 50_000, 47_500, PositionSizingConfig())."""
    result = calculate_position_size(10_000.0, 50_000.0, 47_500.0, PositionSizingConfig())

    assert result.units == pytest.approx(0.04)
    assert result.notional == pytest.approx(2000.0)
    assert result.risk_amount == pytest.approx(100.0)
    assert result.capped_by is None


def test_golden_stop_loss_price() -> None:
    """Generated via compute_stop_loss_price(50_000.0, PositionSide.SHORT)."""
    assert compute_stop_loss_price(50_000.0, PositionSide.SHORT) == pytest.approx(52_500.0)
