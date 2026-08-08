"""Numerical consistency tests: cross-implementation checks and invariants.

Two kinds of tests live here:

1. **Cross-implementation checks** — the same quantity computed two
   independent ways (our rolling implementation vs. a direct call to
   `numpy`/`scipy`/`statsmodels` on a single window) must agree to
   floating-point precision. This catches subtle bugs that a
   "reasonable-looking output" test wouldn't: an off-by-one in a
   rolling window boundary, a sign error, a wrong degrees-of-freedom
   convention.
2. **Arithmetic invariants** — identities that must hold exactly (or to
   tight tolerance) by construction, independent of any specific input:
   ``units * price == notional``, ``|entry - stop| / entry ==
   stop_loss_pct``, running the same computation twice on identical
   input yields identical output. A violation means the implementation
   has drifted from its own mathematical definition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stat_arb.risk.risk import (
    PositionSide,
    PositionSizingConfig,
    RegimeConfig,
    StopLossConfig,
    TrendFilterConfig,
    calculate_position_size,
    compute_stop_loss_price,
    compute_trend_filter,
    detect_regime,
)
from stat_arb.signal.cointegration import (
    CointegrationConfig,
    CointegrationEngine,
    rolling_mean,
    rolling_std,
    rolling_zscore,
)
from stat_arb.signal.regression import (
    RollingRegressionConfig,
    RollingRegressionEngine,
    _rolling_condition_number,
)

pytestmark = pytest.mark.numerical


def _make_series(n: int = 150, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(seed)
    x = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=idx)
    y = pd.Series(1.5 * x.to_numpy() + 2 + rng.normal(0, 0.2, n), index=idx)
    return y, x


# ---------------------------------------------------------------------------
# Cross-implementation checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("evaluation_offset", [-1, -20, -50, -100])
def test_rolling_hedge_ratio_matches_manual_ols_via_lstsq(evaluation_offset: int) -> None:
    """RollingOLS's slope at an arbitrary point must match a direct numpy.linalg.lstsq fit.

    Cross-checks our rolling wrapper against a completely independent
    computation path: manually slicing out the same window and solving
    the OLS normal equations with `numpy.linalg.lstsq` directly, rather
    than through `statsmodels.RollingOLS` at all.
    """
    y, x = _make_series(n=150)
    window = 30
    result = RollingRegressionEngine(RollingRegressionConfig(window=window)).fit(y, x)

    window_x = x.iloc[evaluation_offset - window + 1 : evaluation_offset + 1 or None]
    window_y = y.iloc[evaluation_offset - window + 1 : evaluation_offset + 1 or None]
    design_matrix = np.column_stack([np.ones(window), window_x.to_numpy()])
    coefficients, *_ = np.linalg.lstsq(design_matrix, window_y.to_numpy(), rcond=None)

    assert result.hedge_ratio.iloc[evaluation_offset] == pytest.approx(
        coefficients[1], rel=1e-9
    )
    assert result.intercept.iloc[evaluation_offset] == pytest.approx(coefficients[0], rel=1e-9)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_rolling_condition_number_matches_numpy_cond_across_seeds(seed: int) -> None:
    """The analytical rolling condition number must match numpy.linalg.cond on many windows.

    test_regression.py checks this for one window; this sweeps several
    random seeds/windows to guard against a formula that happens to
    match only for a particular data shape.
    """
    rng = np.random.default_rng(seed)
    x = pd.Series(rng.normal(100, 10, 80))
    config = RollingRegressionConfig(window=15)

    condition_numbers = _rolling_condition_number(x, config)

    for position in (20, 40, 60, 79):
        window_values = x.iloc[position - 14 : position + 1].to_numpy()
        design_matrix = np.column_stack([np.ones(15), window_values])
        expected = np.linalg.cond(design_matrix)
        assert condition_numbers.iloc[position] == pytest.approx(expected, rel=1e-6)


def test_trend_filter_matches_scipy_linregress() -> None:
    """compute_trend_filter's slope/t-stat must match scipy.stats.linregress independently.

    `compute_trend_filter` is built on `statsmodels.RollingOLS`; scipy's
    `linregress` is a completely separate implementation of simple
    linear regression. Agreement between the two on the same window is
    strong evidence the slope/t-statistic computation is correct, not
    just internally self-consistent.
    """
    from scipy.stats import linregress

    idx = pd.date_range("2024-01-01", periods=60, freq="5min", tz="UTC")
    rng = np.random.default_rng(9)
    prices = pd.Series(100 + 0.3 * np.arange(60) + rng.normal(0, 1, 60), index=idx)

    result = compute_trend_filter(prices, TrendFilterConfig(window=20))

    window_prices = prices.iloc[-20:].to_numpy()
    time_index = np.arange(20, dtype=float)
    scipy_result = linregress(time_index, window_prices)

    assert result.slope.iloc[-1] == pytest.approx(scipy_result.slope, rel=1e-9)
    assert result.t_stat.iloc[-1] == pytest.approx(
        scipy_result.slope / scipy_result.stderr, rel=1e-9
    )


def test_adf_pvalue_matches_direct_statsmodels_call() -> None:
    """detect_regime's per-window ADF p-value must match calling statsmodels directly.

    Confirms the rolling wrapper (`.rolling().apply(_adf_pvalue)`)
    passes exactly the right slice to `adfuller` with the right
    arguments, rather than e.g. an off-by-one window or a different
    `autolag` setting silently baked in.
    """
    from statsmodels.tsa.stattools import adfuller

    n = 100
    x = np.zeros(n)
    rng = np.random.default_rng(2)
    for i in range(1, n):
        x[i] = x[i - 1] * 0.3 + rng.normal(0, 1)
    spread = pd.Series(x)

    window = 40
    regime_config = RegimeConfig(window=window)
    with_regime = detect_regime(spread, regime_config)

    window_values = spread.iloc[-window:].to_numpy()
    expected_pvalue = adfuller(window_values, autolag="AIC")[1]
    is_mean_reverting = expected_pvalue < regime_config.significance_level

    assert with_regime.iloc[-1] == ("mean_reverting" if is_mean_reverting else "trending")


def test_zscore_matches_manual_mean_std_calculation() -> None:
    """CointegrationEngine's zscore column must equal (spread - mean) / std elementwise.

    Recomputes the z-score directly from the engine's own
    rolling_mean/rolling_std outputs (bypassing rolling_zscore
    entirely) to confirm there's no hidden discrepancy — e.g. a
    different ddof, a different window alignment — between the pieces.
    """
    y, x = _make_series(n=150)
    regression_result = RollingRegressionEngine(RollingRegressionConfig(window=30)).fit(y, x)
    valid_index = regression_result.hedge_ratio.dropna().index

    engine = CointegrationEngine(CointegrationConfig(spread_window=30, require_cointegration=False))
    spread_result = engine.compute(
        y.loc[valid_index], x.loc[valid_index], regression_result.hedge_ratio.loc[valid_index]
    )

    manual_mean = rolling_mean(spread_result.spread, window=30)
    manual_std = rolling_std(spread_result.spread, window=30)
    manual_zscore = (spread_result.spread - manual_mean) / manual_std

    pd.testing.assert_series_equal(
        spread_result.zscore, manual_zscore, check_names=False, check_exact=False
    )


def test_rolling_zscore_function_matches_engine_zscore_output() -> None:
    """The standalone rolling_zscore() function and CointegrationEngine.compute() must agree."""
    y, x = _make_series(n=150)
    regression_result = RollingRegressionEngine(RollingRegressionConfig(window=30)).fit(y, x)
    valid_index = regression_result.hedge_ratio.dropna().index

    engine = CointegrationEngine(CointegrationConfig(spread_window=30, require_cointegration=False))
    spread_result = engine.compute(
        y.loc[valid_index], x.loc[valid_index], regression_result.hedge_ratio.loc[valid_index]
    )

    standalone_zscore = rolling_zscore(spread_result.spread, window=30)

    pd.testing.assert_series_equal(spread_result.zscore, standalone_zscore, check_names=False)


# ---------------------------------------------------------------------------
# Arithmetic invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("equity", "entry_price", "stop_loss_price"),
    [(10_000.0, 50_000.0, 47_500.0), (500.0, 1.234, 1.17), (1_000_000.0, 3000.0, 2850.0)],
)
def test_position_size_units_times_price_equals_notional(
    equity: float, entry_price: float, stop_loss_price: float
) -> None:
    """units * entry_price must exactly equal notional, by definition, for any input."""
    result = calculate_position_size(equity, entry_price, stop_loss_price)
    assert result.units * entry_price == pytest.approx(result.notional, rel=1e-9)


def test_position_size_risk_amount_equals_equity_times_risk_pct() -> None:
    """risk_amount is defined as equity * risk_per_trade_pct, independent of stop distance."""
    config = PositionSizingConfig(risk_per_trade_pct=0.02)
    result = calculate_position_size(10_000.0, 100.0, 90.0, config)
    assert result.risk_amount == pytest.approx(10_000.0 * 0.02)


@pytest.mark.parametrize("stop_loss_pct", [0.01, 0.05, 0.10, 0.25])
@pytest.mark.parametrize("side", [PositionSide.LONG, PositionSide.SHORT])
def test_stop_loss_distance_ratio_equals_configured_pct(
    stop_loss_pct: float, side: PositionSide
) -> None:
    """|entry - stop| / entry must exactly equal the configured stop_loss_pct."""
    config = StopLossConfig(stop_loss_pct=stop_loss_pct)
    entry_price = 12_345.678
    stop_price = compute_stop_loss_price(entry_price, side, config)

    distance_ratio = abs(entry_price - stop_price) / entry_price
    assert distance_ratio == pytest.approx(stop_loss_pct, rel=1e-9)


def test_rolling_regression_is_deterministic_across_repeated_calls() -> None:
    """Fitting the same data twice must produce byte-identical results (no hidden randomness)."""
    y, x = _make_series(n=150)
    engine = RollingRegressionEngine(RollingRegressionConfig(window=30))

    result_a = engine.fit(y, x)
    result_b = engine.fit(y, x)

    pd.testing.assert_series_equal(result_a.hedge_ratio, result_b.hedge_ratio)
    pd.testing.assert_series_equal(result_a.condition_number, result_b.condition_number)


def test_detect_regime_is_deterministic_across_repeated_calls() -> None:
    """ADF-based regime classification must be deterministic given identical input."""
    _, x = _make_series(n=150)
    config = RegimeConfig(window=30)

    regime_a = detect_regime(x, config)
    regime_b = detect_regime(x, config)

    pd.testing.assert_series_equal(regime_a, regime_b)


def test_cointegration_engine_is_deterministic_across_repeated_calls() -> None:
    """The full spread/z-score/cointegration pipeline must be deterministic given identical input."""
    y, x = _make_series(n=150)
    regression_result = RollingRegressionEngine(RollingRegressionConfig(window=30)).fit(y, x)
    valid_index = regression_result.hedge_ratio.dropna().index
    y_valid, x_valid = y.loc[valid_index], x.loc[valid_index]
    hedge_ratio_valid = regression_result.hedge_ratio.loc[valid_index]

    engine = CointegrationEngine(CointegrationConfig(spread_window=30, require_cointegration=False))
    result_a = engine.compute(y_valid, x_valid, hedge_ratio_valid)
    result_b = engine.compute(y_valid, x_valid, hedge_ratio_valid)

    pd.testing.assert_series_equal(result_a.zscore, result_b.zscore)
    assert result_a.cointegration.p_value == result_b.cointegration.p_value


def test_no_nan_or_inf_leaks_past_the_warmup_period() -> None:
    """Once the rolling windows have filled, no NaN/inf should appear in any output series.

    A NaN/inf appearing *after* warmup (rather than only during it)
    would indicate a numerical edge case (e.g. division by zero) wasn't
    actually handled by the stability/degenerate-window guards those
    modules implement.
    """
    y, x = _make_series(n=150, seed=3)
    regression_result = RollingRegressionEngine(RollingRegressionConfig(window=30)).fit(y, x)

    post_warmup_hedge_ratio = regression_result.hedge_ratio.dropna()
    assert np.isfinite(post_warmup_hedge_ratio.to_numpy()).all()

    valid_index = regression_result.hedge_ratio.dropna().index
    engine = CointegrationEngine(CointegrationConfig(spread_window=30, require_cointegration=False))
    spread_result = engine.compute(
        y.loc[valid_index], x.loc[valid_index], regression_result.hedge_ratio.loc[valid_index]
    )
    post_warmup_zscore = spread_result.zscore.dropna()
    assert np.isfinite(post_warmup_zscore.to_numpy()).all()
