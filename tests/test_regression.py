"""Unit tests for stat_arb.signal.regression.

Covers: correctness of the rolling hedge ratio against a known synthetic
beta, numerical-stability detection (condition number and hedge-ratio
magnitude), forward-fill behavior for unstable windows, input validation,
hedge-ratio-change logging, and a cross-check of the analytical rolling
condition number against ``numpy.linalg.cond``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.unit

from stat_arb.signal.regression import (
    RegressionError,
    RollingRegressionConfig,
    RollingRegressionEngine,
    RollingRegressionResult,
    _rolling_condition_number,
)


def make_series(
    n: int = 200,
    start: str = "2024-01-01",
    freq: str = "5min",
    seed: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """Build a synthetic (y, x) pair with a known true hedge ratio of 2.5."""
    idx = pd.date_range(start=start, periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(seed)
    x = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=idx, name="x")
    noise = rng.normal(0, 0.25, n)
    y = pd.Series(2.5 * x.to_numpy() + 10 + noise, index=idx, name="y")
    return y, x


# ---------------------------------------------------------------------------
# RollingRegressionConfig validation
# ---------------------------------------------------------------------------


def test_config_rejects_window_too_small() -> None:
    with pytest.raises(RegressionError, match="window must be"):
        RollingRegressionConfig(window=2)


def test_config_rejects_min_nobs_below_minimum() -> None:
    with pytest.raises(RegressionError, match="min_nobs must be"):
        RollingRegressionConfig(window=10, min_nobs=2)


def test_config_rejects_min_nobs_above_window() -> None:
    with pytest.raises(RegressionError, match="cannot exceed window"):
        RollingRegressionConfig(window=10, min_nobs=20)


def test_config_defaults_min_nobs_to_window() -> None:
    config = RollingRegressionConfig(window=30)
    assert config._resolved_min_nobs == 30


def test_config_rejects_non_positive_thresholds() -> None:
    with pytest.raises(RegressionError, match="condition_number_threshold"):
        RollingRegressionConfig(condition_number_threshold=0)
    with pytest.raises(RegressionError, match="max_abs_hedge_ratio"):
        RollingRegressionConfig(max_abs_hedge_ratio=-1)
    with pytest.raises(RegressionError, match="hedge_ratio_change_log_threshold"):
        RollingRegressionConfig(hedge_ratio_change_log_threshold=-0.1)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_fit_rejects_non_series_input() -> None:
    engine = RollingRegressionEngine(RollingRegressionConfig(window=10))
    with pytest.raises(RegressionError, match="pandas Series"):
        engine.fit(y=[1, 2, 3], x=[1, 2, 3])  # type: ignore[arg-type]


def test_fit_rejects_mismatched_length() -> None:
    y, x = make_series(n=50)
    engine = RollingRegressionEngine(RollingRegressionConfig(window=10))
    with pytest.raises(RegressionError, match="equal length"):
        engine.fit(y=y, x=x.iloc[:-1])


def test_fit_rejects_misaligned_index() -> None:
    y, x = make_series(n=50)
    x_shifted = x.copy()
    x_shifted.index = x_shifted.index + pd.Timedelta(minutes=1)
    engine = RollingRegressionEngine(RollingRegressionConfig(window=10))
    with pytest.raises(RegressionError, match="identical index"):
        engine.fit(y=y, x=x_shifted)


def test_fit_rejects_missing_values() -> None:
    y, x = make_series(n=50)
    y = y.copy()
    y.iloc[5] = np.nan
    engine = RollingRegressionEngine(RollingRegressionConfig(window=10))
    with pytest.raises(RegressionError, match="missing values"):
        engine.fit(y=y, x=x)


def test_fit_rejects_non_numeric_input() -> None:
    y, x = make_series(n=10)
    y = y.astype(str)
    engine = RollingRegressionEngine(RollingRegressionConfig(window=5))
    with pytest.raises(RegressionError, match="numeric"):
        engine.fit(y=y, x=x)


def test_fit_rejects_series_shorter_than_window() -> None:
    y, x = make_series(n=5)
    engine = RollingRegressionEngine(RollingRegressionConfig(window=10))
    with pytest.raises(RegressionError, match="Need at least 10"):
        engine.fit(y=y, x=x)


def test_fit_rejects_empty_series() -> None:
    y = pd.Series([], dtype=float)
    x = pd.Series([], dtype=float)
    engine = RollingRegressionEngine(RollingRegressionConfig(window=10))
    with pytest.raises(RegressionError, match="must not be empty"):
        engine.fit(y=y, x=x)


# ---------------------------------------------------------------------------
# Correctness of the dynamic hedge ratio
# ---------------------------------------------------------------------------


def test_rolling_hedge_ratio_recovers_true_beta() -> None:
    y, x = make_series(n=300)
    engine = RollingRegressionEngine(RollingRegressionConfig(window=40))

    result = engine.fit(y, x)

    assert isinstance(result, RollingRegressionResult)
    tail = result.hedge_ratio.dropna().tail(50)
    assert (tail - 2.5).abs().max() < 0.1


def test_result_series_share_input_index() -> None:
    y, x = make_series(n=100)
    engine = RollingRegressionEngine(RollingRegressionConfig(window=20))
    result = engine.fit(y, x)

    for series in (
        result.hedge_ratio,
        result.raw_hedge_ratio,
        result.intercept,
        result.r_squared,
        result.condition_number,
        result.is_stable,
    ):
        assert series.index.equals(y.index)


def test_warmup_period_has_no_raw_estimate() -> None:
    y, x = make_series(n=100)
    engine = RollingRegressionEngine(RollingRegressionConfig(window=20))
    result = engine.fit(y, x)

    assert result.raw_hedge_ratio.iloc[: 20 - 1].isna().all()
    assert result.raw_hedge_ratio.iloc[19:].notna().all()


# ---------------------------------------------------------------------------
# Numerical stability detection
# ---------------------------------------------------------------------------


def test_constant_x_window_is_flagged_unstable_and_ffilled() -> None:
    y, x = make_series(n=100)
    x = x.copy()
    y = y.copy()
    # Force a fully flat region wider than the window -> singular design matrix.
    x.iloc[40:70] = 150.0
    y.iloc[40:70] = 2.5 * 150.0 + 10

    engine = RollingRegressionEngine(RollingRegressionConfig(window=20))
    result = engine.fit(y, x)

    flat_window_position = 65  # fully inside the flat region given window=20
    assert not result.is_stable.iloc[flat_window_position]
    assert np.isinf(result.condition_number.iloc[flat_window_position])
    # Forward-filled from the last stable estimate rather than NaN.
    assert not np.isnan(result.hedge_ratio.iloc[flat_window_position])


def test_ffill_unstable_false_leaves_nan() -> None:
    y, x = make_series(n=100)
    x = x.copy()
    y = y.copy()
    x.iloc[40:70] = 150.0
    y.iloc[40:70] = 2.5 * 150.0 + 10

    engine = RollingRegressionEngine(
        RollingRegressionConfig(window=20, ffill_unstable=False)
    )
    result = engine.fit(y, x)

    assert result.hedge_ratio.iloc[65:70].isna().all()


def test_max_abs_hedge_ratio_flags_extreme_beta() -> None:
    idx = pd.date_range("2024-01-01", periods=60, freq="5min", tz="UTC")
    rng = np.random.default_rng(3)
    # x has tiny variance relative to y's swings -> huge, unstable beta.
    x = pd.Series(100 + rng.normal(0, 1e-6, 60), index=idx)
    y = pd.Series(rng.normal(0, 100, 60), index=idx)

    engine = RollingRegressionEngine(
        RollingRegressionConfig(window=15, max_abs_hedge_ratio=10.0)
    )
    result = engine.fit(y, x)

    assert result.n_unstable > 0
    unstable_and_estimated = ~result.is_stable & result.raw_hedge_ratio.notna()
    assert (result.raw_hedge_ratio[unstable_and_estimated].abs() > 10.0).any()


def test_rolling_condition_number_matches_numpy_cond() -> None:
    rng = np.random.default_rng(7)
    x = pd.Series(rng.normal(100, 10, 50))
    config = RollingRegressionConfig(window=10)

    condition_numbers = _rolling_condition_number(x, config)

    window_values = x.iloc[20:30].to_numpy()
    design_matrix = np.column_stack([np.ones(10), window_values])
    expected = np.linalg.cond(design_matrix)

    assert condition_numbers.iloc[29] == pytest.approx(expected)


def test_condition_number_is_inf_for_singular_window() -> None:
    idx = pd.RangeIndex(20)
    x = pd.Series([5.0] * 20, index=idx)
    config = RollingRegressionConfig(window=10)

    condition_numbers = _rolling_condition_number(x, config)

    assert np.isinf(condition_numbers.iloc[15])


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def test_logs_significant_hedge_ratio_changes(caplog: pytest.LogCaptureFixture) -> None:
    idx = pd.date_range("2024-01-01", periods=120, freq="5min", tz="UTC")
    rng = np.random.default_rng(11)
    x = pd.Series(100 + np.cumsum(rng.normal(0, 0.1, 120)), index=idx)

    # Regime change halfway through: beta jumps from 1.0 to 3.0.
    beta = np.where(np.arange(120) < 60, 1.0, 3.0)
    y = pd.Series(beta * x.to_numpy() + rng.normal(0, 0.01, 120), index=idx)

    engine = RollingRegressionEngine(
        RollingRegressionConfig(window=20, hedge_ratio_change_log_threshold=0.05)
    )

    with caplog.at_level(logging.INFO, logger="stat_arb.signal.regression"):
        engine.fit(y, x)

    assert any("Hedge ratio change at" in record.message for record in caplog.records)


def test_logs_unstable_window_warning(caplog: pytest.LogCaptureFixture) -> None:
    y, x = make_series(n=100)
    x = x.copy()
    y = y.copy()
    x.iloc[40:70] = 150.0
    y.iloc[40:70] = 2.5 * 150.0 + 10

    engine = RollingRegressionEngine(RollingRegressionConfig(window=20))

    with caplog.at_level(logging.WARNING, logger="stat_arb.signal.regression"):
        engine.fit(y, x)

    assert any("flagged numerically unstable" in record.message for record in caplog.records)


def test_no_change_log_below_threshold(caplog: pytest.LogCaptureFixture) -> None:
    y, x = make_series(n=100, seed=99)
    engine = RollingRegressionEngine(
        RollingRegressionConfig(window=20, hedge_ratio_change_log_threshold=0.99)
    )

    with caplog.at_level(logging.INFO, logger="stat_arb.signal.regression"):
        engine.fit(y, x)

    assert not any("Hedge ratio change at" in record.message for record in caplog.records)
