"""Unit tests for stat_arb.signal.cointegration.

Covers: spread computation (including hedge-ratio lag correctness),
rolling mean/std/z-score correctness against pandas ground truth,
cointegration test behavior on cointegrated vs. independent series,
input validation, and — directly exercising the "prevent lookahead
bias" requirement — that no output at time *t* changes when future
observations are perturbed.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.unit

from stat_arb.signal.cointegration import (
    CointegrationConfig,
    CointegrationEngine,
    CointegrationError,
    compute_spread,
    rolling_mean,
    rolling_std,
    rolling_zscore,
    test_cointegration as run_cointegration_test,
)


def make_cointegrated_pair(
    n: int = 300, beta: float = 2.0, seed: int = 42
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Build a cointegrated (y, x) pair and a constant true hedge ratio."""
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(seed)
    x = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=idx, name="x")
    y = pd.Series(beta * x.to_numpy() + 5 + rng.normal(0, 0.5, n), index=idx, name="y")
    hedge_ratio = pd.Series(beta, index=idx, name="hedge_ratio")
    return y, x, hedge_ratio


def make_independent_pair(n: int = 300, seed: int = 42) -> tuple[pd.Series, pd.Series]:
    """Build two independent random walks (not cointegrated)."""
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(seed)
    x = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=idx, name="x")
    y = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=idx, name="y")
    return y, x


# ---------------------------------------------------------------------------
# CointegrationConfig validation
# ---------------------------------------------------------------------------


def test_config_rejects_window_too_small() -> None:
    with pytest.raises(CointegrationError, match="spread_window must be"):
        CointegrationConfig(spread_window=1)


def test_config_rejects_min_periods_above_window() -> None:
    with pytest.raises(CointegrationError, match="cannot exceed spread_window"):
        CointegrationConfig(spread_window=10, min_periods=20)


def test_config_rejects_negative_lag() -> None:
    with pytest.raises(CointegrationError, match="hedge_ratio_lag must be"):
        CointegrationConfig(hedge_ratio_lag=-1)


def test_config_rejects_bad_significance_level() -> None:
    with pytest.raises(CointegrationError, match="significance_level must be"):
        CointegrationConfig(significance_level=1.5)


def test_config_rejects_non_positive_std_floor() -> None:
    with pytest.raises(CointegrationError, match="zscore_std_floor must be positive"):
        CointegrationConfig(zscore_std_floor=0)


# ---------------------------------------------------------------------------
# compute_spread
# ---------------------------------------------------------------------------


def test_compute_spread_matches_manual_calculation() -> None:
    y, x, hedge_ratio = make_cointegrated_pair(n=50)
    spread = compute_spread(y, x, hedge_ratio, lag=1)

    expected = y - hedge_ratio.shift(1) * x
    pd.testing.assert_series_equal(spread, expected, check_names=False)
    assert spread.name == "spread"


def test_compute_spread_lag_zero_uses_contemporaneous_hedge_ratio(
    caplog: pytest.LogCaptureFixture,
) -> None:
    y, x, hedge_ratio = make_cointegrated_pair(n=20)
    with caplog.at_level(logging.WARNING, logger="stat_arb.signal.cointegration"):
        spread = compute_spread(y, x, hedge_ratio, lag=0)

    expected = y - hedge_ratio * x
    pd.testing.assert_series_equal(spread, expected, check_names=False)
    assert any("lag=0" in record.message for record in caplog.records)


def test_compute_spread_rejects_mismatched_index() -> None:
    y, x, hedge_ratio = make_cointegrated_pair(n=20)
    shifted = hedge_ratio.copy()
    shifted.index = shifted.index + pd.Timedelta(minutes=1)
    with pytest.raises(CointegrationError, match="identical index"):
        compute_spread(y, x, shifted, lag=1)


def test_compute_spread_rejects_nan_input() -> None:
    y, x, hedge_ratio = make_cointegrated_pair(n=20)
    y = y.copy()
    y.iloc[3] = np.nan
    with pytest.raises(CointegrationError, match="NaN or infinite"):
        compute_spread(y, x, hedge_ratio)


def test_compute_spread_rejects_unsorted_index() -> None:
    y, x, hedge_ratio = make_cointegrated_pair(n=20)
    reversed_y = y.iloc[::-1]
    reversed_x = x.iloc[::-1]
    reversed_hr = hedge_ratio.iloc[::-1]
    with pytest.raises(CointegrationError, match="sorted in increasing order"):
        compute_spread(reversed_y, reversed_x, reversed_hr)


def test_compute_spread_rejects_negative_lag() -> None:
    y, x, hedge_ratio = make_cointegrated_pair(n=20)
    with pytest.raises(CointegrationError, match="lag must be"):
        compute_spread(y, x, hedge_ratio, lag=-1)


# ---------------------------------------------------------------------------
# rolling_mean / rolling_std / rolling_zscore
# ---------------------------------------------------------------------------


def test_rolling_mean_matches_pandas_ground_truth() -> None:
    spread = pd.Series(np.arange(50.0))
    result = rolling_mean(spread, window=10)
    expected = spread.rolling(10, min_periods=10, center=False).mean()
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_rolling_std_matches_pandas_ground_truth() -> None:
    rng = np.random.default_rng(1)
    spread = pd.Series(rng.normal(0, 1, 50))
    result = rolling_std(spread, window=10)
    expected = spread.rolling(10, min_periods=10, center=False).std(ddof=1)
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_rolling_zscore_matches_manual_calculation() -> None:
    rng = np.random.default_rng(2)
    spread = pd.Series(rng.normal(0, 1, 50))
    z = rolling_zscore(spread, window=10)

    mean = spread.rolling(10, min_periods=10).mean()
    std = spread.rolling(10, min_periods=10).std(ddof=1)
    expected = (spread - mean) / std
    pd.testing.assert_series_equal(z, expected, check_names=False)


def test_rolling_zscore_floors_degenerate_std(caplog: pytest.LogCaptureFixture) -> None:
    spread = pd.Series([5.0] * 30)  # perfectly constant -> std == 0 everywhere
    with caplog.at_level(logging.WARNING, logger="stat_arb.signal.cointegration"):
        z = rolling_zscore(spread, window=10, std_floor=1e-8)

    assert z.iloc[9:].isna().all()
    assert any("degenerate" in record.message for record in caplog.records)


def test_rolling_zscore_warmup_is_nan() -> None:
    rng = np.random.default_rng(3)
    spread = pd.Series(rng.normal(0, 1, 30))
    z = rolling_zscore(spread, window=10)
    assert z.iloc[:9].isna().all()
    assert z.iloc[9:].notna().all()


# ---------------------------------------------------------------------------
# Lookahead bias prevention
# ---------------------------------------------------------------------------


def test_rolling_zscore_unaffected_by_future_perturbation() -> None:
    """Changing a future spread value must not change any earlier z-score."""
    rng = np.random.default_rng(4)
    spread = pd.Series(rng.normal(0, 1, 100))

    z_before = rolling_zscore(spread, window=20)

    perturbed = spread.copy()
    perturbed.iloc[-1] += 1000.0  # only the very last observation changes
    z_after = rolling_zscore(perturbed, window=20)

    pd.testing.assert_series_equal(z_before.iloc[:-1], z_after.iloc[:-1])


def test_rolling_mean_and_std_unaffected_by_future_perturbation() -> None:
    rng = np.random.default_rng(5)
    spread = pd.Series(rng.normal(0, 1, 100))

    mean_before = rolling_mean(spread, window=15)
    std_before = rolling_std(spread, window=15)

    perturbed = spread.copy()
    perturbed.iloc[50:] += 500.0  # perturb everything from the midpoint onward

    mean_after = rolling_mean(perturbed, window=15)
    std_after = rolling_std(perturbed, window=15)

    pd.testing.assert_series_equal(mean_before.iloc[:50], mean_after.iloc[:50])
    pd.testing.assert_series_equal(std_before.iloc[:50], std_after.iloc[:50])


def test_compute_spread_hedge_ratio_lag_prevents_same_bar_leakage() -> None:
    """Perturbing the hedge ratio at bar t must only affect the spread at bar t+lag."""
    y, x, hedge_ratio = make_cointegrated_pair(n=50)

    spread_before = compute_spread(y, x, hedge_ratio, lag=1)

    perturbed_hr = hedge_ratio.copy()
    perturbed_hr.iloc[30] += 100.0  # perturb only bar 30's hedge ratio estimate
    spread_after = compute_spread(y, x, perturbed_hr, lag=1)

    # Only bar 31 (which uses lagged bar 30's hedge ratio) should change.
    changed = ~np.isclose(
        spread_before.to_numpy(), spread_after.to_numpy(), equal_nan=True
    )
    assert changed.tolist() == [i == 31 for i in range(50)]


def test_zscore_reacts_only_to_current_and_past_bars_end_to_end() -> None:
    """Full pipeline: a shock at bar t must not move any z-score before bar t."""
    y, x, hedge_ratio = make_cointegrated_pair(n=150)

    config = CointegrationConfig(spread_window=20, require_cointegration=False)
    engine = CointegrationEngine(config)
    before = engine.compute(y, x, hedge_ratio).zscore

    shocked_y = y.copy()
    shocked_y.iloc[100] += 50.0
    after = engine.compute(shocked_y, x, hedge_ratio).zscore

    pd.testing.assert_series_equal(before.iloc[:100], after.iloc[:100])


# ---------------------------------------------------------------------------
# Cointegration test
# ---------------------------------------------------------------------------


def test_cointegration_detects_cointegrated_pair() -> None:
    y, x, _ = make_cointegrated_pair(n=500)
    result = run_cointegration_test(y, x, significance_level=0.05)

    assert result.is_cointegrated is True
    assert result.p_value < 0.05
    assert set(result.critical_values) == {"1%", "5%", "10%"}


def test_cointegration_rejects_independent_series() -> None:
    y, x = make_independent_pair(n=500)
    result = run_cointegration_test(y, x, significance_level=0.05)

    assert result.is_cointegrated is False
    assert result.p_value >= 0.05


def test_cointegration_rejects_bad_significance_level() -> None:
    y, x, _ = make_cointegrated_pair(n=50)
    with pytest.raises(CointegrationError, match="significance_level must be"):
        run_cointegration_test(y, x, significance_level=2.0)


# ---------------------------------------------------------------------------
# CointegrationEngine
# ---------------------------------------------------------------------------


def test_engine_computes_full_result_for_cointegrated_pair() -> None:
    y, x, hedge_ratio = make_cointegrated_pair(n=300)
    engine = CointegrationEngine(CointegrationConfig(spread_window=30))

    result = engine.compute(y, x, hedge_ratio)

    assert result.cointegration.is_cointegrated is True
    assert result.spread.index.equals(y.index)
    assert result.zscore.index.equals(y.index)
    assert result.zscore.dropna().abs().max() < 10  # sane range, no blow-up


def test_engine_raises_when_pair_not_cointegrated_by_default() -> None:
    y, x = make_independent_pair(n=500)
    hedge_ratio = pd.Series(1.0, index=y.index)
    engine = CointegrationEngine(CointegrationConfig(spread_window=30))

    with pytest.raises(CointegrationError, match="failed cointegration validation"):
        engine.compute(y, x, hedge_ratio)


def test_engine_can_skip_cointegration_gate() -> None:
    y, x = make_independent_pair(n=500)
    hedge_ratio = pd.Series(1.0, index=y.index)
    engine = CointegrationEngine(
        CointegrationConfig(spread_window=30, require_cointegration=False)
    )

    result = engine.compute(y, x, hedge_ratio)

    assert result.cointegration.is_cointegrated is False
    assert result.spread.notna().any()


def test_engine_logs_summary(caplog: pytest.LogCaptureFixture) -> None:
    y, x, hedge_ratio = make_cointegrated_pair(n=200)
    engine = CointegrationEngine(CointegrationConfig(spread_window=20))

    with caplog.at_level(logging.INFO, logger="stat_arb.signal.cointegration"):
        engine.compute(y, x, hedge_ratio)

    assert any("Spread computed" in record.message for record in caplog.records)
    assert any("Cointegration test PASSED" in record.message for record in caplog.records)
