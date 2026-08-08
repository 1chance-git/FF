"""Unit tests for stat_arb.risk.risk.

Covers each independent risk control (stop loss, position sizing, regime
detection, trend filter, exposure limits, cooldown logic), input
validation, and the composed `RiskEngine.evaluate_entry` orchestration.
Also verifies the module has zero dependency on freqtrade.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from stat_arb.risk.risk import (
    CooldownConfig,
    CooldownTracker,
    ExposureConfig,
    MarketRegime,
    PositionSide,
    PositionSizingConfig,
    RegimeConfig,
    RiskEngine,
    RiskEngineConfig,
    RiskError,
    StopLossConfig,
    TrendFilterConfig,
    calculate_position_size,
    check_exposure_limits,
    compute_stop_loss_price,
    compute_trend_filter,
    detect_regime,
    is_stop_loss_triggered,
)

UTC_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


def make_mean_reverting_series(n: int = 150, seed: int = 0) -> pd.Series:
    """Ornstein-Uhlenbeck-like mean-reverting series (strongly stationary)."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i - 1] * 0.3 + rng.normal(0, 1)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.Series(x + 100, index=idx)


def make_trending_series(n: int = 150, seed: int = 0) -> pd.Series:
    """Random walk with strong drift (trending, not mean-reverting)."""
    rng = np.random.default_rng(seed)
    drift = np.cumsum(np.full(n, 0.8)) + rng.normal(0, 0.3, n).cumsum() * 0.1
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.Series(100 + drift, index=idx)


# ---------------------------------------------------------------------------
# No Freqtrade dependency
# ---------------------------------------------------------------------------


def test_risk_module_does_not_import_freqtrade() -> None:
    """Parse the module's AST and assert no `import freqtrade...` statement exists.

    A plain substring search would false-positive on the module's own
    docstrings, which discuss the freqtrade-independence design decision
    in prose. Parsing actual import statements is the precise check.
    """
    import ast

    import stat_arb.risk.risk as risk_module

    with open(risk_module.__file__, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())

    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    assert not any(m == "freqtrade" or m.startswith("freqtrade.") for m in imported_modules)


# ---------------------------------------------------------------------------
# Stop loss
# ---------------------------------------------------------------------------


def test_stop_loss_price_long() -> None:
    assert compute_stop_loss_price(100.0, PositionSide.LONG) == pytest.approx(95.0)


def test_stop_loss_price_short() -> None:
    assert compute_stop_loss_price(100.0, PositionSide.SHORT) == pytest.approx(105.0)


def test_stop_loss_price_custom_pct() -> None:
    config = StopLossConfig(stop_loss_pct=0.10)
    assert compute_stop_loss_price(100.0, PositionSide.LONG, config) == pytest.approx(90.0)


def test_stop_loss_rejects_non_positive_entry_price() -> None:
    with pytest.raises(RiskError, match="entry_price must be positive"):
        compute_stop_loss_price(0.0, PositionSide.LONG)


def test_stop_loss_config_rejects_invalid_pct() -> None:
    with pytest.raises(RiskError, match="stop_loss_pct"):
        StopLossConfig(stop_loss_pct=1.5)
    with pytest.raises(RiskError, match="stop_loss_pct"):
        StopLossConfig(stop_loss_pct=0)


def test_is_stop_loss_triggered_long() -> None:
    assert is_stop_loss_triggered(94.0, 100.0, PositionSide.LONG) is True
    assert is_stop_loss_triggered(96.0, 100.0, PositionSide.LONG) is False
    assert is_stop_loss_triggered(95.0, 100.0, PositionSide.LONG) is True  # exactly at stop


def test_is_stop_loss_triggered_short() -> None:
    assert is_stop_loss_triggered(106.0, 100.0, PositionSide.SHORT) is True
    assert is_stop_loss_triggered(104.0, 100.0, PositionSide.SHORT) is False


def test_is_stop_loss_triggered_logs(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="stat_arb.risk.risk"):
        is_stop_loss_triggered(94.0, 100.0, PositionSide.LONG)
    assert any("Stop loss triggered" in r.message for r in caplog.records)


def test_stop_loss_rejects_non_positive_current_price() -> None:
    with pytest.raises(RiskError, match="current_price must be positive"):
        is_stop_loss_triggered(0.0, 100.0, PositionSide.LONG)


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------


def test_position_size_basic_fixed_fractional() -> None:
    # risk 1% of 10,000 = 100; stop distance = 5 -> 20 units, notional 2000
    result = calculate_position_size(10_000.0, 100.0, 95.0)
    assert result.units == pytest.approx(20.0)
    assert result.notional == pytest.approx(2000.0)
    assert result.risk_amount == pytest.approx(100.0)
    assert result.capped_by is None


def test_position_size_capped_by_max_position_pct() -> None:
    # Very tight stop -> huge raw size, must be capped.
    config = PositionSizingConfig(
        risk_per_trade_pct=0.5, max_position_pct_of_equity=0.10, max_leverage=100.0
    )
    result = calculate_position_size(10_000.0, 100.0, 99.99, config)
    assert result.capped_by == "max_position_pct_of_equity"
    assert result.notional == pytest.approx(1000.0)


def test_position_size_capped_by_max_leverage() -> None:
    config = PositionSizingConfig(
        risk_per_trade_pct=0.5, max_position_pct_of_equity=100.0, max_leverage=2.0
    )
    result = calculate_position_size(10_000.0, 100.0, 99.99, config)
    assert result.capped_by == "max_leverage"
    assert result.notional == pytest.approx(20_000.0)


def test_position_size_rejects_zero_stop_distance() -> None:
    with pytest.raises(RiskError, match="differ from entry_price"):
        calculate_position_size(10_000.0, 100.0, 100.0)


def test_position_size_rejects_non_positive_equity() -> None:
    with pytest.raises(RiskError, match="equity must be positive"):
        calculate_position_size(0.0, 100.0, 95.0)


def test_position_sizing_config_validation() -> None:
    with pytest.raises(RiskError, match="risk_per_trade_pct"):
        PositionSizingConfig(risk_per_trade_pct=1.5)
    with pytest.raises(RiskError, match="max_position_pct_of_equity"):
        PositionSizingConfig(max_position_pct_of_equity=0)
    with pytest.raises(RiskError, match="max_leverage"):
        PositionSizingConfig(max_leverage=-1)


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------


def test_detect_regime_flags_mean_reverting_series() -> None:
    series = make_mean_reverting_series(n=200)
    regime = detect_regime(series, RegimeConfig(window=40))

    tail = regime.dropna().tail(30)
    assert (tail == MarketRegime.MEAN_REVERTING.value).mean() > 0.8


def test_detect_regime_flags_trending_series() -> None:
    series = make_trending_series(n=200)
    regime = detect_regime(series, RegimeConfig(window=40))

    tail = regime.dropna().tail(30)
    assert (tail == MarketRegime.TRENDING.value).mean() > 0.8


def test_detect_regime_warmup_is_unknown() -> None:
    series = make_mean_reverting_series(n=100)
    regime = detect_regime(series, RegimeConfig(window=40))
    assert (regime.iloc[: 40 - 1] == MarketRegime.UNKNOWN.value).all()


def test_detect_regime_handles_constant_window() -> None:
    idx = pd.date_range("2024-01-01", periods=60, freq="5min", tz="UTC")
    series = pd.Series([100.0] * 60, index=idx)
    regime = detect_regime(series, RegimeConfig(window=20))
    # Constant windows can't run ADF -> UNKNOWN, not an exception.
    assert (regime.iloc[19:] == MarketRegime.UNKNOWN.value).all()


def test_detect_regime_rejects_short_series() -> None:
    series = make_mean_reverting_series(n=10)
    with pytest.raises(RiskError, match="Need at least"):
        detect_regime(series, RegimeConfig(window=40))


def test_regime_config_validation() -> None:
    with pytest.raises(RiskError, match="window must be"):
        RegimeConfig(window=5)
    with pytest.raises(RiskError, match="significance_level"):
        RegimeConfig(significance_level=2.0)


# ---------------------------------------------------------------------------
# Trend filter
# ---------------------------------------------------------------------------


def test_trend_filter_flags_strong_trend() -> None:
    idx = pd.date_range("2024-01-01", periods=100, freq="5min", tz="UTC")
    rng = np.random.default_rng(1)
    prices = pd.Series(100 + 0.5 * np.arange(100) + rng.normal(0, 0.5, 100), index=idx)

    result = compute_trend_filter(prices, TrendFilterConfig(window=20))

    assert result.is_trending.iloc[-1] is True or bool(result.is_trending.iloc[-1]) is True
    assert result.direction.iloc[-1] == 1


def test_trend_filter_does_not_flag_flat_series() -> None:
    idx = pd.date_range("2024-01-01", periods=100, freq="5min", tz="UTC")
    rng = np.random.default_rng(2)
    prices = pd.Series(100 + rng.normal(0, 1, 100), index=idx)

    result = compute_trend_filter(prices, TrendFilterConfig(window=20))

    assert result.is_trending.tail(30).mean() < 0.2


def test_trend_filter_direction_sign() -> None:
    idx = pd.date_range("2024-01-01", periods=100, freq="5min", tz="UTC")
    prices_down = pd.Series(200 - 0.5 * np.arange(100), index=idx)

    result = compute_trend_filter(prices_down, TrendFilterConfig(window=20))
    assert result.direction.iloc[-1] == -1


def test_trend_filter_result_shares_index() -> None:
    series = make_mean_reverting_series(n=80)
    result = compute_trend_filter(series, TrendFilterConfig(window=20))
    for s in (result.slope, result.t_stat, result.is_trending, result.direction):
        assert s.index.equals(series.index)


def test_trend_filter_config_validation() -> None:
    with pytest.raises(RiskError, match="window must be"):
        TrendFilterConfig(window=1)
    with pytest.raises(RiskError, match="t_stat_threshold"):
        TrendFilterConfig(t_stat_threshold=0)


def test_price_series_validation_shared_by_regime_and_trend() -> None:
    idx = pd.date_range("2024-01-01", periods=20, freq="5min", tz="UTC")
    bad = pd.Series([1.0] * 20, index=idx[::-1])  # unsorted
    with pytest.raises(RiskError, match="sorted"):
        detect_regime(bad, RegimeConfig(window=10))
    with pytest.raises(RiskError, match="sorted"):
        compute_trend_filter(bad, TrendFilterConfig(window=10))


# ---------------------------------------------------------------------------
# Maximum exposure
# ---------------------------------------------------------------------------


def test_exposure_allows_within_limits() -> None:
    config = ExposureConfig(max_gross_exposure_pct=0.5, max_pair_exposure_pct=0.3, max_positions=2)
    result = check_exposure_limits(10_000.0, {}, "BTC/USDC:USDC", 2000.0, config)
    assert result.allowed is True
    assert result.reasons == ()


def test_exposure_blocks_max_positions() -> None:
    config = ExposureConfig(max_positions=1)
    current = {"BTC/USDC:USDC": 1000.0}
    result = check_exposure_limits(10_000.0, current, "ETH/USDC:USDC", 500.0, config)
    assert result.allowed is False
    assert any("max_positions" in r for r in result.reasons)


def test_exposure_allows_adding_to_existing_pair_beyond_position_count() -> None:
    config = ExposureConfig(max_positions=1)
    current = {"BTC/USDC:USDC": 1000.0}
    # Adding more to the SAME pair shouldn't trip max_positions.
    result = check_exposure_limits(10_000.0, current, "BTC/USDC:USDC", 500.0, config)
    assert not any("max_positions" in r for r in result.reasons)


def test_exposure_blocks_gross_limit() -> None:
    config = ExposureConfig(max_gross_exposure_pct=0.2, max_pair_exposure_pct=1.0, max_positions=5)
    current = {"BTC/USDC:USDC": 1900.0}
    result = check_exposure_limits(10_000.0, current, "ETH/USDC:USDC", 200.0, config)
    assert result.allowed is False
    assert any("gross exposure" in r for r in result.reasons)


def test_exposure_blocks_per_pair_limit() -> None:
    config = ExposureConfig(max_gross_exposure_pct=1.0, max_pair_exposure_pct=0.1, max_positions=5)
    result = check_exposure_limits(10_000.0, {}, "BTC/USDC:USDC", 2000.0, config)
    assert result.allowed is False
    assert any("BTC/USDC:USDC exposure" in r for r in result.reasons)


def test_exposure_rejects_non_positive_equity() -> None:
    with pytest.raises(RiskError, match="equity must be positive"):
        check_exposure_limits(0.0, {}, "BTC/USDC:USDC", 100.0)


def test_exposure_config_validation() -> None:
    with pytest.raises(RiskError, match="max_gross_exposure_pct"):
        ExposureConfig(max_gross_exposure_pct=0)
    with pytest.raises(RiskError, match="max_positions"):
        ExposureConfig(max_positions=0)


# ---------------------------------------------------------------------------
# Cooldown logic
# ---------------------------------------------------------------------------


def test_cooldown_not_active_initially() -> None:
    tracker = CooldownTracker(CooldownConfig(cooldown_period=timedelta(hours=4)))
    assert tracker.is_in_cooldown("BTC/USDC:USDC", UTC_NOW) is False


def test_cooldown_active_after_stop_loss_exit() -> None:
    tracker = CooldownTracker(CooldownConfig(cooldown_period=timedelta(hours=4)))
    tracker.record_exit("BTC/USDC:USDC", UTC_NOW, stop_loss_triggered=True)

    assert tracker.is_in_cooldown("BTC/USDC:USDC", UTC_NOW + timedelta(hours=1)) is True
    assert tracker.is_in_cooldown("BTC/USDC:USDC", UTC_NOW + timedelta(hours=5)) is False


def test_cooldown_not_armed_by_non_stop_loss_exit_by_default() -> None:
    tracker = CooldownTracker(CooldownConfig(cooldown_period=timedelta(hours=4)))
    tracker.record_exit("BTC/USDC:USDC", UTC_NOW, stop_loss_triggered=False)
    assert tracker.is_in_cooldown("BTC/USDC:USDC", UTC_NOW + timedelta(minutes=1)) is False


def test_cooldown_apply_to_all_exits() -> None:
    tracker = CooldownTracker(
        CooldownConfig(cooldown_period=timedelta(hours=4), apply_to_all_exits=True)
    )
    tracker.record_exit("BTC/USDC:USDC", UTC_NOW, stop_loss_triggered=False)
    assert tracker.is_in_cooldown("BTC/USDC:USDC", UTC_NOW + timedelta(minutes=1)) is True


def test_cooldown_is_per_pair() -> None:
    tracker = CooldownTracker(CooldownConfig(cooldown_period=timedelta(hours=4)))
    tracker.record_exit("BTC/USDC:USDC", UTC_NOW, stop_loss_triggered=True)
    assert tracker.is_in_cooldown("ETH/USDC:USDC", UTC_NOW) is False


def test_cooldown_remaining_time() -> None:
    tracker = CooldownTracker(CooldownConfig(cooldown_period=timedelta(hours=4)))
    tracker.record_exit("BTC/USDC:USDC", UTC_NOW, stop_loss_triggered=True)
    remaining = tracker.remaining_cooldown("BTC/USDC:USDC", UTC_NOW + timedelta(hours=1))
    assert remaining == timedelta(hours=3)
    assert tracker.remaining_cooldown("ETH/USDC:USDC", UTC_NOW) == timedelta(0)


def test_cooldown_reset() -> None:
    tracker = CooldownTracker(CooldownConfig(cooldown_period=timedelta(hours=4)))
    tracker.record_exit("BTC/USDC:USDC", UTC_NOW, stop_loss_triggered=True)
    tracker.reset("BTC/USDC:USDC")
    assert tracker.is_in_cooldown("BTC/USDC:USDC", UTC_NOW) is False


def test_cooldown_config_rejects_non_positive_period() -> None:
    with pytest.raises(RiskError, match="cooldown_period must be positive"):
        CooldownConfig(cooldown_period=timedelta(0))


# ---------------------------------------------------------------------------
# RiskEngine orchestration
# ---------------------------------------------------------------------------


def test_engine_allows_entry_when_all_checks_pass() -> None:
    series = make_mean_reverting_series(n=200)
    engine = RiskEngine(
        RiskEngineConfig(
            regime=RegimeConfig(window=40), trend_filter=TrendFilterConfig(window=20)
        )
    )

    decision = engine.evaluate_entry(
        pair="BTC/USDC:USDC",
        side=PositionSide.LONG,
        entry_price=float(series.iloc[-1]),
        equity=10_000.0,
        prices=series,
        current_positions={},
        current_time=UTC_NOW,
    )

    assert decision.allowed is True
    assert decision.reasons == ()
    assert decision.position_size is not None
    assert decision.regime is MarketRegime.MEAN_REVERTING


def test_engine_blocks_entry_when_trending() -> None:
    series = make_trending_series(n=200)
    engine = RiskEngine(
        RiskEngineConfig(
            regime=RegimeConfig(window=40), trend_filter=TrendFilterConfig(window=20)
        )
    )

    decision = engine.evaluate_entry(
        pair="BTC/USDC:USDC",
        side=PositionSide.LONG,
        entry_price=float(series.iloc[-1]),
        equity=10_000.0,
        prices=series,
        current_positions={},
        current_time=UTC_NOW,
    )

    assert decision.allowed is False
    assert any("trend" in r or "regime" in r for r in decision.reasons)


def test_engine_blocks_entry_during_cooldown() -> None:
    series = make_mean_reverting_series(n=200)
    engine = RiskEngine(
        RiskEngineConfig(
            regime=RegimeConfig(window=40),
            trend_filter=TrendFilterConfig(window=20),
            cooldown=CooldownConfig(cooldown_period=timedelta(hours=4)),
        )
    )
    engine.record_exit("BTC/USDC:USDC", UTC_NOW, stop_loss_triggered=True)

    decision = engine.evaluate_entry(
        pair="BTC/USDC:USDC",
        side=PositionSide.LONG,
        entry_price=float(series.iloc[-1]),
        equity=10_000.0,
        prices=series,
        current_positions={},
        current_time=UTC_NOW + timedelta(minutes=30),
    )

    assert decision.allowed is False
    assert any("cooldown" in r for r in decision.reasons)


def test_engine_blocks_entry_exceeding_exposure() -> None:
    series = make_mean_reverting_series(n=200)
    engine = RiskEngine(
        RiskEngineConfig(
            regime=RegimeConfig(window=40),
            trend_filter=TrendFilterConfig(window=20),
            exposure=ExposureConfig(max_positions=1),
        )
    )

    decision = engine.evaluate_entry(
        pair="ETH/USDC:USDC",
        side=PositionSide.LONG,
        entry_price=float(series.iloc[-1]),
        equity=10_000.0,
        prices=series,
        current_positions={"BTC/USDC:USDC": 1000.0},
        current_time=UTC_NOW,
    )

    assert decision.allowed is False
    assert any("max_positions" in r for r in decision.reasons)


def test_engine_check_stop_loss_delegates_to_config() -> None:
    engine = RiskEngine(RiskEngineConfig(stop_loss=StopLossConfig(stop_loss_pct=0.05)))
    assert engine.check_stop_loss(94.0, 100.0, PositionSide.LONG) is True
    assert engine.check_stop_loss(96.0, 100.0, PositionSide.LONG) is False


def test_engine_cooldown_expires() -> None:
    series = make_mean_reverting_series(n=200)
    engine = RiskEngine(
        RiskEngineConfig(
            regime=RegimeConfig(window=40),
            trend_filter=TrendFilterConfig(window=20),
            cooldown=CooldownConfig(cooldown_period=timedelta(hours=1)),
        )
    )
    engine.record_exit("BTC/USDC:USDC", UTC_NOW, stop_loss_triggered=True)

    decision = engine.evaluate_entry(
        pair="BTC/USDC:USDC",
        side=PositionSide.LONG,
        entry_price=float(series.iloc[-1]),
        equity=10_000.0,
        prices=series,
        current_positions={},
        current_time=UTC_NOW + timedelta(hours=2),
    )

    assert not any("cooldown" in r for r in decision.reasons)
