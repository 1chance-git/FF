"""Tests for TrendFollowCore: indicators + preliminary entry/invalidation only.

Mirrors the separation used by tests/test_stat_arb_swing.py and
tests/test_strategy_validation.py: pure-function unit tests against
synthetic deterministic OHLCV data (no live exchange access), plus a
handful of whole-strategy tests that load TrendFollowCore through
Freqtrade's own StrategyResolver -- the same real interface contract
any Freqtrade run would enforce.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest
from freqtrade.configuration import Configuration
from freqtrade.configuration.config_validation import validate_config_consistency
from freqtrade.resolvers import StrategyResolver
from freqtrade.strategy import IStrategy

from tests.conftest import CONFIG_PATH, STRATEGIES_PATH, make_ohlcv

pytestmark = pytest.mark.strategy

STRATEGY_PATH = STRATEGIES_PATH / "TrendFollowCore.py"


def _load_strategy_module():
    spec = importlib.util.spec_from_file_location("TrendFollowCore", STRATEGY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_module = _load_strategy_module()
ADX_THRESHOLD = _module.ADX_THRESHOLD
DONCHIAN_PERIOD = _module.DONCHIAN_PERIOD
EMA_PERIOD = _module.EMA_PERIOD
TrendFollowCore = _module.TrendFollowCore
compute_entry_signals = _module.compute_entry_signals
compute_exit_signals = _module.compute_exit_signals
compute_indicators = _module.compute_indicators


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _trending_ohlcv(n: int = 300, *, drift: float = 0.3, noise: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """A steadily uptrending series with a Donchian breakout near the end.

    ``drift`` per bar keeps ADX comfortably above 25 (strong, persistent
    trend) once past warmup, and a final sharp jump guarantees at least
    one bar closes above its own prior-20-candle Donchian upper channel.
    """
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(np.full(n, drift) + rng.normal(0, noise, n))
    # Force an unambiguous breakout on the final bar: jump well above
    # every close in the preceding DONCHIAN_PERIOD bars.
    close[-1] = close[-DONCHIAN_PERIOD - 1 : -1].max() + 10.0
    return make_ohlcv(close, idx)


def _flat_ohlcv(n: int = 60, *, price: float = 100.0) -> pd.DataFrame:
    """Perfectly flat price -- ADX/Donchian breakouts should never fire."""
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    return make_ohlcv(np.full(n, price), idx)


def _ramp_pullback_then_breakout(*, direction: int = 1, plateau_len: int = 4) -> pd.DataFrame:
    """A long, steady ramp (satisfies EMA-200/ADX warmup and keeps ADX high),
    followed by a short flat plateau (no new Donchian extreme -> no breakout),
    followed by one final decisive bar that breaks through the plateau's own
    extreme -- the one scenario that isolates "trend + strong ADX but no
    breakout yet" from "trend + strong ADX + breakout", which a purely
    monotonic ramp can't do (every bar of a monotonic ramp is itself a new
    20-bar extreme, so it can never produce a genuine non-breakout bar).

    ``direction=1`` builds the uptrend/long-breakout scenario;
    ``direction=-1`` builds the exact mirror for the short scenario.
    """
    n = EMA_PERIOD + 60
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    ramp_len = n - plateau_len - 1
    ramp = 100 + direction * np.cumsum(np.full(ramp_len, 0.3))
    plateau = np.full(plateau_len, ramp[-1])
    breakout = ramp[-1] + direction * 10.0
    close = np.concatenate([ramp, plateau, [breakout]])
    return make_ohlcv(close, idx)


# ---------------------------------------------------------------------------
# 1-3: indicators exist
# ---------------------------------------------------------------------------


class TestIndicatorsExist:
    def test_ema200_column_exists_and_is_computed(self):
        df = compute_indicators(_trending_ohlcv())
        assert "ema200" in df.columns
        assert df["ema200"].notna().any()

    def test_adx14_column_exists_and_is_computed(self):
        df = compute_indicators(_trending_ohlcv())
        assert "adx" in df.columns
        assert df["adx"].notna().any()

    def test_donchian_upper_and_lower_columns_exist_and_are_computed(self):
        df = compute_indicators(_trending_ohlcv())
        assert "donchian_upper_prev" in df.columns
        assert "donchian_lower_prev" in df.columns
        assert df["donchian_upper_prev"].notna().any()
        assert df["donchian_lower_prev"].notna().any()

    def test_ema200_matches_talib_directly(self):
        import talib

        raw = _trending_ohlcv()
        df = compute_indicators(raw)
        expected = talib.EMA(raw["close"], timeperiod=EMA_PERIOD)
        pd.testing.assert_series_equal(df["ema200"], expected, check_names=False)


# ---------------------------------------------------------------------------
# 4: Donchian excludes the current candle
# ---------------------------------------------------------------------------


class TestDonchianExcludesCurrentCandle:
    def test_upper_channel_ignores_a_current_candle_spike(self):
        """A huge high on the CURRENT candle must not raise its own threshold."""
        n = 40
        idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
        close = np.full(n, 100.0)
        df = make_ohlcv(close, idx)
        spike_row = n - 1
        df.loc[spike_row, "high"] = 100_000.0  # current-candle spike

        out = compute_indicators(df)
        # If the spike leaked into its own threshold, donchian_upper_prev
        # at spike_row would be ~100_000; it must instead reflect only the
        # flat prior candles (~100 + the make_ohlcv high padding).
        assert out.loc[spike_row, "donchian_upper_prev"] < 200.0

    def test_upper_channel_at_row_n_equals_max_high_of_prior_window_only(self):
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
        rng = np.random.default_rng(1)
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        df = make_ohlcv(close, idx)

        out = compute_indicators(df)
        row = 35
        expected_upper = df["high"].iloc[row - DONCHIAN_PERIOD : row].max()
        assert out.loc[row, "donchian_upper_prev"] == pytest.approx(expected_upper)
        # And explicitly NOT equal to a window that includes the current row.
        including_current = df["high"].iloc[row - DONCHIAN_PERIOD + 1 : row + 1].max()
        if including_current != expected_upper:
            assert out.loc[row, "donchian_upper_prev"] != pytest.approx(including_current)

    def test_lower_channel_at_row_n_equals_min_low_of_prior_window_only(self):
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
        rng = np.random.default_rng(2)
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        df = make_ohlcv(close, idx)

        out = compute_indicators(df)
        row = 35
        expected_lower = df["low"].iloc[row - DONCHIAN_PERIOD : row].min()
        assert out.loc[row, "donchian_lower_prev"] == pytest.approx(expected_lower)


# ---------------------------------------------------------------------------
# 5-6: long/short signals require all three conditions
# ---------------------------------------------------------------------------


class TestEntrySignalConditions:
    def test_long_fires_on_engineered_breakout(self):
        df = compute_indicators(_trending_ohlcv())
        df = compute_entry_signals(df)
        assert df["enter_long"].iloc[-1] == 1
        assert df.loc[df.index[-1], "enter_tag"] == "trend_long_donchian_breakout"

    def test_short_fires_on_engineered_breakdown(self):
        raw = _ramp_pullback_then_breakout(direction=-1)
        df = compute_indicators(raw)
        df = compute_entry_signals(df)
        assert df["enter_short"].iloc[-1] == 1
        assert df.loc[df.index[-1], "enter_tag"] == "trend_short_donchian_breakout"

    def test_no_long_signal_without_donchian_breakout(self):
        """Uptrend (close > ema200, adx > 25) but no breakout on the plateau
        bar just before the engineered final breakout bar."""
        df = compute_indicators(_ramp_pullback_then_breakout(direction=1))
        df = compute_entry_signals(df)
        pre_breakout = df.iloc[-2]  # plateau bar: no new 20-bar high
        assert pre_breakout["close"] > pre_breakout["ema200"]
        assert pre_breakout["adx"] > ADX_THRESHOLD
        assert df["enter_long"].iloc[-2] == 0
        assert df["enter_long"].iloc[-1] == 1  # sanity: the next bar does break out

    def test_no_long_signal_without_adx_above_threshold(self):
        """Flat market: close hovers around ema200-ish and ADX stays low."""
        df = compute_indicators(_flat_ohlcv(n=260))
        df = compute_entry_signals(df)
        assert (df["adx"].dropna() <= ADX_THRESHOLD).all()
        assert df["enter_long"].sum() == 0
        assert df["enter_short"].sum() == 0

    def test_no_long_signal_when_close_below_ema200_even_with_breakout(self):
        """Manually force a Donchian breakout while close < ema200 -- must not fire long."""
        raw = _trending_ohlcv(drift=-0.3)  # downtrend: close ends up < ema200
        df = compute_indicators(raw)
        # Force a spurious "breakout above upper channel" on the last row
        # while the trend/EMA condition still says downtrend.
        df.loc[df.index[-1], "donchian_upper_prev"] = df["close"].iloc[-1] - 1.0
        df = compute_entry_signals(df)
        assert df["close"].iloc[-1] < df["ema200"].iloc[-1]
        assert df["enter_long"].iloc[-1] == 0

    def test_long_and_short_are_mutually_exclusive_per_row(self):
        df = compute_indicators(_trending_ohlcv())
        df = compute_entry_signals(df)
        assert not ((df["enter_long"] == 1) & (df["enter_short"] == 1)).any()


# ---------------------------------------------------------------------------
# 7: NaN/startup periods produce no signals
# ---------------------------------------------------------------------------


class TestStartupProducesNoSignals:
    def test_no_entry_signals_during_ema_warmup(self):
        df = compute_indicators(_trending_ohlcv(n=300))
        df = compute_entry_signals(df)
        warmup = df.iloc[: EMA_PERIOD - 1]
        assert warmup["ema200"].isna().all()
        assert warmup["enter_long"].sum() == 0
        assert warmup["enter_short"].sum() == 0

    def test_no_exit_signals_during_ema_warmup(self):
        df = compute_indicators(_trending_ohlcv(n=300))
        df = compute_exit_signals(df)
        warmup = df.iloc[: EMA_PERIOD - 1]
        assert warmup["exit_long"].sum() == 0
        assert warmup["exit_short"].sum() == 0

    def test_too_short_dataframe_produces_no_signals_anywhere(self):
        """Fewer rows than any indicator's warmup: everything should be NaN-safe."""
        df = compute_indicators(_trending_ohlcv(n=50))
        entry = compute_entry_signals(df)
        exit_ = compute_exit_signals(df)
        assert entry["enter_long"].sum() == 0
        assert entry["enter_short"].sum() == 0
        assert exit_["exit_long"].sum() == 0
        assert exit_["exit_short"].sum() == 0


# ---------------------------------------------------------------------------
# 8: no-lookahead
# ---------------------------------------------------------------------------


class TestNoLookahead:
    def test_indicators_at_row_n_are_unchanged_by_future_rows(self):
        """Truncating the dataframe after row N must not change any indicator
        value already computed at or before row N -- the standard no-lookahead
        contract (see test_strategy_validation.py's equivalent for StatArbSwing)."""
        full = _trending_ohlcv(n=280)
        cutoff = 260

        full_indicators = compute_indicators(full)
        truncated_indicators = compute_indicators(full.iloc[: cutoff + 1])

        cols = ["ema200", "adx", "donchian_upper_prev", "donchian_lower_prev"]
        pd.testing.assert_frame_equal(
            full_indicators.iloc[: cutoff + 1][cols].reset_index(drop=True),
            truncated_indicators[cols].reset_index(drop=True),
        )

    def test_entry_signal_at_row_n_unchanged_by_future_rows(self):
        full = _trending_ohlcv(n=280)
        cutoff = 260

        full_signals = compute_entry_signals(compute_indicators(full))
        truncated_signals = compute_entry_signals(compute_indicators(full.iloc[: cutoff + 1]))

        cols = ["enter_long", "enter_short"]
        pd.testing.assert_frame_equal(
            full_signals.iloc[: cutoff + 1][cols].reset_index(drop=True),
            truncated_signals[cols].reset_index(drop=True),
        )

    def test_donchian_threshold_never_includes_the_signal_candle_itself(self):
        """Direct check: donchian_upper_prev at row N must be identical whether
        or not row N's own high/low is set to an extreme value -- proving the
        threshold was never a function of row N's own candle at all."""
        df = _trending_ohlcv(n=100)
        baseline = compute_indicators(df)

        mutated = df.copy()
        row = 80
        mutated.loc[row, "high"] = 999_999.0
        mutated.loc[row, "low"] = -999_999.0
        mutated_out = compute_indicators(mutated)

        assert mutated_out.loc[row, "donchian_upper_prev"] == pytest.approx(
            baseline.loc[row, "donchian_upper_prev"]
        )
        assert mutated_out.loc[row, "donchian_lower_prev"] == pytest.approx(
            baseline.loc[row, "donchian_lower_prev"]
        )


# ---------------------------------------------------------------------------
# 9-10: strategy loads through Freqtrade; futures long/short config is valid
# ---------------------------------------------------------------------------


def _load_config() -> dict:
    args = {
        "config": [str(CONFIG_PATH)],
        "strategy": "TrendFollowCore",
        "strategy_path": str(STRATEGIES_PATH),
        "timeframe": "4h",
    }
    return Configuration(args, "backtest").get_config()


@pytest.fixture(scope="module")
def resolved_strategy() -> IStrategy:
    return StrategyResolver.load_strategy(_load_config())


class TestStrategyLoadsThroughFreqtrade:
    def test_strategy_resolves_to_the_expected_class(self, resolved_strategy: IStrategy) -> None:
        assert type(resolved_strategy).__name__ == "TrendFollowCore"
        assert isinstance(resolved_strategy, IStrategy)

    def test_config_and_strategy_pass_freqtrade_consistency_validation(self) -> None:
        config = _load_config()
        StrategyResolver.load_strategy(config)
        validate_config_consistency(config)  # must not raise

    def test_required_hooks_are_implemented(self, resolved_strategy: IStrategy) -> None:
        for hook in ("populate_indicators", "populate_entry_trend", "populate_exit_trend"):
            assert callable(getattr(resolved_strategy, hook, None)), hook

    def test_interface_version_is_current(self, resolved_strategy: IStrategy) -> None:
        assert resolved_strategy.INTERFACE_VERSION == 3

    def test_resolved_timeframe_is_4h(self, resolved_strategy: IStrategy) -> None:
        assert resolved_strategy.timeframe == "4h"

    def test_startup_candle_count_exceeds_ema200_warmup(self, resolved_strategy: IStrategy) -> None:
        assert resolved_strategy.startup_candle_count >= EMA_PERIOD

    def test_stoploss_is_a_negative_5_percent(self, resolved_strategy: IStrategy) -> None:
        assert resolved_strategy.stoploss == pytest.approx(-0.05)


class TestFuturesLongShortConfiguration:
    def test_can_short_is_enabled(self, resolved_strategy: IStrategy) -> None:
        assert resolved_strategy.can_short is True

    def test_full_pipeline_produces_valid_columns_for_futures_mode(self, resolved_strategy: IStrategy) -> None:
        """Run the assembled hook pipeline (as Freqtrade would call it) and
        confirm both long and short entry/exit columns exist -- required for
        can_short=True futures operation."""
        raw = _trending_ohlcv(n=280)
        indicators_df = resolved_strategy.populate_indicators(raw.copy(), {"pair": "BTC/USDC:USDC"})
        entry_df = resolved_strategy.populate_entry_trend(indicators_df.copy(), {"pair": "BTC/USDC:USDC"})
        full_df = resolved_strategy.populate_exit_trend(entry_df, {"pair": "BTC/USDC:USDC"})

        for col in ("enter_long", "enter_short", "exit_long", "exit_short"):
            assert col in full_df.columns
            assert full_df[col].isin([0, 1]).all()
