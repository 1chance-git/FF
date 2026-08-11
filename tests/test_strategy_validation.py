"""Strategy validation: StatArbSwing against Freqtrade's own interface contract.

Distinct from `tests/test_stat_arb_swing.py` (which unit-tests the
strategy's pure helper functions and exercises individual hooks in
isolation): this module validates the *assembled* strategy object
against Freqtrade's own configuration/strategy consistency checks, and
validates properties that only make sense at the whole-strategy level —
e.g. that no-lookahead holds through the full populate_indicators ->
populate_entry_trend -> populate_exit_trend pipeline, not just within
each individual `stat_arb` module that feeds it.
"""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from freqtrade.configuration import Configuration
from freqtrade.configuration.config_validation import validate_config_consistency
from freqtrade.resolvers import StrategyResolver
from freqtrade.strategy import IStrategy

from tests.conftest import (
    CONFIG_PATH,
    STRATEGIES_PATH,
    X_PAIR,
    Y_PAIR,
    make_oscillating_cointegrated_pair,
)

pytestmark = pytest.mark.strategy

STRATEGY_FILE = STRATEGIES_PATH / "StatArbSwing.py"


def _load_config() -> dict:
    args = {
        "config": [str(CONFIG_PATH)],
        "strategy": "StatArbSwing",
        "strategy_path": str(STRATEGIES_PATH),
    }
    return Configuration(args, "backtest").get_config()


@pytest.fixture(scope="module")
def resolved_strategy() -> IStrategy:
    """The strategy as Freqtrade's own resolver would load it for a real run."""
    return StrategyResolver.load_strategy(_load_config())


# ---------------------------------------------------------------------------
# Freqtrade's own consistency validation
# ---------------------------------------------------------------------------


def test_config_and_strategy_pass_freqtrade_consistency_validation() -> None:
    """`validate_config_consistency` is the same check Freqtrade runs before every start.

    Catches structural mistakes real to Freqtrade's interface contract
    that unit tests of our own pure functions can't see: conflicting
    ROI/stoploss/trailing-stop settings, invalid order_types
    combinations, futures/margin-mode mismatches, etc.
    """
    config = _load_config()
    StrategyResolver.load_strategy(config)  # populates config["strategy"] object, as real runs do
    validate_config_consistency(config)  # must not raise


def test_strategy_resolves_to_the_expected_class(resolved_strategy: IStrategy) -> None:
    assert type(resolved_strategy).__name__ == "StatArbSwing"
    assert isinstance(resolved_strategy, IStrategy)


# ---------------------------------------------------------------------------
# IStrategy interface conformance
# ---------------------------------------------------------------------------


def test_required_hooks_are_implemented(resolved_strategy: IStrategy) -> None:
    for hook in ("populate_indicators", "populate_entry_trend", "populate_exit_trend"):
        assert callable(getattr(resolved_strategy, hook, None)), hook


def test_futures_short_support_is_enabled(resolved_strategy: IStrategy) -> None:
    """can_short is mandatory for a futures pairs-trading strategy that shorts a leg."""
    assert resolved_strategy.can_short is True


def test_interface_version_is_current(resolved_strategy: IStrategy) -> None:
    assert resolved_strategy.INTERFACE_VERSION == 3


def test_stoploss_is_a_negative_fraction(resolved_strategy: IStrategy) -> None:
    assert resolved_strategy.stoploss < 0
    assert resolved_strategy.stoploss > -1
    assert resolved_strategy.stoploss == pytest.approx(-resolved_strategy.STOP_LOSS_PCT)


def test_minimal_roi_has_the_required_zero_key(resolved_strategy: IStrategy) -> None:
    # StrategyResolver normalizes minimal_roi's keys from str -> int minutes
    # after loading (for its internal sorted-lookup table), so the resolved
    # strategy has an int 0 key, not the "0" string written in the class body.
    assert 0 in resolved_strategy.minimal_roi


def test_startup_candle_count_covers_internal_windows(resolved_strategy: IStrategy) -> None:
    """The bot must not start trading before indicators have real (non-NaN) values.

    The spread/z-score only becomes available after the regression
    window fills *and* the spread rolling window fills on top of that;
    startup_candle_count must be at least their sum, or early candles
    would have indicators=NaN and no signal could ever fire on them.
    """
    required = resolved_strategy.REGRESSION_WINDOW + resolved_strategy.SPREAD_WINDOW
    assert resolved_strategy.startup_candle_count >= required


def test_stop_loss_and_risk_engine_share_one_constant(resolved_strategy: IStrategy) -> None:
    """The base `stoploss` attribute and the risk engine's StopLossConfig must never drift apart."""
    assert (
        resolved_strategy.risk_engine.config.stop_loss.stop_loss_pct
        == resolved_strategy.STOP_LOSS_PCT
    )


# ---------------------------------------------------------------------------
# No-lookahead at the whole-strategy (assembled pipeline) level
# ---------------------------------------------------------------------------


def _load_strategy_module():
    spec = importlib.util.spec_from_file_location("StatArbSwing", STRATEGY_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_full_pipeline(strategy, y_ohlcv: pd.DataFrame, x_ohlcv: pd.DataFrame) -> pd.DataFrame:
    strategy.dp = SimpleNamespace(get_pair_dataframe=lambda pair, timeframe: x_ohlcv)
    indicators_df = strategy.populate_indicators(y_ohlcv.copy(), {"pair": Y_PAIR})
    entry_df = strategy.populate_entry_trend(indicators_df.copy(), {"pair": Y_PAIR})
    full_df = strategy.populate_exit_trend(entry_df, {"pair": Y_PAIR})
    return full_df


def test_full_pipeline_has_no_lookahead_bias() -> None:
    """Perturbing only the *last* candles must not change any earlier signal.

    Runs the real, assembled populate_indicators -> populate_entry_trend
    -> populate_exit_trend pipeline twice — once on the baseline
    synthetic data, once with a large shock applied only to the final 5
    candles — and asserts every row before the shock produces byte-
    identical enter_long/enter_short/exit_long/exit_short/zscore values.
    This is the whole-pipeline counterpart to the per-module lookahead
    tests in test_regression.py / test_cointegration.py: it confirms
    composing those modules inside the strategy didn't reintroduce
    lookahead at the seams (e.g. via the market-data alignment step).
    """
    sas = _load_strategy_module()
    y_ohlcv, x_ohlcv = make_oscillating_cointegrated_pair(n=400)

    strategy_baseline = sas.StatArbSwing({"stake_currency": "USDC", "runmode": None, "user_data_dir": tempfile.mkdtemp(prefix="hermes_test_")})
    baseline = _run_full_pipeline(strategy_baseline, y_ohlcv, x_ohlcv)

    shocked_y = y_ohlcv.copy()
    shocked_y.loc[shocked_y.index[-5:], "close"] += 500.0
    shocked_y.loc[shocked_y.index[-5:], "high"] += 500.0
    shocked_y.loc[shocked_y.index[-5:], "low"] += 500.0
    shocked_y.loc[shocked_y.index[-5:], "open"] += 500.0

    strategy_shocked = sas.StatArbSwing({"stake_currency": "USDC", "runmode": None, "user_data_dir": tempfile.mkdtemp(prefix="hermes_test_")})
    shocked = _run_full_pipeline(strategy_shocked, shocked_y, x_ohlcv)

    unaffected_columns = ["zscore", "enter_long", "enter_short", "exit_long", "exit_short"]
    cutoff = -5
    for column in unaffected_columns:
        pd.testing.assert_series_equal(
            baseline[column].iloc[:cutoff].reset_index(drop=True),
            shocked[column].iloc[:cutoff].reset_index(drop=True),
            check_names=False,
        )


def test_full_pipeline_never_enters_both_directions_on_the_same_candle() -> None:
    """enter_long and enter_short must be mutually exclusive per candle.

    Note this deliberately does *not* assert the same for exit_long/
    exit_short: those legitimately overlap whenever the z-score sits
    within the exit band of the mean (both "if long, exit" and "if
    short, exit" correctly fire together there) — see
    compute_exit_signals's docstring. Freqtrade only ever applies
    whichever exit signal matches a trade's actual open side, so that
    overlap is harmless and expected, not a bug.
    """
    sas = _load_strategy_module()
    y_ohlcv, x_ohlcv = make_oscillating_cointegrated_pair(n=400)
    strategy = sas.StatArbSwing({"stake_currency": "USDC", "runmode": None, "user_data_dir": tempfile.mkdtemp(prefix="hermes_test_")})

    df = _run_full_pipeline(strategy, y_ohlcv, x_ohlcv)

    assert not ((df["enter_long"] == 1) & (df["enter_short"] == 1)).any()


def test_full_pipeline_no_entry_signal_before_startup_candle_count() -> None:
    """No entries should be possible before startup_candle_count candles have elapsed."""
    sas = _load_strategy_module()
    y_ohlcv, x_ohlcv = make_oscillating_cointegrated_pair(n=400)
    strategy = sas.StatArbSwing({"stake_currency": "USDC", "runmode": None, "user_data_dir": tempfile.mkdtemp(prefix="hermes_test_")})

    df = _run_full_pipeline(strategy, y_ohlcv, x_ohlcv)

    warmup = df.iloc[: strategy.startup_candle_count - 1]
    assert (warmup["enter_long"] == 0).all()
    assert (warmup["enter_short"] == 0).all()


# ---------------------------------------------------------------------------
# Hyperopt configuration (optimize/ framework)
# ---------------------------------------------------------------------------


def test_hyperopt_parameters_default_to_the_strategy_constants(resolved_strategy: IStrategy) -> None:
    """entry_zscore_param/exit_zscore_param must default to ENTRY_ZSCORE/EXIT_ZSCORE.

    Outside of an active hyperopt run, `.value` equals each parameter's
    `default=`, so normal dry-run/live/backtest behavior must be
    unaffected by their existence.
    """
    assert resolved_strategy.entry_zscore_param.value == pytest.approx(
        resolved_strategy.ENTRY_ZSCORE
    )
    assert resolved_strategy.exit_zscore_param.value == pytest.approx(
        resolved_strategy.EXIT_ZSCORE
    )


def test_hyperopt_parameters_are_in_the_expected_spaces(resolved_strategy: IStrategy) -> None:
    """entry_zscore_param is a 'buy'-space, exit_zscore_param a 'sell'-space parameter.

    This is what makes `freqtrade hyperopt --spaces buy sell` (see
    optimize.hyperopt_launcher.HyperoptConfig's default spaces) actually
    tune them.
    """
    assert resolved_strategy.entry_zscore_param.space == "buy"
    assert resolved_strategy.exit_zscore_param.space == "sell"


def test_full_pipeline_reads_hyperopt_parameter_value_dynamically() -> None:
    """Overriding entry_zscore_param.value must change actual entry signals.

    This is the behavior Freqtrade's hyperopt engine relies on: it
    mutates a live parameter's `.value` between epochs on one
    long-lived strategy instance (rather than re-constructing the
    strategy), so populate_entry_trend must read `.value` at call time,
    not a value captured once in __init__. Asserting this directly
    (rather than only checking the default) is what actually validates
    the hyperopt wiring works, not just that it's inert by default.
    """
    sas = _load_strategy_module()
    y_ohlcv, x_ohlcv = make_oscillating_cointegrated_pair(n=400)
    strategy = sas.StatArbSwing({"stake_currency": "USDC", "runmode": None, "user_data_dir": tempfile.mkdtemp(prefix="hermes_test_")})

    baseline_df = _run_full_pipeline(strategy, y_ohlcv.copy(), x_ohlcv)
    baseline_entries = int(baseline_df["enter_long"].sum() + baseline_df["enter_short"].sum())

    # A much smaller threshold should fire strictly more (or equal, in a
    # degenerate all-zero case) entry signals than the default 2.0.
    strategy.entry_zscore_param.value = 0.1
    lowered_df = _run_full_pipeline(strategy, y_ohlcv.copy(), x_ohlcv)
    lowered_entries = int(lowered_df["enter_long"].sum() + lowered_df["enter_short"].sum())

    assert lowered_entries > baseline_entries
