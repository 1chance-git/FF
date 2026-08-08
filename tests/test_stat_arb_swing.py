"""Unit tests for user_data/strategies/StatArbSwing.py.

The strategy file lives under `user_data/strategies/`, which is not an
importable package (Freqtrade execs strategy files directly), so it's
loaded here via `importlib` from its file path, exactly as Freqtrade's
own strategy resolver does.

Tests focus on the module-level pure functions (role determination,
market-data alignment, indicator computation, entry/exit signal logic,
position-notional aggregation) since those carry all the actual decision
logic and require no Freqtrade runtime state. A handful of integration
tests instantiate the real `StatArbSwing` class directly (bypassing the
full bot/exchange stack, like test_bot_startup.py does for the base
strategy) to verify the Freqtrade-facing hooks are wired correctly.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STRATEGY_PATH = REPO_ROOT / "user_data" / "strategies" / "StatArbSwing.py"


def _load_strategy_module():
    spec = importlib.util.spec_from_file_location("StatArbSwing", STRATEGY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sas():
    """The loaded StatArbSwing strategy module."""
    return _load_strategy_module()


Y_PAIR = "ETH/USDC:USDC"
X_PAIR = "BTC/USDC:USDC"


def make_ohlcv(close: np.ndarray, idx: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": idx,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 100.0,
        }
    )


def make_cointegrated_legs(
    n: int = 250, beta: float = 2.0, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build cointegrated y (ETH) and x (BTC) OHLCV frames."""
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(seed)
    x = 100 + np.cumsum(rng.normal(0, 1, n))
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = noise[i - 1] * 0.3 + rng.normal(0, 1)
    y = beta * x + 5 + noise
    return make_ohlcv(y, idx), make_ohlcv(x, idx)


# ---------------------------------------------------------------------------
# determine_pair_role
# ---------------------------------------------------------------------------


def test_determine_pair_role_y(sas) -> None:
    assert sas.determine_pair_role(Y_PAIR, Y_PAIR, X_PAIR) == sas.ROLE_Y


def test_determine_pair_role_x(sas) -> None:
    assert sas.determine_pair_role(X_PAIR, Y_PAIR, X_PAIR) == sas.ROLE_X


def test_determine_pair_role_unrelated(sas) -> None:
    assert sas.determine_pair_role("SOL/USDC:USDC", Y_PAIR, X_PAIR) is None


# ---------------------------------------------------------------------------
# build_aligned_closes
# ---------------------------------------------------------------------------


def test_build_aligned_closes_shares_index(sas) -> None:
    y_df, x_df = make_cointegrated_legs(n=100)
    y_close, x_close = sas.build_aligned_closes(Y_PAIR, y_df, X_PAIR, x_df, "5m")
    assert len(y_close) == len(x_close) == 100
    assert y_close.index.equals(x_close.index)


def test_build_aligned_closes_intersects_timestamps(sas) -> None:
    y_df, x_df = make_cointegrated_legs(n=100)
    x_df_trimmed = x_df.iloc[10:].reset_index(drop=True)  # missing first 10 candles

    y_close, x_close = sas.build_aligned_closes(Y_PAIR, y_df, X_PAIR, x_df_trimmed, "5m")

    assert len(y_close) == 90
    assert len(x_close) == 90


def test_build_aligned_closes_raises_on_invalid_data(sas):
    from stat_arb.data.market_data import MarketDataError

    y_df, x_df = make_cointegrated_legs(n=10)
    y_df_broken = y_df.copy()
    y_df_broken.loc[0, "high"] = -1.0  # negative price -> fails validation

    with pytest.raises(MarketDataError):
        sas.build_aligned_closes(Y_PAIR, y_df_broken, X_PAIR, x_df, "5m")


# ---------------------------------------------------------------------------
# compute_pair_indicators
# ---------------------------------------------------------------------------


def test_compute_pair_indicators_produces_expected_columns(sas) -> None:
    from stat_arb.risk.risk import RegimeConfig, TrendFilterConfig
    from stat_arb.signal.cointegration import CointegrationConfig
    from stat_arb.signal.regression import RollingRegressionConfig

    y_df, x_df = make_cointegrated_legs(n=250)
    y_close, x_close = sas.build_aligned_closes(Y_PAIR, y_df, X_PAIR, x_df, "5m")

    indicators = sas.compute_pair_indicators(
        y_close,
        x_close,
        RollingRegressionConfig(window=60),
        CointegrationConfig(spread_window=40),
        RegimeConfig(window=30),
        TrendFilterConfig(window=20),
    )

    expected_columns = {
        "hedge_ratio",
        "spread",
        "spread_mean",
        "spread_std",
        "zscore",
        "regime",
        "is_trending",
        "is_cointegrated",
        "cointegration_pvalue",
    }
    assert expected_columns <= set(indicators.columns)
    assert indicators.index.equals(y_close.index)
    # Hedge ratio should recover something close to the true beta of 2.0.
    assert indicators["hedge_ratio"].dropna().tail(20).mean() == pytest.approx(2.0, abs=0.2)
    # Cointegration should be detected for this genuinely cointegrated pair.
    assert indicators["is_cointegrated"].any()


def test_compute_pair_indicators_warmup_is_nan(sas) -> None:
    from stat_arb.risk.risk import RegimeConfig, TrendFilterConfig
    from stat_arb.signal.cointegration import CointegrationConfig
    from stat_arb.signal.regression import RollingRegressionConfig

    y_df, x_df = make_cointegrated_legs(n=250)
    y_close, x_close = sas.build_aligned_closes(Y_PAIR, y_df, X_PAIR, x_df, "5m")

    indicators = sas.compute_pair_indicators(
        y_close,
        x_close,
        RollingRegressionConfig(window=60),
        CointegrationConfig(spread_window=40),
        RegimeConfig(window=30),
        TrendFilterConfig(window=20),
    )

    assert indicators["hedge_ratio"].iloc[:59].isna().all()


def test_compute_pair_indicators_handles_insufficient_data(sas) -> None:
    from stat_arb.risk.risk import RegimeConfig, TrendFilterConfig
    from stat_arb.signal.cointegration import CointegrationConfig
    from stat_arb.signal.regression import RollingRegressionConfig

    y_df, x_df = make_cointegrated_legs(n=30)  # shorter than regression window
    y_close, x_close = sas.build_aligned_closes(Y_PAIR, y_df, X_PAIR, x_df, "5m")

    indicators = sas.compute_pair_indicators(
        y_close,
        x_close,
        RollingRegressionConfig(window=60),
        CointegrationConfig(spread_window=40),
        RegimeConfig(window=30),
        TrendFilterConfig(window=20),
    )

    assert len(indicators) == 30
    assert indicators["zscore"].isna().all()
    assert not indicators["is_cointegrated"].any()


# ---------------------------------------------------------------------------
# merge_indicators_into_dataframe
# ---------------------------------------------------------------------------


def test_merge_preserves_row_count(sas) -> None:
    y_df, _ = make_cointegrated_legs(n=50)
    indicators = pd.DataFrame(
        {"zscore": np.linspace(-1, 1, 50)},
        index=pd.DatetimeIndex(y_df["date"], name="date"),
    )

    merged = sas.merge_indicators_into_dataframe(y_df, indicators)

    assert len(merged) == len(y_df)
    assert "zscore" in merged.columns


def test_merge_fills_nan_for_missing_indicator_rows(sas) -> None:
    y_df, _ = make_cointegrated_legs(n=50)
    indicators = pd.DataFrame(
        {"zscore": np.linspace(-1, 1, 40)},
        index=pd.DatetimeIndex(y_df["date"].iloc[10:], name="date"),
    )

    merged = sas.merge_indicators_into_dataframe(y_df, indicators)

    assert len(merged) == 50
    assert merged["zscore"].iloc[:10].isna().all()
    assert merged["zscore"].iloc[10:].notna().all()


# ---------------------------------------------------------------------------
# compute_entry_signals / compute_exit_signals
# ---------------------------------------------------------------------------


def make_signal_dataframe(zscores: list[float]) -> pd.DataFrame:
    n = len(zscores)
    return pd.DataFrame(
        {
            "zscore": zscores,
            "regime": ["mean_reverting"] * n,
            "is_trending": [False] * n,
            "is_cointegrated": [True] * n,
        }
    )


def test_entry_signals_y_role_direction(sas) -> None:
    df = make_signal_dataframe([-3.0, 0.0, 3.0])
    enter_long, enter_short = sas.compute_entry_signals(df, sas.ROLE_Y, 2.0, True, True, True)
    assert list(enter_long) == [True, False, False]
    assert list(enter_short) == [False, False, True]


def test_entry_signals_x_role_is_flipped(sas) -> None:
    df = make_signal_dataframe([-3.0, 0.0, 3.0])
    enter_long, enter_short = sas.compute_entry_signals(df, sas.ROLE_X, 2.0, True, True, True)
    assert list(enter_long) == [False, False, True]
    assert list(enter_short) == [True, False, False]


def test_entry_signals_respect_nan_zscore(sas) -> None:
    df = make_signal_dataframe([float("nan"), -3.0])
    enter_long, enter_short = sas.compute_entry_signals(df, sas.ROLE_Y, 2.0, True, True, True)
    assert list(enter_long) == [False, True]


def test_entry_signals_blocked_by_regime(sas) -> None:
    df = make_signal_dataframe([-3.0])
    df["regime"] = "trending"
    enter_long, _ = sas.compute_entry_signals(df, sas.ROLE_Y, 2.0, True, True, True)
    assert not enter_long.iloc[0]


def test_entry_signals_blocked_by_trend_filter(sas) -> None:
    df = make_signal_dataframe([-3.0])
    df["is_trending"] = True
    enter_long, _ = sas.compute_entry_signals(df, sas.ROLE_Y, 2.0, True, True, True)
    assert not enter_long.iloc[0]


def test_entry_signals_blocked_by_cointegration(sas) -> None:
    df = make_signal_dataframe([-3.0])
    df["is_cointegrated"] = False
    enter_long, _ = sas.compute_entry_signals(df, sas.ROLE_Y, 2.0, True, True, True)
    assert not enter_long.iloc[0]


def test_entry_signals_gates_can_be_disabled(sas) -> None:
    df = make_signal_dataframe([-3.0])
    df["regime"] = "trending"
    df["is_trending"] = True
    df["is_cointegrated"] = False
    enter_long, _ = sas.compute_entry_signals(df, sas.ROLE_Y, 2.0, False, False, False)
    assert enter_long.iloc[0]


def test_exit_signals_y_role_direction(sas) -> None:
    df = make_signal_dataframe([-0.2, 0.0, 0.2])
    exit_long, exit_short = sas.compute_exit_signals(df, sas.ROLE_Y, 0.5)
    assert list(exit_long) == [True, True, True]
    assert list(exit_short) == [True, True, True]


def test_exit_signals_only_trigger_on_reversion(sas) -> None:
    df = make_signal_dataframe([-3.0, 3.0])
    exit_long, exit_short = sas.compute_exit_signals(df, sas.ROLE_Y, 0.5)
    # Still far from mean -> not yet reverted for the corresponding side.
    assert not exit_long.iloc[0]
    assert not exit_short.iloc[1]


def test_exit_signals_x_role_is_flipped(sas) -> None:
    df = make_signal_dataframe([-3.0, 3.0])
    exit_long_y, exit_short_y = sas.compute_exit_signals(df, sas.ROLE_Y, 0.5)
    exit_long_x, exit_short_x = sas.compute_exit_signals(df, sas.ROLE_X, 0.5)
    assert list(exit_long_x) == list(exit_short_y)
    assert list(exit_short_x) == list(exit_long_y)


# ---------------------------------------------------------------------------
# positions_notional_from_trades
# ---------------------------------------------------------------------------


def make_trade(pair: str, stake_amount: float, leverage: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(pair=pair, stake_amount=stake_amount, leverage=leverage)


def test_positions_notional_basic(sas) -> None:
    trades = [make_trade(X_PAIR, 500.0, leverage=2.0)]
    result = sas.positions_notional_from_trades(trades, exclude_pair=Y_PAIR)
    assert result == {X_PAIR: 1000.0}


def test_positions_notional_excludes_given_pair(sas) -> None:
    trades = [make_trade(Y_PAIR, 500.0), make_trade(X_PAIR, 300.0)]
    result = sas.positions_notional_from_trades(trades, exclude_pair=Y_PAIR)
    assert result == {X_PAIR: 300.0}


def test_positions_notional_aggregates_same_pair(sas) -> None:
    trades = [make_trade(X_PAIR, 100.0), make_trade(X_PAIR, 50.0)]
    result = sas.positions_notional_from_trades(trades, exclude_pair=Y_PAIR)
    assert result == {X_PAIR: 150.0}


def test_positions_notional_defaults_leverage_to_one(sas) -> None:
    trades = [SimpleNamespace(pair=X_PAIR, stake_amount=200.0, leverage=None)]
    result = sas.positions_notional_from_trades(trades, exclude_pair=Y_PAIR)
    assert result == {X_PAIR: 200.0}


# ---------------------------------------------------------------------------
# Strategy class / Freqtrade-facing hooks
# ---------------------------------------------------------------------------


def make_strategy(sas):
    strategy = sas.StatArbSwing({"stake_currency": "USDC", "runmode": None})
    return strategy


def test_strategy_class_attributes(sas) -> None:
    strategy = make_strategy(sas)
    assert strategy.can_short is True
    assert strategy.stoploss == pytest.approx(-0.05)
    assert strategy.INTERFACE_VERSION == 3
    assert strategy.Y_PAIR == Y_PAIR
    assert strategy.X_PAIR == X_PAIR


def test_strategy_populate_indicators_end_to_end(sas) -> None:
    y_df, x_df = make_cointegrated_legs(n=250)
    strategy = make_strategy(sas)
    strategy.dp = SimpleNamespace(get_pair_dataframe=lambda pair, timeframe: x_df)

    result = strategy.populate_indicators(y_df.copy(), {"pair": Y_PAIR})

    assert len(result) == len(y_df)
    assert "zscore" in result.columns
    assert "hedge_ratio" in result.columns


def test_strategy_populate_indicators_unknown_pair_is_noop(sas) -> None:
    y_df, _ = make_cointegrated_legs(n=50)
    strategy = make_strategy(sas)
    strategy.dp = SimpleNamespace(get_pair_dataframe=lambda pair, timeframe: y_df)

    result = strategy.populate_indicators(y_df.copy(), {"pair": "SOL/USDC:USDC"})

    assert "zscore" not in result.columns


def test_strategy_entry_exit_trend_produce_expected_columns(sas) -> None:
    y_df, x_df = make_cointegrated_legs(n=250)
    strategy = make_strategy(sas)
    strategy.dp = SimpleNamespace(get_pair_dataframe=lambda pair, timeframe: x_df)

    indicators_df = strategy.populate_indicators(y_df.copy(), {"pair": Y_PAIR})
    entry_df = strategy.populate_entry_trend(indicators_df.copy(), {"pair": Y_PAIR})
    exit_df = strategy.populate_exit_trend(indicators_df.copy(), {"pair": Y_PAIR})

    assert set(entry_df["enter_long"].unique()) <= {0, 1}
    assert set(entry_df["enter_short"].unique()) <= {0, 1}
    assert set(exit_df["exit_long"].unique()) <= {0, 1}
    assert set(exit_df["exit_short"].unique()) <= {0, 1}


def test_strategy_custom_stake_amount_uses_risk_engine(sas) -> None:
    strategy = make_strategy(sas)
    strategy.wallets = SimpleNamespace(get_total_stake_amount=lambda: 10_000.0)

    stake = strategy.custom_stake_amount(
        pair=Y_PAIR,
        current_time=datetime.now(timezone.utc),
        current_rate=100.0,
        proposed_stake=1000.0,
        min_stake=10.0,
        max_stake=5000.0,
        leverage=1.0,
        entry_tag=None,
        side="long",
    )

    # risk_per_trade_pct=1% of 10,000 = 100; stop distance = 5 -> 20 units -> notional 2000
    assert stake == pytest.approx(2000.0)


def test_strategy_custom_stake_amount_respects_max_stake(sas) -> None:
    strategy = make_strategy(sas)
    strategy.wallets = SimpleNamespace(get_total_stake_amount=lambda: 10_000.0)

    stake = strategy.custom_stake_amount(
        pair=Y_PAIR,
        current_time=datetime.now(timezone.utc),
        current_rate=100.0,
        proposed_stake=1000.0,
        min_stake=10.0,
        max_stake=500.0,
        leverage=1.0,
        entry_tag=None,
        side="long",
    )

    assert stake == pytest.approx(500.0)


def test_strategy_confirm_trade_entry_blocks_unknown_pair(sas) -> None:
    strategy = make_strategy(sas)
    allowed = strategy.confirm_trade_entry(
        pair="SOL/USDC:USDC",
        order_type="limit",
        amount=1.0,
        rate=100.0,
        time_in_force="GTC",
        current_time=datetime.now(timezone.utc),
        entry_tag=None,
        side="long",
    )
    assert allowed is False


def test_strategy_confirm_trade_entry_blocks_without_analyzed_data(sas) -> None:
    strategy = make_strategy(sas)
    strategy.dp = SimpleNamespace(
        get_analyzed_dataframe=lambda pair, timeframe: (pd.DataFrame(), None)
    )

    allowed = strategy.confirm_trade_entry(
        pair=Y_PAIR,
        order_type="limit",
        amount=1.0,
        rate=100.0,
        time_in_force="GTC",
        current_time=datetime.now(timezone.utc),
        entry_tag=None,
        side="long",
    )
    assert allowed is False


def test_strategy_confirm_trade_exit_arms_cooldown_on_stop_loss(sas) -> None:
    strategy = make_strategy(sas)
    now = datetime.now(timezone.utc)

    result = strategy.confirm_trade_exit(
        pair=Y_PAIR,
        trade=SimpleNamespace(),
        order_type="limit",
        amount=1.0,
        rate=95.0,
        time_in_force="GTC",
        exit_reason="stop_loss",
        current_time=now,
    )

    assert result is True
    assert strategy.risk_engine.is_in_cooldown(Y_PAIR, now + timedelta(minutes=1)) is True


def test_strategy_confirm_trade_exit_does_not_arm_cooldown_on_normal_exit(sas) -> None:
    strategy = make_strategy(sas)
    now = datetime.now(timezone.utc)

    strategy.confirm_trade_exit(
        pair=Y_PAIR,
        trade=SimpleNamespace(),
        order_type="limit",
        amount=1.0,
        rate=105.0,
        time_in_force="GTC",
        exit_reason="exit_signal",
        current_time=now,
    )

    assert strategy.risk_engine.is_in_cooldown(Y_PAIR, now + timedelta(minutes=1)) is False


def test_strategy_resolves_via_freqtrade_strategy_resolver() -> None:
    """Confirm Freqtrade's own resolver (not just direct instantiation) can load it.

    Mirrors tests/test_bot_startup.py's approach: exercises the real
    Configuration + StrategyResolver machinery against the committed
    project config, rather than only the lighter-weight direct
    instantiation used by the other tests in this file.
    """
    from freqtrade.configuration import Configuration
    from freqtrade.resolvers import StrategyResolver

    args = {
        "config": [str(REPO_ROOT / "user_data" / "config.json")],
        "strategy": "StatArbSwing",
        "strategy_path": str(REPO_ROOT / "user_data" / "strategies"),
    }
    config = Configuration(args, "trade").get_config()
    strategy = StrategyResolver.load_strategy(config)

    assert type(strategy).__name__ == "StatArbSwing"
    assert strategy.can_short is True
    assert strategy.stoploss == pytest.approx(-0.05)


def test_strategy_no_ai_dependency(sas) -> None:
    """Confirm no ML/AI framework is imported (freqAI, torch, sklearn, etc.)."""
    import ast

    with open(STRATEGY_PATH, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())

    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    banned_substrings = ("freqai", "torch", "sklearn", "tensorflow", "keras", "xgboost")
    for module_name in imported:
        lowered = module_name.lower()
        assert not any(banned in lowered for banned in banned_substrings), module_name
