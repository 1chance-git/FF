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
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.unit

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


def make_strategy(sas, user_data_dir: str | None = None):
    """Build a StatArbSwing instance, pointing its Hermes memory store at a
    throwaway temp directory rather than the repo's real `user_data/` (each
    call gets its own isolated, disposable SQLite file)."""
    user_data_dir = user_data_dir or tempfile.mkdtemp(prefix="hermes_test_")
    strategy = sas.StatArbSwing(
        {"stake_currency": "USDC", "runmode": None, "user_data_dir": user_data_dir}
    )
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


def test_strategy_confirm_trade_entry_blocks_when_wallets_unavailable(sas) -> None:
    strategy = make_strategy(sas)
    y_df, x_df = make_cointegrated_legs(n=250)
    strategy.dp = SimpleNamespace(
        get_analyzed_dataframe=lambda pair, timeframe: (
            strategy.populate_indicators(y_df.copy(), {"pair": Y_PAIR}),
            None,
        )
    )
    strategy.wallets = None

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


def test_strategy_confirm_trade_entry_fails_closed_on_unexpected_exception(sas) -> None:
    """The core "no risk gate fails open" guarantee: an internal exception must block, not allow.

    Freqtrade wraps confirm_trade_entry with
    strategy_safe_wrapper(..., default_retval=True) -- meaning an
    *unhandled* exception here would make Freqtrade treat the entry as
    confirmed. StatArbSwing must never rely on that fallback: any
    unexpected error inside confirm_trade_entry must itself be caught
    and turned into a blocked (False) entry.
    """
    strategy = make_strategy(sas)

    class ExplodingDataProvider:
        def get_analyzed_dataframe(self, pair, timeframe):
            raise RuntimeError("simulated dataprovider failure")

    strategy.dp = ExplodingDataProvider()

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


def test_strategy_confirm_trade_entry_fails_closed_when_risk_engine_raises(sas) -> None:
    """Same fail-closed guarantee, triggered from inside the risk engine call itself."""
    strategy = make_strategy(sas)
    y_df, x_df = make_cointegrated_legs(n=250)
    strategy.dp = SimpleNamespace(get_pair_dataframe=lambda pair, timeframe: x_df)
    indicators_df = strategy.populate_indicators(y_df.copy(), {"pair": Y_PAIR})
    strategy.dp = SimpleNamespace(
        get_analyzed_dataframe=lambda pair, timeframe: (indicators_df, None)
    )
    strategy.wallets = SimpleNamespace(get_total_stake_amount=lambda: 10_000.0)

    def exploding_evaluate_entry(*args, **kwargs):
        raise RuntimeError("simulated risk engine failure")

    strategy.risk_engine.evaluate_entry = exploding_evaluate_entry

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


def test_strategy_confirm_trade_exit_still_confirms_when_record_exit_raises(sas) -> None:
    """Exits must never be blocked by a cooldown-bookkeeping failure."""
    strategy = make_strategy(sas)

    def exploding_record_exit(*args, **kwargs):
        raise RuntimeError("simulated risk engine failure")

    strategy.risk_engine.record_exit = exploding_record_exit

    result = strategy.confirm_trade_exit(
        pair=Y_PAIR,
        trade=SimpleNamespace(),
        order_type="limit",
        amount=1.0,
        rate=95.0,
        time_in_force="GTC",
        exit_reason="stop_loss",
        current_time=datetime.now(timezone.utc),
    )

    assert result is True


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


# ---------------------------------------------------------------------------
# Hermes memory wiring (confirm_trade_entry / confirm_trade_exit)
# ---------------------------------------------------------------------------


def _strategy_with_allowed_entry(sas, monkeypatch: pytest.MonkeyPatch):
    """A strategy set up so confirm_trade_entry's risk gate says yes."""
    strategy = make_strategy(sas)
    y_df, x_df = make_cointegrated_legs(n=250)
    strategy.dp = SimpleNamespace(get_pair_dataframe=lambda pair, timeframe: x_df)
    indicators_df = strategy.populate_indicators(y_df.copy(), {"pair": Y_PAIR})
    strategy.dp = SimpleNamespace(
        get_analyzed_dataframe=lambda pair, timeframe: (indicators_df, None)
    )
    strategy.wallets = SimpleNamespace(get_total_stake_amount=lambda: 10_000.0)

    # No live DB/session is set up in these unit tests; get_open_trades()
    # only needs to return "no other open positions" for this scenario.
    monkeypatch.setattr(sas.Trade, "get_open_trades", staticmethod(lambda: []))

    from stat_arb.risk.risk import EntryDecision

    strategy.risk_engine.evaluate_entry = lambda **kwargs: EntryDecision(
        allowed=True,
        reasons=(),
        position_size=None,
        stop_loss_price=None,
        regime=None,
        is_trending=None,
    )
    return strategy


class FakeTrade:
    """Minimal stand-in for freqtrade.persistence.Trade with just what
    _record_exit reads: open_rate/open_date_utc, fee costs, funding, and
    the two P&L calculation methods."""

    def __init__(self, open_rate: float = 100.0) -> None:
        self.open_rate = open_rate
        self.open_date_utc = datetime.now(timezone.utc) - timedelta(hours=2)
        self.fee_open_cost = 1.5
        self.fee_close_cost = 1.2
        self.funding_fees = 0.3

    def calc_profit(self, rate: float) -> float:
        return rate - self.open_rate

    def calc_profit_ratio(self, rate: float) -> float:
        return (rate - self.open_rate) / self.open_rate


def test_confirm_trade_entry_records_a_trade_when_allowed(sas, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = _strategy_with_allowed_entry(sas, monkeypatch)
    now = datetime.now(timezone.utc)

    allowed = strategy.confirm_trade_entry(
        pair=Y_PAIR,
        order_type="limit",
        amount=1.0,
        rate=123.0,
        time_in_force="GTC",
        current_time=now,
        entry_tag=None,
        side="long",
    )

    assert allowed is True
    [saved] = strategy.memory_store.get_trades(pair=Y_PAIR)
    assert saved.entry_price == 123.0
    assert saved.side == "long"
    assert saved.entry_time == now
    # entry_zscore/hedge_ratio come straight from the analyzed dataframe.
    assert saved.entry_zscore is not None
    assert saved.hedge_ratio is not None


def test_confirm_trade_entry_does_not_record_when_blocked(sas) -> None:
    strategy = make_strategy(sas)  # unknown pair -> blocked before any recording

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
    assert strategy.memory_store.get_trades() == []


def test_confirm_trade_exit_records_a_completed_trade(sas, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = _strategy_with_allowed_entry(sas, monkeypatch)
    entry_time = datetime.now(timezone.utc)
    strategy.confirm_trade_entry(
        pair=Y_PAIR,
        order_type="limit",
        amount=1.0,
        rate=100.0,
        time_in_force="GTC",
        current_time=entry_time,
        entry_tag=None,
        side="long",
    )

    exit_time = entry_time + timedelta(hours=2)
    trade = FakeTrade(open_rate=100.0)
    result = strategy.confirm_trade_exit(
        pair=Y_PAIR,
        trade=trade,
        order_type="limit",
        amount=1.0,
        rate=110.0,
        time_in_force="GTC",
        exit_reason="exit_signal",
        current_time=exit_time,
    )

    assert result is True
    trades = strategy.memory_store.get_trades(pair=Y_PAIR)
    # One row from the entry, one combined row written at exit.
    assert len(trades) == 2
    completed = trades[-1]
    assert completed.exit_price == 110.0
    assert completed.pnl == pytest.approx(10.0)
    assert completed.pnl_pct == pytest.approx(0.1)
    assert completed.fees == pytest.approx(2.7)
    assert completed.funding == pytest.approx(0.3)
    assert completed.exit_reason == "exit_signal"
    assert completed.holding_time_seconds == pytest.approx(7200.0)
    assert completed.entry_price == 100.0
    # The pending context was consumed.
    assert Y_PAIR not in strategy._pending_entry_context


def test_confirm_trade_exit_without_prior_entry_context_still_records(sas) -> None:
    """A restart between entry and exit shouldn't lose the exit record, only the entry context."""
    strategy = make_strategy(sas)
    now = datetime.now(timezone.utc)
    trade = FakeTrade(open_rate=100.0)

    result = strategy.confirm_trade_exit(
        pair=Y_PAIR,
        trade=trade,
        order_type="limit",
        amount=1.0,
        rate=90.0,
        time_in_force="GTC",
        exit_reason="stop_loss",
        current_time=now,
    )

    assert result is True
    [saved] = strategy.memory_store.get_trades(pair=Y_PAIR)
    assert saved.exit_price == 90.0
    assert saved.pnl == pytest.approx(-10.0)
    assert saved.entry_zscore is None  # no entry context available
    assert saved.entry_price == 100.0  # fell back to trade.open_rate


def test_memory_store_failure_does_not_block_entry_or_exit(sas, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = _strategy_with_allowed_entry(sas, monkeypatch)

    class ExplodingMemoryStore:
        def record_trade(self, *args, **kwargs):
            raise RuntimeError("simulated disk failure")

    strategy.memory_store = ExplodingMemoryStore()

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
    assert allowed is True

    result = strategy.confirm_trade_exit(
        pair=Y_PAIR,
        trade=FakeTrade(),
        order_type="limit",
        amount=1.0,
        rate=105.0,
        time_in_force="GTC",
        exit_reason="exit_signal",
        current_time=datetime.now(timezone.utc),
    )
    assert result is True


def test_memory_store_none_does_not_block_entry_or_exit(sas, monkeypatch: pytest.MonkeyPatch) -> None:
    """If the store failed to initialize entirely (self.memory_store is None)."""
    strategy = _strategy_with_allowed_entry(sas, monkeypatch)
    strategy.memory_store = None

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
    assert allowed is True

    result = strategy.confirm_trade_exit(
        pair=Y_PAIR,
        trade=FakeTrade(),
        order_type="limit",
        amount=1.0,
        rate=105.0,
        time_in_force="GTC",
        exit_reason="exit_signal",
        current_time=datetime.now(timezone.utc),
    )
    assert result is True


def test_latest_signal_context_pure_helper(sas) -> None:
    df = pd.DataFrame({"zscore": [float("nan"), 1.5, 2.0], "hedge_ratio": [1.1, 1.2, 1.3]})
    context = sas.latest_signal_context(df)
    assert context["zscore"] == 2.0
    assert context["hedge_ratio"] == 1.3
    assert context["regime"] is None  # column absent


def test_latest_signal_context_handles_all_nan_column(sas) -> None:
    df = pd.DataFrame({"zscore": [float("nan"), float("nan")]})
    context = sas.latest_signal_context(df)
    assert context["zscore"] is None


# ---------------------------------------------------------------------------
# Hermes memory wiring: strategy error recording
# ---------------------------------------------------------------------------


class _ExplodingDataProvider:
    """Raises on any analyzed-dataframe lookup, to trigger confirm_trade_entry's
    outer fail-closed exception handler."""

    def get_analyzed_dataframe(self, pair, timeframe):
        raise RuntimeError("simulated dataprovider failure")


def test_confirm_trade_entry_exception_is_persisted_with_pair_and_type(sas) -> None:
    """Category 1 + 2: an existing (already fail-closed) exception is persisted,
    and includes the pair plus the exception type/message."""
    strategy = make_strategy(sas)
    strategy.dp = _ExplodingDataProvider()

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

    assert allowed is False  # existing fail-closed behavior preserved
    [error] = strategy.memory_store.get_errors()
    assert error.source == "confirm_trade_entry"
    assert Y_PAIR in error.message
    assert "RuntimeError" in error.message
    assert "simulated dataprovider failure" in error.message
    assert error.severity == "error"


def test_populate_indicators_market_data_error_is_recorded_and_dataframe_unchanged(sas) -> None:
    """Category 1 + 6: an indicator/calculation failure (MarketDataError) is
    persisted, and populate_indicators still returns the dataframe unchanged --
    the existing fail-safe/no-signal-change behavior is untouched."""
    strategy = make_strategy(sas)
    y_df, x_df = make_cointegrated_legs(n=50)
    x_df_broken = x_df.copy()
    x_df_broken.loc[0, "high"] = -1.0  # negative price -> MarketDataError
    strategy.dp = SimpleNamespace(get_pair_dataframe=lambda pair, timeframe: x_df_broken)

    y_df_input = y_df.copy()
    result = strategy.populate_indicators(y_df_input, {"pair": Y_PAIR})

    pd.testing.assert_frame_equal(result, y_df)  # unchanged, exactly as before this change
    [error] = strategy.memory_store.get_errors()
    assert error.source == "populate_indicators"
    assert Y_PAIR in error.message
    assert "MarketDataError" in error.message


def test_confirm_trade_exit_risk_engine_failure_is_persisted_and_still_confirms(sas) -> None:
    """Category 1 + 6: a trade-callback failure (risk_engine.record_exit raising)
    is persisted, and confirm_trade_exit still returns True unchanged."""
    strategy = make_strategy(sas)

    def exploding_record_exit(*args, **kwargs):
        raise RuntimeError("simulated risk engine failure")

    strategy.risk_engine.record_exit = exploding_record_exit

    result = strategy.confirm_trade_exit(
        pair=Y_PAIR,
        trade=SimpleNamespace(),
        order_type="limit",
        amount=1.0,
        rate=95.0,
        time_in_force="GTC",
        exit_reason="stop_loss",
        current_time=datetime.now(timezone.utc),
    )

    assert result is True  # existing behavior: exit is never blocked
    errors = strategy.memory_store.get_errors()
    assert any(e.source == "confirm_trade_exit" and Y_PAIR in e.message for e in errors)


def test_record_strategy_error_handles_missing_pair_and_note(sas) -> None:
    """Category 3: optional pair/note context is handled safely when absent."""
    strategy = make_strategy(sas)

    strategy._record_strategy_error("some_source", RuntimeError("boom"))

    [error] = strategy.memory_store.get_errors()
    assert error.source == "some_source"
    assert error.message == "RuntimeError: boom"


def test_record_strategy_error_is_a_noop_without_a_memory_store(sas) -> None:
    """Category 3 (and part of 4): missing memory_store is handled safely, not as a crash."""
    strategy = make_strategy(sas)
    strategy.memory_store = None

    strategy._record_strategy_error("some_source", RuntimeError("boom"), pair=Y_PAIR)  # must not raise


def test_memory_failure_during_error_recording_does_not_change_fail_closed_behavior(sas) -> None:
    """Category 4: if persisting the error *itself* fails, the original fail-safe
    decision (blocked entry) must be completely unaffected."""
    strategy = make_strategy(sas)
    strategy.dp = _ExplodingDataProvider()

    class ExplodingMemoryStore:
        def record_error(self, *args, **kwargs):
            raise RuntimeError("simulated disk failure")

    strategy.memory_store = ExplodingMemoryStore()

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

    assert allowed is False  # unchanged: still fails closed


def test_memory_failure_during_error_recording_does_not_block_exit(sas) -> None:
    """Category 4, exit side: a broken memory_store must not prevent confirm_trade_exit
    from still confirming the exit, even while a trade-callback failure is also occurring."""
    strategy = make_strategy(sas)

    def exploding_record_exit(*args, **kwargs):
        raise RuntimeError("simulated risk engine failure")

    strategy.risk_engine.record_exit = exploding_record_exit

    class ExplodingMemoryStore:
        def record_error(self, *args, **kwargs):
            raise RuntimeError("simulated disk failure")

        def record_trade(self, *args, **kwargs):
            raise RuntimeError("simulated disk failure")

    strategy.memory_store = ExplodingMemoryStore()

    result = strategy.confirm_trade_exit(
        pair=Y_PAIR,
        trade=FakeTrade(),
        order_type="limit",
        amount=1.0,
        rate=95.0,
        time_in_force="GTC",
        exit_reason="stop_loss",
        current_time=datetime.now(timezone.utc),
    )

    assert result is True  # unchanged: exit is never blocked


def test_no_secrets_from_config_leak_into_persisted_errors(sas) -> None:
    """Category 5: the persisted error message is built only from the exception's
    type/message plus pair/note -- never from self.config, which may hold exchange
    credentials (walletAddress/privateKey per user_data/config.json)."""
    strategy = make_strategy(sas)
    strategy.config = {
        **strategy.config,
        "exchange": {
            "privateKey": "SECRET_KEY_ABC123",
            "walletAddress": "0xSECRETADDRESS",
        },
    }
    strategy.dp = _ExplodingDataProvider()

    strategy.confirm_trade_entry(
        pair=Y_PAIR,
        order_type="limit",
        amount=1.0,
        rate=100.0,
        time_in_force="GTC",
        current_time=datetime.now(timezone.utc),
        entry_tag=None,
        side="long",
    )

    errors = strategy.memory_store.get_errors()
    assert errors
    for error in errors:
        assert "SECRET_KEY_ABC123" not in error.message
        assert "0xSECRETADDRESS" not in error.message


def test_error_recording_does_not_change_behavior_with_or_without_memory_store(sas) -> None:
    """Category 6: confirm_trade_entry's return value is identical whether or not
    a memory_store is present -- error persistence is strictly additive."""
    strategy_with_store = make_strategy(sas)
    strategy_with_store.dp = _ExplodingDataProvider()

    strategy_without_store = make_strategy(sas)
    strategy_without_store.memory_store = None
    strategy_without_store.dp = _ExplodingDataProvider()

    kwargs = dict(
        pair=Y_PAIR,
        order_type="limit",
        amount=1.0,
        rate=100.0,
        time_in_force="GTC",
        current_time=datetime.now(timezone.utc),
        entry_tag=None,
        side="long",
    )

    assert strategy_with_store.confirm_trade_entry(**kwargs) is False
    assert strategy_without_store.confirm_trade_entry(**kwargs) is False


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
