"""Backtest validation: run a real, complete Freqtrade backtest offline.

Unlike the per-module unit tests, this exercises Freqtrade's actual
``Backtesting`` engine end to end — real config validation, real
strategy resolution, the real event-driven backtest loop, real trade
accounting — against synthetic local candle data, with only the network
boundary (exchange market loading) mocked via
``tests.conftest.mocked_hyperliquid_exchange`` (the same technique
``test_bot_startup.py`` uses). This is deliberately a different kind of
test than the module-level unit tests: it validates that the assembled
system (config + `StatArbSwing` + all four `stat_arb` modules) actually
produces a coherent backtest result, not just that each piece works in
isolation.

Why this needs futures-specific fixture data
---------------------------------------------
Freqtrade's futures backtesting additionally requires 1h mark-price and
funding-rate candle history (to compute funding fees over the life of
each trade) — this is loaded from disk exactly like OHLCV candles, so
the fixture writes minimal synthetic 1h candles for both alongside the
5m OHLCV data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from freqtrade.configuration import Configuration
from freqtrade.data.history.datahandlers import get_datahandler
from freqtrade.enums import CandleType

from tests.conftest import (
    CONFIG_PATH,
    STRATEGIES_PATH,
    X_PAIR,
    Y_PAIR,
    fake_futures_market,
    make_ohlcv,
    make_oscillating_cointegrated_pair,
    mocked_hyperliquid_exchange,
)

pytestmark = pytest.mark.backtest


def _write_futures_fixture_data(datadir: Path, y_ohlcv: pd.DataFrame, x_ohlcv: pd.DataFrame) -> None:
    """Write 5m OHLCV plus the 1h mark-price/funding-rate data futures backtesting needs."""
    datadir.mkdir(parents=True, exist_ok=True)
    handler = get_datahandler(datadir, data_format="feather")

    handler.ohlcv_store(Y_PAIR, "5m", y_ohlcv, CandleType.FUTURES)
    handler.ohlcv_store(X_PAIR, "5m", x_ohlcv, CandleType.FUTURES)

    n_hours = 80
    idx_1h = pd.date_range(y_ohlcv["date"].iloc[0], periods=n_hours, freq="1h", tz="UTC")
    for pair, ohlcv in ((Y_PAIR, y_ohlcv), (X_PAIR, x_ohlcv)):
        resampled_close = np.interp(
            np.linspace(0, len(ohlcv) - 1, n_hours),
            np.arange(len(ohlcv)),
            ohlcv["close"].to_numpy(),
        )
        mark_df = make_ohlcv(resampled_close, idx_1h)
        handler.ohlcv_store(pair, "1h", mark_df, CandleType.FUTURES)

        funding_df = make_ohlcv(np.zeros(n_hours), idx_1h)
        handler.ohlcv_store(pair, "1h", funding_df, CandleType.FUNDING_RATE)


def _run_backtest(tmp_path: Path, timerange: str) -> dict:
    """Write synthetic fixture data and run a real, fully offline backtest.

    Returns
    -------
    dict
        ``Backtesting.results`` after ``start()`` completes.
    """
    y_ohlcv, x_ohlcv = make_oscillating_cointegrated_pair(n=600)
    datadir = tmp_path / "hyperliquid"
    _write_futures_fixture_data(datadir, y_ohlcv, x_ohlcv)

    args = {
        "config": [str(CONFIG_PATH)],
        "strategy": "StatArbSwing",
        "strategy_path": str(STRATEGIES_PATH),
        "datadir": str(datadir),
        "timerange": timerange,
        "export": "none",
    }
    config = Configuration(args, "backtest").get_config()
    config["dry_run"] = True
    config["fee"] = 0.001  # short-circuits Exchange.get_fee's market lookup

    markets = {
        Y_PAIR: fake_futures_market(Y_PAIR, "ETH"),
        X_PAIR: fake_futures_market(X_PAIR, "BTC"),
    }

    with mocked_hyperliquid_exchange(markets):
        from freqtrade.optimize.backtesting import Backtesting

        backtesting = Backtesting(config)
        backtesting.start()

    return backtesting.results


@pytest.fixture(scope="module")
def backtest_results() -> dict:
    """Run one real offline backtest, shared read-only across this module's tests.

    Scoped to the module (rather than per-test) since the backtest
    itself takes a few seconds; the assertions below only read its
    output and don't mutate it.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        return _run_backtest(Path(tmp_dir), timerange="20240101-20240103")


@pytest.fixture(scope="module")
def strategy_results(backtest_results: dict) -> dict:
    """The `StatArbSwing`-specific slice of the backtest results."""
    return backtest_results["strategy"]["StatArbSwing"]


def test_backtest_completes_and_produces_results(backtest_results: dict) -> None:
    """The engine ran to completion and produced the expected top-level structure."""
    assert "strategy" in backtest_results
    assert "StatArbSwing" in backtest_results["strategy"]
    assert "strategy_comparison" in backtest_results


def test_backtest_produces_trades_on_the_engineered_data(strategy_results: dict) -> None:
    """The oscillating synthetic spread must actually trigger trades.

    A backtest that "completes" but silently produces zero trades could
    still be hiding a broken entry-signal pipeline; this fixture's data
    is deliberately engineered (see
    make_oscillating_cointegrated_pair) to cross the entry z-score
    threshold repeatedly, so at least one trade is a meaningful
    assertion, not a tautology.
    """
    trades = strategy_results["trades"]
    assert len(trades) > 0


def test_backtest_trades_both_legs_of_the_pair(strategy_results: dict) -> None:
    """Genuine pairs trading requires both legs to actually get traded."""
    traded_pairs = {trade["pair"] for trade in strategy_results["trades"]}
    assert traded_pairs <= {Y_PAIR, X_PAIR}
    # With enough oscillations in the fixture window, both legs should trade;
    # a single-leg-only result would indicate the role-flipping logic is broken.
    assert traded_pairs == {Y_PAIR, X_PAIR}


def test_backtest_trades_have_sane_numeric_fields(strategy_results: dict) -> None:
    """No NaN/inf/non-positive prices or amounts anywhere in the trade log."""
    for trade in strategy_results["trades"]:
        assert trade["open_rate"] > 0
        assert np.isfinite(trade["open_rate"])
        assert trade["amount"] > 0
        assert np.isfinite(trade["amount"])
        assert np.isfinite(trade["profit_abs"])
        assert np.isfinite(trade["profit_ratio"])
        if not trade["is_open"]:
            assert trade["close_rate"] > 0
            assert np.isfinite(trade["close_rate"])


def test_backtest_trades_respect_configured_stop_loss(strategy_results: dict) -> None:
    """Every trade's stop-loss ratio matches the strategy's configured 5%."""
    for trade in strategy_results["trades"]:
        assert trade["stop_loss_ratio"] == pytest.approx(-0.05)


def test_backtest_position_sizing_arithmetic_is_consistent(strategy_results: dict) -> None:
    """amount * open_rate must reconcile with stake_amount * leverage for every trade."""
    for trade in strategy_results["trades"]:
        notional = trade["amount"] * trade["open_rate"]
        expected_notional = trade["stake_amount"] * trade["leverage"]
        assert notional == pytest.approx(expected_notional, rel=0.02)


def test_backtest_entry_tags_identify_the_strategy_signal(strategy_results: dict) -> None:
    """Trades opened by populate_entry_trend carry the strategy's enter_tag."""
    for trade in strategy_results["trades"]:
        assert trade["enter_tag"] == "stat_arb_spread_reversion"


def test_backtest_is_deterministic_across_repeated_runs(tmp_path: Path) -> None:
    """Re-running the same fixture data must produce byte-identical trade outcomes.

    A stochastic or state-leaking pipeline (e.g. a global RNG seed not
    reset, or cross-test contamination from the CooldownTracker) would
    show up here as a mismatch between two runs given the exact same
    input.
    """
    results_a = _run_backtest(tmp_path / "run_a", timerange="20240101-20240102")
    results_b = _run_backtest(tmp_path / "run_b", timerange="20240101-20240102")

    trades_a = results_a["strategy"]["StatArbSwing"]["trades"]
    trades_b = results_b["strategy"]["StatArbSwing"]["trades"]

    assert len(trades_a) == len(trades_b)
    for trade_a, trade_b in zip(trades_a, trades_b, strict=True):
        assert trade_a["pair"] == trade_b["pair"]
        assert trade_a["open_rate"] == pytest.approx(trade_b["open_rate"])
        assert trade_a["profit_abs"] == pytest.approx(trade_b["profit_abs"])
