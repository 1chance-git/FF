"""Shared fixtures and helpers for the whole test suite.

Test categories (selectable via ``pytest -m <marker>``, registered in
``pyproject.toml``):

* ``unit`` — fast, isolated tests of a single function/class. Most tests
  in this repo fall here implicitly (unmarked tests are still run by
  default; the marker exists for the subset that's useful to select
  explicitly alongside the categories below).
* ``strategy`` — validates ``StatArbSwing`` against Freqtrade's own
  config/strategy consistency checks and end-to-end no-lookahead
  behavior (see ``test_strategy_validation.py``).
* ``backtest`` — runs a real, complete `freqtrade` backtest offline
  against synthetic local data, with only the network boundary mocked
  (see ``test_backtest_validation.py``).
* ``regression`` — golden/pinned-value tests: fixed inputs with
  hand-verified expected outputs, guarding against unintended behavior
  drift across future changes (see ``test_golden_values.py``).
* ``numerical`` — cross-implementation consistency and invariant checks
  (e.g. our rolling condition number vs. `numpy.linalg.cond`, position
  sizing arithmetic identities) (see ``test_numerical_consistency.py``).

This module provides the fixtures/helpers several of those categories
share: synthetic OHLCV generation and the mocked-exchange context
manager used to run Freqtrade's real engine (`FreqtradeBot`,
`Backtesting`) fully offline, first proven out in
``test_bot_startup.py`` and reused here for the backtest-validation
suite.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "user_data" / "config.json"
STRATEGIES_PATH = REPO_ROOT / "user_data" / "strategies"

Y_PAIR = "ETH/USDC:USDC"
X_PAIR = "BTC/USDC:USDC"
TIMEFRAME = "5m"


@pytest.fixture
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


def make_ohlcv(close: np.ndarray, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Build a minimal, internally-consistent OHLCV dataframe from a close series."""
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "date": idx,
            "open": close,
            "high": close + np.abs(close) * 0.001 + 0.01,
            "low": close - np.abs(close) * 0.001 - 0.01,
            "close": close,
            "volume": 100.0,
        }
    )


def make_oscillating_cointegrated_pair(
    n: int = 600,
    beta: float = 2.0,
    start: str = "2024-01-01",
    freq: str = "5min",
    oscillation_amplitude: float = 6.0,
    oscillation_period_bars: int = 60,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a cointegrated (y, x) OHLCV pair with a deliberate oscillating spread.

    Unlike a pure AR(1) noise residual (used elsewhere in the test
    suite for statistical realism), this adds a slow sinusoidal
    deviation on top of the cointegrating relationship so the spread
    reliably swings past typical entry z-score thresholds multiple
    times — needed for tests that must observe the strategy actually
    trade (e.g. backtest validation), not merely run without error.

    Parameters
    ----------
    n:
        Number of candles.
    beta:
        True hedge ratio: ``y = beta * x + intercept + oscillation + noise``.
    start, freq:
        Passed to ``pandas.date_range``.
    oscillation_amplitude:
        Amplitude (in price units) of the deliberate spread oscillation.
    oscillation_period_bars:
        Period of the oscillation, in bars.
    seed:
        RNG seed, for determinism.

    Returns
    -------
    tuple[DataFrame, DataFrame]
        ``(y_ohlcv, x_ohlcv)`` for :data:`Y_PAIR` / :data:`X_PAIR`.
    """
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    x = 100 + np.cumsum(rng.normal(0, 0.3, n))
    oscillation = oscillation_amplitude * np.sin(2 * np.pi * t / oscillation_period_bars)
    noise = rng.normal(0, 0.3, n)
    y = beta * x + 5 + oscillation + noise

    return make_ohlcv(y, idx), make_ohlcv(x, idx)


def fake_futures_market(pair: str, base: str) -> dict:
    """A minimal but complete ccxt-style futures market dict for ``pair``."""
    return {
        "id": base + "USDC",
        "symbol": pair,
        "base": base,
        "quote": "USDC",
        "settle": "USDC",
        "spot": False,
        "swap": True,
        "linear": True,
        "active": True,
        "type": "swap",
        "contractSize": 1,
        "precision": {"price": 2, "amount": 3},
        "limits": {
            "amount": {"min": 0.001, "max": 1000},
            "price": {"min": None, "max": None},
            "leverage": {"min": 1, "max": 20},
            "cost": {"min": 1.0, "max": None},
        },
    }


@contextmanager
def mocked_hyperliquid_exchange(markets: dict[str, dict]):
    """Patch ``Exchange`` so it never touches the network, exposing ``markets``.

    This is the same technique ``test_bot_startup.py`` uses to construct
    a real ``FreqtradeBot``/``Backtesting`` fully offline: only the
    network boundary (ccxt client construction and market loading) is
    mocked; every other code path (config validation, exchange/strategy
    resolution, the actual backtest/trading loop) runs unmodified.

    Parameters
    ----------
    markets:
        Mapping of pair -> ccxt-style market dict (see
        :func:`fake_futures_market`).

    Yields
    ------
    None
    """
    fake_ccxt_api = MagicMock()
    fake_ccxt_api.timeframes = {TIMEFRAME: TIMEFRAME, "1h": "1h"}
    fake_ccxt_api.markets = markets
    fake_ccxt_api.has = {
        key: True
        for key in (
            "fetchOHLCV",
            "fetchL2OrderBook",
            "fetchTicker",
            "fetchTickers",
            "fetchTrades",
            "cancelOrder",
            "createOrder",
            "fetchOrder",
            "fetchBalance",
            "fetchPositions",
            "fetchLeverageTiers",
            "fetchMarketLeverageTiers",
            "createMarketOrder",
            "createLimitOrder",
            "createStopLossOrder",
            "editOrder",
        )
    }
    fake_ccxt_api.precisionMode = 2
    fake_ccxt_api.options = {}
    fake_ccxt_api.id = "hyperliquid"
    fake_ccxt_api.name = "Hyperliquid"

    with (
        patch(
            "freqtrade.exchange.exchange.Exchange._load_async_markets",
            return_value=markets,
        ),
        patch(
            "freqtrade.exchange.exchange.Exchange.reload_markets",
            lambda self, *a, **k: setattr(self, "_markets", markets),
        ),
        patch(
            "freqtrade.exchange.exchange.Exchange.validate_required_startup_candles",
            return_value=None,
        ),
        patch("freqtrade.exchange.exchange.Exchange._init_ccxt", return_value=fake_ccxt_api),
    ):
        yield
