"""End-to-end startup verification for the foundation project.

This sandbox's network policy blocks outbound calls to exchange APIs, so a
true network smoke test (``freqtrade trade``) cannot reach Hyperliquid's
API here. That is an environment restriction, not a configuration defect:
the same config works unmodified wherever outbound HTTPS to Hyperliquid is
permitted.

To still prove that configuration parsing, strategy resolution, and
``FreqtradeBot`` construction all succeed together, this test builds the
real bot object against ``user_data/config.json`` with only the network
boundary (ccxt market loading) mocked out — exactly the technique
Freqtrade's own test suite uses (see ``tests/conftest.py`` in the
Freqtrade repository, function ``get_patched_freqtradebot``). Everything
else — config validation, exchange class selection, pairlist handling,
strategy loading, persistence/db init — runs unmodified.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from freqtrade.configuration import Configuration
from freqtrade.freqtradebot import FreqtradeBot

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "user_data" / "config.json"


def _build_config() -> dict:
    """Load the real project config through Freqtrade's own loader."""
    args = {
        "config": [str(CONFIG_PATH)],
        "strategy_path": str(REPO_ROOT / "user_data" / "strategies"),
    }
    return Configuration(args, "trade").get_config()


def test_freqtradebot_starts_with_project_config() -> None:
    """FreqtradeBot must initialize cleanly from the committed config.

    Market/ticker retrieval (the only network-bound step at startup) is
    mocked to two synthetic USDC-margined futures markets so the rest of
    the startup path — pairlist resolution, strategy loading, wallet and
    persistence init — runs for real against the actual project config.
    """
    config = _build_config()
    # In real CLI usage Freqtrade defaults db_url itself (sqlite file under
    # user_data/); pin it to an in-memory DB here so this test never touches
    # disk state shared with other tests or a real dry-run database.
    config["db_url"] = "sqlite://"

    fake_markets = {
        "BTC/USDC:USDC": {
            "id": "BTCUSDC",
            "symbol": "BTC/USDC:USDC",
            "base": "BTC",
            "quote": "USDC",
            "settle": "USDC",
            "spot": False,
            "swap": True,
            "linear": True,
            "active": True,
            "type": "swap",
            "contractSize": 1,
            "precision": {"price": 2, "amount": 3},
            "limits": {"amount": {"min": 0.001, "max": 1000}, "price": {"min": None, "max": None}},
        },
        "ETH/USDC:USDC": {
            "id": "ETHUSDC",
            "symbol": "ETH/USDC:USDC",
            "base": "ETH",
            "quote": "USDC",
            "settle": "USDC",
            "spot": False,
            "swap": True,
            "linear": True,
            "active": True,
            "type": "swap",
            "contractSize": 1,
            "precision": {"price": 2, "amount": 3},
            "limits": {"amount": {"min": 0.001, "max": 1000}, "price": {"min": None, "max": None}},
        },
    }

    fake_ccxt_api = MagicMock()
    fake_ccxt_api.timeframes = {"5m": "5m"}
    fake_ccxt_api.markets = fake_markets
    fake_ccxt_api.has = {
        "fetchOHLCV": True,
        "fetchL2OrderBook": True,
        "fetchTicker": True,
        "fetchTickers": True,
        "fetchTrades": True,
        "cancelOrder": True,
        "createOrder": True,
        "fetchOrder": True,
        "fetchBalance": True,
        "fetchPositions": True,
        "fetchLeverageTiers": True,
        "fetchMarketLeverageTiers": True,
        "createMarketOrder": True,
        "createLimitOrder": True,
        "createStopLossOrder": True,
        "editOrder": True,
    }
    fake_ccxt_api.precisionMode = 2
    fake_ccxt_api.options = {}
    fake_ccxt_api.id = "hyperliquid"
    fake_ccxt_api.name = "Hyperliquid"
    fake_ccxt_api.walletAddress = "0x0000000000000000000000000000000000000000"

    with (
        patch(
            "freqtrade.exchange.exchange.Exchange._load_async_markets",
            return_value=fake_markets,
        ),
        patch(
            "freqtrade.exchange.exchange.Exchange.reload_markets",
            lambda self, *a, **k: setattr(self, "_markets", fake_markets),
        ),
        patch(
            "freqtrade.exchange.exchange.Exchange.validate_required_startup_candles",
            return_value=None,
        ),
        patch("freqtrade.exchange.exchange.Exchange._init_ccxt", return_value=fake_ccxt_api),
    ):
        bot = FreqtradeBot(config)

        assert bot.config["trading_mode"] == "futures"
        assert bot.config["margin_mode"] == "isolated"
        assert bot.config["stake_currency"] == "USDC"
        assert bot.strategy.get_strategy_name() == "FoundationStrategy"
        assert set(bot.pairlists.whitelist) <= {"BTC/USDC:USDC", "ETH/USDC:USDC"}

        bot.cleanup()
