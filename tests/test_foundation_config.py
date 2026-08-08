"""Unit tests for the project foundation configuration and skeleton strategy.

These tests do not touch the network or an exchange. They verify that:

* ``user_data/config.json`` is valid JSON and encodes the required
  trading-mode / margin-mode / stake-currency / pairlist decisions.
* The skeleton strategy module is importable and satisfies the minimal
  ``IStrategy`` contract, so ``freqtrade trade`` / ``freqtrade backtesting``
  can load it without error.
"""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "user_data" / "config.json"
STRATEGY_PATH = REPO_ROOT / "user_data" / "strategies" / "FoundationStrategy.py"


@pytest.fixture(scope="module")
def config() -> dict:
    """Load and parse the foundation config.json."""
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def test_config_file_exists() -> None:
    """The base config must exist at the conventional Freqtrade location."""
    assert CONFIG_PATH.is_file()


def test_futures_isolated_usdc(config: dict) -> None:
    """Trading mode, margin mode, and stake currency must match requirements."""
    assert config["trading_mode"] == "futures"
    assert config["margin_mode"] == "isolated"
    assert config["stake_currency"] == "USDC"


def test_static_pairlist_restricted_to_btc_eth_usdc(config: dict) -> None:
    """Only BTC/USDC and ETH/USDC futures pairs should be tradable."""
    methods = [entry["method"] for entry in config["pairlists"]]
    assert methods == ["StaticPairList"]

    whitelist = set(config["exchange"]["pair_whitelist"])
    assert whitelist == {"BTC/USDC:USDC", "ETH/USDC:USDC"}


def test_dry_run_enabled_by_default(config: dict) -> None:
    """The foundation config must default to dry-run for safety."""
    assert config["dry_run"] is True


def test_log_config_present(config: dict) -> None:
    """A dictConfig-style log_config with console and rotating file handlers."""
    log_config = config["log_config"]
    assert set(log_config["handlers"]) == {"console", "file"}
    assert log_config["handlers"]["file"]["class"] == "logging.handlers.RotatingFileHandler"


def test_strategy_module_importable() -> None:
    """FoundationStrategy must import and subclass IStrategy with no-op hooks."""
    import importlib.util

    from freqtrade.strategy import IStrategy

    spec = importlib.util.spec_from_file_location("FoundationStrategy", STRATEGY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    strategy_cls = module.FoundationStrategy
    assert issubclass(strategy_cls, IStrategy)
    assert strategy_cls.can_short is True
    assert strategy_cls.INTERFACE_VERSION == 3
