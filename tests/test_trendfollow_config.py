"""Compatibility tests for `user_data/config-research-trendfollow.json`.

TrendFollowCore's frozen trading universe is BTC+ETH only (SOL was
excluded after a BTC+ETH-vs-SOL isolation audit over the frozen
39-trade baseline; see TrendFollowCore.py's module docstring for the
rationale). This test module guards that decision the same way
`test_foundation_config.py` guards `config.json`'s pair whitelist --
so a future edit can't silently reintroduce SOL or drift
`max_open_trades` out of sync with the whitelist without a test
failing.
"""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "user_data" / "config-research-trendfollow.json"


@pytest.fixture(scope="module")
def config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def test_config_file_exists() -> None:
    assert CONFIG_PATH.is_file()


def test_static_pairlist_restricted_to_btc_eth(config: dict) -> None:
    """Only BTC/USDC:USDC and ETH/USDC:USDC should be tradable -- SOL was
    excluded from the frozen universe, not merely deprioritized."""
    methods = [entry["method"] for entry in config["pairlists"]]
    assert methods == ["StaticPairList"]

    whitelist = set(config["exchange"]["pair_whitelist"])
    assert whitelist == {"BTC/USDC:USDC", "ETH/USDC:USDC"}


def test_max_open_trades_matches_pair_whitelist_size(config: dict) -> None:
    """`max_open_trades` must track the whitelist size -- it was 3 when
    the universe was BTC+ETH+SOL and must be 2 now that SOL is excluded,
    so the bot can't silently under- or over-allocate concurrent slots."""
    assert config["max_open_trades"] == len(config["exchange"]["pair_whitelist"])


def test_strategy_is_trendfollow_core(config: dict) -> None:
    """This config must stay pointed at TrendFollowCore, not drift to
    another strategy without a corresponding pair-whitelist review."""
    assert config["strategy"] == "TrendFollowCore"


def test_dry_run_enabled(config: dict) -> None:
    """Research config must never place real orders."""
    assert config["dry_run"] is True
