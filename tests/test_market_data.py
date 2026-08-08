"""Unit tests for stat_arb.data.market_data.

Processing functions (``clean_and_fill``, ``validate_ohlcv``,
``align_pairs``) are tested with synthetic in-memory dataframes and touch
neither the filesystem nor the network. ``MarketDataLoader`` /
``MarketDataService`` tests write small candle files to a temporary
directory using Freqtrade's own data handler, so they exercise real
loading code without depending on any pre-downloaded dataset or exchange
connectivity.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.unit
from freqtrade.data.history.datahandlers import get_datahandler
from freqtrade.enums import CandleType

from stat_arb.data.market_data import (
    MarketDataError,
    MarketDataLoader,
    MarketDataService,
    align_pairs,
    clean_and_fill,
    validate_ohlcv,
)

TIMEFRAME = "5m"


def make_ohlcv(
    start: str = "2024-01-01 00:00:00",
    periods: int = 10,
    freq: str = "5min",
    price: float = 100.0,
) -> pd.DataFrame:
    """Build a clean, gap-free synthetic OHLCV dataframe for tests."""
    dates = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "date": dates,
            "open": [price + i for i in range(periods)],
            "high": [price + i + 1 for i in range(periods)],
            "low": [price + i - 1 for i in range(periods)],
            "close": [price + i + 0.5 for i in range(periods)],
            "volume": [10.0 + i for i in range(periods)],
        }
    )


# ---------------------------------------------------------------------------
# validate_ohlcv
# ---------------------------------------------------------------------------


def test_validate_ohlcv_passes_for_clean_data() -> None:
    df = make_ohlcv()
    validate_ohlcv(df, "BTC/USDC:USDC")  # must not raise


def test_validate_ohlcv_detects_missing_columns() -> None:
    df = make_ohlcv().drop(columns=["volume"])
    with pytest.raises(MarketDataError, match="missing required column"):
        validate_ohlcv(df, "BTC/USDC:USDC")


def test_validate_ohlcv_detects_empty_dataframe() -> None:
    df = make_ohlcv(periods=0)
    with pytest.raises(MarketDataError, match="empty"):
        validate_ohlcv(df, "BTC/USDC:USDC")


def test_validate_ohlcv_detects_duplicate_timestamps() -> None:
    df = make_ohlcv()
    df.loc[1, "date"] = df.loc[0, "date"]
    with pytest.raises(MarketDataError, match="duplicate timestamp"):
        validate_ohlcv(df, "BTC/USDC:USDC")


def test_validate_ohlcv_detects_unsorted_dates() -> None:
    df = make_ohlcv()
    df = df.iloc[::-1].reset_index(drop=True)
    with pytest.raises(MarketDataError, match="not sorted"):
        validate_ohlcv(df, "BTC/USDC:USDC")


def test_validate_ohlcv_detects_high_below_low() -> None:
    df = make_ohlcv()
    df.loc[2, "high"] = df.loc[2, "low"] - 5
    with pytest.raises(MarketDataError, match="high < low"):
        validate_ohlcv(df, "BTC/USDC:USDC")


def test_validate_ohlcv_detects_high_below_close() -> None:
    df = make_ohlcv()
    df.loc[2, "high"] = df.loc[2, "close"] - 100
    with pytest.raises(MarketDataError, match="high < open or high < close"):
        validate_ohlcv(df, "BTC/USDC:USDC")


def test_validate_ohlcv_detects_low_above_open() -> None:
    df = make_ohlcv()
    df.loc[2, "low"] = df.loc[2, "open"] + 100
    with pytest.raises(MarketDataError, match="low > open or low > close"):
        validate_ohlcv(df, "BTC/USDC:USDC")


def test_validate_ohlcv_detects_negative_volume() -> None:
    df = make_ohlcv()
    df.loc[3, "volume"] = -1.0
    with pytest.raises(MarketDataError, match="negative value"):
        validate_ohlcv(df, "BTC/USDC:USDC")


def test_validate_ohlcv_detects_non_positive_price() -> None:
    df = make_ohlcv()
    df.loc[4, "close"] = 0.0
    with pytest.raises(MarketDataError, match="non-positive price"):
        validate_ohlcv(df, "BTC/USDC:USDC")


def test_validate_ohlcv_reports_multiple_issues_together() -> None:
    df = make_ohlcv()
    df.loc[1, "date"] = df.loc[0, "date"]
    df.loc[3, "volume"] = -1.0
    with pytest.raises(MarketDataError) as exc_info:
        validate_ohlcv(df, "BTC/USDC:USDC")
    message = str(exc_info.value)
    assert "duplicate timestamp" in message
    assert "negative value" in message


# ---------------------------------------------------------------------------
# clean_and_fill
# ---------------------------------------------------------------------------


def test_clean_and_fill_fills_missing_candle() -> None:
    df = make_ohlcv(periods=10)
    # Remove the candle at index 5, creating a single-candle gap.
    gapped = df.drop(index=5).reset_index(drop=True)
    assert len(gapped) == 9

    filled = clean_and_fill(gapped, TIMEFRAME, "BTC/USDC:USDC")

    assert len(filled) == 10
    validate_ohlcv(filled, "BTC/USDC:USDC")

    # The filled candle should carry the previous close forward with zero volume.
    filled_row = filled.iloc[5]
    prev_close = filled.iloc[4]["close"]
    assert filled_row["open"] == prev_close
    assert filled_row["high"] == prev_close
    assert filled_row["low"] == prev_close
    assert filled_row["close"] == prev_close
    assert filled_row["volume"] == 0


def test_clean_and_fill_deduplicates_timestamps() -> None:
    df = make_ohlcv(periods=5)
    duplicated = pd.concat([df, df.iloc[[2]]], ignore_index=True)
    assert len(duplicated) == 6

    cleaned = clean_and_fill(duplicated, TIMEFRAME, "BTC/USDC:USDC")

    assert len(cleaned) == 5
    validate_ohlcv(cleaned, "BTC/USDC:USDC")


def test_clean_and_fill_can_drop_incomplete_last_candle() -> None:
    df = make_ohlcv(periods=5)
    result = clean_and_fill(df, TIMEFRAME, "BTC/USDC:USDC", drop_incomplete=True)
    assert len(result) == 4


# ---------------------------------------------------------------------------
# align_pairs
# ---------------------------------------------------------------------------


def test_align_pairs_intersects_timestamps() -> None:
    btc = make_ohlcv(periods=10)
    eth = make_ohlcv(periods=10).drop(index=[0, 9]).reset_index(drop=True)  # missing first/last

    aligned = align_pairs({"BTC/USDC:USDC": btc, "ETH/USDC:USDC": eth})

    assert set(aligned) == {"BTC/USDC:USDC", "ETH/USDC:USDC"}
    assert len(aligned["BTC/USDC:USDC"]) == 8
    assert len(aligned["ETH/USDC:USDC"]) == 8
    assert list(aligned["BTC/USDC:USDC"]["date"]) == list(aligned["ETH/USDC:USDC"]["date"])


def test_align_pairs_single_pair_passthrough() -> None:
    btc = make_ohlcv(periods=5)
    aligned = align_pairs({"BTC/USDC:USDC": btc})
    assert len(aligned["BTC/USDC:USDC"]) == 5


def test_align_pairs_raises_on_empty_input() -> None:
    with pytest.raises(MarketDataError, match="no pairs to align"):
        align_pairs({})


def test_align_pairs_raises_on_no_overlap() -> None:
    btc = make_ohlcv(start="2024-01-01 00:00:00", periods=5)
    eth = make_ohlcv(start="2024-06-01 00:00:00", periods=5)
    with pytest.raises(MarketDataError, match="No overlapping timestamps"):
        align_pairs({"BTC/USDC:USDC": btc, "ETH/USDC:USDC": eth})


# ---------------------------------------------------------------------------
# MarketDataLoader / MarketDataService (write via Freqtrade's data handler)
# ---------------------------------------------------------------------------


def _store(datadir: Path, pair: str, dataframe: pd.DataFrame) -> None:
    handler = get_datahandler(datadir, data_format="feather")
    handler.ohlcv_store(pair, TIMEFRAME, dataframe, CandleType.FUTURES)


def test_loader_raises_when_no_data_available(tmp_path: Path) -> None:
    loader = MarketDataLoader(tmp_path, TIMEFRAME, candle_type=CandleType.FUTURES)
    with pytest.raises(MarketDataError, match="No local candle data found"):
        loader.load_pair("BTC/USDC:USDC")


def test_loader_loads_stored_data(tmp_path: Path) -> None:
    _store(tmp_path, "BTC/USDC:USDC", make_ohlcv(periods=10))
    loader = MarketDataLoader(tmp_path, TIMEFRAME, candle_type=CandleType.FUTURES)

    dataframe = loader.load_pair("BTC/USDC:USDC")

    assert len(dataframe) == 10
    assert list(dataframe.columns[:6]) == ["date", "open", "high", "low", "close", "volume"]


def test_loader_loads_multiple_pairs(tmp_path: Path) -> None:
    _store(tmp_path, "BTC/USDC:USDC", make_ohlcv(periods=10))
    _store(tmp_path, "ETH/USDC:USDC", make_ohlcv(periods=10))
    loader = MarketDataLoader(tmp_path, TIMEFRAME, candle_type=CandleType.FUTURES)

    data = loader.load_pairs(["BTC/USDC:USDC", "ETH/USDC:USDC"])

    assert set(data) == {"BTC/USDC:USDC", "ETH/USDC:USDC"}
    assert len(data["BTC/USDC:USDC"]) == 10
    assert len(data["ETH/USDC:USDC"]) == 10


def test_market_data_service_full_pipeline(tmp_path: Path) -> None:
    btc = make_ohlcv(periods=20).drop(index=5).reset_index(drop=True)  # internal gap
    eth = make_ohlcv(periods=20).drop(index=[0, 19]).reset_index(drop=True)  # edge gaps

    _store(tmp_path, "BTC/USDC:USDC", btc)
    _store(tmp_path, "ETH/USDC:USDC", eth)

    loader = MarketDataLoader(tmp_path, TIMEFRAME, candle_type=CandleType.FUTURES)
    service = MarketDataService(loader)

    result = service.get_aligned_market_data(["BTC/USDC:USDC", "ETH/USDC:USDC"])

    assert set(result) == {"BTC/USDC:USDC", "ETH/USDC:USDC"}
    # BTC's internal gap is filled by clean_and_fill, so after alignment
    # only ETH's edge trimming should reduce the shared candle count.
    assert len(result["BTC/USDC:USDC"]) == len(result["ETH/USDC:USDC"]) == 18
    assert list(result["BTC/USDC:USDC"]["date"]) == list(result["ETH/USDC:USDC"]["date"])
    for pair, dataframe in result.items():
        validate_ohlcv(dataframe, pair)


def test_market_data_service_raises_on_missing_pair(tmp_path: Path) -> None:
    _store(tmp_path, "BTC/USDC:USDC", make_ohlcv(periods=10))
    loader = MarketDataLoader(tmp_path, TIMEFRAME, candle_type=CandleType.FUTURES)
    service = MarketDataService(loader)

    with pytest.raises(MarketDataError, match="No local candle data found"):
        service.get_aligned_market_data(["BTC/USDC:USDC", "ETH/USDC:USDC"])
