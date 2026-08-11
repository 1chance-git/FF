"""Tests for research/hyperliquid_ohlcv.py.

All network access is mocked (via a fake ``fetch`` callable, or by
patching ``urllib.request.urlopen`` for the raw-HTTP-layer tests) --
this module never makes a real network call in the test suite.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from research.hyperliquid_ohlcv import (
    COINS,
    INTERVALS,
    PAIR_FOR_COIN,
    HyperliquidAPIError,
    dedupe_and_sort,
    default_window_for,
    fetch_candle_snapshot,
    interval_to_ms,
    paginate_candles,
    paginate_raw,
    run_pipeline_matrix,
    to_dataframe,
    validate_ohlcv,
)


def make_candle(t: int, interval_ms: int, *, o=100.0, h=101.0, l=99.0, c=100.5, v=10.0):
    return {
        "t": t,
        "T": t + interval_ms - 1,
        "s": "BTC",
        "i": "5m",
        "o": str(o),
        "h": str(h),
        "l": str(l),
        "c": str(c),
        "v": str(v),
        "n": 42,
    }


FIVE_MIN_MS = 5 * 60_000


class TestIntervalToMs:
    def test_known_intervals(self):
        assert interval_to_ms("5m") == 5 * 60_000
        assert interval_to_ms("1h") == 60 * 60_000
        assert interval_to_ms("1d") == 24 * 60 * 60_000

    def test_unknown_interval_raises(self):
        with pytest.raises(ValueError, match="Unsupported interval"):
            interval_to_ms("3w")


class TestCandleConversion:
    def test_to_dataframe_maps_fields_correctly(self):
        candles = [make_candle(1_700_000_000_000, FIVE_MIN_MS, o=50000.0, h=50100.0, l=49900.0, c=50050.0, v=12.5)]
        df = to_dataframe(candles)

        assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
        assert len(df) == 1
        row = df.iloc[0]
        assert row["open"] == 50000.0
        assert row["high"] == 50100.0
        assert row["low"] == 49900.0
        assert row["close"] == 50050.0
        assert row["volume"] == 12.5
        assert row["date"] == pd.Timestamp(1_700_000_000_000, unit="ms", tz="UTC")

    def test_to_dataframe_casts_string_fields_to_float(self):
        candles = [make_candle(0, FIVE_MIN_MS, o="123.456")]
        df = to_dataframe(candles)
        assert df.iloc[0]["open"] == pytest.approx(123.456)
        assert isinstance(df.iloc[0]["open"], float)

    def test_to_dataframe_empty_input(self):
        df = to_dataframe([])
        assert df.empty
        assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]


class TestDedupeAndSort:
    def test_removes_duplicate_timestamps(self):
        candles = [make_candle(1000, FIVE_MIN_MS), make_candle(1000, FIVE_MIN_MS)]
        result = dedupe_and_sort(candles)
        assert len(result) == 1

    def test_sorts_chronologically(self):
        candles = [make_candle(3000, FIVE_MIN_MS), make_candle(1000, FIVE_MIN_MS), make_candle(2000, FIVE_MIN_MS)]
        result = dedupe_and_sort(candles)
        assert [c["t"] for c in result] == [1000, 2000, 3000]

    def test_dedup_across_overlapping_pages(self):
        # Simulates two adjacent pagination pages sharing one boundary candle.
        page_a = [make_candle(t, FIVE_MIN_MS) for t in (0, FIVE_MIN_MS, 2 * FIVE_MIN_MS)]
        page_b = [make_candle(t, FIVE_MIN_MS) for t in (2 * FIVE_MIN_MS, 3 * FIVE_MIN_MS)]
        result = dedupe_and_sort(page_a + page_b)
        assert [c["t"] for c in result] == [0, FIVE_MIN_MS, 2 * FIVE_MIN_MS, 3 * FIVE_MIN_MS]

    def test_empty_input(self):
        assert dedupe_and_sort([]) == []


class TestValidateOhlcv:
    def _valid_df(self, n=5, start=0):
        candles = [make_candle(start + i * FIVE_MIN_MS, FIVE_MIN_MS) for i in range(n)]
        return to_dataframe(candles)

    def test_valid_data_passes(self):
        result = validate_ohlcv(self._valid_df(), "5m")
        assert result.ok is True
        assert result.issues == []

    def test_empty_dataframe_fails(self):
        result = validate_ohlcv(to_dataframe([]), "5m")
        assert result.ok is False
        assert "empty" in result.issues[0]

    def test_detects_duplicate_timestamps(self):
        candles = [make_candle(0, FIVE_MIN_MS), make_candle(0, FIVE_MIN_MS)]
        df = pd.concat([to_dataframe([candles[0]]), to_dataframe([candles[1]])], ignore_index=True)
        result = validate_ohlcv(df, "5m")
        assert result.ok is False
        assert any("duplicate timestamp" in issue for issue in result.issues)

    def test_detects_non_chronological_order(self):
        df = self._valid_df(n=3)
        df = df.iloc[::-1].reset_index(drop=True)  # reverse order
        result = validate_ohlcv(df, "5m")
        assert result.ok is False
        assert any("chronological" in issue for issue in result.issues)

    def test_detects_invalid_ohlc_relationship(self):
        candle = make_candle(0, FIVE_MIN_MS, o=100.0, h=90.0, l=99.0, c=100.5)  # high < low
        df = to_dataframe([candle])
        result = validate_ohlcv(df, "5m")
        assert result.ok is False
        assert any("invalid OHLC" in issue for issue in result.issues)

    def test_detects_negative_volume(self):
        candle = make_candle(0, FIVE_MIN_MS, v=-5.0)
        df = to_dataframe([candle])
        result = validate_ohlcv(df, "5m")
        assert result.ok is False
        assert any("negative volume" in issue for issue in result.issues)

    def test_detects_wrong_spacing(self):
        candles = [make_candle(0, FIVE_MIN_MS), make_candle(FIVE_MIN_MS * 3, FIVE_MIN_MS)]  # gap, not 5m
        df = to_dataframe(candles)
        result = validate_ohlcv(df, "5m")
        assert result.ok is False
        assert any("spacing" in issue for issue in result.issues)


class TestFetchCandleSnapshotMalformedResponses:
    def _mock_response(self, body: bytes, status: int = 200):
        cm = MagicMock()
        cm.__enter__.return_value = cm
        cm.status = status
        cm.read.return_value = body
        return cm

    def test_non_list_json_response_raises(self):
        with patch("research.hyperliquid_ohlcv.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response(json.dumps({"error": "bad coin"}).encode())
            with pytest.raises(HyperliquidAPIError, match="expected a JSON list"):
                fetch_candle_snapshot("BTC", "5m", 0, 1000)

    def test_invalid_json_raises(self):
        with patch("research.hyperliquid_ohlcv.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response(b"not json at all")
            with pytest.raises(HyperliquidAPIError, match="not valid JSON"):
                fetch_candle_snapshot("BTC", "5m", 0, 1000)

    def test_non_200_status_raises(self):
        with patch("research.hyperliquid_ohlcv.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response(b"[]", status=500)
            with pytest.raises(HyperliquidAPIError, match="unexpected HTTP status"):
                fetch_candle_snapshot("BTC", "5m", 0, 1000)

    def test_valid_list_response_passes_through(self):
        candles = [make_candle(0, FIVE_MIN_MS)]
        with patch("research.hyperliquid_ohlcv.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response(json.dumps(candles).encode())
            result = fetch_candle_snapshot("BTC", "5m", 0, 1000)
        assert result == candles


class TestPaginationBoundaryHandling:
    def test_single_request_covers_whole_range_by_default(self):
        candles = [make_candle(t, FIVE_MIN_MS) for t in range(0, 5 * FIVE_MIN_MS, FIVE_MIN_MS)]
        fetch = MagicMock(return_value=candles)
        result = paginate_raw("BTC", "5m", 0, 5 * FIVE_MIN_MS, fetch=fetch)
        assert result.requests_made == 1
        assert fetch.call_count == 1

    def test_small_window_forces_multiple_requests(self):
        # 5 candles total, window covers only 2 candles at a time -> 3 requests.
        all_candles = [make_candle(t, FIVE_MIN_MS) for t in range(0, 5 * FIVE_MIN_MS, FIVE_MIN_MS)]

        def fake_fetch(coin, interval, start_ms, end_ms):
            return [c for c in all_candles if start_ms <= c["t"] <= end_ms]

        result = paginate_raw(
            "BTC", "5m", 0, 5 * FIVE_MIN_MS, request_window_ms=2 * FIVE_MIN_MS, fetch=fake_fetch
        )
        assert result.requests_made >= 3
        deduped = dedupe_and_sort(result.raw_candles)
        assert [c["t"] for c in deduped] == [c["t"] for c in all_candles]

    def test_boundary_candles_overlap_and_get_deduped(self):
        # Adjacent windows both include the shared boundary timestamp,
        # exactly like Hyperliquid's real inclusive-boundary behavior
        # observed in the manual OHLCV audit (RAILWAY.md).
        all_candles = [make_candle(t, FIVE_MIN_MS) for t in range(0, 3 * FIVE_MIN_MS, FIVE_MIN_MS)]

        def fake_fetch(coin, interval, start_ms, end_ms):
            # inclusive on both ends, like the real API
            return [c for c in all_candles if start_ms <= c["t"] <= end_ms]

        result = paginate_candles(
            "BTC", "5m", 0, 3 * FIVE_MIN_MS - 1, request_window_ms=FIVE_MIN_MS, fetch=fake_fetch
        )
        timestamps = [c["t"] for c in result]
        assert timestamps == sorted(set(timestamps))  # no duplicates
        assert timestamps == [0, FIVE_MIN_MS, 2 * FIVE_MIN_MS]

    def test_empty_page_stops_pagination(self):
        fetch = MagicMock(return_value=[])
        result = paginate_raw("BTC", "5m", 0, 10 * FIVE_MIN_MS, request_window_ms=FIVE_MIN_MS, fetch=fetch)
        assert result.requests_made == 1
        assert result.raw_candles == []

    def test_exceeding_max_requests_raises(self):
        # fetch always makes only 1ms of progress per call, so covering
        # a large range takes far more requests than max_requests allows --
        # this must be caught explicitly, not loop forever.
        def fake_fetch(coin, interval, start_ms, end_ms):
            return [make_candle(end_ms - 1, FIVE_MIN_MS)]

        with pytest.raises(HyperliquidAPIError, match="exceeded max_requests"):
            paginate_raw(
                "BTC", "5m", 0, 100 * FIVE_MIN_MS,
                request_window_ms=FIVE_MIN_MS, max_requests=3, fetch=fake_fetch,
            )

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError, match="must be <"):
            paginate_raw("BTC", "5m", 1000, 1000, fetch=MagicMock())

    def test_non_positive_window_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            paginate_raw("BTC", "5m", 0, 1000, request_window_ms=0, fetch=MagicMock())


def _synthetic_fetch(coin, interval, start_ms, end_ms):
    """Generic fake ``fetch`` usable for any coin/interval: generates a
    real interval-spaced candle series covering [start_ms, end_ms]
    inclusive, matching Hyperliquid's real inclusive-boundary behavior
    (observed in the manual OHLCV audit, see RAILWAY.md)."""
    step = interval_to_ms(interval)
    first = ((start_ms + step - 1) // step) * step  # first grid point >= start_ms
    timestamps = []
    t = first
    while t <= end_ms:
        timestamps.append(t)
        t += step
    return [make_candle(t, step) for t in timestamps]


class TestDefaultWindowFor:
    def test_scales_with_interval(self):
        for interval in INTERVALS:
            start_ms, end_ms, request_window_ms = default_window_for("BTC", interval)
            step = interval_to_ms(interval)
            assert end_ms - start_ms == 13 * step
            assert request_window_ms == 5 * step
            assert start_ms < end_ms

    def test_custom_candles_per_page_and_total(self):
        start_ms, end_ms, request_window_ms = default_window_for(
            "BTC", "1h", candles_per_page=2, total_candles=6
        )
        step = interval_to_ms("1h")
        assert end_ms - start_ms == 6 * step
        assert request_window_ms == 2 * step


class TestMatrixConstants:
    def test_all_coins_have_a_pair_mapping(self):
        for coin in COINS:
            assert coin in PAIR_FOR_COIN
            assert "/" in PAIR_FOR_COIN[coin]

    def test_expected_scope(self):
        assert set(COINS) == {"BTC", "ETH", "SOL"}
        assert set(INTERVALS) == {"5m", "1h", "4h", "1d"}


class TestRunPipelineMatrix:
    def test_runs_every_combination(self, tmp_path):
        results = run_pipeline_matrix(
            coins=["BTC", "ETH"],
            intervals=["5m", "1h"],
            pair_for_coin={"BTC": "BTC/USDC:USDC", "ETH": "ETH/USDC:USDC"},
            datadir=tmp_path,
            fetch=_synthetic_fetch,
        )
        assert set(results.keys()) == {("BTC", "5m"), ("BTC", "1h"), ("ETH", "5m"), ("ETH", "1h")}

    def test_every_combination_passes_with_clean_synthetic_data(self, tmp_path):
        results = run_pipeline_matrix(
            coins=["BTC", "ETH", "SOL"],
            intervals=["5m", "1h", "4h", "1d"],
            pair_for_coin=PAIR_FOR_COIN,
            datadir=tmp_path,
            fetch=_synthetic_fetch,
        )
        assert len(results) == 12
        for (coin, interval), result in results.items():
            assert result.api_ok, f"{coin} {interval}: API call failed"
            assert result.download_ok, f"{coin} {interval}: no candles downloaded"
            assert result.validation.ok, f"{coin} {interval}: validation failed: {result.validation.issues}"
            assert result.freqtrade_format_ok, f"{coin} {interval}: Freqtrade write failed"
            assert result.freqtrade_readback_ok, f"{coin} {interval}: Freqtrade read-back failed"

    def test_each_pair_writes_to_a_distinct_file(self, tmp_path):
        results = run_pipeline_matrix(
            coins=["BTC", "ETH"],
            intervals=["5m", "1h"],
            pair_for_coin={"BTC": "BTC/USDC:USDC", "ETH": "ETH/USDC:USDC"},
            datadir=tmp_path,
            fetch=_synthetic_fetch,
        )
        output_paths = {result.output_path for result in results.values()}
        assert len(output_paths) == 4  # no accidental filename collisions across coins/intervals

    def test_malformed_response_fails_only_that_combination(self, tmp_path):
        def flaky_fetch(coin, interval, start_ms, end_ms):
            if coin == "ETH" and interval == "1h":
                raise HyperliquidAPIError("simulated exchange error")
            return _synthetic_fetch(coin, interval, start_ms, end_ms)

        results = run_pipeline_matrix(
            coins=["BTC", "ETH"],
            intervals=["5m", "1h"],
            pair_for_coin={"BTC": "BTC/USDC:USDC", "ETH": "ETH/USDC:USDC"},
            datadir=tmp_path,
            fetch=flaky_fetch,
        )
        assert results[("ETH", "1h")].api_ok is False
        assert results[("BTC", "5m")].api_ok is True
        assert results[("BTC", "1h")].api_ok is True
        assert results[("ETH", "5m")].api_ok is True
