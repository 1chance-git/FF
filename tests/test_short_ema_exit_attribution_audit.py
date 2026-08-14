"""Focused unit tests for `hermes.short_ema_exit_attribution_audit` (research-only)."""

from __future__ import annotations

import pandas as pd
import pytest

from hermes.short_ema_exit_attribution_audit import (
    PERSISTENT_KEYS,
    aggregate_group,
    classify_persistent_or_ordinary,
    diagnose_trade,
    find_first_invalidation,
    find_first_invalidation_index,
    reconstruct_trade_candles,
)
from hermes.trade_report import Trade


def _trade(
    pair="BTC/USDC:USDC", direction="SHORT", entry_time="2026-01-01T00:00:00Z",
    exit_time="2026-01-05T00:00:00Z", entry_price=100.0, profit_pct=5.0, profit_abs=50.0,
    duration_minutes=5760.0, exit_reason="exit_signal",
):
    return Trade(
        pair=pair, direction=direction, entry_time=entry_time, exit_time=exit_time,
        entry_price=entry_price, exit_price=95.0, enter_tag=None, exit_reason=exit_reason,
        profit_abs=profit_abs, profit_pct=profit_pct, duration_minutes=duration_minutes, is_open=False,
    )


def _candles(rows):
    return pd.DataFrame(
        [{"date": pd.Timestamp(d, tz="UTC"), "open": o, "high": h, "low": l, "close": c} for d, o, h, l, c in rows]
    )


def _flat_series(n, start="2026-01-01T00:00", freq="4h", close=90.0):
    dates = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({
        "date": dates, "open": [close] * n, "high": [close + 1] * n,
        "low": [close - 1] * n, "close": [close] * n,
    })


# ---------------------------------------------------------------------------
# classify_persistent_or_ordinary
# ---------------------------------------------------------------------------


def test_classify_persistent_exact_match():
    trade = _trade(pair="ETH/USDC:USDC", entry_time="2026-01-20 16:00:00+00:00")
    assert classify_persistent_or_ordinary(trade) == "PERSISTENT"


def test_classify_ordinary_when_no_match():
    trade = _trade(pair="ETH/USDC:USDC", entry_time="2026-02-01 00:00:00+00:00")
    assert classify_persistent_or_ordinary(trade) == "ORDINARY"


def test_classify_wrong_pair_same_time_is_ordinary():
    trade = _trade(pair="BTC/USDC:USDC", entry_time="2026-01-20 16:00:00+00:00")  # ETH's time, BTC pair
    assert classify_persistent_or_ordinary(trade) == "ORDINARY"


def test_persistent_keys_has_exactly_three():
    assert len(PERSISTENT_KEYS) == 3


# ---------------------------------------------------------------------------
# reconstruct_trade_candles
# ---------------------------------------------------------------------------


def test_reconstruct_trade_candles_none_ohlcv():
    assert reconstruct_trade_candles(None, "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", 100.0, "SHORT") == []


def test_reconstruct_trade_candles_basic_short():
    candles = _flat_series(20, close=90.0)
    result = reconstruct_trade_candles(candles, "2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z", 100.0, "SHORT")
    assert len(result) > 0
    assert all(r.date >= pd.Timestamp("2026-01-01T00:00", tz="UTC") for r in result)
    assert all(r.date <= pd.Timestamp("2026-01-03T00:00", tz="UTC") for r in result)


def test_reconstruct_trade_candles_never_uses_candles_after_exit():
    candles = _flat_series(50, close=90.0)
    result = reconstruct_trade_candles(candles, "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", 100.0, "SHORT")
    assert result[-1].date <= pd.Timestamp("2026-01-02T00:00", tz="UTC")


def test_reconstruct_trade_candles_cumulative_mfe_mae_short():
    rows = [
        ("2026-01-01T00:00", 100, 102, 96, 100),
        ("2026-01-01T04:00", 100, 103, 90, 91),
        ("2026-01-01T08:00", 91, 105, 88, 92),  # high spikes but shouldn't reduce cumulative MFE
    ]
    candles = _candles(rows)
    result = reconstruct_trade_candles(candles, "2026-01-01T00:00:00Z", "2026-01-01T08:00:00Z", 100.0, "SHORT")
    assert len(result) == 3
    assert result[0].cumulative_mfe_pct == pytest.approx(4.0)  # (100-96)/100
    assert result[1].cumulative_mfe_pct == pytest.approx(10.0)  # (100-90)/100
    assert result[2].cumulative_mfe_pct == pytest.approx(12.0)  # (100-88)/100, monotonic
    assert result[2].cumulative_mae_pct >= result[1].cumulative_mae_pct  # never decreases


def test_reconstruct_trade_candles_days_since_entry():
    candles = _flat_series(10, close=90.0)
    result = reconstruct_trade_candles(candles, "2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z", 100.0, "SHORT")
    assert result[0].days_since_entry == pytest.approx(0.0)
    assert result[-1].days_since_entry > 0


def test_reconstruct_trade_candles_exit_before_entry_returns_empty():
    candles = _flat_series(10)
    assert reconstruct_trade_candles(candles, "2026-01-05T00:00:00Z", "2026-01-01T00:00:00Z", 100.0, "SHORT") == []


# ---------------------------------------------------------------------------
# find_first_invalidation / find_first_invalidation_index
# ---------------------------------------------------------------------------


def test_find_first_invalidation_short_price_rises_above_ema():
    # Entry at 100, price stays below a flat EMA (~90) for a while then rises above it.
    rows = [("2026-01-01T00:00", 90, 91, 89, 90)] * 250  # warm up EMA at ~90
    rows += [("2026-02-01T00:00", 95, 96, 94, 95)]  # close 95 > ema ~90 -> invalidated
    dates = pd.date_range("2026-01-01T00:00", periods=len(rows), freq="4h", tz="UTC")
    candles = pd.DataFrame([
        {"date": d, "open": r[1], "high": r[2], "low": r[3], "close": r[4]} for d, r in zip(dates, rows)
    ])
    result = reconstruct_trade_candles(candles, dates[0], dates[-1], 100.0, "SHORT")
    idx, invalidation = find_first_invalidation_index(result)
    assert invalidation is not None
    assert invalidation.exit_condition_true is True
    assert invalidation.date == dates[-1]


def test_find_first_invalidation_none_when_never_invalidated():
    candles = _flat_series(300, close=90.0)  # price always below ema, never invalidates
    result = reconstruct_trade_candles(candles, "2026-01-01T00:00:00Z", None, 100.0, "SHORT")
    result = result[:250]  # trim so exit_condition_true is well-defined (post warmup) and stays False
    invalidation = find_first_invalidation(result)
    assert invalidation is None


def test_find_first_invalidation_on_exit_candle_itself():
    dates = pd.date_range("2026-01-01T00:00", periods=251, freq="4h", tz="UTC")
    closes = [90.0] * 250 + [95.0]
    candles = pd.DataFrame({
        "date": dates, "open": closes, "high": [c + 1 for c in closes],
        "low": [c - 1 for c in closes], "close": closes,
    })
    result = reconstruct_trade_candles(candles, dates[0], dates[-1], 100.0, "SHORT")
    idx, invalidation = find_first_invalidation_index(result)
    assert idx == len(result) - 1  # invalidation is the very last (exit) candle


# ---------------------------------------------------------------------------
# diagnose_trade
# ---------------------------------------------------------------------------


def test_diagnose_trade_empty_candles():
    trade = _trade()
    diagnosis = diagnose_trade(trade, [])
    assert diagnosis.n_candles_reconstructed == 0
    assert diagnosis.first_invalidation_time is None


def test_diagnose_trade_no_invalidation_found():
    candles = _flat_series(300, close=90.0)
    trade = _trade(entry_time="2026-01-01T00:00:00Z", exit_time=str(candles["date"].iloc[100]))
    reconstructed = reconstruct_trade_candles(candles, trade.entry_time, trade.exit_time, 100.0, "SHORT")
    diagnosis = diagnose_trade(trade, reconstructed)
    assert diagnosis.first_invalidation_time is None
    assert diagnosis.hours_invalidation_to_exit is None


def test_diagnose_trade_group_classification_flows_through():
    trade = _trade(pair="ETH/USDC:USDC", entry_time="2026-01-20 16:00:00+00:00")
    diagnosis = diagnose_trade(trade, [])
    assert diagnosis.group == "PERSISTENT"


# ---------------------------------------------------------------------------
# aggregate_group
# ---------------------------------------------------------------------------


def test_aggregate_group_thin_sample_flag():
    diagnoses = [diagnose_trade(_trade(), []) for _ in range(3)]
    agg = aggregate_group(diagnoses)
    assert agg.n == 3
    assert agg.is_thin_sample is True


def test_aggregate_group_empty():
    agg = aggregate_group([])
    assert agg.n == 0
    assert agg.n_with_invalidation_found == 0
    assert agg.mean_hours_invalidation_to_exit is None
