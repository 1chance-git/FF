"""Focused unit tests for `hermes.short_trend_persistence_audit` (research-only)."""

from __future__ import annotations

import pandas as pd
import pytest

from hermes.short_trend_persistence_audit import (
    CHECKPOINTS,
    PERSISTENT_KEYS,
    THIN_SAMPLE_THRESHOLD,
    aggregate_group_checkpoint,
    all_checkpoint_snapshots,
    checkpoint_snapshot,
    classify_group,
    compute_trend_persistence,
    reconstruct_full_trade,
)
from hermes.trade_report import Trade


def _trade(
    pair="BTC/USDC:USDC", direction="SHORT", entry_time="2026-01-01T00:00:00Z",
    exit_time="2026-01-10T00:00:00Z", entry_price=100.0, profit_pct=5.0, profit_abs=50.0,
    duration_minutes=12960.0, exit_reason="exit_signal",
):
    return Trade(
        pair=pair, direction=direction, entry_time=entry_time, exit_time=exit_time,
        entry_price=entry_price, exit_price=95.0, enter_tag=None, exit_reason=exit_reason,
        profit_abs=profit_abs, profit_pct=profit_pct, duration_minutes=duration_minutes, is_open=False,
    )


def _flat_series(n, start="2026-01-01T00:00", freq="4h", close=90.0):
    dates = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({
        "date": dates, "open": [close] * n, "high": [close + 1] * n,
        "low": [close - 1] * n, "close": [close] * n,
    })


# ---------------------------------------------------------------------------
# classify_group
# ---------------------------------------------------------------------------


def test_classify_group_persistent():
    trade = _trade(pair="ETH/USDC:USDC", entry_time="2026-01-20 16:00:00+00:00", profit_abs=10.0)
    assert classify_group(trade) == "PERSISTENT"


def test_classify_group_ordinary_winner():
    trade = _trade(pair="ETH/USDC:USDC", entry_time="2026-02-01 00:00:00+00:00", profit_abs=10.0)
    assert classify_group(trade) == "ORDINARY"


def test_classify_group_loser():
    trade = _trade(profit_abs=-10.0)
    assert classify_group(trade) == "LOSER"


def test_persistent_keys_exactly_three():
    assert len(PERSISTENT_KEYS) == 3


# ---------------------------------------------------------------------------
# CHECKPOINTS constant
# ---------------------------------------------------------------------------


def test_checkpoints_include_all_required_labels():
    assert set(CHECKPOINTS.keys()) == {"4h", "12h", "24h", "48h", "3d", "7d", "14d", "21d", "30d", "45d"}
    assert CHECKPOINTS["45d"] == 45 * 24 * 60


# ---------------------------------------------------------------------------
# reconstruct_full_trade
# ---------------------------------------------------------------------------


def test_reconstruct_full_trade_none_ohlcv():
    assert reconstruct_full_trade(None, "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", 100.0, "SHORT") == []


def test_reconstruct_full_trade_never_uses_candles_after_exit():
    candles = _flat_series(50, close=90.0)
    result = reconstruct_full_trade(candles, "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", 100.0, "SHORT")
    assert result[-1].date <= pd.Timestamp("2026-01-02T00:00", tz="UTC")


def test_reconstruct_full_trade_short_aligned_below_ema():
    # Price starts at 110 (keeping the lagging EMA elevated) then drops
    # and holds at 90, well below the still-lagging EMA -- so SHORT stays
    # aligned (close < ema200) for the later candles.
    dates = pd.date_range("2026-01-01T00:00", periods=300, freq="4h", tz="UTC")
    closes = [110.0] * 50 + [90.0] * 250
    candles = pd.DataFrame({
        "date": dates, "open": closes, "high": [c + 1 for c in closes],
        "low": [c - 1 for c in closes], "close": closes,
    })
    result = reconstruct_full_trade(candles, dates[50], "2026-01-20T00:00:00Z", 100.0, "SHORT")
    late_candles = [c for c in result if c.days_since_entry > 5]
    assert len(late_candles) > 0
    assert all(c.structurally_aligned is True for c in late_candles)


def test_reconstruct_full_trade_cumulative_mfe_never_decreases():
    candles = _flat_series(20, close=90.0)
    result = reconstruct_full_trade(candles, "2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z", 100.0, "SHORT")
    mfes = [c.cumulative_mfe_pct for c in result]
    assert all(b >= a for a, b in zip(mfes, mfes[1:]))


def test_reconstruct_full_trade_adx_above_threshold_flag_type():
    candles = _flat_series(300, close=90.0)
    result = reconstruct_full_trade(candles, "2026-01-01T00:00:00Z", "2026-01-10T00:00:00Z", 100.0, "SHORT")
    for c in result:
        assert c.adx_above_threshold is None or isinstance(c.adx_above_threshold, bool)


# ---------------------------------------------------------------------------
# checkpoint_snapshot / all_checkpoint_snapshots
# ---------------------------------------------------------------------------


def test_checkpoint_snapshot_empty_input():
    snap = checkpoint_snapshot([], "4h", 240.0)
    assert snap.n_candles_in_subset == 0
    assert snap.closed_before_checkpoint is False


def test_checkpoint_snapshot_closed_before_flag():
    candles = _flat_series(300, close=90.0)
    trade_candles = reconstruct_full_trade(candles, "2026-01-01T00:00:00Z", "2026-01-01T08:00:00Z", 100.0, "SHORT")
    snap = checkpoint_snapshot(trade_candles, "7d", 10080.0)
    assert snap.closed_before_checkpoint is True
    assert snap.n_candles_in_subset == len(trade_candles)


def test_checkpoint_snapshot_not_closed_before_when_still_running():
    candles = _flat_series(300, close=90.0)
    trade_candles = reconstruct_full_trade(candles, "2026-01-01T00:00:00Z", None, 100.0, "SHORT")
    snap = checkpoint_snapshot(trade_candles, "4h", 240.0)
    assert snap.closed_before_checkpoint is False


def test_all_checkpoint_snapshots_returns_every_label():
    candles = _flat_series(300, close=90.0)
    trade_candles = reconstruct_full_trade(candles, "2026-01-01T00:00:00Z", "2026-01-10T00:00:00Z", 100.0, "SHORT")
    snapshots = all_checkpoint_snapshots(trade_candles)
    assert set(snapshots.keys()) == set(CHECKPOINTS.keys())


# ---------------------------------------------------------------------------
# compute_trend_persistence
# ---------------------------------------------------------------------------


def test_compute_trend_persistence_all_aligned():
    dates = pd.date_range("2026-01-01T00:00", periods=300, freq="4h", tz="UTC")
    closes = [110.0] * 50 + [90.0] * 250
    candles = pd.DataFrame({
        "date": dates, "open": closes, "high": [c + 1 for c in closes],
        "low": [c - 1 for c in closes], "close": closes,
    })
    trade_candles = reconstruct_full_trade(candles, dates[60], "2026-01-20T00:00:00Z", 100.0, "SHORT")
    stats = compute_trend_persistence(trade_candles, "SHORT")
    assert stats.pct_aligned == pytest.approx(100.0)
    assert stats.longest_run_aligned == len(trade_candles)


def test_compute_trend_persistence_empty():
    stats = compute_trend_persistence([], "SHORT")
    assert stats.pct_aligned is None
    assert stats.longest_run_aligned == 0


def test_compute_trend_persistence_mixed_alignment():
    from hermes.short_trend_persistence_audit import TradeCandle
    import pandas as pd
    base = pd.Timestamp("2026-01-01", tz="UTC")
    candles = [
        TradeCandle(date=base, close=1, ema200=1, ema_distance_pct=0, adx=30, donchian_breakout_pct=0,
                    realized_vol_pct=None, price_change_pct=0, cumulative_mfe_pct=0, cumulative_mae_pct=0,
                    structurally_aligned=True, adx_above_threshold=True, days_since_entry=0),
        TradeCandle(date=base, close=1, ema200=1, ema_distance_pct=0, adx=30, donchian_breakout_pct=0,
                    realized_vol_pct=None, price_change_pct=0, cumulative_mfe_pct=0, cumulative_mae_pct=0,
                    structurally_aligned=False, adx_above_threshold=True, days_since_entry=1),
        TradeCandle(date=base, close=1, ema200=1, ema_distance_pct=0, adx=30, donchian_breakout_pct=0,
                    realized_vol_pct=None, price_change_pct=0, cumulative_mfe_pct=0, cumulative_mae_pct=0,
                    structurally_aligned=True, adx_above_threshold=True, days_since_entry=2),
        TradeCandle(date=base, close=1, ema200=1, ema_distance_pct=0, adx=30, donchian_breakout_pct=0,
                    realized_vol_pct=None, price_change_pct=0, cumulative_mfe_pct=0, cumulative_mae_pct=0,
                    structurally_aligned=True, adx_above_threshold=True, days_since_entry=3),
    ]
    stats = compute_trend_persistence(candles, "SHORT")
    assert stats.pct_aligned == pytest.approx(75.0)
    assert stats.longest_run_aligned == 2  # last two candles


# ---------------------------------------------------------------------------
# aggregate_group_checkpoint
# ---------------------------------------------------------------------------


def test_aggregate_group_checkpoint_thin_sample():
    candles = _flat_series(300, close=90.0)
    trades = [_trade(entry_time=f"2026-01-{i:02d}T00:00:00Z", exit_time=f"2026-01-{i+8:02d}T00:00:00Z") for i in range(1, 4)]
    per_trade = [all_checkpoint_snapshots(reconstruct_full_trade(candles, t.entry_time, t.exit_time, 100.0, "SHORT")) for t in trades]
    agg = aggregate_group_checkpoint(per_trade, "4h")
    assert agg.n_total == 3
    assert agg.is_thin_sample is True
    assert THIN_SAMPLE_THRESHOLD == 5


def test_aggregate_group_checkpoint_empty():
    agg = aggregate_group_checkpoint([], "4h")
    assert agg.n_total == 0
    assert agg.n_reached == 0
    assert agg.mean_ema_distance_pct is None
