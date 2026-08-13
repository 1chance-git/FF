"""Focused unit tests for `hermes.short_runner_lifecycle_audit` (research-only)."""

from __future__ import annotations

import pandas as pd
import pytest

from hermes.short_runner_lifecycle_audit import (
    CHECKPOINTS,
    THIN_SAMPLE_THRESHOLD,
    aggregate_all_checkpoints,
    aggregate_checkpoint,
    compute_checkpoint_metrics,
    compute_checkpoint_series,
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


def _hourly_candles(n, start="2026-01-01T00:00", freq="4h", base=100.0, high_add=1.0, low_sub=1.0):
    dates = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({
        "date": dates,
        "open": [base] * n,
        "high": [base + high_add] * n,
        "low": [base - low_sub] * n,
        "close": [base] * n,
    })


# ---------------------------------------------------------------------------
# CHECKPOINTS constant
# ---------------------------------------------------------------------------


def test_checkpoints_fixed_set():
    assert CHECKPOINTS == {"4h": 240.0, "12h": 720.0, "24h": 1440.0, "48h": 2880.0, "3d": 4320.0, "7d": 10080.0}


# ---------------------------------------------------------------------------
# compute_checkpoint_metrics
# ---------------------------------------------------------------------------


def test_compute_checkpoint_metrics_none_ohlcv():
    result = compute_checkpoint_metrics(None, "2026-01-01T00:00:00Z", "2026-01-05T00:00:00Z", 100.0, "SHORT", "4h", 240.0)
    assert result.n_candles_in_window == 0
    assert result.mfe_pct is None


def test_compute_checkpoint_metrics_short_favorable_uses_low():
    candles = _hourly_candles(300, base=100.0, high_add=2.0, low_sub=5.0)
    result = compute_checkpoint_metrics(
        candles, "2026-01-01T00:00:00Z", "2026-01-10T00:00:00Z", 100.0, "SHORT", "4h", 240.0
    )
    assert result.n_candles_in_window >= 1
    assert result.mfe_pct == pytest.approx(5.0, rel=1e-2)  # (100-95)/100*100
    assert result.mae_pct == pytest.approx(2.0, rel=1e-2)  # (102-100)/100*100


def test_compute_checkpoint_metrics_trade_closed_before_checkpoint():
    candles = _hourly_candles(300)
    result = compute_checkpoint_metrics(
        candles, "2026-01-01T00:00:00Z", "2026-01-01T02:00:00Z", 100.0, "SHORT", "24h", 1440.0
    )
    assert result.trade_closed_before_checkpoint is True


def test_compute_checkpoint_metrics_not_yet_closed_at_checkpoint():
    candles = _hourly_candles(300)
    result = compute_checkpoint_metrics(
        candles, "2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z", 100.0, "SHORT", "4h", 240.0
    )
    assert result.trade_closed_before_checkpoint is False


def test_compute_checkpoint_metrics_still_favorable_side_short():
    candles = _hourly_candles(300, base=95.0)  # price dropped from entry 100
    result = compute_checkpoint_metrics(
        candles, "2026-01-01T00:00:00Z", "2026-01-10T00:00:00Z", 100.0, "SHORT", "4h", 240.0
    )
    assert result.still_favorable_side is True
    assert result.price_change_pct < 0


def test_compute_checkpoint_metrics_meaningful_adverse_flag():
    candles = _hourly_candles(300, base=100.0, high_add=5.0, low_sub=0.5)
    result = compute_checkpoint_metrics(
        candles, "2026-01-01T00:00:00Z", "2026-01-10T00:00:00Z", 100.0, "SHORT", "4h", 240.0
    )
    assert result.meaningful_adverse_seen is True  # MAE ~5% >= 1% threshold


def test_compute_checkpoint_metrics_missing_direction():
    candles = _hourly_candles(10)
    result = compute_checkpoint_metrics(
        candles, "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", 100.0, None, "4h", 240.0
    )
    assert result.n_candles_in_window == 0


def test_compute_checkpoint_metrics_never_uses_candles_past_checkpoint():
    # A huge spike far beyond the 4h checkpoint must not leak into the 4h result.
    dates = pd.date_range("2026-01-01T00:00", periods=10, freq="4h", tz="UTC")
    rows = []
    for i, d in enumerate(dates):
        if i <= 1:
            rows.append({"date": d, "open": 100, "high": 101, "low": 99, "close": 100})
        else:
            rows.append({"date": d, "open": 100, "high": 500, "low": 1, "close": 100})
    candles = pd.DataFrame(rows)
    result = compute_checkpoint_metrics(
        candles, "2026-01-01T00:00:00Z", "2026-01-10T00:00:00Z", 100.0, "SHORT", "4h", 240.0
    )
    assert result.mfe_pct < 5.0  # only row 0 and row 1 (at exactly 4h) should be in window


# ---------------------------------------------------------------------------
# compute_checkpoint_series
# ---------------------------------------------------------------------------


def test_compute_checkpoint_series_has_all_checkpoints():
    candles = _hourly_candles(300)
    trade = _trade()
    series = compute_checkpoint_series(candles, trade)
    assert set(series.keys()) == set(CHECKPOINTS.keys())


# ---------------------------------------------------------------------------
# aggregate_checkpoint / aggregate_all_checkpoints
# ---------------------------------------------------------------------------


def test_aggregate_checkpoint_thin_sample_flag():
    candles = _hourly_candles(300)
    trades = [_trade(entry_time=f"2026-01-{i:02d}T00:00:00Z", exit_time=f"2026-01-{i+9:02d}T00:00:00Z") for i in range(1, 4)]
    per_trade = [compute_checkpoint_series(candles, t) for t in trades]
    agg = aggregate_checkpoint(per_trade, "4h")
    assert agg.n_trades_total == 3
    assert agg.is_thin_sample is True
    assert THIN_SAMPLE_THRESHOLD == 5


def test_aggregate_checkpoint_excludes_unreached_from_means():
    candles = _hourly_candles(300)
    reached_trade = _trade(entry_time="2026-01-01T00:00:00Z", exit_time="2026-01-10T00:00:00Z")
    unreached_trade = _trade(pair="ETH/USDC:USDC", entry_time="2026-01-01T00:00:00Z", exit_time="2026-01-10T00:00:00Z")
    per_trade = [
        compute_checkpoint_series(candles, reached_trade),
        compute_checkpoint_series(None, unreached_trade),  # no OHLCV -> unreached
    ]
    agg = aggregate_checkpoint(per_trade, "4h")
    assert agg.n_trades_total == 2
    assert agg.n_trades_reached == 1


def test_aggregate_all_checkpoints_returns_every_label():
    candles = _hourly_candles(300)
    trades = [_trade()]
    per_trade = [compute_checkpoint_series(candles, t) for t in trades]
    result = aggregate_all_checkpoints(per_trade)
    assert set(result.keys()) == set(CHECKPOINTS.keys())


def test_aggregate_checkpoint_empty_group():
    agg = aggregate_checkpoint([], "4h")
    assert agg.n_trades_total == 0
    assert agg.n_trades_reached == 0
    assert agg.is_thin_sample is True
    assert agg.mean_mfe_pct is None
