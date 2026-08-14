"""Focused unit tests for `hermes.short_runner_divergence_audit` (research-only)."""

from __future__ import annotations

import pandas as pd
import pytest

from hermes.short_runner_divergence_audit import (
    CHECKPOINT_ORDER,
    CHECKPOINTS,
    DivergenceRow,
    LONG_DURATION_ORDINARY_KEY,
    ORDINARY_KEYS,
    PERSISTENT_KEYS,
    THIN_SAMPLE_THRESHOLD,
    aggregate_group_checkpoint,
    all_checkpoint_snapshots,
    classify_group,
    compute_divergence_table,
    earliest_divergence_by_variable,
    first_sign_consistent_checkpoint,
    is_long_duration_ordinary,
    overall_earliest_divergence,
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


def _trending_series(n_flat, n_trend, start="2026-01-01T00:00", freq="4h", flat_close=110.0, trend_close=90.0):
    dates = pd.date_range(start, periods=n_flat + n_trend, freq=freq, tz="UTC")
    closes = [flat_close] * n_flat + [trend_close] * n_trend
    return pd.DataFrame({
        "date": dates, "open": closes, "high": [c + 1 for c in closes],
        "low": [c - 1 for c in closes], "close": closes,
    })


# ---------------------------------------------------------------------------
# reconciliation identities
# ---------------------------------------------------------------------------


def test_persistent_keys_exactly_three():
    assert len(PERSISTENT_KEYS) == 3


def test_ordinary_keys_exactly_five():
    assert len(ORDINARY_KEYS) == 5


def test_persistent_and_ordinary_keys_disjoint():
    assert PERSISTENT_KEYS.isdisjoint(ORDINARY_KEYS)


def test_long_duration_ordinary_key_is_btc_20251029():
    assert LONG_DURATION_ORDINARY_KEY == ("BTC/USDC:USDC", "2025-10-29 20:00:00+00:00")
    assert LONG_DURATION_ORDINARY_KEY in ORDINARY_KEYS


def test_classify_group_persistent():
    trade = _trade(pair="ETH/USDC:USDC", entry_time="2026-01-20 16:00:00+00:00", profit_abs=10.0)
    assert classify_group(trade) == "PERSISTENT"


def test_classify_group_ordinary_winner():
    trade = _trade(pair="BTC/USDC:USDC", entry_time="2025-10-29 20:00:00+00:00", profit_abs=10.0)
    assert classify_group(trade) == "ORDINARY"


def test_classify_group_loser():
    trade = _trade(profit_abs=-10.0)
    assert classify_group(trade) == "LOSER"


def test_is_long_duration_ordinary_true_only_for_exact_identity():
    target = _trade(pair="BTC/USDC:USDC", entry_time="2025-10-29 20:00:00+00:00", profit_abs=10.0)
    other = _trade(pair="BTC/USDC:USDC", entry_time="2026-06-01 12:00:00+00:00", profit_abs=10.0)
    assert is_long_duration_ordinary(target) is True
    assert is_long_duration_ordinary(other) is False


# ---------------------------------------------------------------------------
# checkpoint ladder
# ---------------------------------------------------------------------------


def test_checkpoint_ladder_matches_required_labels():
    assert list(CHECKPOINTS.keys()) == [
        "4h", "12h", "24h", "48h", "3d", "7d", "10d", "14d", "17d", "21d", "24d", "30d",
    ]
    assert CHECKPOINTS["30d"] == 30 * 24 * 60
    assert CHECKPOINT_ORDER == tuple(CHECKPOINTS.keys())


# ---------------------------------------------------------------------------
# reconstruct_full_trade: no lookahead, no candles past exit
# ---------------------------------------------------------------------------


def test_reconstruct_full_trade_none_ohlcv():
    assert reconstruct_full_trade(None, "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", 100.0, "SHORT") == []


def test_reconstruct_full_trade_never_uses_candles_after_exit():
    candles = _flat_series(50, close=90.0)
    result = reconstruct_full_trade(candles, "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", 100.0, "SHORT")
    assert result[-1].date <= pd.Timestamp("2026-01-02T00:00", tz="UTC")
    assert all(c.date <= pd.Timestamp("2026-01-02T00:00", tz="UTC") for c in result)


def test_reconstruct_full_trade_cumulative_mfe_never_decreases():
    candles = _flat_series(20, close=90.0)
    result = reconstruct_full_trade(candles, "2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z", 100.0, "SHORT")
    mfes = [c.cumulative_mfe_pct for c in result]
    assert all(b >= a for a, b in zip(mfes, mfes[1:]))


# ---------------------------------------------------------------------------
# checkpoint boundaries / frozen closed-trade checkpoints
# ---------------------------------------------------------------------------


def test_checkpoint_snapshot_closed_before_flag_and_frozen_metrics():
    candles = _flat_series(300, close=90.0)
    trade_candles = reconstruct_full_trade(candles, "2026-01-01T00:00:00Z", "2026-01-01T08:00:00Z", 100.0, "SHORT")
    snapshots = all_checkpoint_snapshots(trade_candles)
    snap_4h = snapshots["4h"]
    snap_30d = snapshots["30d"]
    assert snap_4h.closed_before_checkpoint is False  # exit is exactly at 8h > 4h boundary check uses <=
    assert snap_30d.closed_before_checkpoint is True
    # a checkpoint reached after exit must use exactly the same frozen
    # subset as the trade's real (shorter) window -- never fabricated.
    assert snap_30d.n_candles_in_subset == len(trade_candles)
    assert snap_30d.mfe_pct == trade_candles[-1].cumulative_mfe_pct


def test_checkpoint_snapshot_never_uses_candle_after_checkpoint():
    candles = _flat_series(300, close=90.0)
    trade_candles = reconstruct_full_trade(candles, "2026-01-01T00:00:00Z", "2026-01-20T00:00:00Z", 100.0, "SHORT")
    snapshots = all_checkpoint_snapshots(trade_candles)
    checkpoint_days = CHECKPOINTS["24h"] / 1440.0
    snap = snapshots["24h"]
    assert snap.n_candles_in_subset <= sum(1 for c in trade_candles if c.days_since_entry <= checkpoint_days)


def test_all_checkpoint_snapshots_returns_every_ladder_label():
    candles = _flat_series(300, close=90.0)
    trade_candles = reconstruct_full_trade(candles, "2026-01-01T00:00:00Z", "2026-02-10T00:00:00Z", 100.0, "SHORT")
    snapshots = all_checkpoint_snapshots(trade_candles)
    assert set(snapshots.keys()) == set(CHECKPOINTS.keys())


def test_checkpoint_snapshot_empty_trade():
    snapshots = all_checkpoint_snapshots([])
    for label in CHECKPOINTS:
        assert snapshots[label].n_candles_in_subset == 0
        assert snapshots[label].closed_before_checkpoint is False


# ---------------------------------------------------------------------------
# checkpoint-to-checkpoint deltas
# ---------------------------------------------------------------------------


def test_delta_ema_distance_is_none_at_first_checkpoint():
    candles = _flat_series(300, close=90.0)
    trade_candles = reconstruct_full_trade(candles, "2026-01-01T00:00:00Z", "2026-02-10T00:00:00Z", 100.0, "SHORT")
    snapshots = all_checkpoint_snapshots(trade_candles)
    assert snapshots["4h"].delta_ema_distance_pct is None
    assert snapshots["4h"].delta_adx is None


def test_delta_ema_distance_computed_between_consecutive_ladder_points():
    candles = _trending_series(60, 240, flat_close=110.0, trend_close=90.0)
    entry_date = candles["date"].iloc[60]
    trade_candles = reconstruct_full_trade(candles, entry_date, "2026-03-01T00:00:00Z", 100.0, "SHORT")
    snapshots = all_checkpoint_snapshots(trade_candles)
    later_labels = [label for label in CHECKPOINT_ORDER if snapshots[label].mean_ema_distance_pct is not None]
    assert len(later_labels) >= 2
    second = later_labels[1]
    first_val = snapshots[later_labels[0]].mean_ema_distance_pct
    second_val = snapshots[second].mean_ema_distance_pct
    assert snapshots[second].delta_ema_distance_pct == pytest.approx(second_val - first_val)


# ---------------------------------------------------------------------------
# group aggregation
# ---------------------------------------------------------------------------


def test_aggregate_group_checkpoint_thin_sample_for_persistent_n():
    candles = _flat_series(300, close=90.0)
    trades = [_trade(entry_time=f"2026-01-{i:02d}T00:00:00Z", exit_time=f"2026-01-{i + 8:02d}T00:00:00Z") for i in range(1, 4)]
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


# ---------------------------------------------------------------------------
# divergence detection
# ---------------------------------------------------------------------------


def test_first_sign_consistent_checkpoint_all_positive():
    pairs = [("4h", 1.0), ("12h", 2.0), ("24h", 3.0)]
    assert first_sign_consistent_checkpoint(pairs) == "4h"


def test_first_sign_consistent_checkpoint_flips_then_settles():
    pairs = [("4h", -1.0), ("12h", 1.0), ("24h", 2.0), ("48h", 3.0)]
    assert first_sign_consistent_checkpoint(pairs) == "12h"


def test_first_sign_consistent_checkpoint_never_settles():
    pairs = [("4h", 1.0), ("12h", -1.0), ("24h", 1.0), ("48h", -1.0)]
    assert first_sign_consistent_checkpoint(pairs) is None


def test_first_sign_consistent_checkpoint_none_gap_in_tail_breaks_it():
    pairs = [("4h", 1.0), ("12h", None), ("24h", 1.0)]
    # any candidate start whose tail contains a None is skipped; here every
    # tail from "4h" onward contains the None at "12h", so no start works
    # except one whose tail excludes it -- there is none, so overall None.
    assert first_sign_consistent_checkpoint(pairs) is None


def test_first_sign_consistent_checkpoint_too_short():
    assert first_sign_consistent_checkpoint([("4h", 1.0)]) is None
    assert first_sign_consistent_checkpoint([]) is None


def test_compute_divergence_table_gap_is_persistent_minus_ordinary():
    candles = _trending_series(60, 240, flat_close=110.0, trend_close=80.0)
    dates = pd.date_range("2026-01-01T00:00", periods=60, freq="4h", tz="UTC")
    p_trade_candles = reconstruct_full_trade(candles, dates[59], "2026-03-01T00:00:00Z", 100.0, "SHORT")
    o_candles = _flat_series(300, close=95.0)
    o_trade_candles = reconstruct_full_trade(o_candles, "2026-01-01T00:00:00Z", "2026-03-01T00:00:00Z", 100.0, "SHORT")
    p_agg = {label: aggregate_group_checkpoint([all_checkpoint_snapshots(p_trade_candles)], label) for label in CHECKPOINT_ORDER}
    o_agg = {label: aggregate_group_checkpoint([all_checkpoint_snapshots(o_trade_candles)], label) for label in CHECKPOINT_ORDER}
    rows = compute_divergence_table(p_agg, o_agg)
    assert all(isinstance(r, DivergenceRow) for r in rows)
    for r in rows:
        if r.persistent_value is not None and r.ordinary_value is not None:
            assert r.gap == pytest.approx(r.persistent_value - r.ordinary_value)
        else:
            assert r.gap is None


def test_earliest_divergence_by_variable_and_overall():
    rows = [
        DivergenceRow("4h", "mean_ema_distance_pct", -1.0, -1.0, 0.0),
        DivergenceRow("12h", "mean_ema_distance_pct", -2.0, -1.0, -1.0),
        DivergenceRow("24h", "mean_ema_distance_pct", -3.0, -1.0, -2.0),
        DivergenceRow("4h", "mean_adx", 30.0, 30.0, 0.0),
        DivergenceRow("12h", "mean_adx", 30.0, 30.0, 0.0),
        DivergenceRow("24h", "mean_adx", 40.0, 20.0, 20.0),
    ]
    order = ["4h", "12h", "24h"]
    per_var = earliest_divergence_by_variable(rows, order)
    assert per_var["mean_ema_distance_pct"] == "12h"
    # mean_adx only separates on the final checkpoint -- a lone trailing
    # point is not "sustained" separation, so no divergence point qualifies.
    assert per_var["mean_adx"] is None
    assert overall_earliest_divergence(rows, order) == "12h"


def test_overall_earliest_divergence_none_when_no_variable_diverges():
    rows = [
        DivergenceRow("4h", "mean_ema_distance_pct", 1.0, -1.0, 2.0),
        DivergenceRow("12h", "mean_ema_distance_pct", -1.0, 1.0, -2.0),
    ]
    assert overall_earliest_divergence(rows, ["4h", "12h"]) is None


# ---------------------------------------------------------------------------
# per-trade consistency: three persistent trades individually reconstructable
# ---------------------------------------------------------------------------


def test_each_persistent_trade_reconstructs_independently():
    candles = _trending_series(60, 240, flat_close=110.0, trend_close=90.0)
    dates = pd.date_range("2026-01-01T00:00", periods=60, freq="4h", tz="UTC")
    per_trade_results = []
    for offset in (0, 5, 10):
        tc = reconstruct_full_trade(candles, dates[55 + offset] if 55 + offset < 60 else dates[59], "2026-03-01T00:00:00Z", 100.0, "SHORT")
        per_trade_results.append(tc)
    assert all(len(tc) > 0 for tc in per_trade_results)
    # each trade's candle count differs (later entries see fewer candles)
    # -- confirms these are independent reconstructions, not one shared list.
    assert len({len(tc) for tc in per_trade_results}) >= 2
