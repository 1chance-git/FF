"""Focused unit tests for `hermes.short_runner_persistence_threshold_audit`
(research-only)."""

from __future__ import annotations

import pandas as pd
import pytest

from hermes.short_runner_divergence_audit import (
    aggregate_group_checkpoint,
    all_checkpoint_snapshots,
    reconstruct_full_trade,
)
from hermes.short_runner_persistence_threshold_audit import (
    CHECKPOINT_ORDER,
    CHECKPOINTS,
    GapRow,
    THIN_SAMPLE_THRESHOLD,
    VARIABLES,
    compute_gap_table,
    earliest_meaningful_divergence_by_variable,
    first_meaningful_divergence_checkpoint,
    per_trade_support,
    variables_agreeing_on_window,
)
from hermes.trade_report import Trade


def _trade(
    pair="BTC/USDC:USDC", direction="SHORT", entry_time="2026-01-01T00:00:00Z",
    exit_time="2026-03-01T00:00:00Z", entry_price=100.0, profit_pct=5.0, profit_abs=50.0,
    duration_minutes=12960.0, exit_reason="exit_signal",
):
    return Trade(
        pair=pair, direction=direction, entry_time=entry_time, exit_time=exit_time,
        entry_price=entry_price, exit_price=95.0, enter_tag=None, exit_reason=exit_reason,
        profit_abs=profit_abs, profit_pct=profit_pct, duration_minutes=duration_minutes, is_open=False,
    )


def _trending_series(n_flat, n_trend, start="2026-01-01T00:00", freq="4h", flat_close=110.0, trend_close=90.0):
    dates = pd.date_range(start, periods=n_flat + n_trend, freq=freq, tz="UTC")
    closes = [flat_close] * n_flat + [trend_close] * n_trend
    return pd.DataFrame({
        "date": dates, "open": closes, "high": [c + 1 for c in closes],
        "low": [c - 1 for c in closes], "close": closes,
    })


def _flat_series(n, start="2026-01-01T00:00", freq="4h", close=90.0):
    dates = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({
        "date": dates, "open": [close] * n, "high": [close + 1] * n,
        "low": [close - 1] * n, "close": [close] * n,
    })


# ---------------------------------------------------------------------------
# checkpoint ladder / variable list (block's own literal requirements)
# ---------------------------------------------------------------------------


def test_checkpoint_ladder_matches_block_requirement():
    assert list(CHECKPOINTS.keys()) == ["7d", "10d", "14d", "17d", "21d", "24d", "30d"]
    assert CHECKPOINTS["30d"] == 30 * 24 * 60
    assert CHECKPOINT_ORDER == tuple(CHECKPOINTS.keys())


def test_variables_are_exactly_the_six_named():
    assert set(VARIABLES) == {
        "mean_ema_distance_pct", "mean_adx", "mean_donchian_breakout_pct",
        "mean_mfe_pct", "pct_structurally_aligned", "pct_adx_above_threshold",
    }
    assert len(VARIABLES) == 6


# ---------------------------------------------------------------------------
# compute_gap_table
# ---------------------------------------------------------------------------


def test_compute_gap_table_gap_is_persistent_minus_ordinary():
    candles = _trending_series(60, 240, flat_close=110.0, trend_close=80.0)
    dates = candles["date"]
    p_trade_candles = reconstruct_full_trade(candles, dates.iloc[59], "2026-04-01T00:00:00Z", 100.0, "SHORT")
    o_candles = _flat_series(300, close=95.0)
    o_trade_candles = reconstruct_full_trade(o_candles, "2026-01-01T00:00:00Z", "2026-04-01T00:00:00Z", 100.0, "SHORT")
    p_agg = {label: aggregate_group_checkpoint([all_checkpoint_snapshots(p_trade_candles)], label) for label in CHECKPOINT_ORDER}
    o_agg = {label: aggregate_group_checkpoint([all_checkpoint_snapshots(o_trade_candles)], label) for label in CHECKPOINT_ORDER}
    rows = compute_gap_table(p_agg, o_agg)
    assert all(isinstance(r, GapRow) for r in rows)
    assert len(rows) == len(CHECKPOINT_ORDER) * len(VARIABLES)
    for r in rows:
        if r.persistent_value is not None and r.ordinary_value is not None:
            assert r.gap == pytest.approx(r.persistent_value - r.ordinary_value)
        else:
            assert r.gap is None


def test_compute_gap_table_empty_aggregates_gives_none_gaps():
    empty_agg = {label: aggregate_group_checkpoint([], label) for label in CHECKPOINT_ORDER}
    rows = compute_gap_table(empty_agg, empty_agg)
    assert all(r.gap is None for r in rows)


# ---------------------------------------------------------------------------
# first_meaningful_divergence_checkpoint -- NOT a sign-only rule
# ---------------------------------------------------------------------------


def test_meaningful_divergence_requires_new_magnitude_high_water_mark():
    # tiny sign flips that never exceed the initial noise never qualify.
    pairs = [("7d", 0.5), ("10d", -0.3), ("14d", 0.4), ("17d", -0.2), ("21d", 0.1), ("24d", -0.05), ("30d", 0.02)]
    assert first_meaningful_divergence_checkpoint(pairs) is None


def test_meaningful_divergence_needs_both_magnitude_and_sustained_direction():
    # a spike that reverses right after should not count.
    pairs = [("7d", 0.5), ("10d", 5.0), ("14d", -5.0), ("17d", -5.0), ("21d", -5.0)]
    # 10d is a new high-water mark (5.0 > 0.5) but its tail includes 14d
    # which flips sign -- 10d itself doesn't hold. 14d IS a new
    # high-water mark in magnitude? magnitude at 14d is 5.0, not greater
    # than 5.0 at 10d, so 14d doesn't qualify as a *new* high-water mark
    # either. Expect None here since no checkpoint is both a strict new
    # max AND sign-consistent through the end.
    assert first_meaningful_divergence_checkpoint(pairs) is None


def test_meaningful_divergence_detects_genuine_transition():
    pairs = [("7d", 0.3), ("10d", -0.4), ("14d", 0.2), ("17d", 3.0), ("21d", 4.0), ("24d", 5.0), ("30d", 6.0)]
    # 17d (3.0) is a new high-water mark against the prior max magnitude
    # (0.4), and every checkpoint from 17d onward is positive.
    assert first_meaningful_divergence_checkpoint(pairs) == "17d"


def test_meaningful_divergence_none_in_tail_disqualifies():
    pairs = [("7d", 0.1), ("10d", 3.0), ("14d", None), ("17d", 3.0)]
    assert first_meaningful_divergence_checkpoint(pairs) is None


def test_meaningful_divergence_too_short_tail_disqualifies():
    pairs = [("7d", 0.1), ("10d", 5.0)]
    # 10d is a new high-water mark but its own tail has length 1 -- not
    # "held afterward" by this definition.
    assert first_meaningful_divergence_checkpoint(pairs) is None


def test_meaningful_divergence_empty_and_single():
    assert first_meaningful_divergence_checkpoint([]) is None
    assert first_meaningful_divergence_checkpoint([("7d", 1.0)]) is None


# ---------------------------------------------------------------------------
# earliest_meaningful_divergence_by_variable / variables_agreeing_on_window
# ---------------------------------------------------------------------------


def test_earliest_meaningful_divergence_by_variable_and_grouping():
    rows = [
        GapRow("7d", "mean_ema_distance_pct", -1.0, -1.3, 0.3),
        GapRow("10d", "mean_ema_distance_pct", -1.0, -1.4, 0.4),
        GapRow("14d", "mean_ema_distance_pct", -3.0, -0.9, -2.1),
        GapRow("17d", "mean_ema_distance_pct", -5.0, -0.8, -4.2),
        GapRow("7d", "mean_adx", 30.0, 29.8, 0.2),
        GapRow("10d", "mean_adx", 30.0, 29.6, 0.4),
        GapRow("14d", "mean_adx", 40.0, 30.0, 10.0),
        GapRow("17d", "mean_adx", 42.0, 30.0, 12.0),
    ]
    order = ["7d", "10d", "14d", "17d"]
    per_var = earliest_meaningful_divergence_by_variable(rows, order)
    assert per_var["mean_ema_distance_pct"] == "14d"
    # mean_adx's 10d gap (0.4) is already a new high-water mark over 7d's
    # (0.2), and stays positive through 17d -- it qualifies at 10d, one
    # checkpoint earlier than mean_ema_distance_pct.
    assert per_var["mean_adx"] == "10d"
    grouped = variables_agreeing_on_window(per_var, order)
    assert grouped["14d"] == ["mean_ema_distance_pct"]
    assert grouped["10d"] == ["mean_adx"]


# ---------------------------------------------------------------------------
# per_trade_support
# ---------------------------------------------------------------------------


def test_per_trade_support_flags_agreement_and_disagreement():
    candles = _trending_series(60, 240, flat_close=110.0, trend_close=80.0)
    dates = candles["date"]
    trade_a = reconstruct_full_trade(candles, dates.iloc[59], "2026-04-01T00:00:00Z", 100.0, "SHORT")
    trade_b = reconstruct_full_trade(_flat_series(300, close=95.0), "2026-01-01T00:00:00Z", "2026-04-01T00:00:00Z", 100.0, "SHORT")
    o_candles = _flat_series(300, close=98.0)
    o_trade_candles = reconstruct_full_trade(o_candles, "2026-01-01T00:00:00Z", "2026-04-01T00:00:00Z", 100.0, "SHORT")
    o_agg = {label: aggregate_group_checkpoint([all_checkpoint_snapshots(o_trade_candles)], label) for label in CHECKPOINT_ORDER}

    snaps_a = all_checkpoint_snapshots(trade_a)
    snaps_b = all_checkpoint_snapshots(trade_b)
    checkpoint = "30d"
    p_agg = aggregate_group_checkpoint([snaps_a, snaps_b], checkpoint)
    group_gap = p_agg.mean_ema_distance_pct - o_agg[checkpoint].mean_ema_distance_pct

    results = per_trade_support(
        [snaps_a, snaps_b], ["trade_a", "trade_b"], o_agg, "mean_ema_distance_pct", checkpoint, group_gap,
    )
    assert len(results) == 2
    for r in results:
        assert r.checkpoint_label == checkpoint
        assert r.variable == "mean_ema_distance_pct"
        assert r.supports_direction in (True, False, None)


def test_per_trade_support_mfe_field_mapping_works():
    candles = _flat_series(300, close=90.0)
    trade = reconstruct_full_trade(candles, "2026-01-01T00:00:00Z", "2026-04-01T00:00:00Z", 100.0, "SHORT")
    snaps = all_checkpoint_snapshots(trade)
    o_agg = {label: aggregate_group_checkpoint([snaps], label) for label in CHECKPOINT_ORDER}
    results = per_trade_support([snaps], ["trade"], o_agg, "mean_mfe_pct", "30d", 0.0)
    assert results[0].trade_value == snaps["30d"].mfe_pct


def test_per_trade_support_none_when_trade_value_missing():
    o_candles = _flat_series(300, close=95.0)
    o_trade_candles = reconstruct_full_trade(o_candles, "2026-01-01T00:00:00Z", "2026-04-01T00:00:00Z", 100.0, "SHORT")
    o_agg = {label: aggregate_group_checkpoint([all_checkpoint_snapshots(o_trade_candles)], label) for label in CHECKPOINT_ORDER}
    empty_snaps = all_checkpoint_snapshots([])
    results = per_trade_support([empty_snaps], ["empty_trade"], o_agg, "mean_ema_distance_pct", "30d", 1.0)
    assert results[0].supports_direction is None


# ---------------------------------------------------------------------------
# reconciliation-adjacent: sample-size discipline reused, not re-derived
# ---------------------------------------------------------------------------


def test_thin_sample_threshold_unchanged_from_prior_blocks():
    assert THIN_SAMPLE_THRESHOLD == 5
