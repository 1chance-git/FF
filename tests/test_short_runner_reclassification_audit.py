"""Focused unit tests for `hermes.short_runner_reclassification_audit`
(research-only, diagnosis-only)."""

from __future__ import annotations

import pandas as pd
import pytest

from hermes.short_runner_reclassification_audit import (
    LONG_DURATION_LABEL,
    PERSISTENT_KEYS,
    StructuralTradeRecord,
    TRAJECTORY_CHECKPOINTS,
    TRAJECTORY_ORDER,
    build_structural_record,
    build_structural_table,
    find_largest_gap,
    is_structurally_persistent,
    list_short_winners,
)
from hermes.trade_report import Trade


def _trade(
    pair="BTC/USDC:USDC", direction="SHORT", entry_time="2026-01-01T00:00:00Z",
    exit_time="2026-03-01T00:00:00Z", entry_price=100.0, profit_pct=5.0, profit_abs=50.0,
    duration_minutes=86400.0, exit_reason="exit_signal", is_open=False,
):
    return Trade(
        pair=pair, direction=direction, entry_time=entry_time, exit_time=exit_time,
        entry_price=entry_price, exit_price=95.0, enter_tag=None, exit_reason=exit_reason,
        profit_abs=profit_abs, profit_pct=profit_pct, duration_minutes=duration_minutes, is_open=is_open,
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
# list_short_winners / identities reused verbatim
# ---------------------------------------------------------------------------


def test_persistent_keys_exactly_three():
    assert len(PERSISTENT_KEYS) == 3


def test_list_short_winners_filters_direction_and_outcome():
    trades = [
        _trade(direction="SHORT", profit_abs=10.0),
        _trade(direction="SHORT", profit_abs=-10.0),
        _trade(direction="LONG", profit_abs=10.0),
    ]
    winners = list_short_winners(trades)
    assert len(winners) == 1
    assert winners[0].direction == "SHORT"
    assert winners[0].is_winner is True


def test_list_short_winners_preserves_order_never_resorted():
    trades = [
        _trade(pair="ETH/USDC:USDC", entry_time="2026-05-01T00:00:00Z", profit_abs=1.0),
        _trade(pair="BTC/USDC:USDC", entry_time="2026-01-01T00:00:00Z", profit_abs=100.0),
    ]
    winners = list_short_winners(trades)
    assert [w.pair for w in winners] == ["ETH/USDC:USDC", "BTC/USDC:USDC"]


# ---------------------------------------------------------------------------
# trajectory ladder is a subset of the existing checkpoint ladder
# ---------------------------------------------------------------------------


def test_trajectory_checkpoints_are_a_subset_of_existing_ladder():
    assert set(TRAJECTORY_CHECKPOINTS.keys()) == {"4h", "7d", "14d", "21d", "30d"}
    assert TRAJECTORY_ORDER == ("4h", "7d", "14d", "21d", "30d")


# ---------------------------------------------------------------------------
# build_structural_record: no lookahead, no fabricated values
# ---------------------------------------------------------------------------


def test_build_structural_record_none_ohlcv_returns_empty_fields():
    trade = _trade()
    record = build_structural_record(trade, None)
    assert record.pct_structurally_aligned is None
    assert record.longest_aligned_run_candles == 0
    assert record.first_invalidation_time is None
    assert all(v is None for v in record.ema_distance_trajectory.values())
    assert all(v is None for v in record.mfe_trajectory.values())


def test_build_structural_record_pl_label_matches_classify_group():
    persistent_trade = _trade(pair="ETH/USDC:USDC", entry_time="2026-01-20 16:00:00+00:00", profit_abs=10.0)
    candles = _flat_series(300, close=90.0)
    record = build_structural_record(persistent_trade, candles)
    assert record.pl_label == "PERSISTENT"

    ordinary_trade = _trade(pair="ETH/USDC:USDC", entry_time="2026-02-01T00:00:00Z", profit_abs=10.0)
    record2 = build_structural_record(ordinary_trade, candles)
    assert record2.pl_label == "ORDINARY"


def test_build_structural_record_detects_first_invalidation():
    # price starts below EMA (elevated lag from a prior high block) then
    # rises above it -- SHORT should be invalidated once close > ema200.
    dates = pd.date_range("2026-01-01T00:00", periods=300, freq="4h", tz="UTC")
    closes = [110.0] * 60 + [90.0] * 40 + [130.0] * 200
    candles = pd.DataFrame({
        "date": dates, "open": closes, "high": [c + 1 for c in closes],
        "low": [c - 1 for c in closes], "close": closes,
    })
    trade = _trade(entry_time=str(dates[60]), exit_time=str(dates[200]))
    record = build_structural_record(trade, candles)
    assert record.first_invalidation_time is not None
    assert record.hours_entry_to_invalidation is not None
    assert record.hours_entry_to_invalidation > 0


def test_build_structural_record_longest_aligned_run_never_exceeds_candle_count():
    candles = _flat_series(300, close=90.0)
    trade = _trade(entry_time="2026-01-01T00:00:00Z", exit_time="2026-01-10T00:00:00Z")
    record = build_structural_record(trade, candles)
    assert record.longest_aligned_run_candles <= 60  # ~9 days of 4h candles


def test_build_structural_table_preserves_input_order():
    candles = _flat_series(300, close=90.0)
    trades = [_trade(pair="BTC/USDC:USDC", profit_abs=1.0), _trade(pair="ETH/USDC:USDC", profit_abs=2.0)]
    table = build_structural_table(trades, {"BTC/USDC:USDC": candles, "ETH/USDC:USDC": candles})
    assert [r.pair for r in table] == ["BTC/USDC:USDC", "ETH/USDC:USDC"]


# ---------------------------------------------------------------------------
# is_structurally_persistent -- descriptive only, never rewrites pl_label
# ---------------------------------------------------------------------------


def _record(duration_days, mfe_30d, pl_label="ORDINARY"):
    return StructuralTradeRecord(
        pair="BTC/USDC:USDC", entry_time="2026-01-01T00:00:00Z", pl_label=pl_label,
        final_profit_pct=5.0, duration_minutes=duration_days * 1440.0, duration_days=duration_days,
        pct_structurally_aligned=95.0, longest_aligned_run_candles=10,
        ema_distance_trajectory={"30d": -5.0}, first_invalidation_time=None,
        hours_entry_to_invalidation=None, mfe_trajectory={"30d": mfe_30d},
    )


def test_is_structurally_persistent_above_median_both_dims():
    records = [_record(5, 2.0), _record(10, 5.0), _record(45, 30.0)]
    assert is_structurally_persistent(records[2], records) is True
    assert is_structurally_persistent(records[0], records) is False


def test_is_structurally_persistent_never_touches_pl_label():
    record = _record(45, 30.0, pl_label="ORDINARY")
    is_structurally_persistent(record, [record, _record(5, 1.0)])
    assert record.pl_label == "ORDINARY"  # untouched -- frozen dataclass, never reassigned


def test_is_structurally_persistent_missing_data_returns_false():
    empty = StructuralTradeRecord(
        pair="X", entry_time=None, pl_label="ORDINARY", final_profit_pct=None,
        duration_minutes=None, duration_days=None, pct_structurally_aligned=None,
        longest_aligned_run_candles=0, ema_distance_trajectory={}, first_invalidation_time=None,
        hours_entry_to_invalidation=None, mfe_trajectory={},
    )
    assert is_structurally_persistent(empty, [empty, _record(5, 1.0)]) is False


# ---------------------------------------------------------------------------
# find_largest_gap -- descriptive, never adopted as a threshold
# ---------------------------------------------------------------------------


def test_find_largest_gap_identifies_biggest_jump():
    values = [1.0, 2.0, 3.0, 20.0, 21.0]
    gap = find_largest_gap(values)
    assert gap.gap_index == 2
    assert gap.value_before == 3.0
    assert gap.value_after == 20.0
    assert gap.gap_size == pytest.approx(17.0)


def test_find_largest_gap_dominant_flag():
    dominant = find_largest_gap([1.0, 2.0, 3.0, 50.0])
    assert dominant.is_dominant is True
    even = find_largest_gap([1.0, 5.0, 9.0, 13.0])
    assert even.is_dominant is False


def test_find_largest_gap_too_few_values():
    assert find_largest_gap([]) is None
    assert find_largest_gap([1.0]) is None


def test_find_largest_gap_eight_trade_scale():
    # sample matching the 8 SHORT winners' rough scale -- ensures no
    # crash / off-by-one across a realistic-sized list.
    durations = sorted([3.5, 12.0, 17.5, 20.2, 29.0, 36.2, 43.3, 60.3])
    gap = find_largest_gap(durations)
    assert gap is not None
    assert gap.gap_size > 0
