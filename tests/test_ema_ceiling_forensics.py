"""Focused unit tests for `hermes.ema_ceiling_forensics` (research-only)."""

from __future__ import annotations

import pytest

from hermes.ema_ceiling_forensics import (
    BaselineTradeRecord,
    aggregate_kept_trades,
    baseline_records_from_forensics_json,
    classify_all,
    compute_abs_ema_distance_pct,
    passes_ceiling,
    summarize_fate,
)


# ---------------------------------------------------------------------------
# compute_abs_ema_distance_pct
# ---------------------------------------------------------------------------


def test_abs_ema_distance_long_matches_spec_formula():
    # close=110, ema200=100 -> abs(110-100)/100*100 = 10.0
    assert compute_abs_ema_distance_pct(110.0, 100.0, "LONG") == pytest.approx(10.0)


def test_abs_ema_distance_short_matches_spec_formula():
    # close=90, ema200=100 -> abs(90-100)/100*100 = 10.0
    assert compute_abs_ema_distance_pct(90.0, 100.0, "SHORT") == pytest.approx(10.0)


def test_abs_ema_distance_is_symmetric_regardless_of_direction():
    """The spec requires distance to measure extension regardless of
    direction -- a LONG 10% above EMA and a SHORT 10% below EMA must
    produce the same magnitude."""
    long_dist = compute_abs_ema_distance_pct(110.0, 100.0, "LONG")
    short_dist = compute_abs_ema_distance_pct(90.0, 100.0, "SHORT")
    assert long_dist == pytest.approx(short_dist)


def test_abs_ema_distance_none_on_missing_inputs():
    assert compute_abs_ema_distance_pct(None, 100.0, "LONG") is None
    assert compute_abs_ema_distance_pct(110.0, None, "LONG") is None
    assert compute_abs_ema_distance_pct(110.0, 100.0, None) is None
    assert compute_abs_ema_distance_pct(110.0, 0.0, "LONG") is None


# ---------------------------------------------------------------------------
# passes_ceiling
# ---------------------------------------------------------------------------


def test_passes_ceiling_no_threshold_always_true():
    assert passes_ceiling(50.0, None) is True
    assert passes_ceiling(None, None) is True


def test_passes_ceiling_at_and_under_threshold():
    assert passes_ceiling(4.0, 4.0) is True
    assert passes_ceiling(3.99, 4.0) is True


def test_passes_ceiling_over_threshold_fails():
    assert passes_ceiling(4.01, 4.0) is False


def test_passes_ceiling_unknown_distance_never_passes_real_ceiling():
    assert passes_ceiling(None, 4.0) is False


# ---------------------------------------------------------------------------
# classify_all / summarize_fate: trade-fate diff logic
# ---------------------------------------------------------------------------


def _record(num, is_winner, exit_reason, dist, profit_pct=1.0):
    return BaselineTradeRecord(
        trade_number=num,
        pair="BTC/USDC:USDC",
        direction="LONG",
        exit_reason=exit_reason,
        profit_pct=profit_pct if is_winner else -profit_pct,
        is_winner=is_winner,
        signal_ema_distance_pct=dist,
    )


def test_classify_all_keeps_low_distance_eliminates_high_distance():
    records = [
        _record(1, True, "exit_signal", 2.0),
        _record(2, False, "stop_loss", 8.0),
    ]
    entries = classify_all(records, threshold_pct=5.0)
    kept = [e for e in entries if e.kept]
    eliminated = [e for e in entries if e.eliminated]
    assert [e.trade.trade_number for e in kept] == [1]
    assert [e.trade.trade_number for e in eliminated] == [2]


def test_summarize_fate_breaks_down_eliminated_by_outcome_and_exit_reason():
    records = [
        _record(1, True, "exit_signal", 1.0),   # kept
        _record(2, False, "stop_loss", 7.0),    # eliminated loser, stop_loss
        _record(3, True, "stop_loss", 9.0),     # eliminated winner, stop_loss (rare but possible)
        _record(4, False, "exit_signal", 6.0),  # eliminated loser, exit_signal
        _record(5, False, "force_exit", 6.5),   # eliminated loser, other exit
    ]
    entries = classify_all(records, threshold_pct=5.0)
    summary = summarize_fate(entries)

    assert summary.threshold_pct == 5.0
    assert summary.kept_count == 1
    assert summary.eliminated_count == 4
    assert {t.trade_number for t in summary.eliminated_winners} == {3}
    assert {t.trade_number for t in summary.eliminated_losers} == {2, 4, 5}
    assert {t.trade_number for t in summary.eliminated_stop_loss} == {2, 3}
    assert {t.trade_number for t in summary.eliminated_exit_signal} == {4}
    assert {t.trade_number for t in summary.eliminated_other_exit} == {5}
    assert summary.stop_trades_prevented == 2
    assert summary.winners_prevented == 1


def test_summarize_fate_no_ceiling_eliminates_nothing():
    records = [_record(1, True, "exit_signal", 12.0), _record(2, False, "stop_loss", 20.0)]
    entries = classify_all(records, threshold_pct=None)
    summary = summarize_fate(entries)
    assert summary.kept_count == 2
    assert summary.eliminated_count == 0


def test_summarize_fate_requires_at_least_one_entry():
    with pytest.raises(ValueError):
        summarize_fate([])


# ---------------------------------------------------------------------------
# aggregate_kept_trades
# ---------------------------------------------------------------------------


def test_aggregate_kept_trades_computes_win_rate_and_profit_over_kept_only():
    records = [
        _record(1, True, "exit_signal", 1.0, profit_pct=5.0),
        _record(2, False, "stop_loss", 8.0, profit_pct=3.0),  # eliminated at threshold 5
    ]
    entries = classify_all(records, threshold_pct=5.0)
    agg = aggregate_kept_trades(entries)
    assert agg.trades == 1
    assert agg.winners == 1
    assert agg.losers == 0
    assert agg.win_rate_pct == pytest.approx(100.0)
    assert agg.total_profit_pct == pytest.approx(5.0)
    assert agg.average_profit_pct == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# baseline_records_from_forensics_json
# ---------------------------------------------------------------------------


def test_baseline_records_from_forensics_json_converts_directional_to_abs_pct():
    payload = {
        "trades": [
            {
                "trade_number": 1,
                "pair": "ETH/USDC:USDC",
                "direction": "SHORT",
                "exit_reason": "stop_loss",
                "profit_pct": -5.0,
                "is_winner": False,
                "ema_distance_pct": -0.0730,  # fractional, directional (7.30% adverse-scale here)
            },
            {
                "trade_number": 2,
                "pair": "SOL/USDC:USDC",
                "direction": "LONG",
                "exit_reason": "exit_signal",
                "profit_pct": 2.0,
                "is_winner": True,
                "ema_distance_pct": None,
            },
        ]
    }
    records = baseline_records_from_forensics_json(payload)
    assert len(records) == 2
    assert records[0].signal_ema_distance_pct == pytest.approx(7.30)
    assert records[1].signal_ema_distance_pct is None
