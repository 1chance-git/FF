"""Focused unit tests for `hermes.ema_ceiling_mechanism_audit` (research-only)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from hermes.ema_ceiling_mechanism_audit import (
    CATEGORY_FORCE_EXIT,
    CATEGORY_LOSING_EXIT_SIGNAL,
    CATEGORY_STOP_LOSS,
    CATEGORY_UNRESOLVED,
    CATEGORY_WINNING_EXIT_SIGNAL,
    MechanismTradeRecord,
    classify_outcome_category,
    compare_all_metrics,
    compare_metric,
    compute_elimination_rates,
    compute_winner_preservation,
    merge_mechanism_records,
    split_removed_retained,
    stratify_by_group,
)


def _rec(
    trade_number=1,
    pair="BTC/USDC:USDC",
    direction="LONG",
    entry_time="2026-01-01T00:00:00",
    exit_reason="stop_loss",
    profit_pct=-5.0,
    profit_abs=-50.0,
    is_winner=False,
    ema_distance_pct=7.0,
    breakout_distance_pct=1.0,
    adx14=25.0,
    atr_pct=2.0,
    realized_vol=0.03,
):
    return MechanismTradeRecord(
        trade_number=trade_number,
        pair=pair,
        direction=direction,
        entry_time=entry_time,
        exit_reason=exit_reason,
        profit_pct=profit_pct,
        profit_abs=profit_abs,
        is_winner=is_winner,
        ema_distance_pct=ema_distance_pct,
        breakout_distance_pct=breakout_distance_pct,
        adx14=adx14,
        atr_pct=atr_pct,
        realized_vol=realized_vol,
    )


# ---------------------------------------------------------------------------
# classify_outcome_category
# ---------------------------------------------------------------------------


def test_classify_stop_loss():
    r = _rec(exit_reason="stop_loss", is_winner=False)
    assert classify_outcome_category(r) == CATEGORY_STOP_LOSS


def test_classify_losing_exit_signal():
    r = _rec(exit_reason="exit_signal", is_winner=False)
    assert classify_outcome_category(r) == CATEGORY_LOSING_EXIT_SIGNAL


def test_classify_winning_exit_signal():
    r = _rec(exit_reason="exit_signal", is_winner=True)
    assert classify_outcome_category(r) == CATEGORY_WINNING_EXIT_SIGNAL


def test_classify_force_exit():
    r = _rec(exit_reason="force_exit", is_winner=True)
    assert classify_outcome_category(r) == CATEGORY_FORCE_EXIT


def test_classify_unresolved_on_missing_exit_reason():
    r = _rec(exit_reason=None, is_winner=True)
    assert classify_outcome_category(r) == CATEGORY_UNRESOLVED


def test_classify_unresolved_on_missing_is_winner():
    r = _rec(exit_reason="exit_signal", is_winner=None)
    assert classify_outcome_category(r) == CATEGORY_UNRESOLVED


def test_classify_unresolved_on_unknown_exit_reason():
    r = _rec(exit_reason="something_else", is_winner=True)
    assert classify_outcome_category(r) == CATEGORY_UNRESOLVED


# ---------------------------------------------------------------------------
# split_removed_retained
# ---------------------------------------------------------------------------


def test_split_removed_retained_basic():
    kept = _rec(trade_number=1, entry_time="t1")
    dropped = _rec(trade_number=2, entry_time="t2")
    records = [kept, dropped]
    kept_ids = {("BTC/USDC:USDC", "t1", "LONG")}
    removed, retained = split_removed_retained(records, kept_ids)
    assert removed == [dropped]
    assert retained == [kept]


# ---------------------------------------------------------------------------
# compare_metric / compare_all_metrics
# ---------------------------------------------------------------------------


def test_compare_metric_means_and_medians():
    removed = [_rec(ema_distance_pct=8.0), _rec(ema_distance_pct=10.0)]
    retained = [_rec(ema_distance_pct=2.0), _rec(ema_distance_pct=4.0)]
    cmp = compare_metric("ema_distance_pct", removed, retained)
    assert cmp.removed_mean == pytest.approx(9.0)
    assert cmp.removed_median == pytest.approx(9.0)
    assert cmp.retained_mean == pytest.approx(3.0)
    assert cmp.retained_median == pytest.approx(3.0)
    assert cmp.removed_n == 2
    assert cmp.retained_n == 2


def test_compare_metric_ignores_none_values():
    removed = [_rec(ema_distance_pct=None), _rec(ema_distance_pct=10.0)]
    cmp = compare_metric("ema_distance_pct", removed, [])
    assert cmp.removed_n == 1
    assert cmp.removed_mean == pytest.approx(10.0)


def test_compare_metric_empty_is_none():
    cmp = compare_metric("ema_distance_pct", [], [])
    assert cmp.removed_mean is None
    assert cmp.removed_median is None
    assert cmp.removed_n == 0


def test_compare_all_metrics_covers_all_five_fields():
    removed = [_rec()]
    retained = [_rec()]
    result = compare_all_metrics(removed, retained)
    assert set(result.keys()) == {
        "ema_distance_pct",
        "breakout_distance_pct",
        "realized_vol",
        "atr_pct",
        "adx14",
    }


# ---------------------------------------------------------------------------
# compute_elimination_rates
# ---------------------------------------------------------------------------


def test_elimination_rates_stop_loss_higher_than_winner():
    baseline = [
        _rec(trade_number=1, exit_reason="stop_loss", is_winner=False),
        _rec(trade_number=2, exit_reason="stop_loss", is_winner=False),
        _rec(trade_number=3, exit_reason="exit_signal", is_winner=True),
        _rec(trade_number=4, exit_reason="exit_signal", is_winner=True),
    ]
    # Removes both stop-loss trades, keeps both winners.
    removed = baseline[:2]
    rates = compute_elimination_rates("th5", baseline, removed)
    assert rates.baseline_stop_loss_count == 2
    assert rates.removed_stop_loss_count == 2
    assert rates.stop_loss_removed_pct == pytest.approx(100.0)
    assert rates.baseline_winner_count == 2
    assert rates.removed_winner_count == 0
    assert rates.winner_removed_pct == pytest.approx(0.0)
    assert rates.removes_stop_losses_faster_than_winners is True


def test_elimination_rates_undefined_when_no_baseline_stop_losses():
    baseline = [_rec(exit_reason="exit_signal", is_winner=True)]
    rates = compute_elimination_rates("th5", baseline, [])
    assert rates.stop_loss_removed_pct is None
    assert rates.removes_stop_losses_faster_than_winners is None


# ---------------------------------------------------------------------------
# compute_winner_preservation
# ---------------------------------------------------------------------------


def test_winner_preservation_basic():
    winner_kept = _rec(trade_number=1, is_winner=True, profit_abs=20.0, profit_pct=4.0)
    winner_removed = _rec(trade_number=2, is_winner=True, profit_abs=15.0, profit_pct=3.0)
    loser = _rec(trade_number=3, is_winner=False)
    removed = [winner_removed]
    retained = [winner_kept, loser]
    wp = compute_winner_preservation("th5", removed, retained)
    assert wp.baseline_winner_count == 2
    assert wp.winners_removed == (winner_removed,)
    assert wp.winners_retained == (winner_kept,)
    assert wp.winners_retained_pct == pytest.approx(50.0)
    assert wp.removed_winner_total_profit_abs == pytest.approx(15.0)
    assert wp.removed_winner_total_profit_pct == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# stratify_by_group
# ---------------------------------------------------------------------------


def test_stratify_by_group_pair_filter():
    baseline = [
        _rec(trade_number=1, pair="BTC/USDC:USDC", exit_reason="stop_loss", is_winner=False, ema_distance_pct=8.0),
        _rec(trade_number=2, pair="SOL/USDC:USDC", exit_reason="stop_loss", is_winner=False, ema_distance_pct=9.0),
    ]
    removed = baseline  # both removed
    strat = stratify_by_group(
        "BTC/USDC:USDC", "th5", baseline, removed, lambda r: r.pair == "BTC/USDC:USDC"
    )
    assert strat.baseline_count == 1
    assert strat.removed_count == 1
    assert strat.removed_ema_distance_mean == pytest.approx(8.0)
    assert strat.elimination_rates.baseline_stop_loss_count == 1
    assert strat.elimination_rates.removed_stop_loss_count == 1


# ---------------------------------------------------------------------------
# merge_mechanism_records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeTrade:
    pair: str
    direction: str
    entry_time: str
    exit_reason: str
    profit_pct: float
    profit_abs: float
    is_winner: bool


def test_merge_mechanism_records_joins_by_identity():
    trade = _FakeTrade(
        pair="BTC/USDC:USDC",
        direction="LONG",
        entry_time="2026-01-01T00:00:00",
        exit_reason="stop_loss",
        profit_pct=-5.0,
        profit_abs=-50.0,
        is_winner=False,
    )
    key = ("BTC/USDC:USDC", "2026-01-01T00:00:00", "LONG")
    sig_by_id = {
        key: {
            "trade_number": 7,
            "ema_distance_pct": 0.07,  # fraction -> 7.0 pct
            "breakout_distance_pct": 0.01,
            "adx14": 22.0,
        }
    }
    vol_by_id = {key: {"trade_number": 7, "atr_pct": 2.5, "realized_vol": 0.02}}

    [merged] = merge_mechanism_records([trade], sig_by_id, vol_by_id)
    assert merged.trade_number == 7
    assert merged.ema_distance_pct == pytest.approx(7.0)
    assert merged.breakout_distance_pct == pytest.approx(1.0)
    assert merged.adx14 == pytest.approx(22.0)
    assert merged.atr_pct == pytest.approx(2.5)
    assert merged.realized_vol == pytest.approx(0.02)
    assert merged.is_winner is False


def test_merge_mechanism_records_missing_lookup_stays_none():
    trade = _FakeTrade(
        pair="ETH/USDC:USDC",
        direction="SHORT",
        entry_time="2026-02-01T00:00:00",
        exit_reason="exit_signal",
        profit_pct=2.0,
        profit_abs=10.0,
        is_winner=True,
    )
    [merged] = merge_mechanism_records([trade], {}, {})
    assert merged.ema_distance_pct is None
    assert merged.breakout_distance_pct is None
    assert merged.adx14 is None
    assert merged.atr_pct is None
    assert merged.realized_vol is None
    assert merged.trade_number is None
    assert merged.is_winner is True
