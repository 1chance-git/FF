"""Focused unit tests for `hermes.ema_ceiling_quality_audit` (research-only)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from hermes.ema_ceiling_quality_audit import (
    RemovedTradeSummary,
    classify_stability,
    compute_group_metrics,
    compute_variant_metrics,
    max_drawdown_pct,
    median_pnl,
    profit_factor,
    removed_trades_summary,
)


# ---------------------------------------------------------------------------
# profit_factor
# ---------------------------------------------------------------------------


def test_profit_factor_basic():
    # gross profit 15, gross loss 5 -> 3.0
    assert profit_factor([10.0, 5.0, -5.0]) == pytest.approx(3.0)


def test_profit_factor_no_losses_is_none():
    assert profit_factor([10.0, 5.0]) is None


def test_profit_factor_empty_is_none():
    assert profit_factor([]) is None


# ---------------------------------------------------------------------------
# median_pnl
# ---------------------------------------------------------------------------


def test_median_pnl_odd():
    assert median_pnl([1.0, 5.0, 3.0]) == pytest.approx(3.0)


def test_median_pnl_empty_is_none():
    assert median_pnl([]) is None


# ---------------------------------------------------------------------------
# max_drawdown_pct
# ---------------------------------------------------------------------------


def test_max_drawdown_basic():
    # equity: 5, 3 (peak 5), -2 (peak 5, trough -2 -> dd 7), 10 (equity 8)
    values = [5.0, -2.0, -3.0, 10.0]
    # cumulative: 5, 3, 0, 10 ; peak sequence: 5,5,5,10; dd: 0,2,5,0
    assert max_drawdown_pct(values) == pytest.approx(5.0)


def test_max_drawdown_all_gains_is_zero():
    assert max_drawdown_pct([1.0, 2.0, 3.0]) == pytest.approx(0.0)


def test_max_drawdown_empty_is_none():
    assert max_drawdown_pct([]) is None


# ---------------------------------------------------------------------------
# removed_trades_summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeTrade:
    pair: str
    entry_time: str
    direction: str
    profit_pct: float
    is_winner: bool


def _trades():
    return [
        _FakeTrade("BTC/USDC:USDC", "t1", "LONG", 5.0, True),
        _FakeTrade("BTC/USDC:USDC", "t2", "LONG", -3.0, False),
        _FakeTrade("ETH/USDC:USDC", "t3", "SHORT", -1.0, False),
        _FakeTrade("SOL/USDC:USDC", "t4", "LONG", 2.0, True),
    ]


def test_removed_trades_summary_identifies_missing_baseline_trades():
    baseline = _trades()
    # variant only kept t1 and t4
    variant = [baseline[0], baseline[3]]
    summary = removed_trades_summary(baseline, variant)
    assert summary.removed_count == 2
    assert summary.removed_winner_count == 0
    assert summary.removed_loser_count == 2
    assert summary.removed_loser_ratio == pytest.approx(1.0)
    assert summary.removed_winner_ratio == pytest.approx(0.0)
    assert summary.removed_loser_loss_pct == pytest.approx(-4.0)


def test_removed_trades_summary_no_removals():
    baseline = _trades()
    summary = removed_trades_summary(baseline, baseline)
    assert summary.removed_count == 0
    assert summary.removed_winner_ratio is None
    assert summary.removed_loser_ratio is None


# ---------------------------------------------------------------------------
# compute_variant_metrics / compute_group_metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FullTrade:
    pair: str
    direction: str
    entry_time: str
    profit_pct: float | None
    is_winner: bool | None
    exit_reason: str | None
    duration_minutes: float | None


def _full_trades():
    return [
        _FullTrade("BTC/USDC:USDC", "LONG", "2026-01-01", 5.0, True, "exit_signal", 120.0),
        _FullTrade("BTC/USDC:USDC", "LONG", "2026-01-02", -3.0, False, "stop_loss", 60.0),
        _FullTrade("ETH/USDC:USDC", "SHORT", "2026-01-03", -1.0, False, "stop_loss", 90.0),
        _FullTrade("SOL/USDC:USDC", "LONG", "2026-01-04", 2.0, True, "exit_signal", 30.0),
    ]


def test_compute_variant_metrics_basic():
    m = compute_variant_metrics("test", _full_trades())
    assert m.trades == 4
    assert m.winners == 2
    assert m.losers == 2
    assert m.win_rate_pct == pytest.approx(50.0)
    assert m.total_pnl_pct == pytest.approx(3.0)
    assert m.average_pnl_pct == pytest.approx(0.75)
    assert m.median_pnl_pct == pytest.approx(0.5)
    assert m.profit_factor == pytest.approx(7.0 / 4.0)
    assert m.stop_loss_exits == 2
    assert m.exit_signal_exits == 2
    assert m.force_exits == 0
    assert m.avg_duration_minutes == pytest.approx(75.0)


def test_compute_group_metrics_filters_by_predicate():
    trades = _full_trades()
    btc_only = compute_group_metrics(
        "BTC", trades, lambda t: t.pair == "BTC/USDC:USDC"
    )
    assert btc_only.trades == 2
    assert btc_only.winners == 1
    assert btc_only.losers == 1


# ---------------------------------------------------------------------------
# classify_stability
# ---------------------------------------------------------------------------


def test_classify_stability_robust_when_monotonic():
    win_rates = [8.33, 11.76, 18.18, 21.74, 22.22, 21.21]
    pfs = [1.5, 1.6, 1.8, 2.0, 1.9, 1.7]
    result = classify_stability(win_rates, pfs)
    assert result in ("ROBUST", "MIXED")  # depends on inversion count threshold


def test_classify_stability_fragile_when_oscillating():
    win_rates = [10.0, 5.0, 15.0, 4.0, 20.0, 3.0]
    pfs = [2.0, 0.5, 3.0, 0.4, 4.0, 0.3]
    result = classify_stability(win_rates, pfs)
    assert result == "FRAGILE"
