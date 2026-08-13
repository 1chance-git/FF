"""Focused unit tests for `hermes.og_universe_isolation_audit` (research-only)."""

from __future__ import annotations

from dataclasses import dataclass

from hermes.og_universe_isolation_audit import (
    BTC_PAIR,
    ETH_PAIR,
    OG_PAIRS,
    SOL_PAIR,
    compute_basic_stats,
    compute_contribution,
    filter_by_pairs,
)


@dataclass(frozen=True)
class _T:
    pair: str
    exit_reason: str | None
    profit_pct: float | None
    profit_abs: float | None

    @property
    def is_winner(self) -> bool | None:
        if self.profit_abs is None:
            return None
        return self.profit_abs > 0


def _btc(profit_pct, profit_abs, exit_reason="exit_signal"):
    return _T(BTC_PAIR, exit_reason, profit_pct, profit_abs)


def _eth(profit_pct, profit_abs, exit_reason="exit_signal"):
    return _T(ETH_PAIR, exit_reason, profit_pct, profit_abs)


def _sol(profit_pct, profit_abs, exit_reason="exit_signal"):
    return _T(SOL_PAIR, exit_reason, profit_pct, profit_abs)


# ---------------------------------------------------------------------------
# filter_by_pairs
# ---------------------------------------------------------------------------


def test_filter_by_pairs_keeps_only_matching():
    trades = [_btc(1, 1), _eth(1, 1), _sol(1, 1)]
    filtered = filter_by_pairs(trades, OG_PAIRS)
    assert {t.pair for t in filtered} == {BTC_PAIR, ETH_PAIR}
    assert len(filtered) == 2


def test_filter_by_pairs_empty_result():
    trades = [_sol(1, 1)]
    assert filter_by_pairs(trades, OG_PAIRS) == []


# ---------------------------------------------------------------------------
# compute_basic_stats
# ---------------------------------------------------------------------------


def test_compute_basic_stats_reconciles_counts():
    trades = [
        _btc(-5.0, -50.0, "stop_loss"),
        _btc(10.0, 100.0, "exit_signal"),
        _eth(-2.0, -20.0, "exit_signal"),
        _eth(3.0, 30.0, "force_exit"),
    ]
    stats = compute_basic_stats(trades)
    assert stats.trade_count == 4
    assert stats.winners == 2
    assert stats.losers == 2
    assert stats.win_rate_pct == 50.0
    assert stats.stop_loss_count == 1
    assert stats.exit_signal_count == 2
    assert stats.force_exit_count == 1
    assert stats.unresolved_count == 0
    assert stats.total_profit_pct == 6.0
    assert stats.avg_profit_pct == 1.5
    assert stats.median_profit_pct == 0.5


def test_compute_basic_stats_profit_factor():
    trades = [_btc(10.0, 100.0), _btc(-5.0, -50.0), _eth(-5.0, -50.0)]
    stats = compute_basic_stats(trades)
    assert stats.profit_factor == 1.0  # 100 / (50+50)


def test_compute_basic_stats_profit_factor_none_with_no_losers():
    trades = [_btc(10.0, 100.0), _eth(5.0, 50.0)]
    stats = compute_basic_stats(trades)
    assert stats.profit_factor is None


def test_compute_basic_stats_empty():
    stats = compute_basic_stats([])
    assert stats.trade_count == 0
    assert stats.win_rate_pct is None
    assert stats.total_profit_pct is None
    assert stats.profit_factor is None


def test_compute_basic_stats_missing_profit_abs_not_treated_as_zero():
    trades = [_T(BTC_PAIR, "exit_signal", 5.0, None)]
    stats = compute_basic_stats(trades)
    assert stats.unresolved_count == 1
    assert stats.total_profit_pct == 5.0  # profit_pct still known
    assert stats.total_profit_abs is None  # profit_abs unknown, not zero


# ---------------------------------------------------------------------------
# compute_contribution
# ---------------------------------------------------------------------------


def test_compute_contribution_shares():
    whole = [
        _btc(-5.0, -50.0, "stop_loss"),
        _eth(10.0, 100.0, "exit_signal"),
        _sol(-8.0, -80.0, "stop_loss"),
        _sol(-3.0, -30.0, "exit_signal"),
    ]
    sol = filter_by_pairs(whole, [SOL_PAIR])
    contrib = compute_contribution("SOL", sol, whole)
    assert contrib.trade_share_pct == 50.0  # 2 of 4
    # 3 losers total (BTC stop_loss + both SOL trades); SOL contributed 2 of 3
    assert contrib.loser_share_pct == 100.0 * 2 / 3
    assert contrib.stop_loss_share_pct == 50.0  # 1 of 2 stop-losses
    assert contrib.winner_share_pct == 0.0  # SOL contributed 0 of 1 winner


def test_compute_contribution_profit_share_sign_reflects_drag():
    whole = [_btc(10.0, 100.0), _sol(-15.0, -150.0)]
    sol = filter_by_pairs(whole, [SOL_PAIR])
    contrib = compute_contribution("SOL", sol, whole)
    # whole total_profit_pct = -5.0, SOL's = -15.0 -> share = 300%
    assert contrib.profit_pct_share_pct == 300.0


def test_compute_contribution_handles_empty_group():
    whole = [_btc(5.0, 50.0)]
    sol = filter_by_pairs(whole, [SOL_PAIR])
    contrib = compute_contribution("SOL", sol, whole)
    assert contrib.trade_share_pct == 0.0
    assert contrib.profit_pct_share_pct is None  # group has no profit_pct trades
