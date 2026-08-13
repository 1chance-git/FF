"""OG universe (BTC + ETH) vs SOL isolation audit (research/analysis only).

Answers one question only: within the already-frozen 39-trade baseline
(BTC + ETH + SOL, TrendFollowCore), does restricting the universe to just
BTC and ETH produce a materially different -- and specifically, materially
*cleaner* -- trade sample than the full 3-pair set, or is SOL's apparent
drag not clearly separable from noise at this sample size?

This module never launches a backtest, never adds/removes an entry/exit
condition, and never touches `TrendFollowCore.py`, config, or any pair
list -- it only filters and aggregates the already-loaded baseline trade
list (`hermes.trade_report.Trade`) and the already-joined mechanism
records (`hermes.ema_ceiling_mechanism_audit.MechanismTradeRecord`) by
`.pair`, reusing `hermes.ema_ceiling_temporal_audit.chronological_split`
for the EARLY/LATE split and
`hermes.ema_ceiling_mechanism_audit.{classify_outcome_category,compare_metric}`
for outcome classification and EMA-distance/ADX/breakout comparison,
rather than re-deriving any of that arithmetic.

Three independent pieces live here:

1. **`filter_by_pairs`** -- restrict any trade/record list to one or more
   pairs, the one primitive every other function in this module builds on.
2. **`compute_basic_stats`** -- trade count, win rate, total/avg/median
   P&L, profit factor, and exit-reason breakdown for a trade subset --
   the same shape Step 1 and Step 4 (per-asset LONG/SHORT) both need, so
   they can't silently diverge.
3. **`compute_contribution`** -- one group's (e.g. SOL's) share of a
   whole set's trades, P&L, winners, losers, and stop-losses, framed as
   "contributed X% of Y%", never as a causal claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Sequence

BTC_PAIR = "BTC/USDC:USDC"
ETH_PAIR = "ETH/USDC:USDC"
SOL_PAIR = "SOL/USDC:USDC"
OG_PAIRS = (BTC_PAIR, ETH_PAIR)


def filter_by_pairs(trades: Sequence[Any], pairs: Sequence[str]) -> list[Any]:
    """Restrict `trades` (anything with a `.pair` attribute) to `pairs`."""
    pair_set = set(pairs)
    return [t for t in trades if t.pair in pair_set]


# ---------------------------------------------------------------------------
# Basic stats (Step 1 / per-asset Step 4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BasicStats:
    trade_count: int
    winners: int
    losers: int
    win_rate_pct: float | None
    total_profit_pct: float | None
    total_profit_abs: float | None
    avg_profit_pct: float | None
    median_profit_pct: float | None
    profit_factor: float | None
    stop_loss_count: int
    exit_signal_count: int
    force_exit_count: int
    unresolved_count: int


def _pct(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return 100.0 * numerator / denominator


def compute_basic_stats(trades: Sequence[Any]) -> BasicStats:
    """Aggregate stats for a trade subset. `profit_factor` is
    `sum(winning profit_abs) / abs(sum(losing profit_abs))`, `None` if
    there are no losers (undefined, never reported as infinite) or no
    trades carry `profit_abs`. Every P&L figure excludes `None` entries
    rather than treating them as zero."""
    n = len(trades)
    winners = [t for t in trades if t.is_winner is True]
    losers = [t for t in trades if t.is_winner is False]
    unresolved = [t for t in trades if t.is_winner is None]

    profit_pcts = [t.profit_pct for t in trades if t.profit_pct is not None]
    profit_abses = [t.profit_abs for t in trades if t.profit_abs is not None]

    win_gross = sum(t.profit_abs for t in winners if t.profit_abs is not None)
    loss_gross = sum(t.profit_abs for t in losers if t.profit_abs is not None)
    profit_factor = (win_gross / abs(loss_gross)) if loss_gross < 0 else None

    return BasicStats(
        trade_count=n,
        winners=len(winners),
        losers=len(losers),
        win_rate_pct=_pct(len(winners), n),
        total_profit_pct=sum(profit_pcts) if profit_pcts else None,
        total_profit_abs=sum(profit_abses) if profit_abses else None,
        avg_profit_pct=(sum(profit_pcts) / len(profit_pcts)) if profit_pcts else None,
        median_profit_pct=median(profit_pcts) if profit_pcts else None,
        profit_factor=profit_factor,
        stop_loss_count=sum(1 for t in trades if t.exit_reason == "stop_loss"),
        exit_signal_count=sum(1 for t in trades if t.exit_reason == "exit_signal"),
        force_exit_count=sum(1 for t in trades if t.exit_reason == "force_exit"),
        unresolved_count=len(unresolved),
    )


# ---------------------------------------------------------------------------
# Contribution (Step 3 -- descriptive share, never causal)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Contribution:
    group_label: str
    trade_share_pct: float | None
    profit_pct_share_pct: float | None
    winner_share_pct: float | None
    loser_share_pct: float | None
    stop_loss_share_pct: float | None
    avg_trade_result_pct: float | None


def compute_contribution(group_label: str, group: Sequence[Any], whole: Sequence[Any]) -> Contribution:
    """`group`'s share of `whole` across trade count, aggregate P&L
    (`profit_pct` sum), winners, losers, and stop-losses. Purely
    descriptive: this function computes a percentage-of-total, not a
    counterfactual ("if SOL were removed, P&L would be...") -- that
    inference is left to the caller/report text, and even there must use
    non-causal phrasing per the spec."""
    group_stats = compute_basic_stats(group)
    whole_stats = compute_basic_stats(whole)

    whole_profit_pct_sum = whole_stats.total_profit_pct
    group_profit_pct_sum = group_stats.total_profit_pct
    profit_share = None
    if whole_profit_pct_sum not in (None, 0) and group_profit_pct_sum is not None:
        profit_share = 100.0 * group_profit_pct_sum / whole_profit_pct_sum

    return Contribution(
        group_label=group_label,
        trade_share_pct=_pct(group_stats.trade_count, whole_stats.trade_count),
        profit_pct_share_pct=profit_share,
        winner_share_pct=_pct(group_stats.winners, whole_stats.winners),
        loser_share_pct=_pct(group_stats.losers, whole_stats.losers),
        stop_loss_share_pct=_pct(group_stats.stop_loss_count, whole_stats.stop_loss_count),
        avg_trade_result_pct=group_stats.avg_profit_pct,
    )
