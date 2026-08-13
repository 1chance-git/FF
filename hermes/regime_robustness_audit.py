"""BTC+ETH trend-following chronological regime robustness audit (research
only).

Answers one question only: does the extended BTC+ETH backtest's aggregate
result (positive, SHORT-driven) hold up when the tradeable period is split
into chronological quarters by entry order, or is it concentrated in one
favorable period / one or two extreme trades? This module never runs a
backtest, never touches `TrendFollowCore.py`, config, or the pair
whitelist, and never picks a threshold or recommends a strategy change --
it only segments and re-aggregates an already-loaded
`hermes.trade_report.Trade` list.

Design decisions
-----------------
* **Segmentation is by TRADE ORDER (entry time, ascending), split into
  ~equal-count quarters -- never by calendar date chosen after looking at
  P&L.** This is the same "mechanical segmentation decided before looking
  at results" discipline used by
  `hermes.ema_ceiling_temporal_audit.chronological_split` for the
  EARLY/LATE split; this module generalizes that same idea to N
  (default 4) roughly-equal groups via `chronological_quarters`, with any
  remainder distributed to the earliest quarters one trade at a time (so
  no quarter differs from another by more than one trade) -- a fixed,
  reproducible rule, not a judgment call.
* **Every quarter's stats reuse `hermes.extended_baseline_report`'s
  `SummaryStats`/`compute_summary_stats`/`split_by_direction` rather than
  recomputing win rate / P&L / profit factor a second, possibly
  inconsistent way.** This module only adds what that one doesn't already
  have: the segmentation itself, per-quarter date ranges, outlier-removal
  sensitivity, and a `THIN_SAMPLE` flag.
* **`THIN_SAMPLE` is a fixed, documented threshold (`THIN_SAMPLE_THRESHOLD
  = 5` trades), not a judgment call made per quarter.** A quarter at or
  below this count still gets every requested statistic computed and
  reported -- never suppressed -- but is flagged so a reader doesn't treat
  a 3-trade quarter's percentage return as comparable in weight to a
  15-trade quarter's.
* **Outlier sensitivity never claims significance.** `outlier_removal_pnl`
  reports total P&L with the top 1 and top 2 winners removed, purely
  descriptively -- it does not compute a p-value, a confidence interval,
  or a "would still be profitable" verdict; that judgment is left to the
  report text, per this research program's repeated instruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from hermes.extended_baseline_report import (
    SummaryStats,
    compute_summary_stats,
    date_range,
    split_by_direction,
)
from hermes.trade_report import Trade

THIN_SAMPLE_THRESHOLD = 5


def chronological_quarters(trades: Sequence[Trade], n_quarters: int = 4) -> list[list[Trade]]:
    """Split `trades` (sorted ascending by `entry_time`; trades with a
    `None` entry_time are excluded, never guessed) into `n_quarters`
    roughly-equal, contiguous groups by trade order -- never by a
    calendar date chosen after looking at performance. Any remainder from
    uneven division is given one extra trade each to the earliest
    quarters (a fixed, reproducible rule), so no quarter differs from
    another by more than one trade. Returns fewer than `n_quarters` groups
    only if there are fewer trades than `n_quarters`."""
    ordered = sorted((t for t in trades if t.entry_time is not None), key=lambda t: t.entry_time)
    n = len(ordered)
    if n == 0:
        return [[] for _ in range(n_quarters)]

    base_size, remainder = divmod(n, n_quarters)
    quarters: list[list[Trade]] = []
    start = 0
    for i in range(n_quarters):
        size = base_size + (1 if i < remainder else 0)
        quarters.append(ordered[start:start + size])
        start += size
    return quarters


@dataclass(frozen=True)
class OutlierSensitivity:
    largest_winner_pct: float | None
    second_largest_winner_pct: float | None
    total_pnl_pct: float | None
    total_pnl_excl_top1_pct: float | None
    total_pnl_excl_top2_pct: float | None


def outlier_removal_pnl(trades: Sequence[Trade]) -> OutlierSensitivity:
    """Total simple-sum P&L with the single largest winner, then the two
    largest winners, removed -- descriptive only, never a significance
    claim. Winners are ranked by `profit_pct` (trades with a `None`
    profit_pct are excluded from both the ranking and the totals)."""
    resolved = sorted(
        (t for t in trades if t.profit_pct is not None), key=lambda t: t.profit_pct, reverse=True
    )
    total = sum(t.profit_pct for t in resolved) if resolved else None
    winners = [t for t in resolved if t.profit_pct > 0]

    largest = winners[0].profit_pct if len(winners) >= 1 else None
    second = winners[1].profit_pct if len(winners) >= 2 else None

    excl1 = (total - largest) if (total is not None and largest is not None) else total
    excl2 = (excl1 - second) if (excl1 is not None and second is not None) else excl1

    return OutlierSensitivity(
        largest_winner_pct=largest,
        second_largest_winner_pct=second,
        total_pnl_pct=total,
        total_pnl_excl_top1_pct=excl1,
        total_pnl_excl_top2_pct=excl2,
    )


@dataclass(frozen=True)
class DirectionBreakdown:
    n: int
    total_profit_pct: float | None
    avg_profit_pct: float | None
    win_rate_pct: float | None
    stop_loss_count: int


def direction_breakdown(trades: Sequence[Trade], direction: str) -> DirectionBreakdown:
    """`SummaryStats`, narrowed to the four fields this block's DIRECTION
    CHECK asks for, for `direction`'s trades within `trades`."""
    stats = compute_summary_stats(split_by_direction(trades, direction))
    return DirectionBreakdown(
        n=stats.n,
        total_profit_pct=stats.total_profit_pct,
        avg_profit_pct=stats.avg_profit_pct,
        win_rate_pct=stats.win_rate_pct,
        stop_loss_count=stats.stop_loss_count,
    )


@dataclass(frozen=True)
class StopLossRates:
    overall_rate_pct: float | None
    long_rate_pct: float | None
    short_rate_pct: float | None


def stop_loss_rates(trades: Sequence[Trade]) -> StopLossRates:
    """Stop-loss trades / total trades, overall and split by direction.
    `None` (never `0.0`) when a group has zero trades -- an undefined
    rate is not the same as a zero rate."""
    def _rate(group: Sequence[Trade]) -> float | None:
        if not group:
            return None
        stop_losses = sum(1 for t in group if t.is_stop_loss is True)
        return 100.0 * stop_losses / len(group)

    return StopLossRates(
        overall_rate_pct=_rate(trades),
        long_rate_pct=_rate(split_by_direction(trades, "LONG")),
        short_rate_pct=_rate(split_by_direction(trades, "SHORT")),
    )


@dataclass(frozen=True)
class QuarterReport:
    index: int
    n_trades: int
    period_start: str | None
    period_end: str | None
    is_thin_sample: bool
    summary: SummaryStats
    long: DirectionBreakdown
    short: DirectionBreakdown
    stop_loss_rates: StopLossRates
    outliers: OutlierSensitivity
    max_winner_pct: float | None
    max_loser_pct: float | None


def build_quarter_report(index: int, trades: Sequence[Trade]) -> QuarterReport:
    """Every metric this block's per-quarter reporting requirement asks
    for, for one already-segmented quarter's trade list."""
    start, end = date_range(trades)
    summary = compute_summary_stats(trades)
    resolved_pcts = [t.profit_pct for t in trades if t.profit_pct is not None]

    return QuarterReport(
        index=index,
        n_trades=len(trades),
        period_start=start,
        period_end=end,
        is_thin_sample=(len(trades) <= THIN_SAMPLE_THRESHOLD),
        summary=summary,
        long=direction_breakdown(trades, "LONG"),
        short=direction_breakdown(trades, "SHORT"),
        stop_loss_rates=stop_loss_rates(trades),
        outliers=outlier_removal_pnl(trades),
        max_winner_pct=(max(resolved_pcts) if resolved_pcts else None),
        max_loser_pct=(min(resolved_pcts) if resolved_pcts else None),
    )


def build_all_quarter_reports(trades: Sequence[Trade], n_quarters: int = 4) -> list[QuarterReport]:
    """`build_quarter_report` for every quarter produced by
    `chronological_quarters(trades, n_quarters)`, in chronological order."""
    quarters = chronological_quarters(trades, n_quarters)
    return [build_quarter_report(i + 1, q) for i, q in enumerate(quarters)]
