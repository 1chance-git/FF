"""LONG vs SHORT directional forensics audit (research/analysis only).

Answers one question only: within the now-frozen BTC+ETH TrendFollowCore
universe (see `hermes.og_universe_isolation_audit` and the pair-list
freeze in commit a25b4f8), is the previously observed LONG/SHORT P&L
asymmetry structural, regime-dependent, asset-specific, outlier-driven,
or a small-sample artifact -- and, as far as the already-persisted
exports allow, where in the entry/development/exit pipeline it
originates?

This module never launches a backtest, never touches
`TrendFollowCore.py`, config, or the pair whitelist, and never picks or
recommends a parameter change -- it only filters and aggregates the
already-loaded baseline trade list (`hermes.trade_report.Trade`) and
the already-joined mechanism records
(`hermes.ema_ceiling_mechanism_audit.MechanismTradeRecord`) by
`.direction`, reusing:

* `hermes.og_universe_isolation_audit.filter_by_pairs` /
  `BTC_PAIR` / `ETH_PAIR` / `OG_PAIRS` for pair scoping,
* `hermes.ema_ceiling_temporal_audit.chronological_split` for the
  EARLY/LATE split (same methodology as the isolation audit -- no new
  segmentation scheme invented here),
* `hermes.ema_ceiling_mechanism_audit.{merge_mechanism_records,
  classify_outcome_category, compare_metric, CATEGORY_STOP_LOSS}` for
  the EMA-distance/ADX/breakout join and outcome classification,

rather than re-deriving any of that arithmetic a second time.

Five independent pieces live here:

1. **`split_by_direction`** -- restrict a trade list to LONG or SHORT,
   the one primitive every other function in this module builds on.
2. **`DirectionalStats` / `compute_directional_stats`** -- the fuller
   scorecard this block's questions need beyond
   `og_universe_isolation_audit.compute_basic_stats`: gross profit,
   gross loss, largest winner, largest loser, and expectancy per trade,
   in addition to win rate / P&L / profit factor / stop count.
3. **`outlier_sensitivity`** -- recompute a direction's totals with its
   N largest-magnitude trades removed, so "would the conclusion change
   without the top 1-2 trades" can be answered directly rather than
   eyeballed.
4. **`compare_duration_by_direction`** -- mean/median
   `duration_minutes` for LONG vs SHORT; the only entry/development
   signal the frozen trade export actually carries (the export has no
   MFE/MAE field, so that part of Question 5 is reported as missing
   evidence, never fabricated -- see the module's test suite and the
   forensic report for the explicit call-out).
5. **`ema_distance_by_direction`** -- Question 6: does the existing
   stop-loss-vs-exit-signal EMA-distance separation
   (`hermes.ema_ceiling_mechanism_audit`) differ between LONG and
   SHORT trades, restricted to BTC+ETH.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Sequence

from hermes.ema_ceiling_mechanism_audit import (
    CATEGORY_STOP_LOSS,
    MechanismTradeRecord,
    classify_outcome_category,
    compare_metric,
    merge_mechanism_records,
)

LONG = "LONG"
SHORT = "SHORT"


def split_by_direction(trades: Sequence[Any]) -> tuple[list[Any], list[Any]]:
    """`(long_trades, short_trades)` -- any trade whose `.direction` is
    neither `"LONG"` nor `"SHORT"` (missing/unexpected) is excluded from
    both, never guessed into one bucket."""
    longs = [t for t in trades if t.direction == LONG]
    shorts = [t for t in trades if t.direction == SHORT]
    return longs, shorts


# ---------------------------------------------------------------------------
# DirectionalStats (Subagent A's scorecard)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectionalStats:
    trade_count: int
    winners: int
    losers: int
    win_rate_pct: float | None
    total_profit_pct: float | None
    total_profit_abs: float | None
    avg_profit_pct: float | None
    median_profit_pct: float | None
    profit_factor: float | None
    gross_profit_abs: float | None
    gross_loss_abs: float | None
    largest_winner_pct: float | None
    largest_loser_pct: float | None
    expectancy_pct: float | None
    stop_loss_count: int


def _pct(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return 100.0 * numerator / denominator


def compute_directional_stats(trades: Sequence[Any]) -> DirectionalStats:
    """Full scorecard for one trade subset. `expectancy_pct` is the mean
    `profit_pct` across all trades with a known value (identical to
    `avg_profit_pct` -- kept as a separate, explicitly-named field
    because "expectancy" is what Question 1/Subagent A ask for by name,
    and a reader should not have to infer the two are the same
    quantity). All P&L fields exclude `None` entries rather than
    treating them as zero; `profit_factor` is `None` when there are no
    losers (undefined, never reported as infinite)."""
    n = len(trades)
    winners = [t for t in trades if t.is_winner is True]
    losers = [t for t in trades if t.is_winner is False]

    profit_pcts = [t.profit_pct for t in trades if t.profit_pct is not None]
    profit_abses = [t.profit_abs for t in trades if t.profit_abs is not None]
    winner_pcts = [t.profit_pct for t in winners if t.profit_pct is not None]
    loser_pcts = [t.profit_pct for t in losers if t.profit_pct is not None]

    gross_profit = sum(t.profit_abs for t in winners if t.profit_abs is not None)
    gross_loss = sum(t.profit_abs for t in losers if t.profit_abs is not None)
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < 0 else None

    return DirectionalStats(
        trade_count=n,
        winners=len(winners),
        losers=len(losers),
        win_rate_pct=_pct(len(winners), n),
        total_profit_pct=(sum(profit_pcts) if profit_pcts else None),
        total_profit_abs=(sum(profit_abses) if profit_abses else None),
        avg_profit_pct=((sum(profit_pcts) / len(profit_pcts)) if profit_pcts else None),
        median_profit_pct=(median(profit_pcts) if profit_pcts else None),
        profit_factor=profit_factor,
        gross_profit_abs=(gross_profit if profit_abses else None),
        gross_loss_abs=(gross_loss if profit_abses else None),
        largest_winner_pct=(max(winner_pcts) if winner_pcts else None),
        largest_loser_pct=(min(loser_pcts) if loser_pcts else None),
        expectancy_pct=((sum(profit_pcts) / len(profit_pcts)) if profit_pcts else None),
        stop_loss_count=sum(1 for t in trades if t.exit_reason == "stop_loss"),
    )


# ---------------------------------------------------------------------------
# Outlier sensitivity (Subagent C / Question 4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutlierSensitivity:
    n_removed: int
    removed_trade_profit_pcts: tuple[float, ...]
    full_total_profit_pct: float | None
    without_outliers_total_profit_pct: float | None
    full_avg_profit_pct: float | None
    without_outliers_avg_profit_pct: float | None

    @property
    def conclusion_would_flip(self) -> bool | None:
        """`True` if removing the outliers flips the *sign* of the
        total P&L (the only "would the conclusion change" test this
        function claims to answer -- it does not judge magnitude
        changes as a flip). `None` if either total is unavailable."""
        if self.full_total_profit_pct is None or self.without_outliers_total_profit_pct is None:
            return None
        full_sign = self.full_total_profit_pct > 0
        without_sign = self.without_outliers_total_profit_pct > 0
        return full_sign != without_sign


def outlier_sensitivity(trades: Sequence[Any], n_remove: int = 2) -> OutlierSensitivity:
    """Recompute a direction's total/avg P&L with its `n_remove`
    largest-*magnitude* `profit_pct` trades removed (winners or losers,
    whichever moves the total most, ranked by `abs(profit_pct)` -- not
    "top winners only", since a single large loser can dominate a
    direction's total just as easily as a large winner)."""
    with_pnl = [t for t in trades if t.profit_pct is not None]
    ranked = sorted(with_pnl, key=lambda t: abs(t.profit_pct), reverse=True)
    removed = ranked[:n_remove]
    remaining = ranked[n_remove:]

    full_total = sum(t.profit_pct for t in with_pnl) if with_pnl else None
    full_avg = (full_total / len(with_pnl)) if with_pnl else None
    remaining_total = sum(t.profit_pct for t in remaining) if remaining else None
    remaining_avg = (remaining_total / len(remaining)) if remaining else None

    return OutlierSensitivity(
        n_removed=len(removed),
        removed_trade_profit_pcts=tuple(t.profit_pct for t in removed),
        full_total_profit_pct=full_total,
        without_outliers_total_profit_pct=remaining_total,
        full_avg_profit_pct=full_avg,
        without_outliers_avg_profit_pct=remaining_avg,
    )


# ---------------------------------------------------------------------------
# Duration comparison (Subagent D / Question 5 -- the one entry/development
# signal actually present in the frozen trade export)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DurationComparison:
    long_mean_minutes: float | None
    long_median_minutes: float | None
    long_n: int
    short_mean_minutes: float | None
    short_median_minutes: float | None
    short_n: int


def compare_duration_by_direction(long_trades: Sequence[Any], short_trades: Sequence[Any]) -> DurationComparison:
    long_durations = [t.duration_minutes for t in long_trades if t.duration_minutes is not None]
    short_durations = [t.duration_minutes for t in short_trades if t.duration_minutes is not None]
    return DurationComparison(
        long_mean_minutes=(sum(long_durations) / len(long_durations)) if long_durations else None,
        long_median_minutes=(median(long_durations) if long_durations else None),
        long_n=len(long_durations),
        short_mean_minutes=(sum(short_durations) / len(short_durations)) if short_durations else None,
        short_median_minutes=(median(short_durations) if short_durations else None),
        short_n=len(short_durations),
    )


# ---------------------------------------------------------------------------
# EMA-distance-by-direction (Question 6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectionalEmaFinding:
    direction: str
    metric: str
    stop_loss_mean: float | None
    stop_loss_median: float | None
    stop_loss_n: int
    exit_signal_mean: float | None
    exit_signal_median: float | None
    exit_signal_n: int


def ema_distance_by_direction(
    trades: Sequence[Any],
    signal_by_identity: dict[tuple, dict[str, Any]],
    volatility_by_identity: dict[tuple, dict[str, Any]],
) -> list[DirectionalEmaFinding]:
    """For each of LONG and SHORT, and each of `ema_distance_pct` /
    `breakout_distance_pct` / `adx14`, compare stop-loss trades against
    exit-signal trades -- the same comparison
    `hermes.ema_ceiling_mechanism_audit.compare_metric` already makes
    for the whole universe, just restricted to one direction at a time,
    so Question 6 can be answered without re-deriving the join."""
    findings: list[DirectionalEmaFinding] = []
    long_trades, short_trades = split_by_direction(trades)
    for direction, direction_trades in ((LONG, long_trades), (SHORT, short_trades)):
        records = merge_mechanism_records(direction_trades, signal_by_identity, volatility_by_identity)
        sl_records = [r for r in records if classify_outcome_category(r) == CATEGORY_STOP_LOSS]
        exit_signal_records = [r for r in records if r.exit_reason == "exit_signal"]
        for metric in ("ema_distance_pct", "breakout_distance_pct", "adx14"):
            cmp = compare_metric(metric, sl_records, exit_signal_records)
            findings.append(
                DirectionalEmaFinding(
                    direction=direction,
                    metric=metric,
                    stop_loss_mean=cmp.removed_mean,
                    stop_loss_median=cmp.removed_median,
                    stop_loss_n=cmp.removed_n,
                    exit_signal_mean=cmp.retained_mean,
                    exit_signal_median=cmp.retained_median,
                    exit_signal_n=cmp.retained_n,
                )
            )
    return findings
