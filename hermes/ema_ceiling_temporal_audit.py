"""EMA200-ceiling TEMPORAL ROBUSTNESS audit (research/analysis only).

Answers one question only: does the EMA200-distance entry-ceiling effect
already measured by `hermes.ema_ceiling_forensics` /
`hermes.ema_ceiling_quality_audit` hold across time, or is it concentrated
in one portion of the sample? This module never launches a backtest,
never picks a "winning" threshold, and never touches `TrendFollowCore.py`,
config, or strategy entry/exit/stoploss logic -- it only computes pure
arithmetic over already-loaded trade lists (the
`hermes.trade_report.Trade` interface: `.entry_time`, `.profit_pct`,
`.is_winner`, ...) and reuses `hermes.ema_ceiling_quality_audit`'s
profit-factor/drawdown/median helpers rather than re-deriving them.

Two independent pieces live here:

1. **Chronological split** -- a pure, time-ordered (never random) split
   of a baseline trade list into an EARLY and a LATE period, by trade
   count (first ~50% / remaining ~50%, ordered by `entry_time`), the
   split the spec requires for Step 2.
2. **Temporal robustness classification** -- given a baseline period
   metric and a ceiling-variant period metric for the EARLY period and
   again for the LATE period, classify the ceiling's effect as one of
   ROBUST / MIXED / FAILED / INSUFFICIENT (Step 4), using a minimum
   trade-count floor so a 1-2 trade period never manufactures a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

# Minimum number of *baseline* trades a period must contain before this
# module will classify a ceiling's effect in that period at all -- below
# this, "improved" or "worsened" is noise, not signal, per the spec's
# explicit INSUFFICIENT category (Step 4).
MIN_TRADES_FOR_CLASSIFICATION = 5


@dataclass(frozen=True)
class ChronologicalSplit:
    """A trade list split by TIME (entry_time order), never by random
    assignment -- the EARLY/LATE halves Step 2 asks for."""

    early: tuple[Any, ...]
    late: tuple[Any, ...]
    split_index: int
    early_start: str | None
    early_end: str | None
    late_start: str | None
    late_end: str | None


def chronological_split(
    trades: Sequence[Any], early_fraction: float = 0.5
) -> ChronologicalSplit:
    """Split `trades` into EARLY (first ~`early_fraction`) / LATE (rest),
    ordered by `.entry_time` (ISO strings sort chronologically).

    The split point is by trade COUNT (first ~50% of trades chronologically),
    the "preferred split" the spec names for Step 2 -- not a calendar
    midpoint, which could land on a period with wildly different trade
    density. Ties (`early_fraction * n` not an integer) round down for the
    early half via `int()` truncation, so the late half never has fewer
    trades than the early half for an odd count.
    """
    if not trades:
        return ChronologicalSplit(
            early=(), late=(), split_index=0,
            early_start=None, early_end=None, late_start=None, late_end=None,
        )
    ordered = sorted(trades, key=lambda t: t.entry_time)
    split_index = int(len(ordered) * early_fraction)
    early = tuple(ordered[:split_index])
    late = tuple(ordered[split_index:])
    return ChronologicalSplit(
        early=early,
        late=late,
        split_index=split_index,
        early_start=early[0].entry_time if early else None,
        early_end=early[-1].entry_time if early else None,
        late_start=late[0].entry_time if late else None,
        late_end=late[-1].entry_time if late else None,
    )


# ---------------------------------------------------------------------------
# Temporal robustness classification (Step 4) -- pure, no search
# ---------------------------------------------------------------------------

TemporalVerdict = str  # one of the constants below, kept as plain str

ROBUST = "ROBUST"
MIXED = "MIXED"
FAILED = "FAILED"
INSUFFICIENT = "INSUFFICIENT"


def _improved(baseline_value: float | None, variant_value: float | None) -> bool | None:
    """`True` if `variant_value` is a genuine improvement over
    `baseline_value` (higher = better, e.g. profit factor, avg P&L).
    `None` if either side is unavailable (can't be judged)."""
    if baseline_value is None or variant_value is None:
        return None
    return variant_value > baseline_value


def classify_temporal_robustness(
    *,
    early_baseline_trades: int,
    late_baseline_trades: int,
    early_baseline_metric: float | None,
    early_variant_metric: float | None,
    late_baseline_metric: float | None,
    late_variant_metric: float | None,
    min_trades: int = MIN_TRADES_FOR_CLASSIFICATION,
) -> TemporalVerdict:
    """Classify one ceiling's effect on one metric (e.g. average P&L/trade
    or profit factor) as ROBUST / MIXED / FAILED / INSUFFICIENT, per Step 4:

    - INSUFFICIENT: either period has fewer than `min_trades` baseline
      trades, or a metric could not be computed in either period (e.g.
      profit factor undefined with zero losers).
    - ROBUST: improvement present in BOTH the early and late periods.
    - FAILED: the ceiling worsens (or fails to improve) the metric in
      BOTH periods.
    - MIXED: improvement in exactly one period but not the other.
    """
    if early_baseline_trades < min_trades or late_baseline_trades < min_trades:
        return INSUFFICIENT

    early_improved = _improved(early_baseline_metric, early_variant_metric)
    late_improved = _improved(late_baseline_metric, late_variant_metric)

    if early_improved is None or late_improved is None:
        return INSUFFICIENT

    if early_improved and late_improved:
        return ROBUST
    if not early_improved and not late_improved:
        return FAILED
    return MIXED


# ---------------------------------------------------------------------------
# Trade-set stability (Step 7): removal-ratio + sequencing classification
# ---------------------------------------------------------------------------


def classify_removal_mode(
    *,
    baseline_losers: int,
    baseline_winners: int,
    removed_losers: int,
    removed_winners: int,
    tolerance_pct: float = 15.0,
) -> str:
    """A/B/C label for Step 7: is a ceiling primarily removing losers (A),
    primarily removing winners (B), or removing both roughly
    proportionally (C)?

    Computed from the *ratio* of each bucket removed (removed / baseline
    count in that bucket), not raw counts, since baseline winner/loser
    counts are themselves unequal (8 winners vs 31 losers) -- comparing
    raw removed counts would always look loser-dominated regardless of
    whether the ceiling is actually loser-selective. `tolerance_pct` is
    the max percentage-point gap between the two removal ratios still
    called "proportional" (C); a larger gap favoring the loser ratio is A,
    favoring the winner ratio is B. Returns "INSUFFICIENT" if either
    baseline bucket is empty (a ratio can't be computed).
    """
    if baseline_losers == 0 or baseline_winners == 0:
        return "INSUFFICIENT"
    loser_ratio_pct = 100.0 * removed_losers / baseline_losers
    winner_ratio_pct = 100.0 * removed_winners / baseline_winners
    gap = loser_ratio_pct - winner_ratio_pct
    if abs(gap) <= tolerance_pct:
        return "C_BOTH_PROPORTIONALLY"
    if gap > 0:
        return "A_PRIMARILY_LOSERS"
    return "B_PRIMARILY_WINNERS"
