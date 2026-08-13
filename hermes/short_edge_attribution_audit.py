"""BTC+ETH SHORT-edge attribution audit (research/analysis only).

Answers one question only: for the extended 39-trade BTC+ETH sample, does
SHORT's persistent P&L advantage over LONG come primarily from (A) better
entries, (B) greater favorable development after entry, (C) lower adverse
excursion, (D) longer persistence, (E) better exit timing, (F) a
combination, or (G) the evidence can't distinguish these -- using MFE/MAE
reconstruction, trade-development classification, duration, and
outlier-removal sensitivity, split by direction, pair, and chronological
quarter.

This module never runs a backtest, never touches `TrendFollowCore.py`,
config, or the pair whitelist, and never optimizes anything. It is a thin
composition layer over already-existing, already-tested primitives:

* `hermes.mfe_mae_forensics` for MFE/MAE reconstruction and A/B/C/D
  trade-development classification -- reused verbatim, not reimplemented.
* `hermes.extended_baseline_report` for `SummaryStats`/`compute_summary_stats`/
  `split_by_direction`/`split_by_pair`.
* `hermes.regime_robustness_audit` for `chronological_quarters`.

The only genuinely new logic here is: joining a `Trade` to its MFE/MAE
result (`attach_mfe_mae`), removing the top-N winners for outlier
robustness at an arbitrary N (`remove_top_n_winners`), and grouping
MFE/MAE results by pair+direction or by quarter+direction
(`group_mfe_mae_by`). Nothing here re-derives P&L, win rate, or profit
factor a second way.

Design decisions
-----------------
* **`attach_mfe_mae` never fabricates a result for a trade whose OHLCV
  window can't be sliced.** A trade with an unresolved `MfeMaeResult`
  (missing candles, bad timestamps) is still included in the returned
  list -- callers that need only resolved results filter with
  `resolved_only`, rather than this function silently dropping data.
* **`remove_top_n_winners` ranks by `profit_pct` only, generalizing
  `hermes.regime_robustness_audit.outlier_removal_pnl`'s fixed top-1/top-2
  to an arbitrary `n`** (this block's spec asks for top-1/2/3), without
  duplicating that module's own top-2-only logic -- callers that only need
  top-1/top-2 can keep using the existing function; this one exists
  because top-3 is newly required here.
* **No causal language anywhere in this module.** Every returned value is
  a descriptive statistic (a mean, a count, a rate); interpretive labels
  like "entry-quality signal" vs. "development/exit signal" are left
  entirely to the report text, never encoded as a boolean or a verdict
  field here.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Sequence

from hermes.extended_baseline_report import (
    SummaryStats,
    compute_summary_stats,
    split_by_direction,
    split_by_pair,
)
from hermes.mfe_mae_forensics import MfeMaeResult, compute_mfe_mae, slice_trade_window
from hermes.trade_report import Trade

import pandas as pd


# ---------------------------------------------------------------------------
# Joining trades to their MFE/MAE result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradeMfeMae:
    trade: Trade
    result: MfeMaeResult


def attach_mfe_mae(
    trades: Sequence[Trade], ohlcv_by_pair: dict[str, pd.DataFrame]
) -> list[TradeMfeMae]:
    """`compute_mfe_mae` for every trade in `trades`, using
    `ohlcv_by_pair[trade.pair]` as the candle source. A trade whose pair
    has no entry in `ohlcv_by_pair`, or whose window can't be sliced, gets
    an unresolved `MfeMaeResult` (via `compute_mfe_mae`'s own missing-data
    handling) -- never dropped, never silently zero-filled."""
    joined: list[TradeMfeMae] = []
    for trade in trades:
        ohlcv = ohlcv_by_pair.get(trade.pair) if trade.pair else None
        window = slice_trade_window(ohlcv, trade.entry_time, trade.exit_time) if ohlcv is not None else None
        result = compute_mfe_mae(trade.direction, trade.entry_price, window)
        joined.append(TradeMfeMae(trade=trade, result=result))
    return joined


def resolved_only(joined: Sequence[TradeMfeMae]) -> list[TradeMfeMae]:
    """`joined` filtered to entries whose `MfeMaeResult.is_resolved` is `True`."""
    return [j for j in joined if j.result.is_resolved]


# ---------------------------------------------------------------------------
# MFE/MAE aggregate stats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MfeMaeAggregate:
    n: int
    mean_mfe_pct: float | None
    median_mfe_pct: float | None
    mean_mae_pct: float | None
    median_mae_pct: float | None


def aggregate_mfe_mae(joined: Sequence[TradeMfeMae]) -> MfeMaeAggregate:
    """Mean/median MFE and MAE across the resolved entries of `joined`."""
    resolved = resolved_only(joined)
    mfes = [j.result.mfe_pct for j in resolved]
    maes = [j.result.mae_pct for j in resolved]
    return MfeMaeAggregate(
        n=len(resolved),
        mean_mfe_pct=(mean(mfes) if mfes else None),
        median_mfe_pct=(median(mfes) if mfes else None),
        mean_mae_pct=(mean(maes) if maes else None),
        median_mae_pct=(median(maes) if maes else None),
    )


def filter_by_winner(joined: Sequence[TradeMfeMae], is_winner: bool) -> list[TradeMfeMae]:
    """Entries whose `.trade.is_winner` matches `is_winner` exactly (never
    matching a `None`/unresolved outcome to either `True` or `False`)."""
    return [j for j in joined if j.trade.is_winner is is_winner]


# ---------------------------------------------------------------------------
# Duration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DurationStats:
    n: int
    mean_minutes: float | None
    median_minutes: float | None


def duration_stats(trades: Sequence[Trade]) -> DurationStats:
    """Mean/median `duration_minutes` across `trades` with a known
    duration (`None` values excluded, never treated as zero)."""
    durations = [t.duration_minutes for t in trades if t.duration_minutes is not None]
    return DurationStats(
        n=len(durations),
        mean_minutes=(mean(durations) if durations else None),
        median_minutes=(median(durations) if durations else None),
    )


# ---------------------------------------------------------------------------
# Outlier robustness: remove top-N winners by profit_pct
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutlierRobustnessResult:
    n_removed: int
    total_profit_pct: float | None
    avg_profit_pct: float | None
    median_profit_pct: float | None


def remove_top_n_winners(trades: Sequence[Trade], n: int) -> list[Trade]:
    """`trades` with the `n` largest-`profit_pct` trades removed (ties
    broken by original order; trades with a `None` profit_pct are never
    candidates for removal and are always kept). `n=0` returns `trades`
    unchanged (as a new list)."""
    if n <= 0:
        return list(trades)
    resolved = [t for t in trades if t.profit_pct is not None]
    ranked = sorted(resolved, key=lambda t: t.profit_pct, reverse=True)
    to_remove = set(id(t) for t in ranked[:n])
    return [t for t in trades if id(t) not in to_remove]


def outlier_robustness_series(trades: Sequence[Trade], max_n: int = 3) -> list[OutlierRobustnessResult]:
    """`OutlierRobustnessResult` for removing 0, 1, ..., `max_n` largest
    winners in turn (index 0 = no removal, the unmodified baseline)."""
    results = []
    for n in range(max_n + 1):
        remaining = remove_top_n_winners(trades, n)
        stats = compute_summary_stats(remaining)
        results.append(
            OutlierRobustnessResult(
                n_removed=n,
                total_profit_pct=stats.total_profit_pct,
                avg_profit_pct=stats.avg_profit_pct,
                median_profit_pct=stats.median_profit_pct,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Pair x direction / quarter x direction grouping
# ---------------------------------------------------------------------------


def group_trades_by_pair_direction(
    trades: Sequence[Trade], pairs: Sequence[str], directions: Sequence[str] = ("LONG", "SHORT")
) -> dict[tuple[str, str], list[Trade]]:
    """`{(pair, direction): [trades]}` for every combination of `pairs` x
    `directions` -- an empty list (never a missing key) for a combination
    with no trades."""
    groups: dict[tuple[str, str], list[Trade]] = {}
    for pair in pairs:
        pair_trades = split_by_pair(trades, pair)
        for direction in directions:
            groups[(pair, direction)] = split_by_direction(pair_trades, direction)
    return groups


def group_mfe_mae_by_pair_direction(
    joined: Sequence[TradeMfeMae], pairs: Sequence[str], directions: Sequence[str] = ("LONG", "SHORT")
) -> dict[tuple[str, str], list[TradeMfeMae]]:
    """Same grouping as `group_trades_by_pair_direction`, but over an
    already-joined `TradeMfeMae` list (so MFE/MAE results travel with
    their trade)."""
    groups: dict[tuple[str, str], list[TradeMfeMae]] = {}
    for pair in pairs:
        for direction in directions:
            groups[(pair, direction)] = [
                j for j in joined if j.trade.pair == pair and j.trade.direction == direction
            ]
    return groups
