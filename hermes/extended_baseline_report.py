"""Descriptive summary/comparison helpers for the extended BTC+ETH baseline
backtest (research only).

Answers one question only: given a list of already-backtested trades (an
already-loaded `hermes.trade_report.Trade` sequence -- this module never
runs a backtest itself, never touches Freqtrade, and never re-derives P&L),
what do the required summary statistics (trade count, win rate, P&L,
profit factor, stop-loss/exit-signal counts) look like, split by direction
or by pair -- and how does one such summary compare, purely descriptively,
to another (e.g. the frozen 39/23-trade baseline vs. an extended-sample
run)?

Design decisions
-----------------
* **No significance testing, no "which is better" verdict.** `compare_summaries`
  reports the raw deltas between two `SummaryStats`; it never computes a
  p-value, a confidence interval, or a conclusion -- the caller (a human,
  or a report-writing step) is responsible for any interpretation, per
  this research program's repeated instruction not to overinterpret small
  samples.
* **`profit_factor` is `None`, never `inf` or a fabricated large number,
  when there are zero losing trades.** Reporting `None` for an undefined
  ratio is more honest than pretending an arbitrarily large number is
  meaningful.
* **`split_by_direction`/`split_by_pair` are pure filters, reused from the
  same pattern as `hermes.long_short_directional_audit.split_by_direction`
  and `hermes.og_universe_isolation_audit.filter_by_pairs`** -- not
  reimplemented differently here, just narrowed to a single pair/direction
  each since that's all this block's questions need.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Sequence

from hermes.trade_report import Trade


@dataclass(frozen=True)
class SummaryStats:
    n: int
    winners: int
    losers: int
    win_rate_pct: float | None
    total_profit_pct: float | None
    total_profit_abs: float | None
    avg_profit_pct: float | None
    median_profit_pct: float | None
    profit_factor: float | None
    gross_profit_pct: float | None
    gross_loss_pct: float | None
    stop_loss_count: int
    exit_signal_count: int


def split_by_direction(trades: Sequence[Trade], direction: str) -> list[Trade]:
    """Trades whose `.direction` equals `direction` exactly (case-sensitive,
    matching Freqtrade's own `"LONG"`/`"SHORT"` export values)."""
    return [t for t in trades if t.direction == direction]


def split_by_pair(trades: Sequence[Trade], pair: str) -> list[Trade]:
    """Trades whose `.pair` equals `pair` exactly."""
    return [t for t in trades if t.pair == pair]


def compute_summary_stats(trades: Sequence[Trade]) -> SummaryStats:
    """The required descriptive scorecard for `trades`: count, win rate,
    P&L (total/avg/median), profit factor, and stop-loss/exit-signal
    counts. Trades with an unknown `profit_pct`/`profit_abs` (still open,
    or a malformed export row) are excluded from the P&L aggregates but
    still counted in `n`; `winners`/`losers` only count trades where
    `is_winner` is not `None`."""
    n = len(trades)
    winners = sum(1 for t in trades if t.is_winner is True)
    losers = sum(1 for t in trades if t.is_winner is False)
    resolved = winners + losers

    profit_pcts = [t.profit_pct for t in trades if t.profit_pct is not None]
    profit_abs_values = [t.profit_abs for t in trades if t.profit_abs is not None]

    gross_profit = sum(p for p in profit_pcts if p > 0)
    gross_loss = sum(p for p in profit_pcts if p < 0)

    return SummaryStats(
        n=n,
        winners=winners,
        losers=losers,
        win_rate_pct=(100.0 * winners / resolved) if resolved else None,
        total_profit_pct=(sum(profit_pcts) if profit_pcts else None),
        total_profit_abs=(sum(profit_abs_values) if profit_abs_values else None),
        avg_profit_pct=(mean(profit_pcts) if profit_pcts else None),
        median_profit_pct=(median(profit_pcts) if profit_pcts else None),
        profit_factor=(gross_profit / abs(gross_loss)) if gross_loss < 0 else None,
        gross_profit_pct=(gross_profit if profit_pcts else None),
        gross_loss_pct=(gross_loss if profit_pcts else None),
        stop_loss_count=sum(1 for t in trades if t.is_stop_loss is True),
        exit_signal_count=sum(1 for t in trades if t.exit_reason == "exit_signal"),
    )


def date_range(trades: Sequence[Trade]) -> tuple[str | None, str | None]:
    """`(earliest_entry_time, latest_exit_time)` across `trades`, as the raw
    string values already on each `Trade` (no re-parsing/re-formatting) --
    `(None, None)` for an empty list."""
    entries = [t.entry_time for t in trades if t.entry_time is not None]
    exits = [t.exit_time for t in trades if t.exit_time is not None]
    if not entries or not exits:
        return None, None
    return min(entries), max(exits)


@dataclass(frozen=True)
class SummaryComparison:
    n_delta: int
    win_rate_pct_delta: float | None
    total_profit_pct_delta: float | None
    avg_profit_pct_delta: float | None
    profit_factor_delta: float | None
    stop_loss_count_delta: int


def compare_summaries(extended: SummaryStats, baseline: SummaryStats) -> SummaryComparison:
    """`extended` minus `baseline`, field by field -- a raw descriptive
    delta, never a significance claim or a verdict."""
    def _delta(a: float | None, b: float | None) -> float | None:
        if a is None or b is None:
            return None
        return a - b

    return SummaryComparison(
        n_delta=extended.n - baseline.n,
        win_rate_pct_delta=_delta(extended.win_rate_pct, baseline.win_rate_pct),
        total_profit_pct_delta=_delta(extended.total_profit_pct, baseline.total_profit_pct),
        avg_profit_pct_delta=_delta(extended.avg_profit_pct, baseline.avg_profit_pct),
        profit_factor_delta=_delta(extended.profit_factor, baseline.profit_factor),
        stop_loss_count_delta=extended.stop_loss_count - baseline.stop_loss_count,
    )
