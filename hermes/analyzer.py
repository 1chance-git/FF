"""Read-only analysis of Hermes' trade history: "what are we learning?"

Reads `hermes.memory.MemoryStore` and turns the raw trade log into a
small set of standard trading-performance statistics, plus a short list
of plain-English observations and hypotheses. It never writes to the
store, never touches strategy parameters, and never changes or deploys
anything — it answers a question, it doesn't act on the answer.

Design decisions
-----------------
* **Pure functions over a list of `TradeRecord`, not a class holding
  state.** Every statistic here is a function of the trade history and
  nothing else, so each is implemented as a small, independently
  testable function; `analyze()` composes them into one report rather
  than duplicating the math inline.
* **Only completed trades count.** A `TradeRecord` with `pnl is None`
  represents a trade Hermes doesn't yet know the outcome of (or never
  will, e.g. a bare process/error event never reaches this module in
  the first place). Every statistic here silently excludes those rather
  than treating a missing P&L as zero, which would understate expectancy
  and quietly corrupt every downstream number.
* **Empty or all-incomplete history is a valid input, not an error.**
  `analyze()` on zero completed trades returns a report with zeroed/`None`
  statistics and no findings — never raises — since Hermes may simply
  not have accumulated any history yet.
* **Findings are generated, not inferred by a model.** Every
  `[OBSERVATION]`/`[HYPOTHESIS]` pair here comes from a fixed,
  human-authored rule evaluated against the computed statistics (e.g.
  "a z-score bucket's expectancy is below the overall average by more
  than X" -> a specific templated sentence). There is no LLM or
  optimizer in this module, in keeping with its one job: describe what
  already happened, in plain language, for a human to decide whether to
  act on.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import mean

from hermes.memory import MemoryStore, TradeRecord

#: Z-score is bucketed to this width when grouping "performance by entry
#: z-score" — narrow enough to be informative, wide enough that a bucket
#: usually has more than one trade in it.
_ZSCORE_BUCKET_WIDTH = 0.5

#: A bucket's expectancy must be at least this much worse than the overall
#: expectancy (in absolute currency terms) before it's flagged as a finding,
#: so a single noisy trade in a bucket doesn't generate a spurious hypothesis.
_UNDERPERFORMANCE_THRESHOLD = 0.0

#: A bucket needs at least this many trades before its statistics are
#: considered meaningful enough to report a finding about.
_MIN_TRADES_FOR_FINDING = 3


@dataclass(frozen=True)
class BucketStats:
    """Trade statistics for one group of trades (e.g. one z-score bucket)."""

    label: str
    trade_count: int
    win_rate: float | None
    average_pnl: float | None
    expectancy: float | None


@dataclass(frozen=True)
class Finding:
    """One `[OBSERVATION]` and, optionally, its paired `[HYPOTHESIS]`."""

    observation: str
    hypothesis: str | None = None

    def render(self) -> str:
        """Render as the two-line `[OBSERVATION]`/`[HYPOTHESIS]` text format."""
        lines = [f"[OBSERVATION] {self.observation}"]
        if self.hypothesis:
            lines.append(f"[HYPOTHESIS] {self.hypothesis}")
        return "\n".join(lines)


@dataclass(frozen=True)
class AnalysisReport:
    """The full analysis result: headline statistics plus findings.

    Every statistic is `None` (not zero, not NaN) when there isn't
    enough data to compute it, so a caller can distinguish "we
    calculated this and it's zero" from "we have no basis to say".
    """

    trade_count: int
    win_rate: float | None
    average_pnl: float | None
    expectancy: float | None
    profit_factor: float | None
    max_drawdown: float | None
    average_holding_time_seconds: float | None
    total_fees: float
    total_funding: float
    fees_and_funding_drag_pct: float | None
    largest_losses: list[TradeRecord]
    max_consecutive_losses: int
    by_entry_zscore: list[BucketStats] = field(default_factory=list)
    by_exit_zscore: list[BucketStats] = field(default_factory=list)
    by_regime: list[BucketStats] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    def render_findings(self) -> str:
        """Render all findings as `[OBSERVATION]`/`[HYPOTHESIS]` text blocks."""
        return "\n\n".join(finding.render() for finding in self.findings)


def _completed(trades: list[TradeRecord]) -> list[TradeRecord]:
    """Trades with a known outcome; everything else is excluded from statistics."""
    return [t for t in trades if t.pnl is not None]


def win_rate(trades: list[TradeRecord]) -> float | None:
    """Fraction of completed trades with `pnl > 0`, or `None` if there are none."""
    completed = _completed(trades)
    if not completed:
        return None
    wins = sum(1 for t in completed if t.pnl > 0)
    return wins / len(completed)


def average_pnl(trades: list[TradeRecord]) -> float | None:
    """Mean P&L across completed trades, or `None` if there are none."""
    completed = _completed(trades)
    if not completed:
        return None
    return mean(t.pnl for t in completed)


def expectancy(trades: list[TradeRecord]) -> float | None:
    """Average P&L per trade, accounting for win rate and average win/loss size.

    `expectancy = win_rate * avg_win - loss_rate * avg_loss_magnitude`,
    the standard trading-system definition; equivalent to `average_pnl`
    but computed via win/loss decomposition so a caller can see both
    views. Returns `None` if there are no completed trades.
    """
    completed = _completed(trades)
    if not completed:
        return None
    wins = [t.pnl for t in completed if t.pnl > 0]
    losses = [t.pnl for t in completed if t.pnl <= 0]
    win_fraction = len(wins) / len(completed)
    loss_fraction = len(losses) / len(completed)
    avg_win = mean(wins) if wins else 0.0
    avg_loss_magnitude = abs(mean(losses)) if losses else 0.0
    return win_fraction * avg_win - loss_fraction * avg_loss_magnitude


def profit_factor(trades: list[TradeRecord]) -> float | None:
    """Gross profit / gross loss. `None` with no completed trades, `inf` with no losses."""
    completed = _completed(trades)
    if not completed:
        return None
    gross_profit = sum(t.pnl for t in completed if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in completed if t.pnl < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else None
    return gross_profit / gross_loss


def max_drawdown(trades: list[TradeRecord]) -> float | None:
    """Largest peak-to-trough drop in cumulative P&L, in currency terms.

    Trades are taken in `recorded_at` order (falling back to insertion
    order for trades with no timestamp) to build a cumulative equity
    curve; `None` if there are no completed trades.
    """
    completed = _completed(trades)
    if not completed:
        return None
    ordered = sorted(completed, key=lambda t: t.recorded_at or _epoch())
    cumulative = 0.0
    peak = 0.0
    worst_drawdown = 0.0
    for trade in ordered:
        cumulative += trade.pnl
        peak = max(peak, cumulative)
        worst_drawdown = min(worst_drawdown, cumulative - peak)
    return abs(worst_drawdown)


def _epoch() -> datetime:
    return datetime.min.replace(tzinfo=timezone.utc)


def average_holding_time_seconds(trades: list[TradeRecord]) -> float | None:
    """Mean holding time across trades that recorded one, or `None`."""
    values = [t.holding_time_seconds for t in trades if t.holding_time_seconds is not None]
    if not values:
        return None
    return mean(values)


def fees_and_funding_drag_pct(trades: list[TradeRecord]) -> tuple[float, float, float | None]:
    """Total fees, total funding, and their combined drag as a % of gross profit.

    Returns
    -------
    tuple[float, float, float | None]
        `(total_fees, total_funding, drag_pct)`. `drag_pct` is `None`
        when there's no gross profit to express the drag as a
        percentage of.
    """
    completed = _completed(trades)
    total_fees = sum(t.fees for t in completed if t.fees is not None)
    total_funding = sum(t.funding for t in completed if t.funding is not None)
    gross_profit = sum(t.pnl for t in completed if t.pnl > 0)
    if gross_profit == 0:
        return total_fees, total_funding, None
    drag_pct = (total_fees + abs(total_funding)) / gross_profit * 100.0
    return total_fees, total_funding, drag_pct


def largest_losses(trades: list[TradeRecord], top_n: int = 5) -> list[TradeRecord]:
    """The `top_n` worst completed trades by P&L, most negative first."""
    completed = [t for t in _completed(trades) if t.pnl < 0]
    return sorted(completed, key=lambda t: t.pnl)[:top_n]


def max_consecutive_losses(trades: list[TradeRecord]) -> int:
    """Longest streak of consecutive losing trades, in `recorded_at` order."""
    completed = _completed(trades)
    if not completed:
        return 0
    ordered = sorted(completed, key=lambda t: t.recorded_at or _epoch())
    longest = current = 0
    for trade in ordered:
        if trade.pnl < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _bucket_stats(groups: dict[str, list[TradeRecord]]) -> list[BucketStats]:
    stats = []
    for label, group in sorted(groups.items()):
        stats.append(
            BucketStats(
                label=label,
                trade_count=len(group),
                win_rate=win_rate(group),
                average_pnl=average_pnl(group),
                expectancy=expectancy(group),
            )
        )
    return stats


def _zscore_bucket_label(value: float) -> str:
    lower = (
        int(value / _ZSCORE_BUCKET_WIDTH)
        if value >= 0
        else -(int(-value / _ZSCORE_BUCKET_WIDTH) + 1)
    ) * _ZSCORE_BUCKET_WIDTH
    upper = lower + _ZSCORE_BUCKET_WIDTH
    return f"{lower:.1f} to {upper:.1f}"


def performance_by_entry_zscore(trades: list[TradeRecord]) -> list[BucketStats]:
    """Win rate/avg P&L/expectancy grouped into entry z-score buckets."""
    groups: dict[str, list[TradeRecord]] = defaultdict(list)
    for trade in _completed(trades):
        if trade.entry_zscore is not None:
            groups[_zscore_bucket_label(trade.entry_zscore)].append(trade)
    return _bucket_stats(groups)


def performance_by_exit_zscore(trades: list[TradeRecord]) -> list[BucketStats]:
    """Win rate/avg P&L/expectancy grouped into exit z-score buckets."""
    groups: dict[str, list[TradeRecord]] = defaultdict(list)
    for trade in _completed(trades):
        if trade.exit_zscore is not None:
            groups[_zscore_bucket_label(trade.exit_zscore)].append(trade)
    return _bucket_stats(groups)


def performance_by_regime(trades: list[TradeRecord]) -> list[BucketStats]:
    """Win rate/avg P&L/expectancy grouped by recorded market regime."""
    groups: dict[str, list[TradeRecord]] = defaultdict(list)
    for trade in _completed(trades):
        if trade.regime is not None:
            groups[trade.regime].append(trade)
    return _bucket_stats(groups)


def _generate_findings(
    overall_expectancy: float | None,
    by_entry_zscore: list[BucketStats],
    by_exit_zscore: list[BucketStats],
    by_regime: list[BucketStats],
    consecutive_losses: int,
) -> list[Finding]:
    """Fixed, human-authored rules over the computed statistics; no inference."""
    findings: list[Finding] = []
    if overall_expectancy is not None:
        for dimension_name, buckets, param_name in (
            ("entry z-score", by_entry_zscore, "entry threshold"),
            ("exit z-score", by_exit_zscore, "exit threshold"),
        ):
            for bucket in buckets:
                if bucket.trade_count < _MIN_TRADES_FOR_FINDING:
                    continue
                if bucket.expectancy is None:
                    continue
                shortfall = overall_expectancy - bucket.expectancy
                if shortfall > _UNDERPERFORMANCE_THRESHOLD:
                    findings.append(
                        Finding(
                            observation=(
                                f"Trades with {dimension_name} between {bucket.label} "
                                f"(n={bucket.trade_count}) had expectancy {bucket.expectancy:.2f}, "
                                f"below the overall {overall_expectancy:.2f}."
                            ),
                            hypothesis=(
                                f"A stricter {param_name} around this range may improve "
                                "trade quality — worth investigating further, not acting on "
                                "directly."
                            ),
                        )
                    )

        best_regime = max(
            (b for b in by_regime if b.trade_count >= _MIN_TRADES_FOR_FINDING and b.expectancy is not None),
            key=lambda b: b.expectancy,
            default=None,
        )
        worst_regime = min(
            (b for b in by_regime if b.trade_count >= _MIN_TRADES_FOR_FINDING and b.expectancy is not None),
            key=lambda b: b.expectancy,
            default=None,
        )
        if best_regime is not None and worst_regime is not None and best_regime.label != worst_regime.label:
            findings.append(
                Finding(
                    observation=(
                        f"Regime '{best_regime.label}' (n={best_regime.trade_count}) had expectancy "
                        f"{best_regime.expectancy:.2f}, vs. '{worst_regime.label}' "
                        f"(n={worst_regime.trade_count}) at {worst_regime.expectancy:.2f}."
                    ),
                    hypothesis=(
                        f"Entries during '{worst_regime.label}' may warrant tighter risk gating "
                        "or exclusion — worth investigating further."
                    ),
                )
            )

    if consecutive_losses >= _MIN_TRADES_FOR_FINDING:
        findings.append(
            Finding(
                observation=f"The longest losing streak in this history is {consecutive_losses} trades.",
                hypothesis=(
                    "A streak this long may warrant a cooldown or exposure review during "
                    "similar conditions — worth investigating further."
                ),
            )
        )

    return findings


def analyze(store: MemoryStore) -> AnalysisReport:
    """Compute the full performance report from a Hermes memory store.

    Reads the entire trade history via `store.get_trades()`; never
    writes anything, never raises on an empty or trade-less store.

    Parameters
    ----------
    store:
        The memory store to read from.

    Returns
    -------
    AnalysisReport
        Statistics plus a list of `[OBSERVATION]`/`[HYPOTHESIS]` findings.
    """
    trades = store.get_trades()
    completed = _completed(trades)

    total_fees, total_funding, drag_pct = fees_and_funding_drag_pct(trades)
    by_entry_zscore = performance_by_entry_zscore(trades)
    by_exit_zscore = performance_by_exit_zscore(trades)
    by_regime = performance_by_regime(trades)
    overall_expectancy = expectancy(trades)
    consecutive_losses = max_consecutive_losses(trades)

    findings = _generate_findings(
        overall_expectancy, by_entry_zscore, by_exit_zscore, by_regime, consecutive_losses
    )

    return AnalysisReport(
        trade_count=len(completed),
        win_rate=win_rate(trades),
        average_pnl=average_pnl(trades),
        expectancy=overall_expectancy,
        profit_factor=profit_factor(trades),
        max_drawdown=max_drawdown(trades),
        average_holding_time_seconds=average_holding_time_seconds(trades),
        total_fees=total_fees,
        total_funding=total_funding,
        fees_and_funding_drag_pct=drag_pct,
        largest_losses=largest_losses(trades),
        max_consecutive_losses=consecutive_losses,
        by_entry_zscore=by_entry_zscore,
        by_exit_zscore=by_exit_zscore,
        by_regime=by_regime,
        findings=findings,
    )
