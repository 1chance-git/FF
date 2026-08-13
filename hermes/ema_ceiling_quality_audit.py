"""EMA200-ceiling trade-set QUALITY / ROBUSTNESS audit (research/analysis only).

Adds the calculations `hermes.ema_ceiling_forensics` does not already
provide, needed for the "is this a genuine quality improvement, or just
trade removal?" audit: profit factor, sequential-equity maximum drawdown,
median P&L per retained trade, and explicit removed-winner /
removed-loser trade listings with the ratios the audit spec asks for.

This module never launches a backtest, never chooses/recommends a
threshold, and never touches `TrendFollowCore.py`, config, or strategy
entry/exit/stoploss logic -- it only computes pure arithmetic over
already-loaded `hermes.trade_report.Trade` objects (or the equivalent
`BaselineTradeRecord`s from `hermes.ema_ceiling_forensics`).
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# Profit factor / drawdown / median -- pure arithmetic over a P&L series
# ---------------------------------------------------------------------------


def profit_factor(profit_values: Sequence[float]) -> float | None:
    """Gross profit / gross loss (absolute), over any P&L series.

    Returns `None` if there are no losing values (division by zero is
    undefined, not infinity-by-convention here) or the series is empty.
    """
    if not profit_values:
        return None
    gross_profit = sum(v for v in profit_values if v > 0)
    gross_loss = -sum(v for v in profit_values if v < 0)
    if gross_loss == 0:
        return None
    return gross_profit / gross_loss


def median_pnl(profit_values: Sequence[float]) -> float | None:
    if not profit_values:
        return None
    return median(profit_values)


def max_drawdown_pct(chronological_profit_values: Sequence[float]) -> float | None:
    """Maximum peak-to-trough drawdown (in the same units as the input,
    e.g. percentage points) over the sequential equity curve built by
    cumulatively summing `chronological_profit_values` **in the order
    given** -- callers must already have sorted trades chronologically
    (e.g. by `entry_time`) before calling this.

    Equity curve starts at 0 (not 100) since Freqtrade `profit_pct`/
    `profit_ratio` values are additive P&L deltas, not portfolio levels;
    drawdown is measured as (peak - trough) along that curve, so it is
    directly comparable across variants with different starting trade
    counts. Returns `None` for an empty series.
    """
    if not chronological_profit_values:
        return None
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for v in chronological_profit_values:
        equity += v
        if equity > peak:
            peak = equity
        drawdown = peak - equity
        if drawdown > max_dd:
            max_dd = drawdown
    return max_dd


def average_duration_minutes(durations: Sequence[float]) -> float | None:
    values = [d for d in durations if d is not None]
    if not values:
        return None
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# Removed-trade classification (baseline trades not present in a variant)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RemovedTradeSummary:
    """Winner/loser split of the baseline trades absent from a ceiling
    variant's kept set, plus the ratios the WINNER-REMOVAL ANALYSIS asks
    for."""

    removed_trades: tuple[Any, ...]
    removed_winners: tuple[Any, ...]
    removed_losers: tuple[Any, ...]

    @property
    def removed_count(self) -> int:
        return len(self.removed_trades)

    @property
    def removed_winner_count(self) -> int:
        return len(self.removed_winners)

    @property
    def removed_loser_count(self) -> int:
        return len(self.removed_losers)

    @property
    def removed_winner_profit_pct(self) -> float:
        return sum(
            t.profit_pct for t in self.removed_winners if t.profit_pct is not None
        )

    @property
    def removed_loser_loss_pct(self) -> float:
        return sum(
            t.profit_pct for t in self.removed_losers if t.profit_pct is not None
        )

    @property
    def removed_loser_ratio(self) -> float | None:
        """removed-losers / removed-trades."""
        if not self.removed_trades:
            return None
        return self.removed_loser_count / self.removed_count

    @property
    def removed_winner_ratio(self) -> float | None:
        """removed-winners / removed-trades."""
        if not self.removed_trades:
            return None
        return self.removed_winner_count / self.removed_count


def _trade_identity(trade: Any) -> tuple:
    """A stable identity tuple for matching a trade across two exports of
    the *same* backtest window/pairs -- (pair, entry_time, direction) is
    unique per trade in this dataset (no pyramiding/duplicate entries at
    the same candle observed in the frozen baseline)."""
    return (
        getattr(trade, "pair", None),
        getattr(trade, "entry_time", None),
        getattr(trade, "direction", None),
    )


def removed_trades_summary(
    baseline_trades: Sequence[Any], variant_trades: Sequence[Any]
) -> RemovedTradeSummary:
    """Baseline trades whose identity does not appear in `variant_trades`."""
    kept_identities = {_trade_identity(t) for t in variant_trades}
    removed = [t for t in baseline_trades if _trade_identity(t) not in kept_identities]
    winners = tuple(t for t in removed if t.is_winner is True)
    losers = tuple(t for t in removed if t.is_winner is False)
    return RemovedTradeSummary(
        removed_trades=tuple(removed), removed_winners=winners, removed_losers=losers
    )


# ---------------------------------------------------------------------------
# Per-variant full metric bundle (question 1 / question 3 of the audit)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantMetrics:
    label: str
    trades: int
    winners: int
    losers: int
    win_rate_pct: float | None
    total_pnl_pct: float | None
    average_pnl_pct: float | None
    median_pnl_pct: float | None
    profit_factor: float | None
    max_drawdown_pct: float | None
    stop_loss_exits: int
    exit_signal_exits: int
    force_exits: int
    other_exits: int
    avg_duration_minutes: float | None


def compute_variant_metrics(label: str, trades: Sequence[Any]) -> VariantMetrics:
    """Full metric bundle for one variant's trade list.

    `trades` must expose `.profit_pct`, `.is_winner`, `.exit_reason`,
    `.duration_minutes`, `.entry_time` (the `hermes.trade_report.Trade`
    interface). For drawdown, trades are sorted by `entry_time` (string
    ISO timestamps sort chronologically) before building the equity
    curve -- callers do not need to pre-sort.
    """
    known = [t for t in trades if t.is_winner is not None]
    winners = sum(1 for t in known if t.is_winner)
    losers = sum(1 for t in known if not t.is_winner)
    profit_values = [t.profit_pct for t in trades if t.profit_pct is not None]

    chronological = sorted(
        (t for t in trades if t.profit_pct is not None and t.entry_time is not None),
        key=lambda t: t.entry_time,
    )
    chrono_values = [t.profit_pct for t in chronological]

    exit_counts: dict[str, int] = {}
    for t in trades:
        key = t.exit_reason or "N/A"
        exit_counts[key] = exit_counts.get(key, 0) + 1

    return VariantMetrics(
        label=label,
        trades=len(trades),
        winners=winners,
        losers=losers,
        win_rate_pct=(100.0 * winners / len(known)) if known else None,
        total_pnl_pct=sum(profit_values) if profit_values else None,
        average_pnl_pct=(sum(profit_values) / len(profit_values)) if profit_values else None,
        median_pnl_pct=median_pnl(profit_values),
        profit_factor=profit_factor(profit_values),
        max_drawdown_pct=max_drawdown_pct(chrono_values),
        stop_loss_exits=exit_counts.get("stop_loss", 0),
        exit_signal_exits=exit_counts.get("exit_signal", 0),
        force_exits=exit_counts.get("force_exit", 0),
        other_exits=sum(
            v
            for k, v in exit_counts.items()
            if k not in ("stop_loss", "exit_signal", "force_exit")
        ),
        avg_duration_minutes=average_duration_minutes(
            [t.duration_minutes for t in trades]
        ),
    )


def compute_group_metrics(
    label: str, trades: Sequence[Any], predicate
) -> VariantMetrics:
    """`compute_variant_metrics` restricted to trades matching `predicate`
    -- the per-pair / per-direction breakdown helper (questions 4 and 5)."""
    return compute_variant_metrics(label, [t for t in trades if predicate(t)])


# ---------------------------------------------------------------------------
# Threshold-stability classification (question 7) -- pure, no search
# ---------------------------------------------------------------------------


def classify_stability(
    per_threshold_win_rate_pct: Sequence[float],
    per_threshold_profit_factor: Sequence[float | None],
) -> str:
    """ROBUST / MIXED / FRAGILE based on monotonic-ish persistence of
    improvement across the ordered (ascending threshold) sequences given.

    This does NOT search for or select a threshold -- it only classifies
    the shape of an already-fixed sequence of six results. "ROBUST" means
    both series stay non-decreasing (or flat) with only minor local dips
    (<=1 inversion) across the whole ordered sequence; "FRAGILE" means the
    direction reverses in the majority of adjacent steps; otherwise
    "MIXED".
    """
    def _inversions(series: Sequence[float | None]) -> int:
        clean = [v for v in series if v is not None]
        count = 0
        for a, b in zip(clean, clean[1:]):
            if b < a:
                count += 1
        return count

    n_steps = max(len(per_threshold_win_rate_pct) - 1, 1)
    wr_inv = _inversions(per_threshold_win_rate_pct)
    pf_inv = _inversions(per_threshold_profit_factor)
    worst = max(wr_inv, pf_inv)

    if worst <= 1:
        return "ROBUST"
    if worst >= (n_steps + 1) // 2:
        return "FRAGILE"
    return "MIXED"
