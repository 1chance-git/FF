"""Performance reporting: summary statistics and formatted output for backtest/walk-forward results.

Built entirely on `freqtrade.data.metrics` — the same mature,
already-tested Sharpe/Sortino/Calmar/drawdown/expectancy/SQN
implementations Freqtrade's own backtest report and hyperopt loss
functions use — rather than recomputing any of that math independently.
This module's job is assembling those statistics into one
`PerformanceReport` and rendering it (via `rich`, matching `hermes`'s
CLI-output style) or a walk-forward run's per-window breakdown.

Design decisions
-----------------
* **Every statistic here is a direct call into `freqtrade.data.metrics`,
  never a reimplementation.** Sharpe, Sortino, Calmar, max drawdown,
  expectancy, and SQN each have real subtleties (annualization
  convention, whether drawdown is computed on equity or on trade
  P&L, degrees-of-freedom in the ratio's denominator) that Freqtrade's
  own implementations already get right and this project's other test
  suites (`test_backtest_validation.py`) already exercise against.
* **`compute_performance_report` accepts a plain trades DataFrame**, the
  same shape Freqtrade's own backtest results use (`close_date`,
  `profit_abs`, `pair`, ...) — not a `Backtesting` object or a
  `BacktestResult` — so it composes with any of: a real `Backtesting`
  run's `results["strategy"][name]["trades"]`, a walk-forward window's
  test-period trades, or a hand-built DataFrame in a test.
* **Rendering is `rich`-based, consistent with `hermes.cli`.** Both
  tools are meant to be used together operationally; sharing a visual
  style (tables via `rich.table.Table`) means output from one doesn't
  look like it came from a different, unrelated project.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from rich.console import Console
from rich.table import Table

from freqtrade.data.metrics import (
    calculate_calmar,
    calculate_expectancy,
    calculate_max_drawdown,
    calculate_sharpe,
    calculate_sortino,
    calculate_sqn,
)

logger = logging.getLogger(__name__)


class ReportingError(Exception):
    """Raised when a report cannot be computed (e.g. no trades)."""


@dataclass(frozen=True)
class PerformanceReport:
    """Summary performance statistics for a set of trades.

    Attributes
    ----------
    total_trades:
        Number of trades.
    win_rate:
        Fraction of trades with positive `profit_abs`.
    total_profit_abs:
        Sum of `profit_abs` across all trades.
    total_profit_pct:
        `total_profit_abs` as a fraction of `starting_balance`.
    sharpe, sortino, calmar:
        Risk-adjusted return ratios (`freqtrade.data.metrics`).
    max_drawdown_pct, max_drawdown_abs:
        Maximum peak-to-trough drawdown, relative and absolute.
    expectancy, expectancy_ratio:
        Average profit per trade, and profit-factor-style ratio
        (`freqtrade.data.metrics.calculate_expectancy`).
    sqn:
        System Quality Number (`freqtrade.data.metrics.calculate_sqn`).
    per_pair_profit:
        Mapping of pair -> total `profit_abs` for that pair.
    """

    total_trades: int
    win_rate: float
    total_profit_abs: float
    total_profit_pct: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown_pct: float
    max_drawdown_abs: float
    expectancy: float
    expectancy_ratio: float
    sqn: float
    per_pair_profit: dict[str, float]


def compute_performance_report(
    trades: pd.DataFrame,
    starting_balance: float,
    min_date: datetime | None = None,
    max_date: datetime | None = None,
) -> PerformanceReport:
    """Compute a full performance report from a trades dataframe.

    Parameters
    ----------
    trades:
        Trades dataframe in Freqtrade's standard shape: at minimum
        `pair`, `close_date`, `profit_abs` columns (exactly what
        `Backtesting.results["strategy"][name]["trades"]` and
        `hermes`/`optimize` launchers' outputs already provide).
    starting_balance:
        Starting account balance, needed to express drawdown/profit as
        fractions.
    min_date, max_date:
        Period bounds for the Sharpe/Sortino/Calmar calculations.
        Default to the trades' own `close_date` min/max.

    Returns
    -------
    PerformanceReport

    Raises
    ------
    ReportingError
        If ``trades`` is empty.
    """
    if len(trades) == 0:
        raise ReportingError("Cannot compute a performance report from zero trades")

    trades = trades.copy()
    trades["close_date"] = pd.to_datetime(trades["close_date"])
    resolved_min_date = min_date or trades["close_date"].min()
    resolved_max_date = max_date or trades["close_date"].max()

    total_profit_abs = float(trades["profit_abs"].sum())
    win_rate = float((trades["profit_abs"] > 0).mean())

    drawdown_result = calculate_max_drawdown(
        trades, starting_balance=starting_balance, relative=True
    )
    expectancy, expectancy_ratio = calculate_expectancy(trades)

    per_pair_profit = trades.groupby("pair")["profit_abs"].sum().to_dict()

    report = PerformanceReport(
        total_trades=len(trades),
        win_rate=win_rate,
        total_profit_abs=total_profit_abs,
        total_profit_pct=total_profit_abs / starting_balance if starting_balance else float("nan"),
        sharpe=calculate_sharpe(trades, resolved_min_date, resolved_max_date, starting_balance),
        sortino=calculate_sortino(trades, resolved_min_date, resolved_max_date, starting_balance),
        calmar=calculate_calmar(trades, resolved_min_date, resolved_max_date, starting_balance),
        max_drawdown_pct=float(drawdown_result.relative_account_drawdown),
        max_drawdown_abs=float(drawdown_result.drawdown_abs),
        expectancy=float(expectancy),
        expectancy_ratio=float(expectancy_ratio),
        sqn=calculate_sqn(trades, starting_balance),
        per_pair_profit={str(k): float(v) for k, v in per_pair_profit.items()},
    )

    logger.info(
        "Performance report: %d trades, win_rate=%.1f%%, sharpe=%.2f, max_drawdown=%.1f%%",
        report.total_trades,
        report.win_rate * 100,
        report.sharpe,
        report.max_drawdown_pct * 100,
    )
    return report


def render_performance_report(report: PerformanceReport, title: str = "Performance report") -> Table:
    """Render a `PerformanceReport` as a `rich.table.Table`.

    Parameters
    ----------
    report:
        Report to render.
    title:
        Table title.

    Returns
    -------
    Table
    """
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    rows = [
        ("Total trades", str(report.total_trades)),
        ("Win rate", f"{report.win_rate:.1%}"),
        ("Total profit", f"{report.total_profit_abs:,.2f} ({report.total_profit_pct:.2%})"),
        ("Sharpe", f"{report.sharpe:.3f}"),
        ("Sortino", f"{report.sortino:.3f}"),
        ("Calmar", f"{report.calmar:.3f}"),
        ("Max drawdown", f"{report.max_drawdown_abs:,.2f} ({report.max_drawdown_pct:.2%})"),
        ("Expectancy", f"{report.expectancy:.4f} (ratio {report.expectancy_ratio:.3f})"),
        ("SQN", f"{report.sqn:.3f}"),
    ]
    for metric, value in rows:
        table.add_row(metric, value)

    for pair, profit in sorted(report.per_pair_profit.items()):
        table.add_row(f"  {pair} profit", f"{profit:,.2f}")

    return table


def print_performance_report(
    report: PerformanceReport, title: str = "Performance report", console: Console | None = None
) -> None:
    """Print a `PerformanceReport` to the terminal via `rich`."""
    console = console or Console()
    console.print(render_performance_report(report, title))
