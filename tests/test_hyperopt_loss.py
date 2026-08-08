"""Unit tests for optimize.hyperopt_loss.StatArbHyperOptLoss."""

from __future__ import annotations

import pandas as pd
import pytest

pytestmark = pytest.mark.unit

from optimize.hyperopt_loss import (
    DRAWDOWN_PENALTY_WEIGHT,
    INSUFFICIENT_TRADES_LOSS,
    MIN_TRADES_FOR_VALID_LOSS,
    StatArbHyperOptLoss,
)


def make_trades(profits: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(profits), freq="D")
    return pd.DataFrame({"close_date": dates, "profit_abs": profits})


def test_insufficient_trades_returns_sentinel_loss() -> None:
    trades = make_trades([10.0, -5.0, 3.0])  # fewer than MIN_TRADES_FOR_VALID_LOSS
    assert len(trades) < MIN_TRADES_FOR_VALID_LOSS

    loss = StatArbHyperOptLoss.hyperopt_loss_function(
        results=trades,
        trade_count=len(trades),
        min_date=trades["close_date"].min(),
        max_date=trades["close_date"].max(),
        config={},
        processed={},
        backtest_stats={},
        starting_balance=1000.0,
    )

    assert loss == INSUFFICIENT_TRADES_LOSS


def test_sufficient_trades_returns_finite_loss() -> None:
    profits = [10, -5, 8, 12, -3, 6, 9, -8, 4, 11, -2, 7]
    trades = make_trades(profits)
    assert len(trades) >= MIN_TRADES_FOR_VALID_LOSS

    loss = StatArbHyperOptLoss.hyperopt_loss_function(
        results=trades,
        trade_count=len(trades),
        min_date=trades["close_date"].min(),
        max_date=trades["close_date"].max(),
        config={},
        processed={},
        backtest_stats={},
        starting_balance=1000.0,
    )

    assert loss != INSUFFICIENT_TRADES_LOSS
    import math

    assert math.isfinite(loss)


def test_lower_loss_for_higher_sharpe_same_drawdown() -> None:
    """A strategy with steadier, more consistent profits should score a lower (better) loss."""
    steady_profits = [5.0] * 12
    volatile_profits = [30, -20, 25, -18, 22, -15, 28, -22, 20, -19, 24, -17]

    steady_trades = make_trades(steady_profits)
    volatile_trades = make_trades(volatile_profits)

    def compute(trades: pd.DataFrame) -> float:
        return StatArbHyperOptLoss.hyperopt_loss_function(
            results=trades,
            trade_count=len(trades),
            min_date=trades["close_date"].min(),
            max_date=trades["close_date"].max(),
            config={},
            processed={},
            backtest_stats={},
            starting_balance=1000.0,
        )

    steady_loss = compute(steady_trades)
    volatile_loss = compute(volatile_trades)

    assert steady_loss < volatile_loss


def test_higher_drawdown_penalty_weight_increases_loss_for_drawdown_heavy_results() -> None:
    """Directly verifies the drawdown term is actually included (not just Sharpe)."""
    # Trades with an early large loss (drawdown) followed by recovery.
    profits = [-50, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    trades = make_trades(profits)

    from freqtrade.data.metrics import calculate_max_drawdown, calculate_sharpe

    sharpe = calculate_sharpe(trades, trades["close_date"].min(), trades["close_date"].max(), 1000.0)
    drawdown = calculate_max_drawdown(trades, starting_balance=1000.0, relative=True)

    expected_loss = -sharpe + DRAWDOWN_PENALTY_WEIGHT * float(drawdown.relative_account_drawdown)

    actual_loss = StatArbHyperOptLoss.hyperopt_loss_function(
        results=trades,
        trade_count=len(trades),
        min_date=trades["close_date"].min(),
        max_date=trades["close_date"].max(),
        config={},
        processed={},
        backtest_stats={},
        starting_balance=1000.0,
    )

    assert actual_loss == pytest.approx(expected_loss)


def test_is_a_valid_ihyperoptloss_subclass() -> None:
    from freqtrade.optimize.hyperopt_loss.hyperopt_loss_interface import IHyperOptLoss

    assert issubclass(StatArbHyperOptLoss, IHyperOptLoss)
