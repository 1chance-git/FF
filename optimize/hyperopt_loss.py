"""Custom Freqtrade hyperopt loss function for the stat-arb strategy.

Implements `freqtrade.optimize.hyperopt.IHyperOptLoss`, the abstract
interface Freqtrade's own hyperopt engine calls once per epoch — reused
directly rather than writing a parallel optimization/scoring framework.
The loss itself is built from `freqtrade.data.metrics`'s Sharpe ratio and
max-drawdown calculations (the same mature, already-tested functions
Freqtrade's built-in `SharpeHyperOptLoss`/`MaxDrawDownHyperOptLoss` use),
combined into one risk-adjusted objective rather than reimplementing
either statistic.

Usage: point `freqtrade hyperopt` at this file's directory via
`--hyperopt-path optimize/` and select it with
`--hyperopt-loss StatArbHyperOptLoss` (see
:func:`optimize.hyperopt_launcher.build_hyperopt_command`, which wires
both flags automatically).

Design decisions
-----------------
* **Sharpe ratio alone is an incomplete objective for a mean-reversion
  pairs strategy.** A strategy that rides a wide, growing hedge-ratio
  drift straight through an emerging trend can produce a superficially
  attractive Sharpe ratio right up until a large drawdown. Penalizing
  drawdown directly, in the same objective, discourages the optimizer
  from selecting parameter combinations that only look good on
  volatility-normalized average return.
* **The combined objective is a simple, transparent weighted sum, not a
  black-box multi-objective solver.** `loss = -sharpe + drawdown_weight
  * max_drawdown_pct` is easy to reason about and to re-derive by hand
  from the two underlying statistics — appropriate for a hyperopt loss
  function, which is called thousands of times and needs to be both
  fast and inspectable, unlike the statistical models elsewhere in this
  project where a more sophisticated method is directly the point.
* **A minimum trade count gate returns a fixed, large loss rather than
  `inf`/`nan`.** Too few trades makes the Sharpe/drawdown estimate
  statistically meaningless (e.g. a single lucky trade can produce an
  enormous apparent Sharpe ratio); returning a large-but-finite
  sentinel loss keeps the optimizer's comparisons well-defined instead
  of producing `nan` propagation Freqtrade's optimizer can't rank.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pandas import DataFrame

from freqtrade.data.metrics import calculate_max_drawdown, calculate_sharpe
from freqtrade.optimize.hyperopt_loss.hyperopt_loss_interface import IHyperOptLoss

#: Loss assigned when there are fewer than MIN_TRADES_FOR_VALID_LOSS trades,
#: so the optimizer can still rank these epochs (worse than any real result)
#: instead of receiving a NaN/inf it can't compare.
INSUFFICIENT_TRADES_LOSS = 100.0

#: Trades below this count make Sharpe/drawdown statistically unreliable.
MIN_TRADES_FOR_VALID_LOSS = 10

#: Weight applied to max-drawdown-pct when combining it with -Sharpe.
#: Higher values push the optimizer harder toward low-drawdown parameter
#: regions at the cost of some raw Sharpe.
DRAWDOWN_PENALTY_WEIGHT = 2.0


class StatArbHyperOptLoss(IHyperOptLoss):
    """Risk-adjusted hyperopt loss: penalized Sharpe ratio.

    ``loss = -sharpe_ratio + DRAWDOWN_PENALTY_WEIGHT * max_drawdown_fraction``

    Lower is better (Freqtrade's hyperopt always minimizes), so this
    rewards a high Sharpe ratio while penalizing large drawdowns.
    """

    @staticmethod
    def hyperopt_loss_function(
        *,
        results: DataFrame,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        starting_balance: float,
        **kwargs: Any,
    ) -> float:
        """Compute the loss for one hyperopt epoch's backtest results.

        Parameters
        ----------
        results:
            Trade results dataframe for this epoch (Freqtrade's standard
            backtest trades format).
        trade_count:
            Number of trades in ``results``.
        min_date, max_date:
            Backtest period bounds.
        starting_balance:
            Starting account balance, needed to express drawdown as a
            fraction.
        **kwargs:
            Other fields Freqtrade's `IHyperOptLoss` interface passes
            (config, processed data, etc.), unused here.

        Returns
        -------
        float
            Loss value; lower is better.
        """
        if trade_count < MIN_TRADES_FOR_VALID_LOSS:
            return INSUFFICIENT_TRADES_LOSS

        sharpe_ratio = calculate_sharpe(results, min_date, max_date, starting_balance)

        drawdown_result = calculate_max_drawdown(
            results, starting_balance=starting_balance, relative=True
        )
        max_drawdown_fraction = float(drawdown_result.relative_account_drawdown)

        return -sharpe_ratio + DRAWDOWN_PENALTY_WEIGHT * max_drawdown_fraction
