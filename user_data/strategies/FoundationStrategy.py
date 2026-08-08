"""Foundation strategy skeleton.

This strategy exists solely to satisfy Freqtrade's requirement that a
strategy class be loadable at startup. It contains no indicators and no
entry/exit logic — those will be added in a later module (statistical
arbitrage signal generation).

Design decisions
-----------------
* Subclasses ``freqtrade.strategy.IStrategy`` directly rather than any
  third-party base class: Freqtrade's own interface is the mature,
  well-tested contract the bot's engine expects, so there is nothing to
  reinvent here.
* All three mandatory hooks (``populate_indicators``,
  ``populate_entry_trend``, ``populate_exit_trend``) are implemented as
  no-ops that return the dataframe unchanged. ``can_short`` is enabled
  because the project trades USDC-margined futures, where short entries
  are valid.
* ``minimal_roi``, ``stoploss`` and ``timeframe`` are set to safe,
  conservative placeholders. They are intentionally not tuned — that is
  a task for the risk-management module, not the foundation module.
"""

import logging
from typing import Any

from pandas import DataFrame

from freqtrade.strategy import IStrategy


logger = logging.getLogger(__name__)


class FoundationStrategy(IStrategy):
    """Minimal, no-op strategy used to verify the bot starts correctly."""

    INTERFACE_VERSION = 3

    # Futures/short support is required for statistical-arbitrage pair trades.
    can_short: bool = True

    timeframe: str = "5m"

    # Conservative placeholders; not yet tuned. No trades are generated
    # by this strategy, so these values are never exercised in practice.
    minimal_roi: dict[str, float] = {"0": 0.10}
    stoploss: float = -0.10

    startup_candle_count: int = 30

    def populate_indicators(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        """Compute indicators. No-op placeholder for the foundation module.

        Parameters
        ----------
        dataframe:
            OHLCV candle data for the given pair.
        metadata:
            Pair metadata supplied by Freqtrade (e.g. ``{"pair": "BTC/USDC:USDC"}``).

        Returns
        -------
        DataFrame
            The dataframe, unmodified.
        """
        logger.debug("populate_indicators called for %s", metadata.get("pair"))
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        """Generate entry signals. No-op placeholder for the foundation module.

        Parameters
        ----------
        dataframe:
            OHLCV candle data with indicators attached.
        metadata:
            Pair metadata supplied by Freqtrade.

        Returns
        -------
        DataFrame
            The dataframe, unmodified — no entry signals are set.
        """
        logger.debug("populate_entry_trend called for %s", metadata.get("pair"))
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        """Generate exit signals. No-op placeholder for the foundation module.

        Parameters
        ----------
        dataframe:
            OHLCV candle data with indicators attached.
        metadata:
            Pair metadata supplied by Freqtrade.

        Returns
        -------
        DataFrame
            The dataframe, unmodified — no exit signals are set.
        """
        logger.debug("populate_exit_trend called for %s", metadata.get("pair"))
        return dataframe
