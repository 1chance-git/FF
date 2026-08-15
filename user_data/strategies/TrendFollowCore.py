"""TrendFollowCore: indicators-and-signal-only trend-following strategy.

Path A's first production-oriented strategy, distinct from and
unrelated to :mod:`StatArbSwing` (the retained mean-reversion pairs
strategy for Path B research). Trades two independent futures pairs
-- ``BTC/USDC:USDC``, ``ETH/USDC:USDC`` -- each signaled purely from
its own price action, with no cross-pair statistics (no hedge ratio,
no spread, no z-score, no cointegration). SOL/USDC:USDC was part of
the initial research universe but was excluded from the frozen
trading universe after a BTC+ETH-vs-SOL isolation audit over the
frozen 39-trade baseline showed BTC+ETH alone with a materially
cleaner profile (higher win rate, higher profit factor, roughly half
the stop-loss count) and SOL as a net drag on aggregate P/L; SOL is
deferred, not permanently ruled out, pending a larger sample.

**SHORT-ONLY.** LONG entry and exit logic has been removed
structurally, not gated behind a configuration flag: ``compute_entry_signals``
never evaluates the long-side EMA/ADX/Donchian condition at all (there
is no ``long_condition`` in the code), and ``compute_exit_signals``
never evaluates the long-side EMA-200 invalidation. ``enter_long`` and
``exit_long`` columns are still produced (Freqtrade requires them) but
are unconditionally ``0`` -- there is no code path, under any market
condition, that can set either to ``1``. The SHORT entry condition
(``close < ema200`` and ``adx > 25`` and Donchian-lower breakout) and
SHORT exit condition (EMA-200 invalidation) are unchanged from the
long/short version of this file.

Scope, deliberately narrow: this file implements indicators and
preliminary entry/invalidation logic ONLY.

* No leverage selection, no position sizing, no TWAP/execution slicing.
* No exits beyond the single structural EMA-200 invalidation described
  below (plus Freqtrade's mandatory ``stoploss``).
* No AI/ML, no FreqAI, no Hermes learning/memory wiring.
* No backtesting, optimization, or parameter tuning performed as part
  of building this file -- that is explicitly future work.

Design
------
* **Pure, Freqtrade-independent functions do the actual computation.**
  ``compute_indicators``, ``compute_entry_signals``, and
  ``compute_exit_signals`` take and return plain ``pandas.DataFrame``
  objects and are unit tested directly, without instantiating a
  Freqtrade strategy or touching a dataprovider/wallet/exchange --
  mirroring the same separation ``StatArbSwing`` uses.
  ``TrendFollowCore`` itself is a thin ``IStrategy`` adapter that wires
  Freqtrade's hooks to these functions.
* **Donchian channel lookback deliberately excludes the current
  candle.** ``high``/``low`` are shifted forward by one bar *before*
  the rolling max/min window is applied, so the breakout threshold for
  candle ``N`` is computed only from candles ``[N-20, N-1]`` -- never
  candle ``N`` itself. This is what "use only completed candles"
  requires: comparing today's close against a channel that included
  today's own high/low would let today's own price movement help
  create the very breakout threshold it's being tested against
  (lookahead through self-inclusion), not a leak of literally future
  (post-today) data -- but excluding it is still the correct, safer
  contract for a breakout signal to have.
* **NaN indicators produce no signal, explicitly.** Pandas comparisons
  against NaN already evaluate to ``False`` (so ``close > NaN`` never
  fires), but every entry/exit column is additionally, explicitly
  masked to ``0``/``False`` wherever any required indicator is NaN --
  making "no signal during warmup" a checked invariant of this file
  rather than an incidental consequence of NaN comparison semantics.
* **The EMA-200 trend filter is also the sole invalidation.** A short
  is invalidated the moment price closes back above EMA-200 (the same
  condition that would have blocked a *new* short entry), so entry and
  exit share one trend definition rather than inventing a second one.
"""

from __future__ import annotations

import logging

import pandas as pd
import talib

from freqtrade.strategy import IStrategy

logger = logging.getLogger(__name__)

EMA_PERIOD = 200
ADX_PERIOD = 14
DONCHIAN_PERIOD = 20
ADX_THRESHOLD = 25.0

_REQUIRED_INDICATOR_COLUMNS = ("ema200", "adx", "donchian_upper_prev", "donchian_lower_prev")


# ---------------------------------------------------------------------------
# Pure helper functions (no Freqtrade dependency; independently unit-tested)
# ---------------------------------------------------------------------------


def compute_indicators(
    dataframe: pd.DataFrame,
    *,
    ema_period: int = EMA_PERIOD,
    adx_period: int = ADX_PERIOD,
    donchian_period: int = DONCHIAN_PERIOD,
) -> pd.DataFrame:
    """Add ``ema200``, ``adx``, ``donchian_upper_prev``, ``donchian_lower_prev``.

    The two Donchian columns are the rolling max of ``high`` / min of
    ``low`` over the *previous* ``donchian_period`` completed candles --
    computed by shifting ``high``/``low`` one bar forward before the
    rolling window, so the current candle never contributes to its own
    breakout threshold (see the module docstring).
    """
    df = dataframe.copy()

    df["ema200"] = talib.EMA(df["close"], timeperiod=ema_period)
    df["adx"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod=adx_period)

    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    df["donchian_upper_prev"] = prev_high.rolling(window=donchian_period).max()
    df["donchian_lower_prev"] = prev_low.rolling(window=donchian_period).min()

    return df


def compute_entry_signals(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add ``enter_short`` (0/1) plus its ``enter_tag``. SHORT-only, structurally.

    Short requires all three: ``close < ema200``, ``adx > 25``, and the
    close breaking below the previous-20-candle Donchian lower channel
    -- unchanged from the long/short version of this strategy. Any row
    missing a required indicator (warmup/NaN) gets no signal,
    explicitly.

    ``enter_long`` is still produced (Freqtrade requires the column)
    but is unconditionally ``0``: there is no long-side condition
    computed anywhere in this function, so no code path can set it to
    ``1`` under any market data.
    """
    df = dataframe.copy()

    valid = df[list(_REQUIRED_INDICATOR_COLUMNS)].notna().all(axis=1)

    short_condition = (
        valid
        & (df["close"] < df["ema200"])
        & (df["adx"] > ADX_THRESHOLD)
        & (df["close"] < df["donchian_lower_prev"])
    )

    df["enter_long"] = 0
    df["enter_short"] = short_condition.astype(int)
    df.loc[short_condition, "enter_tag"] = "trend_short_donchian_breakout"

    return df


def compute_exit_signals(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add ``exit_short`` (0/1): the EMA-200 structural invalidation. SHORT-only.

    A short exits the moment price closes back above EMA-200 --
    unchanged from the long/short version of this strategy. NaN
    ``ema200`` (warmup) produces no exit signal, explicitly, same as
    the entry side.

    ``exit_long`` is still produced (Freqtrade requires the column)
    but is unconditionally ``0``: there is no long-side condition
    computed anywhere in this function.
    """
    df = dataframe.copy()

    valid = df["ema200"].notna()

    df["exit_long"] = 0
    df["exit_short"] = (valid & (df["close"] > df["ema200"])).astype(int)

    return df


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class TrendFollowCore(IStrategy):
    """SHORT-only trend-following strategy for BTC/ETH futures: EMA-200
    direction filter, ADX-14 trend-strength gate, and a 20-candle
    Donchian breakout trigger. See the module docstring for full scope,
    rationale, and the SHORT-only structural note.
    """

    INTERFACE_VERSION = 3

    can_short: bool = True

    # Fallback only: StrategyResolver overrides this from config if the
    # config specifies a timeframe (same convention StatArbSwing uses).
    timeframe: str = "4h"

    # startup_candle_count must safely exceed every indicator's warmup
    # requirement:
    #   - EMA-200 needs 200 candles before talib.EMA stops returning NaN.
    #   - The Donchian channel needs donchian_period (20) prior candles
    #     on top of the 1-candle shift, i.e. 21.
    #   - ADX-14 (a recursive/smoothed indicator, like EMA) is documented
    #     by TA-Lib as needing extra bars beyond its bare `timeperiod` to
    #     converge past its internal "unstable period" -- 14 is nowhere
    #     near enough for a numerically stable value.
    # EMA-200 dominates all three by a wide margin, so 200 plus a fixed
    # safety margin (covers Donchian's +21 and ADX's convergence bars,
    # with room to spare) is sufficient without needing per-indicator
    # arithmetic. 250 was chosen as that margin, not guessed silently.
    startup_candle_count: int = 250

    STOP_LOSS_PCT: float = 0.05
    stoploss: float = -STOP_LOSS_PCT

    # No ROI target: the only intended exit is the EMA-200 structural
    # invalidation (populate_exit_trend) plus the stop loss above --
    # per this module's explicit scope, no further exit logic is added.
    minimal_roi: dict[str, float] = {"0": 10.0}

    use_exit_signal: bool = True
    exit_profit_only: bool = False

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return compute_indicators(dataframe)

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return compute_entry_signals(dataframe)

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return compute_exit_signals(dataframe)
