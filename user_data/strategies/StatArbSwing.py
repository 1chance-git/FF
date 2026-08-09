"""StatArbSwing: a statistical-arbitrage pairs-trading Freqtrade strategy.

Trades the two-pair universe this project is scoped to (see
``user_data/config.json``'s ``StaticPairList``) as a single market-neutral
pair: ``ETH/USDC:USDC`` (the "y" leg) against ``BTC/USDC:USDC`` (the "x"
leg, used as the hedge). It assembles the four modules built earlier in
this project:

* :mod:`stat_arb.data.market_data` — cleans/gap-fills each leg's OHLCV
  and aligns both legs onto a shared timestamp index before any
  cross-pair statistic is computed.
* :mod:`stat_arb.signal.regression` — a rolling OLS hedge ratio between
  the two legs.
* :mod:`stat_arb.signal.cointegration` — the spread, its rolling
  mean/std/z-score, and an Engle-Granger cointegration gate.
* :mod:`stat_arb.risk.risk` — the 5% stop loss, position sizing, regime
  detection, trend filter, exposure limits, and cooldown logic that
  decide whether, and how large, an entry is allowed.

No AI/ML component is used anywhere in this strategy — every signal and
decision is produced by the closed-form statistical and rule-based logic
in the four modules above.

Design decisions
-----------------
* **Only pure, freqtrade-independent functions do the actual computation
  and decision logic.** Every module-level function in this file
  (``determine_pair_role``, ``build_aligned_closes``,
  ``compute_pair_indicators``, ``compute_entry_signals``,
  ``compute_exit_signals``, ``positions_notional_from_trades``) takes
  and returns plain `pandas`/primitive types and can be, and is, unit
  tested without instantiating a Freqtrade strategy or touching a
  dataprovider/wallet/exchange. ``StatArbSwing`` itself is a thin
  ``IStrategy`` adapter that wires Freqtrade's hooks to these functions.
* **Both legs are traded, synchronized by shared spread statistics.**
  Genuine pairs trading requires both legs executed with opposite
  exposure. Freqtrade calls ``populate_indicators``/
  ``populate_entry_trend``/``populate_exit_trend`` once per pair, so this
  strategy computes the *same* spread/z-score (a property of the pair,
  not of either leg alone) inside each call, then flips the entry/exit
  direction based on which leg (`y` or `x`) is currently being
  evaluated (see ``determine_pair_role`` / ``compute_entry_signals``).
* **The hedge ratio and cointegration status are computed once per
  ``populate_indicators`` refresh, not per row.** The rolling hedge
  ratio (`RollingRegressionEngine`) is inherently per-row/vectorized, so
  that part is cheap and computed every refresh. The Engle-Granger
  cointegration test, however, is a whole-sample test by construction
  (`stat_arb.signal.cointegration.test_cointegration` is not a rolling
  statistic) — it is intended to periodically re-validate that the pair
  is still cointegrated, not to be re-run for every historical bar. Its
  result is broadcast as a single (pvalue, is_cointegrated) pair across
  the whole indicator refresh, which naturally updates on each new
  ``populate_indicators`` call as more data arrives.
* **Regime detection and the trend filter run on the *spread*, not on
  raw price.** The risk engine's `detect_regime`/`compute_trend_filter`
  are asset-agnostic; feeding them the spread (rather than either leg's
  raw price) answers the question this strategy actually needs answered:
  is the *spread* currently mean-reverting, and is it currently
  trending away from its mean strongly enough to override that.
* **Backtesting and live/dry-run use two complementary gating layers.**
  ``populate_entry_trend`` applies the vectorized statistical filters
  (z-score threshold, regime, trend filter, cointegration) that
  Freqtrade's backtester evaluates historically. ``confirm_trade_entry``
  additionally applies the risk engine's *stateful* controls — cooldown
  and portfolio exposure — which require live position/wallet state that
  simply doesn't exist mid-backtest-vectorization. Both layers share the
  same `RiskEngine` instance and configuration, so there is exactly one
  source of truth for every threshold.
* **The strategy's own `timeframe` class attribute is a fallback
  default, not a hard requirement.** Freqtrade's `StrategyResolver`
  overrides a strategy's `timeframe` with whatever ``user_data/config.json``
  specifies, so this file matches the project's existing committed
  timeframe (``"5m"``) rather than silently diverging from it; operators
  can still override via config as with any Freqtrade strategy.
* **Position sizing and the base stop loss share one constant.**
  `STOP_LOSS_PCT` is used both for the strategy's mandatory `stoploss`
  class attribute (Freqtrade enforces this regardless of any other
  logic) and for the `RiskEngine`'s `StopLossConfig`, so the two can
  never drift out of sync.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Make the project's `stat_arb` package importable regardless of the
# working directory Freqtrade was launched from (it execs strategy files
# directly rather than importing them as part of an installed package).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IStrategy

from stat_arb.data.market_data import (
    MarketDataError,
    align_pairs,
    clean_and_fill,
    validate_ohlcv,
)
from stat_arb.signal.regression import (
    RollingRegressionConfig,
    RollingRegressionEngine,
)
from stat_arb.signal.cointegration import (
    CointegrationConfig,
    CointegrationEngine,
    CointegrationError,
)
from stat_arb.risk.risk import (
    MarketRegime,
    PositionSide,
    RegimeConfig,
    RiskEngine,
    RiskEngineConfig,
    StopLossConfig,
    TrendFilterConfig,
    calculate_position_size,
    compute_stop_loss_price,
)

from hermes.memory import MemoryStore, TradeRecord

logger = logging.getLogger(__name__)

ROLE_Y = "y"
ROLE_X = "x"


# ---------------------------------------------------------------------------
# Pure helper functions (no Freqtrade dependency; independently unit-tested)
# ---------------------------------------------------------------------------


def determine_pair_role(pair: str, y_pair: str, x_pair: str) -> str | None:
    """Classify ``pair`` as the "y" leg, the "x" leg, or unrelated.

    Returns
    -------
    str | None
        ``"y"`` if ``pair == y_pair``, ``"x"`` if ``pair == x_pair``,
        otherwise ``None`` (e.g. a pair present in the whitelist that
        this strategy doesn't know how to pair-trade).
    """
    if pair == y_pair:
        return ROLE_Y
    if pair == x_pair:
        return ROLE_X
    return None


def build_aligned_closes(
    y_pair: str,
    y_ohlcv: pd.DataFrame,
    x_pair: str,
    x_ohlcv: pd.DataFrame,
    timeframe: str,
) -> tuple[pd.Series, pd.Series]:
    """Clean, gap-fill, and align both legs' OHLCV, returning aligned close series.

    Parameters
    ----------
    y_pair, x_pair:
        Pair names, used only for logging inside the market-data layer.
    y_ohlcv, x_ohlcv:
        Raw OHLCV dataframes for each leg (Freqtrade candle format).
    timeframe:
        Candle timeframe, used to compute expected candle spacing when
        filling gaps.

    Returns
    -------
    tuple[Series, Series]
        ``(y_close, x_close)``, indexed identically by ``date``.

    Raises
    ------
    MarketDataError
        If either leg's data is invalid, or the legs share no common
        timestamps (see :mod:`stat_arb.data.market_data`).
    """
    y_clean = clean_and_fill(y_ohlcv, timeframe, y_pair)
    x_clean = clean_and_fill(x_ohlcv, timeframe, x_pair)
    validate_ohlcv(y_clean, y_pair)
    validate_ohlcv(x_clean, x_pair)
    aligned = align_pairs({y_pair: y_clean, x_pair: x_clean})
    y_close = aligned[y_pair].set_index("date")["close"]
    x_close = aligned[x_pair].set_index("date")["close"]
    return y_close, x_close


def latest_signal_context(analyzed: pd.DataFrame) -> dict[str, float | str | None]:
    """Extract the most recent zscore/hedge_ratio/regime from an analyzed dataframe.

    Pure, no I/O — used to snapshot the signal context worth remembering
    alongside a trade. Each field is ``None`` if the column is absent or
    has no non-null value yet (e.g. still warming up), never raises.

    Parameters
    ----------
    analyzed:
        A dataframe as produced by ``populate_indicators``/
        ``merge_indicators_into_dataframe``.

    Returns
    -------
    dict[str, float | str | None]
        Keys ``"zscore"``, ``"hedge_ratio"``, ``"regime"``.
    """

    def _latest(column: str) -> float | str | None:
        if column not in analyzed.columns:
            return None
        values = analyzed[column].dropna()
        return values.iloc[-1] if not values.empty else None

    return {
        "zscore": _latest("zscore"),
        "hedge_ratio": _latest("hedge_ratio"),
        "regime": _latest("regime"),
    }


def compute_pair_indicators(
    y_close: pd.Series,
    x_close: pd.Series,
    regression_config: RollingRegressionConfig,
    cointegration_config: CointegrationConfig,
    regime_config: RegimeConfig,
    trend_config: TrendFilterConfig,
) -> pd.DataFrame:
    """Compute hedge ratio, spread/z-score, cointegration, regime, and trend.

    Parameters
    ----------
    y_close, x_close:
        Aligned close-price series for the y and x legs (see
        :func:`build_aligned_closes`).
    regression_config:
        Config for the rolling hedge-ratio regression.
    cointegration_config:
        Config for spread/z-score computation. ``require_cointegration``
        is ignored here (this function never raises on a failed test —
        gating on cointegration is a strategy-level decision made from
        the resulting ``is_cointegrated`` column, not a reason to
        produce no indicators at all).
    regime_config, trend_config:
        Configs for regime detection / the trend filter, applied to the
        spread.

    Returns
    -------
    DataFrame
        Indexed like ``y_close``, with columns ``hedge_ratio``,
        ``spread``, ``spread_mean``, ``spread_std``, ``zscore``,
        ``regime``, ``is_trending``, ``is_cointegrated``,
        ``cointegration_pvalue``. Rows before enough history has
        accumulated are ``NaN``/``False`` as appropriate, never dropped.
    """
    result = pd.DataFrame(index=y_close.index)
    for col in ("hedge_ratio", "spread", "spread_mean", "spread_std", "zscore"):
        result[col] = float("nan")
    result["regime"] = MarketRegime.UNKNOWN.value
    result["is_trending"] = False
    result["is_cointegrated"] = False
    result["cointegration_pvalue"] = float("nan")

    if len(y_close) < regression_config.window:
        logger.debug(
            "Not enough data for regression window (%d < %d); returning empty indicators",
            len(y_close),
            regression_config.window,
        )
        return result

    regression_engine = RollingRegressionEngine(regression_config)
    regression_result = regression_engine.fit(y=y_close, x=x_close)
    result["hedge_ratio"] = regression_result.hedge_ratio

    valid_index = regression_result.hedge_ratio.dropna().index
    if len(valid_index) < cointegration_config.spread_window + 1:
        logger.debug(
            "Not enough valid hedge-ratio observations for spread window "
            "(%d < %d); hedge ratio computed but spread/regime/trend left empty",
            len(valid_index),
            cointegration_config.spread_window + 1,
        )
        return result

    y_valid = y_close.loc[valid_index]
    x_valid = x_close.loc[valid_index]
    hedge_ratio_valid = regression_result.hedge_ratio.loc[valid_index]

    coint_config_permissive = CointegrationConfig(
        spread_window=cointegration_config.spread_window,
        min_periods=cointegration_config.min_periods,
        hedge_ratio_lag=cointegration_config.hedge_ratio_lag,
        significance_level=cointegration_config.significance_level,
        require_cointegration=False,
        zscore_std_floor=cointegration_config.zscore_std_floor,
    )
    cointegration_engine = CointegrationEngine(coint_config_permissive)
    spread_result = cointegration_engine.compute(y_valid, x_valid, hedge_ratio_valid)

    result.loc[valid_index, "spread"] = spread_result.spread
    result.loc[valid_index, "spread_mean"] = spread_result.rolling_mean
    result.loc[valid_index, "spread_std"] = spread_result.rolling_std
    result.loc[valid_index, "zscore"] = spread_result.zscore
    result.loc[valid_index, "is_cointegrated"] = spread_result.cointegration.is_cointegrated
    result.loc[valid_index, "cointegration_pvalue"] = spread_result.cointegration.p_value

    spread_valid = spread_result.spread.dropna()
    if len(spread_valid) >= max(regime_config.window, trend_config.window):
        regime_series = _detect_regime_safe(spread_valid, regime_config)
        trend_result = _compute_trend_filter_safe(spread_valid, trend_config)
        result.loc[regime_series.index, "regime"] = regime_series
        result.loc[trend_result.index, "is_trending"] = trend_result
    else:
        logger.debug(
            "Not enough spread history for regime/trend (%d available)", len(spread_valid)
        )

    return result


def _detect_regime_safe(spread: pd.Series, config: RegimeConfig) -> pd.Series:
    """Wrap :func:`stat_arb.risk.risk.detect_regime`, imported lazily to avoid a cycle."""
    from stat_arb.risk.risk import detect_regime

    return detect_regime(spread, config)


def _compute_trend_filter_safe(spread: pd.Series, config: TrendFilterConfig) -> pd.Series:
    """Wrap :func:`stat_arb.risk.risk.compute_trend_filter`, returning just ``is_trending``."""
    from stat_arb.risk.risk import compute_trend_filter

    return compute_trend_filter(spread, config).is_trending


def merge_indicators_into_dataframe(
    dataframe: pd.DataFrame, indicators: pd.DataFrame
) -> pd.DataFrame:
    """Left-merge computed indicators onto Freqtrade's per-pair dataframe by date.

    Preserves ``dataframe``'s original row count and order (a Freqtrade
    requirement for ``populate_indicators``) even though ``indicators``
    may be shorter (e.g. during warm-up) or built from a differently
    trimmed date range.

    Parameters
    ----------
    dataframe:
        The dataframe Freqtrade passed to ``populate_indicators``.
    indicators:
        Output of :func:`compute_pair_indicators`, indexed by ``date``.

    Returns
    -------
    DataFrame
        ``dataframe`` with the indicator columns added.
    """
    indicators_reset = indicators.reset_index().rename(columns={"index": "date"})
    return dataframe.merge(indicators_reset, on="date", how="left")


def compute_entry_signals(
    dataframe: pd.DataFrame,
    role: str,
    entry_zscore: float,
    require_mean_reverting_regime: bool,
    block_when_trending: bool,
    require_cointegration: bool,
) -> tuple[pd.Series, pd.Series]:
    """Compute vectorized long/short entry conditions for one leg of the pair.

    The spread and z-score are identical for both legs (they are a
    property of the pair); which leg goes long vs. short when the spread
    deviates is flipped based on ``role`` (see module docstring).

    Parameters
    ----------
    dataframe:
        Per-pair dataframe with ``compute_pair_indicators``' columns
        already merged in.
    role:
        ``"y"`` or ``"x"``, from :func:`determine_pair_role`.
    entry_zscore:
        Absolute z-score threshold that triggers an entry.
    require_mean_reverting_regime:
        If ``True``, only enter when ``regime == "mean_reverting"``.
    block_when_trending:
        If ``True``, never enter when ``is_trending`` is ``True``.
    require_cointegration:
        If ``True``, only enter when ``is_cointegrated`` is ``True``.

    Returns
    -------
    tuple[Series, Series]
        ``(enter_long, enter_short)`` boolean masks aligned to
        ``dataframe``'s index.
    """
    zscore = dataframe["zscore"]
    base_ok = zscore.notna()

    if require_mean_reverting_regime:
        base_ok &= dataframe["regime"] == MarketRegime.MEAN_REVERTING.value
    if block_when_trending:
        base_ok &= ~dataframe["is_trending"].astype(bool)
    if require_cointegration:
        base_ok &= dataframe["is_cointegrated"].astype(bool)

    spread_low = zscore <= -entry_zscore
    spread_high = zscore >= entry_zscore

    if role == ROLE_Y:
        enter_long = base_ok & spread_low
        enter_short = base_ok & spread_high
    else:
        enter_long = base_ok & spread_high
        enter_short = base_ok & spread_low

    return enter_long, enter_short


def compute_exit_signals(
    dataframe: pd.DataFrame, role: str, exit_zscore: float
) -> tuple[pd.Series, pd.Series]:
    """Compute vectorized long/short exit conditions (spread reversion) for one leg.

    Parameters
    ----------
    dataframe:
        Per-pair dataframe with the ``zscore`` column merged in.
    role:
        ``"y"`` or ``"x"``.
    exit_zscore:
        Absolute z-score threshold at which a reverted spread triggers
        an exit (typically smaller than the entry threshold).

    Returns
    -------
    tuple[Series, Series]
        ``(exit_long, exit_short)`` boolean masks aligned to
        ``dataframe``'s index. The two masks can be simultaneously
        ``True`` whenever the z-score sits within the exit band of the
        mean (``-exit_zscore <= zscore <= exit_zscore``) — that is
        intentional, not a bug: Freqtrade only ever applies whichever
        exit signal matches a given trade's actual open side, so "exit
        if long" and "exit if short" both being true near the mean is
        exactly the desired behavior, not a contradictory signal.
    """
    zscore = dataframe["zscore"]
    valid = zscore.notna()
    reverted_up = zscore >= -exit_zscore
    reverted_down = zscore <= exit_zscore

    if role == ROLE_Y:
        exit_long = valid & reverted_up
        exit_short = valid & reverted_down
    else:
        exit_long = valid & reverted_down
        exit_short = valid & reverted_up

    return exit_long, exit_short


def positions_notional_from_trades(open_trades: list[Any], exclude_pair: str) -> dict[str, float]:
    """Build a pair -> notional-exposure mapping from open trade objects.

    Duck-types on ``.pair``, ``.stake_amount``, and ``.leverage`` so it
    can be tested with plain stub objects rather than requiring a live
    Freqtrade database session.

    Parameters
    ----------
    open_trades:
        Open trade-like objects (e.g. from ``Trade.get_open_trades()``).
    exclude_pair:
        Pair to omit (typically the pair currently being evaluated for a
        new entry, since it isn't "other" exposure).

    Returns
    -------
    dict[str, float]
        Mapping of pair -> notional exposure (``stake_amount * leverage``).
    """
    positions: dict[str, float] = {}
    for trade in open_trades:
        if trade.pair == exclude_pair:
            continue
        leverage = getattr(trade, "leverage", None) or 1.0
        notional = abs(trade.stake_amount * leverage)
        positions[trade.pair] = positions.get(trade.pair, 0.0) + notional
    return positions


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class StatArbSwing(IStrategy):
    """Market-neutral pairs-trading strategy for ETH/USDC vs. BTC/USDC futures.

    See the module docstring for the full design rationale. In summary:
    a rolling-OLS hedge ratio and Engle-Granger-validated spread z-score
    drive entries/exits on both legs simultaneously, gated by the
    independent risk engine's regime detection, trend filter, exposure
    limits, and cooldown logic, with a fixed 5% stop loss throughout.
    """

    INTERFACE_VERSION = 3

    can_short: bool = True
    # Fallback only: StrategyResolver overrides this from config.json.
    timeframe: str = "5m"
    startup_candle_count: int = 160

    # Freqtrade requires a numeric `stoploss`; kept in sync with the risk
    # engine's StopLossConfig via the shared STOP_LOSS_PCT constant below.
    STOP_LOSS_PCT: float = 0.05
    stoploss: float = -STOP_LOSS_PCT

    # No ROI target: exits are driven entirely by spread reversion
    # (populate_exit_trend) and the stop loss, which is standard for a
    # mean-reversion pairs strategy.
    minimal_roi: dict[str, float] = {"0": 10.0}

    use_exit_signal: bool = True
    exit_profit_only: bool = False

    # -- Pair roles -----------------------------------------------------
    Y_PAIR: str = "ETH/USDC:USDC"
    X_PAIR: str = "BTC/USDC:USDC"

    # -- Signal thresholds ------------------------------------------------
    ENTRY_ZSCORE: float = 2.0
    EXIT_ZSCORE: float = 0.5
    REQUIRE_MEAN_REVERTING_REGIME: bool = True
    BLOCK_ENTRY_WHEN_TRENDING: bool = True
    REQUIRE_COINTEGRATION: bool = True

    # Hyperopt-tunable counterparts of ENTRY_ZSCORE/EXIT_ZSCORE, read by
    # populate_entry_trend/populate_exit_trend in place of the plain
    # constants above. `.value` defaults to ENTRY_ZSCORE/EXIT_ZSCORE
    # outside of a hyperopt run, so normal dry-run/live/backtest behavior
    # is unchanged; `freqtrade hyperopt --spaces buy sell` (see
    # optimize/hyperopt_launcher.py) tunes these directly. Deliberately
    # scoped to entry/exit thresholds only, not the window sizes that
    # feed indicator computation (REGRESSION_WINDOW etc.) — tuning those
    # would require Freqtrade's more expensive "indicator space" hyperopt
    # mode and reconstructing `risk_engine` (and its stateful
    # CooldownTracker) per epoch, which isn't worth the added complexity
    # and risk to this strategy's live-trading semantics for the
    # marginal benefit over tuning just the thresholds.
    entry_zscore_param = DecimalParameter(
        1.0, 4.0, default=ENTRY_ZSCORE, decimals=2, space="buy", optimize=True, load=True
    )
    exit_zscore_param = DecimalParameter(
        0.0, 1.5, default=EXIT_ZSCORE, decimals=2, space="sell", optimize=True, load=True
    )

    # -- Window configuration (bars) --------------------------------------
    REGRESSION_WINDOW: int = 60
    SPREAD_WINDOW: int = 40
    REGIME_WINDOW: int = 30
    TREND_WINDOW: int = 20

    def __init__(self, config: dict) -> None:
        """Initialize the strategy and its independent risk engine.

        Parameters
        ----------
        config:
            Freqtrade configuration dict, as passed by the resolver.
        """
        super().__init__(config)

        self._regression_config = RollingRegressionConfig(window=self.REGRESSION_WINDOW)
        self._cointegration_config = CointegrationConfig(spread_window=self.SPREAD_WINDOW)
        self._regime_config = RegimeConfig(window=self.REGIME_WINDOW)
        self._trend_config = TrendFilterConfig(window=self.TREND_WINDOW)

        self.risk_engine = RiskEngine(
            RiskEngineConfig(
                stop_loss=StopLossConfig(stop_loss_pct=self.STOP_LOSS_PCT),
                regime=self._regime_config,
                trend_filter=self._trend_config,
                require_mean_reverting_regime=self.REQUIRE_MEAN_REVERTING_REGIME,
                block_entry_when_trending=self.BLOCK_ENTRY_WHEN_TRENDING,
            )
        )

        # Per-pair signal context captured at entry (zscore/hedge_ratio/regime,
        # plus entry price/time), held in memory until the trade closes and a
        # single combined TradeRecord can be written. Process-local: a Hermes
        # restart between entry and exit loses it, degrading those fields to
        # None on the eventual record rather than losing the record itself.
        self._pending_entry_context: dict[str, dict[str, Any]] = {}

        self.memory_store: MemoryStore | None = None
        try:
            user_data_dir = Path(config.get("user_data_dir", "user_data"))
            self.memory_store = MemoryStore(user_data_dir / "hermes_memory.sqlite3")
        except Exception:
            logger.exception(
                "Failed to initialize Hermes memory store; trade history will not be "
                "recorded (trading is unaffected)"
            )

    def informative_pairs(self) -> list[tuple[str, str, str]]:
        """Both legs are already in the main whitelist; nothing extra needed."""
        return []

    # -- Indicators ---------------------------------------------------------

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Compute hedge ratio, spread/z-score, cointegration, regime, and trend.

        Fetches the *other* leg's OHLCV via the dataprovider, aligns both
        legs (:mod:`stat_arb.data.market_data`), fits the rolling hedge
        ratio (:mod:`stat_arb.signal.regression`), computes the spread
        and its z-score with a cointegration check
        (:mod:`stat_arb.signal.cointegration`), and classifies the
        spread's regime/trend (:mod:`stat_arb.risk.risk`). See the
        module docstring for why cointegration is evaluated once per
        refresh rather than per row.
        """
        pair = metadata["pair"]
        role = determine_pair_role(pair, self.Y_PAIR, self.X_PAIR)
        if role is None:
            logger.warning("StatArbSwing does not know how to pair-trade %s; skipping", pair)
            return dataframe

        other_pair = self.X_PAIR if role == ROLE_Y else self.Y_PAIR
        try:
            other_ohlcv = self._get_other_leg_dataframe(other_pair)
            if role == ROLE_Y:
                y_close, x_close = build_aligned_closes(
                    self.Y_PAIR, dataframe, self.X_PAIR, other_ohlcv, self.timeframe
                )
            else:
                y_close, x_close = build_aligned_closes(
                    self.Y_PAIR, other_ohlcv, self.X_PAIR, dataframe, self.timeframe
                )
        except MarketDataError as exc:
            logger.warning("Could not align %s/%s data for %s: %s", self.Y_PAIR, self.X_PAIR, pair, exc)
            return dataframe

        indicators = compute_pair_indicators(
            y_close,
            x_close,
            self._regression_config,
            self._cointegration_config,
            self._regime_config,
            self._trend_config,
        )
        return merge_indicators_into_dataframe(dataframe, indicators)

    def _get_other_leg_dataframe(self, other_pair: str) -> pd.DataFrame:
        """Fetch the other leg's raw OHLCV via the dataprovider."""
        if self.dp is None:
            raise MarketDataError("No dataprovider available to fetch the other leg's data")
        return self.dp.get_pair_dataframe(pair=other_pair, timeframe=self.timeframe)

    # -- Entries / exits ------------------------------------------------

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Set enter_long/enter_short from the spread z-score, regime, and trend filter."""
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0

        role = determine_pair_role(metadata["pair"], self.Y_PAIR, self.X_PAIR)
        if role is None or "zscore" not in dataframe.columns:
            return dataframe

        enter_long, enter_short = compute_entry_signals(
            dataframe,
            role,
            self.entry_zscore_param.value,
            self.REQUIRE_MEAN_REVERTING_REGIME,
            self.BLOCK_ENTRY_WHEN_TRENDING,
            self.REQUIRE_COINTEGRATION,
        )
        dataframe.loc[enter_long, "enter_long"] = 1
        dataframe.loc[enter_long, "enter_tag"] = "stat_arb_spread_reversion"
        dataframe.loc[enter_short, "enter_short"] = 1
        dataframe.loc[enter_short, "enter_tag"] = "stat_arb_spread_reversion"
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Set exit_long/exit_short once the spread z-score has reverted toward its mean."""
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0

        role = determine_pair_role(metadata["pair"], self.Y_PAIR, self.X_PAIR)
        if role is None or "zscore" not in dataframe.columns:
            return dataframe

        exit_long, exit_short = compute_exit_signals(dataframe, role, self.exit_zscore_param.value)
        dataframe.loc[exit_long, "exit_long"] = 1
        dataframe.loc[exit_short, "exit_short"] = 1
        return dataframe

    # -- Risk engine integration: sizing, entry/exit gating, cooldown -----

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> float:
        """Size the position via the risk engine's fixed-fractional sizing."""
        equity = self.wallets.get_total_stake_amount() if self.wallets else proposed_stake
        position_side = PositionSide.LONG if side == "long" else PositionSide.SHORT
        stop_price = compute_stop_loss_price(
            current_rate, position_side, self.risk_engine.config.stop_loss
        )
        sizing = calculate_position_size(
            equity, current_rate, stop_price, self.risk_engine.config.position_sizing
        )
        stake = sizing.notional / leverage if leverage else sizing.notional
        stake = min(stake, max_stake)
        if min_stake is not None:
            stake = max(stake, min_stake)
        return stake

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> bool:
        """Apply the risk engine's stateful gates (cooldown, exposure) before entry.

        The vectorized statistical filters (z-score, regime, trend,
        cointegration) already ran in ``populate_entry_trend``; this hook
        adds the checks that require live portfolio state and therefore
        can't run during vectorized backtesting.

        Every code path below is wrapped so this method can never raise:
        Freqtrade calls ``confirm_trade_entry`` through
        ``strategy_safe_wrapper(..., default_retval=True)``
        (``freqtradebot.py``), meaning an *unhandled* exception here
        would make Freqtrade log a warning and then treat the entry as
        **confirmed** — the exact opposite of what a risk gate should do
        on failure. Any unexpected error is therefore caught explicitly
        and turned into a blocked entry (``False``) with a logged
        reason, so this gate fails closed, not open.
        """
        try:
            return self._confirm_trade_entry_unsafe(
                pair=pair, rate=rate, current_time=current_time, side=side
            )
        except Exception:
            logger.exception(
                "confirm_trade_entry raised an unexpected error for %s; blocking entry "
                "(fail closed)",
                pair,
            )
            return False

    def _confirm_trade_entry_unsafe(
        self, *, pair: str, rate: float, current_time: datetime, side: str
    ) -> bool:
        """The real confirm_trade_entry logic; see the public method's docstring for why
        every call site of this goes through a blanket try/except that fails closed."""
        role = determine_pair_role(pair, self.Y_PAIR, self.X_PAIR)
        if role is None:
            return False

        analyzed = self.dp.get_analyzed_dataframe(pair, self.timeframe)[0] if self.dp else None
        if analyzed is None or analyzed.empty or "spread" not in analyzed.columns:
            logger.info("No analyzed spread data available yet for %s; blocking entry", pair)
            return False

        spread = analyzed["spread"].dropna()
        if spread.empty:
            logger.info("Spread not yet available for %s; blocking entry", pair)
            return False

        if self.wallets is None:
            logger.warning("No wallets available for %s; blocking entry", pair)
            return False
        equity = self.wallets.get_total_stake_amount()

        open_trades = Trade.get_open_trades()
        current_positions = positions_notional_from_trades(open_trades, exclude_pair=pair)
        position_side = PositionSide.LONG if side == "long" else PositionSide.SHORT

        decision = self.risk_engine.evaluate_entry(
            pair=pair,
            side=position_side,
            entry_price=rate,
            equity=equity,
            prices=spread,
            current_positions=current_positions,
            current_time=current_time,
        )
        if not decision.allowed:
            logger.info("Blocking entry for %s: %s", pair, "; ".join(decision.reasons))
            return False

        try:
            self._record_entry(pair, side, rate, current_time, analyzed)
        except Exception:
            logger.exception(
                "Failed to record entry to Hermes memory for %s; trade proceeds", pair
            )
        return True

    def _record_entry(
        self,
        pair: str,
        side: str,
        rate: float,
        current_time: datetime,
        analyzed: pd.DataFrame,
    ) -> None:
        """Snapshot entry-time signal context and, if available, log it to Hermes.

        Never raises to its caller by design of the components it calls:
        `latest_signal_context` is pure, and `MemoryStore.record_trade`
        catches its own exceptions. The caller still wraps this call, as
        defense in depth against a future change here.
        """
        context = latest_signal_context(analyzed)
        self._pending_entry_context[pair] = {
            **context,
            "side": side,
            "entry_price": rate,
            "entry_time": current_time,
        }
        if self.memory_store is None:
            return
        self.memory_store.record_trade(
            TradeRecord(
                pair=pair,
                side=side,
                entry_time=current_time,
                entry_price=rate,
                entry_zscore=context["zscore"],
                hedge_ratio=context["hedge_ratio"],
                regime=context["regime"],
            )
        )

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs: Any,
    ) -> bool:
        """Record the exit with the risk engine, arming the cooldown on a stop loss.

        Never blocks the exit itself: a failure to record cooldown
        bookkeeping is a (logged) internal error, not a reason to leave
        an open position stuck unable to close. Freqtrade also defaults
        an unhandled exception here to "confirmed" anyway
        (``strategy_safe_wrapper(..., default_retval=True)`` in
        ``freqtradebot.py``), but catching it explicitly avoids an
        alarming "Unexpected error" log for what is, from a trading
        perspective, a non-fatal bookkeeping failure.
        """
        stop_loss_triggered = exit_reason in ("stop_loss", "stoploss")
        try:
            self.risk_engine.record_exit(pair, current_time, stop_loss_triggered=stop_loss_triggered)
        except Exception:
            logger.exception(
                "Failed to record exit with the risk engine for %s; cooldown may not be "
                "armed, but the exit itself is not blocked",
                pair,
            )

        try:
            self._record_exit(pair, trade, rate, exit_reason, current_time)
        except Exception:
            logger.exception(
                "Failed to record exit to Hermes memory for %s; exit proceeds", pair
            )

        return True

    def _record_exit(
        self,
        pair: str,
        trade: Trade,
        rate: float,
        exit_reason: str,
        current_time: datetime,
    ) -> None:
        """Combine the entry context captured in `_record_entry` with the closing
        trade's P&L/fees into a single completed-trade record for Hermes.

        Best-effort throughout: a missing entry context (e.g. Hermes
        restarted mid-trade) or an un-computable P&L degrades individual
        fields to `None` rather than skipping the record.
        """
        entry_context = self._pending_entry_context.pop(pair, {})

        if self.memory_store is None:
            return

        try:
            pnl = trade.calc_profit(rate)
            pnl_pct = trade.calc_profit_ratio(rate)
        except Exception:
            logger.warning("Could not compute P&L for %s at exit; recording without it", pair)
            pnl = None
            pnl_pct = None

        fee_components = [
            fee for fee in (trade.fee_open_cost, trade.fee_close_cost) if fee is not None
        ]
        fees = sum(fee_components) if fee_components else None
        funding = getattr(trade, "funding_fees", None)

        entry_time = entry_context.get("entry_time") or trade.open_date_utc
        holding_time_seconds = (
            (current_time - entry_time).total_seconds() if entry_time is not None else None
        )

        exit_context: dict[str, float | str | None] = {}
        dp = getattr(self, "dp", None)
        if dp is not None:
            exit_analyzed = dp.get_analyzed_dataframe(pair, self.timeframe)[0]
            if exit_analyzed is not None and not exit_analyzed.empty:
                exit_context = latest_signal_context(exit_analyzed)

        self.memory_store.record_trade(
            TradeRecord(
                pair=pair,
                side=entry_context.get("side"),
                entry_time=entry_time,
                exit_time=current_time,
                entry_price=entry_context.get("entry_price", trade.open_rate),
                exit_price=rate,
                pnl=pnl,
                pnl_pct=pnl_pct,
                fees=fees,
                funding=funding,
                entry_zscore=entry_context.get("zscore"),
                exit_zscore=exit_context.get("zscore"),
                hedge_ratio=entry_context.get("hedge_ratio"),
                holding_time_seconds=holding_time_seconds,
                exit_reason=exit_reason,
                regime=entry_context.get("regime"),
            )
        )
