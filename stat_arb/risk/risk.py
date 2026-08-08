"""Independent risk engine: stop loss, sizing, regime/trend filters, exposure, cooldown.

This module has **no dependency on Freqtrade**. Every function and class
here operates on plain `pandas`/`numpy` data and primitive Python types
(`float`, `str`, `datetime`), so it can be unit-tested, driven from a
backtester, or wired into a completely different execution engine
without ever importing `freqtrade`. The only third-party dependency is
`statsmodels`, used for the same mature statistical routines the rest of
`stat_arb` already relies on.

Covers, independently of one another:

* **Stop loss** — a fixed 5% adverse-move stop, computed and checked
  per position.
* **Position sizing** — fixed-fractional sizing from the distance to the
  stop loss, capped by configurable exposure/leverage bounds.
* **Regime detection** — classifies each rolling window of a price
  series as mean-reverting or trending via the Augmented Dickey-Fuller
  test.
* **Trend filter** — a separate, faster diagnostic: the statistical
  significance of a rolling linear time trend, used to veto new
  mean-reversion entries while a strong directional trend is in force.
* **Maximum exposure** — gross, per-pair, and position-count limits
  checked before sizing a new entry.
* **Cooldown logic** — a stateful tracker that blocks new entries on a
  pair for a configurable period after a stop-loss exit.

`RiskEngine` composes all of the above into a single `evaluate_entry`
call, but every building block is a free function or small dataclass
that can be used and tested on its own.

Design decisions
-----------------
* **No Freqtrade dependency, by requirement.** The risk engine is a pure
  decision layer: given prices, equity, and current positions (all
  plain Python/pandas types), it decides whether and how large an entry
  should be. Keeping it free of `freqtrade` imports means it can be
  exercised in isolation (as the tests do), reused from a research
  notebook or a different execution venue, and changed without touching
  anything strategy-integration-related.
* **Regime detection reuses `statsmodels.tsa.stattools.adfuller`**, the
  same mature Augmented Dickey-Fuller implementation used elsewhere in
  this project (see `stat_arb.signal.cointegration`), rather than a
  custom stationarity heuristic. Running it per rolling window is more
  expensive than a closed-form indicator (documented on
  `detect_regime`), but it is directly the same statistical test that
  validated the pair's cointegration in the first place, applied
  locally.
* **The trend filter reuses `statsmodels.regression.rolling.RollingOLS`**
  (already a dependency, see `stat_arb.signal.regression`) to regress
  price on a linear time index per rolling window and test the slope's
  significance (`t = slope / std_err`), rather than a hand-rolled
  moving-average-crossover heuristic. This gives a statistically
  grounded, cheaply computed answer to "is there a real, non-noise
  trend right now" that is independent of (and much faster than) the
  ADF-based regime classification, so the two can disagree — that
  disagreement is informative, not a bug.
* **Position sizing is fixed-fractional, capped by explicit exposure
  bounds.** Sizing purely off the stop-loss distance
  (`risk_amount / stop_distance`) can produce an unbounded position size
  when the stop is very tight; every size is therefore also capped by
  `max_position_pct_of_equity` and `max_leverage`, with the binding cap
  reported on the result rather than applied silently.
* **Cooldown state is intentionally mutable, unlike every other
  dataclass in this module.** A cooldown tracker's entire purpose is to
  remember exit history across calls — making it an immutable frozen
  dataclass like the rest of the module's configuration objects would
  defeat that purpose. It is deliberately the one stateful class here,
  and it is isolated to exactly the state it needs (`pair -> last
  stop-loss exit time`).
* **Only stop-loss exits trigger a cooldown by default.** A trade closed
  for a normal reason (e.g. mean-reversion completed, take-profit) isn't
  evidence the pair or the timing was bad; a stop-loss exit is. Set
  `CooldownTracker.record_exit(..., stop_loss_triggered=True)`
  explicitly to arm the cooldown, or pass ``apply_to_all_exits=True`` on
  `CooldownConfig` to cool down after every exit instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import adfuller

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


class RiskError(Exception):
    """Raised when risk-engine inputs are invalid."""


class PositionSide(str, Enum):
    """Direction of a position."""

    LONG = "long"
    SHORT = "short"


class MarketRegime(str, Enum):
    """Classification of a price series' rolling-window regime."""

    MEAN_REVERTING = "mean_reverting"
    TRENDING = "trending"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Stop loss
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StopLossConfig:
    """Fixed percentage stop-loss configuration.

    Parameters
    ----------
    stop_loss_pct:
        Adverse move, as a fraction of entry price, that triggers the
        stop. Defaults to ``0.05`` (5%).
    """

    stop_loss_pct: float = 0.05

    def __post_init__(self) -> None:
        if not 0 < self.stop_loss_pct < 1:
            raise RiskError(f"stop_loss_pct must be in (0, 1), got {self.stop_loss_pct}")


def compute_stop_loss_price(
    entry_price: float,
    side: PositionSide,
    config: StopLossConfig = StopLossConfig(),
) -> float:
    """Compute the stop-loss price for a position.

    Parameters
    ----------
    entry_price:
        The position's entry price. Must be positive.
    side:
        Position direction.
    config:
        Stop-loss configuration.

    Returns
    -------
    float
        For a long position, ``entry_price * (1 - stop_loss_pct)``; for
        a short, ``entry_price * (1 + stop_loss_pct)``.

    Raises
    ------
    RiskError
        If ``entry_price`` is not positive.
    """
    if entry_price <= 0:
        raise RiskError(f"entry_price must be positive, got {entry_price}")

    if side is PositionSide.LONG:
        return entry_price * (1 - config.stop_loss_pct)
    return entry_price * (1 + config.stop_loss_pct)


def is_stop_loss_triggered(
    current_price: float,
    entry_price: float,
    side: PositionSide,
    config: StopLossConfig = StopLossConfig(),
) -> bool:
    """Check whether the current price has breached the stop-loss level.

    Parameters
    ----------
    current_price:
        The latest observed price. Must be positive.
    entry_price:
        The position's entry price.
    side:
        Position direction.
    config:
        Stop-loss configuration.

    Returns
    -------
    bool
        ``True`` if the stop has been breached.

    Raises
    ------
    RiskError
        If ``current_price`` is not positive.
    """
    if current_price <= 0:
        raise RiskError(f"current_price must be positive, got {current_price}")

    stop_price = compute_stop_loss_price(entry_price, side, config)
    if side is PositionSide.LONG:
        triggered = current_price <= stop_price
    else:
        triggered = current_price >= stop_price

    if triggered:
        logger.info(
            "Stop loss triggered: side=%s, entry=%.6f, stop=%.6f, current=%.6f",
            side.value,
            entry_price,
            stop_price,
            current_price,
        )
    return triggered


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PositionSizingConfig:
    """Fixed-fractional position sizing configuration.

    Parameters
    ----------
    risk_per_trade_pct:
        Fraction of equity to risk (i.e. lose if the stop is hit) on a
        single trade.
    max_position_pct_of_equity:
        Hard cap on a single position's notional value, as a fraction of
        equity, regardless of what fixed-fractional sizing would imply.
    max_leverage:
        Hard cap on a single position's notional value expressed as a
        multiple of equity (i.e. leverage).
    """

    risk_per_trade_pct: float = 0.01
    max_position_pct_of_equity: float = 0.25
    max_leverage: float = 3.0

    def __post_init__(self) -> None:
        if not 0 < self.risk_per_trade_pct < 1:
            raise RiskError(
                f"risk_per_trade_pct must be in (0, 1), got {self.risk_per_trade_pct}"
            )
        if self.max_position_pct_of_equity <= 0:
            raise RiskError("max_position_pct_of_equity must be positive")
        if self.max_leverage <= 0:
            raise RiskError("max_leverage must be positive")


@dataclass(frozen=True)
class PositionSizeResult:
    """Result of :func:`calculate_position_size`.

    Attributes
    ----------
    units:
        Position size in units of the traded asset.
    notional:
        Position size in quote-currency notional (``units * entry_price``).
    risk_amount:
        Equity amount that would be lost if the stop loss is hit
        (before any exposure cap was applied).
    capped_by:
        Which exposure bound (if any) reduced the position below what
        fixed-fractional risk sizing alone would have produced:
        ``"max_position_pct_of_equity"``, ``"max_leverage"``, or
        ``None`` if uncapped.
    """

    units: float
    notional: float
    risk_amount: float
    capped_by: str | None


def calculate_position_size(
    equity: float,
    entry_price: float,
    stop_loss_price: float,
    config: PositionSizingConfig = PositionSizingConfig(),
) -> PositionSizeResult:
    """Calculate position size via fixed-fractional risk, capped by exposure bounds.

    The raw size is ``risk_amount / |entry_price - stop_loss_price|``
    units, i.e. the position that loses exactly ``risk_amount`` if the
    stop is hit. That raw size is then capped so the position's notional
    never exceeds either ``max_position_pct_of_equity`` or
    ``max_leverage`` of equity.

    Parameters
    ----------
    equity:
        Current account equity. Must be positive.
    entry_price:
        Intended entry price. Must be positive.
    stop_loss_price:
        Stop-loss price for the position (e.g. from
        :func:`compute_stop_loss_price`). Must differ from
        ``entry_price``.
    config:
        Position sizing configuration.

    Returns
    -------
    PositionSizeResult

    Raises
    ------
    RiskError
        If ``equity``/``entry_price`` are not positive, or
        ``stop_loss_price`` equals ``entry_price`` (zero stop distance).
    """
    if equity <= 0:
        raise RiskError(f"equity must be positive, got {equity}")
    if entry_price <= 0:
        raise RiskError(f"entry_price must be positive, got {entry_price}")
    if stop_loss_price <= 0:
        raise RiskError(f"stop_loss_price must be positive, got {stop_loss_price}")
    stop_distance = abs(entry_price - stop_loss_price)
    if stop_distance == 0:
        raise RiskError("stop_loss_price must differ from entry_price")

    risk_amount = equity * config.risk_per_trade_pct
    raw_units = risk_amount / stop_distance
    raw_notional = raw_units * entry_price

    max_notional_by_pct = equity * config.max_position_pct_of_equity
    max_notional_by_leverage = equity * config.max_leverage
    notional_cap = min(max_notional_by_pct, max_notional_by_leverage)

    if raw_notional > notional_cap:
        notional = notional_cap
        units = notional / entry_price
        capped_by = (
            "max_position_pct_of_equity"
            if max_notional_by_pct <= max_notional_by_leverage
            else "max_leverage"
        )
        logger.info(
            "Position size capped by %s: raw_notional=%.2f -> %.2f",
            capped_by,
            raw_notional,
            notional,
        )
    else:
        notional = raw_notional
        units = raw_units
        capped_by = None

    return PositionSizeResult(
        units=units, notional=notional, risk_amount=risk_amount, capped_by=capped_by
    )


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeConfig:
    """Configuration for :func:`detect_regime`.

    Parameters
    ----------
    window:
        Rolling window size (in bars) for the ADF test.
    min_nobs:
        Minimum observations required within a window. Defaults to
        ``window``.
    significance_level:
        ADF p-value threshold. A window's p-value below this classifies
        it :attr:`MarketRegime.MEAN_REVERTING`; at or above,
        :attr:`MarketRegime.TRENDING`.
    """

    window: int = 60
    min_nobs: int | None = None
    significance_level: float = 0.05
    _resolved_min_nobs: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.window < 10:
            raise RiskError(
                f"window must be >= 10 for a meaningful ADF test, got {self.window}"
            )
        resolved = self.min_nobs if self.min_nobs is not None else self.window
        if resolved < 10:
            raise RiskError(f"min_nobs must be >= 10, got {resolved}")
        if resolved > self.window:
            raise RiskError(f"min_nobs ({resolved}) cannot exceed window ({self.window})")
        if not 0 < self.significance_level < 1:
            raise RiskError(
                f"significance_level must be in (0, 1), got {self.significance_level}"
            )
        object.__setattr__(self, "_resolved_min_nobs", resolved)


def _adf_pvalue(window_values: np.ndarray) -> float:
    """Run ADF on one rolling window, returning NaN if it can't be computed.

    ``adfuller`` raises ``ValueError`` on a constant (zero-variance)
    window; that is treated as "regime unknown" rather than propagating
    the exception and aborting the whole rolling computation.
    """
    try:
        return adfuller(window_values, autolag="AIC")[1]
    except ValueError:
        return np.nan


def detect_regime(prices: pd.Series, config: RegimeConfig = RegimeConfig()) -> pd.Series:
    """Classify each rolling window of ``prices`` as mean-reverting or trending.

    Runs the Augmented Dickey-Fuller test on each trailing window; a
    window whose ADF p-value is below ``config.significance_level``
    rejects the unit-root null hypothesis and is classified
    :attr:`MarketRegime.MEAN_REVERTING`. Otherwise it is classified
    :attr:`MarketRegime.TRENDING`. Windows without enough data (warm-up)
    or with a degenerate (constant) window are
    :attr:`MarketRegime.UNKNOWN`.

    .. note::
       This runs one ADF regression per rolling window, which is
       considerably more expensive than a closed-form indicator. It is
       intended for periodic regime checks (e.g. once per bar on a
       modest-frequency timeframe), not for high-frequency use on very
       long series without downsampling.

    Parameters
    ----------
    prices:
        Price series (e.g. the spread from
        :mod:`stat_arb.signal.cointegration`, or a raw pair leg).
    config:
        Regime detection configuration.

    Returns
    -------
    Series
        A series of :class:`MarketRegime` values (as their ``str``
        value), named ``"regime"``, sharing ``prices``' index.

    Raises
    ------
    RiskError
        If ``prices`` is invalid (see :func:`_validate_price_series`).
    """
    _validate_price_series(prices, config.window)

    pvalues = prices.rolling(config.window, min_periods=config._resolved_min_nobs).apply(
        _adf_pvalue, raw=True
    )

    def _classify(p: float) -> str:
        if pd.isna(p):
            return MarketRegime.UNKNOWN.value
        if p < config.significance_level:
            return MarketRegime.MEAN_REVERTING.value
        return MarketRegime.TRENDING.value

    regime = pvalues.apply(_classify)
    regime.name = "regime"

    counts = regime.value_counts()
    logger.info(
        "Regime detection: window=%d, mean_reverting=%d, trending=%d, unknown=%d",
        config.window,
        int(counts.get(MarketRegime.MEAN_REVERTING.value, 0)),
        int(counts.get(MarketRegime.TRENDING.value, 0)),
        int(counts.get(MarketRegime.UNKNOWN.value, 0)),
    )
    return regime


# ---------------------------------------------------------------------------
# Trend filter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrendFilterConfig:
    """Configuration for :func:`compute_trend_filter`.

    Parameters
    ----------
    window:
        Rolling window size (in bars) for the linear trend regression.
    min_nobs:
        Minimum observations required within a window. Defaults to
        ``window``.
    t_stat_threshold:
        Minimum absolute t-statistic of the rolling slope for a window
        to be flagged trending. ``2.0`` corresponds to roughly the 5%
        significance level for large samples.
    """

    window: int = 30
    min_nobs: int | None = None
    t_stat_threshold: float = 2.0
    _resolved_min_nobs: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.window < 3:
            raise RiskError(f"window must be >= 3, got {self.window}")
        resolved = self.min_nobs if self.min_nobs is not None else self.window
        if resolved < 3:
            raise RiskError(f"min_nobs must be >= 3, got {resolved}")
        if resolved > self.window:
            raise RiskError(f"min_nobs ({resolved}) cannot exceed window ({self.window})")
        if self.t_stat_threshold <= 0:
            raise RiskError("t_stat_threshold must be positive")
        object.__setattr__(self, "_resolved_min_nobs", resolved)


@dataclass(frozen=True)
class TrendFilterResult:
    """Output of :func:`compute_trend_filter`. All series share the input index.

    Attributes
    ----------
    slope:
        Rolling linear trend slope (price units per bar).
    t_stat:
        Rolling slope t-statistic (``slope / std_err``).
    is_trending:
        ``True`` where ``abs(t_stat) >= t_stat_threshold``.
    direction:
        ``1`` where trending up, ``-1`` where trending down, ``0``
        where not trending (including warm-up/unavailable windows).
    """

    slope: pd.Series
    t_stat: pd.Series
    is_trending: pd.Series
    direction: pd.Series


def compute_trend_filter(
    prices: pd.Series, config: TrendFilterConfig = TrendFilterConfig()
) -> TrendFilterResult:
    """Detect statistically significant rolling linear trends in ``prices``.

    Regresses ``prices`` on a linear time index (``0, 1, 2, ...``) over
    each trailing window via ``statsmodels.regression.rolling.RollingOLS``
    and tests the slope's significance. Used as a gate to veto new
    mean-reversion entries while a strong directional trend is active —
    a pair can be cointegrated over the long run yet still trend hard
    enough, for long enough, to blow through a stop before reverting.

    Parameters
    ----------
    prices:
        Price series to test for a trend.
    config:
        Trend filter configuration.

    Returns
    -------
    TrendFilterResult

    Raises
    ------
    RiskError
        If ``prices`` is invalid.
    """
    _validate_price_series(prices, config.window)

    time_index = np.arange(len(prices), dtype=float)
    exog = pd.DataFrame({"const": 1.0, "t": time_index}, index=prices.index)

    model = RollingOLS(
        prices.astype(float), exog, window=config.window, min_nobs=config._resolved_min_nobs
    )
    fit_result = model.fit(params_only=False)

    slope = fit_result.params["t"]
    slope.name = "slope"
    std_err = fit_result.bse["t"]

    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = slope / std_err
    t_stat.name = "t_stat"

    is_trending = t_stat.abs() >= config.t_stat_threshold
    is_trending = is_trending.fillna(False)
    is_trending.name = "is_trending"

    direction = pd.Series(0, index=prices.index, dtype=int, name="direction")
    direction.loc[is_trending & (slope > 0)] = 1
    direction.loc[is_trending & (slope < 0)] = -1

    logger.info(
        "Trend filter: window=%d, %d/%d bar(s) flagged trending",
        config.window,
        int(is_trending.sum()),
        int(slope.notna().sum()),
    )

    return TrendFilterResult(
        slope=slope, t_stat=t_stat, is_trending=is_trending, direction=direction
    )


def _validate_price_series(prices: pd.Series, window: int) -> None:
    """Shared validation for regime/trend-filter price series inputs."""
    if not isinstance(prices, pd.Series):
        raise RiskError("prices must be a pandas Series")
    if len(prices) == 0:
        raise RiskError("prices must not be empty")
    if not pd.api.types.is_numeric_dtype(prices):
        raise RiskError("prices must be numeric")
    if not np.isfinite(prices.to_numpy(dtype=float)).all():
        raise RiskError("prices contains NaN or infinite values")
    if not prices.index.is_monotonic_increasing:
        raise RiskError("prices index must be sorted in increasing order")
    if len(prices) < window:
        raise RiskError(f"Need at least {window} observations, got {len(prices)}")


# ---------------------------------------------------------------------------
# Maximum exposure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExposureConfig:
    """Portfolio-level exposure limits.

    Parameters
    ----------
    max_gross_exposure_pct:
        Maximum total notional across all open positions, as a fraction
        of equity.
    max_pair_exposure_pct:
        Maximum notional in a single pair, as a fraction of equity.
    max_positions:
        Maximum number of distinct concurrently open pairs.
    """

    max_gross_exposure_pct: float = 0.5
    max_pair_exposure_pct: float = 0.3
    max_positions: int = 2

    def __post_init__(self) -> None:
        if self.max_gross_exposure_pct <= 0:
            raise RiskError("max_gross_exposure_pct must be positive")
        if self.max_pair_exposure_pct <= 0:
            raise RiskError("max_pair_exposure_pct must be positive")
        if self.max_positions < 1:
            raise RiskError("max_positions must be >= 1")


@dataclass(frozen=True)
class ExposureCheckResult:
    """Result of :func:`check_exposure_limits`.

    Attributes
    ----------
    allowed:
        ``True`` if adding the proposed position would not breach any
        configured limit.
    reasons:
        Human-readable reasons for each breached limit (empty if
        ``allowed`` is ``True``).
    """

    allowed: bool
    reasons: tuple[str, ...]


def check_exposure_limits(
    equity: float,
    current_positions: Mapping[str, float],
    new_pair: str,
    new_notional: float,
    config: ExposureConfig = ExposureConfig(),
) -> ExposureCheckResult:
    """Check whether adding a new position would breach portfolio exposure limits.

    Parameters
    ----------
    equity:
        Current account equity. Must be positive.
    current_positions:
        Mapping of pair -> current notional exposure for all open
        positions (excluding the proposed new one).
    new_pair:
        Pair the proposed new position is on.
    new_notional:
        Notional size of the proposed new position. Must be non-negative.
    config:
        Exposure limit configuration.

    Returns
    -------
    ExposureCheckResult

    Raises
    ------
    RiskError
        If ``equity`` is not positive or ``new_notional`` is negative.
    """
    if equity <= 0:
        raise RiskError(f"equity must be positive, got {equity}")
    if new_notional < 0:
        raise RiskError(f"new_notional must be non-negative, got {new_notional}")

    reasons: list[str] = []

    is_new_pair = new_pair not in current_positions
    if is_new_pair and len(current_positions) >= config.max_positions:
        reasons.append(
            f"max_positions limit reached ({len(current_positions)}/{config.max_positions})"
        )

    gross_exposure = sum(current_positions.values()) + new_notional
    gross_limit = equity * config.max_gross_exposure_pct
    if gross_exposure > gross_limit:
        reasons.append(
            f"gross exposure {gross_exposure:.2f} exceeds limit {gross_limit:.2f} "
            f"({config.max_gross_exposure_pct:.0%} of equity)"
        )

    pair_exposure = current_positions.get(new_pair, 0.0) + new_notional
    pair_limit = equity * config.max_pair_exposure_pct
    if pair_exposure > pair_limit:
        reasons.append(
            f"{new_pair} exposure {pair_exposure:.2f} exceeds limit {pair_limit:.2f} "
            f"({config.max_pair_exposure_pct:.0%} of equity)"
        )

    result = ExposureCheckResult(allowed=not reasons, reasons=tuple(reasons))
    if not result.allowed:
        logger.warning("Exposure limits breached for %s: %s", new_pair, "; ".join(reasons))
    return result


# ---------------------------------------------------------------------------
# Cooldown logic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CooldownConfig:
    """Configuration for :class:`CooldownTracker`.

    Parameters
    ----------
    cooldown_period:
        Duration a pair stays in cooldown after a qualifying exit.
    apply_to_all_exits:
        If ``False`` (default), only stop-loss exits arm the cooldown.
        If ``True``, every recorded exit arms it.
    """

    cooldown_period: timedelta = timedelta(hours=4)
    apply_to_all_exits: bool = False

    def __post_init__(self) -> None:
        if self.cooldown_period <= timedelta(0):
            raise RiskError("cooldown_period must be positive")


class CooldownTracker:
    """Tracks recent stop-loss exits per pair and blocks re-entry during cooldown.

    Unlike the rest of this module's configuration objects, this class
    is intentionally mutable: its purpose is to remember exit history
    across calls to :meth:`record_exit` / :meth:`is_in_cooldown`.
    """

    def __init__(self, config: CooldownConfig | None = None) -> None:
        """Initialize an empty tracker.

        Parameters
        ----------
        config:
            Cooldown configuration. Defaults to ``CooldownConfig()``.
        """
        self.config = config or CooldownConfig()
        self._last_qualifying_exit: dict[str, datetime] = {}

    def record_exit(self, pair: str, exit_time: datetime, stop_loss_triggered: bool) -> None:
        """Record a position exit.

        Parameters
        ----------
        pair:
            Pair that was exited.
        exit_time:
            Timestamp of the exit.
        stop_loss_triggered:
            Whether the exit was caused by the stop loss. Only
            stop-loss exits arm the cooldown unless
            ``config.apply_to_all_exits`` is ``True``.
        """
        if stop_loss_triggered or self.config.apply_to_all_exits:
            self._last_qualifying_exit[pair] = exit_time
            logger.info(
                "Cooldown armed for %s at %s (stop_loss=%s), period=%s",
                pair,
                exit_time,
                stop_loss_triggered,
                self.config.cooldown_period,
            )

    def is_in_cooldown(self, pair: str, current_time: datetime) -> bool:
        """Check whether ``pair`` is currently in cooldown.

        Parameters
        ----------
        pair:
            Pair to check.
        current_time:
            Current timestamp.

        Returns
        -------
        bool
            ``True`` if ``pair`` had a qualifying exit within
            ``config.cooldown_period`` of ``current_time``.
        """
        last_exit = self._last_qualifying_exit.get(pair)
        if last_exit is None:
            return False
        in_cooldown = (current_time - last_exit) < self.config.cooldown_period
        return in_cooldown

    def remaining_cooldown(self, pair: str, current_time: datetime) -> timedelta:
        """Time remaining until ``pair``'s cooldown expires (zero if none active)."""
        last_exit = self._last_qualifying_exit.get(pair)
        if last_exit is None:
            return timedelta(0)
        remaining = self.config.cooldown_period - (current_time - last_exit)
        return max(remaining, timedelta(0))

    def reset(self, pair: str | None = None) -> None:
        """Clear cooldown state for ``pair``, or for all pairs if ``None``."""
        if pair is None:
            self._last_qualifying_exit.clear()
        else:
            self._last_qualifying_exit.pop(pair, None)


# ---------------------------------------------------------------------------
# RiskEngine orchestrator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskEngineConfig:
    """Aggregate configuration for :class:`RiskEngine`.

    Parameters
    ----------
    stop_loss, position_sizing, regime, trend_filter, exposure, cooldown:
        Sub-configurations for each risk control.
    require_mean_reverting_regime:
        If ``True`` (default), entries are only allowed when the latest
        detected regime is :attr:`MarketRegime.MEAN_REVERTING`.
    block_entry_when_trending:
        If ``True`` (default), entries are blocked when
        :func:`compute_trend_filter` currently flags the pair as
        trending.
    """

    stop_loss: StopLossConfig = field(default_factory=StopLossConfig)
    position_sizing: PositionSizingConfig = field(default_factory=PositionSizingConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    trend_filter: TrendFilterConfig = field(default_factory=TrendFilterConfig)
    exposure: ExposureConfig = field(default_factory=ExposureConfig)
    cooldown: CooldownConfig = field(default_factory=CooldownConfig)
    require_mean_reverting_regime: bool = True
    block_entry_when_trending: bool = True


@dataclass(frozen=True)
class EntryDecision:
    """Result of :meth:`RiskEngine.evaluate_entry`.

    Attributes
    ----------
    allowed:
        ``True`` if the entry passes every configured risk control.
    reasons:
        Human-readable reasons for each control that blocked the entry
        (empty if ``allowed`` is ``True``).
    position_size:
        Sizing result, if computed (``None`` only when validation fails
        before sizing would make sense).
    stop_loss_price:
        Computed stop-loss price for the proposed entry.
    regime:
        Latest detected market regime.
    is_trending:
        Latest trend-filter flag.
    """

    allowed: bool
    reasons: tuple[str, ...]
    position_size: PositionSizeResult | None
    stop_loss_price: float | None
    regime: MarketRegime | None
    is_trending: bool | None


class RiskEngine:
    """Composes stop loss, sizing, regime/trend filters, exposure, and cooldown.

    Usage::

        engine = RiskEngine(RiskEngineConfig())
        decision = engine.evaluate_entry(
            pair="BTC/USDC:USDC",
            side=PositionSide.LONG,
            entry_price=50_000.0,
            equity=10_000.0,
            prices=spread_series,
            current_positions={},
            current_time=datetime.now(timezone.utc),
        )
        if decision.allowed:
            ...  # place the order sized at decision.position_size.units
        engine.record_exit("BTC/USDC:USDC", exit_time, stop_loss_triggered=True)
    """

    def __init__(self, config: RiskEngineConfig | None = None) -> None:
        """Initialize the engine.

        Parameters
        ----------
        config:
            Aggregate risk configuration. Defaults to
            ``RiskEngineConfig()``.
        """
        self.config = config or RiskEngineConfig()
        self._cooldown = CooldownTracker(self.config.cooldown)

    def evaluate_entry(
        self,
        pair: str,
        side: PositionSide,
        entry_price: float,
        equity: float,
        prices: pd.Series,
        current_positions: Mapping[str, float],
        current_time: datetime,
    ) -> EntryDecision:
        """Evaluate whether a proposed entry passes all configured risk controls.

        Parameters
        ----------
        pair:
            Pair to enter.
        side:
            Proposed position direction.
        entry_price:
            Proposed entry price.
        equity:
            Current account equity.
        prices:
            Recent price/spread series for ``pair``, used for regime
            detection and the trend filter. Its last value should be at
            or before ``current_time``.
        current_positions:
            Mapping of pair -> current notional exposure for all open
            positions (excluding ``pair``, if not currently held).
        current_time:
            Current timestamp, used for the cooldown check.

        Returns
        -------
        EntryDecision
        """
        reasons: list[str] = []

        if self._cooldown.is_in_cooldown(pair, current_time):
            remaining = self._cooldown.remaining_cooldown(pair, current_time)
            reasons.append(f"{pair} is in cooldown for another {remaining}")

        regime_series = detect_regime(prices, self.config.regime)
        latest_regime = MarketRegime(regime_series.iloc[-1])
        if (
            self.config.require_mean_reverting_regime
            and latest_regime is not MarketRegime.MEAN_REVERTING
        ):
            reasons.append(f"regime is {latest_regime.value}, not mean_reverting")

        trend_result = compute_trend_filter(prices, self.config.trend_filter)
        is_trending_now = bool(trend_result.is_trending.iloc[-1])
        if self.config.block_entry_when_trending and is_trending_now:
            reasons.append("trend filter is currently flagging a significant trend")

        stop_loss_price = compute_stop_loss_price(entry_price, side, self.config.stop_loss)
        position_size = calculate_position_size(
            equity, entry_price, stop_loss_price, self.config.position_sizing
        )

        exposure_result = check_exposure_limits(
            equity, current_positions, pair, position_size.notional, self.config.exposure
        )
        reasons.extend(exposure_result.reasons)

        allowed = not reasons
        logger.info(
            "Entry evaluation for %s (%s): allowed=%s%s",
            pair,
            side.value,
            allowed,
            f", reasons={reasons}" if reasons else "",
        )

        return EntryDecision(
            allowed=allowed,
            reasons=tuple(reasons),
            position_size=position_size,
            stop_loss_price=stop_loss_price,
            regime=latest_regime,
            is_trending=is_trending_now,
        )

    def check_stop_loss(self, current_price: float, entry_price: float, side: PositionSide) -> bool:
        """Check whether the current price has breached the configured stop loss."""
        return is_stop_loss_triggered(current_price, entry_price, side, self.config.stop_loss)

    def record_exit(self, pair: str, exit_time: datetime, stop_loss_triggered: bool) -> None:
        """Record a position exit, arming the cooldown per configuration."""
        self._cooldown.record_exit(pair, exit_time, stop_loss_triggered)

    def is_in_cooldown(self, pair: str, current_time: datetime) -> bool:
        """Check whether ``pair`` is currently in cooldown."""
        return self._cooldown.is_in_cooldown(pair, current_time)
