"""Rolling OLS hedge-ratio engine for statistical arbitrage pairs.

Given two aligned price series (typically the ``close`` columns of two
pairs produced by :mod:`stat_arb.data.market_data`), this module fits a
rolling ordinary least squares regression of one series on the other and
produces a time-varying ("dynamic") hedge ratio — the beta a pairs trader
would use to size the offsetting leg of a spread trade at each point in
time.

This module contains **no signal generation or trading logic** (no
z-scores, no entry/exit rules) — it only produces the hedge ratio and the
diagnostics needed to trust it.

Design decisions
-----------------
* **`statsmodels.regression.rolling.RollingOLS` does the regression
  math.** It is a mature, actively maintained implementation of rolling
  OLS with correct handling of window edges, missing data, and standard
  errors. Re-deriving rolling least squares by hand (e.g. looping over
  windows and calling ``numpy.linalg.lstsq``) would duplicate well-tested
  code and be far slower, since ``RollingOLS`` updates its window
  incrementally rather than refitting from scratch.
* **The hedge ratio is the slope of `y = alpha + beta * x`.** By
  convention here, ``y`` is the "dependent" leg of the pair and ``x`` is
  the "independent" leg; ``beta`` (the hedge ratio) is the number of
  units of ``x`` needed to offset one unit of ``y`` in a market-neutral
  spread. Which physical pair leg plays ``y`` vs. ``x`` is a decision for
  the (not-yet-built) pair-selection module — this engine is symmetric
  and pair-agnostic.
* **Numerical stability is validated per window, not assumed.** A rolling
  window can become numerically degenerate — e.g. the independent
  variable is nearly constant within the window, making the design
  matrix nearly singular — and OLS will still return *a* number even
  though it is numerically meaningless. Each window's condition number
  is computed (see :func:`_rolling_condition_number`) and combined with
  a sanity bound on the hedge ratio's magnitude to flag windows as
  unstable; unstable estimates are not silently trusted.
* **Condition number is computed analytically, not via a per-window
  `numpy.linalg.cond` loop.** For the two-column design matrix
  ``[1, x]``, ``X'X`` is a 2x2 matrix built entirely from rolling sums of
  ``x`` and ``x**2`` (both vectorized `pandas` rolling operations), and a
  2x2 symmetric matrix has a closed-form eigenvalue solution. This gives
  the same 2-norm condition number ``numpy.linalg.cond`` would return
  (verified in the unit tests) without an O(n) Python loop calling SVD on
  every window.
* **Unstable windows are forward-filled by default, not silently
  dropped.** Downstream consumers (e.g. a spread/z-score module) need a
  hedge ratio at every timestamp. Forward-filling the last numerically
  trustworthy estimate is a conservative, explicit choice — a stale but
  valid hedge ratio is safer than an unstable one, and the choice can be
  disabled (``ffill_unstable=False``) for callers that want to see raw
  ``NaN`` gaps instead.
* **Only economically significant hedge-ratio changes are logged.**
  Logging every window-to-window fluctuation of a rolling statistic
  would flood the log with noise. Instead, a change is logged only when
  it exceeds ``hedge_ratio_change_log_threshold`` (a relative move, by
  default 5%), which is the kind of jump a pairs trader actually cares
  about — e.g. because it changes position sizing materially.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.regression.rolling import RollingOLS

logger = logging.getLogger(__name__)

#: Minimum rolling window size: 2 regression parameters (intercept, slope)
#: plus at least one residual degree of freedom.
_MIN_WINDOW = 3


class RegressionError(Exception):
    """Raised when regression inputs are invalid or a fit cannot be produced."""


@dataclass(frozen=True)
class RollingRegressionConfig:
    """Configuration for :class:`RollingRegressionEngine`.

    Parameters
    ----------
    window:
        Number of observations in each rolling window.
    min_nobs:
        Minimum number of non-missing observations required within a
        window before it produces an estimate. Defaults to ``window``
        (i.e. no partial-window estimates at the start of the series).
    condition_number_threshold:
        A window's design-matrix condition number above this value marks
        the window's estimate as numerically unstable. ``1e8`` is a
        standard rule-of-thumb threshold for OLS ill-conditioning.
    max_abs_hedge_ratio:
        A window whose estimated hedge ratio exceeds this magnitude is
        also flagged unstable, independent of the condition number —
        catching cases where the regression is technically well-posed
        but has produced an economically implausible hedge ratio.
    hedge_ratio_change_log_threshold:
        Minimum relative change (e.g. ``0.05`` = 5%) between consecutive
        stable hedge-ratio estimates required to emit a log entry.
    ffill_unstable:
        If ``True`` (default), unstable windows have their hedge ratio
        forward-filled from the last stable estimate. If ``False``,
        unstable windows are left as ``NaN``.
    """

    window: int = 60
    min_nobs: int | None = None
    condition_number_threshold: float = 1e8
    max_abs_hedge_ratio: float = 50.0
    hedge_ratio_change_log_threshold: float = 0.05
    ffill_unstable: bool = True
    _resolved_min_nobs: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.window < _MIN_WINDOW:
            raise RegressionError(
                f"window must be >= {_MIN_WINDOW} (need at least one residual "
                f"degree of freedom for a 2-parameter regression), got {self.window}"
            )
        resolved = self.min_nobs if self.min_nobs is not None else self.window
        if resolved < _MIN_WINDOW:
            raise RegressionError(f"min_nobs must be >= {_MIN_WINDOW}, got {resolved}")
        if resolved > self.window:
            raise RegressionError(
                f"min_nobs ({resolved}) cannot exceed window ({self.window})"
            )
        if self.condition_number_threshold <= 0:
            raise RegressionError("condition_number_threshold must be positive")
        if self.max_abs_hedge_ratio <= 0:
            raise RegressionError("max_abs_hedge_ratio must be positive")
        if self.hedge_ratio_change_log_threshold < 0:
            raise RegressionError("hedge_ratio_change_log_threshold must be non-negative")
        object.__setattr__(self, "_resolved_min_nobs", resolved)


@dataclass(frozen=True)
class RollingRegressionResult:
    """Output of a rolling OLS fit: ``y = intercept + hedge_ratio * x``.

    All series share the same index as the input data.

    Attributes
    ----------
    hedge_ratio:
        The dynamic hedge ratio (rolling OLS slope), with numerically
        unstable windows handled per
        :attr:`RollingRegressionConfig.ffill_unstable`.
    raw_hedge_ratio:
        The hedge ratio exactly as estimated by ``RollingOLS``, before
        any stability-driven forward-fill is applied.
    intercept:
        The rolling OLS intercept.
    r_squared:
        The rolling R-squared of each window's fit.
    condition_number:
        The design matrix's 2-norm condition number for each window.
    is_stable:
        ``True`` where the window's estimate passed both the condition-
        number and hedge-ratio-magnitude checks.
    config:
        The configuration used to produce this result.
    """

    hedge_ratio: pd.Series
    raw_hedge_ratio: pd.Series
    intercept: pd.Series
    r_squared: pd.Series
    condition_number: pd.Series
    is_stable: pd.Series
    config: RollingRegressionConfig

    @property
    def n_unstable(self) -> int:
        """Number of windows without a stable hedge-ratio estimate.

        Includes both numerically unstable windows and warm-up windows
        before ``min_nobs`` observations are available (which have no
        estimate at all, rather than an unstable one).
        """
        return int((~self.is_stable).sum())


def _validate_inputs(y: pd.Series, x: pd.Series, config: RollingRegressionConfig) -> None:
    """Validate regression inputs before fitting.

    Raises
    ------
    RegressionError
        If ``y``/``x`` are not compatible, non-numeric, contain missing
        values, or are too short for the configured window.
    """
    if not isinstance(y, pd.Series) or not isinstance(x, pd.Series):
        raise RegressionError("y and x must both be pandas Series")

    if len(y) != len(x):
        raise RegressionError(f"y and x must have equal length, got {len(y)} vs {len(x)}")

    if len(y) == 0:
        raise RegressionError("y and x must not be empty")

    if not y.index.equals(x.index):
        raise RegressionError("y and x must share an identical index (align them first)")

    if not pd.api.types.is_numeric_dtype(y) or not pd.api.types.is_numeric_dtype(x):
        raise RegressionError("y and x must both be numeric")

    if y.isna().any() or x.isna().any():
        raise RegressionError(
            "y and x must not contain missing values (fill gaps before regressing, "
            "e.g. via stat_arb.data.market_data.clean_and_fill)"
        )

    if len(y) < config.window:
        raise RegressionError(
            f"Need at least {config.window} observations for window={config.window}, "
            f"got {len(y)}"
        )


def _rolling_condition_number(x: pd.Series, config: RollingRegressionConfig) -> pd.Series:
    """Compute the 2-norm condition number of the design matrix ``[1, x]`` per window.

    For design matrix ``X = [1, x]``, ``X'X = [[n, sum(x)], [sum(x), sum(x^2)]]``
    is a 2x2 symmetric matrix whose eigenvalues have a closed form. The
    2-norm condition number of ``X`` equals ``sqrt(eig_max / eig_min)`` of
    ``X'X``. Computing it this way is fully vectorized via rolling sums,
    rather than materializing and decomposing each window's matrix.

    Parameters
    ----------
    x:
        The independent variable series.
    config:
        Regression configuration (supplies window size / min_nobs).

    Returns
    -------
    Series
        Condition number per window, aligned to ``x``'s index. ``inf``
        where the window's design matrix is singular (e.g. constant
        ``x``); ``NaN`` where too few observations are available.
    """
    n = x.rolling(config.window, min_periods=config._resolved_min_nobs).count()
    sum_x = x.rolling(config.window, min_periods=config._resolved_min_nobs).sum()
    sum_x2 = (x**2).rolling(config.window, min_periods=config._resolved_min_nobs).sum()

    a, b, d = n, sum_x, sum_x2
    mean_term = (a + d) / 2
    half_diff = (a - d) / 2
    spread = np.sqrt(half_diff**2 + b**2)
    eig_max = mean_term + spread
    eig_min = mean_term - spread

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = eig_max / eig_min
        condition_number = np.sqrt(ratio)

    # Non-positive/zero eig_min means a singular (or numerically singular)
    # design matrix for that window.
    condition_number = condition_number.where(eig_min > 0, np.inf)
    return condition_number


def _log_hedge_ratio_changes(hedge_ratio: pd.Series, config: RollingRegressionConfig) -> None:
    """Log economically significant changes in the (stable) hedge ratio.

    Only relative changes at or above
    :attr:`RollingRegressionConfig.hedge_ratio_change_log_threshold` are
    logged, to avoid flooding the log with routine window-to-window
    noise.
    """
    valid = hedge_ratio.dropna()
    if len(valid) < 2:
        return

    previous = valid.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_change = (valid - previous).abs() / previous.abs()

    significant = relative_change[relative_change >= config.hedge_ratio_change_log_threshold]
    for timestamp, pct_change in significant.items():
        logger.info(
            "Hedge ratio change at %s: %.6f -> %.6f (%.2f%%)",
            timestamp,
            previous.loc[timestamp],
            valid.loc[timestamp],
            pct_change * 100,
        )

    if len(significant):
        logger.info(
            "%d hedge ratio change(s) >= %.1f%% logged out of %d stable estimates",
            len(significant),
            config.hedge_ratio_change_log_threshold * 100,
            len(valid),
        )


class RollingRegressionEngine:
    """Fits a rolling OLS regression to produce a dynamic hedge ratio.

    Usage::

        engine = RollingRegressionEngine(RollingRegressionConfig(window=60))
        result = engine.fit(y=btc_close, x=eth_close)
        result.hedge_ratio  # dynamic hedge ratio series
    """

    def __init__(self, config: RollingRegressionConfig | None = None) -> None:
        """Initialize the engine.

        Parameters
        ----------
        config:
            Regression configuration. Defaults to
            ``RollingRegressionConfig()`` if not provided.
        """
        self.config = config or RollingRegressionConfig()

    def fit(self, y: pd.Series, x: pd.Series) -> RollingRegressionResult:
        """Fit a rolling OLS regression ``y = intercept + hedge_ratio * x``.

        Parameters
        ----------
        y:
            Dependent variable series (e.g. one pair leg's close price).
        x:
            Independent variable series (e.g. the other leg's close
            price), sharing ``y``'s index.

        Returns
        -------
        RollingRegressionResult
            The dynamic hedge ratio and its stability diagnostics.

        Raises
        ------
        RegressionError
            If the inputs are invalid (see :func:`_validate_inputs`).
        """
        config = self.config
        _validate_inputs(y, x, config)

        logger.info(
            "Fitting rolling OLS: n=%d, window=%d, min_nobs=%d",
            len(y),
            config.window,
            config._resolved_min_nobs,
        )

        exog = pd.DataFrame({"const": 1.0, "x": x.to_numpy(dtype=float)}, index=x.index)
        endog = y.astype(float)

        model = RollingOLS(
            endog, exog, window=config.window, min_nobs=config._resolved_min_nobs
        )
        fit_result = model.fit(params_only=False)

        raw_hedge_ratio = fit_result.params["x"]
        intercept = fit_result.params["const"]
        r_squared = fit_result.rsquared

        condition_number = _rolling_condition_number(x, config)

        is_stable = (
            raw_hedge_ratio.notna()
            & np.isfinite(raw_hedge_ratio)
            & (raw_hedge_ratio.abs() <= config.max_abs_hedge_ratio)
            & (condition_number <= config.condition_number_threshold)
        )

        hedge_ratio = raw_hedge_ratio.where(is_stable)
        if config.ffill_unstable:
            hedge_ratio = hedge_ratio.ffill()

        n_unstable = int((~is_stable & raw_hedge_ratio.notna()).sum())
        if n_unstable:
            logger.warning(
                "%d/%d window(s) flagged numerically unstable "
                "(condition_number > %.1e or |hedge_ratio| > %.1f)%s",
                n_unstable,
                int(raw_hedge_ratio.notna().sum()),
                config.condition_number_threshold,
                config.max_abs_hedge_ratio,
                " — forward-filled from last stable estimate" if config.ffill_unstable else "",
            )

        _log_hedge_ratio_changes(hedge_ratio, config)

        return RollingRegressionResult(
            hedge_ratio=hedge_ratio,
            raw_hedge_ratio=raw_hedge_ratio,
            intercept=intercept,
            r_squared=r_squared,
            condition_number=condition_number,
            is_stable=is_stable,
            config=config,
        )
