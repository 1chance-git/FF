"""Statistical arbitrage spread engine: spread, rolling z-score, cointegration gate.

Given two aligned price series and a (possibly time-varying) hedge ratio
— typically the output of :mod:`stat_arb.signal.regression` — this module
computes the pair spread, its rolling mean/standard deviation/z-score,
and validates that the pair is actually cointegrated before any of that
is trusted as a trading signal input.

This module contains **no entry/exit rule logic** — it produces the
statistics a signal-generation module would consume, not trading
decisions themselves.

Design decisions
-----------------
* **Cointegration is tested with `statsmodels.tsa.stattools.coint`
  (Engle-Granger).** This is the standard, well-tested implementation of
  the two-step Engle-Granger cointegration test. Reimplementing the
  augmented Dickey-Fuller regression and critical-value tables that
  underlie it would duplicate mature statistical code for no benefit.
* **Cointegration validation gates spread computation by default.**
  A spread computed from a non-cointegrated pair is not mean-reverting
  and any z-score built on it is not a meaningful trading signal — it is
  just noise. `CointegrationEngine.compute()` therefore raises
  `CointegrationError` when the pair fails the test, unless the caller
  explicitly opts out via `CointegrationConfig(require_cointegration=False)`
  (e.g. for research/exploration where fitting anyway is useful).
* **Lookahead bias is prevented at its two possible entry points:**

  1. *Rolling statistics never see the future.* `rolling_mean`,
     `rolling_std`, and `rolling_zscore` all use `pandas.Series.rolling`
     with `center` hardcoded to `False` (not exposed as a parameter) —
     a centered rolling window would, by construction, use future
     observations to describe the present, which is lookahead bias by
     definition.
  2. *The hedge ratio is lagged before it is used to build the spread.*
     `compute_spread` shifts the hedge ratio series by
     `hedge_ratio_lag` bars (default 1) before combining it with `y`/`x`.
     This matters even though a *trailing* rolling regression only uses
     data through the current bar (so using it "as of" the current bar
     is not, strictly, using future information): when a bar's own
     price is part of the very regression window that produces that
     bar's hedge ratio, OLS is explicitly minimizing that bar's squared
     residual as part of the fit. The resulting in-sample residual is
     therefore systematically smaller than a genuine out-of-sample
     residual would be — an in-sample shrinkage bias that inflates
     apparent mean-reversion. Lagging the hedge ratio by at least one
     bar removes this: the hedge ratio applied at bar *t* was estimated
     using no information from bar *t* itself.
* **A degenerate (near-zero) rolling standard deviation yields `NaN`,
  not `inf`.** Dividing by a standard deviation that has collapsed to
  numerical noise would produce an enormous, meaningless z-score.
  `rolling_zscore` floors the standard deviation at
  `zscore_std_floor` and treats windows below it as unavailable data
  rather than fabricating an extreme signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

logger = logging.getLogger(__name__)

#: Minimum rolling window for spread statistics: need at least 2 points
#: for a standard deviation to be defined.
_MIN_WINDOW = 2


class CointegrationError(Exception):
    """Raised for invalid inputs, or when a pair fails cointegration validation."""


@dataclass(frozen=True)
class CointegrationTestResult:
    """Result of an Engle-Granger cointegration test between two series.

    Attributes
    ----------
    test_statistic:
        The Engle-Granger test statistic (more negative = stronger
        evidence of cointegration).
    p_value:
        p-value of the test. Small values (below the configured
        significance level) reject the null hypothesis of no
        cointegration.
    critical_values:
        Critical values at the 1%, 5%, and 10% significance levels.
    significance_level:
        The significance level used to determine :attr:`is_cointegrated`.
    is_cointegrated:
        ``True`` if ``p_value < significance_level``.
    """

    test_statistic: float
    p_value: float
    critical_values: dict[str, float]
    significance_level: float
    is_cointegrated: bool


@dataclass(frozen=True)
class CointegrationConfig:
    """Configuration for :class:`CointegrationEngine`.

    Parameters
    ----------
    spread_window:
        Rolling window size (in bars) for the spread's mean, standard
        deviation, and z-score.
    min_periods:
        Minimum observations required within a window to produce a
        statistic. Defaults to ``spread_window`` (no partial-window
        estimates at the start of the series).
    hedge_ratio_lag:
        Number of bars to lag the hedge ratio before applying it to
        compute the spread. Must be >= 1 to avoid in-sample shrinkage
        bias (see module docstring); set to ``0`` only with a full
        understanding of that tradeoff — doing so logs a warning.
    significance_level:
        Significance level for the Engle-Granger cointegration test.
    require_cointegration:
        If ``True`` (default), :meth:`CointegrationEngine.compute` raises
        :class:`CointegrationError` when the pair fails the
        cointegration test at ``significance_level``. If ``False``, the
        test still runs and its result is returned, but spread
        statistics are computed regardless.
    zscore_std_floor:
        Rolling standard deviation values at or below this are treated
        as numerically degenerate; the corresponding z-score is ``NaN``
        rather than an inflated or infinite value.
    """

    spread_window: int = 60
    min_periods: int | None = None
    hedge_ratio_lag: int = 1
    significance_level: float = 0.05
    require_cointegration: bool = True
    zscore_std_floor: float = 1e-8
    _resolved_min_periods: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.spread_window < _MIN_WINDOW:
            raise CointegrationError(
                f"spread_window must be >= {_MIN_WINDOW}, got {self.spread_window}"
            )
        resolved = self.min_periods if self.min_periods is not None else self.spread_window
        if resolved < _MIN_WINDOW:
            raise CointegrationError(f"min_periods must be >= {_MIN_WINDOW}, got {resolved}")
        if resolved > self.spread_window:
            raise CointegrationError(
                f"min_periods ({resolved}) cannot exceed spread_window ({self.spread_window})"
            )
        if self.hedge_ratio_lag < 0:
            raise CointegrationError(
                f"hedge_ratio_lag must be >= 0, got {self.hedge_ratio_lag}"
            )
        if not 0 < self.significance_level < 1:
            raise CointegrationError(
                f"significance_level must be in (0, 1), got {self.significance_level}"
            )
        if self.zscore_std_floor <= 0:
            raise CointegrationError("zscore_std_floor must be positive")
        object.__setattr__(self, "_resolved_min_periods", resolved)


@dataclass(frozen=True)
class SpreadResult:
    """Output of :meth:`CointegrationEngine.compute`.

    All series share the same index as the input data.

    Attributes
    ----------
    spread:
        ``y - lagged_hedge_ratio * x``.
    rolling_mean:
        Trailing rolling mean of :attr:`spread`.
    rolling_std:
        Trailing rolling standard deviation of :attr:`spread`.
    zscore:
        Trailing rolling z-score of :attr:`spread`.
    cointegration:
        Result of the Engle-Granger cointegration test on ``(y, x)``.
    config:
        The configuration used to produce this result.
    """

    spread: pd.Series
    rolling_mean: pd.Series
    rolling_std: pd.Series
    zscore: pd.Series
    cointegration: CointegrationTestResult
    config: CointegrationConfig


def _validate_series_inputs(series: dict[str, pd.Series]) -> None:
    """Validate a set of named series share an index and contain no NaNs/inf.

    Parameters
    ----------
    series:
        Mapping of a descriptive name (used in error messages) to the
        series to validate, e.g. ``{"y": y, "x": x}``.

    Raises
    ------
    CointegrationError
        If any series is empty, non-numeric, contains missing/infinite
        values, has an unsorted index, or the series don't share an
        identical index.
    """
    names = list(series)
    reference_name, reference = names[0], series[names[0]]

    for name, s in series.items():
        if not isinstance(s, pd.Series):
            raise CointegrationError(f"'{name}' must be a pandas Series")
        if len(s) == 0:
            raise CointegrationError(f"'{name}' must not be empty")
        if not pd.api.types.is_numeric_dtype(s):
            raise CointegrationError(f"'{name}' must be numeric")
        if not np.isfinite(s.to_numpy(dtype=float)).all():
            raise CointegrationError(f"'{name}' contains NaN or infinite values")
        if not s.index.is_monotonic_increasing:
            raise CointegrationError(
                f"'{name}' index must be sorted in increasing order "
                "(an unsorted index can silently introduce lookahead bias "
                "into rolling computations)"
            )
        if not s.index.equals(reference.index):
            raise CointegrationError(
                f"'{name}' must share an identical index with '{reference_name}'"
            )


def test_cointegration(
    y: pd.Series,
    x: pd.Series,
    significance_level: float = 0.05,
    trend: str = "c",
) -> CointegrationTestResult:
    """Run an Engle-Granger cointegration test between ``y`` and ``x``.

    Parameters
    ----------
    y:
        First price series.
    x:
        Second price series, sharing ``y``'s index.
    significance_level:
        Significance level used to determine
        :attr:`CointegrationTestResult.is_cointegrated`.
    trend:
        Deterministic trend term passed to
        ``statsmodels.tsa.stattools.coint`` (``"c"`` = constant only,
        the standard choice for price-level pairs).

    Returns
    -------
    CointegrationTestResult

    Raises
    ------
    CointegrationError
        If the inputs are invalid.
    """
    _validate_series_inputs({"y": y, "x": x})
    if not 0 < significance_level < 1:
        raise CointegrationError(
            f"significance_level must be in (0, 1), got {significance_level}"
        )

    test_statistic, p_value, critical_values = coint(
        y.to_numpy(dtype=float), x.to_numpy(dtype=float), trend=trend
    )
    critical_value_map = {
        "1%": float(critical_values[0]),
        "5%": float(critical_values[1]),
        "10%": float(critical_values[2]),
    }
    is_cointegrated = bool(p_value < significance_level)

    result = CointegrationTestResult(
        test_statistic=float(test_statistic),
        p_value=float(p_value),
        critical_values=critical_value_map,
        significance_level=significance_level,
        is_cointegrated=is_cointegrated,
    )

    if is_cointegrated:
        logger.info(
            "Cointegration test PASSED: statistic=%.4f, p_value=%.4g (< %.2f)",
            result.test_statistic,
            result.p_value,
            significance_level,
        )
    else:
        logger.warning(
            "Cointegration test FAILED: statistic=%.4f, p_value=%.4g (>= %.2f). "
            "Spread built on this pair is not statistically validated as mean-reverting.",
            result.test_statistic,
            result.p_value,
            significance_level,
        )
    return result


def compute_spread(
    y: pd.Series,
    x: pd.Series,
    hedge_ratio: pd.Series,
    lag: int = 1,
) -> pd.Series:
    """Compute the pair spread ``y - lag(hedge_ratio, lag) * x``.

    Parameters
    ----------
    y:
        Dependent leg's price series.
    x:
        Independent leg's price series, sharing ``y``'s index.
    hedge_ratio:
        Hedge ratio series (e.g. from
        :class:`stat_arb.signal.regression.RollingRegressionEngine`),
        sharing ``y``'s index.
    lag:
        Number of bars to shift ``hedge_ratio`` forward before applying
        it, so the hedge ratio used at bar *t* was estimated using no
        information from bar *t* itself (see module docstring for why
        this matters even for a trailing rolling regression). ``lag=0``
        disables this protection and logs a warning.

    Returns
    -------
    Series
        The spread, named ``"spread"``, sharing the input index. Bars
        without a (lagged) hedge ratio available are ``NaN``.

    Raises
    ------
    CointegrationError
        If the inputs are invalid.
    """
    _validate_series_inputs({"y": y, "x": x, "hedge_ratio": hedge_ratio})
    if lag < 0:
        raise CointegrationError(f"lag must be >= 0, got {lag}")
    if lag == 0:
        logger.warning(
            "compute_spread called with lag=0: the hedge ratio at bar t may have "
            "been estimated using bar t's own price, which can bias the spread "
            "toward apparent mean-reversion (in-sample shrinkage). Prefer lag>=1."
        )

    causal_hedge_ratio = hedge_ratio.shift(lag) if lag > 0 else hedge_ratio
    spread = y - causal_hedge_ratio * x
    spread.name = "spread"
    return spread


def rolling_mean(spread: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Trailing rolling mean of the spread (never uses future observations)."""
    result = spread.rolling(window, min_periods=min_periods or window, center=False).mean()
    result.name = "rolling_mean"
    return result


def rolling_std(spread: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Trailing rolling standard deviation of the spread (sample, ddof=1)."""
    result = spread.rolling(window, min_periods=min_periods or window, center=False).std(ddof=1)
    result.name = "rolling_std"
    return result


def rolling_zscore(
    spread: pd.Series,
    window: int,
    min_periods: int | None = None,
    std_floor: float = 1e-8,
) -> pd.Series:
    """Trailing rolling z-score of the spread.

    ``z_t = (spread_t - rolling_mean_t) / rolling_std_t``, computed
    entirely from trailing windows (``rolling(..., center=False)``), so
    ``z_t`` depends only on data available at or before bar *t*.

    Parameters
    ----------
    spread:
        Spread series, e.g. from :func:`compute_spread`.
    window:
        Rolling window size in bars.
    min_periods:
        Minimum observations required within a window. Defaults to
        ``window``.
    std_floor:
        Rolling standard deviations at or below this are treated as
        numerically degenerate; the z-score there is ``NaN`` instead of
        an inflated or infinite value.

    Returns
    -------
    Series
        The rolling z-score, named ``"zscore"``.
    """
    mean = rolling_mean(spread, window, min_periods)
    std = rolling_std(spread, window, min_periods)

    degenerate = std.notna() & (std <= std_floor)
    n_degenerate = int(degenerate.sum())
    if n_degenerate:
        logger.warning(
            "%d/%d rolling window(s) have standard deviation <= %.1e "
            "(degenerate/near-constant spread); z-score set to NaN there",
            n_degenerate,
            int(std.notna().sum()),
            std_floor,
        )

    safe_std = std.where(~degenerate)
    zscore = (spread - mean) / safe_std
    zscore.name = "zscore"
    return zscore


class CointegrationEngine:
    """Orchestrates cointegration validation and spread/z-score computation.

    Usage::

        engine = CointegrationEngine(CointegrationConfig(spread_window=60))
        result = engine.compute(y=btc_close, x=eth_close, hedge_ratio=hedge_ratio)
        result.zscore  # rolling z-score, safe to use as a signal input
    """

    def __init__(self, config: CointegrationConfig | None = None) -> None:
        """Initialize the engine.

        Parameters
        ----------
        config:
            Configuration. Defaults to ``CointegrationConfig()`` if not
            provided.
        """
        self.config = config or CointegrationConfig()

    def compute(self, y: pd.Series, x: pd.Series, hedge_ratio: pd.Series) -> SpreadResult:
        """Validate cointegration and compute the spread and its rolling z-score.

        Parameters
        ----------
        y:
            Dependent leg's price series.
        x:
            Independent leg's price series, sharing ``y``'s index.
        hedge_ratio:
            Hedge ratio series, sharing ``y``'s index.

        Returns
        -------
        SpreadResult

        Raises
        ------
        CointegrationError
            If the inputs are invalid, or the pair fails the
            cointegration test and
            :attr:`CointegrationConfig.require_cointegration` is ``True``.
        """
        config = self.config
        _validate_series_inputs({"y": y, "x": x, "hedge_ratio": hedge_ratio})

        cointegration = test_cointegration(y, x, significance_level=config.significance_level)
        if not cointegration.is_cointegrated and config.require_cointegration:
            raise CointegrationError(
                f"Pair failed cointegration validation: p_value={cointegration.p_value:.4g} "
                f">= significance_level={config.significance_level}. "
                "Set require_cointegration=False to compute the spread anyway."
            )

        min_periods = config._resolved_min_periods
        spread = compute_spread(y, x, hedge_ratio, lag=config.hedge_ratio_lag)
        mean = rolling_mean(spread, config.spread_window, min_periods)
        std = rolling_std(spread, config.spread_window, min_periods)
        zscore = rolling_zscore(
            spread, config.spread_window, min_periods, config.zscore_std_floor
        )

        logger.info(
            "Spread computed: n=%d, window=%d, hedge_ratio_lag=%d, "
            "%d/%d bars have a usable z-score",
            len(spread),
            config.spread_window,
            config.hedge_ratio_lag,
            int(zscore.notna().sum()),
            len(zscore),
        )

        return SpreadResult(
            spread=spread,
            rolling_mean=mean,
            rolling_std=std,
            zscore=zscore,
            cointegration=cointegration,
            config=config,
        )
