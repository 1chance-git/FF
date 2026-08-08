"""Statistical signal generation building blocks (hedge ratios, spreads, z-scores, ...)."""

from stat_arb.signal.cointegration import (
    CointegrationConfig,
    CointegrationEngine,
    CointegrationError,
    CointegrationTestResult,
    SpreadResult,
    compute_spread,
    rolling_mean,
    rolling_std,
    rolling_zscore,
    test_cointegration,
)
from stat_arb.signal.regression import (
    RegressionError,
    RollingRegressionConfig,
    RollingRegressionEngine,
    RollingRegressionResult,
)

__all__ = [
    "CointegrationConfig",
    "CointegrationEngine",
    "CointegrationError",
    "CointegrationTestResult",
    "RegressionError",
    "RollingRegressionConfig",
    "RollingRegressionEngine",
    "RollingRegressionResult",
    "SpreadResult",
    "compute_spread",
    "rolling_mean",
    "rolling_std",
    "rolling_zscore",
    "test_cointegration",
]
