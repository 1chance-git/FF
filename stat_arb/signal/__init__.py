"""Statistical signal generation building blocks (hedge ratios, spreads, z-scores, ...)."""

from stat_arb.signal.regression import (
    RegressionError,
    RollingRegressionConfig,
    RollingRegressionEngine,
    RollingRegressionResult,
)

__all__ = [
    "RegressionError",
    "RollingRegressionConfig",
    "RollingRegressionEngine",
    "RollingRegressionResult",
]
