"""Hermes: operational tooling for running and monitoring the stat-arb bot.

Structured logging, health checks against a running bot's REST API, a
backtest launcher, process lifecycle/restart support, and a CLI tying
them together — all independent of (built on top of, never inside)
Freqtrade's own internals. See `hermes/cli.py` for the command surface
and each submodule's docstring for its own design rationale.
"""

from hermes.backtest import BacktestConfig, BacktestError, BacktestLauncher, BacktestResult
from hermes.health import CheckResult, HealthChecker, HealthReport, HealthStatus
from hermes.logging_config import LoggingConfig, configure_logging, get_logger
from hermes.process import (
    BotProcessManager,
    ProcessConfig,
    ProcessError,
    RestartSupervisor,
    RestartSupervisorConfig,
)

__all__ = [
    "BacktestConfig",
    "BacktestError",
    "BacktestLauncher",
    "BacktestResult",
    "BotProcessManager",
    "CheckResult",
    "HealthChecker",
    "HealthReport",
    "HealthStatus",
    "LoggingConfig",
    "ProcessConfig",
    "ProcessError",
    "RestartSupervisor",
    "RestartSupervisorConfig",
    "configure_logging",
    "get_logger",
]
