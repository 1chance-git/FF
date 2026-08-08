"""Health checks for a running Freqtrade bot process.

Queries the bot's REST API via `freqtrade_client.FtRestClient` — the
official, mature client library shipped by the Freqtrade project itself
— rather than hand-rolling HTTP calls against the API. This keeps Hermes
correctly in sync with whatever the API server actually exposes across
Freqtrade versions, instead of duplicating and inevitably drifting from
that contract.

Design decisions
-----------------
* **The REST client is injected, not constructed internally.**
  `HealthChecker` takes any object exposing the small subset of
  `FtRestClient`'s interface it needs (`ping`, `health`, `version`,
  `sysinfo`). This is the same dependency-injection pattern used
  throughout this project (e.g. `stat_arb.data.market_data.MarketDataLoader`
  takes a datadir rather than reaching for a global) — it makes every
  check testable with a plain stub, with no real HTTP server required.
* **A single unreachable API is a health result, not an exception.** A
  bot that isn't running (or whose API server is disabled/unreachable)
  is an entirely expected, common state — during startup, a planned
  restart, or simply because the operator hasn't started it yet. Health
  checks report that as `HealthStatus.UNHEALTHY` with a clear reason,
  never raise, so a caller can always get a report to act on rather than
  having to wrap every check in a try/except.
* **Checks are independent and individually reported.** A `HealthReport`
  aggregates a list of `CheckResult`s rather than collapsing straight to
  a single boolean, so a caller (or the CLI) can see *which* check
  failed and why, not just that something is wrong.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Outcome of an individual health check or the aggregate report."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class SupportsFreqtradeApi(Protocol):
    """The subset of `FtRestClient`'s interface :class:`HealthChecker` needs.

    Any object satisfying this (the real `FtRestClient`, or a test
    stub) can be passed to `HealthChecker`.
    """

    def ping(self) -> dict[str, Any]: ...

    def health(self) -> dict[str, Any]: ...

    def version(self) -> dict[str, Any]: ...

    def sysinfo(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single health check.

    Attributes
    ----------
    name:
        Short identifier for the check, e.g. ``"api_reachable"``.
    status:
        Outcome of this check.
    message:
        Human-readable summary.
    details:
        Any structured data backing the result (e.g. the raw API
        response), for logging/debugging.
    """

    name: str
    status: HealthStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HealthReport:
    """Aggregate result of running all configured health checks.

    Attributes
    ----------
    status:
        Overall status: :attr:`HealthStatus.UNHEALTHY` if any check is
        unhealthy, else :attr:`HealthStatus.DEGRADED` if any check is
        degraded, else :attr:`HealthStatus.HEALTHY`.
    checks:
        Individual check results, in the order they were run.
    duration_seconds:
        Wall-clock time taken to run all checks.
    """

    status: HealthStatus
    checks: tuple[CheckResult, ...]
    duration_seconds: float

    @property
    def is_healthy(self) -> bool:
        """``True`` only if every check reported healthy."""
        return self.status is HealthStatus.HEALTHY

    def failed_checks(self) -> tuple[CheckResult, ...]:
        """Checks that did not report healthy, in run order."""
        return tuple(c for c in self.checks if c.status is not HealthStatus.HEALTHY)


#: Sysinfo CPU/RAM usage percentages at or above this are DEGRADED, not
#: HEALTHY, since the bot is still running but under load worth noticing.
DEFAULT_RESOURCE_WARNING_PCT = 85.0


class HealthChecker:
    """Runs a fixed set of health checks against a Freqtrade bot's REST API.

    Usage::

        client = FtRestClient("http://127.0.0.1:8080", "user", "pass")
        checker = HealthChecker(client)
        report = checker.run()
        if not report.is_healthy:
            for check in report.failed_checks():
                print(check.name, check.message)
    """

    def __init__(
        self,
        client: SupportsFreqtradeApi,
        resource_warning_pct: float = DEFAULT_RESOURCE_WARNING_PCT,
    ) -> None:
        """Initialize the checker.

        Parameters
        ----------
        client:
            A `FtRestClient`-compatible object (see
            :class:`SupportsFreqtradeApi`).
        resource_warning_pct:
            CPU/RAM usage percentage at or above which the ``resources``
            check is reported :attr:`HealthStatus.DEGRADED` instead of
            healthy.
        """
        self.client = client
        self.resource_warning_pct = resource_warning_pct

    def run(self) -> HealthReport:
        """Run all health checks and return the aggregate report.

        Never raises: any exception from an individual check (e.g. a
        connection error) is caught and turned into an
        :attr:`HealthStatus.UNHEALTHY` result for that check.
        """
        start = time.monotonic()
        checks = (
            self._check_api_reachable(),
            self._check_bot_health(),
            self._check_resources(),
        )
        duration = time.monotonic() - start

        if any(c.status is HealthStatus.UNHEALTHY for c in checks):
            status = HealthStatus.UNHEALTHY
        elif any(c.status is HealthStatus.DEGRADED for c in checks):
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY

        report = HealthReport(status=status, checks=checks, duration_seconds=duration)
        logger.info(
            "Health check complete: status=%s, duration=%.3fs, failed=%s",
            status.value,
            duration,
            [c.name for c in report.failed_checks()],
        )
        return report

    def _check_api_reachable(self) -> CheckResult:
        try:
            response = self.client.ping()
        except Exception as exc:  # noqa: BLE001 - any client failure is a health result
            return CheckResult(
                name="api_reachable",
                status=HealthStatus.UNHEALTHY,
                message=f"Could not reach bot API: {exc}",
            )

        if response.get("status") == "pong":
            return CheckResult(
                name="api_reachable",
                status=HealthStatus.HEALTHY,
                message="Bot API reachable and running",
                details=response,
            )
        return CheckResult(
            name="api_reachable",
            status=HealthStatus.UNHEALTHY,
            message=f"Bot API reachable but not running (state: {response})",
            details=response,
        )

    def _check_bot_health(self) -> CheckResult:
        try:
            health_response = self.client.health()
            version_response = self.client.version()
        except Exception as exc:  # noqa: BLE001
            return CheckResult(
                name="bot_health",
                status=HealthStatus.UNHEALTHY,
                message=f"Could not query bot health/version: {exc}",
            )

        details = {"health": health_response, "version": version_response}
        return CheckResult(
            name="bot_health",
            status=HealthStatus.HEALTHY,
            message=f"Bot version {version_response.get('version', 'unknown')} responding",
            details=details,
        )

    def _check_resources(self) -> CheckResult:
        try:
            sysinfo = self.client.sysinfo()
        except Exception as exc:  # noqa: BLE001
            return CheckResult(
                name="resources",
                status=HealthStatus.UNHEALTHY,
                message=f"Could not query system resources: {exc}",
            )

        cpu_pct = sysinfo.get("cpu_pct")
        ram_pct = sysinfo.get("ram_pct")
        cpu_values = cpu_pct if isinstance(cpu_pct, list) else [cpu_pct] if cpu_pct else []
        worst_cpu = max(cpu_values, default=0.0)

        if worst_cpu >= self.resource_warning_pct or (ram_pct or 0) >= self.resource_warning_pct:
            return CheckResult(
                name="resources",
                status=HealthStatus.DEGRADED,
                message=f"High resource usage: cpu={worst_cpu}%, ram={ram_pct}%",
                details=sysinfo,
            )
        return CheckResult(
            name="resources",
            status=HealthStatus.HEALTHY,
            message=f"Resource usage nominal: cpu={worst_cpu}%, ram={ram_pct}%",
            details=sysinfo,
        )
