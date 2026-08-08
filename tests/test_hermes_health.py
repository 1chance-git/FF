"""Unit tests for hermes.health.

Uses plain stub clients (satisfying SupportsFreqtradeApi by duck typing)
rather than a real FtRestClient/HTTP server, per the module's
dependency-injection design.
"""

from __future__ import annotations

from typing import Any

import pytest

from hermes.health import HealthChecker, HealthStatus


class StubClient:
    """A configurable stub satisfying hermes.health.SupportsFreqtradeApi."""

    def __init__(
        self,
        ping_response: dict[str, Any] | None = None,
        health_response: dict[str, Any] | None = None,
        version_response: dict[str, Any] | None = None,
        sysinfo_response: dict[str, Any] | None = None,
        raise_on: str | None = None,
    ) -> None:
        self.ping_response = ping_response or {"status": "pong"}
        self.health_response = health_response or {}
        self.version_response = version_response or {"version": "2026.7"}
        self.sysinfo_response = sysinfo_response or {"cpu_pct": [10.0], "ram_pct": 20.0}
        self.raise_on = raise_on

    def ping(self):
        if self.raise_on == "ping":
            raise ConnectionError("connection refused")
        return self.ping_response

    def health(self):
        if self.raise_on == "health":
            raise ConnectionError("connection refused")
        return self.health_response

    def version(self):
        if self.raise_on == "version":
            raise ConnectionError("connection refused")
        return self.version_response

    def sysinfo(self):
        if self.raise_on == "sysinfo":
            raise ConnectionError("connection refused")
        return self.sysinfo_response


def test_all_checks_healthy() -> None:
    report = HealthChecker(StubClient()).run()
    assert report.status is HealthStatus.HEALTHY
    assert report.is_healthy is True
    assert report.failed_checks() == ()
    assert {c.name for c in report.checks} == {"api_reachable", "bot_health", "resources"}


def test_unreachable_api_is_unhealthy() -> None:
    report = HealthChecker(StubClient(raise_on="ping")).run()
    assert report.status is HealthStatus.UNHEALTHY
    assert report.is_healthy is False
    failed_names = {c.name for c in report.failed_checks()}
    assert "api_reachable" in failed_names


def test_ping_not_running_is_unhealthy() -> None:
    client = StubClient(ping_response={"status": "not_running"})
    report = HealthChecker(client).run()
    api_check = next(c for c in report.checks if c.name == "api_reachable")
    assert api_check.status is HealthStatus.UNHEALTHY


def test_bot_health_check_failure_marks_unhealthy() -> None:
    report = HealthChecker(StubClient(raise_on="health")).run()
    bot_health_check = next(c for c in report.checks if c.name == "bot_health")
    assert bot_health_check.status is HealthStatus.UNHEALTHY
    assert report.status is HealthStatus.UNHEALTHY


def test_high_resource_usage_is_degraded_not_unhealthy() -> None:
    client = StubClient(sysinfo_response={"cpu_pct": [95.0], "ram_pct": 40.0})
    report = HealthChecker(client, resource_warning_pct=85.0).run()

    resource_check = next(c for c in report.checks if c.name == "resources")
    assert resource_check.status is HealthStatus.DEGRADED
    assert report.status is HealthStatus.DEGRADED  # degraded, not unhealthy overall
    assert report.is_healthy is False


def test_high_ram_usage_is_degraded() -> None:
    client = StubClient(sysinfo_response={"cpu_pct": [10.0], "ram_pct": 99.0})
    report = HealthChecker(client, resource_warning_pct=85.0).run()
    resource_check = next(c for c in report.checks if c.name == "resources")
    assert resource_check.status is HealthStatus.DEGRADED


def test_unhealthy_takes_priority_over_degraded() -> None:
    client = StubClient(raise_on="ping", sysinfo_response={"cpu_pct": [95.0], "ram_pct": 10.0})
    report = HealthChecker(client, resource_warning_pct=85.0).run()
    assert report.status is HealthStatus.UNHEALTHY


def test_report_includes_duration() -> None:
    report = HealthChecker(StubClient()).run()
    assert report.duration_seconds >= 0


def test_check_result_details_carry_raw_response() -> None:
    client = StubClient(version_response={"version": "2026.7", "extra": "field"})
    report = HealthChecker(client).run()
    bot_health_check = next(c for c in report.checks if c.name == "bot_health")
    assert bot_health_check.details["version"]["extra"] == "field"


def test_never_raises_even_if_all_checks_fail() -> None:
    client = StubClient(raise_on="ping")

    class AllFailClient(StubClient):
        def health(self):
            raise TimeoutError("timeout")

        def sysinfo(self):
            raise TimeoutError("timeout")

    report = HealthChecker(AllFailClient(raise_on="ping")).run()
    assert report.status is HealthStatus.UNHEALTHY
    assert len(report.failed_checks()) == 3
