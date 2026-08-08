"""Unit tests for hermes.logging_config."""

from __future__ import annotations

import json
import logging

import pytest

from hermes.logging_config import LoggingConfig, configure_logging, get_logger


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """Restore the root logger's handlers after each test."""
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    yield
    root.handlers = original_handlers
    root.setLevel(original_level)


def test_config_rejects_invalid_level(tmp_path) -> None:
    with pytest.raises(ValueError, match="Invalid logging level"):
        LoggingConfig(level="NOT_A_LEVEL")


def test_configure_logging_writes_json_file(tmp_path) -> None:
    log_file = tmp_path / "hermes.log"
    configure_logging(LoggingConfig(json_log_file=log_file, console=False, level="INFO"))

    logger = get_logger("hermes.test.json")
    logger.info("hello", extra={"pair": "BTC/USDC:USDC"})
    for handler in logging.getLogger().handlers:
        handler.flush()

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["message"] == "hello"
    assert record["pair"] == "BTC/USDC:USDC"
    assert record["level"] == "INFO"


def test_configure_logging_creates_parent_directories(tmp_path) -> None:
    log_file = tmp_path / "nested" / "dir" / "hermes.log"
    configure_logging(LoggingConfig(json_log_file=log_file, console=False))
    assert log_file.parent.is_dir()


def test_configure_logging_is_idempotent(tmp_path) -> None:
    log_file = tmp_path / "hermes.log"
    configure_logging(LoggingConfig(json_log_file=log_file, console=True))
    handler_count_first = len(logging.getLogger().handlers)

    configure_logging(LoggingConfig(json_log_file=log_file, console=True))
    handler_count_second = len(logging.getLogger().handlers)

    assert handler_count_first == handler_count_second


def test_configure_logging_respects_level(tmp_path) -> None:
    log_file = tmp_path / "hermes.log"
    configure_logging(LoggingConfig(json_log_file=log_file, console=False, level="WARNING"))

    logger = get_logger("hermes.test.level")
    logger.info("should not appear")
    logger.warning("should appear")
    for handler in logging.getLogger().handlers:
        handler.flush()

    content = log_file.read_text(encoding="utf-8")
    assert "should not appear" not in content
    assert "should appear" in content


def test_configure_logging_no_channels_uses_null_handler(tmp_path) -> None:
    root = configure_logging(LoggingConfig(console=False, json_log_file=None))
    hermes_handlers = [h for h in root.handlers if getattr(h, "_hermes_managed", False)]
    assert any(isinstance(h, logging.NullHandler) for h in hermes_handlers)


def test_get_logger_returns_named_logger() -> None:
    logger = get_logger("hermes.foo")
    assert logger.name == "hermes.foo"
