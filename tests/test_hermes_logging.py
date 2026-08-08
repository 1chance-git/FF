"""Unit tests for hermes.logging_config."""

from __future__ import annotations

import json
import logging
import logging.handlers

import pytest

pytestmark = pytest.mark.unit

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


def test_config_rejects_non_positive_max_bytes() -> None:
    with pytest.raises(ValueError, match="json_log_max_bytes must be positive"):
        LoggingConfig(json_log_max_bytes=0)


def test_config_rejects_negative_backup_count() -> None:
    with pytest.raises(ValueError, match="json_log_backup_count must be non-negative"):
        LoggingConfig(json_log_backup_count=-1)


def test_json_log_file_handler_rotates_instead_of_growing_unbounded(tmp_path) -> None:
    """The JSON log file must use a RotatingFileHandler, not a plain FileHandler.

    A long-lived operational tool writing an unbounded log file is a
    real disk-usage risk; this pins the handler type so a future change
    can't silently regress back to unbounded growth.
    """
    log_file = tmp_path / "hermes.log"
    configure_logging(
        LoggingConfig(
            json_log_file=log_file,
            console=False,
            json_log_max_bytes=1024,
            json_log_backup_count=3,
        )
    )

    handlers = [
        h
        for h in logging.getLogger().handlers
        if isinstance(h, logging.handlers.RotatingFileHandler)
    ]
    assert len(handlers) == 1
    assert handlers[0].maxBytes == 1024
    assert handlers[0].backupCount == 3


def test_json_log_file_actually_rotates_when_size_limit_exceeded(tmp_path) -> None:
    log_file = tmp_path / "hermes.log"
    configure_logging(
        LoggingConfig(
            json_log_file=log_file,
            console=False,
            json_log_max_bytes=200,
            json_log_backup_count=2,
        )
    )

    logger = get_logger("hermes.test.rotation")
    for i in range(50):
        logger.info("a fairly long log message to fill up the rotation limit quickly %d", i)
    for handler in logging.getLogger().handlers:
        handler.flush()

    rotated_file = tmp_path / "hermes.log.1"
    assert rotated_file.exists()
