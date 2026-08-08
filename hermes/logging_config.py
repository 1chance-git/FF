"""Structured logging setup for Hermes and the processes it manages.

Provides two independent output channels, since they serve different
audiences:

* **Structured (JSON) logs** — one JSON object per line, suitable for
  shipping to a log aggregator (ELK, Loki, CloudWatch, ...) or `jq`-based
  ad-hoc querying. Built on `python-json-logger`
  (`pythonjsonlogger.json.JsonFormatter`), a small, mature, widely used
  library that already handles the fiddly parts correctly (exception
  formatting, non-serializable extra fields, timestamp formatting) —
  reimplementing a JSON `logging.Formatter` by hand would just recreate
  bugs that library has already fixed.
* **CLI-friendly console output** — human-readable, colorized terminal
  output for interactive use, built on `rich`
  (`rich.logging.RichHandler`), a mature library purpose-built for this.

Both channels attach to the same standard-library `logging` hierarchy,
so any code that does ordinary `logging.getLogger(__name__)` calls
automatically gets both without changing a line — Hermes' own modules,
and (via Freqtrade's own `log_config` dictConfig support) the trading
bot process it launches, can share this configuration.

Design decisions
-----------------
* **Two formatters, one hierarchy**, rather than one "pick a format" flag.
  Structured logs and human console output are consumed by different
  things (machines vs. a person watching a terminal) and optimizing a
  single format for both inevitably makes it worse at each. Attaching
  both handlers to the same root logger costs nothing extra and lets a
  caller enable either, both, or neither independently.
* **JSON logs go to a file by default, console output to the terminal.**
  This mirrors how the trading bot is already configured (see
  `user_data/config.json`'s `log_config`) and avoids interleaving
  machine-oriented JSON with the human-oriented console during
  interactive `hermes` CLI use.
* **`configure_logging` is idempotent.** Calling it more than once (e.g.
  once from `hermes.cli` and again from a test) removes any handlers it
  previously installed before adding new ones, rather than stacking
  duplicate handlers that would double-log everything.
* **The JSON log file rotates.** A plain, non-rotating file handler
  would grow without bound for a long-lived operational tool — the same
  reason the trading bot's own `log_config` (see
  `user_data/config.json`) uses a `RotatingFileHandler` rather than a
  bare one. Hermes matches that convention (`RotatingFileHandler`,
  configurable max size/backup count) instead of introducing a second,
  inconsistent logging behavior for the same kind of file.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from dataclasses import dataclass
from pathlib import Path

from pythonjsonlogger.json import JsonFormatter
from rich.logging import RichHandler

#: Marker attribute set on handlers installed by configure_logging, so a
#: repeat call can find and remove exactly its own handlers (and no
#: others a caller may have added independently).
_HERMES_HANDLER_MARKER = "_hermes_managed"

_JSON_LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"


@dataclass(frozen=True)
class LoggingConfig:
    """Configuration for :func:`configure_logging`.

    Parameters
    ----------
    level:
        Root logger level, e.g. ``"INFO"``, ``"DEBUG"``.
    json_log_file:
        Path to write structured JSON logs to. ``None`` disables the
        JSON channel entirely.
    console:
        If ``True`` (default), attach a `rich`-based human-readable
        handler to the terminal (stderr).
    console_show_path:
        Whether the console handler shows the source file/line for each
        log record. Useful for development, noisy for routine
        operational use.
    json_log_max_bytes:
        Rotate the JSON log file once it reaches this size. Matches the
        trading bot's own `log_config` default (10 MB).
    json_log_backup_count:
        Number of rotated JSON log files to retain.
    """

    level: str = "INFO"
    json_log_file: Path | None = None
    console: bool = True
    console_show_path: bool = False
    json_log_max_bytes: int = 10 * 1024 * 1024
    json_log_backup_count: int = 10

    def __post_init__(self) -> None:
        valid_levels = logging.getLevelNamesMapping() if hasattr(
            logging, "getLevelNamesMapping"
        ) else {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.level.upper() not in valid_levels:
            raise ValueError(f"Invalid logging level: {self.level!r}")
        if self.json_log_max_bytes <= 0:
            raise ValueError("json_log_max_bytes must be positive")
        if self.json_log_backup_count < 0:
            raise ValueError("json_log_backup_count must be non-negative")


def configure_logging(config: LoggingConfig | None = None) -> logging.Logger:
    """Configure the root logger with structured JSON and/or console output.

    Idempotent: any handlers previously installed by this function are
    removed before new ones are added, so calling it again (e.g. to
    change the log level) does not double-log.

    Parameters
    ----------
    config:
        Logging configuration. Defaults to ``LoggingConfig()`` (INFO
        level, console only, no JSON file).

    Returns
    -------
    logging.Logger
        The configured root logger.
    """
    config = config or LoggingConfig()
    root_logger = logging.getLogger()
    root_logger.setLevel(config.level.upper())

    for handler in list(root_logger.handlers):
        if getattr(handler, _HERMES_HANDLER_MARKER, False):
            root_logger.removeHandler(handler)

    if config.json_log_file is not None:
        config.json_log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            config.json_log_file,
            maxBytes=config.json_log_max_bytes,
            backupCount=config.json_log_backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(
            JsonFormatter(
                _JSON_LOG_FORMAT, rename_fields={"asctime": "timestamp", "levelname": "level"}
            )
        )
        setattr(file_handler, _HERMES_HANDLER_MARKER, True)
        root_logger.addHandler(file_handler)

    if config.console:
        console_handler = RichHandler(
            show_path=config.console_show_path,
            rich_tracebacks=True,
            markup=False,
        )
        console_handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        setattr(console_handler, _HERMES_HANDLER_MARKER, True)
        root_logger.addHandler(console_handler)

    if not config.json_log_file and not config.console:
        null_handler = logging.NullHandler()
        setattr(null_handler, _HERMES_HANDLER_MARKER, True)
        root_logger.addHandler(null_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Return a module logger. Thin convenience wrapper around ``logging.getLogger``."""
    return logging.getLogger(name)
