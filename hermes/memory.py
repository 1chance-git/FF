"""Persistent, append-only history of trading and operational events, in SQLite.

Hermes' memory of what happened: trades and their P&L/fees/funding/
signal context, bot start/stop/crash events, important errors, and
backtest results. This module only remembers — it has no opinion about
whether any of it was good or bad (that's `hermes.analyzer`), and it
never mutates strategy behavior.

Design decisions
-----------------
* **Plain `sqlite3` from the standard library, no ORM.** A handful of
  narrow, append-only tables don't need an ORM's abstraction; a schema
  this small is more legible as literal SQL than as a stack of model
  classes, and it keeps this module dependency-free.
* **Append-only: `INSERT` only, no `UPDATE`/`DELETE` in the public
  API.** Every table has an autoincrement `id` and a `recorded_at`
  timestamp filled in automatically if not supplied. History is a log,
  not a mutable record — "never silently overwrite history" is enforced
  by the API surface itself, not by convention.
* **Every write function catches its own exceptions and returns
  `bool`, never raises.** A future integration point (the strategy or
  the risk engine recording a trade's context) will call these
  functions from inside Freqtrade's own process. A locked database, a
  full disk, or a schema surprise must degrade to a logged failure and
  a `False` return, never an exception that could interrupt trading.
  Schema initialization (`MemoryStore.__init__`) is the one exception:
  it runs once, outside any trading hot path, and a store that can't
  create its own tables should fail loudly and immediately rather than
  pretend to work.
* **Optional fields default to `None`, not a sentinel or a required
  join.** Funding isn't available on every exchange, z-scores aren't
  known for every event, hedge ratio doesn't apply to non-trade events
  — every record dataclass makes these fields optional so a caller
  supplies what it has and nothing it doesn't.
* **`extra`/`metrics` are a JSON blob column, not more columns.**
  Backtest metrics in particular vary by Freqtrade version and command;
  rather than chase that with schema migrations, `BacktestResult.metrics`
  stores the caller's dict as-is. `TradeRecord.extra` gives the same
  escape hatch for any per-trade detail that doesn't warrant its own
  column.
* **WAL journal mode.** Freqtrade (writing trade events, potentially)
  and an analysis/CLI process (reading) can otherwise contend for
  SQLite's single-writer lock; WAL lets readers proceed without
  blocking on a writer, and vice versa, for the common case here.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MemoryStoreError(Exception):
    """Raised for unrecoverable memory-store failures (schema init only)."""


# -- Where Hermes memory lives -----------------------------------------------
#
# One source of truth, shared by every writer and reader of this database
# (hermes.cli's backtest/analyze/backtest-report commands and
# hermes.supervisor), so they can never silently resolve to different files
# — which is exactly the failure mode that made a completed backtest's
# recorded result unreadable on Railway.
#
# The directory is deliberately *configurable* rather than hard-coded:
# deployments whose persistent storage is mounted somewhere other than the
# repo-relative default (e.g. Railway, whose Volume mounts at
# `/app/user_data/data`) point HERMES_USER_DATA_DIR at that mount instead of
# passing --user-data-dir to every single command. No deployment-specific
# path appears anywhere in Hermes' own logic.

MEMORY_DB_FILENAME = "hermes_memory.sqlite3"
USER_DATA_DIR_ENV_VAR = "HERMES_USER_DATA_DIR"
DEFAULT_USER_DATA_DIR = Path("user_data")


def default_user_data_dir() -> Path:
    """The user-data directory to use when a caller didn't name one.

    Reads :data:`USER_DATA_DIR_ENV_VAR` at call time (not import time), so
    tests and processes can set it per-invocation, falling back to
    :data:`DEFAULT_USER_DATA_DIR` — which keeps local development and the
    existing test suite on exactly the behavior they had before.
    """
    from os import environ

    configured = environ.get(USER_DATA_DIR_ENV_VAR)
    return Path(configured) if configured else DEFAULT_USER_DATA_DIR


def memory_db_path(user_data_dir: Path | str | None = None) -> Path:
    """Resolve the Hermes memory database path.

    Passing ``user_data_dir`` (e.g. from a ``--user-data-dir`` flag) wins;
    passing ``None`` defers to :func:`default_user_data_dir`.
    """
    base = Path(user_data_dir) if user_data_dir is not None else default_user_data_dir()
    return base / MEMORY_DB_FILENAME


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at TEXT NOT NULL,
    pair TEXT NOT NULL,
    side TEXT,
    entry_time TEXT,
    exit_time TEXT,
    entry_price REAL,
    exit_price REAL,
    pnl REAL,
    pnl_pct REAL,
    fees REAL,
    funding REAL,
    entry_zscore REAL,
    exit_zscore REAL,
    hedge_ratio REAL,
    holding_time_seconds REAL,
    exit_reason TEXT,
    regime TEXT,
    extra TEXT
);

CREATE TABLE IF NOT EXISTS process_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at TEXT NOT NULL,
    event_type TEXT NOT NULL,
    pid INTEGER,
    message TEXT
);

CREATE TABLE IF NOT EXISTS error_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at TEXT NOT NULL,
    source TEXT NOT NULL,
    message TEXT NOT NULL,
    severity TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS backtest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at TEXT NOT NULL,
    strategy TEXT NOT NULL,
    timerange TEXT,
    metrics TEXT NOT NULL
);
"""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _parse_iso(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


@dataclass(frozen=True)
class TradeRecord:
    """A single completed (or in-progress) pairs trade and its context.

    All fields beyond `pair` are optional: a trade can be logged at
    entry (only entry_* known) and, separately, at exit — callers
    decide whether to record one row per lifecycle event or one row
    per completed trade; this module doesn't enforce either.
    """

    pair: str
    side: str | None = None
    entry_time: datetime | None = None
    exit_time: datetime | None = None
    entry_price: float | None = None
    exit_price: float | None = None
    pnl: float | None = None
    pnl_pct: float | None = None
    fees: float | None = None
    funding: float | None = None
    entry_zscore: float | None = None
    exit_zscore: float | None = None
    hedge_ratio: float | None = None
    holding_time_seconds: float | None = None
    exit_reason: str | None = None
    regime: str | None = None
    extra: dict[str, Any] | None = None
    recorded_at: datetime | None = None
    id: int | None = field(default=None, compare=False)


@dataclass(frozen=True)
class ProcessEvent:
    """A bot lifecycle event: start, stop, or crash."""

    event_type: str
    pid: int | None = None
    message: str | None = None
    recorded_at: datetime | None = None
    id: int | None = field(default=None, compare=False)


@dataclass(frozen=True)
class ErrorEvent:
    """A notable error worth remembering, independent of any single trade."""

    source: str
    message: str
    severity: str = "error"
    recorded_at: datetime | None = None
    id: int | None = field(default=None, compare=False)


@dataclass(frozen=True)
class BacktestResult:
    """The outcome of a single backtest run."""

    strategy: str
    metrics: dict[str, Any]
    timerange: str | None = None
    recorded_at: datetime | None = None
    id: int | None = field(default=None, compare=False)


class MemoryStore:
    """Hermes' append-only SQLite history of trades and operational events.

    Usage::

        store = MemoryStore(Path("user_data/hermes_memory.sqlite3"))
        store.record_trade(TradeRecord(pair="BTC/USDC:USDC", pnl=12.3, ...))
        recent = store.get_trades(pair="BTC/USDC:USDC", limit=50)
    """

    def __init__(self, db_path: Path) -> None:
        """Open (creating if needed) the store at `db_path`.

        Raises
        ------
        MemoryStoreError
            If the schema cannot be created. This is the one place this
            class raises rather than logging and returning `False` —
            it happens once, at startup, never from inside a live
            trading call.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._connect() as conn:
                conn.executescript(_SCHEMA_SQL)
        except sqlite3.Error as exc:
            raise MemoryStoreError(
                f"Failed to initialize Hermes memory schema at {self.db_path}: {exc}"
            ) from exc

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    # -- writes: never raise, always safe to call from the trading path --

    def record_trade(self, record: TradeRecord) -> bool:
        """Append a trade record. Returns `False` (and logs) on any failure."""
        return self._safe_insert(
            "trades",
            {
                "recorded_at": _iso(record.recorded_at or _utc_now()),
                "pair": record.pair,
                "side": record.side,
                "entry_time": _iso(record.entry_time),
                "exit_time": _iso(record.exit_time),
                "entry_price": record.entry_price,
                "exit_price": record.exit_price,
                "pnl": record.pnl,
                "pnl_pct": record.pnl_pct,
                "fees": record.fees,
                "funding": record.funding,
                "entry_zscore": record.entry_zscore,
                "exit_zscore": record.exit_zscore,
                "hedge_ratio": record.hedge_ratio,
                "holding_time_seconds": record.holding_time_seconds,
                "exit_reason": record.exit_reason,
                "regime": record.regime,
                "extra": json.dumps(record.extra) if record.extra is not None else None,
            },
        )

    def record_process_event(self, event: ProcessEvent) -> bool:
        """Append a bot start/stop/crash event. Returns `False` (and logs) on failure."""
        return self._safe_insert(
            "process_events",
            {
                "recorded_at": _iso(event.recorded_at or _utc_now()),
                "event_type": event.event_type,
                "pid": event.pid,
                "message": event.message,
            },
        )

    def record_error(self, error: ErrorEvent) -> bool:
        """Append a notable error. Returns `False` (and logs) on failure."""
        return self._safe_insert(
            "error_events",
            {
                "recorded_at": _iso(error.recorded_at or _utc_now()),
                "source": error.source,
                "message": error.message,
                "severity": error.severity,
            },
        )

    def record_backtest_result(self, result: BacktestResult) -> bool:
        """Append a backtest result. Returns `False` (and logs) on failure."""
        return self._safe_insert(
            "backtest_results",
            {
                "recorded_at": _iso(result.recorded_at or _utc_now()),
                "strategy": result.strategy,
                "timerange": result.timerange,
                "metrics": json.dumps(result.metrics),
            },
        )

    def _safe_insert(self, table: str, columns: dict[str, Any]) -> bool:
        placeholders = ", ".join("?" for _ in columns)
        column_names = ", ".join(columns)
        sql = f"INSERT INTO {table} ({column_names}) VALUES ({placeholders})"
        try:
            with self._connect() as conn:
                conn.execute(sql, tuple(columns.values()))
        except (sqlite3.Error, TypeError, ValueError) as exc:
            logger.error("Hermes memory write to %s failed (dropped, not raised): %s", table, exc)
            return False
        return True

    # -- reads -------------------------------------------------------

    def get_trades(
        self,
        pair: str | None = None,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[TradeRecord]:
        """Return trade records, oldest first, optionally filtered and capped.

        Parameters
        ----------
        pair:
            Restrict to this pair only, if given.
        since:
            Restrict to records with `recorded_at >= since`, if given.
        limit:
            If given, return at most the `limit` most recent matching
            records (still ordered oldest-first).
        """
        rows = self._select("trades", pair=pair, since=since, limit=limit, pair_column="pair")
        return [_row_to_trade(row) for row in rows]

    def get_process_events(
        self, since: datetime | None = None, limit: int | None = None
    ) -> list[ProcessEvent]:
        """Return process lifecycle events, oldest first."""
        rows = self._select("process_events", since=since, limit=limit)
        return [_row_to_process_event(row) for row in rows]

    def get_errors(
        self, since: datetime | None = None, limit: int | None = None
    ) -> list[ErrorEvent]:
        """Return recorded errors, oldest first."""
        rows = self._select("error_events", since=since, limit=limit)
        return [_row_to_error_event(row) for row in rows]

    def get_backtest_results(
        self, since: datetime | None = None, limit: int | None = None
    ) -> list[BacktestResult]:
        """Return recorded backtest results, oldest first."""
        rows = self._select("backtest_results", since=since, limit=limit)
        return [_row_to_backtest_result(row) for row in rows]

    def _select(
        self,
        table: str,
        *,
        since: datetime | None = None,
        limit: int | None = None,
        pair: str | None = None,
        pair_column: str | None = None,
    ) -> Sequence[sqlite3.Row]:
        clauses: list[str] = []
        params: list[Any] = []
        if since is not None:
            clauses.append("recorded_at >= ?")
            params.append(_iso(since))
        if pair is not None and pair_column is not None:
            clauses.append(f"{pair_column} = ?")
            params.append(pair)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        if limit is not None:
            # Take the most recent `limit` rows, but return them oldest-first
            # so callers always see history in chronological order.
            sql = (
                f"SELECT * FROM (SELECT * FROM {table} {where} ORDER BY id DESC LIMIT ?) "
                "sub ORDER BY id ASC"
            )
            params.append(limit)
        else:
            sql = f"SELECT * FROM {table} {where} ORDER BY id ASC"

        with self._connect() as conn:
            return conn.execute(sql, tuple(params)).fetchall()


def _row_to_trade(row: sqlite3.Row) -> TradeRecord:
    return TradeRecord(
        id=row["id"],
        recorded_at=_parse_iso(row["recorded_at"]),
        pair=row["pair"],
        side=row["side"],
        entry_time=_parse_iso(row["entry_time"]),
        exit_time=_parse_iso(row["exit_time"]),
        entry_price=row["entry_price"],
        exit_price=row["exit_price"],
        pnl=row["pnl"],
        pnl_pct=row["pnl_pct"],
        fees=row["fees"],
        funding=row["funding"],
        entry_zscore=row["entry_zscore"],
        exit_zscore=row["exit_zscore"],
        hedge_ratio=row["hedge_ratio"],
        holding_time_seconds=row["holding_time_seconds"],
        exit_reason=row["exit_reason"],
        regime=row["regime"],
        extra=json.loads(row["extra"]) if row["extra"] is not None else None,
    )


def _row_to_process_event(row: sqlite3.Row) -> ProcessEvent:
    return ProcessEvent(
        id=row["id"],
        recorded_at=_parse_iso(row["recorded_at"]),
        event_type=row["event_type"],
        pid=row["pid"],
        message=row["message"],
    )


def _row_to_error_event(row: sqlite3.Row) -> ErrorEvent:
    return ErrorEvent(
        id=row["id"],
        recorded_at=_parse_iso(row["recorded_at"]),
        source=row["source"],
        message=row["message"],
        severity=row["severity"],
    )


def _row_to_backtest_result(row: sqlite3.Row) -> BacktestResult:
    return BacktestResult(
        id=row["id"],
        recorded_at=_parse_iso(row["recorded_at"]),
        strategy=row["strategy"],
        timerange=row["timerange"],
        metrics=json.loads(row["metrics"]),
    )
