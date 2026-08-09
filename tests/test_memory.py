"""Unit tests for hermes.memory."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from hermes.memory import (
    BacktestResult,
    ErrorEvent,
    MemoryStore,
    MemoryStoreError,
    ProcessEvent,
    TradeRecord,
)


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "hermes_memory.sqlite3")


# -- schema / init -----------------------------------------------------


def test_opening_the_same_path_twice_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "hermes_memory.sqlite3"
    MemoryStore(db_path)
    second = MemoryStore(db_path)  # must not raise or wipe the schema
    assert second.get_trades() == []


def test_init_raises_memory_store_error_when_schema_cannot_be_created(tmp_path: Path) -> None:
    # A directory can't be opened as a sqlite file; sqlite3.connect() will
    # succeed lazily but the first real statement fails.
    directory_as_db_path = tmp_path / "not_a_file"
    directory_as_db_path.mkdir()

    with pytest.raises(MemoryStoreError):
        MemoryStore(directory_as_db_path)


def test_creates_parent_directories(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "memory.sqlite3"
    MemoryStore(nested)
    assert nested.exists()


# -- trades --------------------------------------------------------------


def test_record_and_retrieve_a_full_trade(store: MemoryStore) -> None:
    entry = datetime(2026, 1, 1, tzinfo=timezone.utc)
    exit_ = entry + timedelta(hours=3)
    record = TradeRecord(
        pair="BTC/USDC:USDC",
        side="long",
        entry_time=entry,
        exit_time=exit_,
        entry_price=50000.0,
        exit_price=50500.0,
        pnl=125.0,
        pnl_pct=0.01,
        fees=2.5,
        funding=0.75,
        entry_zscore=2.1,
        exit_zscore=0.1,
        hedge_ratio=1.32,
        holding_time_seconds=10800.0,
        exit_reason="z_score_exit",
        regime="mean_reverting",
        extra={"note": "test", "leg": "Y"},
    )

    assert store.record_trade(record) is True

    [saved] = store.get_trades()
    assert saved.pair == "BTC/USDC:USDC"
    assert saved.entry_time == entry
    assert saved.exit_time == exit_
    assert saved.pnl == 125.0
    assert saved.fees == 2.5
    assert saved.funding == 0.75
    assert saved.entry_zscore == 2.1
    assert saved.exit_zscore == 0.1
    assert saved.hedge_ratio == 1.32
    assert saved.holding_time_seconds == 10800.0
    assert saved.exit_reason == "z_score_exit"
    assert saved.regime == "mean_reverting"
    assert saved.extra == {"note": "test", "leg": "Y"}
    assert saved.id is not None


def test_missing_optional_fields_default_safely(store: MemoryStore) -> None:
    assert store.record_trade(TradeRecord(pair="ETH/USDC:USDC")) is True

    [saved] = store.get_trades()
    assert saved.pair == "ETH/USDC:USDC"
    assert saved.funding is None
    assert saved.entry_zscore is None
    assert saved.exit_reason is None
    assert saved.extra is None


def test_recorded_at_is_filled_in_automatically(store: MemoryStore) -> None:
    before = datetime.now(timezone.utc)
    store.record_trade(TradeRecord(pair="BTC/USDC:USDC"))
    after = datetime.now(timezone.utc)

    [saved] = store.get_trades()
    assert before <= saved.recorded_at <= after


def test_explicit_recorded_at_is_preserved(store: MemoryStore) -> None:
    when = datetime(2020, 5, 5, tzinfo=timezone.utc)
    store.record_trade(TradeRecord(pair="BTC/USDC:USDC", recorded_at=when))

    [saved] = store.get_trades()
    assert saved.recorded_at == when


def test_trades_are_append_only_and_never_overwritten(store: MemoryStore) -> None:
    for i in range(5):
        store.record_trade(TradeRecord(pair="BTC/USDC:USDC", pnl=float(i)))

    trades = store.get_trades()
    assert len(trades) == 5
    assert [t.pnl for t in trades] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert len({t.id for t in trades}) == 5  # every row kept its own id


def test_get_trades_filters_by_pair(store: MemoryStore) -> None:
    store.record_trade(TradeRecord(pair="BTC/USDC:USDC"))
    store.record_trade(TradeRecord(pair="ETH/USDC:USDC"))
    store.record_trade(TradeRecord(pair="BTC/USDC:USDC"))

    btc_trades = store.get_trades(pair="BTC/USDC:USDC")
    assert len(btc_trades) == 2
    assert all(t.pair == "BTC/USDC:USDC" for t in btc_trades)


def test_get_trades_filters_by_since(store: MemoryStore) -> None:
    old = datetime(2020, 1, 1, tzinfo=timezone.utc)
    recent = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store.record_trade(TradeRecord(pair="BTC/USDC:USDC", recorded_at=old))
    store.record_trade(TradeRecord(pair="BTC/USDC:USDC", recorded_at=recent))

    results = store.get_trades(since=datetime(2025, 1, 1, tzinfo=timezone.utc))
    assert len(results) == 1
    assert results[0].recorded_at == recent


def test_get_trades_limit_returns_most_recent_in_chronological_order(
    store: MemoryStore,
) -> None:
    for i in range(10):
        store.record_trade(TradeRecord(pair="BTC/USDC:USDC", pnl=float(i)))

    latest_three = store.get_trades(limit=3)
    assert [t.pnl for t in latest_three] == [7.0, 8.0, 9.0]


def test_get_trades_on_empty_store_returns_empty_list(store: MemoryStore) -> None:
    assert store.get_trades() == []


# -- process events / errors / backtest results ---------------------------


def test_record_and_retrieve_process_events(store: MemoryStore) -> None:
    store.record_process_event(ProcessEvent(event_type="start", pid=1234))
    store.record_process_event(ProcessEvent(event_type="crash", message="OOM killed"))
    store.record_process_event(ProcessEvent(event_type="stop"))

    events = store.get_process_events()
    assert [e.event_type for e in events] == ["start", "crash", "stop"]
    assert events[0].pid == 1234
    assert events[1].message == "OOM killed"
    assert events[2].pid is None


def test_record_and_retrieve_errors(store: MemoryStore) -> None:
    store.record_error(ErrorEvent(source="risk_engine", message="ADF test failed", severity="warning"))

    [saved] = store.get_errors()
    assert saved.source == "risk_engine"
    assert saved.message == "ADF test failed"
    assert saved.severity == "warning"


def test_error_severity_defaults_to_error(store: MemoryStore) -> None:
    store.record_error(ErrorEvent(source="hermes", message="disk full"))
    [saved] = store.get_errors()
    assert saved.severity == "error"


def test_record_and_retrieve_backtest_results(store: MemoryStore) -> None:
    metrics = {"profit_total": 123.45, "trades": 42, "win_rate": 0.55}
    store.record_backtest_result(
        BacktestResult(strategy="StatArbSwing", timerange="20250101-20260101", metrics=metrics)
    )

    [saved] = store.get_backtest_results()
    assert saved.strategy == "StatArbSwing"
    assert saved.timerange == "20250101-20260101"
    assert saved.metrics == metrics


def test_backtest_metrics_survive_nested_json_round_trip(store: MemoryStore) -> None:
    metrics = {"per_pair": {"BTC/USDC:USDC": {"trades": 3, "wins": 2}}, "tags": ["a", "b"]}
    store.record_backtest_result(BacktestResult(strategy="S", metrics=metrics))

    [saved] = store.get_backtest_results()
    assert saved.metrics == metrics


# -- failures never raise -------------------------------------------------


def test_write_failure_is_logged_and_returns_false_not_raised(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.ERROR, logger="hermes.memory")

    def broken_connect() -> sqlite3.Connection:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(store, "_connect", broken_connect)

    result = store.record_trade(TradeRecord(pair="BTC/USDC:USDC"))

    assert result is False
    assert "Hermes memory write" in caplog.text


def test_write_failure_on_process_event_does_not_raise(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_connect() -> sqlite3.Connection:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(store, "_connect", broken_connect)

    assert store.record_process_event(ProcessEvent(event_type="crash")) is False
    assert store.record_error(ErrorEvent(source="x", message="y")) is False
    assert store.record_backtest_result(BacktestResult(strategy="S", metrics={})) is False
