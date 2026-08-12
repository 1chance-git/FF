"""Unit tests for hermes.trade_report."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from hermes.trade_report import (
    TradeReportError,
    find_latest_export_file,
    load_trades_export,
    render_trade_report,
)

_LONG_WIN = {
    "pair": "BTC/USDC:USDC",
    "is_short": False,
    "open_date": "2026-01-20 00:00:00",
    "close_date": "2026-02-10 00:00:00",
    "open_rate": 100.0,
    "close_rate": 110.0,
    "enter_tag": "trend_long_donchian_breakout",
    "exit_reason": "exit_signal",
    "profit_abs": 10.0,
    "profit_ratio": 0.10,
    "trade_duration": 30240,
    "is_open": False,
}

_SHORT_LOSS = {
    "pair": "SOL/USDC:USDC",
    "is_short": True,
    "open_date": "2026-03-01 00:00:00",
    "close_date": "2026-03-01 12:00:00",
    "open_rate": 50.0,
    "close_rate": 52.5,
    "enter_tag": "trend_short_donchian_breakout",
    "exit_reason": "stop_loss",
    "profit_abs": -5.0,
    "profit_ratio": -0.05,
    "trade_duration": 720,
    "is_open": False,
}

_MISSING_FIELDS = {
    "pair": "ETH/USDC:USDC",
    # everything else genuinely absent from this trade dict
}


# ---------------------------------------------------------------------------
# find_latest_export_file
# ---------------------------------------------------------------------------


def test_find_latest_export_file_returns_none_for_missing_directory(tmp_path: Path) -> None:
    assert find_latest_export_file(tmp_path / "does_not_exist") is None


def test_find_latest_export_file_returns_none_for_empty_directory(tmp_path: Path) -> None:
    assert find_latest_export_file(tmp_path) is None


def test_find_latest_export_file_ignores_meta_json(tmp_path: Path) -> None:
    (tmp_path / "backtest-result-2026-01-01_00-00-00.meta.json").write_text("{}")
    assert find_latest_export_file(tmp_path) is None


def test_find_latest_export_file_picks_most_recently_modified(tmp_path: Path) -> None:
    older = tmp_path / "backtest-result-2026-01-01_00-00-00.json"
    newer = tmp_path / "backtest-result-2026-02-01_00-00-00.json"
    older.write_text("[]")
    newer.write_text("[]")
    # Ensure a clear ordering regardless of filesystem timestamp resolution.
    import os
    import time

    os.utime(older, (time.time() - 100, time.time() - 100))

    assert find_latest_export_file(tmp_path) == newer


# ---------------------------------------------------------------------------
# load_trades_export: shapes
# ---------------------------------------------------------------------------


def test_load_flat_list_shape(tmp_path: Path) -> None:
    path = tmp_path / "export.json"
    path.write_text(json.dumps([_LONG_WIN, _SHORT_LOSS]))

    report = load_trades_export(path)

    assert report.total_trades == 2
    assert report.source_path == path


def test_load_wrapped_single_strategy_shape(tmp_path: Path) -> None:
    path = tmp_path / "export.json"
    path.write_text(json.dumps({"strategy": {"TrendFollowCore": {"trades": [_LONG_WIN]}}}))

    report = load_trades_export(path)

    assert report.total_trades == 1
    assert report.trades[0].pair == "BTC/USDC:USDC"


def test_load_wrapped_selects_named_strategy_among_several(tmp_path: Path) -> None:
    path = tmp_path / "export.json"
    path.write_text(
        json.dumps(
            {
                "strategy": {
                    "TrendFollowCore": {"trades": [_LONG_WIN]},
                    "StatArbSwing": {"trades": [_SHORT_LOSS]},
                }
            }
        )
    )

    report = load_trades_export(path, strategy="StatArbSwing")

    assert report.total_trades == 1
    assert report.trades[0].pair == "SOL/USDC:USDC"


def test_load_wrapped_multiple_strategies_without_filter_raises(tmp_path: Path) -> None:
    path = tmp_path / "export.json"
    path.write_text(
        json.dumps(
            {
                "strategy": {
                    "TrendFollowCore": {"trades": [_LONG_WIN]},
                    "StatArbSwing": {"trades": [_SHORT_LOSS]},
                }
            }
        )
    )

    with pytest.raises(TradeReportError, match="multiple strategies"):
        load_trades_export(path)


def test_load_bare_trades_key_shape(tmp_path: Path) -> None:
    path = tmp_path / "export.json"
    path.write_text(json.dumps({"trades": [_LONG_WIN]}))

    report = load_trades_export(path)

    assert report.total_trades == 1


def test_load_unrecognized_shape_raises(tmp_path: Path) -> None:
    path = tmp_path / "export.json"
    path.write_text(json.dumps({"unexpected": "shape"}))

    with pytest.raises(TradeReportError, match="unrecognized"):
        load_trades_export(path)


def test_load_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "export.json"
    path.write_text("not json{{{")

    with pytest.raises(TradeReportError, match="cannot read"):
        load_trades_export(path)


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(TradeReportError, match="cannot read"):
        load_trades_export(tmp_path / "does_not_exist.json")


# ---------------------------------------------------------------------------
# Per-trade field mapping and "missing stays missing"
# ---------------------------------------------------------------------------


def test_trade_fields_map_from_raw_dict(tmp_path: Path) -> None:
    path = tmp_path / "export.json"
    path.write_text(json.dumps([_LONG_WIN]))

    [trade] = load_trades_export(path).trades

    assert trade.pair == "BTC/USDC:USDC"
    assert trade.direction == "LONG"
    assert trade.entry_time == "2026-01-20 00:00:00"
    assert trade.exit_time == "2026-02-10 00:00:00"
    assert trade.entry_price == 100.0
    assert trade.exit_price == 110.0
    assert trade.enter_tag == "trend_long_donchian_breakout"
    assert trade.exit_reason == "exit_signal"
    assert trade.profit_abs == 10.0
    assert trade.profit_pct == pytest.approx(10.0)
    assert trade.duration_minutes == 30240
    assert trade.is_open is False
    assert trade.is_winner is True
    assert trade.is_stop_loss is False


def test_short_trade_direction_is_short(tmp_path: Path) -> None:
    path = tmp_path / "export.json"
    path.write_text(json.dumps([_SHORT_LOSS]))

    [trade] = load_trades_export(path).trades

    assert trade.direction == "SHORT"
    assert trade.is_winner is False
    assert trade.is_stop_loss is True


def test_missing_fields_become_none_not_fabricated(tmp_path: Path) -> None:
    path = tmp_path / "export.json"
    path.write_text(json.dumps([_MISSING_FIELDS]))

    [trade] = load_trades_export(path).trades

    assert trade.pair == "ETH/USDC:USDC"
    assert trade.direction is None
    assert trade.entry_time is None
    assert trade.exit_time is None
    assert trade.entry_price is None
    assert trade.exit_price is None
    assert trade.enter_tag is None
    assert trade.exit_reason is None
    assert trade.profit_abs is None
    assert trade.profit_pct is None
    assert trade.duration_minutes is None
    assert trade.is_open is None
    assert trade.is_winner is None
    assert trade.is_stop_loss is None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregation_over_multiple_trades(tmp_path: Path) -> None:
    path = tmp_path / "export.json"
    path.write_text(json.dumps([_LONG_WIN, _SHORT_LOSS, _MISSING_FIELDS]))

    report = load_trades_export(path)

    assert report.total_trades == 3
    assert report.winners == 1
    assert report.losers == 1
    assert report.win_rate_pct == pytest.approx(50.0)  # only 2 of 3 have known outcome
    assert report.long_trades == 1
    assert report.short_trades == 1


def test_win_rate_is_none_when_no_trade_has_known_outcome(tmp_path: Path) -> None:
    path = tmp_path / "export.json"
    path.write_text(json.dumps([_MISSING_FIELDS]))

    report = load_trades_export(path)

    assert report.win_rate_pct is None


def test_per_pair_counts(tmp_path: Path) -> None:
    path = tmp_path / "export.json"
    path.write_text(json.dumps([_LONG_WIN, _SHORT_LOSS, _LONG_WIN]))

    report = load_trades_export(path)

    assert report.per_pair_counts() == {"BTC/USDC:USDC": 2, "SOL/USDC:USDC": 1}
    assert report.trades_for_pair("BTC/USDC:USDC") == (report.trades[0], report.trades[2])


def test_exit_reason_and_enter_tag_counts(tmp_path: Path) -> None:
    path = tmp_path / "export.json"
    path.write_text(json.dumps([_LONG_WIN, _SHORT_LOSS, _MISSING_FIELDS]))

    report = load_trades_export(path)

    assert report.exit_reason_counts() == {"exit_signal": 1, "stop_loss": 1, "N/A": 1}
    assert report.enter_tag_counts() == {
        "trend_long_donchian_breakout": 1,
        "trend_short_donchian_breakout": 1,
        "N/A": 1,
    }


def test_empty_export_has_zero_trades(tmp_path: Path) -> None:
    path = tmp_path / "export.json"
    path.write_text(json.dumps([]))

    report = load_trades_export(path)

    assert report.total_trades == 0
    assert report.winners == 0
    assert report.losers == 0
    assert report.win_rate_pct is None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_render_reports_na_for_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "export.json"
    path.write_text(json.dumps([_MISSING_FIELDS]))

    rendered = render_trade_report(load_trades_export(path))

    assert "N/A" in rendered
    assert "[TREND][TRADE FORENSICS]" in rendered


def test_render_empty_report_says_no_trades(tmp_path: Path) -> None:
    path = tmp_path / "export.json"
    path.write_text(json.dumps([]))

    rendered = render_trade_report(load_trades_export(path))

    assert "No trades found" in rendered


def test_render_includes_aggregate_totals(tmp_path: Path) -> None:
    path = tmp_path / "export.json"
    path.write_text(json.dumps([_LONG_WIN, _SHORT_LOSS]))

    rendered = render_trade_report(load_trades_export(path))

    assert "TOTAL TRADES: 2" in rendered
    assert "WINNERS: 1" in rendered
    assert "LOSERS: 1" in rendered
    assert "LONG TRADES: 1" in rendered
    assert "SHORT TRADES: 1" in rendered


# ---------------------------------------------------------------------------
# Native Freqtrade `.zip` export (real schema: freqtrade 2026.7,
# freqtrade/optimize/optimize_reports/bt_storage.py store_backtest_results)
# ---------------------------------------------------------------------------


def _write_freqtrade_zip(
    tmp_path: Path,
    stem: str,
    payload: dict | list,
    *,
    extra_members: dict[str, bytes] | None = None,
) -> Path:
    """Build a zip matching Freqtrade's own on-disk shape: `<stem>.zip`
    containing `<stem>.json` (the stats payload) plus whatever other
    members a real export also carries (sanitized config copy, etc.),
    which this module must never open."""
    zip_path = tmp_path / f"{stem}.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(f"{stem}.json", json.dumps(payload))
        for name, content in (extra_members or {}).items():
            archive.writestr(name, content)
    return zip_path


_WRAPPED_PAYLOAD = {
    "strategy": {"TrendFollowCore": {"trades": [_LONG_WIN, _SHORT_LOSS]}},
    "strategy_comparison": [{"key": "TrendFollowCore", "trades": 2}],
}


def test_zip_json_member_name_matches_freqtrade_convention(tmp_path: Path) -> None:
    from hermes.trade_report import _zip_json_member_name

    zip_path = tmp_path / "backtest-result-2026-08-12_15-11-31.zip"
    assert _zip_json_member_name(zip_path) == "backtest-result-2026-08-12_15-11-31.json"


def test_load_zip_export_reads_wrapped_shape(tmp_path: Path) -> None:
    zip_path = _write_freqtrade_zip(
        tmp_path,
        "backtest-result-2026-08-12_15-11-31",
        _WRAPPED_PAYLOAD,
        extra_members={"backtest-result-2026-08-12_15-11-31_config.json": b"{}"},
    )

    report = load_trades_export(zip_path, strategy="TrendFollowCore")

    assert report.total_trades == 2
    assert report.source_path == zip_path
    assert {t.pair for t in report.trades} == {"BTC/USDC:USDC", "SOL/USDC:USDC"}


def test_load_zip_export_reads_flat_list_shape(tmp_path: Path) -> None:
    zip_path = _write_freqtrade_zip(tmp_path, "backtest-result-flat", [_LONG_WIN])

    report = load_trades_export(zip_path)

    assert report.total_trades == 1
    assert report.trades[0].pair == "BTC/USDC:USDC"


def test_zip_and_json_produce_equivalent_trade_records(tmp_path: Path) -> None:
    json_path = tmp_path / "export.json"
    json_path.write_text(json.dumps(_WRAPPED_PAYLOAD))
    zip_path = _write_freqtrade_zip(tmp_path, "backtest-result-equiv", _WRAPPED_PAYLOAD)

    json_report = load_trades_export(json_path, strategy="TrendFollowCore")
    zip_report = load_trades_export(zip_path, strategy="TrendFollowCore")

    assert json_report.trades == zip_report.trades


def test_load_zip_export_ignores_other_zip_members_and_extracts_nothing_to_disk(
    tmp_path: Path,
) -> None:
    """A crafted member name elsewhere in the archive (including one that
    looks like a path-traversal attempt) must never be opened or written
    anywhere -- this module only ever reads the one member name it itself
    computes."""
    before = set(tmp_path.iterdir())
    zip_path = _write_freqtrade_zip(
        tmp_path,
        "backtest-result-traversal",
        _WRAPPED_PAYLOAD,
        extra_members={
            "backtest-result-traversal_config.json": b"{}",
            "../../../../tmp/evil_hermes_test_marker.json": b"malicious",
        },
    )

    report = load_trades_export(zip_path, strategy="TrendFollowCore")

    assert report.total_trades == 2
    # Nothing was extracted: the only new filesystem entry is the zip itself.
    after = set(tmp_path.iterdir())
    assert after - before == {zip_path}
    assert not (Path("/tmp") / "evil_hermes_test_marker.json").exists()


def test_load_zip_export_missing_expected_member_raises(tmp_path: Path) -> None:
    zip_path = tmp_path / "backtest-result-orphan.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("something-else.json", json.dumps(_WRAPPED_PAYLOAD))

    with pytest.raises(TradeReportError, match="not found"):
        load_trades_export(zip_path)


def test_load_zip_export_malformed_zip_raises(tmp_path: Path) -> None:
    zip_path = tmp_path / "backtest-result-broken.zip"
    zip_path.write_bytes(b"this is not a zip file")

    with pytest.raises(TradeReportError, match="not a valid zip"):
        load_trades_export(zip_path)


def test_load_zip_export_malformed_json_inside_zip_raises(tmp_path: Path) -> None:
    zip_path = tmp_path / "backtest-result-badjson.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("backtest-result-badjson.json", "not json{{{")

    with pytest.raises(TradeReportError, match="cannot read export file"):
        load_trades_export(zip_path)


def test_load_zip_export_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(TradeReportError, match="cannot read export file"):
        load_trades_export(tmp_path / "does_not_exist.zip")


# ---------------------------------------------------------------------------
# find_latest_export_file: `.last_result.json` pointer + `.zip` support
# ---------------------------------------------------------------------------


def test_find_latest_export_file_prefers_last_result_pointer(tmp_path: Path) -> None:
    import os
    import time

    older = _write_freqtrade_zip(tmp_path, "backtest-result-2026-01-01_00-00-00", _WRAPPED_PAYLOAD)
    newer = _write_freqtrade_zip(tmp_path, "backtest-result-2026-02-01_00-00-00", _WRAPPED_PAYLOAD)
    os.utime(older, (time.time() - 100, time.time() - 100))
    # Pointer explicitly names the OLDER file -- proves the pointer wins over mtime.
    (tmp_path / ".last_result.json").write_text(
        json.dumps({"latest_backtest": older.name})
    )

    assert find_latest_export_file(tmp_path) == older


def test_find_latest_export_file_falls_back_when_pointer_invalid(tmp_path: Path) -> None:
    zip_path = _write_freqtrade_zip(tmp_path, "backtest-result-only", _WRAPPED_PAYLOAD)
    (tmp_path / ".last_result.json").write_text("not json{{{")

    assert find_latest_export_file(tmp_path) == zip_path


def test_find_latest_export_file_falls_back_when_pointer_points_to_missing_file(
    tmp_path: Path,
) -> None:
    zip_path = _write_freqtrade_zip(tmp_path, "backtest-result-only", _WRAPPED_PAYLOAD)
    (tmp_path / ".last_result.json").write_text(
        json.dumps({"latest_backtest": "backtest-result-does-not-exist.zip"})
    )

    assert find_latest_export_file(tmp_path) == zip_path


def test_find_latest_export_file_finds_zip_via_glob_without_pointer(tmp_path: Path) -> None:
    import os
    import time

    older = _write_freqtrade_zip(tmp_path, "backtest-result-2026-01-01_00-00-00", _WRAPPED_PAYLOAD)
    newer = _write_freqtrade_zip(tmp_path, "backtest-result-2026-02-01_00-00-00", _WRAPPED_PAYLOAD)
    os.utime(older, (time.time() - 100, time.time() - 100))

    assert find_latest_export_file(tmp_path) == newer


def test_find_latest_export_file_still_ignores_meta_json_alongside_zip(tmp_path: Path) -> None:
    zip_path = _write_freqtrade_zip(tmp_path, "backtest-result-x", _WRAPPED_PAYLOAD)
    (tmp_path / "backtest-result-x.meta.json").write_text("{}")

    assert find_latest_export_file(tmp_path) == zip_path
