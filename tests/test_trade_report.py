"""Unit tests for hermes.trade_report."""

from __future__ import annotations

import json
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
