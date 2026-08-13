"""Focused unit tests for `hermes.short_edge_attribution_audit` (research-only)."""

from __future__ import annotations

import pandas as pd
import pytest

from hermes.short_edge_attribution_audit import (
    aggregate_mfe_mae,
    attach_mfe_mae,
    duration_stats,
    filter_by_winner,
    group_mfe_mae_by_pair_direction,
    group_trades_by_pair_direction,
    outlier_robustness_series,
    remove_top_n_winners,
    resolved_only,
)
from hermes.trade_report import Trade


def _trade(
    pair="BTC/USDC:USDC", direction="LONG", entry_time="2026-01-01T00:00:00Z",
    exit_time="2026-01-01T08:00:00Z", entry_price=100.0, profit_pct=1.0, profit_abs=10.0,
    duration_minutes=480.0, exit_reason="exit_signal",
):
    return Trade(
        pair=pair, direction=direction, entry_time=entry_time, exit_time=exit_time,
        entry_price=entry_price, exit_price=101.0, enter_tag=None, exit_reason=exit_reason,
        profit_abs=profit_abs, profit_pct=profit_pct, duration_minutes=duration_minutes, is_open=False,
    )


def _candles(rows):
    return pd.DataFrame(
        [{"date": pd.Timestamp(d, tz="UTC"), "open": o, "high": h, "low": l, "close": c} for d, o, h, l, c in rows]
    )


# ---------------------------------------------------------------------------
# attach_mfe_mae / resolved_only
# ---------------------------------------------------------------------------


def test_attach_mfe_mae_resolves_when_ohlcv_present():
    trade = _trade(pair="BTC/USDC:USDC", direction="LONG", entry_time="2026-01-01T00:00:00Z",
                    exit_time="2026-01-01T04:00:00Z", entry_price=100.0)
    ohlcv = {"BTC/USDC:USDC": _candles([
        ("2026-01-01T00:00", 100, 108, 95, 101),
        ("2026-01-01T04:00", 101, 106, 99, 102),
    ])}
    joined = attach_mfe_mae([trade], ohlcv)
    assert len(joined) == 1
    assert joined[0].result.is_resolved
    assert joined[0].result.mfe_pct == pytest.approx(8.0)


def test_attach_mfe_mae_unresolved_when_pair_missing_from_ohlcv():
    trade = _trade(pair="SOL/USDC:USDC")
    joined = attach_mfe_mae([trade], {"BTC/USDC:USDC": _candles([("2026-01-01T00:00", 100, 105, 95, 101)])})
    assert joined[0].result.is_resolved is False


def test_resolved_only_filters_unresolved():
    trade_ok = _trade(pair="BTC/USDC:USDC", entry_time="2026-01-01T00:00:00Z", exit_time="2026-01-01T04:00:00Z")
    trade_bad = _trade(pair="SOL/USDC:USDC")
    ohlcv = {"BTC/USDC:USDC": _candles([
        ("2026-01-01T00:00", 100, 105, 95, 101), ("2026-01-01T04:00", 101, 106, 99, 102),
    ])}
    joined = attach_mfe_mae([trade_ok, trade_bad], ohlcv)
    assert len(resolved_only(joined)) == 1


# ---------------------------------------------------------------------------
# aggregate_mfe_mae / filter_by_winner
# ---------------------------------------------------------------------------


def test_aggregate_mfe_mae_basic():
    trade1 = _trade(pair="BTC/USDC:USDC", entry_time="2026-01-01T00:00:00Z", exit_time="2026-01-01T04:00:00Z")
    trade2 = _trade(pair="BTC/USDC:USDC", entry_time="2026-01-02T00:00:00Z", exit_time="2026-01-02T04:00:00Z")
    ohlcv = {"BTC/USDC:USDC": _candles([
        ("2026-01-01T00:00", 100, 110, 95, 101), ("2026-01-01T04:00", 101, 106, 99, 102),
        ("2026-01-02T00:00", 100, 104, 90, 101), ("2026-01-02T04:00", 101, 103, 98, 102),
    ])}
    joined = attach_mfe_mae([trade1, trade2], ohlcv)
    agg = aggregate_mfe_mae(joined)
    assert agg.n == 2
    assert agg.mean_mfe_pct == pytest.approx((10.0 + 4.0) / 2)


def test_aggregate_mfe_mae_empty():
    agg = aggregate_mfe_mae([])
    assert agg.n == 0
    assert agg.mean_mfe_pct is None


def test_filter_by_winner():
    win = _trade(profit_abs=10.0)
    lose = _trade(profit_abs=-5.0)
    ohlcv = {"BTC/USDC:USDC": _candles([
        ("2026-01-01T00:00", 100, 105, 95, 101), ("2026-01-01T08:00", 101, 106, 99, 102),
    ])}
    joined = attach_mfe_mae([win, lose], ohlcv)
    assert len(filter_by_winner(joined, True)) == 1
    assert len(filter_by_winner(joined, False)) == 1


# ---------------------------------------------------------------------------
# duration_stats
# ---------------------------------------------------------------------------


def test_duration_stats_basic():
    trades = [_trade(duration_minutes=120.0), _trade(duration_minutes=240.0)]
    stats = duration_stats(trades)
    assert stats.n == 2
    assert stats.mean_minutes == 180.0
    assert stats.median_minutes == 180.0


def test_duration_stats_excludes_none():
    t1 = _trade(duration_minutes=100.0)
    t2 = Trade(pair="BTC/USDC:USDC", direction="LONG", entry_time="t", exit_time=None,
               entry_price=100.0, exit_price=None, enter_tag=None, exit_reason=None,
               profit_abs=None, profit_pct=None, duration_minutes=None, is_open=True)
    stats = duration_stats([t1, t2])
    assert stats.n == 1
    assert stats.mean_minutes == 100.0


# ---------------------------------------------------------------------------
# remove_top_n_winners / outlier_robustness_series
# ---------------------------------------------------------------------------


def test_remove_top_n_winners_removes_largest():
    t1 = _trade(profit_pct=10.0)
    t2 = _trade(profit_pct=5.0)
    t3 = _trade(profit_pct=-2.0)
    remaining = remove_top_n_winners([t1, t2, t3], 1)
    assert remaining == [t2, t3]


def test_remove_top_n_winners_zero_is_noop():
    trades = [_trade(profit_pct=1.0), _trade(profit_pct=2.0)]
    assert remove_top_n_winners(trades, 0) == trades


def test_remove_top_n_winners_n_exceeds_length():
    trades = [_trade(profit_pct=1.0)]
    assert remove_top_n_winners(trades, 5) == []


def test_outlier_robustness_series_progressive_removal():
    trades = [_trade(profit_pct=10.0), _trade(profit_pct=5.0), _trade(profit_pct=-2.0)]
    series = outlier_robustness_series(trades, max_n=2)
    assert len(series) == 3
    assert series[0].n_removed == 0
    assert series[0].total_profit_pct == pytest.approx(13.0)
    assert series[1].n_removed == 1
    assert series[1].total_profit_pct == pytest.approx(3.0)
    assert series[2].n_removed == 2
    assert series[2].total_profit_pct == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# group_trades_by_pair_direction / group_mfe_mae_by_pair_direction
# ---------------------------------------------------------------------------


def test_group_trades_by_pair_direction():
    trades = [
        _trade(pair="BTC/USDC:USDC", direction="LONG"),
        _trade(pair="BTC/USDC:USDC", direction="SHORT"),
        _trade(pair="ETH/USDC:USDC", direction="LONG"),
    ]
    groups = group_trades_by_pair_direction(trades, ["BTC/USDC:USDC", "ETH/USDC:USDC"])
    assert len(groups[("BTC/USDC:USDC", "LONG")]) == 1
    assert len(groups[("BTC/USDC:USDC", "SHORT")]) == 1
    assert len(groups[("ETH/USDC:USDC", "LONG")]) == 1
    assert groups[("ETH/USDC:USDC", "SHORT")] == []


def test_group_mfe_mae_by_pair_direction():
    trade = _trade(pair="BTC/USDC:USDC", direction="SHORT", entry_time="2026-01-01T00:00:00Z",
                    exit_time="2026-01-01T04:00:00Z")
    ohlcv = {"BTC/USDC:USDC": _candles([
        ("2026-01-01T00:00", 100, 105, 95, 101), ("2026-01-01T04:00", 101, 106, 99, 102),
    ])}
    joined = attach_mfe_mae([trade], ohlcv)
    groups = group_mfe_mae_by_pair_direction(joined, ["BTC/USDC:USDC"])
    assert len(groups[("BTC/USDC:USDC", "SHORT")]) == 1
    assert groups[("BTC/USDC:USDC", "LONG")] == []
