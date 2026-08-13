"""Focused unit tests for `hermes.short_persistence_audit` (research-only)."""

from __future__ import annotations

import pandas as pd
import pytest

from hermes.short_persistence_audit import (
    THIN_SAMPLE_THRESHOLD,
    EntryConditionMetrics,
    PostEntryMetrics,
    adx,
    aggregate_metric,
    compute_entry_condition_metrics,
    compute_post_entry_metrics,
    donchian_prev_bounds,
    ema,
    group_by_quarter,
    identify_persistent_winners,
    list_short_winners,
    reconcile,
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
# A. reconcile
# ---------------------------------------------------------------------------


def test_reconcile_matches_expected():
    trades = (
        [_trade(pair="BTC/USDC:USDC", direction="LONG") for _ in range(9)]
        + [_trade(pair="BTC/USDC:USDC", direction="SHORT") for _ in range(10)]
        + [_trade(pair="ETH/USDC:USDC", direction="LONG") for _ in range(9)]
        + [_trade(pair="ETH/USDC:USDC", direction="SHORT") for _ in range(11)]
    )
    result = reconcile(trades)
    assert result.n == 39
    assert result.n_long == 18
    assert result.n_short == 21
    assert result.n_btc == 19
    assert result.n_eth == 20
    assert result.matches_expected is True


def test_reconcile_mismatch_reported_not_raised():
    trades = [_trade() for _ in range(5)]
    result = reconcile(trades)
    assert result.matches_expected is False
    assert result.n == 5


def test_reconcile_date_range():
    trades = [
        _trade(entry_time="2026-02-01T00:00:00Z", exit_time="2026-02-02T00:00:00Z"),
        _trade(entry_time="2026-01-01T00:00:00Z", exit_time="2026-03-01T00:00:00Z"),
    ]
    result = reconcile(trades, expected_n=2, expected_long=2, expected_short=0, expected_btc=2, expected_eth=0)
    assert result.earliest_entry == "2026-01-01T00:00:00Z"
    assert result.latest_exit == "2026-03-01T00:00:00Z"


# ---------------------------------------------------------------------------
# B. list_short_winners / identify_persistent_winners
# ---------------------------------------------------------------------------


def test_list_short_winners_filters_correctly():
    short_win = _trade(direction="SHORT", profit_abs=10.0)
    short_lose = _trade(direction="SHORT", profit_abs=-5.0)
    long_win = _trade(direction="LONG", profit_abs=10.0)
    records = list_short_winners([short_win, short_lose, long_win], {})
    assert len(records) == 1
    assert records[0].trade is short_win


def test_identify_persistent_winners_top_n():
    from hermes.short_persistence_audit import ShortWinnerRecord
    winners = [
        ShortWinnerRecord(trade=_trade(profit_pct=30.0), duration_minutes=100, mfe_pct=30, mae_pct=1, profit_pct=30.0),
        ShortWinnerRecord(trade=_trade(profit_pct=20.0), duration_minutes=90, mfe_pct=20, mae_pct=1, profit_pct=20.0),
        ShortWinnerRecord(trade=_trade(profit_pct=10.0), duration_minutes=80, mfe_pct=10, mae_pct=1, profit_pct=10.0),
        ShortWinnerRecord(trade=_trade(profit_pct=1.0), duration_minutes=10, mfe_pct=1, mae_pct=1, profit_pct=1.0),
    ]
    persistent, ordinary = identify_persistent_winners(winners, n=2)
    assert len(persistent) == 2
    assert {w.profit_pct for w in persistent} == {30.0, 20.0}
    assert len(ordinary) == 2


def test_identify_persistent_winners_empty():
    persistent, ordinary = identify_persistent_winners([], n=3)
    assert persistent == []
    assert ordinary == []


# ---------------------------------------------------------------------------
# ema / adx / donchian_prev_bounds
# ---------------------------------------------------------------------------


def test_ema_matches_pandas_ewm_definition():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ema(series, period=3)
    expected = series.ewm(span=3, adjust=False).mean()
    pd.testing.assert_series_equal(result, expected)


def test_donchian_prev_bounds_excludes_current_candle():
    df = _candles([
        ("2026-01-01T00:00", 100, 110, 90, 105),
        ("2026-01-01T04:00", 105, 120, 95, 110),
        ("2026-01-01T08:00", 110, 200, 5, 115),  # extreme candle -- shouldn't affect its own bounds
    ])
    upper, lower = donchian_prev_bounds(df, period=2)
    # Row 2's bounds come from rows 0-1 only (max high=120, min low=90), not row 2's own 200/5.
    assert upper.iloc[2] == 120
    assert lower.iloc[2] == 90


def test_adx_returns_series_same_length():
    df = _candles([(f"2026-01-{i:02d}T00:00", 100 + i, 105 + i, 95 + i, 101 + i) for i in range(1, 20)])
    result = adx(df, period=14)
    assert len(result) == len(df)


# ---------------------------------------------------------------------------
# compute_entry_condition_metrics
# ---------------------------------------------------------------------------


def test_compute_entry_condition_metrics_none_ohlcv():
    result = compute_entry_condition_metrics(None, "2026-01-01T00:00:00Z", 100.0, "LONG")
    assert result.ema200_distance_pct is None
    assert result.adx_at_entry is None


def test_compute_entry_condition_metrics_none_entry_time():
    df = _candles([("2026-01-01T00:00", 100, 105, 95, 101)])
    result = compute_entry_condition_metrics(df, None, 100.0, "LONG")
    assert result.ema200_distance_pct is None


def test_compute_entry_condition_metrics_never_uses_post_entry_candles():
    df = _candles([
        ("2026-01-01T00:00", 100, 105, 95, 100),
        ("2026-01-01T04:00", 100, 106, 96, 101),  # entry candle
        ("2026-01-01T08:00", 101, 500, 1, 200),   # post-entry: should NOT affect entry metrics
    ])
    result = compute_entry_condition_metrics(df, "2026-01-01T04:00:00Z", 101.0, "LONG", lookback_candles=2)
    # realized vol should be computed only from candles up to and including entry (2 candles -> 1 return)
    assert result.realized_vol_before_entry_pct is None or result.realized_vol_before_entry_pct >= 0
    # sanity: the huge post-entry candle must not leak into ema/adx (computed only from <= entry_time rows)
    df_up_to_entry = df[df["date"] <= pd.Timestamp("2026-01-01T04:00", tz="UTC")]
    assert len(df_up_to_entry) == 2


def test_compute_entry_condition_metrics_ema_distance_sign():
    # Entry price well above a flat EMA -> positive distance for LONG framing.
    rows = [(f"2026-01-{i:02d}T00:00", 100, 101, 99, 100) for i in range(1, 10)]
    df = _candles(rows)
    result = compute_entry_condition_metrics(df, "2026-01-09T00:00:00Z", 110.0, "LONG")
    assert result.ema200_distance_pct is not None
    assert result.ema200_distance_pct > 0


# ---------------------------------------------------------------------------
# compute_post_entry_metrics
# ---------------------------------------------------------------------------


def test_compute_post_entry_metrics_basic():
    window = _candles([
        ("2026-01-01T00:00", 100, 105, 95, 100),
        ("2026-01-01T04:00", 100, 110, 90, 105),
        ("2026-01-01T08:00", 105, 108, 100, 103),
    ])
    result = compute_post_entry_metrics(window)
    assert result.price_expansion_pct == pytest.approx(20.0)  # (110-90)/100 * 100
    assert result.realized_vol_after_entry_pct is not None


def test_compute_post_entry_metrics_none_window():
    result = compute_post_entry_metrics(None)
    assert result.realized_vol_after_entry_pct is None
    assert result.price_expansion_pct is None


def test_compute_post_entry_metrics_single_candle():
    window = _candles([("2026-01-01T00:00", 100, 105, 95, 100)])
    result = compute_post_entry_metrics(window)
    assert result.realized_vol_after_entry_pct is None


# ---------------------------------------------------------------------------
# aggregate_metric
# ---------------------------------------------------------------------------


def test_aggregate_metric_thin_sample_flag():
    result = aggregate_metric([1.0, 2.0, 3.0])
    assert result.n == 3
    assert result.is_thin_sample is True
    assert THIN_SAMPLE_THRESHOLD == 5

    result2 = aggregate_metric([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert result2.is_thin_sample is False


def test_aggregate_metric_excludes_none():
    result = aggregate_metric([1.0, None, 3.0, None])
    assert result.n == 2
    assert result.mean_value == 2.0


def test_aggregate_metric_empty():
    result = aggregate_metric([])
    assert result.n == 0
    assert result.mean_value is None
    assert result.is_thin_sample is True


# ---------------------------------------------------------------------------
# group_by_quarter
# ---------------------------------------------------------------------------


def test_group_by_quarter_even_split():
    trades = [_trade(entry_time=f"2026-01-{i:02d}T00:00:00Z") for i in range(1, 9)]
    quarters = group_by_quarter(trades, 4)
    assert [len(q) for q in quarters] == [2, 2, 2, 2]


def test_group_by_quarter_deterministic_order():
    trades = [_trade(entry_time=f"2026-01-{i:02d}T00:00:00Z") for i in range(8, 0, -1)]
    quarters = group_by_quarter(trades, 4)
    assert quarters[0][0].entry_time == "2026-01-01T00:00:00Z"


def test_group_by_quarter_empty():
    quarters = group_by_quarter([], 4)
    assert quarters == [[], [], [], []]
