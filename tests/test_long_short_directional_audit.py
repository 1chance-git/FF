"""Focused unit tests for `hermes.long_short_directional_audit` (research-only)."""

from __future__ import annotations

from dataclasses import dataclass

from hermes.long_short_directional_audit import (
    LONG,
    SHORT,
    compare_duration_by_direction,
    compute_directional_stats,
    ema_distance_by_direction,
    outlier_sensitivity,
    split_by_direction,
)


@dataclass(frozen=True)
class _T:
    pair: str
    direction: str
    entry_time: str
    exit_reason: str | None
    profit_pct: float | None
    profit_abs: float | None
    duration_minutes: float | None = None

    @property
    def is_winner(self) -> bool | None:
        if self.profit_abs is None:
            return None
        return self.profit_abs > 0


def _long(profit_pct, profit_abs, exit_reason="exit_signal", duration=None, entry_time="2026-01-01"):
    return _T("BTC/USDC:USDC", LONG, entry_time, exit_reason, profit_pct, profit_abs, duration)


def _short(profit_pct, profit_abs, exit_reason="exit_signal", duration=None, entry_time="2026-01-01"):
    return _T("BTC/USDC:USDC", SHORT, entry_time, exit_reason, profit_pct, profit_abs, duration)


# ---------------------------------------------------------------------------
# split_by_direction
# ---------------------------------------------------------------------------


def test_split_by_direction_separates_long_and_short():
    trades = [_long(1, 1), _short(1, 1), _long(2, 2)]
    longs, shorts = split_by_direction(trades)
    assert len(longs) == 2
    assert len(shorts) == 1


def test_split_by_direction_excludes_unknown_direction():
    weird = _T("BTC/USDC:USDC", "FLAT", "2026-01-01", "exit_signal", 1.0, 1.0)
    longs, shorts = split_by_direction([weird])
    assert longs == []
    assert shorts == []


# ---------------------------------------------------------------------------
# compute_directional_stats
# ---------------------------------------------------------------------------


def test_compute_directional_stats_basic_counts():
    trades = [
        _long(-5.0, -50.0, "stop_loss"),
        _long(10.0, 100.0, "exit_signal"),
        _long(-2.0, -20.0, "exit_signal"),
    ]
    stats = compute_directional_stats(trades)
    assert stats.trade_count == 3
    assert stats.winners == 1
    assert stats.losers == 2
    assert stats.win_rate_pct == 100.0 / 3
    assert stats.stop_loss_count == 1
    assert stats.total_profit_pct == 3.0
    assert stats.avg_profit_pct == 1.0
    assert stats.expectancy_pct == stats.avg_profit_pct


def test_compute_directional_stats_profit_factor_and_gross():
    trades = [_long(10.0, 100.0), _long(-5.0, -50.0), _long(-5.0, -50.0)]
    stats = compute_directional_stats(trades)
    assert stats.gross_profit_abs == 100.0
    assert stats.gross_loss_abs == -100.0
    assert stats.profit_factor == 1.0


def test_compute_directional_stats_profit_factor_none_with_no_losers():
    trades = [_long(10.0, 100.0)]
    stats = compute_directional_stats(trades)
    assert stats.profit_factor is None


def test_compute_directional_stats_largest_winner_and_loser():
    trades = [_long(3.0, 30.0), _long(9.0, 90.0), _long(-1.0, -10.0), _long(-7.0, -70.0)]
    stats = compute_directional_stats(trades)
    assert stats.largest_winner_pct == 9.0
    assert stats.largest_loser_pct == -7.0


def test_compute_directional_stats_empty():
    stats = compute_directional_stats([])
    assert stats.trade_count == 0
    assert stats.win_rate_pct is None
    assert stats.largest_winner_pct is None
    assert stats.largest_loser_pct is None
    assert stats.profit_factor is None


# ---------------------------------------------------------------------------
# outlier_sensitivity
# ---------------------------------------------------------------------------


def test_outlier_sensitivity_removes_largest_magnitude_trades():
    trades = [_short(20.0, 200.0), _short(-1.0, -10.0), _short(1.0, 10.0), _short(-15.0, -150.0)]
    result = outlier_sensitivity(trades, n_remove=2)
    assert result.n_removed == 2
    assert set(result.removed_trade_profit_pcts) == {20.0, -15.0}
    assert result.full_total_profit_pct == 5.0
    assert result.without_outliers_total_profit_pct == 0.0


def test_outlier_sensitivity_conclusion_flip_detection():
    # Total is positive only because of one huge winner; removing it flips the sign.
    trades = [_short(50.0, 500.0), _short(-2.0, -20.0), _short(-3.0, -30.0)]
    result = outlier_sensitivity(trades, n_remove=1)
    assert result.full_total_profit_pct == 45.0
    assert result.without_outliers_total_profit_pct == -5.0
    assert result.conclusion_would_flip is True


def test_outlier_sensitivity_no_flip_when_sign_holds():
    trades = [_short(50.0, 500.0), _short(10.0, 100.0), _short(-3.0, -30.0)]
    result = outlier_sensitivity(trades, n_remove=1)
    assert result.conclusion_would_flip is False


def test_outlier_sensitivity_fewer_trades_than_n_remove():
    trades = [_short(5.0, 50.0)]
    result = outlier_sensitivity(trades, n_remove=2)
    assert result.n_removed == 1
    assert result.without_outliers_total_profit_pct is None


# ---------------------------------------------------------------------------
# compare_duration_by_direction
# ---------------------------------------------------------------------------


def test_compare_duration_by_direction():
    longs = [_long(1, 1, duration=100.0), _long(1, 1, duration=300.0)]
    shorts = [_short(1, 1, duration=50.0)]
    result = compare_duration_by_direction(longs, shorts)
    assert result.long_mean_minutes == 200.0
    assert result.long_median_minutes == 200.0
    assert result.long_n == 2
    assert result.short_mean_minutes == 50.0
    assert result.short_n == 1


def test_compare_duration_by_direction_missing_duration_excluded():
    longs = [_long(1, 1, duration=None)]
    shorts = []
    result = compare_duration_by_direction(longs, shorts)
    assert result.long_n == 0
    assert result.long_mean_minutes is None


# ---------------------------------------------------------------------------
# ema_distance_by_direction
# ---------------------------------------------------------------------------


def test_ema_distance_by_direction_splits_and_classifies():
    trades = [
        _long(-5.0, -50.0, "stop_loss", entry_time="2026-01-01"),
        _long(3.0, 30.0, "exit_signal", entry_time="2026-01-02"),
        _short(-5.0, -50.0, "stop_loss", entry_time="2026-01-03"),
        _short(4.0, 40.0, "exit_signal", entry_time="2026-01-04"),
    ]
    signal_idx = {
        ("BTC/USDC:USDC", "2026-01-01", LONG): {"ema_distance_pct": 0.08, "breakout_distance_pct": 0.01, "adx14": 30.0},
        ("BTC/USDC:USDC", "2026-01-02", LONG): {"ema_distance_pct": 0.02, "breakout_distance_pct": 0.01, "adx14": 30.0},
        ("BTC/USDC:USDC", "2026-01-03", SHORT): {"ema_distance_pct": 0.10, "breakout_distance_pct": 0.01, "adx14": 30.0},
        ("BTC/USDC:USDC", "2026-01-04", SHORT): {"ema_distance_pct": 0.03, "breakout_distance_pct": 0.01, "adx14": 30.0},
    }
    findings = ema_distance_by_direction(trades, signal_idx, {})
    ema_findings = {f.direction: f for f in findings if f.metric == "ema_distance_pct"}
    assert ema_findings[LONG].stop_loss_mean == 8.0
    assert ema_findings[LONG].exit_signal_mean == 2.0
    assert ema_findings[SHORT].stop_loss_mean == 10.0
    assert ema_findings[SHORT].exit_signal_mean == 3.0


def test_ema_distance_by_direction_covers_all_three_metrics_both_directions():
    findings = ema_distance_by_direction([_long(1, 1), _short(1, 1)], {}, {})
    metrics_seen = {(f.direction, f.metric) for f in findings}
    assert metrics_seen == {
        (LONG, "ema_distance_pct"), (LONG, "breakout_distance_pct"), (LONG, "adx14"),
        (SHORT, "ema_distance_pct"), (SHORT, "breakout_distance_pct"), (SHORT, "adx14"),
    }
