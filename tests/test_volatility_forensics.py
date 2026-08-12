"""Unit tests for hermes.volatility_forensics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.unit

from hermes.volatility_forensics import (
    VolatilityEntryContext,
    audit_no_lookahead_volatility,
    build_volatility_dataframe,
    compute_atr,
    compute_atr_pct,
    compute_log_returns,
    compute_realized_volatility,
    compute_true_range,
    compute_volatility_ema_correlations,
    pearson_correlation,
    quartile_buckets,
    reconstruct_all_volatility_contexts,
    reconstruct_volatility_context,
    save_volatility_dataset,
    spearman_correlation,
    summarize_entry_volatility_by_pair,
    summarize_pair_volatility,
    summarize_stop_distance_in_atr_by_pair,
    summarize_stop_loss_vs_exit_signal,
    summarize_winner_vs_loser,
)


def _synthetic_ohlcv(n: int = 100, *, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    noise = rng.normal(0, 1.0, n)
    close = 100 + np.cumsum(noise * 0.5)
    high = close + np.abs(rng.normal(0.5, 0.2, n))
    low = close - np.abs(rng.normal(0.5, 0.2, n))
    return pd.DataFrame({"date": dates, "high": high, "low": low, "close": close})


# ---------------------------------------------------------------------------
# 1. True Range
# ---------------------------------------------------------------------------


def test_true_range_uses_max_of_three_components() -> None:
    df = pd.DataFrame(
        {
            "high": [10.0, 12.0, 9.0],
            "low": [9.0, 10.0, 7.0],
            "close": [9.5, 11.0, 7.5],
        }
    )
    tr = compute_true_range(df)

    # Row 0 has no previous close, so only high-low is available; pandas'
    # max(axis=1) skips the NaN gap terms rather than propagating NaN.
    assert tr.iloc[0] == pytest.approx(1.0)
    # row 1: high-low=2.0, |high-prevclose|=|12-9.5|=2.5, |low-prevclose|=|10-9.5|=0.5 -> max=2.5
    assert tr.iloc[1] == pytest.approx(2.5)
    # row 2: high-low=2.0, |high-prevclose|=|9-11|=2.0, |low-prevclose|=|7-11|=4.0 -> max=4.0
    assert tr.iloc[2] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# 2. ATR
# ---------------------------------------------------------------------------


def test_atr_is_rolling_mean_of_true_range() -> None:
    df = _synthetic_ohlcv(n=30)
    atr = compute_atr(df, period=14)

    assert pd.isna(atr.iloc[12])  # not enough candles yet (need 14 TR values)
    assert pd.notna(atr.iloc[14])
    tr = compute_true_range(df)
    assert atr.iloc[20] == pytest.approx(tr.iloc[7:21].mean())


# ---------------------------------------------------------------------------
# 3. ATR% normalization
# ---------------------------------------------------------------------------


def test_atr_pct_normalizes_by_close() -> None:
    df = _synthetic_ohlcv(n=30)
    atr = compute_atr(df, period=14)
    atr_pct = compute_atr_pct(df, period=14)

    idx = 20
    assert atr_pct.iloc[idx] == pytest.approx(atr.iloc[idx] / df["close"].iloc[idx] * 100.0)


# ---------------------------------------------------------------------------
# 4. log-return calculation
# ---------------------------------------------------------------------------


def test_log_return_calculation() -> None:
    df = pd.DataFrame({"close": [100.0, 110.0, 99.0]})
    returns = compute_log_returns(df)

    assert pd.isna(returns.iloc[0])
    assert returns.iloc[1] == pytest.approx(np.log(110.0 / 100.0))
    assert returns.iloc[2] == pytest.approx(np.log(99.0 / 110.0))


# ---------------------------------------------------------------------------
# 5. realized volatility calculation
# ---------------------------------------------------------------------------


def test_realized_volatility_is_rolling_std_of_log_returns() -> None:
    df = _synthetic_ohlcv(n=100)
    rv = compute_realized_volatility(df, window=42)

    assert pd.isna(rv.iloc[40])  # fewer than 42 returns available
    assert pd.notna(rv.iloc[42])
    returns = compute_log_returns(df)
    assert rv.iloc[60] == pytest.approx(returns.iloc[19:61].std())


# ---------------------------------------------------------------------------
# 6. timestamp alignment (entry-candle lookup via find_entry_candle)
# ---------------------------------------------------------------------------


def test_reconstruct_volatility_context_aligns_to_exact_entry_candle() -> None:
    df = _synthetic_ohlcv(n=100)
    vol_df = build_volatility_dataframe(df, atr_period=14, realized_vol_window=42)
    row = vol_df.iloc[70]

    signal_trade = {
        "trade_number": 1,
        "pair": "BTC/USDC:USDC",
        "direction": "LONG",
        "entry_time": str(row["date"]),
        "entry_price": 100.0,
        "adx14": 30.0,
        "ema_distance_pct": 0.05,
        "exit_reason": "exit_signal",
        "profit_pct": 5.0,
        "duration_minutes": 1000,
        "is_winner": True,
    }

    context = reconstruct_volatility_context(signal_trade, vol_df)

    assert context.candle_matched is True
    assert context.atr_pct == pytest.approx(float(row["atr_pct"]))
    assert context.realized_vol == pytest.approx(float(row["realized_vol"]))


# ---------------------------------------------------------------------------
# 7. no-lookahead behavior
# ---------------------------------------------------------------------------


def test_audit_no_lookahead_volatility_passes_for_real_calculations() -> None:
    df = _synthetic_ohlcv(n=100)
    assert audit_no_lookahead_volatility(df, at_index=90) is True
    assert audit_no_lookahead_volatility(df, at_index=60) is True


def test_audit_no_lookahead_volatility_detects_a_leaky_calculation(monkeypatch) -> None:
    """A deliberately centered (lookahead) ATR stand-in must FAIL the audit."""
    import hermes.volatility_forensics as vf_module

    def leaky_build_volatility_dataframe(ohlcv, *, atr_period=14, realized_vol_window=42):
        df = ohlcv.copy()
        tr = vf_module.compute_true_range(df)
        # Centered window: uses future rows relative to each row's own position.
        df["atr_pct"] = tr.rolling(window=atr_period, center=True, min_periods=1).mean() / df["close"] * 100.0
        df["realized_vol"] = vf_module.compute_log_returns(df).rolling(
            window=realized_vol_window, center=True, min_periods=1
        ).std()
        return df

    monkeypatch.setattr(vf_module, "build_volatility_dataframe", leaky_build_volatility_dataframe)

    df = _synthetic_ohlcv(n=100)
    assert vf_module.audit_no_lookahead_volatility(df, at_index=60) is False


# ---------------------------------------------------------------------------
# 8. missing-data handling
# ---------------------------------------------------------------------------


def test_reconstruct_volatility_context_handles_no_data_for_pair() -> None:
    signal_trade = {
        "trade_number": 1,
        "pair": "SOL/USDC:USDC",
        "direction": "LONG",
        "entry_time": "2026-02-01 00:00:00+00:00",
        "entry_price": 100.0,
        "adx14": 30.0,
        "ema_distance_pct": 0.05,
        "exit_reason": "exit_signal",
        "profit_pct": 5.0,
        "duration_minutes": 1000,
        "is_winner": True,
    }

    context = reconstruct_volatility_context(signal_trade, None)

    assert context.candle_matched is False
    assert context.atr_pct is None
    assert context.realized_vol is None
    assert context.stop_distance_in_atr is None
    # Outcome/signal fields still preserved.
    assert context.exit_reason == "exit_signal"


def test_reconstruct_volatility_context_handles_candle_not_found() -> None:
    df = _synthetic_ohlcv(n=100)
    vol_df = build_volatility_dataframe(df)
    signal_trade = {
        "trade_number": 1,
        "pair": "BTC/USDC:USDC",
        "entry_time": "2099-01-01 00:00:00+00:00",
    }

    context = reconstruct_volatility_context(signal_trade, vol_df)

    assert context.candle_matched is False
    assert context.atr_pct is None


def test_reconstruct_all_volatility_contexts_multiple_trades() -> None:
    df = _synthetic_ohlcv(n=100)
    vol_df = build_volatility_dataframe(df)
    rows = [vol_df.iloc[i] for i in (70, 75, 80)]
    trades = [
        {"trade_number": i + 1, "pair": "BTC/USDC:USDC", "entry_time": str(r["date"])}
        for i, r in enumerate(rows)
    ]

    contexts = reconstruct_all_volatility_contexts(trades, {"BTC/USDC:USDC": vol_df})

    assert len(contexts) == 3
    assert all(c.candle_matched for c in contexts)


# ---------------------------------------------------------------------------
# 9. trade-entry volatility lookup: grouping/summary functions
# ---------------------------------------------------------------------------


def _ctx(**overrides) -> VolatilityEntryContext:
    base = dict(
        trade_number=1,
        pair="BTC/USDC:USDC",
        direction="LONG",
        entry_time="2026-02-01 00:00:00+00:00",
        entry_price=100.0,
        adx14=30.0,
        ema_distance_pct=0.05,
        atr_pct=2.0,
        realized_vol=0.01,
        exit_reason="exit_signal",
        profit_pct=5.0,
        duration_minutes=1000,
        is_winner=True,
        candle_matched=True,
    )
    base.update(overrides)
    return VolatilityEntryContext(**base)


def test_summarize_entry_volatility_by_pair() -> None:
    contexts = [
        _ctx(pair="BTC/USDC:USDC", atr_pct=2.0),
        _ctx(pair="BTC/USDC:USDC", atr_pct=4.0),
        _ctx(pair="SOL/USDC:USDC", atr_pct=6.0),
    ]

    result = summarize_entry_volatility_by_pair(contexts)

    assert result["BTC/USDC:USDC"]["trade_count"] == 2
    assert result["BTC/USDC:USDC"]["mean_atr_pct"] == pytest.approx(3.0)
    assert result["SOL/USDC:USDC"]["trade_count"] == 1


def test_summarize_stop_loss_vs_exit_signal() -> None:
    contexts = [
        _ctx(exit_reason="stop_loss", atr_pct=5.0),
        _ctx(exit_reason="stop_loss", atr_pct=7.0),
        _ctx(exit_reason="exit_signal", atr_pct=2.0),
        _ctx(exit_reason="force_exit", atr_pct=1.0),
    ]

    result = summarize_stop_loss_vs_exit_signal(contexts)

    assert result["stop_loss"]["trade_count"] == 2
    assert result["stop_loss"]["mean_atr_pct"] == pytest.approx(6.0)
    assert result["exit_signal"]["trade_count"] == 1
    assert result["force_exit"]["trade_count"] == 1


def test_summarize_winner_vs_loser_and_exit_signal_sign_split() -> None:
    contexts = [
        _ctx(is_winner=True, exit_reason="exit_signal", profit_pct=5.0),
        _ctx(is_winner=False, exit_reason="exit_signal", profit_pct=-3.0),
        _ctx(is_winner=False, exit_reason="stop_loss", profit_pct=-5.0),
    ]

    result = summarize_winner_vs_loser(contexts)

    assert result["winners"]["trade_count"] == 1
    assert result["losers"]["trade_count"] == 2
    assert result["stop_loss"]["trade_count"] == 1
    assert result["negative_exit_signal"]["trade_count"] == 1
    assert result["positive_exit_signal"]["trade_count"] == 1


def test_quartile_buckets_splits_into_four_groups() -> None:
    contexts = [_ctx(trade_number=i, atr_pct=float(i)) for i in range(1, 13)]

    buckets = quartile_buckets(contexts, metric="atr_pct")

    assert set(buckets.keys()) <= {"Q1", "Q2", "Q3", "Q4"}
    total = sum(b["trade_count"] for b in buckets.values())
    assert total == 12


def test_quartile_buckets_returns_empty_for_too_few_trades() -> None:
    contexts = [_ctx(atr_pct=1.0), _ctx(atr_pct=2.0)]
    assert quartile_buckets(contexts) == {}


# ---------------------------------------------------------------------------
# 10. stop-distance-in-ATR calculation
# ---------------------------------------------------------------------------


def test_stop_distance_in_atr_property() -> None:
    context = _ctx(atr_pct=2.5)
    assert context.stop_distance_in_atr == pytest.approx(5.0 / 2.5)


def test_stop_distance_in_atr_none_when_atr_missing() -> None:
    context = _ctx(atr_pct=None)
    assert context.stop_distance_in_atr is None


def test_stop_distance_in_atr_none_when_atr_is_zero() -> None:
    context = _ctx(atr_pct=0.0)
    assert context.stop_distance_in_atr is None


def test_summarize_stop_distance_in_atr_by_pair() -> None:
    contexts = [
        _ctx(pair="BTC/USDC:USDC", atr_pct=5.0),
        _ctx(pair="SOL/USDC:USDC", atr_pct=2.0),
    ]

    result = summarize_stop_distance_in_atr_by_pair(contexts)

    assert result["BTC/USDC:USDC"]["mean_atr_pct"] == pytest.approx(5.0)
    assert result["BTC/USDC:USDC"]["mean_stop_distance_in_atr"] == pytest.approx(1.0)
    assert result["SOL/USDC:USDC"]["mean_stop_distance_in_atr"] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Pair-level summary
# ---------------------------------------------------------------------------


def test_summarize_pair_volatility() -> None:
    df = _synthetic_ohlcv(n=100)
    vol_df = build_volatility_dataframe(df)

    summary = summarize_pair_volatility(vol_df)

    assert summary["candles"] == 100
    assert summary["atr_pct"]["mean"] is not None
    assert summary["realized_vol"]["mean"] is not None


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------


def test_pearson_correlation_perfect_positive() -> None:
    assert pearson_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_pearson_correlation_none_for_zero_variance() -> None:
    assert pearson_correlation([1, 1, 1], [1, 2, 3]) is None


def test_pearson_correlation_none_for_too_few_points() -> None:
    assert pearson_correlation([1], [2]) is None


def test_spearman_correlation_monotonic_nonlinear() -> None:
    x = [1, 2, 3, 4, 5]
    y = [1, 4, 9, 16, 25]  # nonlinear but perfectly monotonic
    assert spearman_correlation(x, y) == pytest.approx(1.0)
    assert pearson_correlation(x, y) < 1.0  # not perfectly linear


def test_compute_volatility_ema_correlations_reports_n() -> None:
    contexts = [
        _ctx(atr_pct=1.0, ema_distance_pct=0.01, realized_vol=0.01),
        _ctx(atr_pct=2.0, ema_distance_pct=0.02, realized_vol=0.02),
        _ctx(atr_pct=3.0, ema_distance_pct=0.03, realized_vol=0.03),
    ]

    result = compute_volatility_ema_correlations(contexts)

    assert result["atr_pct_vs_ema_distance"]["n"] == 3
    assert result["atr_pct_vs_ema_distance"]["pearson_r"] == pytest.approx(1.0)
    assert result["realized_vol_vs_ema_distance"]["n"] == 3


def test_compute_volatility_ema_correlations_excludes_missing_values() -> None:
    contexts = [
        _ctx(atr_pct=1.0, ema_distance_pct=0.01),
        _ctx(atr_pct=None, ema_distance_pct=0.02),
        _ctx(atr_pct=3.0, ema_distance_pct=None),
    ]

    result = compute_volatility_ema_correlations(contexts)

    assert result["atr_pct_vs_ema_distance"]["n"] == 1


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_volatility_dataset_round_trips(tmp_path: Path) -> None:
    dataset = {"metadata": {"strategy": "TrendFollowCore"}, "trades": [1, 2, 3]}
    output_path = tmp_path / "volatility_forensics.json"

    save_volatility_dataset(dataset, output_path)

    loaded = json.loads(output_path.read_text())
    assert loaded["metadata"]["strategy"] == "TrendFollowCore"
    assert loaded["trades"] == [1, 2, 3]
