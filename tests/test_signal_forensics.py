"""Unit tests for hermes.signal_forensics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.unit

from hermes.signal_forensics import (
    EntryContext,
    SignalForensicsError,
    audit_no_lookahead,
    build_forensic_dataset,
    compute_breakout_distance_pct,
    compute_ema_distance_pct,
    find_entry_candle,
    load_trendfollow_indicator_functions,
    reconcile,
    reconstruct_all,
    reconstruct_entry_context,
    save_forensic_dataset,
    summarize_by_direction,
    summarize_by_enter_tag,
    summarize_by_exit_reason,
    summarize_by_outcome,
    summarize_by_pair,
)
from hermes.trade_report import Trade, TradeReport

REPO_ROOT = Path(__file__).resolve().parents[1]
STRATEGY_PATH = REPO_ROOT / "user_data" / "strategies" / "TrendFollowCore.py"


def _synthetic_ohlcv(n: int = 260, *, seed: int = 7, trend: float = 0.15) -> pd.DataFrame:
    """A deterministic, gently trending synthetic 4h OHLCV series -- long
    enough for EMA200/ADX14/Donchian20 to all produce real (non-NaN)
    values well before the end, which is what these tests need without
    touching any real market data file."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    noise = rng.normal(0, 1.0, n)
    close = 100 + np.cumsum(trend + noise * 0.3)
    high = close + np.abs(rng.normal(0.5, 0.2, n))
    low = close - np.abs(rng.normal(0.5, 0.2, n))
    open_ = close - rng.normal(0, 0.2, n)
    volume = rng.uniform(100, 200, n)
    return pd.DataFrame(
        {"date": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


# ---------------------------------------------------------------------------
# load_trendfollow_indicator_functions
# ---------------------------------------------------------------------------


def test_load_trendfollow_indicator_functions_imports_real_strategy() -> None:
    compute_indicators = load_trendfollow_indicator_functions(STRATEGY_PATH)
    df = _synthetic_ohlcv()

    result = compute_indicators(df)

    for col in ("ema200", "adx", "donchian_upper_prev", "donchian_lower_prev"):
        assert col in result.columns
    # Warmup produces NaN early, real values well before the end.
    assert pd.notna(result.iloc[-1]["ema200"])


def test_load_trendfollow_indicator_functions_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(SignalForensicsError, match="failed to import"):
        load_trendfollow_indicator_functions(tmp_path / "does_not_exist.py")


def test_load_trendfollow_indicator_functions_missing_compute_indicators_raises(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "not_a_strategy.py"
    module_path.write_text("X = 1\n")

    with pytest.raises(SignalForensicsError, match="no compute_indicators"):
        load_trendfollow_indicator_functions(module_path)


# ---------------------------------------------------------------------------
# audit_no_lookahead
# ---------------------------------------------------------------------------


def test_audit_no_lookahead_passes_for_the_real_strategy_indicators() -> None:
    compute_indicators = load_trendfollow_indicator_functions(STRATEGY_PATH)
    df = _synthetic_ohlcv()

    assert audit_no_lookahead(df, compute_indicators, at_index=259) is True
    assert audit_no_lookahead(df, compute_indicators, at_index=230) is True


def test_audit_no_lookahead_detects_a_genuinely_leaky_indicator() -> None:
    """A deliberately lookahead-leaking stand-in must FAIL the audit --
    proving the check actually detects lookahead when it's present, not
    just restating the strategy's own no-lookahead claim."""

    def leaky_compute_indicators(dataframe: pd.DataFrame) -> pd.DataFrame:
        df = dataframe.copy()
        # Centered rolling window: uses future rows relative to each row's
        # own position -- exactly the kind of leak this audit exists to catch.
        df["ema200"] = df["close"].rolling(window=21, center=True, min_periods=1).mean()
        df["adx"] = 30.0
        df["donchian_upper_prev"] = df["high"].rolling(window=20, min_periods=1).max()
        df["donchian_lower_prev"] = df["low"].rolling(window=20, min_periods=1).min()
        return df

    df = _synthetic_ohlcv()

    assert audit_no_lookahead(df, leaky_compute_indicators, at_index=230) is False


# ---------------------------------------------------------------------------
# find_entry_candle
# ---------------------------------------------------------------------------


def test_find_entry_candle_matches_exact_timestamp() -> None:
    df = _synthetic_ohlcv(n=10)
    target = df.iloc[5]["date"]

    found = find_entry_candle(df, str(target))

    assert found is not None
    assert found["date"] == target


def test_find_entry_candle_handles_naive_timestamp_as_utc() -> None:
    df = _synthetic_ohlcv(n=10)
    target = df.iloc[3]["date"]
    naive_str = target.tz_localize(None).isoformat()

    found = find_entry_candle(df, naive_str)

    assert found is not None
    assert found["date"] == target


def test_find_entry_candle_returns_none_for_no_match() -> None:
    df = _synthetic_ohlcv(n=10)
    assert find_entry_candle(df, "2099-01-01 00:00:00+00:00") is None


def test_find_entry_candle_returns_none_for_none_entry_time() -> None:
    df = _synthetic_ohlcv(n=10)
    assert find_entry_candle(df, None) is None


def test_find_entry_candle_returns_none_for_malformed_entry_time() -> None:
    df = _synthetic_ohlcv(n=10)
    assert find_entry_candle(df, "not-a-timestamp-at-all") is None


# ---------------------------------------------------------------------------
# Distance calculations
# ---------------------------------------------------------------------------


def test_compute_ema_distance_pct_long() -> None:
    assert compute_ema_distance_pct(110.0, 100.0, "LONG") == pytest.approx(0.10)


def test_compute_ema_distance_pct_short() -> None:
    assert compute_ema_distance_pct(90.0, 100.0, "SHORT") == pytest.approx(0.10)


def test_compute_ema_distance_pct_missing_inputs_are_none() -> None:
    assert compute_ema_distance_pct(None, 100.0, "LONG") is None
    assert compute_ema_distance_pct(110.0, None, "LONG") is None
    assert compute_ema_distance_pct(110.0, 0.0, "LONG") is None
    assert compute_ema_distance_pct(110.0, 100.0, None) is None


def test_compute_breakout_distance_pct_long() -> None:
    assert compute_breakout_distance_pct(105.0, 100.0, 90.0, "LONG") == pytest.approx(0.05)


def test_compute_breakout_distance_pct_short() -> None:
    assert compute_breakout_distance_pct(85.0, 100.0, 90.0, "SHORT") == pytest.approx(5.0 / 90.0)


def test_compute_breakout_distance_pct_missing_inputs_are_none() -> None:
    assert compute_breakout_distance_pct(None, 100.0, 90.0, "LONG") is None
    assert compute_breakout_distance_pct(105.0, None, 90.0, "LONG") is None
    assert compute_breakout_distance_pct(85.0, 100.0, None, "SHORT") is None
    assert compute_breakout_distance_pct(105.0, 100.0, 90.0, None) is None


# ---------------------------------------------------------------------------
# reconstruct_entry_context / reconstruct_all / reconcile
# ---------------------------------------------------------------------------


def _trade(**overrides) -> Trade:
    base = dict(
        pair="BTC/USDC:USDC",
        direction="LONG",
        entry_time=None,
        exit_time="2026-02-01 00:00:00+00:00",
        entry_price=110.0,
        exit_price=115.0,
        enter_tag="trend_long_donchian_breakout",
        exit_reason="exit_signal",
        profit_abs=5.0,
        profit_pct=4.5,
        duration_minutes=1440,
        is_open=False,
    )
    base.update(overrides)
    return Trade(**base)


def test_reconstruct_entry_context_happy_path() -> None:
    compute_indicators = load_trendfollow_indicator_functions(STRATEGY_PATH)
    df = _synthetic_ohlcv()
    indicators_df = compute_indicators(df)
    entry_row = indicators_df.iloc[240]
    trade = _trade(entry_price=float(entry_row["close"]), entry_time=str(entry_row["date"]))

    context = reconstruct_entry_context(1, trade, indicators_df)

    assert context.candle_matched is True
    assert context.ema200 == pytest.approx(float(entry_row["ema200"]))
    assert context.adx14 == pytest.approx(float(entry_row["adx"]))
    assert context.donchian_upper_prev == pytest.approx(float(entry_row["donchian_upper_prev"]))
    assert context.ema_distance_pct is not None
    assert context.breakout_distance_pct is not None
    # Outcome fields come straight from the trade, untouched.
    assert context.exit_reason == "exit_signal"
    assert context.profit_abs == 5.0
    assert context.is_winner is True


def test_reconstruct_entry_context_no_data_for_pair() -> None:
    trade = _trade(entry_time="2026-02-01 00:00:00+00:00")

    context = reconstruct_entry_context(1, trade, None)

    assert context.candle_matched is False
    assert context.ema200 is None
    assert context.adx14 is None
    assert context.donchian_upper_prev is None
    assert context.ema_distance_pct is None
    assert context.breakout_distance_pct is None
    # Outcome fields are still preserved even without a matched candle.
    assert context.exit_reason == "exit_signal"


def test_reconstruct_entry_context_candle_not_found() -> None:
    compute_indicators = load_trendfollow_indicator_functions(STRATEGY_PATH)
    indicators_df = compute_indicators(_synthetic_ohlcv())
    trade = _trade(entry_time="2099-01-01 00:00:00+00:00")

    context = reconstruct_entry_context(1, trade, indicators_df)

    assert context.candle_matched is False
    assert context.ema200 is None


def test_reconstruct_entry_context_malformed_entry_time_does_not_raise() -> None:
    compute_indicators = load_trendfollow_indicator_functions(STRATEGY_PATH)
    indicators_df = compute_indicators(_synthetic_ohlcv())
    trade = _trade(entry_time="definitely-not-a-timestamp")

    context = reconstruct_entry_context(1, trade, indicators_df)

    assert context.candle_matched is False


def test_reconstruct_all_and_reconcile_full_match() -> None:
    compute_indicators = load_trendfollow_indicator_functions(STRATEGY_PATH)
    df = _synthetic_ohlcv()
    indicators_df = compute_indicators(df)
    rows = [indicators_df.iloc[i] for i in (230, 235, 240, 245, 250)]
    trades = tuple(
        _trade(entry_price=float(r["close"]), entry_time=str(r["date"])) for r in rows
    )
    report = TradeReport(trades=trades)

    contexts = reconstruct_all(report, {"BTC/USDC:USDC": indicators_df})
    result = reconcile(contexts, expected=5)

    assert result.matched == 5
    assert result.unmatched_trade_numbers == ()
    assert result.complete is True


def test_reconcile_reports_unmatched_trade_numbers() -> None:
    compute_indicators = load_trendfollow_indicator_functions(STRATEGY_PATH)
    indicators_df = compute_indicators(_synthetic_ohlcv())
    good_row = indicators_df.iloc[240]
    trades = (
        _trade(entry_price=float(good_row["close"]), entry_time=str(good_row["date"])),
        _trade(entry_time="2099-01-01 00:00:00+00:00"),  # no such candle
        _trade(pair="SOL/USDC:USDC", entry_time="2026-02-01 00:00:00+00:00"),  # no data at all
    )
    report = TradeReport(trades=trades)

    contexts = reconstruct_all(report, {"BTC/USDC:USDC": indicators_df})
    result = reconcile(contexts, expected=3)

    assert result.matched == 1
    assert result.unmatched_trade_numbers == (2, 3)
    assert result.complete is False


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _ctx(**overrides) -> EntryContext:
    base = dict(
        trade_number=1,
        pair="BTC/USDC:USDC",
        direction="LONG",
        entry_time="2026-02-01 00:00:00+00:00",
        entry_price=110.0,
        ema200=100.0,
        adx14=30.0,
        donchian_upper_prev=105.0,
        donchian_lower_prev=95.0,
        ema_distance_pct=0.10,
        breakout_distance_pct=0.05,
        enter_tag="trend_long_donchian_breakout",
        exit_reason="exit_signal",
        profit_abs=5.0,
        profit_pct=4.5,
        duration_minutes=1440,
        is_winner=True,
        candle_matched=True,
    )
    base.update(overrides)
    return EntryContext(**base)


def test_summarize_by_direction() -> None:
    contexts = [
        _ctx(direction="LONG", adx14=20.0, profit_abs=5.0, is_winner=True),
        _ctx(direction="LONG", adx14=30.0, profit_abs=-3.0, is_winner=False),
        _ctx(direction="SHORT", adx14=40.0, profit_abs=10.0, is_winner=True),
    ]

    result = summarize_by_direction(contexts)

    assert result["LONG"]["trade_count"] == 2
    assert result["LONG"]["average_adx"] == pytest.approx(25.0)
    assert result["LONG"]["win_rate_pct"] == pytest.approx(50.0)
    assert result["LONG"]["total_profit_abs"] == pytest.approx(2.0)
    assert result["SHORT"]["trade_count"] == 1
    assert result["SHORT"]["win_rate_pct"] == pytest.approx(100.0)


def test_summarize_by_outcome() -> None:
    contexts = [
        _ctx(is_winner=True, profit_pct=5.0),
        _ctx(is_winner=True, profit_pct=3.0),
        _ctx(is_winner=False, profit_pct=-2.0),
    ]

    result = summarize_by_outcome(contexts)

    assert result["WINNERS"]["trade_count"] == 2
    assert result["WINNERS"]["average_profit_pct"] == pytest.approx(4.0)
    assert result["LOSERS"]["trade_count"] == 1


def test_summarize_by_exit_reason() -> None:
    contexts = [
        _ctx(exit_reason="exit_signal"),
        _ctx(exit_reason="exit_signal"),
        _ctx(exit_reason="stop_loss"),
        _ctx(exit_reason=None),
    ]

    result = summarize_by_exit_reason(contexts)

    assert result["exit_signal"]["trade_count"] == 2
    assert result["stop_loss"]["trade_count"] == 1
    assert None not in result


def test_summarize_by_pair() -> None:
    contexts = [
        _ctx(pair="BTC/USDC:USDC"),
        _ctx(pair="BTC/USDC:USDC"),
        _ctx(pair="ETH/USDC:USDC"),
    ]

    result = summarize_by_pair(contexts)

    assert result["BTC/USDC:USDC"]["trade_count"] == 2
    assert result["ETH/USDC:USDC"]["trade_count"] == 1


def test_summarize_by_enter_tag() -> None:
    contexts = [
        _ctx(enter_tag="trend_long_donchian_breakout"),
        _ctx(enter_tag="trend_short_donchian_breakout"),
        _ctx(enter_tag="trend_short_donchian_breakout"),
    ]

    result = summarize_by_enter_tag(contexts)

    assert result["trend_long_donchian_breakout"]["trade_count"] == 1
    assert result["trend_short_donchian_breakout"]["trade_count"] == 2


def test_group_summary_handles_all_none_gracefully() -> None:
    contexts = [_ctx(adx14=None, ema_distance_pct=None, breakout_distance_pct=None, is_winner=None, profit_abs=None, profit_pct=None)]

    result = summarize_by_direction(contexts)

    assert result["LONG"]["average_adx"] is None
    assert result["LONG"]["win_rate_pct"] is None
    assert result["LONG"]["total_profit_abs"] is None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_build_and_save_forensic_dataset_round_trips(tmp_path: Path) -> None:
    contexts = [_ctx(trade_number=1), _ctx(trade_number=2, direction="SHORT")]
    dataset = build_forensic_dataset(
        contexts, strategy="TrendFollowCore", timeframe="4h", timerange="20260115-20260811"
    )
    output_path = tmp_path / "trend_signal_forensics.json"

    save_forensic_dataset(dataset, output_path)

    loaded = json.loads(output_path.read_text())
    assert loaded["strategy"] == "TrendFollowCore"
    assert len(loaded["trades"]) == 2
    assert loaded["summary"]["by_direction"]["LONG"]["trade_count"] == 1
    assert loaded["summary"]["by_direction"]["SHORT"]["trade_count"] == 1
