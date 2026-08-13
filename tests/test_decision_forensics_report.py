"""Unit tests for hermes.decision_forensics_report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.unit

from hermes.decision_forensics_report import (
    DATA_NOT_AVAILABLE,
    SAME_CANDLE_RESOLUTION,
    FATE_A_RECOVERED,
    FATE_D_LARGER_LOSS,
    FATE_E_NORMAL_EXIT,
    FATE_NO_MATCH,
    build_cross_width_dataset,
    build_decision_forensics_report,
    build_exit_walk_table,
    classify_width_change,
    combined_high_vol_high_ema_diagnostic,
    pair_stratified_diagnostic,
    reconstruct_entry_sequence,
    reconstruct_exit_decision,
    render_decision_forensics_report,
    resolve_exit_mechanism,
    save_decision_forensics_dataset,
    stop_distance_atr_buckets,
)
from hermes.stoploss_forensics import ExitSequenceResult
from hermes.trade_report import Trade, TradeReport
from hermes.volatility_forensics import VolatilityEntryContext


def _indicators_df(
    closes: list[float],
    *,
    ema200: float = 100.0,
    adx: float = 30.0,
    donchian_upper_prev: float = 95.0,
    donchian_lower_prev: float = 105.0,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> pd.DataFrame:
    n = len(closes)
    dates = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "close": closes,
            "high": highs if highs is not None else closes,
            "low": lows if lows is not None else closes,
            "ema200": [ema200] * n,
            "adx": [adx] * n,
            "donchian_upper_prev": [donchian_upper_prev] * n,
            "donchian_lower_prev": [donchian_lower_prev] * n,
        }
    )


# ---------------------------------------------------------------------------
# 1. signal-candle vs execution-candle alignment
# ---------------------------------------------------------------------------


def test_signal_candle_is_the_candle_before_execution() -> None:
    # Signal candle (index 1) satisfies LONG gates; execution/fill candle
    # (index 2, the recorded entry_time) is the *next* candle.
    df = _indicators_df([90.0, 96.0, 97.0, 98.0], ema200=90.0, donchian_upper_prev=95.0)
    entry_time = df.iloc[2]["date"]

    result = reconstruct_entry_sequence(
        df,
        trade_number=1,
        pair="BTC/USDC:USDC",
        direction="LONG",
        entry_time=entry_time,
        entry_price=97.5,
    )

    assert result.signal_candle_available is True
    assert result.signal_candle_time == str(df.iloc[1]["date"])
    assert result.execution_candle_time == str(df.iloc[2]["date"])
    assert result.entry_signal_emitted_on_signal_candle is True


def test_signal_candle_data_not_available_when_execution_is_first_candle() -> None:
    df = _indicators_df([96.0, 97.0], donchian_upper_prev=95.0)
    entry_time = df.iloc[0]["date"]

    result = reconstruct_entry_sequence(
        df, trade_number=2, pair="BTC/USDC:USDC", direction="LONG", entry_time=entry_time, entry_price=96.0
    )

    assert result.signal_candle_available is False
    assert result.signal_candle_time == DATA_NOT_AVAILABLE


def test_signal_candle_data_not_available_when_execution_candle_missing() -> None:
    df = _indicators_df([96.0, 97.0])
    result = reconstruct_entry_sequence(
        df, trade_number=3, pair="BTC/USDC:USDC", direction="LONG", entry_time="2099-01-01", entry_price=96.0
    )
    assert result.signal_candle_available is False


# ---------------------------------------------------------------------------
# 2. long entry gate reconstruction
# ---------------------------------------------------------------------------


def test_long_entry_gate_reconstruction_all_pass() -> None:
    df = _indicators_df([90.0, 106.0, 107.0], ema200=100.0, adx=30.0, donchian_upper_prev=105.0)
    entry_time = df.iloc[2]["date"]
    result = reconstruct_entry_sequence(
        df, trade_number=1, pair="BTC/USDC:USDC", direction="LONG", entry_time=entry_time, entry_price=107.0
    )
    assert result.ema_condition is True
    assert result.adx_condition is True
    assert result.donchian_condition is True
    assert result.entry_signal_emitted_on_signal_candle is True


def test_long_entry_gate_reconstruction_fails_on_donchian() -> None:
    df = _indicators_df([90.0, 101.0, 102.0], ema200=100.0, adx=30.0, donchian_upper_prev=105.0)
    entry_time = df.iloc[2]["date"]
    result = reconstruct_entry_sequence(
        df, trade_number=1, pair="BTC/USDC:USDC", direction="LONG", entry_time=entry_time, entry_price=102.0
    )
    assert result.donchian_condition is False
    assert result.entry_signal_emitted_on_signal_candle is False


# ---------------------------------------------------------------------------
# 3. short entry gate reconstruction
# ---------------------------------------------------------------------------


def test_short_entry_gate_reconstruction_all_pass() -> None:
    df = _indicators_df([110.0, 94.0, 93.0], ema200=100.0, adx=30.0, donchian_lower_prev=95.0)
    entry_time = df.iloc[2]["date"]
    result = reconstruct_entry_sequence(
        df, trade_number=2, pair="ETH/USDC:USDC", direction="SHORT", entry_time=entry_time, entry_price=93.0
    )
    assert result.ema_condition is True
    assert result.donchian_condition is True
    assert result.entry_signal_emitted_on_signal_candle is True


def test_short_entry_gate_reconstruction_fails_on_adx() -> None:
    df = _indicators_df([110.0, 94.0, 93.0], ema200=100.0, adx=10.0, donchian_lower_prev=95.0)
    entry_time = df.iloc[2]["date"]
    result = reconstruct_entry_sequence(
        df, trade_number=2, pair="ETH/USDC:USDC", direction="SHORT", entry_time=entry_time, entry_price=93.0
    )
    assert result.adx_condition is False
    assert result.entry_signal_emitted_on_signal_candle is False


# ---------------------------------------------------------------------------
# 4. EMA200 exit detection
# ---------------------------------------------------------------------------


def test_ema200_exit_detection_long() -> None:
    df = _indicators_df([100.0, 99.0, 98.0])
    result = reconstruct_exit_decision(
        df,
        trade_number=1,
        pair="BTC/USDC:USDC",
        direction="LONG",
        entry_index=0,
        entry_time=str(df.iloc[0]["date"]),
        entry_price=100.0,
        recorded_exit_reason="exit_signal",
        recorded_profit_pct=-1.0,
    )
    assert result.sequence.exit_signal_trigger_index == 1
    assert result.resolved_mechanism == "exit_signal"
    assert result.matches_recorded_exit_reason is True


# ---------------------------------------------------------------------------
# 5. stop-loss detection
# ---------------------------------------------------------------------------


def test_stop_loss_detection_long() -> None:
    df = _indicators_df([100.0, 101.0, 101.0], lows=[100.0, 101.0, 90.0])
    result = reconstruct_exit_decision(
        df,
        trade_number=2,
        pair="BTC/USDC:USDC",
        direction="LONG",
        entry_index=0,
        entry_time=str(df.iloc[0]["date"]),
        entry_price=100.0,
        recorded_exit_reason="stop_loss",
        recorded_profit_pct=-5.0,
    )
    assert result.sequence.stop_trigger_index == 2
    assert result.resolved_mechanism == "stop_loss"
    assert result.matches_recorded_exit_reason is True


# ---------------------------------------------------------------------------
# 6. same-candle stop/exit ordering
# ---------------------------------------------------------------------------


def test_same_candle_conflict_resolves_to_stop_loss() -> None:
    # Candle 1: low wicks to the stop level AND close crosses below ema200.
    df = _indicators_df([100.0, 94.0], lows=[100.0, 90.0])
    result = reconstruct_exit_decision(
        df,
        trade_number=3,
        pair="BTC/USDC:USDC",
        direction="LONG",
        entry_index=0,
        entry_time=str(df.iloc[0]["date"]),
        entry_price=100.0,
        recorded_exit_reason="stop_loss",
        recorded_profit_pct=-5.0,
    )
    assert result.sequence.which_first == "same_candle"
    assert result.same_candle_conflict is True
    assert result.resolved_mechanism == "stop_loss"
    assert resolve_exit_mechanism(
        ExitSequenceResult(0, "t", 0, "t", "same_candle")
    ) == "stop_loss"
    assert SAME_CANDLE_RESOLUTION  # documented, non-empty


# ---------------------------------------------------------------------------
# 7. force-exit boundary handling
# ---------------------------------------------------------------------------


def test_force_exit_neither_mechanism_fires_within_window() -> None:
    df = _indicators_df([100.0, 100.5, 101.0])
    result = reconstruct_exit_decision(
        df,
        trade_number=4,
        pair="SOL/USDC:USDC",
        direction="LONG",
        entry_index=0,
        entry_time=str(df.iloc[0]["date"]),
        entry_price=100.0,
        recorded_exit_reason="force_exit",
        recorded_profit_pct=1.0,
    )
    assert result.sequence.which_first == "neither_within_window"
    assert result.resolved_mechanism == "neither_within_window"
    # force_exit is not one of the two mechanisms this module resolves --
    # matches_recorded_exit_reason stays None rather than False.
    assert result.matches_recorded_exit_reason is None


# ---------------------------------------------------------------------------
# 8. stop-distance-in-ATR calculation
# ---------------------------------------------------------------------------


def _vctx(trade_number, pair, atr_pct, exit_reason="exit_signal", ema_distance_pct=1.0, profit_pct=1.0):
    return VolatilityEntryContext(
        trade_number=trade_number,
        pair=pair,
        direction="LONG",
        entry_time="t",
        entry_price=100.0,
        adx14=30.0,
        ema_distance_pct=ema_distance_pct,
        atr_pct=atr_pct,
        realized_vol=0.01,
        exit_reason=exit_reason,
        profit_pct=profit_pct,
        duration_minutes=100.0,
        is_winner=profit_pct > 0,
        candle_matched=True,
    )


def test_stop_distance_in_atr_is_five_over_entry_atr_pct() -> None:
    ctx = _vctx(1, "BTC/USDC:USDC", atr_pct=2.0)
    assert ctx.stop_distance_in_atr == pytest.approx(2.5)


def test_stop_distance_atr_buckets_quartiles() -> None:
    contexts = [_vctx(i, "BTC/USDC:USDC", atr_pct=float(atr)) for i, atr in enumerate([1, 2, 3, 4, 5, 6, 7, 8], start=1)]
    buckets = stop_distance_atr_buckets(contexts)
    assert set(buckets.keys()) <= {"Q1", "Q2", "Q3", "Q4"}
    total = sum(b["trade_count"] for b in buckets.values())
    assert total == 8


def test_stop_distance_atr_buckets_empty_when_insufficient_data() -> None:
    contexts = [_vctx(1, "BTC/USDC:USDC", atr_pct=2.0)]
    assert stop_distance_atr_buckets(contexts) == {}


# ---------------------------------------------------------------------------
# 9. unmatched downstream trade handling (cross-width matching)
# ---------------------------------------------------------------------------


def _trade(pair, direction, entry_time, exit_reason, profit_pct):
    return Trade(
        pair=pair,
        direction=direction,
        entry_time=entry_time,
        exit_time=None,
        entry_price=100.0,
        exit_price=None,
        enter_tag=None,
        exit_reason=exit_reason,
        profit_abs=profit_pct,
        profit_pct=profit_pct,
        duration_minutes=100.0,
        is_open=False,
    )


def test_cross_width_no_match_when_downstream_sequencing_changed() -> None:
    baseline = TradeReport(trades=(_trade("BTC/USDC:USDC", "LONG", "2026-01-01T00:00:00", "stop_loss", -5.0),))
    widened = TradeReport(trades=())  # trade doesn't exist at this width -- sequencing changed

    rows = build_cross_width_dataset(baseline, {"-6": widened})
    assert rows[0].by_width["-6"]["classification"] == FATE_NO_MATCH


def test_cross_width_recovered_into_profit() -> None:
    baseline = TradeReport(trades=(_trade("BTC/USDC:USDC", "LONG", "2026-01-01T00:00:00", "stop_loss", -5.0),))
    widened = TradeReport(trades=(_trade("BTC/USDC:USDC", "LONG", "2026-01-01T00:00:00", "exit_signal", 3.0),))

    rows = build_cross_width_dataset(baseline, {"-6": widened})
    assert rows[0].by_width["-6"]["classification"] == FATE_A_RECOVERED


def test_cross_width_larger_loss() -> None:
    baseline = TradeReport(trades=(_trade("BTC/USDC:USDC", "LONG", "2026-01-01T00:00:00", "stop_loss", -5.0),))
    widened = TradeReport(trades=(_trade("BTC/USDC:USDC", "LONG", "2026-01-01T00:00:00", "stop_loss", -8.0),))

    rows = build_cross_width_dataset(baseline, {"-8": widened})
    assert rows[0].by_width["-8"]["classification"] == FATE_D_LARGER_LOSS
    assert classify_width_change(baseline.trades[0], widened.trades[0]) == FATE_D_LARGER_LOSS


def test_cross_width_normal_exit_still_negative() -> None:
    baseline = TradeReport(trades=(_trade("BTC/USDC:USDC", "LONG", "2026-01-01T00:00:00", "stop_loss", -5.0),))
    widened = TradeReport(trades=(_trade("BTC/USDC:USDC", "LONG", "2026-01-01T00:00:00", "exit_signal", -1.0),))

    rows = build_cross_width_dataset(baseline, {"-7": widened})
    assert rows[0].by_width["-7"]["classification"] == FATE_E_NORMAL_EXIT


# ---------------------------------------------------------------------------
# 10. deterministic report generation
# ---------------------------------------------------------------------------


def _minimal_dataset() -> dict:
    return build_decision_forensics_report(
        window="2026-01-15 to 2026-08-11",
        entry_sequences=[],
        exit_forensics=[],
        cross_width_rows=[],
        atr_buckets={},
        pair_diagnostic={},
        combined_diagnostic={"data_available": False},
        pair_comparison={},
        long_short_observation={
            "LONG": {"trades": 19, "winners": 2, "total_profit": -162.10},
            "SHORT": {"trades": 20, "winners": 6, "total_profit": 185.00},
        },
        data_quality={},
        entry_reconstruction_status="RESOLVED",
        exit_reconstruction_status="RESOLVED",
        final_verdict="PARTIALLY SUPPORTED",
        evidence_ranking=["item 1", "item 2"],
    )


def test_build_decision_forensics_report_is_deterministic() -> None:
    d1 = _minimal_dataset()
    d2 = _minimal_dataset()
    assert render_decision_forensics_report(d1) == render_decision_forensics_report(d2)


def test_render_decision_forensics_report_contains_all_ten_sections_and_status_block() -> None:
    text = render_decision_forensics_report(_minimal_dataset())

    for section in [
        "1. ENTRY DECISION FORENSICS",
        "2. EXIT DECISION FORENSICS",
        "3. STOP-WIDTH CROSS-SECTION",
        "4. STOP DISTANCE IN ATR",
        "5. VOLATILITY VS EMA-DISTANCE DIAGNOSTICS",
        "6. PAIR COMPARISON",
        "7. LONG/SHORT OBSERVATION",
        "8. DATA QUALITY / UNRESOLVED ITEMS",
        "9. EVIDENCE RANKING",
        "10. FINAL VERDICT",
    ]:
        assert section in text

    assert "[DECISION FORENSICS]" in text
    assert "STATUS:" in text
    assert "ENTRY RECONSTRUCTION: RESOLVED" in text
    assert "EXIT RECONSTRUCTION: RESOLVED" in text
    assert "STOP WIDTH FORENSICS: PASS" in text
    assert "STRATEGY MODIFIED: NO" in text
    assert "HYPEROPT: NO" in text
    assert "DEPLOYMENT: NO" in text
    assert "FINAL RECOMMENDATION: RESEARCH ONLY -- NO PRODUCTION CHANGE" in text


def test_save_decision_forensics_dataset_round_trips(tmp_path: Path) -> None:
    dataset = _minimal_dataset()
    output_path = tmp_path / "decision_forensics.json"
    save_decision_forensics_dataset(dataset, output_path)

    loaded = json.loads(output_path.read_text())
    assert loaded["window"] == "2026-01-15 to 2026-08-11"
    assert loaded["final_verdict"] == "PARTIALLY SUPPORTED"


# ---------------------------------------------------------------------------
# extra coverage: exit walk table, pair-stratified & combined diagnostics
# ---------------------------------------------------------------------------


def test_build_exit_walk_table_stops_at_first_trigger() -> None:
    df = _indicators_df([100.0, 99.0, 98.0, 97.0])
    walk = build_exit_walk_table(df, 0, "LONG", 100.0, stoploss_pct=5.0)
    # exit_signal fires at offset 1 (close=99 < ema200=100) -- table should
    # not extend past that trigger.
    assert walk[-1]["candle_offset"] == 1
    assert walk[-1]["ema_exit_condition"] is True


def test_pair_stratified_diagnostic_splits_by_pair() -> None:
    contexts = [
        _vctx(1, "BTC/USDC:USDC", atr_pct=1.0, exit_reason="stop_loss", ema_distance_pct=7.0, profit_pct=-5.0),
        _vctx(2, "BTC/USDC:USDC", atr_pct=1.0, exit_reason="exit_signal", ema_distance_pct=2.0, profit_pct=1.0),
        _vctx(3, "ETH/USDC:USDC", atr_pct=2.0, exit_reason="stop_loss", ema_distance_pct=6.0, profit_pct=-5.0),
    ]
    diag = pair_stratified_diagnostic(contexts)
    assert set(diag.keys()) == {"BTC/USDC:USDC", "ETH/USDC:USDC"}
    assert diag["BTC/USDC:USDC"]["stop_loss_count"] == 1
    assert diag["BTC/USDC:USDC"]["mean_ema_distance_stop_loss"] == pytest.approx(7.0)


def test_combined_high_vol_high_ema_diagnostic_reports_rates() -> None:
    contexts = [
        _vctx(1, "BTC/USDC:USDC", atr_pct=1.0, exit_reason="exit_signal", ema_distance_pct=1.0, profit_pct=1.0),
        _vctx(2, "BTC/USDC:USDC", atr_pct=2.0, exit_reason="exit_signal", ema_distance_pct=2.0, profit_pct=1.0),
        _vctx(3, "BTC/USDC:USDC", atr_pct=5.0, exit_reason="stop_loss", ema_distance_pct=8.0, profit_pct=-5.0),
        _vctx(4, "BTC/USDC:USDC", atr_pct=6.0, exit_reason="stop_loss", ema_distance_pct=9.0, profit_pct=-5.0),
    ]
    result = combined_high_vol_high_ema_diagnostic(contexts)
    assert result["data_available"] is True
    assert result["stop_loss_rate_pct_high_both"] == 100.0
