"""Unit tests for hermes.stoploss_forensics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.unit

from hermes.stoploss_forensics import (
    FATE_DELAYED_SAME_LOSS,
    FATE_LARGER_LOSS,
    FATE_NORMAL_EXIT,
    FATE_RECOVERED_INTO_PROFIT,
    FATE_REDUCED_LOSS,
    FATE_STILL_OPEN,
    FATE_UNKNOWN,
    build_stoploss_overlay_config,
    classify_stop_trade_fate,
    counterfactual_recovery_after_stop,
    explain_entry_gates,
    reconstruct_exit_sequence,
    render_entry_explanation,
    save_stoploss_forensics_dataset,
    save_stoploss_overlay_config,
)


def _candle(
    *, close=110.0, ema200=100.0, adx=30.0, donchian_upper_prev=105.0, donchian_lower_prev=95.0
) -> pd.Series:
    return pd.Series(
        {
            "close": close,
            "ema200": ema200,
            "adx": adx,
            "donchian_upper_prev": donchian_upper_prev,
            "donchian_lower_prev": donchian_lower_prev,
        }
    )


def _walk_df(
    closes: list[float],
    *,
    ema200: float = 100.0,
    n_pad_ema_none: int = 0,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> pd.DataFrame:
    """Build a minimal OHLCV+ema200 dataframe for exit-sequence walk tests.
    high/low default to close unless explicitly overridden (e.g. to model
    an intracandle wick that doesn't affect the candle's own close)."""
    n = len(closes)
    dates = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    ema_col = [None] * n_pad_ema_none + [ema200] * (n - n_pad_ema_none)
    return pd.DataFrame(
        {
            "date": dates,
            "close": closes,
            "high": highs if highs is not None else closes,
            "low": lows if lows is not None else closes,
            "ema200": ema_col,
        }
    )


# ---------------------------------------------------------------------------
# Config overlay
# ---------------------------------------------------------------------------


def test_build_stoploss_overlay_config_converts_positive_pct_to_negative_fraction() -> None:
    assert build_stoploss_overlay_config(6.0) == {"stoploss": -0.06}
    assert build_stoploss_overlay_config(10.0) == {"stoploss": -0.10}


def test_save_stoploss_overlay_config_writes_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "overlay.json"
    save_stoploss_overlay_config(7.0, path)

    loaded = json.loads(path.read_text())
    assert loaded == {"stoploss": -0.07}


# ---------------------------------------------------------------------------
# Entry-gate reconstruction
# ---------------------------------------------------------------------------


def test_explain_entry_gates_long_all_pass() -> None:
    candle = _candle(close=110.0, ema200=100.0, adx=30.0, donchian_upper_prev=105.0)
    result = explain_entry_gates(candle, "LONG")

    assert result.indicators_valid is True
    assert result.close_above_ema200 is True
    assert result.adx_above_threshold is True
    assert result.close_above_donchian_upper_prev is True
    assert result.entry_fired is True


def test_explain_entry_gates_long_fails_on_adx() -> None:
    candle = _candle(close=110.0, ema200=100.0, adx=20.0, donchian_upper_prev=105.0)
    result = explain_entry_gates(candle, "LONG")

    assert result.adx_above_threshold is False
    assert result.entry_fired is False


def test_explain_entry_gates_short_all_pass() -> None:
    candle = _candle(close=90.0, ema200=100.0, adx=30.0, donchian_lower_prev=95.0)
    result = explain_entry_gates(candle, "SHORT")

    assert result.close_below_ema200 is True
    assert result.close_below_donchian_lower_prev is True
    assert result.entry_fired is True


def test_explain_entry_gates_marks_irrelevant_gates_for_direction() -> None:
    long_result = explain_entry_gates(_candle(), "LONG")
    assert "close_below_ema200" in long_result.irrelevant_gates

    short_result = explain_entry_gates(_candle(), "SHORT")
    assert "close_above_ema200" in short_result.irrelevant_gates


def test_explain_entry_gates_unresolved_when_indicators_missing() -> None:
    candle = pd.Series({"close": 110.0, "ema200": None, "adx": None,
                         "donchian_upper_prev": None, "donchian_lower_prev": None})
    result = explain_entry_gates(candle, "LONG")

    assert result.indicators_valid is False
    assert result.entry_fired is False
    assert result.close_above_ema200 is None


def test_render_entry_explanation_reports_unresolved() -> None:
    candle = pd.Series({"close": 110.0, "ema200": None, "adx": None,
                         "donchian_upper_prev": None, "donchian_lower_prev": None})
    gates = explain_entry_gates(candle, "LONG")
    text = render_entry_explanation(1, "BTC/USDC:USDC", "LONG", "2026-01-01", gates)

    assert "UNRESOLVED -- DATA NOT AVAILABLE" in text


def test_render_entry_explanation_lists_gates_for_long() -> None:
    candle = _candle()
    gates = explain_entry_gates(candle, "LONG")
    text = render_entry_explanation(5, "BTC/USDC:USDC", "LONG", "2026-01-01", gates)

    assert "Trade #5 entered LONG" in text
    assert "ADX14 > 25" in text


# ---------------------------------------------------------------------------
# Exit-sequence forensics
# ---------------------------------------------------------------------------


def test_exit_sequence_stop_fires_before_exit_signal_long() -> None:
    # Entry at index 0, price=100, ema200 constant at 100. Closes stay >= 100
    # (so the EMA-cross exit condition never fires), but candle 2's LOW wicks
    # down to 90 -- an intracandle stop hit at -5% (level=95) while the
    # candle's own close (101) never crosses below EMA200.
    closes = [100.0, 101.0, 101.0, 103.0]
    lows = [100.0, 101.0, 90.0, 103.0]
    df = _walk_df(closes, lows=lows)
    result = reconstruct_exit_sequence(df, entry_index=0, direction="LONG", entry_price=100.0, stoploss_pct=5.0)

    assert result.stop_trigger_index == 2
    assert result.which_first == "stop_loss"


def test_exit_sequence_exit_signal_fires_before_stop_long() -> None:
    # close crosses below ema200 (100) at index 1 (close=99), well before any
    # low ever reaches the -5% stop level of 95.
    df = _walk_df([100.0, 99.0, 98.0, 97.0])
    result = reconstruct_exit_sequence(df, entry_index=0, direction="LONG", entry_price=100.0, stoploss_pct=5.0)

    assert result.exit_signal_trigger_index == 1
    assert result.which_first == "exit_signal"


def test_exit_sequence_neither_fires_within_window() -> None:
    df = _walk_df([100.0, 100.5, 101.0, 101.5])
    result = reconstruct_exit_sequence(df, entry_index=0, direction="LONG", entry_price=100.0, stoploss_pct=5.0)

    assert result.stop_trigger_index is None
    assert result.exit_signal_trigger_index is None
    assert result.which_first == "neither_within_window"


def test_exit_sequence_short_direction() -> None:
    # SHORT entry at 100, ema200 constant at 100. Closes stay <= 100 (so the
    # EMA-cross exit condition never fires), but candle 2's HIGH wicks up to
    # 106 -- an intracandle stop hit at +5% (level=105).
    closes = [100.0, 99.0, 99.0, 97.0]
    highs = [100.0, 99.0, 106.0, 97.0]
    df = _walk_df(closes, highs=highs)
    result = reconstruct_exit_sequence(df, entry_index=0, direction="SHORT", entry_price=100.0, stoploss_pct=5.0)

    assert result.stop_trigger_index == 2
    assert result.which_first == "stop_loss"


def test_exit_sequence_never_looks_before_entry_index() -> None:
    # A stop-triggering low exists at index 0, but entry_index=2 -- must be ignored.
    df = _walk_df([50.0, 50.0, 100.0, 101.0])
    result = reconstruct_exit_sequence(df, entry_index=2, direction="LONG", entry_price=100.0, stoploss_pct=5.0)

    assert result.stop_trigger_index is None


# ---------------------------------------------------------------------------
# Counterfactual recovery
# ---------------------------------------------------------------------------


def test_counterfactual_recovery_none_when_stop_never_triggers() -> None:
    df = _walk_df([100.0, 101.0, 102.0])
    result = counterfactual_recovery_after_stop(df, entry_index=0, direction="LONG", entry_price=100.0)

    assert result.stop_trigger_index is None
    assert result.recovered_above_entry is None


def test_counterfactual_recovery_price_recovers_long() -> None:
    # Stop hit at index 2 (low=90). Afterward price recovers to 110 at index 4.
    df = _walk_df([100.0, 96.0, 90.0, 95.0, 110.0])
    result = counterfactual_recovery_after_stop(df, entry_index=0, direction="LONG", entry_price=100.0, stoploss_pct=5.0)

    assert result.stop_trigger_index == 2
    assert result.recovered_above_entry is True
    assert result.reached_profitability is True
    assert result.favorable_pct == pytest.approx((110.0 - 100.0) / 100.0 * 100.0)


def test_counterfactual_recovery_price_does_not_recover_long() -> None:
    df = _walk_df([100.0, 96.0, 90.0, 85.0, 80.0])
    result = counterfactual_recovery_after_stop(df, entry_index=0, direction="LONG", entry_price=100.0, stoploss_pct=5.0)

    assert result.stop_trigger_index == 2
    assert result.recovered_above_entry is False
    assert result.adverse_pct == pytest.approx((100.0 - 80.0) / 100.0 * 100.0)


def test_counterfactual_recovery_still_open_when_no_data_after_stop() -> None:
    df = _walk_df([100.0, 96.0, 90.0])
    result = counterfactual_recovery_after_stop(df, entry_index=0, direction="LONG", entry_price=100.0, stoploss_pct=5.0)

    assert result.stop_trigger_index == 2
    assert result.still_open_at_window_end is True
    assert result.favorable_pct == 0.0


def test_counterfactual_recovery_short_direction() -> None:
    # SHORT: stop at +5% => 105, hit at index 2 (high=106). Afterward price
    # falls to 90 at index 4 -- favorable for a short.
    df = _walk_df([100.0, 103.0, 106.0, 100.0, 90.0])
    result = counterfactual_recovery_after_stop(df, entry_index=0, direction="SHORT", entry_price=100.0, stoploss_pct=5.0)

    assert result.stop_trigger_index == 2
    assert result.recovered_above_entry is True  # "recovered" means price moved favorably again
    assert result.favorable_pct == pytest.approx((100.0 - 90.0) / 100.0 * 100.0)


# ---------------------------------------------------------------------------
# Fate classification
# ---------------------------------------------------------------------------


def test_classify_fate_recovered_into_profit() -> None:
    fate = classify_stop_trade_fate(-5.0, "exit_signal", 3.5)
    assert fate == FATE_RECOVERED_INTO_PROFIT


def test_classify_fate_normal_exit_still_negative() -> None:
    fate = classify_stop_trade_fate(-5.0, "exit_signal", -1.0)
    assert fate == FATE_NORMAL_EXIT


def test_classify_fate_reduced_loss() -> None:
    fate = classify_stop_trade_fate(-5.0, "stop_loss", -6.5)  # wider stop, smaller pct loss than baseline? see below
    # Note: baseline is -5.0 (at -5% stop); a WIDER stop hitting at e.g. -6.5%
    # is actually a LARGER loss, not reduced -- use a case where test loss is
    # smaller in magnitude than baseline instead.
    fate2 = classify_stop_trade_fate(-7.0, "stop_loss", -6.0)
    assert fate2 == FATE_REDUCED_LOSS


def test_classify_fate_delayed_same_loss() -> None:
    fate = classify_stop_trade_fate(-5.0, "stop_loss", -5.0)
    assert fate == FATE_DELAYED_SAME_LOSS


def test_classify_fate_larger_loss() -> None:
    fate = classify_stop_trade_fate(-5.0, "stop_loss", -8.0)
    assert fate == FATE_LARGER_LOSS


def test_classify_fate_still_open_on_force_exit() -> None:
    fate = classify_stop_trade_fate(-5.0, "force_exit", -3.0)
    assert fate == FATE_STILL_OPEN


def test_classify_fate_unknown_on_missing_data() -> None:
    assert classify_stop_trade_fate(None, "stop_loss", -5.0) == FATE_UNKNOWN
    assert classify_stop_trade_fate(-5.0, None, -5.0) == FATE_UNKNOWN
    assert classify_stop_trade_fate(-5.0, "stop_loss", None) == FATE_UNKNOWN


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_stoploss_forensics_dataset_round_trips(tmp_path: Path) -> None:
    dataset = {"metadata": {"strategy": "TrendFollowCore"}, "tests": [5, 6, 7]}
    output_path = tmp_path / "stoploss_forensics.json"

    save_stoploss_forensics_dataset(dataset, output_path)

    loaded = json.loads(output_path.read_text())
    assert loaded["metadata"]["strategy"] == "TrendFollowCore"
    assert loaded["tests"] == [5, 6, 7]
