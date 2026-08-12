"""Mechanical entry/exit forensics and stop-loss sensitivity, for TrendFollowCore.

Companion to `hermes.signal_forensics` (entry indicator context) and
`hermes.trade_report` (recorded trade outcomes). This module adds two
things neither of those provide:

1. **Mechanical entry-gate reconstruction** -- for a given entry candle,
   which of `TrendFollowCore.compute_entry_signals`'s own conditions were
   true, computed by calling that function directly (never re-derived),
   so "why did this trade enter" is answered from the strategy's actual
   code, not from correlation with the outcome.
2. **Exit-sequence forensics** -- walking forward candle-by-candle from a
   trade's entry to determine which of two events happened first: the
   fixed stop-loss threshold being crossed, or `TrendFollowCore`'s own
   `compute_exit_signals` condition (the EMA200 cross) becoming true.
   This is the only way to honestly answer "did the stop fire before the
   strategy's own exit logic would have," rather than assuming it from
   the recorded `exit_reason` alone.

Also provides the config-overlay mechanism the stop-loss sensitivity
experiment uses: a tiny, temporary JSON file containing only a `stoploss`
key, passed as a *second* `-c` file alongside the frozen research config
(Freqtrade's own config-merge behavior — later files override earlier
keys). This means the six-stop-width experiment never touches
`TrendFollowCore.py`, `config-research-trendfollow.json`, or any other
existing file: the strategy's `stoploss` class attribute is overridden
the same way a caller would override any other config value, and nothing
about entry/exit logic, indicators, or pairs changes between runs.

Nothing here launches a subprocess directly -- the six backtests
themselves still go through `hermes.backtest.BacktestLauncher`, exactly
like every other backtest this session's tooling has run. This module
only supplies the config overlay and the post-hoc forensic reading over
already-downloaded OHLCV and already-recorded trade exports.

Design decisions
-----------------
* **Entry-gate reconstruction calls `TrendFollowCore.compute_entry_signals`
  itself** (via `hermes.signal_forensics.load_trendfollow_indicator_functions`
  style dynamic import), not a hand-copied condition -- exactly the same
  "reuse the real function" principle `hermes.signal_forensics` already
  established, so this module cannot silently drift from what the
  strategy actually checks.
* **Exit-sequence forensics never assumes which mechanism fired first.**
  `reconstruct_exit_sequence` independently finds the first candle where
  the stop threshold is crossed (`low`/`high` breaching the fill-price-
  relative stop level) and the first candle where the strategy's own
  `compute_exit_signals` condition is true, then compares indices -- the
  answer falls out of the comparison, it isn't asserted.
* **Counterfactual recovery is explicitly hypothetical and separately
  labeled**, per this module's build-block instruction not to call it
  MFE/MAE without the calculation supporting that name: `favorable_pct`/
  `adverse_pct` are the largest close-relative favorable/adverse
  percentage moves observed in the counterfactual post-original-stop
  window, which *is* what MFE/MAE conventionally mean, but this module
  keeps its own descriptive field names rather than asserting the
  standard label, leaving that characterization to the report/caller.
* **The stop-loss overlay is one key, one temporary file, never committed.**
  `build_stoploss_overlay_config` returns a plain dict; callers write it
  wherever a caller-scoped temp directory is appropriate (a Railway
  script uses an ephemeral path, since the overlay's only job is to be
  read once as Freqtrade's `-c` argument during a single backtest
  process, not to persist).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from hermes.signal_forensics import find_entry_candle

DEFAULT_STOP_LOSS_PCT = 5.0


# ---------------------------------------------------------------------------
# Config overlay (no strategy or existing config file touched)
# ---------------------------------------------------------------------------


def build_stoploss_overlay_config(stoploss_pct: float) -> dict[str, float]:
    """A minimal Freqtrade config overlay setting only `stoploss`.

    `stoploss_pct` is a positive percentage (e.g. `6.0` for a -6% stop);
    Freqtrade's own config schema expects the negative fraction.
    """
    return {"stoploss": -stoploss_pct / 100.0}


def save_stoploss_overlay_config(stoploss_pct: float, path: Path) -> Path:
    """Write `build_stoploss_overlay_config(stoploss_pct)` as JSON to `path`."""
    path = Path(path)
    path.write_text(json.dumps(build_stoploss_overlay_config(stoploss_pct), indent=2))
    return path


# ---------------------------------------------------------------------------
# Part A: mechanical entry-gate reconstruction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntryGateExplanation:
    """Which of `TrendFollowCore`'s own entry gates were true at one candle,
    for one direction -- computed by calling the strategy's real function."""

    direction: str
    indicators_valid: bool
    close_above_ema200: bool | None
    close_below_ema200: bool | None
    adx_above_threshold: bool | None
    close_above_donchian_upper_prev: bool | None
    close_below_donchian_lower_prev: bool | None
    entry_fired: bool
    irrelevant_gates: tuple[str, ...]


def explain_entry_gates(
    candle: pd.Series, direction: str, *, adx_threshold: float = 25.0
) -> EntryGateExplanation:
    """Mechanically explain why `direction`'s entry gate did/didn't fire at `candle`.

    `candle` must already carry `TrendFollowCore.compute_indicators`'s
    columns (`ema200`, `adx`, `donchian_upper_prev`, `donchian_lower_prev`)
    plus `close`. Every boolean is computed directly from those values --
    nothing is inferred from the trade's eventual outcome.
    """
    required = ("ema200", "adx", "donchian_upper_prev", "donchian_lower_prev")
    valid = all(pd.notna(candle.get(col)) for col in required)

    close_above_ema = close_below_ema = None
    adx_ok = None
    above_donchian = below_donchian = None
    if valid:
        close_above_ema = bool(candle["close"] > candle["ema200"])
        close_below_ema = bool(candle["close"] < candle["ema200"])
        adx_ok = bool(candle["adx"] > adx_threshold)
        above_donchian = bool(candle["close"] > candle["donchian_upper_prev"])
        below_donchian = bool(candle["close"] < candle["donchian_lower_prev"])

    if direction == "LONG":
        fired = bool(valid and close_above_ema and adx_ok and above_donchian)
        irrelevant = ("close_below_ema200", "close_below_donchian_lower_prev")
    elif direction == "SHORT":
        fired = bool(valid and close_below_ema and adx_ok and below_donchian)
        irrelevant = ("close_above_ema200", "close_above_donchian_upper_prev")
    else:
        fired = False
        irrelevant = ()

    return EntryGateExplanation(
        direction=direction,
        indicators_valid=valid,
        close_above_ema200=close_above_ema,
        close_below_ema200=close_below_ema,
        adx_above_threshold=adx_ok,
        close_above_donchian_upper_prev=above_donchian,
        close_below_donchian_lower_prev=below_donchian,
        entry_fired=fired,
        irrelevant_gates=irrelevant,
    )


def render_entry_explanation(
    trade_number: int, pair: str, direction: str, entry_time: str, gates: EntryGateExplanation
) -> str:
    """Human-readable explanation, in the style the build block asked for."""
    if not gates.indicators_valid:
        return f"Trade #{trade_number} ({pair}, {direction}): UNRESOLVED -- DATA NOT AVAILABLE (indicators not valid at entry candle)"

    lines = [f"Trade #{trade_number} entered {direction} ({pair}) at {entry_time} because:"]
    if direction == "LONG":
        lines.append(f"1. price was above EMA200: {gates.close_above_ema200}")
        lines.append(f"2. ADX14 > 25: {gates.adx_above_threshold}")
        lines.append(f"3. close exceeded previous Donchian upper: {gates.close_above_donchian_upper_prev}")
    elif direction == "SHORT":
        lines.append(f"1. price was below EMA200: {gates.close_below_ema200}")
        lines.append(f"2. ADX14 > 25: {gates.adx_above_threshold}")
        lines.append(f"3. close fell below previous Donchian lower: {gates.close_below_donchian_lower_prev}")
    lines.append(f"All required gates fired: {gates.entry_fired}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Part B: exit-sequence forensics (which mechanism fired first)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExitSequenceResult:
    """First candle (if any) where each exit mechanism becomes true, walking
    forward from the entry candle -- and which one came first."""

    stop_trigger_index: int | None
    stop_trigger_time: str | None
    exit_signal_trigger_index: int | None
    exit_signal_trigger_time: str | None
    which_first: str  # "stop_loss" | "exit_signal" | "neither_within_window" | "same_candle"


def reconstruct_exit_sequence(
    indicators_df: pd.DataFrame,
    entry_index: int,
    direction: str,
    entry_price: float,
    stoploss_pct: float = DEFAULT_STOP_LOSS_PCT,
) -> ExitSequenceResult:
    """Walk forward from `entry_index` (inclusive) to find which fires first:
    the fixed `stoploss_pct` stop, or `TrendFollowCore`'s own EMA200-cross
    exit condition. Never looks before `entry_index`.
    """
    stop_idx = stop_time = None
    exit_idx = exit_time = None

    if direction == "LONG":
        stop_price = entry_price * (1 - stoploss_pct / 100.0)
    elif direction == "SHORT":
        stop_price = entry_price * (1 + stoploss_pct / 100.0)
    else:
        stop_price = None

    for i in range(entry_index, len(indicators_df)):
        row = indicators_df.iloc[i]

        if stop_idx is None and stop_price is not None:
            if direction == "LONG" and row["low"] <= stop_price:
                stop_idx, stop_time = i, str(row["date"])
            elif direction == "SHORT" and row["high"] >= stop_price:
                stop_idx, stop_time = i, str(row["date"])

        if exit_idx is None and pd.notna(row.get("ema200")):
            if direction == "LONG" and row["close"] < row["ema200"]:
                exit_idx, exit_time = i, str(row["date"])
            elif direction == "SHORT" and row["close"] > row["ema200"]:
                exit_idx, exit_time = i, str(row["date"])

        if stop_idx is not None and exit_idx is not None:
            break

    if stop_idx is None and exit_idx is None:
        which = "neither_within_window"
    elif stop_idx is None:
        which = "exit_signal"
    elif exit_idx is None:
        which = "stop_loss"
    elif stop_idx < exit_idx:
        which = "stop_loss"
    elif exit_idx < stop_idx:
        which = "exit_signal"
    else:
        which = "same_candle"

    return ExitSequenceResult(
        stop_trigger_index=stop_idx,
        stop_trigger_time=stop_time,
        exit_signal_trigger_index=exit_idx,
        exit_signal_trigger_time=exit_time,
        which_first=which,
    )


# ---------------------------------------------------------------------------
# Part 7 (build block numbering): counterfactual recovery analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CounterfactualRecovery:
    """Hypothetical counterfactual: what the price path did AFTER the
    original -5% stop point would have triggered. Explicitly hypothetical
    -- see module docstring for the MFE/MAE labeling caveat."""

    stop_trigger_index: int | None
    recovered_above_entry: bool | None
    reached_profitability: bool | None
    first_exit_signal_after_stop_time: str | None
    favorable_pct: float | None
    adverse_pct: float | None
    still_open_at_window_end: bool | None


def counterfactual_recovery_after_stop(
    indicators_df: pd.DataFrame,
    entry_index: int,
    direction: str,
    entry_price: float,
    stoploss_pct: float = DEFAULT_STOP_LOSS_PCT,
) -> CounterfactualRecovery:
    """For a trade that WAS stopped out at `stoploss_pct`, look at what price
    did afterward, had the position hypothetically stayed open.
    """
    sequence = reconstruct_exit_sequence(
        indicators_df, entry_index, direction, entry_price, stoploss_pct
    )
    if sequence.stop_trigger_index is None:
        return CounterfactualRecovery(
            stop_trigger_index=None,
            recovered_above_entry=None,
            reached_profitability=None,
            first_exit_signal_after_stop_time=None,
            favorable_pct=None,
            adverse_pct=None,
            still_open_at_window_end=None,
        )

    after = indicators_df.iloc[sequence.stop_trigger_index + 1 :]
    if after.empty:
        return CounterfactualRecovery(
            stop_trigger_index=sequence.stop_trigger_index,
            recovered_above_entry=False,
            reached_profitability=False,
            first_exit_signal_after_stop_time=None,
            favorable_pct=0.0,
            adverse_pct=0.0,
            still_open_at_window_end=True,
        )

    if direction == "LONG":
        favorable_pct = float((after["high"].max() - entry_price) / entry_price * 100.0)
        adverse_pct = float((entry_price - after["low"].min()) / entry_price * 100.0)
        recovered = bool((after["close"] > entry_price).any())
        profitable = recovered
    elif direction == "SHORT":
        favorable_pct = float((entry_price - after["low"].min()) / entry_price * 100.0)
        adverse_pct = float((after["high"].max() - entry_price) / entry_price * 100.0)
        recovered = bool((after["close"] < entry_price).any())
        profitable = recovered
    else:
        favorable_pct = adverse_pct = None
        recovered = profitable = None

    exit_signal_time = None
    for i in range(sequence.stop_trigger_index + 1, len(indicators_df)):
        row = indicators_df.iloc[i]
        if pd.isna(row.get("ema200")):
            continue
        if direction == "LONG" and row["close"] < row["ema200"]:
            exit_signal_time = str(row["date"])
            break
        if direction == "SHORT" and row["close"] > row["ema200"]:
            exit_signal_time = str(row["date"])
            break

    return CounterfactualRecovery(
        stop_trigger_index=sequence.stop_trigger_index,
        recovered_above_entry=recovered,
        reached_profitability=profitable,
        first_exit_signal_after_stop_time=exit_signal_time,
        favorable_pct=favorable_pct,
        adverse_pct=adverse_pct,
        still_open_at_window_end=(exit_signal_time is None),
    )


# ---------------------------------------------------------------------------
# Fate classification across stop widths
# ---------------------------------------------------------------------------

FATE_RECOVERED_INTO_PROFIT = "A: Recovered into profit"
FATE_REDUCED_LOSS = "B: Reduced the loss"
FATE_DELAYED_SAME_LOSS = "C: Delayed the same loss"
FATE_LARGER_LOSS = "D: Created a larger loss"
FATE_NORMAL_EXIT = "E: Eventually exited by normal exit signal"
FATE_STILL_OPEN = "F: Still open / unresolved at window end"
FATE_UNKNOWN = "G: Other -- insufficient data to classify"


def classify_stop_trade_fate(
    baseline_profit_pct: float | None,
    test_exit_reason: str | None,
    test_profit_pct: float | None,
) -> str:
    """Classify one baseline stop-loss trade's fate under a wider stop,
    given that test's actual recorded exit_reason/profit_pct (from a real
    re-run backtest at that stop width -- not simulated here).
    """
    if test_profit_pct is None or test_exit_reason is None or baseline_profit_pct is None:
        return FATE_UNKNOWN

    if test_exit_reason == "exit_signal":
        if test_profit_pct > 0:
            return FATE_RECOVERED_INTO_PROFIT
        return FATE_NORMAL_EXIT

    if test_exit_reason == "force_exit":
        return FATE_STILL_OPEN

    if test_exit_reason == "stop_loss":
        if test_profit_pct > baseline_profit_pct + 1e-9:
            return FATE_REDUCED_LOSS
        if abs(test_profit_pct - baseline_profit_pct) <= 1e-9:
            return FATE_DELAYED_SAME_LOSS
        return FATE_LARGER_LOSS

    return FATE_UNKNOWN


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_stoploss_forensics_dataset(dataset: dict[str, Any], output_path: Path) -> None:
    """Write `dataset` as JSON. Caller validates `output_path` is inside
    persistent storage (see `hermes.export_paths`)."""
    Path(output_path).write_text(json.dumps(dataset, indent=2, default=str))
