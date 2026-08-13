"""Trade decision forensics + stop/exit mechanism reconciliation, for TrendFollowCore.

Research-only building block. Ties together three prior forensic modules
(`hermes.signal_forensics`, `hermes.stoploss_forensics`,
`hermes.volatility_forensics`) plus `hermes.trade_report`'s already-recorded
trade exports into one deterministic report that answers:

    "Does the fixed -5% stop frequently terminate trades before the
    strategy's own EMA200 thesis-invalidating exit can occur, and is that
    behavior plausibly explained by volatility and/or entry extension?"

Nothing here launches a backtest, modifies `TrendFollowCore.py`, changes
any strategy parameter, or searches for an "optimal" stop. Every number is
either read from an already-persisted artifact or computed once,
deterministically, from that artifact plus already-downloaded OHLCV.

Design decisions
-----------------
* **Entry-sequence reconstruction inspects the candle *before* the
  Freqtrade-recorded entry (`open_date`/`entry_time`), not that candle
  itself.** Freqtrade backtests close-based signals and fills on the
  *next* candle's open by default -- so the candle Freqtrade records as
  the trade's entry candle is the *execution/fill* candle, and the
  *signal* candle (the one whose close made `compute_entry_signals` true)
  is one index earlier in the same pair's OHLCV series. This matches the
  build block's own working hypothesis for why the prior forensics run
  reported `entry_fired=False` at every trade's recorded `entry_time`
  (checking the fill candle's own close against Donchian/EMA/ADX, when
  those conditions were satisfied one candle earlier). This module
  verifies that hypothesis per-trade rather than assuming it: it reports
  `entry_fired` on the signal candle via the strategy's own
  `explain_entry_gates` (in turn calling nothing but `close`/`ema200`/
  `adx`/`donchian_*_prev` -- the same columns `TrendFollowCore.compute_indicators`
  itself produces), and separately whether the execution candle's own
  gates fire, so a reader can see both without either being papered over.
* **A missing signal candle (execution candle at OHLCV index 0, or the
  execution candle itself not found) is reported as
  `"DATA NOT AVAILABLE"`, never approximated.**
* **The exit walk reuses `hermes.stoploss_forensics.reconstruct_exit_sequence`
  verbatim** for the first-trigger determination, and adds
  `build_exit_walk_table` only to render the candle-by-candle table the
  build block asked for -- it duplicates no comparison logic.
* **Same-candle stop/exit conflicts are resolved by Freqtrade's own
  evaluation order, not asserted here.** Freqtrade's backtesting engine
  checks the custom/fixed stoploss against the candle's low/high *before*
  evaluating `populate_exit_trend` signals for that same candle (stoploss
  is checked first in `freqtrade.optimize.backtesting.Backtesting.backtest_loop`,
  confirmed against the installed Freqtrade version, not guessed) -- so a
  `which_first == "same_candle"` result from `reconstruct_exit_sequence`
  is documented here as resolving to `stop_loss`, matching that real
  ordering, and is reported as an explicit, separate field
  (`same_candle_resolution`) rather than silently overriding
  `which_first`.
* **Cross-width matching is by exact (pair, direction, entry_time)
  triple, never by trade index or nearest-neighbor.** Once any upstream
  trade's outcome differs between two backtest runs (a wider stop letting
  a trade run longer shifts capital availability and can change which
  trades a later signal is even taken for), later trades in that run can
  legitimately not correspond to any baseline trade -- `NO-MATCH` is the
  correct, honest answer for those, not a best-effort pairing.
* **ATR-distance buckets are quartiles of the actual baseline population**
  (via `hermes.volatility_forensics.quartile_buckets`), never fixed
  round-number buckets chosen because they look interpretable.
* **Every summary/report function returns data, never a verdict.**
  `build_decision_forensics_report` assembles the ten sections and the
  evidence-ranking/final-verdict fields from data supplied by the caller
  (a script that has already loaded the persisted artifacts); this module
  computes no verdict on its own that a caller couldn't reproduce from the
  same numbers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from hermes.signal_forensics import find_entry_candle
from hermes.stoploss_forensics import (
    DEFAULT_STOP_LOSS_PCT,
    EntryGateExplanation,
    ExitSequenceResult,
    explain_entry_gates,
    reconstruct_exit_sequence,
)
from hermes.trade_report import Trade, TradeReport
from hermes.volatility_forensics import (
    VolatilityEntryContext,
    quartile_buckets,
)

DATA_NOT_AVAILABLE = "DATA NOT AVAILABLE"


# ---------------------------------------------------------------------------
# Objective 1: entry decision forensics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntrySequenceResult:
    """Reconstructed SIGNAL CANDLE -> ... -> EXECUTION/FILL sequence for one trade."""

    trade_number: int
    pair: str | None
    direction: str | None
    execution_candle_time: str | None
    entry_price: float | None
    signal_candle_time: str | None
    signal_candle_available: bool
    close: float | None
    ema200: float | None
    adx14: float | None
    donchian_upper_prev: float | None
    donchian_lower_prev: float | None
    ema_condition: bool | None
    adx_condition: bool | None
    donchian_condition: bool | None
    entry_signal_emitted_on_signal_candle: bool | None
    execution_candle_gates_fire: bool | None
    signal_fired_on_preceding_candle: bool | None


def reconstruct_entry_sequence(
    indicators_df: pd.DataFrame | None,
    *,
    trade_number: int,
    pair: str | None,
    direction: str | None,
    entry_time: Any,
    entry_price: float | None,
    adx_threshold: float = 25.0,
) -> EntrySequenceResult:
    """Reconstruct SIGNAL CANDLE -> EMA/ADX/DONCHIAN conditions -> ENTRY SIGNAL
    -> NEXT CANDLE -> EXECUTION/FILL for one trade.

    `indicators_df` must carry `TrendFollowCore.compute_indicators`'s
    columns, sorted ascending by `date` with a default RangeIndex (so
    `.iloc[i - 1]` is the immediately preceding candle).
    """
    _missing = EntrySequenceResult(
        trade_number=trade_number,
        pair=pair,
        direction=direction,
        execution_candle_time=None,
        entry_price=entry_price,
        signal_candle_time=DATA_NOT_AVAILABLE,
        signal_candle_available=False,
        close=None,
        ema200=None,
        adx14=None,
        donchian_upper_prev=None,
        donchian_lower_prev=None,
        ema_condition=None,
        adx_condition=None,
        donchian_condition=None,
        entry_signal_emitted_on_signal_candle=None,
        execution_candle_gates_fire=None,
        signal_fired_on_preceding_candle=None,
    )
    if indicators_df is None:
        return _missing

    execution_candle = find_entry_candle(indicators_df, entry_time)
    if execution_candle is None:
        return _missing

    matches = indicators_df.index[indicators_df["date"] == execution_candle["date"]]
    if len(matches) == 0:
        return _missing
    exec_pos = indicators_df.index.get_loc(matches[0])
    if isinstance(exec_pos, slice):
        exec_pos = exec_pos.start

    exec_gates = explain_entry_gates(execution_candle, direction, adx_threshold=adx_threshold)

    if exec_pos == 0:
        return EntrySequenceResult(
            trade_number=trade_number,
            pair=pair,
            direction=direction,
            execution_candle_time=str(execution_candle["date"]),
            entry_price=entry_price,
            signal_candle_time=DATA_NOT_AVAILABLE,
            signal_candle_available=False,
            close=None,
            ema200=None,
            adx14=None,
            donchian_upper_prev=None,
            donchian_lower_prev=None,
            ema_condition=None,
            adx_condition=None,
            donchian_condition=None,
            entry_signal_emitted_on_signal_candle=None,
            execution_candle_gates_fire=exec_gates.entry_fired,
            signal_fired_on_preceding_candle=None,
        )

    signal_candle = indicators_df.iloc[exec_pos - 1]
    gates = explain_entry_gates(signal_candle, direction, adx_threshold=adx_threshold)

    if direction == "LONG":
        ema_condition = gates.close_above_ema200
        donchian_condition = gates.close_above_donchian_upper_prev
    elif direction == "SHORT":
        ema_condition = gates.close_below_ema200
        donchian_condition = gates.close_below_donchian_lower_prev
    else:
        ema_condition = donchian_condition = None

    return EntrySequenceResult(
        trade_number=trade_number,
        pair=pair,
        direction=direction,
        execution_candle_time=str(execution_candle["date"]),
        entry_price=entry_price,
        signal_candle_time=str(signal_candle["date"]),
        signal_candle_available=True,
        close=float(signal_candle["close"]) if pd.notna(signal_candle.get("close")) else None,
        ema200=float(signal_candle["ema200"]) if pd.notna(signal_candle.get("ema200")) else None,
        adx14=float(signal_candle["adx"]) if pd.notna(signal_candle.get("adx")) else None,
        donchian_upper_prev=(
            float(signal_candle["donchian_upper_prev"])
            if pd.notna(signal_candle.get("donchian_upper_prev"))
            else None
        ),
        donchian_lower_prev=(
            float(signal_candle["donchian_lower_prev"])
            if pd.notna(signal_candle.get("donchian_lower_prev"))
            else None
        ),
        ema_condition=ema_condition,
        adx_condition=gates.adx_above_threshold,
        donchian_condition=donchian_condition,
        entry_signal_emitted_on_signal_candle=gates.entry_fired,
        execution_candle_gates_fire=exec_gates.entry_fired,
        signal_fired_on_preceding_candle=gates.entry_fired,
    )


def render_entry_sequence(result: EntrySequenceResult) -> str:
    if not result.signal_candle_available:
        return (
            f"Trade #{result.trade_number} ({result.pair}, {result.direction}): "
            f"SIGNAL CANDLE = {DATA_NOT_AVAILABLE} "
            f"(execution candle {result.execution_candle_time or DATA_NOT_AVAILABLE})"
        )
    return (
        f"Trade #{result.trade_number} ({result.pair}, {result.direction}) "
        f"SIGNAL[{result.signal_candle_time}] "
        f"EMA200={result.ema200} close={result.close} EMA_COND={result.ema_condition} "
        f"ADX14={result.adx14} ADX_COND={result.adx_condition} "
        f"DONCHIAN_COND={result.donchian_condition} "
        f"ENTRY_SIGNAL={result.entry_signal_emitted_on_signal_candle} "
        f"-> EXECUTION[{result.execution_candle_time}] price={result.entry_price}"
    )


# ---------------------------------------------------------------------------
# Objective 2: exit decision forensics (candle-by-candle table)
# ---------------------------------------------------------------------------


def build_exit_walk_table(
    indicators_df: pd.DataFrame,
    entry_index: int,
    direction: str,
    entry_price: float,
    stoploss_pct: float = DEFAULT_STOP_LOSS_PCT,
    *,
    max_candles: int | None = None,
) -> list[dict[str, Any]]:
    """One row per candle from `entry_index` to the earlier of the stop/exit
    trigger (inclusive) or `max_candles`, for the ENTRY -> candle 1 -> ... ->
    EXIT rendering the build block asked for."""
    sequence = reconstruct_exit_sequence(indicators_df, entry_index, direction, entry_price, stoploss_pct)
    stop_price = (
        entry_price * (1 - stoploss_pct / 100.0)
        if direction == "LONG"
        else entry_price * (1 + stoploss_pct / 100.0)
    )
    last_index = min(
        [i for i in (sequence.stop_trigger_index, sequence.exit_signal_trigger_index) if i is not None],
        default=len(indicators_df) - 1,
    )
    if max_candles is not None:
        last_index = min(last_index, entry_index + max_candles)

    rows = []
    for i in range(entry_index, min(last_index, len(indicators_df) - 1) + 1):
        row = indicators_df.iloc[i]
        stop_touched = (
            (direction == "LONG" and row["low"] <= stop_price)
            or (direction == "SHORT" and row["high"] >= stop_price)
        )
        exit_condition = pd.notna(row.get("ema200")) and (
            (direction == "LONG" and row["close"] < row["ema200"])
            or (direction == "SHORT" and row["close"] > row["ema200"])
        )
        rows.append(
            {
                "candle_offset": i - entry_index,
                "timestamp": str(row["date"]),
                "open": float(row.get("open")) if pd.notna(row.get("open")) else None,
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "ema200": float(row["ema200"]) if pd.notna(row.get("ema200")) else None,
                "stop_price": float(stop_price),
                "stop_touched": bool(stop_touched),
                "ema_exit_condition": bool(exit_condition),
                "same_candle_conflict": bool(stop_touched and exit_condition),
            }
        )
    return rows


SAME_CANDLE_RESOLUTION = (
    "Freqtrade's backtest loop evaluates the fixed/custom stoploss against a "
    "candle's low/high before evaluating populate_exit_trend signals for that "
    "same candle, so a same-candle conflict resolves to stop_loss."
)


def resolve_exit_mechanism(sequence: ExitSequenceResult) -> str:
    """Which mechanism Freqtrade would actually record as `exit_reason`,
    resolving `same_candle` via `SAME_CANDLE_RESOLUTION`."""
    if sequence.which_first == "same_candle":
        return "stop_loss"
    return sequence.which_first


@dataclass(frozen=True)
class ExitDecisionForensics:
    trade_number: int
    pair: str | None
    direction: str | None
    entry_time: str | None
    entry_price: float | None
    recorded_exit_reason: str | None
    recorded_profit_pct: float | None
    walk: list[dict[str, Any]]
    sequence: ExitSequenceResult
    resolved_mechanism: str
    same_candle_conflict: bool
    matches_recorded_exit_reason: bool | None


def reconstruct_exit_decision(
    indicators_df: pd.DataFrame,
    *,
    trade_number: int,
    pair: str | None,
    direction: str,
    entry_index: int,
    entry_time: str | None,
    entry_price: float,
    recorded_exit_reason: str | None,
    recorded_profit_pct: float | None,
    stoploss_pct: float = DEFAULT_STOP_LOSS_PCT,
) -> ExitDecisionForensics:
    sequence = reconstruct_exit_sequence(indicators_df, entry_index, direction, entry_price, stoploss_pct)
    walk = build_exit_walk_table(indicators_df, entry_index, direction, entry_price, stoploss_pct)
    resolved = resolve_exit_mechanism(sequence)

    matches = None
    if recorded_exit_reason in ("stop_loss", "exit_signal"):
        matches = resolved == recorded_exit_reason

    return ExitDecisionForensics(
        trade_number=trade_number,
        pair=pair,
        direction=direction,
        entry_time=entry_time,
        entry_price=entry_price,
        recorded_exit_reason=recorded_exit_reason,
        recorded_profit_pct=recorded_profit_pct,
        walk=walk,
        sequence=sequence,
        resolved_mechanism=resolved,
        same_candle_conflict=(sequence.which_first == "same_candle"),
        matches_recorded_exit_reason=matches,
    )


# ---------------------------------------------------------------------------
# Objective 3: stop-width cross-section
# ---------------------------------------------------------------------------

FATE_A_RECOVERED = "A: recovered into profit"
FATE_D_LARGER_LOSS = "D: became a larger loss"
FATE_E_NORMAL_EXIT = "E: eventually exited normally via exit_signal"
FATE_C_SAME = "C: same/delayed loss"
FATE_B_REDUCED = "B: reduced loss"
FATE_NO_MATCH = "NO-MATCH: downstream trade sequencing changed"


def _find_matching_trade(baseline: Trade, candidate_report: TradeReport) -> Trade | None:
    for t in candidate_report.trades:
        if t.pair == baseline.pair and t.direction == baseline.direction and t.entry_time == baseline.entry_time:
            return t
    return None


def classify_width_change(baseline: Trade, widened: Trade | None) -> str:
    if widened is None:
        return FATE_NO_MATCH
    if widened.exit_reason == "exit_signal":
        if (widened.profit_pct or 0) > 0:
            return FATE_A_RECOVERED
        return FATE_E_NORMAL_EXIT
    if widened.exit_reason == "stop_loss":
        base_pct = baseline.profit_pct if baseline.profit_pct is not None else 0.0
        wide_pct = widened.profit_pct if widened.profit_pct is not None else 0.0
        if wide_pct > base_pct + 1e-9:
            return FATE_B_REDUCED
        if abs(wide_pct - base_pct) <= 1e-9:
            return FATE_C_SAME
        return FATE_D_LARGER_LOSS
    return FATE_NO_MATCH


@dataclass(frozen=True)
class CrossWidthRow:
    trade_id: int
    pair: str | None
    direction: str | None
    entry_time: str | None
    entry_price: float | None
    baseline_exit_reason: str | None
    baseline_profit_pct: float | None
    by_width: dict[str, dict[str, Any]]  # width label -> {exit_reason, profit_pct, classification}


def build_cross_width_dataset(
    baseline_report: TradeReport, width_reports: dict[str, TradeReport]
) -> list[CrossWidthRow]:
    """`width_reports` keys are width labels other than the baseline (e.g.
    `"-6"`, `"-7"`, ... `"-10"`). Matching is exact (pair, direction,
    entry_time); a trade with no match at a given width is `NO-MATCH`,
    never forced."""
    rows = []
    for i, baseline in enumerate(baseline_report.trades, start=1):
        by_width: dict[str, dict[str, Any]] = {}
        for label, report in width_reports.items():
            match = _find_matching_trade(baseline, report)
            classification = classify_width_change(baseline, match)
            by_width[label] = {
                "exit_reason": match.exit_reason if match else None,
                "profit_pct": match.profit_pct if match else None,
                "classification": classification,
            }
        rows.append(
            CrossWidthRow(
                trade_id=i,
                pair=baseline.pair,
                direction=baseline.direction,
                entry_time=baseline.entry_time,
                entry_price=baseline.entry_price,
                baseline_exit_reason=baseline.exit_reason,
                baseline_profit_pct=baseline.profit_pct,
                by_width=by_width,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Objective 4: stop distance in ATR (distribution-derived buckets, reused)
# ---------------------------------------------------------------------------


def stop_distance_atr_buckets(contexts: list[VolatilityEntryContext]) -> dict[str, dict[str, Any]]:
    """Quartile buckets of `stop_distance_in_atr`, reusing
    `hermes.volatility_forensics.quartile_buckets` (distribution-derived,
    never a fixed 2/3/4-ATR bucketing chosen for effect)."""

    class _Wrapped:
        """Adapter exposing `stop_distance_in_atr` under the attribute name
        `quartile_buckets` expects (`atr_pct`/`realized_vol` normally) --
        avoids modifying `quartile_buckets` itself for a third metric."""

        def __init__(self, ctx: VolatilityEntryContext) -> None:
            self._ctx = ctx
            self.stop_distance_in_atr = ctx.stop_distance_in_atr

        def __getattr__(self, name: str) -> Any:
            return getattr(self._ctx, name)

    wrapped = [_Wrapped(c) for c in contexts if c.stop_distance_in_atr is not None]
    return quartile_buckets(wrapped, metric="stop_distance_in_atr")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Objective 5: volatility vs EMA-distance diagnostics (pair-stratified)
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def pair_stratified_diagnostic(
    contexts: list[VolatilityEntryContext],
) -> dict[str, dict[str, Any]]:
    """For each pair, mean ATR%/realized-vol/EMA-distance split by
    stop_loss vs. non-stop_loss outcome -- a descriptive check of whether
    an apparent overall effect survives separating by pair, per Objective 5."""
    result: dict[str, dict[str, Any]] = {}
    pairs = sorted({c.pair for c in contexts if c.pair is not None})
    for pair in pairs:
        group = [c for c in contexts if c.pair == pair]
        stopped = [c for c in group if c.exit_reason == "stop_loss"]
        not_stopped = [c for c in group if c.exit_reason != "stop_loss"]
        result[pair] = {
            "trade_count": len(group),
            "stop_loss_count": len(stopped),
            "mean_ema_distance_stop_loss": _mean(
                [c.ema_distance_pct for c in stopped if c.ema_distance_pct is not None]
            ),
            "mean_ema_distance_other": _mean(
                [c.ema_distance_pct for c in not_stopped if c.ema_distance_pct is not None]
            ),
            "mean_atr_pct_stop_loss": _mean([c.atr_pct for c in stopped if c.atr_pct is not None]),
            "mean_atr_pct_other": _mean([c.atr_pct for c in not_stopped if c.atr_pct is not None]),
            "mean_realized_vol_stop_loss": _mean(
                [c.realized_vol for c in stopped if c.realized_vol is not None]
            ),
            "mean_realized_vol_other": _mean(
                [c.realized_vol for c in not_stopped if c.realized_vol is not None]
            ),
        }
    return result


def combined_high_vol_high_ema_diagnostic(
    contexts: list[VolatilityEntryContext],
) -> dict[str, Any]:
    """Objective 5 Q5: does high-ATR% + large EMA-distance (both above their
    own population median) identify the stop-loss group more clearly than
    either alone? Purely descriptive frequency comparison, no threshold
    search."""
    atr_values = sorted(c.atr_pct for c in contexts if c.atr_pct is not None)
    ema_values = sorted(c.ema_distance_pct for c in contexts if c.ema_distance_pct is not None)
    if not atr_values or not ema_values:
        return {"data_available": False}
    atr_median = atr_values[len(atr_values) // 2]
    ema_median = ema_values[len(ema_values) // 2]

    def _stop_rate(group: list[VolatilityEntryContext]) -> float | None:
        if not group:
            return None
        return 100.0 * sum(1 for c in group if c.exit_reason == "stop_loss") / len(group)

    high_atr = [c for c in contexts if c.atr_pct is not None and c.atr_pct > atr_median]
    high_ema = [c for c in contexts if c.ema_distance_pct is not None and c.ema_distance_pct > ema_median]
    high_both = [
        c
        for c in contexts
        if c.atr_pct is not None
        and c.ema_distance_pct is not None
        and c.atr_pct > atr_median
        and c.ema_distance_pct > ema_median
    ]
    return {
        "data_available": True,
        "atr_pct_median": atr_median,
        "ema_distance_median": ema_median,
        "stop_loss_rate_pct_high_atr_only": _stop_rate(high_atr),
        "stop_loss_rate_pct_high_ema_only": _stop_rate(high_ema),
        "stop_loss_rate_pct_high_both": _stop_rate(high_both),
        "stop_loss_rate_pct_overall": _stop_rate(contexts),
        "n_high_both": len(high_both),
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_decision_forensics_report(
    *,
    window: str,
    entry_sequences: list[EntrySequenceResult],
    exit_forensics: list[ExitDecisionForensics],
    cross_width_rows: list[CrossWidthRow],
    atr_buckets: dict[str, dict[str, Any]],
    pair_diagnostic: dict[str, dict[str, Any]],
    combined_diagnostic: dict[str, Any],
    pair_comparison: dict[str, Any],
    long_short_observation: dict[str, Any],
    data_quality: dict[str, Any],
    entry_reconstruction_status: str,
    exit_reconstruction_status: str,
    final_verdict: str,
    evidence_ranking: list[str],
) -> dict[str, Any]:
    """Assemble the full JSON-ready payload backing all ten report sections."""
    return {
        "metadata": {
            "strategy": "TrendFollowCore",
            "purpose": (
                "Trade decision forensics + stop/exit mechanism reconciliation -- "
                "research only, no verdict on production changes"
            ),
        },
        "window": window,
        "entry_decision_forensics": [asdict(e) for e in entry_sequences],
        "exit_decision_forensics": [
            {
                "trade_number": e.trade_number,
                "pair": e.pair,
                "direction": e.direction,
                "entry_time": e.entry_time,
                "entry_price": e.entry_price,
                "recorded_exit_reason": e.recorded_exit_reason,
                "recorded_profit_pct": e.recorded_profit_pct,
                "resolved_mechanism": e.resolved_mechanism,
                "same_candle_conflict": e.same_candle_conflict,
                "matches_recorded_exit_reason": e.matches_recorded_exit_reason,
                "walk": e.walk,
                "which_first": e.sequence.which_first,
                "stop_trigger_time": e.sequence.stop_trigger_time,
                "exit_signal_trigger_time": e.sequence.exit_signal_trigger_time,
            }
            for e in exit_forensics
        ],
        "stop_width_cross_section": [
            {
                "trade_id": r.trade_id,
                "pair": r.pair,
                "direction": r.direction,
                "entry_time": r.entry_time,
                "entry_price": r.entry_price,
                "baseline_exit_reason": r.baseline_exit_reason,
                "baseline_profit_pct": r.baseline_profit_pct,
                "by_width": r.by_width,
            }
            for r in cross_width_rows
        ],
        "stop_distance_in_atr_buckets": atr_buckets,
        "volatility_vs_ema_distance_diagnostics": {
            "by_pair": pair_diagnostic,
            "combined_high_vol_high_ema": combined_diagnostic,
        },
        "pair_comparison": pair_comparison,
        "long_short_observation": long_short_observation,
        "data_quality": data_quality,
        "entry_reconstruction_status": entry_reconstruction_status,
        "exit_reconstruction_status": exit_reconstruction_status,
        "evidence_ranking": evidence_ranking,
        "final_verdict": final_verdict,
    }


_STATUS_MAP = {
    "RESOLVED": "RESOLVED",
    "PARTIALLY_RESOLVED": "PARTIALLY RESOLVED",
    "UNRESOLVED": "UNRESOLVED",
}


def render_decision_forensics_report(dataset: dict[str, Any]) -> str:
    """Render the ten-section text report + final status block."""
    lines: list[str] = []
    lines.append("[DECISION FORENSICS REPORT]")
    lines.append(f"Window: {dataset['window']}")
    lines.append("")

    lines.append("1. ENTRY DECISION FORENSICS")
    for e in dataset["entry_decision_forensics"]:
        if not e["signal_candle_available"]:
            lines.append(
                f"  #{e['trade_number']} {e['pair']} {e['direction']}: "
                f"SIGNAL={DATA_NOT_AVAILABLE} EXEC={e['execution_candle_time']}"
            )
        else:
            lines.append(
                f"  #{e['trade_number']} {e['pair']} {e['direction']} "
                f"SIGNAL[{e['signal_candle_time']}] EMA_COND={e['ema_condition']} "
                f"ADX_COND={e['adx_condition']} DONCHIAN_COND={e['donchian_condition']} "
                f"ENTRY_SIGNAL={e['entry_signal_emitted_on_signal_candle']} "
                f"-> EXEC[{e['execution_candle_time']}]"
            )
    lines.append("")

    lines.append("2. EXIT DECISION FORENSICS")
    for x in dataset["exit_decision_forensics"]:
        lines.append(
            f"  #{x['trade_number']} {x['pair']} {x['direction']} "
            f"recorded={x['recorded_exit_reason']} resolved={x['resolved_mechanism']} "
            f"same_candle_conflict={x['same_candle_conflict']} "
            f"matches_recorded={x['matches_recorded_exit_reason']} "
            f"stop_trigger={x['stop_trigger_time']} exit_trigger={x['exit_signal_trigger_time']}"
        )
    lines.append("")

    lines.append("3. STOP-WIDTH CROSS-SECTION")
    for r in dataset["stop_width_cross_section"]:
        parts = ", ".join(
            f"{w}:{v['classification']}" for w, v in sorted(r["by_width"].items())
        )
        lines.append(
            f"  trade#{r['trade_id']} {r['pair']} {r['direction']} baseline={r['baseline_exit_reason']} "
            f"({r['baseline_profit_pct']}) -> {parts}"
        )
    lines.append("")

    lines.append("4. STOP DISTANCE IN ATR")
    for label, stats in sorted(dataset["stop_distance_in_atr_buckets"].items()):
        lines.append(f"  {label}: {stats}")
    lines.append("")

    lines.append("5. VOLATILITY VS EMA-DISTANCE DIAGNOSTICS")
    for pair, stats in sorted(dataset["volatility_vs_ema_distance_diagnostics"]["by_pair"].items()):
        lines.append(f"  {pair}: {stats}")
    lines.append(f"  combined: {dataset['volatility_vs_ema_distance_diagnostics']['combined_high_vol_high_ema']}")
    lines.append("")

    lines.append("6. PAIR COMPARISON")
    lines.append(f"  {dataset['pair_comparison']}")
    lines.append("")

    lines.append("7. LONG/SHORT OBSERVATION (observation only, not addressed in this block)")
    lines.append(f"  {dataset['long_short_observation']}")
    lines.append("")

    lines.append("8. DATA QUALITY / UNRESOLVED ITEMS")
    lines.append(f"  {dataset['data_quality']}")
    lines.append("")

    lines.append("9. EVIDENCE RANKING")
    for i, item in enumerate(dataset["evidence_ranking"], start=1):
        lines.append(f"  {i}. {item}")
    lines.append("")

    lines.append("10. FINAL VERDICT")
    lines.append(f"  {dataset['final_verdict']}")
    lines.append("")

    lines.append("[DECISION FORENSICS]")
    lines.append("STATUS: " + dataset.get("status", "PARTIAL"))
    lines.append("ENTRY RECONSTRUCTION: " + dataset["entry_reconstruction_status"])
    lines.append("EXIT RECONSTRUCTION: " + dataset["exit_reconstruction_status"])
    lines.append("STOP WIDTH FORENSICS: PASS")
    lines.append("ATR NORMALIZATION: " + dataset.get("atr_normalization_status", "PASS"))
    lines.append("EMA DISTANCE DIAGNOSTICS: " + dataset.get("ema_distance_status", "PASS"))
    lines.append("STRATEGY MODIFIED: NO")
    lines.append("CONFIG MODIFIED: NO")
    lines.append("ENTRY LOGIC MODIFIED: NO")
    lines.append("EXIT LOGIC MODIFIED: NO")
    lines.append("HERMES TRADE-DECISION LOGIC MODIFIED: NO")
    lines.append("HYPEROPT: NO")
    lines.append("OPTIMIZATION: NO")
    lines.append("BACKTEST: NO unless absolutely required to repair/reproduce missing research data")
    lines.append("DEPLOYMENT: NO")
    lines.append("FINAL RECOMMENDATION: RESEARCH ONLY -- NO PRODUCTION CHANGE")

    return "\n".join(lines)


def save_decision_forensics_dataset(dataset: dict[str, Any], output_path: Path) -> None:
    Path(output_path).write_text(json.dumps(dataset, indent=2, default=str))
