"""Reconstruct TrendFollowCore's indicator context at each frozen trade's entry.

Companion to `hermes.trade_report`: that module recovers *what happened*
(entry/exit price, P&L, exit reason) for each of the frozen baseline's 39
trades from Freqtrade's own trade export. This module answers a different
question -- *what the strategy saw* at the moment each trade entered
(EMA200, ADX14, the Donchian breakout levels) -- purely as a deterministic
read-only calculation over already-downloaded OHLCV candles, with no
subprocess, no Freqtrade backtesting engine, and no strategy modification
of any kind.

Nothing here launches a backtest, a subprocess, or Freqtrade at all. It is
pure post-hoc arithmetic over data that already exists: the persisted
trade export (`hermes.trade_report`) and the already-downloaded OHLCV
candles for each pair.

Design decisions
-----------------
* **Reuses `TrendFollowCore.compute_indicators` itself, dynamically
  imported from the actual strategy file, rather than re-deriving a
  parallel EMA/ADX/Donchian calculation.** A hand-rewritten "equivalent"
  formula is exactly the kind of thing that can silently drift from the
  strategy's real behavior; importing the live function means this module
  is definitionally computing what `TrendFollowCore` itself computes, not
  an approximation of it. `load_trendfollow_indicator_functions` does this
  via `importlib`, matching how Freqtrade itself resolves strategy
  classes from a file path, and never writes to the strategy file.
* **No-lookahead is verified, not assumed.** `audit_no_lookahead`
  recomputes indicators on a dataframe truncated to end at the candle
  under test and compares against the same candle's value computed from
  the full dataframe. If a later candle had influenced the earlier one,
  these two computations would disagree -- so passing this check is a
  genuine proof for that candle, not a restatement of the module
  docstring's claim in `TrendFollowCore.py`.
* **A trade whose entry candle can't be found in the OHLCV data becomes
  `candle_matched=False` with every indicator field `None`, not a
  fabricated approximation from a nearby candle.** Reconciliation against
  the frozen 39 trades is meant to catch exactly this: silently
  interpolating a "close enough" candle would hide a real data gap.
* **Distance metrics are computed here, not read off any indicator
  column**, since Freqtrade's export doesn't carry them -- they are
  observational arithmetic over already-known values (entry price,
  EMA200, Donchian levels), explicitly not a threshold or filter
  suggestion of any kind.
"""

from __future__ import annotations

import importlib.util
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from hermes.trade_report import Trade, TradeReport


class SignalForensicsError(Exception):
    """Raised when the strategy module can't be loaded, or reconciliation fails."""


# ---------------------------------------------------------------------------
# Reuse the strategy's own indicator functions (read-only import)
# ---------------------------------------------------------------------------


def load_trendfollow_indicator_functions(strategy_path: Path):
    """Dynamically import `compute_indicators` from `strategy_path`.

    Reads the file to import it as a module; never writes to it. Using
    `importlib` (the same mechanism Freqtrade itself uses to resolve a
    strategy class from a file) means this calls the actual, current
    `TrendFollowCore.compute_indicators` -- not a copy that could grow
    stale if the strategy is ever legitimately changed elsewhere.
    """
    strategy_path = Path(strategy_path)
    spec = importlib.util.spec_from_file_location(
        "trend_follow_core_readonly_import", strategy_path
    )
    if spec is None or spec.loader is None:
        raise SignalForensicsError(f"cannot import strategy module from {strategy_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise SignalForensicsError(
            f"failed to import strategy module from {strategy_path}: {exc}"
        ) from exc
    if not hasattr(module, "compute_indicators"):
        raise SignalForensicsError(
            f"{strategy_path} has no compute_indicators function"
        )
    return module.compute_indicators


# ---------------------------------------------------------------------------
# No-lookahead audit
# ---------------------------------------------------------------------------

_INDICATOR_COLUMNS = ("ema200", "adx", "donchian_upper_prev", "donchian_lower_prev")


def audit_no_lookahead(
    df: pd.DataFrame,
    compute_indicators_fn,
    at_index: int,
    *,
    tolerance: float = 1e-9,
) -> bool:
    """`True` iff indicators at row `at_index` are identical whether computed
    from the full `df` or from `df` truncated to end at `at_index`.

    If any later row had influenced the indicator value at `at_index`,
    truncating the dataframe there would change that value -- so this is
    a genuine check, not a restatement of trust in the strategy's own
    documented no-lookahead design.
    """
    full = compute_indicators_fn(df)
    truncated = compute_indicators_fn(df.iloc[: at_index + 1])
    full_row = full.iloc[at_index]
    truncated_row = truncated.iloc[-1]
    for col in _INDICATOR_COLUMNS:
        a, b = full_row[col], truncated_row[col]
        a_nan, b_nan = pd.isna(a), pd.isna(b)
        if a_nan and b_nan:
            continue
        if a_nan or b_nan:
            return False
        if abs(a - b) > tolerance:
            return False
    return True


# ---------------------------------------------------------------------------
# Entry-candle matching and context reconstruction
# ---------------------------------------------------------------------------


def _none_if_nan(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return float(value)


def find_entry_candle(indicators_df: pd.DataFrame, entry_time: Any) -> pd.Series | None:
    """The row in `indicators_df` whose `date` exactly matches `entry_time`.

    Returns `None` (not an error) on no exact match -- a genuinely missing
    or misaligned candle is a reportable gap for the caller to surface,
    never something to approximate from a neighboring candle.
    """
    if entry_time is None:
        return None
    try:
        ts = pd.Timestamp(entry_time)
    except (ValueError, TypeError):
        return None
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

    matches = indicators_df.loc[indicators_df["date"] == ts]
    if matches.empty:
        return None
    return matches.iloc[0]


def compute_ema_distance_pct(
    entry_price: float | None, ema200: float | None, direction: str | None
) -> float | None:
    """Observational only -- see module docstring. Never used to gate entries."""
    if entry_price is None or not ema200 or direction not in ("LONG", "SHORT"):
        return None
    if direction == "LONG":
        return (entry_price - ema200) / ema200
    return (ema200 - entry_price) / ema200


def compute_breakout_distance_pct(
    entry_price: float | None,
    donchian_upper_prev: float | None,
    donchian_lower_prev: float | None,
    direction: str | None,
) -> float | None:
    """Observational only -- see module docstring. Never used to gate entries."""
    if entry_price is None or direction not in ("LONG", "SHORT"):
        return None
    if direction == "LONG":
        if not donchian_upper_prev:
            return None
        return (entry_price - donchian_upper_prev) / donchian_upper_prev
    if not donchian_lower_prev:
        return None
    return (donchian_lower_prev - entry_price) / donchian_lower_prev


@dataclass(frozen=True)
class EntryContext:
    """Reconstructed indicator context for one frozen trade's entry, plus
    its already-known outcome (from `hermes.trade_report`)."""

    trade_number: int
    pair: str | None
    direction: str | None
    entry_time: str | None
    entry_price: float | None
    ema200: float | None
    adx14: float | None
    donchian_upper_prev: float | None
    donchian_lower_prev: float | None
    ema_distance_pct: float | None
    breakout_distance_pct: float | None
    enter_tag: str | None
    exit_reason: str | None
    profit_abs: float | None
    profit_pct: float | None
    duration_minutes: float | None
    is_winner: bool | None
    candle_matched: bool


def reconstruct_entry_context(
    trade_number: int, trade: Trade, indicators_df: pd.DataFrame | None
) -> EntryContext:
    """Build one trade's `EntryContext`. `indicators_df` is `None` when no
    OHLCV data exists at all for `trade.pair` -- distinct from "candle not
    found within otherwise-present data" (also reported, via
    `candle_matched=False`)."""
    candle = None if indicators_df is None else find_entry_candle(indicators_df, trade.entry_time)

    ema200 = adx14 = donchian_upper_prev = donchian_lower_prev = None
    if candle is not None:
        ema200 = _none_if_nan(candle.get("ema200"))
        adx14 = _none_if_nan(candle.get("adx"))
        donchian_upper_prev = _none_if_nan(candle.get("donchian_upper_prev"))
        donchian_lower_prev = _none_if_nan(candle.get("donchian_lower_prev"))

    return EntryContext(
        trade_number=trade_number,
        pair=trade.pair,
        direction=trade.direction,
        entry_time=trade.entry_time,
        entry_price=trade.entry_price,
        ema200=ema200,
        adx14=adx14,
        donchian_upper_prev=donchian_upper_prev,
        donchian_lower_prev=donchian_lower_prev,
        ema_distance_pct=compute_ema_distance_pct(trade.entry_price, ema200, trade.direction),
        breakout_distance_pct=compute_breakout_distance_pct(
            trade.entry_price, donchian_upper_prev, donchian_lower_prev, trade.direction
        ),
        enter_tag=trade.enter_tag,
        exit_reason=trade.exit_reason,
        profit_abs=trade.profit_abs,
        profit_pct=trade.profit_pct,
        duration_minutes=trade.duration_minutes,
        is_winner=trade.is_winner,
        candle_matched=candle is not None,
    )


def reconstruct_all(
    trade_report: TradeReport, indicators_by_pair: dict[str, pd.DataFrame]
) -> list[EntryContext]:
    """Reconstruct an `EntryContext` for every trade in `trade_report`, in order."""
    return [
        reconstruct_entry_context(i, trade, indicators_by_pair.get(trade.pair))
        for i, trade in enumerate(trade_report.trades, start=1)
    ]


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReconciliationResult:
    expected: int
    matched: int
    unmatched_trade_numbers: tuple[int, ...]

    @property
    def complete(self) -> bool:
        return self.matched == self.expected and not self.unmatched_trade_numbers


def reconcile(contexts: list[EntryContext], *, expected: int) -> ReconciliationResult:
    unmatched = tuple(c.trade_number for c in contexts if not c.candle_matched)
    matched = len(contexts) - len(unmatched)
    return ReconciliationResult(expected=expected, matched=matched, unmatched_trade_numbers=unmatched)


# ---------------------------------------------------------------------------
# Descriptive summaries (no interpretation -- see module/CLI docstrings)
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _group_summary(group: list[EntryContext]) -> dict[str, Any]:
    adx_values = [c.adx14 for c in group if c.adx14 is not None]
    ema_dist_values = [c.ema_distance_pct for c in group if c.ema_distance_pct is not None]
    breakout_dist_values = [
        c.breakout_distance_pct for c in group if c.breakout_distance_pct is not None
    ]
    known_outcome = [c for c in group if c.is_winner is not None]
    winners = [c for c in known_outcome if c.is_winner]
    profit_values = [c.profit_abs for c in group if c.profit_abs is not None]
    profit_pct_values = [c.profit_pct for c in group if c.profit_pct is not None]

    return {
        "trade_count": len(group),
        "average_adx": _mean(adx_values),
        "median_adx": _median(adx_values),
        "average_ema_distance_pct": _mean(ema_dist_values),
        "median_ema_distance_pct": _median(ema_dist_values),
        "average_breakout_distance_pct": _mean(breakout_dist_values),
        "win_rate_pct": (100.0 * len(winners) / len(known_outcome)) if known_outcome else None,
        "total_profit_abs": sum(profit_values) if profit_values else None,
        "average_profit_pct": _mean(profit_pct_values),
    }


def summarize_by_direction(contexts: list[EntryContext]) -> dict[str, dict[str, Any]]:
    return {
        direction: _group_summary([c for c in contexts if c.direction == direction])
        for direction in ("LONG", "SHORT")
    }


def summarize_by_outcome(contexts: list[EntryContext]) -> dict[str, dict[str, Any]]:
    return {
        "WINNERS": _group_summary([c for c in contexts if c.is_winner is True]),
        "LOSERS": _group_summary([c for c in contexts if c.is_winner is False]),
    }


def summarize_by_exit_reason(contexts: list[EntryContext]) -> dict[str, dict[str, Any]]:
    reasons = sorted({c.exit_reason for c in contexts if c.exit_reason is not None})
    return {reason: _group_summary([c for c in contexts if c.exit_reason == reason]) for reason in reasons}


def summarize_by_pair(contexts: list[EntryContext]) -> dict[str, dict[str, Any]]:
    pairs = sorted({c.pair for c in contexts if c.pair is not None})
    return {pair: _group_summary([c for c in contexts if c.pair == pair]) for pair in pairs}


def summarize_by_enter_tag(contexts: list[EntryContext]) -> dict[str, dict[str, Any]]:
    tags = sorted({c.enter_tag for c in contexts if c.enter_tag is not None})
    return {tag: _group_summary([c for c in contexts if c.enter_tag == tag]) for tag in tags}


# ---------------------------------------------------------------------------
# Persistence (write the forensic dataset -- never the original trade export)
# ---------------------------------------------------------------------------


def build_forensic_dataset(
    contexts: list[EntryContext],
    *,
    strategy: str,
    timeframe: str,
    timerange: str,
) -> dict[str, Any]:
    """Assemble the full forensic payload, ready for `json.dumps`."""
    return {
        "strategy": strategy,
        "timeframe": timeframe,
        "timerange": timerange,
        "trades": [asdict(c) for c in contexts],
        "summary": {
            "by_direction": summarize_by_direction(contexts),
            "by_outcome": summarize_by_outcome(contexts),
            "by_exit_reason": summarize_by_exit_reason(contexts),
            "by_pair": summarize_by_pair(contexts),
            "by_enter_tag": summarize_by_enter_tag(contexts),
        },
    }


def save_forensic_dataset(dataset: dict[str, Any], output_path: Path) -> None:
    """Write `dataset` as JSON to `output_path`. Caller is responsible for
    validating `output_path` is inside persistent storage (see
    `hermes.export_paths.prepare_export_directory`) -- this function only
    writes; it makes no persistence judgment of its own."""
    output_path = Path(output_path)
    output_path.write_text(json.dumps(dataset, indent=2, default=str))
