"""EMA200-distance entry-ceiling forensics (research/backtest only).

Tests whether filtering `TrendFollowCore` entries by their SIGNAL-CANDLE
distance from EMA200 (``abs(close - ema200) / ema200 * 100``) is associated
with better outcomes. This module never modifies `TrendFollowCore.py` and
never chooses/recommends a final threshold -- it only computes the pure
arithmetic and diagnostics a human report is built from.

Two independent things live here:

1. **Pure EMA-distance / threshold-eligibility helpers**, reusing
   `hermes.signal_forensics.compute_ema_distance_pct` (already the
   established, tested definition of directional EMA distance for this
   project) and wrapping it as an absolute percentage plus a simple
   ``<= threshold`` eligibility check -- exactly the definition specified
   for this forensic block: ``abs(close - ema200) / ema200 * 100``,
   evaluated at the signal candle, symmetric for LONG and SHORT.
2. **Trade-fate diff logic**: given the frozen baseline's 39
   `EntryContext`-shaped trades (each carrying its own signal-candle EMA
   distance) and a candidate ceiling, classify every trade as *kept* or
   *eliminated*, and every eliminated trade as
   winner/loser x stop_loss/exit_signal/other -- the exact breakdown the
   spec's TRADE-FATE ANALYSIS and STOP-LOSS TRADE IMPACT sections need.

Nothing here launches a subprocess or a Freqtrade backtest -- that
happens separately, once per threshold, via `hermes.backtest.BacktestLauncher`
against small `populate_entry_trend`-overriding subclass strategy files
(never edits to `TrendFollowCore.py` itself). This module only computes
the ceiling arithmetic and the fate diagnostics both the deploy script and
this module's own tests exercise directly, with no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hermes.signal_forensics import compute_ema_distance_pct

# ---------------------------------------------------------------------------
# EMA-distance ceiling arithmetic (pure)
# ---------------------------------------------------------------------------


def compute_abs_ema_distance_pct(
    close: float | None, ema200: float | None, direction: str | None
) -> float | None:
    """``abs(close - ema200) / ema200 * 100`` -- direction-agnostic extension.

    Reuses `compute_ema_distance_pct` (already the project's tested,
    directional definition -- positive when price is on the "correct"
    side of EMA200 for `direction`) rather than re-deriving the
    close/ema200 arithmetic a second time, then takes the absolute value
    and rescales to a percentage (0-100 range, not a 0-1 fraction) per
    this block's explicit spec: ``abs(close - ema200) / ema200 * 100``.
    """
    fractional = compute_ema_distance_pct(close, ema200, direction)
    if fractional is None:
        return None
    return abs(fractional) * 100.0


def passes_ceiling(ema_distance_pct: float | None, threshold_pct: float | None) -> bool:
    """`True` iff the trade is eligible under `threshold_pct`.

    `threshold_pct is None` means "no ceiling" (the baseline variant) --
    always eligible. A trade whose distance couldn't be computed
    (`ema_distance_pct is None`) is never eligible under any real ceiling:
    an unknown extension can't be proven `<= threshold`.
    """
    if threshold_pct is None:
        return True
    if ema_distance_pct is None:
        return False
    return ema_distance_pct <= threshold_pct


# ---------------------------------------------------------------------------
# Trade-fate diff: baseline trade -> kept/eliminated under a ceiling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineTradeRecord:
    """One frozen baseline trade, with its signal-candle EMA distance already
    known (from `hermes.signal_forensics`/the persisted forensics JSON) --
    this dataclass never recomputes it."""

    trade_number: int
    pair: str | None
    direction: str | None
    exit_reason: str | None
    profit_pct: float | None
    is_winner: bool | None
    signal_ema_distance_pct: float | None


@dataclass(frozen=True)
class TradeFateEntry:
    """A `BaselineTradeRecord` classified against one threshold."""

    trade: BaselineTradeRecord
    threshold_pct: float | None
    kept: bool

    @property
    def eliminated(self) -> bool:
        return not self.kept


def classify_trade(record: BaselineTradeRecord, threshold_pct: float | None) -> TradeFateEntry:
    kept = passes_ceiling(record.signal_ema_distance_pct, threshold_pct)
    return TradeFateEntry(trade=record, threshold_pct=threshold_pct, kept=kept)


def classify_all(
    records: list[BaselineTradeRecord], threshold_pct: float | None
) -> list[TradeFateEntry]:
    return [classify_trade(r, threshold_pct) for r in records]


@dataclass(frozen=True)
class ThresholdFateSummary:
    """The TRADE-FATE ANALYSIS / STOP-LOSS TRADE IMPACT / WINNER REMOVAL
    breakdown for one threshold, derived purely from `TradeFateEntry` list."""

    threshold_pct: float | None
    kept_count: int
    eliminated_count: int
    eliminated_winners: tuple[BaselineTradeRecord, ...]
    eliminated_losers: tuple[BaselineTradeRecord, ...]
    eliminated_stop_loss: tuple[BaselineTradeRecord, ...]
    eliminated_exit_signal: tuple[BaselineTradeRecord, ...]
    eliminated_other_exit: tuple[BaselineTradeRecord, ...]

    @property
    def stop_trades_prevented(self) -> int:
        return len(self.eliminated_stop_loss)

    @property
    def winners_prevented(self) -> int:
        return len(self.eliminated_winners)


def summarize_fate(entries: list[TradeFateEntry]) -> ThresholdFateSummary:
    if not entries:
        raise ValueError("summarize_fate requires at least one TradeFateEntry")
    threshold_pct = entries[0].threshold_pct
    eliminated = [e.trade for e in entries if e.eliminated]
    kept = [e.trade for e in entries if e.kept]
    return ThresholdFateSummary(
        threshold_pct=threshold_pct,
        kept_count=len(kept),
        eliminated_count=len(eliminated),
        eliminated_winners=tuple(t for t in eliminated if t.is_winner is True),
        eliminated_losers=tuple(t for t in eliminated if t.is_winner is False),
        eliminated_stop_loss=tuple(t for t in eliminated if t.exit_reason == "stop_loss"),
        eliminated_exit_signal=tuple(t for t in eliminated if t.exit_reason == "exit_signal"),
        eliminated_other_exit=tuple(
            t for t in eliminated
            if t.exit_reason not in ("stop_loss", "exit_signal")
        ),
    )


# ---------------------------------------------------------------------------
# Threshold-curve aggregate stats (pure, computed over *kept* trades)
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


@dataclass(frozen=True)
class ThresholdAggregate:
    threshold_pct: float | None
    trades: int
    winners: int
    losers: int
    win_rate_pct: float | None
    total_profit_pct: float | None
    average_profit_pct: float | None


def aggregate_kept_trades(entries: list[TradeFateEntry]) -> ThresholdAggregate:
    kept = [e.trade for e in entries if e.kept]
    threshold_pct = entries[0].threshold_pct if entries else None
    known = [t for t in kept if t.is_winner is not None]
    winners = sum(1 for t in known if t.is_winner)
    losers = sum(1 for t in known if not t.is_winner)
    profit_values = [t.profit_pct for t in kept if t.profit_pct is not None]
    return ThresholdAggregate(
        threshold_pct=threshold_pct,
        trades=len(kept),
        winners=winners,
        losers=losers,
        win_rate_pct=(100.0 * winners / len(known)) if known else None,
        total_profit_pct=sum(profit_values) if profit_values else None,
        average_profit_pct=_mean(profit_values),
    )


# ---------------------------------------------------------------------------
# Loading BaselineTradeRecord list from the persisted signal-forensics JSON
# ---------------------------------------------------------------------------


def baseline_records_from_forensics_json(payload: dict[str, Any]) -> list[BaselineTradeRecord]:
    """Build `BaselineTradeRecord`s from `hermes.signal_forensics.build_forensic_dataset`'s
    JSON shape (the already-persisted frozen-baseline forensics file).

    Uses each trade's already-computed `ema_distance_pct` (directional, per
    `compute_ema_distance_pct`) and takes its absolute value * 100 to match
    this block's percentage convention -- never recomputes EMA/close from
    OHLCV here, since the forensics file already did that read-only
    reconstruction once.
    """
    records = []
    for t in payload.get("trades", []):
        directional = t.get("ema_distance_pct")
        abs_pct = abs(directional) * 100.0 if directional is not None else None
        records.append(
            BaselineTradeRecord(
                trade_number=t["trade_number"],
                pair=t.get("pair"),
                direction=t.get("direction"),
                exit_reason=t.get("exit_reason"),
                profit_pct=t.get("profit_pct"),
                is_winner=t.get("is_winner"),
                signal_ema_distance_pct=abs_pct,
            )
        )
    return records
