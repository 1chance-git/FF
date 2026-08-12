"""Read-only, per-trade forensic report over Freqtrade's native trade export.

Companion to `hermes.backtest_report`: that module parses the *aggregate*
tables Freqtrade prints to stdout (already captured and persisted by every
`hermes backtest` run). This module reads the *individual* trades Freqtrade
optionally writes as a JSON export file (requested via
`BacktestConfig.export_types`/`export_directory`, see `hermes.backtest`),
when a caller wants per-trade identification, entry/exit context, and
outcome rather than the pre-aggregated summary.

Nothing here launches a subprocess, computes a trading decision, or
re-derives a number Freqtrade already computed -- every field either comes
straight from the export file or is reported as unavailable. This module
also never assumes an exported file exists yet; `load_trades_export`
returning "no file found" is an expected, reportable outcome, not an
error to work around by launching a backtest.

Design decisions
-----------------
* **The exact JSON layout Freqtrade emits for `--export trades` has
  shifted across versions** (a bare list of trade dicts in some versions;
  a `{"strategy": {<name>: {"trades": [...]}}}` wrapper in others). Rather
  than hard-code one shape and silently mis-parse (or crash on) the other,
  `_extract_trade_dicts` recognizes both and raises a clear
  `TradeReportError` naming the actual top-level shape it found if
  neither matches -- so a genuine format change surfaces immediately
  instead of masquerading as "zero trades."
* **Every per-trade field is read defensively.** Freqtrade's own trade
  dict schema carries fields (`is_short`, `enter_tag`, `exit_reason`, ...)
  that can be absent depending on version/config; every accessor here
  returns `None` (rendered as `"N/A"`) rather than guessing a default,
  mirroring `hermes.backtest_report`'s "missing metrics stay missing"
  rule.
* **Aggregation here is a cross-check, not a replacement, for Freqtrade's
  own aggregate report.** `TradeReport`'s totals are computed directly
  from the same per-trade rows this module parses; they exist so a caller
  can sanity-check them against `hermes.backtest_report`'s parse of
  Freqtrade's printed summary, not to become the canonical source of
  aggregate metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class TradeReportError(Exception):
    """Raised when an export file exists but its shape can't be understood."""


@dataclass(frozen=True)
class Trade:
    """One backtested trade, as reported by Freqtrade's own trade export.

    Every field is `None` if the export didn't contain it -- never a
    fabricated or inferred value.
    """

    pair: str | None
    direction: str | None  # "LONG" / "SHORT" / None
    entry_time: str | None
    exit_time: str | None
    entry_price: float | None
    exit_price: float | None
    enter_tag: str | None
    exit_reason: str | None
    profit_abs: float | None
    profit_pct: float | None
    duration_minutes: float | None
    is_open: bool | None

    @property
    def is_winner(self) -> bool | None:
        """`True`/`False` from `profit_abs`, or `None` if unknown."""
        if self.profit_abs is None:
            return None
        return self.profit_abs > 0

    @property
    def is_stop_loss(self) -> bool | None:
        """`True` if `exit_reason` is Freqtrade's `stop_loss`, `None` if unknown."""
        if self.exit_reason is None:
            return None
        return self.exit_reason == "stop_loss"


@dataclass(frozen=True)
class TradeReport:
    """All trades recovered from one export file, plus a cross-check aggregate."""

    trades: tuple[Trade, ...] = field(default_factory=tuple)
    source_path: Path | None = None

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winners(self) -> int:
        return sum(1 for t in self.trades if t.is_winner is True)

    @property
    def losers(self) -> int:
        return sum(1 for t in self.trades if t.is_winner is False)

    @property
    def win_rate_pct(self) -> float | None:
        known = [t for t in self.trades if t.is_winner is not None]
        if not known:
            return None
        return 100.0 * sum(1 for t in known if t.is_winner) / len(known)

    @property
    def long_trades(self) -> int:
        return sum(1 for t in self.trades if t.direction == "LONG")

    @property
    def short_trades(self) -> int:
        return sum(1 for t in self.trades if t.direction == "SHORT")

    def trades_for_pair(self, pair: str) -> tuple[Trade, ...]:
        return tuple(t for t in self.trades if t.pair == pair)

    def exit_reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in self.trades:
            key = t.exit_reason or "N/A"
            counts[key] = counts.get(key, 0) + 1
        return counts

    def enter_tag_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in self.trades:
            key = t.enter_tag or "N/A"
            counts[key] = counts.get(key, 0) + 1
        return counts

    def per_pair_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in self.trades:
            key = t.pair or "N/A"
            counts[key] = counts.get(key, 0) + 1
        return counts


def find_latest_export_file(export_directory: Path) -> Path | None:
    """The most recently modified `backtest-result-*.json` under `export_directory`.

    Returns `None` (not an error) if the directory doesn't exist yet or
    contains no matching file -- a caller asking before any backtest has
    exported anything is an expected state, not a failure.
    """
    export_directory = Path(export_directory)
    if not export_directory.is_dir():
        return None

    candidates = [
        p
        for p in export_directory.glob("backtest-result-*.json")
        if not p.name.endswith(".meta.json")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _extract_trade_dicts(payload: Any, *, strategy: str | None) -> list[dict[str, Any]]:
    """Recover the raw list of trade dicts from either export shape Freqtrade uses."""
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        strategy_block = payload.get("strategy")
        if isinstance(strategy_block, dict):
            if strategy is not None and strategy in strategy_block:
                inner = strategy_block[strategy]
            elif len(strategy_block) == 1:
                inner = next(iter(strategy_block.values()))
            else:
                raise TradeReportError(
                    "export file has multiple strategies recorded "
                    f"({sorted(strategy_block.keys())}) and no `strategy` filter "
                    "was given to disambiguate"
                )
            trades = inner.get("trades") if isinstance(inner, dict) else None
            if isinstance(trades, list):
                return trades
        if isinstance(payload.get("trades"), list):
            return payload["trades"]

    raise TradeReportError(
        f"unrecognized export file shape (top-level type: {type(payload).__name__}); "
        "expected a list of trade dicts, or a dict with a 'trades' or "
        "'strategy' key"
    )


def _direction(raw: dict[str, Any]) -> str | None:
    is_short = raw.get("is_short")
    if is_short is None:
        return None
    return "SHORT" if is_short else "LONG"


def _parse_trade(raw: dict[str, Any]) -> Trade:
    return Trade(
        pair=raw.get("pair"),
        direction=_direction(raw),
        entry_time=raw.get("open_date"),
        exit_time=raw.get("close_date"),
        entry_price=raw.get("open_rate"),
        exit_price=raw.get("close_rate"),
        enter_tag=raw.get("enter_tag"),
        exit_reason=raw.get("exit_reason"),
        profit_abs=raw.get("profit_abs"),
        profit_pct=(
            raw["profit_ratio"] * 100.0 if raw.get("profit_ratio") is not None else None
        ),
        duration_minutes=raw.get("trade_duration"),
        is_open=raw.get("is_open"),
    )


def load_trades_export(path: Path, *, strategy: str | None = None) -> TradeReport:
    """Parse a Freqtrade trade export JSON file into a `TradeReport`.

    Parameters
    ----------
    path:
        Path to a `backtest-result-*.json` file (not its `.meta.json`
        sibling).
    strategy:
        Which strategy's trades to read, if the export file recorded more
        than one. Ignored for the flat (single-strategy-implicit) export
        shape.

    Raises
    ------
    TradeReportError
        If the file isn't valid JSON, or its shape isn't recognized.
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise TradeReportError(f"cannot read export file {path}: {exc}") from exc

    raw_trades = _extract_trade_dicts(payload, strategy=strategy)
    return TradeReport(trades=tuple(_parse_trade(t) for t in raw_trades), source_path=path)


def _fmt(value: Any) -> str:
    return "N/A" if value is None else str(value)


def render_trade_report(report: TradeReport) -> str:
    """Format a `TradeReport` as the plain-text report `hermes trade-report` prints.

    Pure formatting: every value comes straight from `report`, computes
    nothing new, and reports `"N/A"` for anything `None` -- never a
    fabricated value.
    """
    lines = ["[TREND][TRADE FORENSICS]", ""]

    if not report.trades:
        lines.append("No trades found in the export file.")
        return "\n".join(lines)

    for i, t in enumerate(report.trades, start=1):
        lines.append(f"TRADE {i}")
        lines.append(f"PAIR: {_fmt(t.pair)}")
        lines.append(f"DIRECTION: {_fmt(t.direction)}")
        lines.append(f"ENTRY TIME: {_fmt(t.entry_time)}")
        lines.append(f"EXIT TIME: {_fmt(t.exit_time)}")
        lines.append(f"ENTRY PRICE: {_fmt(t.entry_price)}")
        lines.append(f"EXIT PRICE: {_fmt(t.exit_price)}")
        lines.append(f"ENTRY TAG: {_fmt(t.enter_tag)}")
        lines.append(f"EXIT REASON: {_fmt(t.exit_reason)}")
        lines.append(f"PROFIT: {_fmt(t.profit_abs)}")
        lines.append(f"PROFIT %: {_fmt(t.profit_pct)}")
        lines.append(f"DURATION (min): {_fmt(t.duration_minutes)}")
        lines.append("")

    lines.append("TOTAL TRADES: " + str(report.total_trades))
    lines.append("WINNERS: " + str(report.winners))
    lines.append("LOSERS: " + str(report.losers))
    lines.append("WIN RATE: " + _fmt(report.win_rate_pct))
    lines.append("LONG TRADES: " + str(report.long_trades))
    lines.append("SHORT TRADES: " + str(report.short_trades))

    lines.append("EXIT REASONS:")
    for reason, count in sorted(report.exit_reason_counts().items()):
        lines.append(f"  {reason}: {count}")

    lines.append("ENTRY TAGS:")
    for tag, count in sorted(report.enter_tag_counts().items()):
        lines.append(f"  {tag}: {count}")

    lines.append("PER-PAIR RESULTS:")
    for pair, count in sorted(report.per_pair_counts().items()):
        lines.append(f"  {pair}: {count}")

    return "\n".join(lines)
