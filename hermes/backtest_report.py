"""Read-only extraction of Freqtrade's own results from a recorded backtest.

`hermes.backtest.BacktestLauncher` already captures the `freqtrade
backtesting` subprocess' full stdout and persists it inside the
`metrics` JSON blob of every `hermes.memory.BacktestResult` row (see
that module's ``_record_result``). Freqtrade prints its complete
results — the per-pair breakdown and the summary-metrics table — to
exactly that stdout. So a completed backtest's real numbers are
*already recorded*; they were simply never surfaced, because
``hermes backtest`` only logs a one-line "succeeded/failed" summary.

This module closes that gap **without re-running anything**: it parses
the stored stdout and exposes the numbers Freqtrade already computed.
Nothing here launches a subprocess, touches an exchange, writes to the
memory database, or influences a trading decision — every function is
pure text-in/data-out, and the CLI command built on it
(``hermes backtest-report``) only ever reads.

Design decisions
-----------------
* **Parse Freqtrade's rendered tables, don't recompute anything.**
  Every number reported here is one Freqtrade itself printed. Hermes
  never re-derives win rate, drawdown, or profit factor from raw
  trades — that would risk quietly disagreeing with the engine's own
  accounting, which is the one number that matters.
* **The parser targets `rich` box-drawing tables, because that is what
  Freqtrade actually emits.** Freqtrade renders results via
  `freqtrade.util.rich_tables.print_rich_table`, so captured stdout
  contains `┃`-delimited header rows and `│`-delimited data rows inside
  `━`/`─` borders — not markdown pipes. The exact rendering was
  confirmed by calling Freqtrade's own `text_table_bt_results` /
  `text_table_add_metrics` formatters directly (a pure formatting call,
  no backtest) and matching this parser to their real output.
* **Tables are parsed generically, then selected by title.** Freqtrade
  prints several structurally identical tables ("BACKTESTING REPORT",
  "LEFT OPEN TRADES REPORT", tag stats, ...). Parsing all of them into
  a neutral `ParsedTable` list and *then* picking by title keeps the
  low-level parsing honest and makes each step independently testable,
  rather than entangling "how do I read a table" with "which table do
  I want".
* **Missing metrics stay missing.** Every accessor returns ``None``
  when Freqtrade didn't print that field (or printed ``N/A``). A
  research report that invents a plausible-looking number is worse
  than one that admits the gap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

# rich renders header rows with a heavy vertical bar and body rows with a
# light one; everything else in a table is border/box drawing.
_HEADER_SEP = "┃"
_ROW_SEP = "│"
_BOX_CHARS = frozenset("┏┓┗┛┡┩┢┪├┤┬┴┼╇╈╡╞─━│┃")

PAIR_TABLE_TITLE = "BACKTESTING REPORT"
SUMMARY_TABLE_TITLE = "SUMMARY METRICS"
TOTAL_ROW_KEY = "TOTAL"


@dataclass(frozen=True)
class ParsedTable:
    """One `rich` table recovered from captured stdout, as plain strings."""

    title: str | None
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class PairResult:
    """One row of Freqtrade's per-pair "BACKTESTING REPORT" table.

    Also used for that table's final ``TOTAL`` row, which has the same
    shape (see :attr:`BacktestReport.total`).
    """

    pair: str
    trades: int
    avg_profit_pct: float | None
    profit_total_abs: float | None
    profit_total_pct: float | None
    avg_duration: str
    wins: int
    draws: int
    losses: int
    win_rate_pct: float | None


@dataclass(frozen=True)
class BacktestReport:
    """Everything this module could recover from one backtest's stdout.

    ``summary_metrics`` maps Freqtrade's own metric labels (e.g.
    ``"Profit factor"``) to their printed value strings, verbatim.
    Accessors below normalize the handful of fields a research report
    usually wants; anything not printed comes back ``None``.
    """

    pair_results: tuple[PairResult, ...] = ()
    total: PairResult | None = None
    summary_metrics: Mapping[str, str] = field(default_factory=dict)

    @property
    def parsed_anything(self) -> bool:
        """``True`` if any results table was recovered at all.

        ``False`` means the stdout held no Freqtrade results — e.g. the
        backtest failed before producing any, or produced zero trades.
        """
        return bool(self.pair_results or self.total or self.summary_metrics)

    def metric(self, label: str) -> str | None:
        """Raw printed value for a summary-metrics ``label``, or ``None``.

        ``N/A`` (Freqtrade's own "not computed" marker) is normalized to
        ``None`` so callers never report it as if it were a value.
        """
        value = self.summary_metrics.get(label)
        if value is None or value.strip() in ("", "N/A"):
            return None
        return value.strip()

    # -- Normalized accessors (all ``None`` when Freqtrade didn't print it) --

    @property
    def total_trades(self) -> int | None:
        if self.total is not None:
            return self.total.trades
        # Fall back to the summary table's "Total/Daily Avg Trades" ("21 / 0.1").
        raw = self.metric("Total/Daily Avg Trades")
        if raw is None:
            return None
        return _to_int(raw.split("/")[0])

    @property
    def wins(self) -> int | None:
        return self.total.wins if self.total is not None else None

    @property
    def losses(self) -> int | None:
        return self.total.losses if self.total is not None else None

    @property
    def draws(self) -> int | None:
        return self.total.draws if self.total is not None else None

    @property
    def win_rate_pct(self) -> float | None:
        return self.total.win_rate_pct if self.total is not None else None

    @property
    def profit_total_abs(self) -> float | None:
        if self.total is not None and self.total.profit_total_abs is not None:
            return self.total.profit_total_abs
        return _to_float(self.metric("Absolute profit"))

    @property
    def profit_total_pct(self) -> float | None:
        if self.total is not None and self.total.profit_total_pct is not None:
            return self.total.profit_total_pct
        return _to_float(self.metric("Total profit %"))

    @property
    def profit_factor(self) -> float | None:
        return _to_float(self.metric("Profit factor"))

    @property
    def max_drawdown(self) -> str | None:
        """Freqtrade's "Absolute drawdown" line, e.g. ``"74.5 USDC (7.45%)"``.

        Kept as the printed string rather than split into number+percent:
        it carries both the absolute and account-relative figures, and
        reproducing it verbatim avoids implying a precision or a unit
        Freqtrade didn't state.
        """
        return self.metric("Absolute drawdown")

    @property
    def avg_trade_duration(self) -> str | None:
        if self.total is not None and self.total.avg_duration:
            return self.total.avg_duration
        return None

    @property
    def long_trades(self) -> int | None:
        raw = self.metric("Long / Short trades")
        return _to_int(raw.split("/")[0]) if raw else None

    @property
    def short_trades(self) -> int | None:
        raw = self.metric("Long / Short trades")
        parts = raw.split("/") if raw else []
        return _to_int(parts[1]) if len(parts) > 1 else None

    @property
    def starting_balance(self) -> str | None:
        return self.metric("Starting balance")

    @property
    def final_balance(self) -> str | None:
        return self.metric("Final balance")

    def trades_for_pair(self, pair: str) -> int | None:
        """Trade count for one pair, or ``None`` if that pair has no row."""
        for result in self.pair_results:
            if result.pair == pair:
                return result.trades
        return None


# ---------------------------------------------------------------------------
# Value coercion
# ---------------------------------------------------------------------------


def _to_int(text: str | None) -> int | None:
    """Best-effort int from a printed cell; ``None`` if not a number."""
    if text is None:
        return None
    match = re.search(r"-?\d+", text.replace(",", ""))
    return int(match.group()) if match else None


def _to_float(text: str | None) -> float | None:
    """Best-effort float from a printed cell; ``None`` if not a number.

    Handles Freqtrade's printed forms: bare numbers, percentages
    (``"4.17%"``), and currency-suffixed amounts (``"41.73 USDC"``).
    """
    if text is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(match.group()) if match else None


# ---------------------------------------------------------------------------
# Generic rich-table parsing
# ---------------------------------------------------------------------------


def _split_cells(line: str, separator: str) -> tuple[str, ...]:
    stripped = line.strip()
    if stripped.startswith(separator):
        stripped = stripped[len(separator) :]
    if stripped.endswith(separator):
        stripped = stripped[: -len(separator)]
    return tuple(cell.strip() for cell in stripped.split(separator))


def _first_char(line: str) -> str:
    stripped = line.strip()
    return stripped[0] if stripped else ""


def parse_rich_tables(text: str) -> list[ParsedTable]:
    """Recover every `rich` box-drawing table in ``text``.

    A table's title is the nearest preceding non-blank, non-table line
    (which is exactly where `rich` prints it). Tables without one get
    ``title=None``.
    """
    tables: list[ParsedTable] = []

    pending_title: str | None = None
    headers: tuple[str, ...] | None = None
    rows: list[tuple[str, ...]] = []
    in_table = False

    def flush() -> None:
        nonlocal headers, rows, in_table
        if headers is not None:
            tables.append(
                ParsedTable(title=pending_title, headers=headers, rows=tuple(rows))
            )
        headers = None
        rows = []
        in_table = False

    for line in text.splitlines():
        lead = _first_char(line)

        if lead == _HEADER_SEP:
            if headers is None:
                headers = _split_cells(line, _HEADER_SEP)
                in_table = True
            else:
                # A second heavy-bar row inside one table (rare); treat as data.
                rows.append(_split_cells(line, _HEADER_SEP))
            continue

        if lead == _ROW_SEP:
            if headers is None:
                # Body rows with no header row seen yet — still a table.
                headers = ()
                in_table = True
            rows.append(_split_cells(line, _ROW_SEP))
            continue

        if lead in _BOX_CHARS:
            # Border line: part of the table, carries no data.
            in_table = True
            continue

        # Not a table line.
        if in_table:
            flush()
        if line.strip():
            pending_title = line.strip()

    flush()
    return tables


def find_table(tables: list[ParsedTable], title: str) -> ParsedTable | None:
    """First table whose title matches ``title`` (case-insensitive)."""
    wanted = title.strip().lower()
    for table in tables:
        if table.title and table.title.strip().lower() == wanted:
            return table
    return None


# ---------------------------------------------------------------------------
# Freqtrade-specific extraction
# ---------------------------------------------------------------------------


def _parse_pair_row(row: tuple[str, ...]) -> PairResult | None:
    """Convert one "BACKTESTING REPORT" row into a :class:`PairResult`.

    Expected columns (Freqtrade's own order):
    ``Pair | Trades | Avg Profit % | Tot Profit <stake> | Tot Profit % |
    Avg Duration | Win Draw Loss Win%``.
    """
    if len(row) < 7 or not row[0]:
        return None

    trades = _to_int(row[1])
    if trades is None:
        return None

    # The final column packs four whitespace-separated values.
    wins = draws = losses = 0
    win_rate: float | None = None
    win_loss_parts = row[6].split()
    if len(win_loss_parts) >= 4:
        wins = _to_int(win_loss_parts[0]) or 0
        draws = _to_int(win_loss_parts[1]) or 0
        losses = _to_int(win_loss_parts[2]) or 0
        win_rate = _to_float(win_loss_parts[3])

    return PairResult(
        pair=row[0],
        trades=trades,
        avg_profit_pct=_to_float(row[2]),
        profit_total_abs=_to_float(row[3]),
        profit_total_pct=_to_float(row[4]),
        avg_duration=row[5],
        wins=wins,
        draws=draws,
        losses=losses,
        win_rate_pct=win_rate,
    )


def parse_backtest_stdout(stdout: str) -> BacktestReport:
    """Extract Freqtrade's own results from one backtest's captured stdout.

    Pure: no I/O, no subprocess, no database access. Returns an empty
    :class:`BacktestReport` (``parsed_anything is False``) if the stdout
    contains no recognizable results tables, rather than raising —
    a failed or zero-trade backtest is an expected input here, not an
    error condition.
    """
    if not stdout:
        return BacktestReport()

    tables = parse_rich_tables(stdout)

    pair_results: list[PairResult] = []
    total: PairResult | None = None

    pair_table = find_table(tables, PAIR_TABLE_TITLE)
    if pair_table is not None:
        for row in pair_table.rows:
            parsed = _parse_pair_row(row)
            if parsed is None:
                continue
            if parsed.pair == TOTAL_ROW_KEY:
                total = parsed
            else:
                pair_results.append(parsed)

    summary_metrics: dict[str, str] = {}
    summary_table = find_table(tables, SUMMARY_TABLE_TITLE)
    if summary_table is not None:
        for row in summary_table.rows:
            if len(row) < 2:
                continue
            label = row[0].strip()
            if not label:  # rich renders blank spacer rows between groups
                continue
            summary_metrics[label] = row[1].strip()

    return BacktestReport(
        pair_results=tuple(pair_results),
        total=total,
        summary_metrics=summary_metrics,
    )


def extract_stdout(metrics: Mapping[str, Any] | None) -> str | None:
    """Pull the captured stdout out of a persisted record's ``metrics`` blob.

    Mirrors the key `hermes.backtest.BacktestLauncher._record_result`
    writes. Returns ``None`` when the record predates that field or
    stored something non-textual, so callers can say so plainly instead
    of reporting an empty result as if it were a real one.
    """
    if not metrics:
        return None
    stdout = metrics.get("stdout")
    return stdout if isinstance(stdout, str) and stdout else None
