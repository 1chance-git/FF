"""Hyperliquid candleSnapshot -> Freqtrade OHLCV data-format proof.

Standalone, research-only. Talks to Hyperliquid's public ``info`` API
directly (no exchange credentials, no orders, no CCXT/Freqtrade
exchange connection) and writes the result through Freqtrade's own
data handler so the resulting file is byte-for-byte what
``freqtrade download-data`` would have produced, had Hyperliquid
supported it (see RAILWAY.md's "Hyperliquid reachability" finding for
why this exists: Freqtrade's own ``download-data`` command does not
support Hyperliquid, even though Hyperliquid itself is reachable and
returns valid candle history).

Originally proved narrowly (BTC, 5m only); generalized to cover
BTC/ETH/SOL across 5m/1h/4h/1d once that single-pair/single-timeframe
proof succeeded end-to-end on Railway against live Hyperliquid data
(see RAILWAY.md). The per-request logic (pagination, dedup, validation,
Freqtrade format/read-back) is unchanged from the proven version --
``run_pipeline`` still handles exactly one (coin, interval) pair per
call; ``run_pipeline_matrix`` is the only new piece, and it just calls
``run_pipeline`` once per (coin, interval) combination.

This module is never imported by ``StatArbSwing``, ``hermes``, or any
live-trading path. Running it as a script only ever performs read-only
HTTP GETs/POSTs against Hyperliquid's public market-data endpoint and
local filesystem writes under a research data directory -- it never
places orders and never touches exchange credentials.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)

HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"

# Freqtrade's own OHLCV column order/names -- kept as a module constant
# (rather than re-derived from FeatherDataHandler at call time) so the
# pure conversion/validation functions below don't need Freqtrade
# importable to be unit tested.
OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

_INTERVAL_MS = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}

# The generalized scope, proven one (coin, interval) pair at a time via
# run_pipeline before being run as this full matrix (see RAILWAY.md's
# OHLCV audit, which already confirmed all 12 combinations return valid
# candle history from Hyperliquid's native API).
COINS = ["BTC", "ETH", "SOL"]
INTERVALS = ["5m", "1h", "4h", "1d"]
PAIR_FOR_COIN = {
    "BTC": "BTC/USDC:USDC",
    "ETH": "ETH/USDC:USDC",
    "SOL": "SOL/USDC:USDC",
}


class HyperliquidAPIError(RuntimeError):
    """Raised when Hyperliquid's ``info`` endpoint returns something this
    module can't safely treat as a list of candles."""


def interval_to_ms(interval: str) -> int:
    """Milliseconds per candle for a Hyperliquid interval string.

    Only used to validate spacing and to step ``endTime`` backward
    between pagination requests -- not a general timeframe parser.
    """
    try:
        return _INTERVAL_MS[interval]
    except KeyError:
        raise ValueError(
            f"Unsupported interval {interval!r}; supported: {sorted(_INTERVAL_MS)}"
        ) from None


def fetch_candle_snapshot(
    coin: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    url: str = HYPERLIQUID_INFO_URL,
    timeout: float = 25.0,
) -> list[dict[str, Any]]:
    """One raw call to Hyperliquid's ``candleSnapshot`` info endpoint.

    Returns the raw list of candle dicts exactly as Hyperliquid sends
    them (string-valued o/h/l/c/v, integer ms t/T) -- no conversion,
    dedup, or validation here. Raises ``HyperliquidAPIError`` if the
    response isn't a JSON list (Hyperliquid returns a JSON object with
    an error message, or an empty body, for malformed requests).
    """
    body = json.dumps(
        {
            "type": "candleSnapshot",
            "req": {"coin": coin, "interval": interval, "startTime": start_ms, "endTime": end_ms},
        }
    ).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            raw = resp.read()
    except urllib.error.URLError as exc:
        raise HyperliquidAPIError(f"request to {url} failed: {exc}") from exc

    if status != 200:
        raise HyperliquidAPIError(f"unexpected HTTP status {status} from {url}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HyperliquidAPIError(f"response body was not valid JSON: {exc}") from exc

    if not isinstance(data, list):
        raise HyperliquidAPIError(
            f"expected a JSON list of candles, got {type(data).__name__}: {str(data)[:200]}"
        )
    return data


FetchFn = Callable[..., list[dict[str, Any]]]


@dataclass
class PaginationResult:
    raw_candles: list[dict[str, Any]]
    requests_made: int


def paginate_raw(
    coin: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    request_window_ms: int | None = None,
    max_requests: int = 50,
    fetch: FetchFn = fetch_candle_snapshot,
) -> PaginationResult:
    """Walk ``[start_ms, end_ms]`` backward in ``request_window_ms``-sized
    chunks, never assuming a single request covers the whole range.

    ``request_window_ms`` defaults to the full ``[start_ms, end_ms]``
    span in one request (matching what Hyperliquid itself would return
    up to its own ~5000-candle cap -- see RAILWAY.md's OHLCV audit).
    Passing a smaller window is what actually exercises pagination:
    each request only asks for its own slice, and this function
    stitches the slices back together.

    Returns the raw candles exactly as received (may contain duplicate
    boundary candles between adjacent pages -- see ``dedupe_and_sort``)
    plus how many requests it took, so callers can report both the
    pagination behavior and the dedup work separately.
    """
    if start_ms >= end_ms:
        raise ValueError(f"start_ms ({start_ms}) must be < end_ms ({end_ms})")
    if request_window_ms is not None and request_window_ms <= 0:
        raise ValueError(f"request_window_ms must be positive, got {request_window_ms}")

    window = request_window_ms if request_window_ms is not None else (end_ms - start_ms)

    raw_candles: list[dict[str, Any]] = []
    cursor_end = end_ms
    requests_made = 0

    while cursor_end > start_ms:
        if requests_made >= max_requests:
            raise HyperliquidAPIError(
                f"exceeded max_requests={max_requests} while paginating "
                f"{coin} {interval} [{start_ms}, {end_ms}]; refusing to loop further"
            )
        cursor_start = max(start_ms, cursor_end - window)
        page = fetch(coin, interval, cursor_start, cursor_end)
        requests_made += 1
        raw_candles.extend(page)

        if not page:
            # No data in this slice (e.g. before the exchange existed) --
            # nothing more to learn by continuing to page backward from here.
            break

        earliest_t = min(c["t"] for c in page)
        if earliest_t >= cursor_end:
            # No progress possible (e.g. a single candle spanning the
            # whole requested window) -- stop rather than loop forever.
            break
        cursor_end = earliest_t

    return PaginationResult(raw_candles=raw_candles, requests_made=requests_made)


def paginate_candles(
    coin: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    request_window_ms: int | None = None,
    max_requests: int = 50,
    fetch: FetchFn = fetch_candle_snapshot,
) -> list[dict[str, Any]]:
    """Same as ``paginate_raw``, but deduplicated and chronologically
    sorted -- the convenience entry point when the caller doesn't need
    the raw request count."""
    result = paginate_raw(
        coin,
        interval,
        start_ms,
        end_ms,
        request_window_ms=request_window_ms,
        max_requests=max_requests,
        fetch=fetch,
    )
    return dedupe_and_sort(result.raw_candles)


def dedupe_and_sort(raw_candles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate by open-time ``t`` (later occurrences win -- harmless
    since Hyperliquid returns identical data for the same ``t``/interval,
    but deterministic either way) and sort chronologically."""
    by_open_time: dict[int, dict[str, Any]] = {}
    for candle in raw_candles:
        by_open_time[candle["t"]] = candle
    return sorted(by_open_time.values(), key=lambda c: c["t"])


def to_dataframe(raw_candles: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert Hyperliquid's raw candle dicts to Freqtrade's OHLCV shape.

    t (open ms) -> date (UTC, tz-aware), o/h/l/c/v (strings) -> floats.
    Assumes ``raw_candles`` is already deduplicated/sorted (see
    ``dedupe_and_sort``) -- this function doesn't re-check that.
    """
    if not raw_candles:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    df = pd.DataFrame(
        {
            "date": [pd.Timestamp(c["t"], unit="ms", tz=timezone.utc) for c in raw_candles],
            "open": [float(c["o"]) for c in raw_candles],
            "high": [float(c["h"]) for c in raw_candles],
            "low": [float(c["l"]) for c in raw_candles],
            "close": [float(c["c"]) for c in raw_candles],
            "volume": [float(c["v"]) for c in raw_candles],
        }
    )
    return df[OHLCV_COLUMNS]


@dataclass
class ValidationResult:
    ok: bool
    issues: list[str] = field(default_factory=list)


def validate_ohlcv(df: pd.DataFrame, interval: str) -> ValidationResult:
    """Check the invariants the task spec calls out explicitly:
    no duplicate timestamps, chronological order, valid OHLC relationships,
    non-negative volume, and expected candle spacing.

    Returns a report rather than raising, so callers can decide whether
    a given violation (e.g. one exchange-downtime gap) is fatal.
    """
    issues: list[str] = []

    if df.empty:
        return ValidationResult(ok=False, issues=["dataframe is empty"])

    if df["date"].duplicated().any():
        dupes = int(df["date"].duplicated().sum())
        issues.append(f"{dupes} duplicate timestamp(s)")

    if not df["date"].is_monotonic_increasing:
        issues.append("timestamps are not strictly chronological")

    bad_ohlc = df[(df["high"] < df["low"]) | (df["high"] < df["open"]) | (df["high"] < df["close"])
                  | (df["low"] > df["open"]) | (df["low"] > df["close"])]
    if not bad_ohlc.empty:
        issues.append(f"{len(bad_ohlc)} row(s) with invalid OHLC relationships (high/low bounds violated)")

    non_positive_price = df[(df[["open", "high", "low", "close"]] <= 0).any(axis=1)]
    if not non_positive_price.empty:
        issues.append(f"{len(non_positive_price)} row(s) with non-positive price")

    negative_volume = df[df["volume"] < 0]
    if not negative_volume.empty:
        issues.append(f"{len(negative_volume)} row(s) with negative volume")

    if len(df) > 1:
        expected_step = pd.Timedelta(milliseconds=interval_to_ms(interval))
        deltas = df["date"].diff().dropna()
        wrong_spacing = deltas[deltas != expected_step]
        if not wrong_spacing.empty:
            issues.append(
                f"{len(wrong_spacing)} gap(s) not equal to the expected {interval} spacing "
                f"(min gap {deltas.min()}, max gap {deltas.max()})"
            )

    return ValidationResult(ok=not issues, issues=issues)


def save_to_freqtrade(
    df: pd.DataFrame,
    *,
    datadir: Path,
    pair: str,
    timeframe: str,
    candle_type: str = "futures",
) -> Path:
    """Write ``df`` via Freqtrade's own data handler (feather format),
    so the file is indistinguishable from one ``freqtrade download-data``
    would have produced. Imports Freqtrade lazily so this module's pure
    functions stay importable/testable without Freqtrade installed.
    """
    from freqtrade.data.history.datahandlers.idatahandler import get_datahandler
    from freqtrade.enums import CandleType

    handler = get_datahandler(datadir, data_format="feather")
    handler.ohlcv_store(pair, timeframe, df, CandleType(candle_type))
    return handler._pair_data_filename(datadir, pair, timeframe, CandleType(candle_type))


def read_back_from_freqtrade(
    *,
    datadir: Path,
    pair: str,
    timeframe: str,
    candle_type: str = "futures",
) -> pd.DataFrame:
    """Read the just-written file back through Freqtrade's own data
    handler -- the actual proof that Freqtrade can read what this module
    wrote, not just that a file exists on disk."""
    from freqtrade.data.history.datahandlers.idatahandler import get_datahandler
    from freqtrade.enums import CandleType

    handler = get_datahandler(datadir, data_format="feather")
    return handler.ohlcv_load(
        pair,
        timeframe,
        CandleType(candle_type),
        fill_missing=False,
        drop_incomplete=False,
        warn_no_data=True,
    )


@dataclass
class PipelineResult:
    api_ok: bool
    download_ok: bool
    pagination_requests: int
    dedup_removed: int
    validation: ValidationResult
    freqtrade_format_ok: bool
    freqtrade_readback_ok: bool
    candle_count: int
    first_date: datetime | None
    last_date: datetime | None
    output_path: Path | None


def run_pipeline(
    *,
    coin: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    pair: str,
    datadir: Path,
    request_window_ms: int | None = None,
    candle_type: str = "futures",
    fetch: FetchFn = fetch_candle_snapshot,
) -> PipelineResult:
    """The end-to-end proof: Hyperliquid API -> historical candles ->
    Freqtrade data format -> Freqtrade can read the data. Used by both
    the tests (with a mocked ``fetch``) and the ``__main__`` script
    (with the real one)."""
    try:
        pagination = paginate_raw(
            coin, interval, start_ms, end_ms, request_window_ms=request_window_ms, fetch=fetch
        )
        api_ok = True
    except HyperliquidAPIError:
        return PipelineResult(
            api_ok=False,
            download_ok=False,
            pagination_requests=0,
            dedup_removed=0,
            validation=ValidationResult(ok=False, issues=["API call(s) failed"]),
            freqtrade_format_ok=False,
            freqtrade_readback_ok=False,
            candle_count=0,
            first_date=None,
            last_date=None,
            output_path=None,
        )

    raw = dedupe_and_sort(pagination.raw_candles)
    dedup_removed = len(pagination.raw_candles) - len(raw)
    df = to_dataframe(raw)
    validation = validate_ohlcv(df, interval)

    output_path = None
    freqtrade_format_ok = False
    freqtrade_readback_ok = False
    if validation.ok:
        output_path = save_to_freqtrade(
            df, datadir=datadir, pair=pair, timeframe=interval, candle_type=candle_type
        )
        freqtrade_format_ok = output_path.exists()
        if freqtrade_format_ok:
            read_back = read_back_from_freqtrade(
                datadir=datadir, pair=pair, timeframe=interval, candle_type=candle_type
            )
            freqtrade_readback_ok = len(read_back) == len(df) and list(read_back.columns) == OHLCV_COLUMNS

    return PipelineResult(
        api_ok=api_ok,
        download_ok=len(raw) > 0,
        pagination_requests=pagination.requests_made,
        dedup_removed=dedup_removed,
        validation=validation,
        freqtrade_format_ok=freqtrade_format_ok,
        freqtrade_readback_ok=freqtrade_readback_ok,
        candle_count=len(df),
        first_date=df["date"].iloc[0].to_pydatetime() if not df.empty else None,
        last_date=df["date"].iloc[-1].to_pydatetime() if not df.empty else None,
        output_path=output_path,
    )


def default_window_for(coin: str, interval: str, *, candles_per_page: int = 5, total_candles: int = 13) -> tuple[int, int, int]:
    """Compute (start_ms, end_ms, request_window_ms) for a small,
    interval-scaled proof window: ``total_candles`` candles' worth of
    history, fetched ``candles_per_page`` at a time (so pagination is
    genuinely exercised at every interval, not just 5m). Scales with
    the interval so 1d doesn't request 13 days when 13 5m-candles is
    the intended proportion -- e.g. for "5m" this reproduces exactly
    the originally-proven 65-minute/25-minute-page window.
    """
    interval_ms = interval_to_ms(interval)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    end_ms = now_ms
    start_ms = now_ms - total_candles * interval_ms
    request_window_ms = candles_per_page * interval_ms
    return start_ms, end_ms, request_window_ms


def run_pipeline_matrix(
    *,
    coins: list[str] = COINS,
    intervals: list[str] = INTERVALS,
    pair_for_coin: dict[str, str] = PAIR_FOR_COIN,
    datadir: Path,
    candles_per_page: int = 5,
    total_candles: int = 13,
    candle_type: str = "futures",
    fetch: FetchFn = fetch_candle_snapshot,
) -> dict[tuple[str, str], PipelineResult]:
    """Run ``run_pipeline`` once per (coin, interval) combination in the
    matrix, each with its own interval-scaled small window (see
    ``default_window_for``). Same per-pair logic as the original
    single-pair proof -- this only adds the loop and result collection.
    """
    results: dict[tuple[str, str], PipelineResult] = {}
    for coin in coins:
        pair = pair_for_coin[coin]
        for interval in intervals:
            start_ms, end_ms, request_window_ms = default_window_for(
                coin, interval, candles_per_page=candles_per_page, total_candles=total_candles
            )
            results[(coin, interval)] = run_pipeline(
                coin=coin,
                interval=interval,
                start_ms=start_ms,
                end_ms=end_ms,
                pair=pair,
                datadir=datadir,
                request_window_ms=request_window_ms,
                candle_type=candle_type,
                fetch=fetch,
            )
    return results


def _print_result(coin: str, interval: str, result: PipelineResult) -> None:
    print(f"--- {coin} {interval} ---")
    print(f"API: {'PASS' if result.api_ok else 'FAIL'}")
    print(f"{coin} {interval} download: {'PASS' if result.download_ok else 'FAIL'}")
    print(f"Pagination: {'PASS' if result.pagination_requests > 1 else 'FAIL'} ({result.pagination_requests} requests)")
    print(f"Deduplication: PASS ({result.dedup_removed} duplicate candle(s) removed)")
    print(f"OHLCV validation: {'PASS' if result.validation.ok else 'FAIL'}")
    if not result.validation.ok:
        for issue in result.validation.issues:
            print(f"  - {issue}")
    print(f"Freqtrade format: {'PASS' if result.freqtrade_format_ok else 'FAIL'}")
    print(f"Freqtrade read-back: {'PASS' if result.freqtrade_readback_ok else 'FAIL'}")
    print(f"CANDLES: {result.candle_count}")
    print(f"DATE RANGE: {result.first_date} -> {result.last_date}")
    print(f"OUTPUT: {result.output_path}")
    print("")


def _main() -> None:
    """Small, controlled proof run across the full BTC/ETH/SOL x
    5m/1h/4h/1d matrix: 13 candles' worth of history per (coin,
    interval), fetched 5 candles per page (forcing real pagination at
    every interval) -- not years of data, per the task's explicit
    constraint. For "5m" this is exactly the originally-proven
    65-minute/25-minute-page window; other intervals scale the same
    proportion.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    datadir = Path("user_data/data/hyperliquid")
    print(f"[HL-PIPELINE] running matrix: coins={COINS} intervals={INTERVALS}")
    results = run_pipeline_matrix(datadir=datadir)

    print("[HYPERLIQUID][DATA PIPELINE]")
    for coin in COINS:
        for interval in INTERVALS:
            _print_result(coin, interval, results[(coin, interval)])

    all_pass = all(
        r.api_ok and r.download_ok and r.validation.ok and r.freqtrade_format_ok and r.freqtrade_readback_ok
        for r in results.values()
    )
    print(f"MATRIX RESULT: {'PASS' if all_pass else 'FAIL'} ({len(results)} combinations)")


if __name__ == "__main__":
    _main()
