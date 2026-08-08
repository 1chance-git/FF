"""Reusable market data layer for the stat-arb system.

Loads OHLCV candle data for the project's tradable pairs (currently
``BTC/USDC:USDC`` and ``ETH/USDC:USDC``), cleans and fills gaps in it,
aligns multiple pairs onto a shared timestamp index, and validates the
resulting dataframes before they are handed to any downstream module.

This module intentionally contains **no indicators or trading logic** —
that is out of scope for the market data layer.

Design decisions
-----------------
* **Freqtrade's own history/converter utilities do the heavy lifting.**
  ``freqtrade.data.history.load_pair_history`` already knows how to read
  the on-disk candle store (feather/json, whichever format is present)
  that ``freqtrade download-data`` and the live bot both write to under
  ``user_data/data/<exchange>``. ``freqtrade.data.converter`` already
  implements de-duplication and missing-candle interpolation
  (``clean_ohlcv_dataframe`` / ``ohlcv_fill_up_missing_data``) that has
  been exercised in production by a large user base. Reimplementing
  either would duplicate mature, well-tested code for no benefit — this
  module is a thin, typed, validated orchestration layer on top of them.
* **Loading and processing are separate, independently testable
  functions.** ``MarketDataLoader`` is the only piece that touches disk;
  ``clean_and_fill``, ``validate_ohlcv``, and ``align_pairs`` are pure
  functions over dataframes already in memory, so unit tests can exercise
  gap-filling, validation, and alignment logic directly with synthetic
  data and no filesystem or network dependency.
* **Validation raises rather than warns.** A market data layer that
  silently accepts a corrupt dataframe (duplicate timestamps, an
  ``OHLC`` violation, a gap that wasn't actually filled) would let bad
  data flow straight into statistical models downstream, where it is
  much harder to trace back to its source. ``MarketDataError`` is raised
  eagerly, with every violation found rather than just the first, so
  problems are diagnosed in one pass.
* **Alignment uses an inner join on timestamps.** Statistical arbitrage
  (cointegration, spread z-scores, etc.) requires every pair's series to
  share the exact same timestamp index — a single missing bar in one leg
  silently misaligns the pair. Taking the intersection of available
  timestamps across all requested pairs, rather than forward-filling
  across pair boundaries, guarantees that guarantee without fabricating
  data for one pair that doesn't exist for another.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from freqtrade.data.converter import clean_ohlcv_dataframe
from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType

if TYPE_CHECKING:
    from collections.abc import Sequence

    from freqtrade.configuration import TimeRange


logger = logging.getLogger(__name__)

#: Columns every OHLCV dataframe handled by this module must contain.
REQUIRED_COLUMNS: tuple[str, ...] = ("date", "open", "high", "low", "close", "volume")


class MarketDataError(Exception):
    """Raised when candle data cannot be loaded or fails integrity validation."""


class MarketDataLoader:
    """Loads OHLCV candle data from Freqtrade's on-disk candle store.

    This class is a thin wrapper around
    :func:`freqtrade.data.history.load_pair_history`. It adds nothing to
    the loading mechanics themselves — it only fixes the datadir/timeframe/
    candle-type for a given project and fails loudly when no data is
    available, instead of silently returning an empty dataframe.
    """

    def __init__(
        self,
        datadir: Path | str,
        timeframe: str,
        candle_type: CandleType = CandleType.FUTURES,
    ) -> None:
        """Initialize the loader.

        Parameters
        ----------
        datadir:
            Directory Freqtrade stores/reads candle data in, e.g.
            ``user_data/data/hyperliquid``.
        timeframe:
            Candle timeframe to load, e.g. ``"5m"``.
        candle_type:
            Candle type to load. Defaults to ``CandleType.FUTURES`` since
            this project trades USDC-margined perpetual futures.
        """
        self.datadir = Path(datadir)
        self.timeframe = timeframe
        self.candle_type = candle_type

    def load_pair(self, pair: str, timerange: TimeRange | None = None) -> pd.DataFrame:
        """Load raw candle data for a single pair.

        Parameters
        ----------
        pair:
            Pair to load, e.g. ``"BTC/USDC:USDC"``.
        timerange:
            Optional time range to restrict loading to. ``None`` loads
            all locally available history.

        Returns
        -------
        DataFrame
            Raw candle data as stored on disk (not yet cleaned, gap-filled,
            or validated).

        Raises
        ------
        MarketDataError
            If no local candle data exists for ``pair``.
        """
        logger.info(
            "Loading %s candles for %s from %s", self.timeframe, pair, self.datadir
        )
        dataframe = load_pair_history(
            pair=pair,
            timeframe=self.timeframe,
            datadir=self.datadir,
            timerange=timerange,
            fill_up_missing=False,
            drop_incomplete=False,
            candle_type=self.candle_type,
        )
        if dataframe.empty:
            raise MarketDataError(
                f"No local candle data found for {pair} ({self.timeframe}) in "
                f"{self.datadir}. Download it first, e.g.: "
                f"freqtrade download-data -p {pair} -t {self.timeframe} "
                f"--datadir {self.datadir}"
            )
        logger.debug("Loaded %d raw candles for %s", len(dataframe), pair)
        return dataframe

    def load_pairs(
        self, pairs: Sequence[str], timerange: TimeRange | None = None
    ) -> dict[str, pd.DataFrame]:
        """Load raw candle data for multiple pairs.

        Parameters
        ----------
        pairs:
            Pairs to load, e.g. ``("BTC/USDC:USDC", "ETH/USDC:USDC")``.
        timerange:
            Optional time range to restrict loading to.

        Returns
        -------
        dict[str, DataFrame]
            Mapping of pair -> raw candle dataframe.
        """
        return {pair: self.load_pair(pair, timerange) for pair in pairs}


def clean_and_fill(
    dataframe: pd.DataFrame,
    timeframe: str,
    pair: str,
    *,
    drop_incomplete: bool = False,
) -> pd.DataFrame:
    """De-duplicate, sort, and fill gaps in a single pair's OHLCV data.

    Delegates entirely to Freqtrade's own
    :func:`freqtrade.data.converter.clean_ohlcv_dataframe`, which groups
    duplicate timestamps, optionally drops a trailing incomplete candle,
    and fills missing candles (``ohlcv_fill_up_missing_data``): filled
    candles get an OHLC equal to the previous close and zero volume,
    which is the standard Freqtrade convention for "no trades happened
    in this interval."

    Parameters
    ----------
    dataframe:
        Raw candle data, as returned by :meth:`MarketDataLoader.load_pair`.
    timeframe:
        Candle timeframe, used to compute the expected candle spacing.
    pair:
        Pair name, used only for log messages.
    drop_incomplete:
        Whether to drop the final candle, in case it represents an
        in-progress (not-yet-closed) period. Defaults to ``False`` since
        historical data loaded from disk is normally already closed.

    Returns
    -------
    DataFrame
        Cleaned, gap-filled candle data.
    """
    before = len(dataframe)
    cleaned = clean_ohlcv_dataframe(
        dataframe,
        timeframe,
        pair,
        fill_missing=True,
        drop_incomplete=drop_incomplete,
    )
    filled = len(cleaned) - before
    if filled:
        logger.info("Filled %d missing candle(s) for %s", filled, pair)
    return cleaned


def validate_ohlcv(dataframe: pd.DataFrame, pair: str) -> None:
    """Validate the integrity of a single pair's OHLCV dataframe.

    Checks performed:

    * All of :data:`REQUIRED_COLUMNS` are present and the dataframe is
      non-empty.
    * The ``date`` column has no nulls, no duplicates, and is sorted in
      strictly increasing order.
    * ``open``/``high``/``low``/``close``/``volume`` have no null values.
    * Prices are strictly positive and volume is non-negative.
    * The OHLC relationship holds for every row: ``low <= open, close <=
      high`` and ``low <= high``.

    All violations are collected before raising, so a single failure
    surfaces the full picture rather than just the first problem found.

    Parameters
    ----------
    dataframe:
        Candle data to validate.
    pair:
        Pair name, used in error messages.

    Raises
    ------
    MarketDataError
        If any integrity check fails.
    """
    issues: list[str] = []

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in dataframe.columns]
    if missing_columns:
        raise MarketDataError(
            f"{pair}: dataframe is missing required column(s): {missing_columns}"
        )

    if dataframe.empty:
        raise MarketDataError(f"{pair}: dataframe is empty")

    dates = dataframe["date"]
    if dates.isna().any():
        issues.append("'date' column contains null values")
    if dates.duplicated().any():
        issues.append(f"'date' column has {dates.duplicated().sum()} duplicate timestamp(s)")
    if not dates.is_monotonic_increasing:
        issues.append("'date' column is not sorted in increasing order")

    numeric_cols = ["open", "high", "low", "close", "volume"]
    nulls = dataframe[numeric_cols].isna().any()
    for col in nulls[nulls].index:
        issues.append(f"'{col}' column contains null values")

    price_cols = ["open", "high", "low", "close"]
    non_positive = (dataframe[price_cols] <= 0).any()
    for col in non_positive[non_positive].index:
        issues.append(f"'{col}' column contains non-positive price(s)")

    if (dataframe["volume"] < 0).any():
        issues.append("'volume' column contains negative value(s)")

    high_low_violation = dataframe["high"] < dataframe["low"]
    if high_low_violation.any():
        issues.append(f"{high_low_violation.sum()} row(s) where high < low")

    high_bound_violation = (dataframe["high"] < dataframe["open"]) | (
        dataframe["high"] < dataframe["close"]
    )
    if high_bound_violation.any():
        issues.append(f"{high_bound_violation.sum()} row(s) where high < open or high < close")

    low_bound_violation = (dataframe["low"] > dataframe["open"]) | (
        dataframe["low"] > dataframe["close"]
    )
    if low_bound_violation.any():
        issues.append(f"{low_bound_violation.sum()} row(s) where low > open or low > close")

    if issues:
        formatted = "; ".join(issues)
        raise MarketDataError(f"{pair}: dataframe failed integrity validation: {formatted}")


def align_pairs(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Align multiple pairs' OHLCV data onto a shared timestamp index.

    Takes the intersection of timestamps present in every input
    dataframe and trims each dataframe to exactly that set, in sorted
    order. This is required before any cross-pair statistics (spread,
    correlation, cointegration) can be computed: those calculations
    assume row ``i`` of one pair's series and row ``i`` of another's
    represent the same point in time.

    Parameters
    ----------
    data:
        Mapping of pair -> cleaned candle dataframe (see
        :func:`clean_and_fill`).

    Returns
    -------
    dict[str, DataFrame]
        Mapping of pair -> dataframe trimmed to the shared timestamp
        index, each re-indexed with a fresh ``RangeIndex``.

    Raises
    ------
    MarketDataError
        If ``data`` is empty, or if the pairs share no common timestamps.
    """
    if not data:
        raise MarketDataError("align_pairs() called with no pairs to align")

    if len(data) == 1:
        (pair, dataframe), = data.items()
        return {pair: dataframe.sort_values("date").reset_index(drop=True)}

    timestamp_sets = [set(df["date"]) for df in data.values()]
    common_timestamps = set.intersection(*timestamp_sets)

    if not common_timestamps:
        raise MarketDataError(
            "No overlapping timestamps found across pairs: "
            f"{list(data.keys())}. Cannot align market data."
        )

    aligned: dict[str, pd.DataFrame] = {}
    for pair, dataframe in data.items():
        trimmed = (
            dataframe[dataframe["date"].isin(common_timestamps)]
            .sort_values("date")
            .reset_index(drop=True)
        )
        dropped = len(dataframe) - len(trimmed)
        if dropped:
            logger.info(
                "Dropped %d candle(s) from %s while aligning to shared timestamps",
                dropped,
                pair,
            )
        aligned[pair] = trimmed

    lengths = {pair: len(df) for pair, df in aligned.items()}
    if len(set(lengths.values())) != 1:
        raise MarketDataError(f"Alignment produced mismatched lengths across pairs: {lengths}")

    return aligned


class MarketDataService:
    """High-level orchestrator: load, clean, validate, and align.

    This is the entry point most downstream modules should use — it
    composes :class:`MarketDataLoader`, :func:`clean_and_fill`,
    :func:`validate_ohlcv`, and :func:`align_pairs` into the standard
    pipeline (validate raw candles are usable, fill their internal gaps,
    validate again, then align across pairs and validate once more).
    """

    def __init__(self, loader: MarketDataLoader) -> None:
        """Initialize the service with a configured loader.

        Parameters
        ----------
        loader:
            Loader to use for fetching each pair's raw candle data.
        """
        self.loader = loader

    def get_aligned_market_data(
        self,
        pairs: Sequence[str],
        timerange: TimeRange | None = None,
        *,
        drop_incomplete: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Load, clean, validate, and align candle data for multiple pairs.

        Parameters
        ----------
        pairs:
            Pairs to load, e.g. ``("BTC/USDC:USDC", "ETH/USDC:USDC")``.
        timerange:
            Optional time range to restrict loading to.
        drop_incomplete:
            Forwarded to :func:`clean_and_fill`.

        Returns
        -------
        dict[str, DataFrame]
            Mapping of pair -> cleaned, gap-filled, validated dataframe,
            all sharing an identical ``date`` index.

        Raises
        ------
        MarketDataError
            If any pair fails to load, fails validation, or the pairs
            share no common timestamps.
        """
        raw = self.loader.load_pairs(pairs, timerange)

        cleaned: dict[str, pd.DataFrame] = {}
        for pair, dataframe in raw.items():
            filled = clean_and_fill(
                dataframe, self.loader.timeframe, pair, drop_incomplete=drop_incomplete
            )
            validate_ohlcv(filled, pair)
            cleaned[pair] = filled

        aligned = align_pairs(cleaned)
        for pair, dataframe in aligned.items():
            validate_ohlcv(dataframe, pair)

        candle_count = len(next(iter(aligned.values()))) if aligned else 0
        logger.info(
            "Market data ready: %d pair(s), %d aligned candle(s) each",
            len(aligned),
            candle_count,
        )
        return aligned
