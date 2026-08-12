"""Research-only volatility measurement over already-downloaded OHLCV data.

Companion to `hermes.signal_forensics`: that module reconstructed
EMA200/ADX14/Donchian context at each of the 39 frozen baseline trades'
entry candles. This module answers a distinct, narrower question raised
by that diagnosis -- H4, "is SOL's underperformance a volatility effect?"
-- by computing ATR% and rolling realized volatility over the same OHLCV
data, joining those values to the same 39 trade entries by the same
exact-timestamp matching, and producing purely descriptive comparisons
(pair-level, entry-context, stop-loss-vs-exit-signal, quartile buckets,
correlations). Nothing here computes a verdict; `hermes.cli` or a caller
script decides what SUPPORTED/NOT SUPPORTED/etc. means from these numbers.

Nothing here launches a subprocess, a backtest, or Freqtrade at all, and
nothing here touches `TrendFollowCore.py`'s indicators or entry/exit
logic -- ATR and realized volatility are not part of the strategy and
this module does not make them part of it. This is pure post-hoc
arithmetic over data that already exists.

Design decisions
-----------------
* **Entry-candle matching reuses `hermes.signal_forensics.find_entry_candle`
  rather than re-implementing timestamp lookup.** Both modules need
  exactly the same operation -- "the row in an indicator dataframe whose
  `date` exactly matches a trade's `entry_time`" -- so duplicating it here
  would risk the two implementations silently disagreeing on edge cases
  (timezone handling, no-match behavior). The already-tested
  implementation is reused as-is.
* **ATR and realized volatility are computed with the same no-lookahead
  discipline as `TrendFollowCore`'s own indicators**: `compute_atr` is a
  rolling mean over the trailing `period` candles (pandas
  `rolling(...).mean()`, backward-looking by construction, `min_periods`
  equal to the full window so a partially-filled window produces `NaN`
  rather than a value computed from fewer candles than intended);
  `compute_realized_volatility` is likewise a trailing rolling standard
  deviation. `audit_no_lookahead_volatility` verifies this the same way
  `hermes.signal_forensics.audit_no_lookahead` does -- comparing a value
  computed from the full series against the same value computed from a
  series truncated to end at that row -- rather than merely asserting it.
* **Spearman correlation is computed via rank-transform + Pearson, not a
  `scipy` dependency.** `pearson_correlation` and `spearman_correlation`
  both operate on plain lists of floats with no `scipy` import, keeping
  this module's dependency footprint identical to the rest of `hermes`
  (`pandas`/`numpy`, already required). Degenerate inputs (fewer than 2
  points, or zero variance in either series) return `None` rather than a
  fabricated coefficient.
* **Every aggregate function is presentation-neutral: it returns numbers,
  not a verdict.** Nothing in this module decides whether H4 is
  "supported" -- that judgment call belongs to whoever reads the output
  numbers, consistent with the same "observation vs. interpretation"
  separation `hermes.signal_forensics` and this session's prior forensic
  blocks have maintained throughout.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hermes.signal_forensics import find_entry_candle

DEFAULT_ATR_PERIOD = 14
DEFAULT_REALIZED_VOL_WINDOW = 42  # 42 * 4h = 168h ~= 7 days
DEFAULT_STOP_LOSS_PCT = 5.0  # TrendFollowCore's frozen -5% stop, as a positive percentage


class VolatilityForensicsError(Exception):
    """Raised for unrecoverable input problems (never for "value is missing")."""


# ---------------------------------------------------------------------------
# Core, no-lookahead volatility calculations
# ---------------------------------------------------------------------------


def compute_true_range(df: pd.DataFrame) -> pd.Series:
    """True Range per candle: max(high-low, |high-prev_close|, |low-prev_close|).

    The first row is always `NaN` (no previous close exists) -- warmup,
    not a bug, mirroring how `TrendFollowCore.compute_indicators` treats
    its own first candles.
    """
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def compute_atr(df: pd.DataFrame, period: int = DEFAULT_ATR_PERIOD) -> pd.Series:
    """ATR(period): trailing rolling mean of True Range. Backward-looking only."""
    tr = compute_true_range(df)
    return tr.rolling(window=period, min_periods=period).mean()


def compute_atr_pct(df: pd.DataFrame, period: int = DEFAULT_ATR_PERIOD) -> pd.Series:
    """ATR(period) as a percentage of each candle's own close."""
    return compute_atr(df, period) / df["close"] * 100.0


def compute_log_returns(df: pd.DataFrame) -> pd.Series:
    """4h log returns: ln(close_t / close_{t-1}). First row is `NaN`."""
    return np.log(df["close"] / df["close"].shift(1))


def compute_realized_volatility(
    df: pd.DataFrame, window: int = DEFAULT_REALIZED_VOL_WINDOW
) -> pd.Series:
    """Rolling standard deviation of log returns over `window` candles.

    Not annualized -- the raw rolling value is what every comparison in
    this module uses. `annualized_realized_volatility` exists separately
    and is never substituted for this one.
    """
    return compute_log_returns(df).rolling(window=window, min_periods=window).std()


def annualized_realized_volatility(
    realized_vol: pd.Series, *, candles_per_year: float
) -> pd.Series:
    """Optional, clearly-separate annualized view of `realized_vol`.

    Never used in place of the raw rolling value anywhere else in this
    module -- provided only because callers sometimes want it, per this
    module's own "don't let annualization replace the raw measure" rule.
    """
    return realized_vol * math.sqrt(candles_per_year)


def build_volatility_dataframe(
    ohlcv: pd.DataFrame,
    *,
    atr_period: int = DEFAULT_ATR_PERIOD,
    realized_vol_window: int = DEFAULT_REALIZED_VOL_WINDOW,
) -> pd.DataFrame:
    """`ohlcv` (must have `date`/`high`/`low`/`close`) plus `atr_pct` and
    `realized_vol` columns, ready for `find_entry_candle` lookup."""
    df = ohlcv.copy()
    df["atr_pct"] = compute_atr_pct(df, atr_period)
    df["realized_vol"] = compute_realized_volatility(df, realized_vol_window)
    return df


def audit_no_lookahead_volatility(
    df: pd.DataFrame,
    at_index: int,
    *,
    atr_period: int = DEFAULT_ATR_PERIOD,
    realized_vol_window: int = DEFAULT_REALIZED_VOL_WINDOW,
    tolerance: float = 1e-9,
) -> bool:
    """`True` iff `atr_pct`/`realized_vol` at `at_index` are identical whether
    computed from the full `df` or from `df` truncated to end at `at_index`.

    Same proof-not-assumption logic as
    `hermes.signal_forensics.audit_no_lookahead`.
    """
    full = build_volatility_dataframe(
        df, atr_period=atr_period, realized_vol_window=realized_vol_window
    )
    truncated = build_volatility_dataframe(
        df.iloc[: at_index + 1], atr_period=atr_period, realized_vol_window=realized_vol_window
    )
    for col in ("atr_pct", "realized_vol"):
        a, b = full.iloc[at_index][col], truncated.iloc[-1][col]
        a_nan, b_nan = pd.isna(a), pd.isna(b)
        if a_nan and b_nan:
            continue
        if a_nan or b_nan:
            return False
        if abs(a - b) > tolerance:
            return False
    return True


# ---------------------------------------------------------------------------
# Pair-level (whole-window) summary
# ---------------------------------------------------------------------------


def _describe(values: pd.Series) -> dict[str, float | None]:
    clean = values.dropna()
    if clean.empty:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std()) if len(clean) > 1 else 0.0,
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def summarize_pair_volatility(volatility_df: pd.DataFrame) -> dict[str, Any]:
    """Whole-window ATR%/realized-vol descriptive stats for one pair's data."""
    return {
        "candles": int(len(volatility_df)),
        "atr_pct": _describe(volatility_df["atr_pct"]),
        "realized_vol": _describe(volatility_df["realized_vol"]),
    }


# ---------------------------------------------------------------------------
# Per-trade entry-context volatility
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VolatilityEntryContext:
    """One trade's entry-candle volatility, joined to its already-known
    signal context (from `hermes.signal_forensics`) and outcome (from
    `hermes.trade_report`)."""

    trade_number: int
    pair: str | None
    direction: str | None
    entry_time: str | None
    entry_price: float | None
    adx14: float | None
    ema_distance_pct: float | None
    atr_pct: float | None
    realized_vol: float | None
    exit_reason: str | None
    profit_pct: float | None
    duration_minutes: float | None
    is_winner: bool | None
    candle_matched: bool

    @property
    def stop_distance_in_atr(self) -> float | None:
        """`5% / ATR% at entry` -- see module docstring; descriptive only."""
        if not self.atr_pct:
            return None
        return DEFAULT_STOP_LOSS_PCT / self.atr_pct


def _none_if_nan(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return float(value)


def reconstruct_volatility_context(
    signal_trade: dict[str, Any], volatility_df: pd.DataFrame | None
) -> VolatilityEntryContext:
    """Build one trade's `VolatilityEntryContext` from a signal-forensics
    trade record (a dict shaped like `hermes.signal_forensics.EntryContext`,
    e.g. read back from the persisted forensic JSON) plus a pair's
    `atr_pct`/`realized_vol`-augmented OHLCV dataframe.
    """
    candle = (
        None
        if volatility_df is None
        else find_entry_candle(volatility_df, signal_trade.get("entry_time"))
    )
    atr_pct = _none_if_nan(candle.get("atr_pct")) if candle is not None else None
    realized_vol = _none_if_nan(candle.get("realized_vol")) if candle is not None else None

    return VolatilityEntryContext(
        trade_number=signal_trade["trade_number"],
        pair=signal_trade.get("pair"),
        direction=signal_trade.get("direction"),
        entry_time=signal_trade.get("entry_time"),
        entry_price=signal_trade.get("entry_price"),
        adx14=signal_trade.get("adx14"),
        ema_distance_pct=signal_trade.get("ema_distance_pct"),
        atr_pct=atr_pct,
        realized_vol=realized_vol,
        exit_reason=signal_trade.get("exit_reason"),
        profit_pct=signal_trade.get("profit_pct"),
        duration_minutes=signal_trade.get("duration_minutes"),
        is_winner=signal_trade.get("is_winner"),
        candle_matched=candle is not None,
    )


def reconstruct_all_volatility_contexts(
    signal_trades: list[dict[str, Any]], volatility_by_pair: dict[str, pd.DataFrame]
) -> list[VolatilityEntryContext]:
    return [
        reconstruct_volatility_context(t, volatility_by_pair.get(t.get("pair")))
        for t in signal_trades
    ]


# ---------------------------------------------------------------------------
# Entry-volatility groupings (descriptive only -- no verdicts)
# ---------------------------------------------------------------------------


def _group_stats(group: list[VolatilityEntryContext]) -> dict[str, Any]:
    atr_values = [c.atr_pct for c in group if c.atr_pct is not None]
    rv_values = [c.realized_vol for c in group if c.realized_vol is not None]
    known_outcome = [c for c in group if c.is_winner is not None]
    winners = [c for c in known_outcome if c.is_winner]
    ema_values = [c.ema_distance_pct for c in group if c.ema_distance_pct is not None]
    duration_values = [c.duration_minutes for c in group if c.duration_minutes is not None]
    profit_pct_values = [c.profit_pct for c in group if c.profit_pct is not None]

    def _mean(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    def _median(values: list[float]) -> float | None:
        return float(np.median(values)) if values else None

    return {
        "trade_count": len(group),
        "mean_atr_pct": _mean(atr_values),
        "median_atr_pct": _median(atr_values),
        "mean_realized_vol": _mean(rv_values),
        "median_realized_vol": _median(rv_values),
        "mean_ema_distance_pct": _mean(ema_values),
        "mean_duration_minutes": _mean(duration_values),
        "win_rate_pct": (100.0 * len(winners) / len(known_outcome)) if known_outcome else None,
        "mean_profit_pct": _mean(profit_pct_values),
    }


def summarize_entry_volatility_by_pair(
    contexts: list[VolatilityEntryContext],
) -> dict[str, dict[str, Any]]:
    pairs = sorted({c.pair for c in contexts if c.pair is not None})
    return {pair: _group_stats([c for c in contexts if c.pair == pair]) for pair in pairs}


def summarize_stop_loss_vs_exit_signal(
    contexts: list[VolatilityEntryContext],
) -> dict[str, dict[str, Any]]:
    """`stop_loss` vs `exit_signal`, with `force_exit` reported separately
    (per the build block: excluded from the primary comparison)."""
    return {
        "stop_loss": _group_stats([c for c in contexts if c.exit_reason == "stop_loss"]),
        "exit_signal": _group_stats([c for c in contexts if c.exit_reason == "exit_signal"]),
        "force_exit": _group_stats([c for c in contexts if c.exit_reason == "force_exit"]),
    }


def summarize_winner_vs_loser(
    contexts: list[VolatilityEntryContext],
) -> dict[str, dict[str, Any]]:
    return {
        "winners": _group_stats([c for c in contexts if c.is_winner is True]),
        "losers": _group_stats([c for c in contexts if c.is_winner is False]),
        "stop_loss": _group_stats([c for c in contexts if c.exit_reason == "stop_loss"]),
        "negative_exit_signal": _group_stats(
            [
                c
                for c in contexts
                if c.exit_reason == "exit_signal" and (c.profit_pct or 0) < 0
            ]
        ),
        "positive_exit_signal": _group_stats(
            [
                c
                for c in contexts
                if c.exit_reason == "exit_signal" and (c.profit_pct or 0) >= 0
            ]
        ),
    }


# ---------------------------------------------------------------------------
# Correlations (Pearson + Spearman, no scipy dependency)
# ---------------------------------------------------------------------------


def pearson_correlation(x: list[float], y: list[float]) -> float | None:
    """`None` for fewer than 2 paired points or zero variance in either series."""
    if len(x) != len(y) or len(x) < 2:
        return None
    x_arr, y_arr = np.array(x, dtype=float), np.array(y, dtype=float)
    if np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def spearman_correlation(x: list[float], y: list[float]) -> float | None:
    """Pearson correlation of rank-transformed `x`/`y` -- no `scipy` dependency."""
    if len(x) != len(y) or len(x) < 2:
        return None
    x_ranks = pd.Series(x).rank().to_numpy()
    y_ranks = pd.Series(y).rank().to_numpy()
    return pearson_correlation(list(x_ranks), list(y_ranks))


def compute_volatility_ema_correlations(
    contexts: list[VolatilityEntryContext],
) -> dict[str, dict[str, Any]]:
    paired_atr = [
        (c.atr_pct, c.ema_distance_pct)
        for c in contexts
        if c.atr_pct is not None and c.ema_distance_pct is not None
    ]
    paired_rv = [
        (c.realized_vol, c.ema_distance_pct)
        for c in contexts
        if c.realized_vol is not None and c.ema_distance_pct is not None
    ]

    def _corr_entry(pairs: list[tuple[float, float]]) -> dict[str, Any]:
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        return {
            "pearson_r": pearson_correlation(xs, ys),
            "spearman_rho": spearman_correlation(xs, ys),
            "n": len(pairs),
        }

    return {
        "atr_pct_vs_ema_distance": _corr_entry(paired_atr),
        "realized_vol_vs_ema_distance": _corr_entry(paired_rv),
    }


# ---------------------------------------------------------------------------
# Quartile buckets (distribution-based, never chosen post-hoc for effect)
# ---------------------------------------------------------------------------


def quartile_buckets(
    contexts: list[VolatilityEntryContext], *, metric: str = "atr_pct"
) -> dict[str, dict[str, Any]]:
    """Split `contexts` into quartiles of `metric` (`atr_pct` or
    `realized_vol`), using `pandas.qcut` (distribution-derived boundaries,
    not selected after seeing which bucket looks best)."""
    valid = [c for c in contexts if getattr(c, metric) is not None]
    if len(valid) < 4:
        return {}

    values = pd.Series([getattr(c, metric) for c in valid])
    try:
        labels = pd.qcut(values, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    except ValueError:
        return {}

    buckets: dict[str, list[VolatilityEntryContext]] = {}
    for ctx, label in zip(valid, labels):
        buckets.setdefault(str(label), []).append(ctx)

    result = {}
    for label, group in buckets.items():
        stats = _group_stats(group)
        stop_loss_count = sum(1 for c in group if c.exit_reason == "stop_loss")
        profit_values = [c.profit_pct for c in group if c.profit_pct is not None]
        result[label] = {
            **stats,
            "stop_loss_count": stop_loss_count,
            "stop_loss_frequency_pct": (100.0 * stop_loss_count / len(group)) if group else None,
            "total_profit_pct": float(sum(profit_values)) if profit_values else None,
        }
    return result


# ---------------------------------------------------------------------------
# Stop-distance-in-ATR diagnostic
# ---------------------------------------------------------------------------


def summarize_stop_distance_in_atr_by_pair(
    contexts: list[VolatilityEntryContext], *, stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT
) -> dict[str, dict[str, Any]]:
    """Mean ATR% and mean `stop_loss_pct / ATR%` per pair -- a descriptive
    "how many ATR units does the fixed stop represent" diagnostic, not a
    risk model. See module/build-block docstring for the caveat."""
    pairs = sorted({c.pair for c in contexts if c.pair is not None})
    result: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        group = [c for c in contexts if c.pair == pair]
        atr_values = [c.atr_pct for c in group if c.atr_pct is not None]
        stop_distances = [c.stop_distance_in_atr for c in group if c.stop_distance_in_atr is not None]
        result[pair] = {
            "trade_count": len(group),
            "mean_atr_pct": float(np.mean(atr_values)) if atr_values else None,
            "mean_stop_distance_in_atr": float(np.mean(stop_distances)) if stop_distances else None,
            "stop_loss_pct_used": stop_loss_pct,
        }
    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def build_volatility_dataset(
    *,
    pair_volatility_summary: dict[str, Any],
    contexts: list[VolatilityEntryContext],
    window: str,
    timeframe: str,
    pairs: list[str],
    atr_period: int,
    realized_vol_window: int,
    data_quality: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the full JSON-ready payload described in the build block."""
    return {
        "metadata": {
            "strategy": "TrendFollowCore",
            "purpose": "H4 volatility forensics -- descriptive measurement only, no verdict computed here",
        },
        "window": window,
        "timeframe": timeframe,
        "pairs": pairs,
        "atr_methodology": {
            "period": atr_period,
            "formula": "TR = max(high-low, |high-prev_close|, |low-prev_close|); ATR = rolling_mean(TR, period); ATR% = ATR / close * 100",
        },
        "realized_volatility_methodology": {
            "window_candles": realized_vol_window,
            "approx_days": realized_vol_window * 4 / 24,
            "formula": "log_return = ln(close_t / close_t-1); realized_vol = rolling_std(log_return, window), not annualized",
        },
        "pair_summary": pair_volatility_summary,
        "trade_entry_context": [asdict(c) for c in contexts],
        "stop_loss_vs_exit_signal": summarize_stop_loss_vs_exit_signal(contexts),
        "winner_vs_loser": summarize_winner_vs_loser(contexts),
        "volatility_buckets": {
            "by_atr_pct": quartile_buckets(contexts, metric="atr_pct"),
            "by_realized_vol": quartile_buckets(contexts, metric="realized_vol"),
        },
        "correlations": compute_volatility_ema_correlations(contexts),
        "stop_distance_in_ATR": summarize_stop_distance_in_atr_by_pair(contexts),
        "data_quality": data_quality,
        "conclusion": "NOT COMPUTED -- descriptive measurement only, see accompanying report for interpretation",
    }


def save_volatility_dataset(dataset: dict[str, Any], output_path: Path) -> None:
    """Write `dataset` as JSON. Caller validates `output_path` is inside
    persistent storage (see `hermes.export_paths`) -- this only writes."""
    Path(output_path).write_text(json.dumps(dataset, indent=2, default=str))
