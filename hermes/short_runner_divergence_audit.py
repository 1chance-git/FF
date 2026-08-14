"""SHORT-winner PERSISTENT vs ORDINARY structural-divergence audit
(research/analysis only).

Answers one question only: at which checkpoint in a SHORT winner's
lifecycle -- measured from already-existing OHLCV data at a fixed
checkpoint ladder (4h/12h/24h/48h/3d/7d/10d/14d/17d/21d/24d/30d) -- do the
3 previously identified persistent SHORT winners first become
descriptively distinguishable from the 5 ordinary SHORT winners, and on
which variable? This module never runs a backtest, never touches
`TrendFollowCore.py`, config, or the pair whitelist, never redefines the
persistent/ordinary grouping, and never searches for or proposes a new
threshold or trading rule. It reports descriptive separation only --
never a statistical-significance or causal claim.

Design decisions
-----------------
* **EMA200/ADX14/Donchian20/MFE-MAE/trade reconstruction reuse the exact
  independent pandas reimplementation already established in
  `hermes.short_trend_persistence_audit`** (and, before it,
  `hermes.short_persistence_audit`, `hermes.short_runner_lifecycle_audit`,
  `hermes.short_ema_exit_attribution_audit`) -- same periods, same
  conventions, not a fifth separately-written version.
* **The checkpoint ladder is fixed and literal** (`CHECKPOINTS`), taken
  directly from this block's own required list -- never tuned, added to,
  or removed from based on results. It intentionally differs from the
  prior block's ladder (this one adds 10d/17d/24d and drops 45d) because
  the block asked for a different, finer-grained ladder aimed at locating
  the *earliest* divergence point rather than describing the full
  lifecycle.
* **Persistent/ordinary/loser classification reuses the exact
  `(pair, entry_time)` matching already established in
  `hermes.short_ema_exit_attribution_audit`** -- never re-ranked or
  re-derived here. `ORDINARY_KEYS` is listed explicitly (rather than
  derived by elimination) only because this block's prompt names the 5
  ordinary identities explicitly; membership is still exact-tuple based,
  never P/L-ranked at runtime.
* **Checkpoint-to-checkpoint deltas** (`delta_ema_distance_pct`,
  `delta_adx`) are computed only between *consecutive entries in the
  fixed ladder*, using each checkpoint's own mean-of-subset value -- never
  interpolated, never computed from a checkpoint that was skipped because
  the trade had already closed.
* **Divergence detection is a descriptive, non-statistical heuristic**:
  for a given variable, the earliest checkpoint from which the sign of
  `(persistent_mean - ordinary_mean)` stays constant through every
  remaining checkpoint in the ladder. This is a *consistency-of-direction*
  check, not a magnitude threshold and not a significance test -- no gap
  size, p-value, or z-score is ever computed or compared against an
  invented cutoff. A variable with fewer than 2 remaining checkpoints, or
  whose gap is `None` at any point in the tail, has no detectable
  divergence point under this definition.
* **Thin-sample discipline**: PERSISTENT (n=3) is always labeled
  thin-sample (`THIN_SAMPLE_THRESHOLD = 5`); every aggregate is still
  computed and reported, never suppressed.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Sequence

import pandas as pd

from hermes.trade_report import Trade

LONG = "LONG"
SHORT = "SHORT"

EMA_PERIOD = 200
ADX_PERIOD = 14
DONCHIAN_PERIOD = 20
ADX_ENTRY_THRESHOLD = 25.0  # TrendFollowCore.ADX_THRESHOLD, reused verbatim

THIN_SAMPLE_THRESHOLD = 5

PERSISTENT_KEYS: frozenset[tuple[str, str]] = frozenset({
    ("ETH/USDC:USDC", "2026-01-20 16:00:00+00:00"),
    ("ETH/USDC:USDC", "2026-05-15 16:00:00+00:00"),
    ("BTC/USDC:USDC", "2026-01-20 08:00:00+00:00"),
})

ORDINARY_KEYS: frozenset[tuple[str, str]] = frozenset({
    ("BTC/USDC:USDC", "2025-10-29 20:00:00+00:00"),
    ("BTC/USDC:USDC", "2026-06-01 12:00:00+00:00"),
    ("ETH/USDC:USDC", "2025-11-03 16:00:00+00:00"),
    ("ETH/USDC:USDC", "2025-10-09 12:00:00+00:00"),
    ("BTC/USDC:USDC", "2026-08-11 16:00:00+00:00"),
})

# The specific ordinary-winner identity called out by the block's
# secondary question 7 -- flagged, never re-classified.
LONG_DURATION_ORDINARY_KEY: tuple[str, str] = ("BTC/USDC:USDC", "2025-10-29 20:00:00+00:00")

# Fixed, literal checkpoint ladder (minutes from entry) -- from the
# block's own required list. Never searched or extended based on results.
CHECKPOINTS: dict[str, float] = {
    "4h": 240.0,
    "12h": 720.0,
    "24h": 1440.0,
    "48h": 2880.0,
    "3d": 4320.0,
    "7d": 10080.0,
    "10d": 14400.0,
    "14d": 20160.0,
    "17d": 24480.0,
    "21d": 30240.0,
    "24d": 34560.0,
    "30d": 43200.0,
}
CHECKPOINT_ORDER: tuple[str, ...] = tuple(CHECKPOINTS.keys())


def classify_group(trade: Trade) -> str:
    """`"PERSISTENT"` / `"ORDINARY"` / `"LOSER"` for a SHORT trade, by
    exact `(pair, entry_time)` match against `PERSISTENT_KEYS` for
    winners, and `is_winner` otherwise."""
    if trade.is_winner is False:
        return "LOSER"
    key = (trade.pair, str(trade.entry_time))
    return "PERSISTENT" if key in PERSISTENT_KEYS else "ORDINARY"


def is_long_duration_ordinary(trade: Trade) -> bool:
    """True only for the exact 60.3-day BTC 2025-10-29 20:00 ordinary
    winner named in the block's secondary question 7."""
    return (trade.pair, str(trade.entry_time)) == LONG_DURATION_ORDINARY_KEY


def ema(series: pd.Series, period: int = EMA_PERIOD) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move, down_move = high.diff(), -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def donchian_prev_bounds(df: pd.DataFrame, period: int = DONCHIAN_PERIOD) -> tuple[pd.Series, pd.Series]:
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    return prev_high.rolling(window=period).max(), prev_low.rolling(window=period).min()


# ---------------------------------------------------------------------------
# Full trade reconstruction (entry through exit, one row per candle)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradeCandle:
    date: pd.Timestamp
    close: float
    ema200: float | None
    ema_distance_pct: float | None
    adx: float | None
    donchian_breakout_pct: float | None
    realized_vol_pct: float | None
    price_change_pct: float
    cumulative_mfe_pct: float
    cumulative_mae_pct: float
    structurally_aligned: bool | None
    adx_above_threshold: bool | None
    days_since_entry: float


def reconstruct_full_trade(
    ohlcv: pd.DataFrame | None, entry_time, exit_time, entry_price: float | None, direction: str | None,
) -> list[TradeCandle]:
    """Every candle from `entry_time` through `exit_time` inclusive
    (never later), each carrying EMA200/ADX14/Donchian20 computed from
    all history up to and including that candle (never a later one), a
    trailing realized-volatility figure, cumulative MFE/MAE, and the
    SHORT/LONG-aligned + ADX>threshold flags. Returns an empty list if
    inputs are missing or the window can't be sliced. No candle after
    `exit_time` is ever included."""
    if ohlcv is None or ohlcv.empty or entry_price is None or direction not in (LONG, SHORT):
        return []
    try:
        entry_ts = pd.Timestamp(entry_time)
        exit_ts = pd.Timestamp(exit_time) if exit_time is not None else None
    except (ValueError, TypeError):
        return []
    entry_ts = entry_ts.tz_localize("UTC") if entry_ts.tzinfo is None else entry_ts.tz_convert("UTC")
    if exit_ts is not None:
        exit_ts = exit_ts.tz_localize("UTC") if exit_ts.tzinfo is None else exit_ts.tz_convert("UTC")
        if exit_ts < entry_ts:
            return []

    full_history = ohlcv.sort_values("date").reset_index(drop=True)
    mask = full_history["date"] >= entry_ts
    if exit_ts is not None:
        mask &= full_history["date"] <= exit_ts
    window = full_history.loc[mask]
    if window.empty:
        return []

    records: list[TradeCandle] = []
    cum_mfe, cum_mae = 0.0, 0.0
    trade_closes: list[float] = []
    for _, row in window.iterrows():
        up_to = full_history.loc[full_history["date"] <= row["date"]]
        ema200_series = ema(up_to["close"], EMA_PERIOD)
        adx_series = adx(up_to, ADX_PERIOD)
        upper, lower = donchian_prev_bounds(up_to, DONCHIAN_PERIOD)
        ema_value = ema200_series.iloc[-1]
        adx_value = adx_series.iloc[-1]
        upper_value, lower_value = upper.iloc[-1], lower.iloc[-1]

        close, high, low = float(row["close"]), float(row["high"]), float(row["low"])
        trade_closes.append(close)

        if direction == SHORT:
            favorable = (entry_price - low) / entry_price * 100.0
            adverse = (high - entry_price) / entry_price * 100.0
        else:
            favorable = (high - entry_price) / entry_price * 100.0
            adverse = (entry_price - low) / entry_price * 100.0
        cum_mfe = max(cum_mfe, favorable)
        cum_mae = max(cum_mae, adverse)

        ema_distance = None
        aligned = None
        if pd.notna(ema_value) and ema_value:
            ema_distance = 100.0 * (close - ema_value) / ema_value
            aligned = bool(close < ema_value) if direction == SHORT else bool(close > ema_value)

        breakout = None
        if direction == SHORT and pd.notna(lower_value) and lower_value:
            breakout = 100.0 * (lower_value - close) / lower_value
        elif direction == LONG and pd.notna(upper_value) and upper_value:
            breakout = 100.0 * (close - upper_value) / upper_value

        recent_closes = trade_closes[-21:]
        realized_vol = None
        if len(recent_closes) >= 3:
            s = pd.Series(recent_closes)
            returns = s.pct_change().dropna()
            if len(returns) >= 2:
                realized_vol = float(returns.std()) * 100.0

        price_change_pct = 100.0 * (close - entry_price) / entry_price
        days_since_entry = (row["date"] - entry_ts).total_seconds() / 86400.0

        records.append(TradeCandle(
            date=row["date"], close=close,
            ema200=(float(ema_value) if pd.notna(ema_value) else None),
            ema_distance_pct=ema_distance,
            adx=(float(adx_value) if pd.notna(adx_value) else None),
            donchian_breakout_pct=breakout,
            realized_vol_pct=realized_vol,
            price_change_pct=price_change_pct,
            cumulative_mfe_pct=cum_mfe,
            cumulative_mae_pct=cum_mae,
            structurally_aligned=aligned,
            adx_above_threshold=(bool(adx_value > ADX_ENTRY_THRESHOLD) if pd.notna(adx_value) else None),
            days_since_entry=days_since_entry,
        ))
    return records


# ---------------------------------------------------------------------------
# Checkpoint slicing (+ checkpoint-to-checkpoint deltas) and run-length stats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointSnapshot:
    checkpoint_label: str
    closed_before_checkpoint: bool
    n_candles_in_subset: int
    mean_ema_distance_pct: float | None
    delta_ema_distance_pct: float | None
    mean_adx: float | None
    delta_adx: float | None
    mean_donchian_breakout_pct: float | None
    mean_realized_vol_pct: float | None
    price_change_pct: float | None
    mfe_pct: float | None
    mae_pct: float | None
    pct_structurally_aligned: float | None
    longest_run_aligned: int
    pct_adx_above_threshold: float | None
    longest_run_adx_above_threshold: int


def _longest_run(flags: Sequence[bool | None]) -> int:
    """Longest consecutive run of `True` in `flags`, treating `None` as a
    break (never counted as `True`)."""
    longest = current = 0
    for f in flags:
        if f is True:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _pct_true(flags: Sequence[bool | None]) -> float | None:
    resolved = [f for f in flags if f is not None]
    return (100.0 * sum(1 for f in resolved if f) / len(resolved)) if resolved else None


def _mean_or_none(values):
    resolved = [v for v in values if v is not None]
    return mean(resolved) if resolved else None


def _single_checkpoint_snapshot(
    full_trade_candles: Sequence[TradeCandle], checkpoint_label: str, checkpoint_minutes: float
) -> CheckpointSnapshot:
    """The metrics-table row for one checkpoint, sliced from an
    already-reconstructed full-trade candle list, with `delta_*` fields
    left `None` (filled in afterward by `all_checkpoint_snapshots`, which
    knows the prior checkpoint in ladder order). `closed_before_checkpoint
    =True` when the trade's own last candle is before the checkpoint
    horizon -- the subset then simply contains every candle the trade
    actually had, never a fabricated extension past the real exit."""
    if not full_trade_candles:
        return CheckpointSnapshot(
            checkpoint_label=checkpoint_label, closed_before_checkpoint=False, n_candles_in_subset=0,
            mean_ema_distance_pct=None, delta_ema_distance_pct=None, mean_adx=None, delta_adx=None,
            mean_donchian_breakout_pct=None, mean_realized_vol_pct=None, price_change_pct=None,
            mfe_pct=None, mae_pct=None, pct_structurally_aligned=None, longest_run_aligned=0,
            pct_adx_above_threshold=None, longest_run_adx_above_threshold=0,
        )
    checkpoint_days = checkpoint_minutes / 1440.0
    subset = [c for c in full_trade_candles if c.days_since_entry <= checkpoint_days]
    if not subset:
        subset = [full_trade_candles[0]]
    closed_before = full_trade_candles[-1].days_since_entry < checkpoint_days

    aligned_flags = [c.structurally_aligned for c in subset]
    adx_flags = [c.adx_above_threshold for c in subset]
    last = subset[-1]
    return CheckpointSnapshot(
        checkpoint_label=checkpoint_label,
        closed_before_checkpoint=closed_before,
        n_candles_in_subset=len(subset),
        mean_ema_distance_pct=_mean_or_none([c.ema_distance_pct for c in subset]),
        delta_ema_distance_pct=None,
        mean_adx=_mean_or_none([c.adx for c in subset]),
        delta_adx=None,
        mean_donchian_breakout_pct=_mean_or_none([c.donchian_breakout_pct for c in subset]),
        mean_realized_vol_pct=_mean_or_none([c.realized_vol_pct for c in subset]),
        price_change_pct=last.price_change_pct,
        mfe_pct=last.cumulative_mfe_pct,
        mae_pct=last.cumulative_mae_pct,
        pct_structurally_aligned=_pct_true(aligned_flags),
        longest_run_aligned=_longest_run(aligned_flags),
        pct_adx_above_threshold=_pct_true(adx_flags),
        longest_run_adx_above_threshold=_longest_run(adx_flags),
    )


def all_checkpoint_snapshots(
    full_trade_candles: Sequence[TradeCandle], checkpoints: dict[str, float] = CHECKPOINTS
) -> dict[str, CheckpointSnapshot]:
    """`_single_checkpoint_snapshot` for every entry in `checkpoints`, in
    ladder order, with `delta_ema_distance_pct`/`delta_adx` filled in as
    the change from the *previous ladder checkpoint's* mean value (`None`
    for the first checkpoint, or whenever either side is unavailable).
    Deltas are computed strictly between consecutive fixed-ladder labels
    -- never across a skipped or interpolated point."""
    raw = {label: _single_checkpoint_snapshot(full_trade_candles, label, minutes) for label, minutes in checkpoints.items()}
    ordered_labels = list(checkpoints.keys())
    result: dict[str, CheckpointSnapshot] = {}
    prev_ema, prev_adx = None, None
    for label in ordered_labels:
        snap = raw[label]
        delta_ema = (
            snap.mean_ema_distance_pct - prev_ema
            if snap.mean_ema_distance_pct is not None and prev_ema is not None
            else None
        )
        delta_adx = (
            snap.mean_adx - prev_adx
            if snap.mean_adx is not None and prev_adx is not None
            else None
        )
        result[label] = CheckpointSnapshot(
            checkpoint_label=snap.checkpoint_label,
            closed_before_checkpoint=snap.closed_before_checkpoint,
            n_candles_in_subset=snap.n_candles_in_subset,
            mean_ema_distance_pct=snap.mean_ema_distance_pct,
            delta_ema_distance_pct=delta_ema,
            mean_adx=snap.mean_adx,
            delta_adx=delta_adx,
            mean_donchian_breakout_pct=snap.mean_donchian_breakout_pct,
            mean_realized_vol_pct=snap.mean_realized_vol_pct,
            price_change_pct=snap.price_change_pct,
            mfe_pct=snap.mfe_pct,
            mae_pct=snap.mae_pct,
            pct_structurally_aligned=snap.pct_structurally_aligned,
            longest_run_aligned=snap.longest_run_aligned,
            pct_adx_above_threshold=snap.pct_adx_above_threshold,
            longest_run_adx_above_threshold=snap.longest_run_adx_above_threshold,
        )
        if snap.mean_ema_distance_pct is not None:
            prev_ema = snap.mean_ema_distance_pct
        if snap.mean_adx is not None:
            prev_adx = snap.mean_adx
    return result


# ---------------------------------------------------------------------------
# Group-level aggregation across trades at one checkpoint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupCheckpointAggregate:
    checkpoint_label: str
    n_total: int
    n_reached: int
    is_thin_sample: bool
    mean_ema_distance_pct: float | None
    mean_delta_ema_distance_pct: float | None
    mean_adx: float | None
    mean_delta_adx: float | None
    mean_donchian_breakout_pct: float | None
    mean_realized_vol_pct: float | None
    mean_mfe_pct: float | None
    mean_mae_pct: float | None
    pct_structurally_aligned: float | None
    mean_longest_run_aligned: float | None
    pct_adx_above_threshold: float | None
    mean_longest_run_adx_above_threshold: float | None


def aggregate_group_checkpoint(
    per_trade_snapshots: Sequence[dict[str, CheckpointSnapshot]], checkpoint_label: str
) -> GroupCheckpointAggregate:
    """Group-level aggregate for `checkpoint_label` across
    `per_trade_snapshots` (one `all_checkpoint_snapshots` result per
    trade). A trade with zero candles in its subset at this checkpoint
    contributes to `n_total` but not `n_reached` or any mean."""
    entries = [s[checkpoint_label] for s in per_trade_snapshots if checkpoint_label in s]
    reached = [e for e in entries if e.n_candles_in_subset > 0]

    return GroupCheckpointAggregate(
        checkpoint_label=checkpoint_label,
        n_total=len(entries),
        n_reached=len(reached),
        is_thin_sample=(len(reached) < THIN_SAMPLE_THRESHOLD),
        mean_ema_distance_pct=_mean_or_none([e.mean_ema_distance_pct for e in reached]),
        mean_delta_ema_distance_pct=_mean_or_none([e.delta_ema_distance_pct for e in reached]),
        mean_adx=_mean_or_none([e.mean_adx for e in reached]),
        mean_delta_adx=_mean_or_none([e.delta_adx for e in reached]),
        mean_donchian_breakout_pct=_mean_or_none([e.mean_donchian_breakout_pct for e in reached]),
        mean_realized_vol_pct=_mean_or_none([e.mean_realized_vol_pct for e in reached]),
        mean_mfe_pct=_mean_or_none([e.mfe_pct for e in reached]),
        mean_mae_pct=_mean_or_none([e.mae_pct for e in reached]),
        pct_structurally_aligned=_mean_or_none([e.pct_structurally_aligned for e in reached]),
        mean_longest_run_aligned=_mean_or_none([float(e.longest_run_aligned) for e in reached]),
        pct_adx_above_threshold=_mean_or_none([e.pct_adx_above_threshold for e in reached]),
        mean_longest_run_adx_above_threshold=_mean_or_none(
            [float(e.longest_run_adx_above_threshold) for e in reached]
        ),
    )


# ---------------------------------------------------------------------------
# Divergence detection (descriptive, non-statistical)
# ---------------------------------------------------------------------------

# The variables inspected for divergence, and how to read each one off a
# `GroupCheckpointAggregate`. Fixed, literal list -- not tuned by results.
DIVERGENCE_VARIABLES: tuple[str, ...] = (
    "mean_ema_distance_pct",
    "mean_adx",
    "mean_donchian_breakout_pct",
    "mean_realized_vol_pct",
    "mean_mfe_pct",
    "mean_mae_pct",
    "pct_structurally_aligned",
    "pct_adx_above_threshold",
)


@dataclass(frozen=True)
class DivergenceRow:
    checkpoint_label: str
    variable: str
    persistent_value: float | None
    ordinary_value: float | None
    gap: float | None  # persistent - ordinary


def compute_divergence_table(
    persistent_aggregates: dict[str, GroupCheckpointAggregate],
    ordinary_aggregates: dict[str, GroupCheckpointAggregate],
    checkpoint_order: Sequence[str] = CHECKPOINT_ORDER,
    variables: Sequence[str] = DIVERGENCE_VARIABLES,
) -> list[DivergenceRow]:
    """One `DivergenceRow` per (checkpoint, variable), in ladder order.
    `gap` is `None` whenever either side's value is `None` at that
    checkpoint -- never imputed."""
    rows: list[DivergenceRow] = []
    for label in checkpoint_order:
        p = persistent_aggregates.get(label)
        o = ordinary_aggregates.get(label)
        for var in variables:
            p_val = getattr(p, var) if p is not None else None
            o_val = getattr(o, var) if o is not None else None
            gap = (p_val - o_val) if (p_val is not None and o_val is not None) else None
            rows.append(DivergenceRow(checkpoint_label=label, variable=var, persistent_value=p_val, ordinary_value=o_val, gap=gap))
    return rows


def first_sign_consistent_checkpoint(
    gaps_by_checkpoint: Sequence[tuple[str, float | None]],
) -> str | None:
    """Given `(checkpoint_label, gap)` pairs in fixed ladder order, return
    the earliest checkpoint label from which the sign of `gap` (positive,
    negative, or exactly-zero-as-its-own-sign) stays constant through
    every remaining checkpoint in the sequence. Returns `None` if no such
    point exists (including when the tail contains a `None` gap, when
    there are fewer than 2 checkpoints, or when the only sign-consistent
    tail is a single trailing point -- a lone checkpoint is not "sustained"
    separation by this definition, so a qualifying tail must span at
    least 2 checkpoints). This is a pure consistency-of-direction check --
    it never compares a gap's magnitude against any threshold, and never
    claims statistical significance."""
    n = len(gaps_by_checkpoint)
    if n < 2:
        return None

    def _sign(x: float) -> int:
        if x > 0:
            return 1
        if x < 0:
            return -1
        return 0

    for start in range(n - 1):
        tail = gaps_by_checkpoint[start:]
        if any(g is None for _, g in tail):
            continue
        signs = {_sign(g) for _, g in tail}
        if len(signs) == 1:
            return gaps_by_checkpoint[start][0]
    return None


def earliest_divergence_by_variable(
    divergence_rows: Sequence[DivergenceRow], checkpoint_order: Sequence[str] = CHECKPOINT_ORDER,
) -> dict[str, str | None]:
    """For every variable present in `divergence_rows`, the earliest
    sign-consistent checkpoint per `first_sign_consistent_checkpoint`."""
    by_variable: dict[str, list[tuple[str, float | None]]] = {}
    order_index = {label: i for i, label in enumerate(checkpoint_order)}
    for row in divergence_rows:
        by_variable.setdefault(row.variable, []).append((row.checkpoint_label, row.gap))
    result: dict[str, str | None] = {}
    for var, pairs in by_variable.items():
        pairs_sorted = sorted(pairs, key=lambda pair: order_index.get(pair[0], len(checkpoint_order)))
        result[var] = first_sign_consistent_checkpoint(pairs_sorted)
    return result


def overall_earliest_divergence(
    divergence_rows: Sequence[DivergenceRow], checkpoint_order: Sequence[str] = CHECKPOINT_ORDER,
) -> str | None:
    """The single earliest checkpoint (ladder order) at which *any*
    variable first shows a sign-consistent gap, per
    `earliest_divergence_by_variable`. `None` if no variable ever
    diverges under that definition."""
    per_variable = earliest_divergence_by_variable(divergence_rows, checkpoint_order)
    order_index = {label: i for i, label in enumerate(checkpoint_order)}
    candidates = [label for label in per_variable.values() if label is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda label: order_index.get(label, len(checkpoint_order)))
