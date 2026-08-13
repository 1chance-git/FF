"""EMA200-ceiling MECHANISM VALIDATION audit (research/analysis only).

Answers one question only: does the EMA200-distance entry-ceiling already
measured by `hermes.ema_ceiling_forensics` / `hermes.ema_ceiling_quality_audit`
/ `hermes.ema_ceiling_temporal_audit` primarily remove highly extended,
stop-loss-prone entries while retaining a meaningful portion of winning
breakout trades -- i.e. does the *mechanism* we believe is operating
actually explain the observed improvement, or is the improvement
unrelated to EMA-extension/stop-loss exposure?

This module never launches a backtest, never picks a "winning" threshold,
and never touches `TrendFollowCore.py`, config, or strategy
entry/exit/stoploss logic -- it only computes pure arithmetic and
classification over already-loaded per-trade records that carry both the
already-known outcome (`hermes.trade_report`/`hermes.ema_ceiling_forensics`
shape) and the already-reconstructed signal-candle context
(`hermes.signal_forensics`: EMA distance, Donchian breakout %, ADX) and
volatility context (`hermes.volatility_forensics`: ATR%, realized vol).

Four independent pieces live here:

1. **`MechanismTradeRecord`** -- one baseline trade with every metric the
   mechanism test needs attached, plus `merge_mechanism_records` to build
   a list of these from a `BaselineTradeRecord` list (outcome) joined
   against signal-forensics and volatility-forensics per-trade dicts
   (context), by `(pair, entry_time, direction)` identity -- the same
   identity key `hermes.ema_ceiling_quality_audit._trade_identity` already
   uses for baseline/variant matching, so this module doesn't invent a
   second one.
2. **A/B/C/D/E outcome classification** (`classify_outcome_category`) --
   the exact five-way split the REMOVED-TRADE ANALYSIS step of the spec
   asks for: stop-loss / losing exit-signal / winning exit-signal / force
   exit / unresolved. Never infers a missing `exit_reason` or `is_winner`
   into a category; both missing means `E_UNRESOLVED`.
3. **Removed-vs-retained metric comparison** (`compare_removed_vs_retained`)
   -- mean/median of EMA distance, Donchian breakout %, ADX, ATR%, and
   realized vol for the removed set vs. the retained set at one
   threshold, plus `stop_loss_removed_pct`/`winner_removed_pct` so the
   caller can directly answer "does the ceiling remove stop-loss trades
   at a higher rate than winners" (Step 5) without recomputing the ratio
   itself.
4. **Pair/direction stratification** (`stratify_by_group`) -- the same
   removed/retained comparison restricted to one BTC/ETH/SOL pair or one
   LONG/SHORT direction, reusing `compare_removed_vs_retained` rather
   than a parallel implementation, so Step 7 can never silently diverge
   from Step 4/5's math.

Every function here operates on values already computed elsewhere
(`hermes.signal_forensics`, `hermes.volatility_forensics`,
`hermes.ema_ceiling_forensics`); nothing recomputes an EMA, ADX, Donchian
level, ATR, or realized-vol value from OHLCV. A `None` metric is always
excluded from mean/median rather than treated as zero, and a group with
zero trades in a bucket returns `None` for that bucket's stats rather
than fabricating a value from an empty list.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# MechanismTradeRecord: outcome + signal context + volatility context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MechanismTradeRecord:
    """One frozen baseline trade with outcome + every mechanism-test metric
    already attached. Any metric that couldn't be reconstructed for this
    trade stays `None` -- never fabricated or interpolated."""

    trade_number: int | None
    pair: str | None
    direction: str | None
    entry_time: str | None
    exit_reason: str | None
    profit_pct: float | None
    profit_abs: float | None
    is_winner: bool | None
    ema_distance_pct: float | None  # abs, percentage points (0-100 scale)
    breakout_distance_pct: float | None  # abs, percentage points
    adx14: float | None
    atr_pct: float | None
    realized_vol: float | None


def _trade_identity(trade: Any) -> tuple:
    """Same identity convention as `hermes.ema_ceiling_quality_audit`:
    `(pair, entry_time, direction)` uniquely identifies a trade within one
    backtest window/pair set (no pyramiding observed in the frozen
    baseline)."""
    return (
        getattr(trade, "pair", None),
        getattr(trade, "entry_time", None),
        getattr(trade, "direction", None),
    )


def merge_mechanism_records(
    baseline_trades: Sequence[Any],
    signal_trades_by_identity: dict[tuple, dict[str, Any]],
    volatility_trades_by_identity: dict[tuple, dict[str, Any]],
) -> list[MechanismTradeRecord]:
    """Join outcome (`baseline_trades`, `hermes.trade_report.Trade`-shaped)
    against signal context and volatility context, by identity.

    `signal_trades_by_identity`/`volatility_trades_by_identity` are dicts
    keyed by `_trade_identity(...)`-shaped tuples, built by the caller from
    `hermes.signal_forensics`/`hermes.volatility_forensics` per-trade JSON
    (each entry already carries `ema_distance_pct` as a *fraction*, per
    `compute_ema_distance_pct`'s convention -- this function converts to
    the 0-100 percentage scale `hermes.ema_ceiling_forensics` uses, via
    `abs(x) * 100`, so every metric here is on a consistent percentage
    scale). A trade absent from either lookup gets `None` for that
    lookup's metrics, not a KeyError or a guessed value.
    """
    records = []
    for t in baseline_trades:
        key = _trade_identity(t)
        sig = signal_trades_by_identity.get(key)
        vol = volatility_trades_by_identity.get(key)

        ema_distance_pct = None
        breakout_distance_pct = None
        adx14 = None
        if sig is not None:
            raw_ema = sig.get("ema_distance_pct")
            ema_distance_pct = abs(raw_ema) * 100.0 if raw_ema is not None else None
            raw_breakout = sig.get("breakout_distance_pct")
            breakout_distance_pct = (
                abs(raw_breakout) * 100.0 if raw_breakout is not None else None
            )
            adx14 = sig.get("adx14")

        atr_pct = None
        realized_vol = None
        if vol is not None:
            atr_pct = vol.get("atr_pct")
            realized_vol = vol.get("realized_vol")
            # Fall back to volatility-forensics' own adx14/ema_distance_pct
            # only if signal-forensics didn't have them (both modules
            # reconstruct the same values independently; signal_forensics
            # is preferred as the more complete EntryContext, but a trade
            # missing from it should not lose an otherwise-available value).
            if adx14 is None:
                adx14 = vol.get("adx14")
            if ema_distance_pct is None:
                raw_ema_v = vol.get("ema_distance_pct")
                ema_distance_pct = (
                    abs(raw_ema_v) * 100.0 if raw_ema_v is not None else None
                )

        records.append(
            MechanismTradeRecord(
                trade_number=(sig or {}).get("trade_number") or (vol or {}).get("trade_number"),
                pair=t.pair,
                direction=t.direction,
                entry_time=t.entry_time,
                exit_reason=t.exit_reason,
                profit_pct=t.profit_pct,
                profit_abs=t.profit_abs,
                is_winner=t.is_winner,
                ema_distance_pct=ema_distance_pct,
                breakout_distance_pct=breakout_distance_pct,
                adx14=adx14,
                atr_pct=atr_pct,
                realized_vol=realized_vol,
            )
        )
    return records


# ---------------------------------------------------------------------------
# A/B/C/D/E outcome classification (REMOVED-TRADE ANALYSIS step)
# ---------------------------------------------------------------------------

CATEGORY_STOP_LOSS = "A_STOP_LOSS"
CATEGORY_LOSING_EXIT_SIGNAL = "B_LOSING_EXIT_SIGNAL"
CATEGORY_WINNING_EXIT_SIGNAL = "C_WINNING_EXIT_SIGNAL"
CATEGORY_FORCE_EXIT = "D_FORCE_EXIT"
CATEGORY_UNRESOLVED = "E_UNRESOLVED"


def classify_outcome_category(record: MechanismTradeRecord) -> str:
    """A/B/C/D/E per the spec's REMOVED-TRADE ANALYSIS classification.

    A trade whose `exit_reason` or `is_winner` is unknown is `E_UNRESOLVED`
    -- never guessed into A-D from partial information (e.g. a `None`
    `is_winner` with `exit_reason == "exit_signal"` is unresolved, not
    assumed to be a loser)."""
    if record.exit_reason is None or record.is_winner is None:
        return CATEGORY_UNRESOLVED
    if record.exit_reason == "stop_loss":
        return CATEGORY_STOP_LOSS
    if record.exit_reason == "exit_signal":
        return (
            CATEGORY_WINNING_EXIT_SIGNAL if record.is_winner else CATEGORY_LOSING_EXIT_SIGNAL
        )
    if record.exit_reason == "force_exit":
        return CATEGORY_FORCE_EXIT
    return CATEGORY_UNRESOLVED


# ---------------------------------------------------------------------------
# Removed-vs-retained split (given a set of kept identities from a ceiling)
# ---------------------------------------------------------------------------


def split_removed_retained(
    records: Sequence[MechanismTradeRecord], kept_identities: set[tuple]
) -> tuple[list[MechanismTradeRecord], list[MechanismTradeRecord]]:
    """Split `records` into (removed, retained) using `kept_identities` --
    a set of `(pair, entry_time, direction)` tuples present in a ceiling
    variant's trade export (the same identity convention `_trade_identity`
    uses). A record whose identity is in `kept_identities` is retained;
    every other record was removed by the ceiling."""
    removed, retained = [], []
    for r in records:
        key = (r.pair, r.entry_time, r.direction)
        (retained if key in kept_identities else removed).append(r)
    return removed, retained


# ---------------------------------------------------------------------------
# Removed-vs-retained metric comparison (MECHANISM TEST, Step 4)
# ---------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return (sum(clean) / len(clean)) if clean else None


def _median(values: Sequence[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return median(clean) if clean else None


@dataclass(frozen=True)
class MetricComparison:
    metric: str
    removed_mean: float | None
    removed_median: float | None
    retained_mean: float | None
    retained_median: float | None
    removed_n: int
    retained_n: int


_METRIC_FIELDS = (
    "ema_distance_pct",
    "breakout_distance_pct",
    "realized_vol",
    "atr_pct",
    "adx14",
)


def compare_metric(
    metric: str, removed: Sequence[MechanismTradeRecord], retained: Sequence[MechanismTradeRecord]
) -> MetricComparison:
    removed_values = [getattr(r, metric) for r in removed if getattr(r, metric) is not None]
    retained_values = [getattr(r, metric) for r in retained if getattr(r, metric) is not None]
    return MetricComparison(
        metric=metric,
        removed_mean=_mean(removed_values),
        removed_median=_median(removed_values),
        retained_mean=_mean(retained_values),
        retained_median=_median(retained_values),
        removed_n=len(removed_values),
        retained_n=len(retained_values),
    )


def compare_all_metrics(
    removed: Sequence[MechanismTradeRecord], retained: Sequence[MechanismTradeRecord]
) -> dict[str, MetricComparison]:
    """`compare_metric` for every mechanism-test metric (Step 4's full
    removed-vs-retained table for one threshold)."""
    return {m: compare_metric(m, removed, retained) for m in _METRIC_FIELDS}


# ---------------------------------------------------------------------------
# Stop-loss / winner elimination rates (Step 5, Step 6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EliminationRates:
    """Step 5's core comparison: does the ceiling remove stop-loss trades
    at a higher rate than it removes winners?"""

    threshold_label: str
    baseline_stop_loss_count: int
    removed_stop_loss_count: int
    stop_loss_removed_pct: float | None
    baseline_winner_count: int
    removed_winner_count: int
    winner_removed_pct: float | None
    baseline_losing_exit_signal_count: int
    removed_losing_exit_signal_count: int
    losing_exit_signal_removed_pct: float | None

    @property
    def removes_stop_losses_faster_than_winners(self) -> bool | None:
        """`True`/`False` if both rates are known, `None` if either is
        undefined (e.g. zero baseline stop-loss trades)."""
        if self.stop_loss_removed_pct is None or self.winner_removed_pct is None:
            return None
        return self.stop_loss_removed_pct > self.winner_removed_pct


def _pct(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return 100.0 * numerator / denominator


def compute_elimination_rates(
    threshold_label: str,
    all_baseline: Sequence[MechanismTradeRecord],
    removed: Sequence[MechanismTradeRecord],
) -> EliminationRates:
    baseline_categories = [classify_outcome_category(r) for r in all_baseline]
    removed_categories = [classify_outcome_category(r) for r in removed]

    baseline_sl = baseline_categories.count(CATEGORY_STOP_LOSS)
    removed_sl = removed_categories.count(CATEGORY_STOP_LOSS)

    baseline_winners = sum(1 for r in all_baseline if r.is_winner is True)
    removed_winners = sum(1 for r in removed if r.is_winner is True)

    baseline_losing_exit = baseline_categories.count(CATEGORY_LOSING_EXIT_SIGNAL)
    removed_losing_exit = removed_categories.count(CATEGORY_LOSING_EXIT_SIGNAL)

    return EliminationRates(
        threshold_label=threshold_label,
        baseline_stop_loss_count=baseline_sl,
        removed_stop_loss_count=removed_sl,
        stop_loss_removed_pct=_pct(removed_sl, baseline_sl),
        baseline_winner_count=baseline_winners,
        removed_winner_count=removed_winners,
        winner_removed_pct=_pct(removed_winners, baseline_winners),
        baseline_losing_exit_signal_count=baseline_losing_exit,
        removed_losing_exit_signal_count=removed_losing_exit,
        losing_exit_signal_removed_pct=_pct(removed_losing_exit, baseline_losing_exit),
    )


# ---------------------------------------------------------------------------
# Winner preservation (Step 6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WinnerPreservation:
    threshold_label: str
    baseline_winner_count: int
    winners_removed: tuple[MechanismTradeRecord, ...]
    winners_retained: tuple[MechanismTradeRecord, ...]

    @property
    def winners_retained_pct(self) -> float | None:
        return _pct(len(self.winners_retained), self.baseline_winner_count)

    @property
    def removed_winner_total_profit_abs(self) -> float:
        return sum(w.profit_abs for w in self.winners_removed if w.profit_abs is not None)

    @property
    def removed_winner_total_profit_pct(self) -> float:
        return sum(w.profit_pct for w in self.winners_removed if w.profit_pct is not None)


def compute_winner_preservation(
    threshold_label: str, removed: Sequence[MechanismTradeRecord], retained: Sequence[MechanismTradeRecord]
) -> WinnerPreservation:
    winners_removed = tuple(r for r in removed if r.is_winner is True)
    winners_retained = tuple(r for r in retained if r.is_winner is True)
    return WinnerPreservation(
        threshold_label=threshold_label,
        baseline_winner_count=len(winners_removed) + len(winners_retained),
        winners_removed=winners_removed,
        winners_retained=winners_retained,
    )


# ---------------------------------------------------------------------------
# Pair / direction stratification (Step 7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupStratification:
    group_label: str
    baseline_count: int
    removed_count: int
    elimination_rates: EliminationRates
    removed_ema_distance_mean: float | None
    removed_ema_distance_median: float | None


def stratify_by_group(
    group_label: str,
    threshold_label: str,
    all_baseline: Sequence[MechanismTradeRecord],
    removed: Sequence[MechanismTradeRecord],
    predicate,
) -> GroupStratification:
    """`compute_elimination_rates` restricted to records matching
    `predicate` (e.g. `lambda r: r.pair == "BTC/USDC:USDC"` or
    `lambda r: r.direction == "LONG"`) -- reuses the exact same rate
    arithmetic as Step 5 rather than a parallel per-group implementation,
    so a pair/direction breakdown can never silently disagree with the
    whole-baseline numbers it is a subset of."""
    group_baseline = [r for r in all_baseline if predicate(r)]
    group_removed = [r for r in removed if predicate(r)]
    rates = compute_elimination_rates(threshold_label, group_baseline, group_removed)
    removed_ema = [r.ema_distance_pct for r in group_removed if r.ema_distance_pct is not None]
    return GroupStratification(
        group_label=group_label,
        baseline_count=len(group_baseline),
        removed_count=len(group_removed),
        elimination_rates=rates,
        removed_ema_distance_mean=_mean(removed_ema),
        removed_ema_distance_median=_median(removed_ema),
    )
