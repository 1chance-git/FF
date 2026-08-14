"""SHORT-winner PERSISTENT vs ORDINARY persistence-threshold audit
(research/analysis only).

Answers one question only: "At what post-entry age, if any, does a SHORT
winner's structural behavior begin to look meaningfully different from a
normal SHORT winner?" Restricted to a coarser 7d-30d checkpoint ladder and
exactly 6 variables (EMA200 distance, ADX, Donchian breakout distance,
MFE, %time structurally aligned, %time ADX>25). This module never runs a
backtest, never touches `TrendFollowCore.py`, config, or the pair
whitelist, never redefines the persistent/ordinary grouping, and never
proposes a trading rule or optimizes a threshold.

Design decisions
-----------------
* **All trade reconstruction and checkpoint-slicing machinery is reused
  verbatim from `hermes.short_runner_divergence_audit`** -- same
  EMA200/ADX14/Donchian20 reimplementation, same
  `reconstruct_full_trade`/`all_checkpoint_snapshots`/
  `aggregate_group_checkpoint`, same `classify_group`/`PERSISTENT_KEYS`
  identity matching. Nothing here is a sixth separately-written
  reimplementation of those primitives.
* **Only the 8 SHORT winners are used** (PERSISTENT n=3, ORDINARY n=5) --
  SHORT losers are out of scope for this block, unlike the prior
  divergence-audit block.
* **`CHECKPOINTS` is the block's own literal 7-point ladder** (7d, 10d,
  14d, 17d, 21d, 24d, 30d) -- coarser than, and a strict subset of, the
  12-point ladder used in `short_runner_divergence_audit`. Never
  extended or tuned based on results.
* **"Meaningful" divergence is explicitly NOT a sign-only rule.** A gap
  at checkpoint *i* only counts as the first meaningful divergence if (a)
  its magnitude exceeds every strictly-earlier checkpoint's gap magnitude
  in this same variable/dataset (i.e. it is a genuine new high-water mark
  against the noise already observed in this run, not an arbitrary
  invented cutoff), and (b) the sign from that checkpoint through the
  final checkpoint stays constant (direction holds up, not a one-off
  spike). Both conditions are computed purely from the data already
  produced by this same run -- no external threshold, no
  optimization, no significance test.
* **Per-trade support** checks, at the checkpoint each variable is first
  flagged as meaningfully diverging, whether each of the 3 individual
  PERSISTENT trades sits on the same side of the ORDINARY group's mean as
  the group-level gap direction -- descriptive agreement, not a vote or a
  statistical test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from hermes.short_runner_divergence_audit import (
    LONG,
    SHORT,
    PERSISTENT_KEYS,
    THIN_SAMPLE_THRESHOLD,
    GroupCheckpointAggregate,
    TradeCandle,
    all_checkpoint_snapshots,
    aggregate_group_checkpoint,
    classify_group,
    reconstruct_full_trade,
)

# The block's own literal 7-point checkpoint ladder -- a strict subset of
# short_runner_divergence_audit.CHECKPOINTS, never extended or tuned.
CHECKPOINTS: dict[str, float] = {
    "7d": 10080.0,
    "10d": 14400.0,
    "14d": 20160.0,
    "17d": 24480.0,
    "21d": 30240.0,
    "24d": 34560.0,
    "30d": 43200.0,
}
CHECKPOINT_ORDER: tuple[str, ...] = tuple(CHECKPOINTS.keys())

# Exactly the 6 variables named in this block -- no additional indicators.
VARIABLES: tuple[str, ...] = (
    "mean_ema_distance_pct",
    "mean_adx",
    "mean_donchian_breakout_pct",
    "mean_mfe_pct",
    "pct_structurally_aligned",
    "pct_adx_above_threshold",
)

VARIABLE_LABELS: dict[str, str] = {
    "mean_ema_distance_pct": "EMA200 distance %",
    "mean_adx": "ADX",
    "mean_donchian_breakout_pct": "Donchian breakout distance %",
    "mean_mfe_pct": "MFE %",
    "pct_structurally_aligned": "% time structurally aligned",
    "pct_adx_above_threshold": "% time ADX>25",
}


@dataclass(frozen=True)
class GapRow:
    checkpoint_label: str
    variable: str
    persistent_value: float | None
    ordinary_value: float | None
    gap: float | None  # persistent - ordinary


def compute_gap_table(
    persistent_aggregates: dict[str, GroupCheckpointAggregate],
    ordinary_aggregates: dict[str, GroupCheckpointAggregate],
    checkpoint_order: Sequence[str] = CHECKPOINT_ORDER,
    variables: Sequence[str] = VARIABLES,
) -> list[GapRow]:
    """One `GapRow` per (checkpoint, variable), in ladder order. `gap` is
    `None` whenever either side's value is `None` at that checkpoint --
    never imputed."""
    rows: list[GapRow] = []
    for label in checkpoint_order:
        p = persistent_aggregates.get(label)
        o = ordinary_aggregates.get(label)
        for var in variables:
            p_val = getattr(p, var) if p is not None else None
            o_val = getattr(o, var) if o is not None else None
            gap = (p_val - o_val) if (p_val is not None and o_val is not None) else None
            rows.append(GapRow(checkpoint_label=label, variable=var, persistent_value=p_val, ordinary_value=o_val, gap=gap))
    return rows


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def first_meaningful_divergence_checkpoint(
    gaps_by_checkpoint: Sequence[tuple[str, float | None]],
) -> str | None:
    """The earliest checkpoint at which this variable's gap is BOTH (a) a
    new magnitude high-water-mark against every strictly-earlier
    checkpoint's gap magnitude in this same sequence, and (b) sign-stable
    from that checkpoint through the end of the sequence (requires at
    least 2 checkpoints in that tail -- a lone trailing point is not
    "held afterward"). This is explicitly NOT a sign-only rule: a tiny
    gap that merely flips sign never qualifies unless it is also the
    largest-magnitude gap seen so far. Returns `None` if no checkpoint
    satisfies both conditions. A `None` gap resets nothing -- it is
    simply skipped when updating the noise floor."""
    n = len(gaps_by_checkpoint)
    noise_floor: float | None = None  # undefined until at least one earlier checkpoint has been observed
    for i in range(n):
        label, gap = gaps_by_checkpoint[i]
        if gap is None:
            continue
        magnitude = abs(gap)
        # A checkpoint can only "exceed the earlier noise" once there IS
        # earlier noise to compare against -- the first observed gap
        # never qualifies on its own.
        if noise_floor is not None and magnitude > noise_floor:
            tail = gaps_by_checkpoint[i:]
            if len(tail) >= 2 and all(g is not None for _, g in tail):
                signs = {_sign(g) for _, g in tail}
                if len(signs) == 1:
                    return label
        noise_floor = magnitude if noise_floor is None else max(noise_floor, magnitude)
    return None


def earliest_meaningful_divergence_by_variable(
    gap_rows: Sequence[GapRow], checkpoint_order: Sequence[str] = CHECKPOINT_ORDER,
) -> dict[str, str | None]:
    """For every variable present in `gap_rows`, the earliest meaningful
    divergence checkpoint per `first_meaningful_divergence_checkpoint`."""
    order_index = {label: i for i, label in enumerate(checkpoint_order)}
    by_variable: dict[str, list[tuple[str, float | None]]] = {}
    for row in gap_rows:
        by_variable.setdefault(row.variable, []).append((row.checkpoint_label, row.gap))
    result: dict[str, str | None] = {}
    for var, pairs in by_variable.items():
        pairs_sorted = sorted(pairs, key=lambda pair: order_index.get(pair[0], len(checkpoint_order)))
        result[var] = first_meaningful_divergence_checkpoint(pairs_sorted)
    return result


def variables_agreeing_on_window(
    earliest_by_variable: dict[str, str | None], checkpoint_order: Sequence[str] = CHECKPOINT_ORDER,
) -> dict[str, list[str]]:
    """Groups variables by their earliest-meaningful-divergence
    checkpoint (variables with `None` are grouped under the key `None`).
    Used to answer "which variables agree on the same transition window"
    descriptively -- not a clustering algorithm, just a groupby."""
    groups: dict[str, list[str]] = {}
    for var, label in earliest_by_variable.items():
        groups.setdefault(label, []).append(var)
    return groups


# ---------------------------------------------------------------------------
# Per-trade support check
# ---------------------------------------------------------------------------


# `GroupCheckpointAggregate` and the per-trade `CheckpointSnapshot` name
# the MFE field differently (`mean_mfe_pct` vs `mfe_pct`); every other
# variable in `VARIABLES` shares its name across both dataclasses. This
# maps a group-aggregate field name to its per-trade-snapshot equivalent.
_TRADE_SNAPSHOT_FIELD: dict[str, str] = {"mean_mfe_pct": "mfe_pct"}


@dataclass(frozen=True)
class TradeSupport:
    trade_label: str
    variable: str
    checkpoint_label: str
    trade_value: float | None
    ordinary_mean: float | None
    group_gap_sign: int
    supports_direction: bool | None  # None when either value is unavailable


def per_trade_support(
    persistent_trade_snapshots: Sequence[dict],
    persistent_trade_labels: Sequence[str],
    ordinary_aggregates: dict[str, GroupCheckpointAggregate],
    variable: str,
    checkpoint_label: str,
    group_gap: float,
) -> list[TradeSupport]:
    """For each individual PERSISTENT trade, whether its own value at
    `checkpoint_label` sits on the same side of the ORDINARY group's mean
    as the group-level gap's sign. A trade with a `None` value at this
    checkpoint yields `supports_direction=None` (never coerced to
    False)."""
    group_sign = _sign(group_gap)
    ordinary_mean = getattr(ordinary_aggregates.get(checkpoint_label), variable, None) if ordinary_aggregates.get(checkpoint_label) is not None else None
    results: list[TradeSupport] = []
    snapshot_field = _TRADE_SNAPSHOT_FIELD.get(variable, variable)
    for label, snaps in zip(persistent_trade_labels, persistent_trade_snapshots):
        snap = snaps.get(checkpoint_label)
        trade_value = getattr(snap, snapshot_field, None) if snap is not None else None
        supports = None
        if trade_value is not None and ordinary_mean is not None and group_sign != 0:
            trade_diff = trade_value - ordinary_mean
            supports = _sign(trade_diff) == group_sign
        results.append(TradeSupport(
            trade_label=label, variable=variable, checkpoint_label=checkpoint_label,
            trade_value=trade_value, ordinary_mean=ordinary_mean, group_gap_sign=group_sign,
            supports_direction=supports,
        ))
    return results
