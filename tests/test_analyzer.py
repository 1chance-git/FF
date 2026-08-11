"""Unit tests for hermes.analyzer."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from hermes.analyzer import (
    AnalysisReport,
    analyze,
    average_holding_time_seconds,
    average_pnl,
    expectancy,
    fees_and_funding_drag_pct,
    largest_losses,
    max_consecutive_losses,
    max_drawdown,
    performance_by_entry_zscore,
    performance_by_exit_zscore,
    performance_by_regime,
    profit_factor,
    win_rate,
)
from hermes.memory import MemoryStore, TradeRecord

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _trade(offset_hours: float = 0, **kwargs) -> TradeRecord:
    defaults = dict(pair="BTC/USDC:USDC", recorded_at=_T0 + timedelta(hours=offset_hours))
    defaults.update(kwargs)
    return TradeRecord(**defaults)


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "hermes_memory.sqlite3")


# -- empty / incomplete history is handled gracefully ---------------------


def test_analyze_on_empty_store_returns_report_with_no_data(store: MemoryStore) -> None:
    report = analyze(store)

    assert isinstance(report, AnalysisReport)
    assert report.trade_count == 0
    assert report.win_rate is None
    assert report.average_pnl is None
    assert report.expectancy is None
    assert report.profit_factor is None
    assert report.max_drawdown is None
    assert report.average_holding_time_seconds is None
    assert report.total_fees == 0.0
    assert report.total_funding == 0.0
    assert report.fees_and_funding_drag_pct is None
    assert report.largest_losses == []
    assert report.max_consecutive_losses == 0
    assert report.by_entry_zscore == []
    assert report.by_exit_zscore == []
    assert report.by_regime == []
    assert report.findings == []


def test_trades_without_a_pnl_are_excluded_from_statistics() -> None:
    # An in-progress trade recorded at entry only (pnl not yet known).
    open_trade = _trade(pnl=None, entry_zscore=2.0)
    closed_trade = _trade(offset_hours=1, pnl=10.0)

    assert win_rate([open_trade, closed_trade]) == 1.0
    assert average_pnl([open_trade, closed_trade]) == 10.0


def test_all_incomplete_history_behaves_like_empty_history() -> None:
    trades = [_trade(pnl=None), _trade(offset_hours=1, pnl=None)]

    assert win_rate(trades) is None
    assert average_pnl(trades) is None
    assert expectancy(trades) is None
    assert profit_factor(trades) is None
    assert max_drawdown(trades) is None


# -- individual statistics -------------------------------------------------


def test_win_rate() -> None:
    trades = [_trade(pnl=10.0), _trade(pnl=-5.0), _trade(pnl=3.0), _trade(pnl=-1.0)]
    assert win_rate(trades) == 0.5


def test_average_pnl() -> None:
    trades = [_trade(pnl=10.0), _trade(pnl=-4.0)]
    assert average_pnl(trades) == 3.0


def test_expectancy_matches_win_loss_decomposition() -> None:
    # 2 wins of +10, 1 loss of -20: win_frac=2/3, avg_win=10, loss_frac=1/3, avg_loss=20
    trades = [_trade(pnl=10.0), _trade(pnl=10.0), _trade(pnl=-20.0)]
    expected = (2 / 3) * 10.0 - (1 / 3) * 20.0
    assert expectancy(trades) == pytest.approx(expected)
    assert expectancy(trades) == pytest.approx(average_pnl(trades))


def test_profit_factor_is_gross_profit_over_gross_loss() -> None:
    trades = [_trade(pnl=30.0), _trade(pnl=-10.0)]
    assert profit_factor(trades) == pytest.approx(3.0)


def test_profit_factor_is_infinite_with_no_losses() -> None:
    trades = [_trade(pnl=5.0), _trade(pnl=2.0)]
    assert profit_factor(trades) == float("inf")


def test_profit_factor_is_none_with_no_completed_trades() -> None:
    assert profit_factor([]) is None


def test_max_drawdown_on_a_known_equity_curve() -> None:
    # cumulative: 10, 25 (peak), 15 (dd=10), 5 (dd=20), 30 (new peak)
    trades = [
        _trade(offset_hours=0, pnl=10.0),
        _trade(offset_hours=1, pnl=15.0),
        _trade(offset_hours=2, pnl=-10.0),
        _trade(offset_hours=3, pnl=-10.0),
        _trade(offset_hours=4, pnl=25.0),
    ]
    assert max_drawdown(trades) == pytest.approx(20.0)


def test_max_drawdown_orders_by_recorded_at_not_insertion_order() -> None:
    later = _trade(offset_hours=5, pnl=-50.0)
    earlier = _trade(offset_hours=0, pnl=10.0)
    # Inserted out of chronological order.
    assert max_drawdown([later, earlier]) == pytest.approx(50.0)


def test_average_holding_time_ignores_trades_without_one() -> None:
    trades = [
        _trade(pnl=1.0, holding_time_seconds=100.0),
        _trade(pnl=1.0, holding_time_seconds=300.0),
        _trade(pnl=1.0, holding_time_seconds=None),
    ]
    assert average_holding_time_seconds(trades) == pytest.approx(200.0)


def test_average_holding_time_is_none_with_no_data() -> None:
    assert average_holding_time_seconds([_trade(pnl=1.0)]) is None


def test_fees_and_funding_drag() -> None:
    trades = [
        _trade(pnl=100.0, fees=2.0, funding=-1.0),
        _trade(pnl=50.0, fees=1.0, funding=0.5),
    ]
    total_fees, total_funding, drag_pct = fees_and_funding_drag_pct(trades)
    assert total_fees == pytest.approx(3.0)
    assert total_funding == pytest.approx(-0.5)
    # gross profit = 150, drag = (3.0 + 0.5) / 150 * 100
    assert drag_pct == pytest.approx((3.0 + 0.5) / 150 * 100)


def test_fees_and_funding_drag_pct_is_none_with_no_gross_profit() -> None:
    trades = [_trade(pnl=-10.0, fees=1.0)]
    _, _, drag_pct = fees_and_funding_drag_pct(trades)
    assert drag_pct is None


def test_largest_losses_sorted_most_negative_first() -> None:
    trades = [_trade(pnl=-5.0), _trade(pnl=-50.0), _trade(pnl=10.0), _trade(pnl=-1.0)]
    losses = largest_losses(trades, top_n=2)
    assert [t.pnl for t in losses] == [-50.0, -5.0]


def test_largest_losses_on_all_wins_is_empty() -> None:
    assert largest_losses([_trade(pnl=1.0), _trade(pnl=2.0)]) == []


def test_max_consecutive_losses() -> None:
    trades = [
        _trade(offset_hours=0, pnl=1.0),
        _trade(offset_hours=1, pnl=-1.0),
        _trade(offset_hours=2, pnl=-1.0),
        _trade(offset_hours=3, pnl=-1.0),
        _trade(offset_hours=4, pnl=1.0),
        _trade(offset_hours=5, pnl=-1.0),
    ]
    assert max_consecutive_losses(trades) == 3


def test_max_consecutive_losses_with_no_losses_is_zero() -> None:
    assert max_consecutive_losses([_trade(pnl=1.0), _trade(pnl=2.0)]) == 0


# -- grouping ---------------------------------------------------------


def test_performance_by_entry_zscore_buckets_and_excludes_missing() -> None:
    trades = [
        _trade(pnl=10.0, entry_zscore=2.1),
        _trade(pnl=-5.0, entry_zscore=2.3),
        _trade(pnl=1.0, entry_zscore=None),  # excluded: no z-score
    ]
    buckets = performance_by_entry_zscore(trades)
    assert len(buckets) == 1
    assert buckets[0].label == "2.0 to 2.5"
    assert buckets[0].trade_count == 2


def test_performance_by_exit_zscore_buckets() -> None:
    trades = [_trade(pnl=5.0, exit_zscore=0.2), _trade(pnl=-5.0, exit_zscore=0.4)]
    buckets = performance_by_exit_zscore(trades)
    assert len(buckets) == 1
    assert buckets[0].label == "0.0 to 0.5"


def test_performance_by_entry_zscore_handles_negative_values() -> None:
    trades = [_trade(pnl=5.0, entry_zscore=-2.3)]
    buckets = performance_by_entry_zscore(trades)
    assert buckets[0].label == "-2.5 to -2.0"


def test_performance_by_entry_zscore_negative_boundary_matches_positive_mirror() -> None:
    """A value exactly on a bucket boundary buckets the same way on both sides of zero.

    Regression test: the bucketing formula used to floor negative
    boundary values (e.g. -0.5) into the bucket *below* the boundary
    (-1.0 to -0.5) while a positive boundary value (e.g. 0.5) correctly
    bucketed into the bucket *starting at* the boundary (0.5 to 1.0) —
    an inconsistent, off-by-one-bucket asymmetry.
    """
    positive_buckets = performance_by_entry_zscore([_trade(pnl=5.0, entry_zscore=0.5)])
    negative_buckets = performance_by_entry_zscore([_trade(pnl=5.0, entry_zscore=-0.5)])
    assert positive_buckets[0].label == "0.5 to 1.0"
    assert negative_buckets[0].label == "-0.5 to 0.0"


def test_performance_by_regime_groups_by_label() -> None:
    trades = [
        _trade(pnl=10.0, regime="mean_reverting"),
        _trade(pnl=-10.0, regime="trending"),
        _trade(pnl=5.0, regime="mean_reverting"),
        _trade(pnl=1.0, regime=None),  # excluded
    ]
    buckets = performance_by_regime(trades)
    labels = {b.label: b for b in buckets}
    assert set(labels) == {"mean_reverting", "trending"}
    assert labels["mean_reverting"].trade_count == 2


# -- findings -----------------------------------------------------------


def test_findings_are_generated_for_an_underperforming_zscore_bucket() -> None:
    # Bucket 2.0-2.5: 3 trades, all losers -> clearly worse than overall.
    trades = [
        _trade(pnl=-10.0, entry_zscore=2.1),
        _trade(pnl=-8.0, entry_zscore=2.2),
        _trade(pnl=-6.0, entry_zscore=2.4),
        _trade(pnl=50.0, entry_zscore=3.5),
        _trade(pnl=40.0, entry_zscore=3.6),
        _trade(pnl=45.0, entry_zscore=3.7),
    ]
    report_trades_analysis = performance_by_entry_zscore(trades)
    assert any(b.label == "2.0 to 2.5" and b.expectancy < 0 for b in report_trades_analysis)


def test_analyze_end_to_end_produces_findings_text(store: MemoryStore) -> None:
    for i in range(3):
        store.record_trade(
            TradeRecord(
                pair="BTC/USDC:USDC",
                pnl=-10.0 - i,
                entry_zscore=2.0 + i * 0.05,
                recorded_at=_T0 + timedelta(hours=i),
            )
        )
    for i in range(3):
        store.record_trade(
            TradeRecord(
                pair="BTC/USDC:USDC",
                pnl=50.0 + i,
                entry_zscore=3.5 + i * 0.05,
                recorded_at=_T0 + timedelta(hours=10 + i),
            )
        )

    report = analyze(store)

    assert report.trade_count == 6
    assert report.findings, "expected at least one finding from the underperforming bucket"
    rendered = report.render_findings()
    assert "[OBSERVATION]" in rendered
    assert "[HYPOTHESIS]" in rendered


def test_bucket_below_minimum_trade_count_is_not_flagged(store: MemoryStore) -> None:
    # Only 2 trades in the losing bucket: below _MIN_TRADES_FOR_FINDING (3).
    store.record_trade(TradeRecord(pair="BTC/USDC:USDC", pnl=-10.0, entry_zscore=2.1))
    store.record_trade(TradeRecord(pair="BTC/USDC:USDC", pnl=-8.0, entry_zscore=2.2))
    for i in range(3):
        store.record_trade(
            TradeRecord(pair="BTC/USDC:USDC", pnl=50.0 + i, entry_zscore=3.5 + i * 0.05)
        )

    report = analyze(store)

    small_bucket = next(b for b in report.by_entry_zscore if b.label == "2.0 to 2.5")
    assert small_bucket.trade_count == 2
    assert not any("2.0 to 2.5" in f.observation for f in report.findings)


def test_consecutive_loss_streak_generates_a_finding(store: MemoryStore) -> None:
    for i in range(4):
        store.record_trade(
            TradeRecord(pair="BTC/USDC:USDC", pnl=-5.0, recorded_at=_T0 + timedelta(hours=i))
        )

    report = analyze(store)

    assert report.max_consecutive_losses == 4
    assert any("losing streak" in f.observation for f in report.findings)


def test_findings_render_format_matches_spec() -> None:
    from hermes.analyzer import Finding

    finding = Finding(
        observation="Trades entered between Z=2.0 and Z=2.5 had lower expectancy.",
        hypothesis="A stricter entry threshold may improve trade quality.",
    )
    rendered = finding.render()
    assert rendered == (
        "[OBSERVATION] Trades entered between Z=2.0 and Z=2.5 had lower expectancy.\n"
        "[HYPOTHESIS] A stricter entry threshold may improve trade quality."
    )


def test_finding_without_hypothesis_omits_that_line() -> None:
    from hermes.analyzer import Finding

    finding = Finding(observation="Just an observation.")
    assert finding.render() == "[OBSERVATION] Just an observation."


# -- never modifies anything ------------------------------------------


def test_analyze_does_not_write_to_the_store(store: MemoryStore) -> None:
    store.record_trade(TradeRecord(pair="BTC/USDC:USDC", pnl=10.0))
    before = store.get_trades()

    analyze(store)

    after = store.get_trades()
    assert before == after
