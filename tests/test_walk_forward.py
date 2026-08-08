"""Unit tests for optimize.walk_forward.

WalkForwardRunner tests use synthetic optimizer/evaluator callables
(no real backtest/hyperopt), consistent with the module's dependency-
injection design — this validates the orchestration logic itself
(window sequencing, aggregation), independent of whatever real
optimizer/backtester a production caller would inject.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

pytestmark = pytest.mark.unit

from optimize.walk_forward import (
    WalkForwardError,
    WalkForwardRunner,
    WalkForwardWindow,
    generate_windows,
)


# ---------------------------------------------------------------------------
# generate_windows
# ---------------------------------------------------------------------------


def test_generate_windows_basic_contiguous_coverage() -> None:
    windows = generate_windows(
        start=date(2024, 1, 1),
        end=date(2024, 4, 1),
        train_period=timedelta(days=30),
        test_period=timedelta(days=10),
    )

    assert len(windows) > 0
    for window in windows:
        assert window.train_end == window.test_start
        assert window.test_end - window.test_start == timedelta(days=10)
        assert window.train_end - window.train_start == timedelta(days=30)


def test_generate_windows_default_step_produces_contiguous_test_windows() -> None:
    windows = generate_windows(
        start=date(2024, 1, 1),
        end=date(2024, 3, 1),
        train_period=timedelta(days=20),
        test_period=timedelta(days=10),
    )
    # With step defaulting to test_period, consecutive windows' test
    # periods should be back-to-back.
    for a, b in zip(windows, windows[1:]):
        assert a.test_end == b.test_start


def test_generate_windows_custom_step_overlaps() -> None:
    windows = generate_windows(
        start=date(2024, 1, 1),
        end=date(2024, 3, 1),
        train_period=timedelta(days=20),
        test_period=timedelta(days=10),
        step=timedelta(days=5),
    )
    assert windows[1].train_start - windows[0].train_start == timedelta(days=5)


def test_generate_windows_too_short_range_produces_no_windows() -> None:
    windows = generate_windows(
        start=date(2024, 1, 1),
        end=date(2024, 1, 10),
        train_period=timedelta(days=30),
        test_period=timedelta(days=10),
    )
    assert windows == []


def test_generate_windows_rejects_end_before_start() -> None:
    with pytest.raises(WalkForwardError, match="end must be after start"):
        generate_windows(date(2024, 2, 1), date(2024, 1, 1), timedelta(days=1), timedelta(days=1))


def test_generate_windows_rejects_non_positive_train_period() -> None:
    with pytest.raises(WalkForwardError, match="train_period must be positive"):
        generate_windows(
            date(2024, 1, 1), date(2024, 2, 1), timedelta(days=0), timedelta(days=1)
        )


def test_generate_windows_rejects_non_positive_test_period() -> None:
    with pytest.raises(WalkForwardError, match="test_period must be positive"):
        generate_windows(
            date(2024, 1, 1), date(2024, 2, 1), timedelta(days=1), timedelta(days=0)
        )


def test_generate_windows_rejects_non_positive_step() -> None:
    with pytest.raises(WalkForwardError, match="step must be positive"):
        generate_windows(
            date(2024, 1, 1),
            date(2024, 2, 1),
            timedelta(days=1),
            timedelta(days=1),
            step=timedelta(0),
        )


def test_window_timerange_formatting() -> None:
    window = WalkForwardWindow(
        train_start=date(2024, 1, 1),
        train_end=date(2024, 1, 31),
        test_start=date(2024, 1, 31),
        test_end=date(2024, 2, 10),
    )
    assert window.train_timerange() == "20240101-20240131"
    assert window.test_timerange() == "20240131-20240210"


# ---------------------------------------------------------------------------
# WalkForwardRunner
# ---------------------------------------------------------------------------


def make_windows(n: int = 3) -> list[WalkForwardWindow]:
    return generate_windows(
        date(2024, 1, 1),
        date(2024, 1, 1) + timedelta(days=10 * (n + 3)),
        train_period=timedelta(days=20),
        test_period=timedelta(days=10),
    )[:n]


def test_runner_calls_optimizer_and_evaluator_for_each_window() -> None:
    windows = make_windows(3)
    optimizer_calls = []
    evaluator_calls = []

    def optimizer(window):
        optimizer_calls.append(window)
        return {"entry_zscore": 2.0}

    def evaluator(window, params):
        evaluator_calls.append((window, params))
        return 1.0

    report = WalkForwardRunner(optimizer, evaluator).run(windows)

    assert len(optimizer_calls) == 3
    assert len(evaluator_calls) == 3
    assert len(report.results) == 3


def test_runner_preserves_window_order() -> None:
    windows = make_windows(3)
    report = WalkForwardRunner(lambda w: {}, lambda w, p: 1.0).run(windows)
    assert [r.window for r in report.results] == windows


def test_runner_passes_optimizer_output_to_evaluator() -> None:
    windows = make_windows(2)

    def optimizer(window):
        return {"entry_zscore": 2.5, "window_start": window.train_start.isoformat()}

    received_params = []

    def evaluator(window, params):
        received_params.append(params)
        return 1.0

    WalkForwardRunner(optimizer, evaluator).run(windows)

    assert received_params[0]["entry_zscore"] == 2.5
    assert received_params[0]["window_start"] == windows[0].train_start.isoformat()


def test_runner_rejects_empty_window_list() -> None:
    with pytest.raises(WalkForwardError, match="must not be empty"):
        WalkForwardRunner(lambda w: {}, lambda w, p: 1.0).run([])


def test_report_mean_out_of_sample_score() -> None:
    windows = make_windows(4)
    scores = iter([1.0, -0.5, 2.0, 0.5])

    report = WalkForwardRunner(lambda w: {}, lambda w, p: next(scores)).run(windows)

    assert report.mean_out_of_sample_score == pytest.approx(0.75)


def test_report_worst_window() -> None:
    windows = make_windows(3)
    scores = iter([1.0, -3.0, 2.0])

    report = WalkForwardRunner(lambda w: {}, lambda w, p: next(scores)).run(windows)

    assert report.worst_window.out_of_sample_score == -3.0
    assert report.worst_window.window == windows[1]


def test_report_fraction_profitable_windows() -> None:
    windows = make_windows(4)
    scores = iter([1.0, -1.0, 2.0, -0.5])

    report = WalkForwardRunner(lambda w: {}, lambda w, p: next(scores)).run(windows)

    assert report.fraction_profitable_windows == pytest.approx(0.5)


def test_report_out_of_sample_scores_preserves_order() -> None:
    windows = make_windows(3)
    scores = iter([3.0, 1.0, 2.0])

    report = WalkForwardRunner(lambda w: {}, lambda w, p: next(scores)).run(windows)

    assert report.out_of_sample_scores == (3.0, 1.0, 2.0)
