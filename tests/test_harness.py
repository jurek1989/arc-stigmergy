"""The harness is the foundation everything later stands on, so it gets the most scrutiny.

Two things are being pinned down here. First, that the official ARC metric is computed
*exactly* right -- two attempts per test input, pixel-perfect, all test inputs required.
Getting this subtly wrong would quietly invalidate every experiment that follows, and the
error would be invisible because there is nothing to compare against.

Second, that a misbehaving solver degrades to a bad score rather than killing a run that
may be several hundred tasks in. Search-based solvers fail in creative ways; the harness
has to survive all of them.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from conftest import make_task

from arc.harness import MAX_ATTEMPTS, evaluate, evaluate_task
from arc.task import Task


# -- solvers used by the tests --------------------------------------------------


class Oracle:
    """Returns the ground truth. Must score exactly 1.0 or the harness is broken."""

    name = "oracle"
    steps = 7

    def solve(self, task: Task) -> list[list[np.ndarray]]:
        return [[np.asarray(pair.output)] for pair in task.test]


class Constant:
    """Always returns the same grid, whatever the task."""

    name = "constant"

    def __init__(self, grid) -> None:
        self.grid = np.asarray(grid, dtype=np.uint8)

    def solve(self, task: Task) -> list[list[np.ndarray]]:
        return [[self.grid] for _ in task.test]


# -- the official metric --------------------------------------------------------


def test_oracle_scores_one(single_test_task, multi_test_task, reshaping_task):
    run = evaluate(Oracle(), tasks=[single_test_task, multi_test_task, reshaping_task])
    assert run.score == 1.0
    assert run.summary()["n_solved"] == 3
    assert run.summary()["test_input_exact_rate"] == 1.0
    assert run.summary()["cell_accuracy"] == 1.0
    assert run.summary()["foreground_iou"] == 1.0


def test_wrong_answer_scores_zero(single_test_task):
    result = evaluate_task(Constant([[0, 0], [0, 0]]), single_test_task)
    assert result.solved is False
    assert result.n_exact == 0


def test_all_test_inputs_must_be_correct(multi_test_task):
    """Hitting one of two test inputs is worth nothing under the official rule."""

    class HalfRight:
        name = "half-right"

        def solve(self, task):
            return [
                [np.asarray(task.test[0].output)],
                [np.zeros((2, 2), dtype=np.uint8)],
            ]

    result = evaluate_task(HalfRight(), multi_test_task)
    assert result.n_test == 2
    assert result.n_exact == 1
    assert result.solved is False
    assert result.outcomes[0].exact is True
    assert result.outcomes[1].exact is False


def test_second_attempt_counts(single_test_task):
    """Two attempts are allowed and either one hitting is a solve."""

    class WrongThenRight:
        name = "wrong-then-right"

        def solve(self, task):
            return [
                [np.zeros((2, 2), dtype=np.uint8), np.asarray(pair.output)]
                for pair in task.test
            ]

    result = evaluate_task(WrongThenRight(), single_test_task)
    assert result.solved is True
    assert result.outcomes[0].n_attempts == MAX_ATTEMPTS


def test_third_attempt_does_not_count(single_test_task):
    """More than two attempts is against the rules; the extras must be discarded, not
    quietly used. A correct third attempt still scores zero."""

    class ThreeAttempts:
        name = "three-attempts"

        def solve(self, task):
            return [
                [
                    np.zeros((2, 2), dtype=np.uint8),
                    np.ones((2, 2), dtype=np.uint8),
                    np.asarray(pair.output),  # correct, but too late
                ]
                for pair in task.test
            ]

    result = evaluate_task(ThreeAttempts(), single_test_task)
    assert result.solved is False
    assert result.outcomes[0].n_attempts == MAX_ATTEMPTS
    assert any("only the first" in w for w in result.warnings)


def test_best_attempt_is_reported_for_partial_metrics(single_test_task):
    """Partial measures come from the better of the two attempts, not the first or last."""

    class GoodThenGarbage:
        name = "good-then-garbage"

        def solve(self, task):
            near_miss = np.asarray(task.test[0].output).copy()
            near_miss[0, 0] = (near_miss[0, 0] + 1) % 10
            return [[near_miss, np.zeros((1, 1), dtype=np.uint8)]]

    result = evaluate_task(GoodThenGarbage(), single_test_task)
    assert result.solved is False
    assert result.outcomes[0].best.shape_correct is True
    assert result.outcomes[0].best.cell_accuracy == pytest.approx(0.75)


# -- surviving broken solvers ---------------------------------------------------


def test_exception_is_recorded_not_raised(single_test_task):
    class Crashes:
        name = "crashes"

        def solve(self, task):
            raise RuntimeError("boom")

    result = evaluate_task(Crashes(), single_test_task)
    assert result.solved is False
    assert result.error is not None
    assert "RuntimeError" in result.error and "boom" in result.error


def test_run_continues_past_a_crash(single_test_task, multi_test_task):
    """One exploding task must not take the other 119 with it."""

    class CrashesOnFirst:
        name = "crashes-on-first"

        def solve(self, task):
            if task.task_id == "00000001":
                raise ValueError("nope")
            return [[np.asarray(pair.output)] for pair in task.test]

    run = evaluate(CrashesOnFirst(), tasks=[single_test_task, multi_test_task])
    assert run.n_tasks == 2
    assert run.summary()["n_errors"] == 1
    assert run.solved_task_ids() == ["00000002"]


@pytest.mark.parametrize(
    "returned",
    [
        None,
        "not a list",
        [],
        [[]],
        [["not a grid"]],
        [[np.full((2, 2), 99)]],
        [[np.zeros((40, 40))]],
        [[np.zeros(())]],
        [[np.array([1.5, 2.5])]],
    ],
    ids=[
        "none", "string", "empty", "empty-attempts", "string-attempt",
        "colour-out-of-range", "oversized", "zero-dim", "non-integer",
    ],
)
def test_malformed_predictions_are_repaired(single_test_task, returned):
    class Malformed:
        name = "malformed"

        def solve(self, task):
            return returned

    result = evaluate_task(Malformed(), single_test_task)
    assert result.solved is False
    assert result.error is None, "malformed output is a warning, not an exception"
    assert result.warnings, "the repair must be visible, not silent"
    assert len(result.outcomes) == single_test_task.n_test


def test_missing_prediction_for_one_test_input(multi_test_task):
    class TooShort:
        name = "too-short"

        def solve(self, task):
            return [[np.asarray(task.test[0].output)]]

    result = evaluate_task(TooShort(), multi_test_task)
    assert len(result.outcomes) == 2
    assert result.outcomes[0].exact is True
    assert result.outcomes[1].exact is False
    assert result.solved is False


def test_extra_predictions_are_ignored(single_test_task):
    class TooLong:
        name = "too-long"

        def solve(self, task):
            return [[np.asarray(task.test[0].output)], [np.zeros((2, 2), dtype=np.uint8)]]

    result = evaluate_task(TooLong(), single_test_task)
    assert len(result.outcomes) == 1
    assert result.solved is True
    assert any("extra ignored" in w for w in result.warnings)


def test_bare_grid_instead_of_attempt_list_is_accepted(single_test_task):
    """An easy mistake, and unambiguous, so it is accepted rather than scored as zero."""

    class BareGrid:
        name = "bare-grid"

        def solve(self, task):
            return [np.asarray(pair.output) for pair in task.test]

    result = evaluate_task(BareGrid(), single_test_task)
    assert result.solved is True
    assert result.warnings == []


def test_mutating_a_task_grid_is_blocked(single_test_task):
    """Tasks are cached per process, so in-place mutation would poison later runs. It has
    to fail loudly at the point of the mistake."""

    class Mutates:
        name = "mutates"

        def solve(self, task):
            task.train[0].input[0, 0] = 5
            return [[np.asarray(pair.output)] for pair in task.test]

    result = evaluate_task(Mutates(), single_test_task)
    assert result.error is not None
    assert "read-only" in result.error
    assert single_test_task.train[0].input[0, 0] == 1


# -- bookkeeping ----------------------------------------------------------------


def test_steps_are_read_from_the_solver(single_test_task):
    assert evaluate_task(Oracle(), single_test_task).steps == 7


def test_steps_are_none_when_the_solver_does_not_report_them(single_test_task):
    assert evaluate_task(Constant([[0]]), single_test_task).steps is None
    run = evaluate(Constant([[0]]), tasks=[single_test_task])
    assert run.summary()["steps_total"] is None


def test_timing_is_recorded(single_test_task):
    assert evaluate_task(Oracle(), single_test_task).seconds >= 0.0


def test_summary_counts_add_up(single_test_task, multi_test_task):
    run = evaluate(Oracle(), tasks=[single_test_task, multi_test_task])
    summary = run.summary()
    assert summary["n_tasks"] == 2
    assert summary["score"] == summary["n_solved"] / summary["n_tasks"]
    assert run.failed_task_ids() == []


def test_cell_accuracy_is_undefined_not_zero_when_shapes_are_wrong(reshaping_task):
    """A solver that gets every shape wrong must report `None`, not a misleading 0.0."""
    run = evaluate(Constant([[0]]), tasks=[reshaping_task])
    summary = run.summary()
    assert summary["cell_accuracy"] is None
    assert summary["cell_accuracy_n_tasks"] == 0
    assert summary["cell_accuracy_padded"] is not None


def test_run_saves_json_and_csv(tmp_path, single_test_task, multi_test_task):
    run = evaluate(Oracle(), tasks=[single_test_task, multi_test_task])
    json_path, csv_path = run.save(tmp_path)

    payload = json.loads(json_path.read_text())
    assert payload["summary"]["score"] == 1.0
    assert len(payload["tasks"]) == 2
    assert payload["tasks"][0]["outcomes"][0]["best"]["exact_match"] is True
    assert "python" in payload["metadata"] and "numpy" in payload["metadata"]

    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 2
    assert {r["task_id"] for r in rows} == {"00000001", "00000002"}
    assert all(r["solved"] == "1" for r in rows)
    assert all(r["cell_accuracy"] == "1.000000" for r in rows)


def test_evaluate_respects_limit(single_test_task, multi_test_task, reshaping_task):
    run = evaluate(
        Oracle(), tasks=[single_test_task, multi_test_task, reshaping_task], limit=2
    )
    assert run.n_tasks == 2


def test_task_without_ground_truth_is_rejected():
    """The harness scores offline. A held-out pair it cannot score must be an error, not
    a silently miscounted zero."""
    task = make_task(train=[([[1]], [[2]])], test=[([[1]], [[2]])])
    stripped = Task(
        task_id=task.task_id,
        train=task.train,
        test=(type(task.test[0])(input=task.test[0].input, output=None),),
        source=task.source,
    )
    with pytest.raises(ValueError, match="no ground-truth output"):
        evaluate_task(Oracle(), stripped)
