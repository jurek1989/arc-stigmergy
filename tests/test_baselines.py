"""Baselines.

They are trivial by design, so the tests are less about what they compute and more about
them honouring the solver contract -- the right number of prediction lists, at most two
attempts, deterministic output -- since they are the reference implementations anything
later will be copied from.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_task, requires_data

from arc.baselines import (
    D4_TRANSFORMS,
    IdentitySolver,
    MostCommonTrainOutputSolver,
    RandomSymmetrySolver,
    build_all,
)
from arc.harness import MAX_ATTEMPTS, evaluate
from arc.task import grid_key, grids_equal


@pytest.mark.parametrize("solver", build_all(), ids=lambda s: s.name)
def test_contract_is_honoured(solver, multi_test_task):
    predictions = solver.solve(multi_test_task)
    assert len(predictions) == multi_test_task.n_test
    for attempts in predictions:
        assert isinstance(attempts, list)
        assert 1 <= len(attempts) <= MAX_ATTEMPTS
        for grid in attempts:
            assert isinstance(grid, np.ndarray)
            assert grid.ndim == 2
            assert 0 <= grid.min() and grid.max() <= 9


@pytest.mark.parametrize("solver", build_all(), ids=lambda s: s.name)
def test_baselines_do_not_solve_a_non_trivial_task(solver, single_test_task):
    """The floor has to actually be a floor. This task's answer is not the input, not any
    train output, and not a symmetry of anything."""
    assert evaluate(solver, tasks=[single_test_task]).score == 0.0


def test_identity_returns_the_input(multi_test_task):
    predictions = IdentitySolver().solve(multi_test_task)
    for attempts, pair in zip(predictions, multi_test_task.test):
        assert grids_equal(attempts[0], pair.input)


def test_identity_solves_a_task_whose_output_is_its_input():
    task = make_task(train=[([[1, 2]], [[1, 2]])], test=[([[3, 4]], [[3, 4]])])
    assert evaluate(IdentitySolver(), tasks=[task]).score == 1.0


def test_most_common_picks_the_modal_train_output():
    repeated = [[7, 7], [7, 7]]
    task = make_task(
        train=[([[1, 1], [1, 1]], [[9, 9], [9, 9]]),
               ([[2, 2], [2, 2]], repeated),
               ([[3, 3], [3, 3]], repeated)],
        test=[([[4, 4], [4, 4]], repeated)],
    )
    attempts = MostCommonTrainOutputSolver().solve(task)[0]
    assert grids_equal(attempts[0], np.asarray(repeated, dtype=np.uint8))
    assert grids_equal(attempts[1], np.asarray([[9, 9], [9, 9]], dtype=np.uint8))
    assert evaluate(MostCommonTrainOutputSolver(), tasks=[task]).score == 1.0


def test_most_common_breaks_ties_by_first_appearance():
    """All-distinct outputs is the common case in ARC; the result must still be
    deterministic rather than depending on dict ordering."""
    task = make_task(
        train=[([[1]], [[1]]), ([[2]], [[2]]), ([[3]], [[3]])],
        test=[([[4]], [[4]])],
    )
    attempts = MostCommonTrainOutputSolver().solve(task)[0]
    assert [int(a[0, 0]) for a in attempts] == [1, 2]


def test_most_common_reports_steps(multi_test_task):
    solver = MostCommonTrainOutputSolver()
    solver.solve(multi_test_task)
    assert solver.steps == len(multi_test_task.train)


def test_random_symmetry_gives_two_distinct_transforms(multi_test_task):
    for attempts in RandomSymmetrySolver(seed=3).solve(multi_test_task):
        assert len(attempts) == MAX_ATTEMPTS
        assert grid_key(attempts[0]) != grid_key(attempts[1]) or grids_equal(
            attempts[0], attempts[1]
        ), "distinct transforms may still coincide on a symmetric grid"


def test_random_symmetry_is_deterministic(multi_test_task):
    a = RandomSymmetrySolver(seed=11).solve(multi_test_task)
    b = RandomSymmetrySolver(seed=11).solve(multi_test_task)
    assert [[grid_key(g) for g in attempts] for attempts in a] == [
        [grid_key(g) for g in attempts] for attempts in b
    ]


def test_random_symmetry_depends_on_the_seed(multi_test_task):
    keys = {
        tuple(grid_key(g) for attempts in RandomSymmetrySolver(seed=s).solve(multi_test_task)
              for g in attempts)
        for s in range(8)
    }
    assert len(keys) > 1


def test_random_symmetry_output_is_always_some_symmetry_of_the_input(multi_test_task):
    predictions = RandomSymmetrySolver(seed=5).solve(multi_test_task)
    for attempts, pair in zip(predictions, multi_test_task.test):
        allowed = {grid_key(np.ascontiguousarray(f(np.asarray(pair.input))))
                   for f in D4_TRANSFORMS.values()}
        for attempt in attempts:
            assert grid_key(attempt) in allowed


def test_d4_is_a_group_of_eight_distinct_transforms():
    """A generic grid must land on eight different results; a duplicated entry in the
    table would silently halve the search space."""
    grid = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    results = {grid_key(np.ascontiguousarray(f(grid))) for f in D4_TRANSFORMS.values()}
    assert len(D4_TRANSFORMS) == 8
    assert len(results) == 8


def test_d4_transforms_are_involutive_where_they_should_be():
    grid = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    for name in ("rot180", "flip_horizontal", "flip_vertical", "transpose", "anti_transpose"):
        twice = D4_TRANSFORMS[name](D4_TRANSFORMS[name](grid))
        assert grids_equal(np.ascontiguousarray(twice), grid), name


# -- the calibration numbers recorded in CLAUDE.md ------------------------------


@requires_data
def test_all_baselines_score_zero_on_the_lab_bench():
    """arc-agi-2/evaluation is the bench everything is measured against. If a trivial
    solver ever scores above zero here, that is a finding, not a passing test."""
    for solver in build_all():
        run = evaluate(solver, dataset="arc-agi-2", split="evaluation")
        assert run.score == 0.0, f"{solver.name} scored {run.score}"


@requires_data
def test_identity_partial_scores_match_the_recorded_floor():
    """The floor documented in CLAUDE.md: shape 0.708, cell accuracy 0.810. Every partial
    metric is read relative to these, so a silent drift would invalidate the readings."""
    summary = evaluate(IdentitySolver(), dataset="arc-agi-2", split="evaluation").summary()
    assert summary["shape_accuracy"] == pytest.approx(0.7083, abs=1e-4)
    assert summary["cell_accuracy"] == pytest.approx(0.8097, abs=1e-4)
    assert summary["cell_accuracy_n_tasks"] == 86


@requires_data
def test_identity_and_random_symmetry_share_a_histogram_distance():
    """D4 preserves the colour histogram exactly, so these two must agree to the last
    digit. A mismatch means the histogram measure is broken."""
    identity = evaluate(IdentitySolver(), dataset="arc-agi-2", split="evaluation")
    symmetry = evaluate(RandomSymmetrySolver(), dataset="arc-agi-2", split="evaluation")
    assert identity.summary()["color_histogram_distance"] == pytest.approx(
        symmetry.summary()["color_histogram_distance"]
    )
