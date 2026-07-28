"""Grid comparison.

The partial measures only earn their place if each one isolates a different failure mode,
so most of these tests construct a grid pair that fails in exactly one way and check that
the corresponding measure -- and only it -- notices.
"""

from __future__ import annotations

import numpy as np
import pytest

from arc.metrics import comparison_rank, compare_grids, diff_mask


def grid(rows) -> np.ndarray:
    return np.asarray(rows, dtype=np.uint8)


# -- exact match ----------------------------------------------------------------


def test_identical_grids():
    c = compare_grids(grid([[1, 2], [3, 4]]), grid([[1, 2], [3, 4]]))
    assert c.exact_match is True
    assert c.shape_correct is True
    assert c.cell_accuracy == 1.0
    assert c.hamming_distance == 0
    assert c.cell_accuracy_padded == 1.0
    assert c.color_histogram_distance == 0.0
    assert c.foreground_iou == 1.0


def test_one_wrong_cell():
    c = compare_grids(grid([[1, 2], [3, 4]]), grid([[1, 2], [3, 5]]))
    assert c.exact_match is False
    assert c.shape_correct is True
    assert c.cell_accuracy == 0.75
    assert c.hamming_distance == 1


# -- shape handling -------------------------------------------------------------


def test_cell_accuracy_is_none_when_shapes_differ():
    """None rather than 0.0, so "wrong shape" cannot masquerade as "right shape, all
    cells wrong". Those are different failures and get separate signals."""
    c = compare_grids(grid([[1, 2], [3, 4]]), grid([[1, 2, 0], [3, 4, 0], [0, 0, 0]]))
    assert c.shape_correct is False
    assert c.cell_accuracy is None
    assert c.hamming_distance is None


def test_padded_accuracy_uses_the_union_rectangle():
    """2x2 prediction against a 3x3 target, agreeing on the whole 2x2 overlap: 4 agreeing
    cells out of the 9-cell union. Area the prediction is missing counts against it, so
    guessing a tiny grid buys nothing."""
    c = compare_grids(grid([[1, 2], [3, 4]]), grid([[1, 2, 0], [3, 4, 0], [0, 0, 0]]))
    assert c.cell_accuracy_padded == pytest.approx(4 / 9)


def test_padded_accuracy_equals_cell_accuracy_when_shapes_agree():
    c = compare_grids(grid([[1, 2], [3, 4]]), grid([[1, 9], [3, 9]]))
    assert c.cell_accuracy_padded == c.cell_accuracy


def test_shrinking_the_prediction_does_not_inflate_the_padded_score():
    big = grid([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    assert compare_grids(grid([[1]]), big).cell_accuracy_padded == pytest.approx(1 / 9)
    assert compare_grids(grid([[1, 1], [1, 1]]), big).cell_accuracy_padded == pytest.approx(4 / 9)


def test_shapes_are_reported_as_plain_tuples():
    """They end up in JSON, so numpy ints would not serialise."""
    c = compare_grids(grid([[1, 2]]), grid([[1], [2]]))
    assert c.predicted_shape == (1, 2)
    assert c.expected_shape == (2, 1)
    assert all(isinstance(x, int) for x in c.predicted_shape + c.expected_shape)


# -- colour histogram -----------------------------------------------------------


def test_histogram_distance_ignores_position():
    """Same colours in the same proportions, rearranged: this measure must not react."""
    c = compare_grids(grid([[1, 2], [3, 4]]), grid([[4, 3], [2, 1]]))
    assert c.color_histogram_distance == pytest.approx(0.0)
    assert c.exact_match is False


def test_histogram_distance_catches_wrong_colours():
    c = compare_grids(grid([[1, 1], [1, 1]]), grid([[2, 2], [2, 2]]))
    assert c.color_histogram_distance == pytest.approx(1.0)


def test_histogram_distance_is_bounded():
    for a, b in [
        (grid([[0]]), grid([[9] * 5] * 5)),
        (grid([[1, 2, 3]]), grid([[4, 5, 6], [7, 8, 9]])),
    ]:
        assert 0.0 <= compare_grids(a, b).color_histogram_distance <= 1.0


def test_histogram_distance_is_invariant_under_d4():
    """D4 permutes cells, so it preserves the colour histogram exactly. This is the
    internal consistency check behind identity and random-symmetry scoring the same
    histogram distance on the real data -- if it ever breaks, the measure is wrong."""
    a = grid([[1, 2, 3], [4, 5, 6]])
    target = grid([[7, 7, 7], [8, 8, 0]])
    reference = compare_grids(a, target).color_histogram_distance
    for transformed in (np.rot90(a, 2), np.fliplr(a), np.flipud(a), a.T, np.rot90(a, 1)):
        assert compare_grids(transformed, target).color_histogram_distance == pytest.approx(
            reference
        )


# -- foreground IoU -------------------------------------------------------------


def test_foreground_iou_catches_wrong_position():
    """Identical ink, moved. Histogram distance stays 0; IoU is what notices."""
    c = compare_grids(grid([[1, 0], [0, 0]]), grid([[0, 0], [0, 1]]))
    assert c.color_histogram_distance == pytest.approx(0.0)
    assert c.foreground_iou == 0.0


def test_foreground_iou_ignores_which_colour():
    c = compare_grids(grid([[1, 0], [0, 1]]), grid([[7, 0], [0, 7]]))
    assert c.foreground_iou == 1.0
    assert c.exact_match is False


def test_foreground_iou_partial_overlap():
    c = compare_grids(grid([[1, 1], [0, 0]]), grid([[1, 0], [1, 0]]))
    assert c.foreground_iou == pytest.approx(1 / 3)


def test_foreground_iou_of_two_blank_grids_is_one():
    assert compare_grids(grid([[0, 0]]), grid([[0, 0]])).foreground_iou == 1.0


def test_foreground_iou_treats_missing_area_as_background():
    """Padding with background, not with a sentinel: absent ink is not foreground."""
    c = compare_grids(grid([[1]]), grid([[1, 0], [0, 0]]))
    assert c.foreground_iou == 1.0


# -- ranking and diffing --------------------------------------------------------


def test_exact_match_outranks_everything():
    exact = compare_grids(grid([[1, 2], [3, 4]]), grid([[1, 2], [3, 4]]))
    near = compare_grids(grid([[1, 2], [3, 5]]), grid([[1, 2], [3, 4]]))
    assert comparison_rank(exact) > comparison_rank(near)


def test_correct_shape_outranks_more_matching_cells_at_the_wrong_shape():
    """Shape is trusted above raw cell agreement: getting the dimensions right is a real
    step, and a bigger wrong-shaped grid can otherwise accumulate more agreeing cells."""
    target = grid([[1, 1], [1, 1]])
    right_shape = compare_grids(grid([[1, 0], [0, 0]]), target)
    wrong_shape = compare_grids(grid([[1, 1, 1], [1, 1, 1]]), target)
    assert comparison_rank(right_shape) > comparison_rank(wrong_shape)


def test_diff_mask_covers_the_union_rectangle():
    mask = diff_mask(grid([[1, 2]]), grid([[1, 9], [0, 0]]))
    assert mask.shape == (2, 2)
    assert mask[0, 0] == False  # noqa: E712 - comparing numpy bools
    assert mask[0, 1] == True  # noqa: E712
    assert mask[1].all(), "area the prediction does not cover counts as different"


def test_diff_mask_is_empty_on_an_exact_match():
    assert not diff_mask(grid([[1, 2], [3, 4]]), grid([[1, 2], [3, 4]])).any()


def test_comparison_serialises_to_json():
    import json

    c = compare_grids(grid([[1, 2]]), grid([[1], [2]]))
    assert json.loads(json.dumps(c.as_dict()))["expected_shape"] == [2, 1]
