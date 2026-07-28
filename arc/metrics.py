"""Comparing a predicted grid against the expected one.

The official ARC metric is binary: pixel-perfect or nothing. That is the right metric for
a leaderboard and a terrible one for diagnosis -- it cannot distinguish "produced noise"
from "got everything except three cells". Everything here exists to recover that lost
signal.

The partial measures are deliberately kept **separate** rather than blended into one
number, because they fail in different ways and mixing them hides which failure happened:

* ``shape_correct`` -- did the solver even get the output dimensions right? About half of
  ARC tasks change the grid size, so this is a real hurdle on its own.
* ``cell_accuracy`` -- fraction of matching cells, defined **only** when the shapes agree.
  ``None`` otherwise, rather than 0.0, so that "wrong shape" never masquerades as "right
  shape, all cells wrong".
* ``cell_accuracy_padded`` -- a soft fallback that is defined for any pair of shapes.
* ``color_histogram_distance`` -- catches "right shape, wrong colours".
* ``foreground_iou`` -- catches "right colours, wrong position".

All are O(number of cells).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .task import BACKGROUND_COLOR, N_COLORS, Grid


@dataclass(frozen=True)
class GridComparison:
    """How close one predicted grid is to the expected one."""

    exact_match: bool
    """The official criterion: identical shape and identical cells."""

    shape_correct: bool
    predicted_shape: tuple[int, int]
    expected_shape: tuple[int, int]

    cell_accuracy: float | None
    """Fraction of matching cells, in [0, 1]. ``None`` when the shapes differ."""

    hamming_distance: int | None
    """Number of differing cells. ``None`` when the shapes differ."""

    cell_accuracy_padded: float
    """Shape-tolerant variant of ``cell_accuracy``, in [0, 1].

    Both grids are anchored at the top-left corner. A cell counts as correct only if it
    exists in *both* grids and the values agree; the denominator is the area of the
    smallest rectangle containing both. Area present in one grid but not the other is
    therefore counted as wrong, so there is no free credit for guessing a tiny grid.
    Equals ``cell_accuracy`` whenever the shapes agree.
    """

    color_histogram_distance: float
    """Total variation distance between the two colour distributions, in [0, 1].

    0.0 means the two grids use the ten colours in exactly the same proportions, which
    says nothing about where. Shape-independent by construction, so it stays meaningful
    when the dimensions are wrong.
    """

    foreground_iou: float
    """Intersection-over-union of the non-background masks, in [0, 1].

    Computed on the union rectangle, treating colour 0 as background. Measures whether the
    ink landed in the right places, ignoring which colour it is. 1.0 when both grids are
    entirely background. The complementary distance is ``1 - foreground_iou``.
    """

    def as_dict(self) -> dict:
        return asdict(self)


def _union_shape(a: Grid, b: Grid) -> tuple[int, int]:
    return max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])


def _intersection_shape(a: Grid, b: Grid) -> tuple[int, int]:
    return min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])


def _color_histogram(grid: Grid) -> np.ndarray:
    counts = np.bincount(np.asarray(grid).ravel(), minlength=N_COLORS).astype(np.float64)
    return counts / counts.sum()


def _pad_to(grid: Grid, shape: tuple[int, int], fill: int) -> np.ndarray:
    padded = np.full(shape, fill, dtype=np.int16)
    padded[: grid.shape[0], : grid.shape[1]] = grid
    return padded


def compare_grids(predicted: Grid, expected: Grid) -> GridComparison:
    """Compare a prediction against ground truth. See :class:`GridComparison`."""
    predicted = np.asarray(predicted)
    expected = np.asarray(expected)

    shape_correct = predicted.shape == expected.shape
    exact_match = shape_correct and bool(np.array_equal(predicted, expected))

    if shape_correct:
        matching = int(np.count_nonzero(predicted == expected))
        cell_accuracy: float | None = matching / predicted.size
        hamming: int | None = predicted.size - matching
    else:
        cell_accuracy = None
        hamming = None

    ih, iw = _intersection_shape(predicted, expected)
    uh, uw = _union_shape(predicted, expected)
    agreeing = int(np.count_nonzero(predicted[:ih, :iw] == expected[:ih, :iw]))
    cell_accuracy_padded = agreeing / (uh * uw)

    hist_distance = 0.5 * float(
        np.abs(_color_histogram(predicted) - _color_histogram(expected)).sum()
    )

    # Padding with the background colour makes area outside a grid count as background,
    # which is exactly right: absent ink is not foreground.
    pred_fg = _pad_to(predicted, (uh, uw), fill=BACKGROUND_COLOR) != BACKGROUND_COLOR
    exp_fg = _pad_to(expected, (uh, uw), fill=BACKGROUND_COLOR) != BACKGROUND_COLOR
    union = int(np.count_nonzero(pred_fg | exp_fg))
    foreground_iou = 1.0 if union == 0 else int(np.count_nonzero(pred_fg & exp_fg)) / union

    return GridComparison(
        exact_match=exact_match,
        shape_correct=shape_correct,
        predicted_shape=tuple(int(x) for x in predicted.shape),
        expected_shape=tuple(int(x) for x in expected.shape),
        cell_accuracy=cell_accuracy,
        hamming_distance=hamming,
        cell_accuracy_padded=cell_accuracy_padded,
        color_histogram_distance=hist_distance,
        foreground_iou=foreground_iou,
    )


def comparison_rank(comparison: GridComparison) -> tuple:
    """Sort key for picking the best of several attempts. Larger is better.

    Ordered by how much each signal is trusted: an exact match beats everything, then a
    correct shape, then how many cells agree.
    """
    return (
        comparison.exact_match,
        comparison.shape_correct,
        comparison.cell_accuracy_padded,
        comparison.foreground_iou,
        -comparison.color_histogram_distance,
    )


def diff_mask(predicted: Grid, expected: Grid) -> np.ndarray:
    """Boolean mask over the union rectangle marking cells that disagree.

    Cells present in only one of the two grids count as disagreeing. Used by the
    visualisation to highlight where a prediction went wrong.
    """
    uh, uw = _union_shape(predicted, expected)
    ih, iw = _intersection_shape(predicted, expected)
    mask = np.ones((uh, uw), dtype=bool)
    mask[:ih, :iw] = predicted[:ih, :iw] != expected[:ih, :iw]
    return mask
