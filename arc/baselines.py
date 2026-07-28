"""Trivial solvers, so the harness has something to measure.

None of these look at the task in any meaningful way. They exist to calibrate the
instrument: they establish the floor of every metric, they prove the harness handles
multiple test inputs and two attempts correctly, and they are what any real solver has to
beat before it is worth talking about.

Expect scores at or very near zero on the official metric. That is the point -- if a
baseline scores well on some measure, that measure is measuring the wrong thing.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .harness import MAX_ATTEMPTS
from .task import Grid, Task, grid_key

D4_TRANSFORMS = {
    "identity": lambda g: g,
    "rot90": lambda g: np.rot90(g, 1),
    "rot180": lambda g: np.rot90(g, 2),
    "rot270": lambda g: np.rot90(g, 3),
    "flip_horizontal": lambda g: np.fliplr(g),
    "flip_vertical": lambda g: np.flipud(g),
    "transpose": lambda g: g.T,
    "anti_transpose": lambda g: np.rot90(g, 2).T,
}
"""The eight symmetries of the square, the dihedral group D4."""


class IdentitySolver:
    """Return the test input unchanged.

    The absolute floor. Its only non-trivial score is on tasks whose output happens to
    equal its input, which in ARC essentially never happens -- but its *partial* scores
    are informative: they tell you how much of a typical output is already present in the
    input, which is the baseline any "edit the input" solver has to beat.
    """

    name = "identity"

    def solve(self, task: Task) -> list[list[Grid]]:
        return [[np.asarray(pair.input)] for pair in task.test]


class MostCommonTrainOutputSolver:
    """Ignore the input; return the output grid seen most often among the train pairs.

    The second attempt is the runner-up. On the many ARC tasks where every train output is
    distinct this degenerates to "the first train output, then the second", which is
    exactly as arbitrary as it sounds.

    Reports ``steps`` (the number of train outputs it tallied) purely to exercise the
    harness's step-counting path.
    """

    name = "most-common-train-output"

    def __init__(self) -> None:
        self.steps = 0

    def solve(self, task: Task) -> list[list[Grid]]:
        outputs = [np.asarray(pair.output) for pair in task.train if pair.output is not None]
        self.steps = len(outputs)
        if not outputs:
            return [[np.asarray(pair.input)] for pair in task.test]

        counts: Counter[bytes] = Counter(grid_key(g) for g in outputs)
        first_seen: dict[bytes, Grid] = {}
        for grid in outputs:
            first_seen.setdefault(grid_key(grid), grid)

        # Ties broken by first appearance among the train pairs, so the result is
        # deterministic and does not depend on dict ordering.
        order = sorted(
            counts,
            key=lambda key: (-counts[key], list(first_seen).index(key)),
        )
        attempts = [first_seen[key] for key in order[:MAX_ATTEMPTS]]
        return [list(attempts) for _ in task.test]


class RandomSymmetrySolver:
    """Apply a randomly chosen element of D4 to the test input.

    Two attempts means two *distinct* symmetries, drawn without replacement from all eight
    (including the identity). Seeded, so a run is reproducible; the seed is mixed with the
    task id so that different tasks get different draws rather than the same one.
    """

    name = "random-symmetry"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._names = list(D4_TRANSFORMS)

    def solve(self, task: Task) -> list[list[Grid]]:
        # Deriving the per-task seed from the task id keeps each task independent of the
        # order in which the harness happens to visit tasks. ARC ids are 8 hex digits, but
        # do not rely on it.
        try:
            task_number = int(task.task_id, 16)
        except ValueError:
            task_number = int.from_bytes(task.task_id.encode(), "little")
        rng = np.random.default_rng((self.seed * 1_000_003 + task_number) % (2**32))

        predictions = []
        for pair in task.test:
            chosen = rng.choice(len(self._names), size=MAX_ATTEMPTS, replace=False)
            predictions.append(
                [
                    np.ascontiguousarray(D4_TRANSFORMS[self._names[i]](np.asarray(pair.input)))
                    for i in chosen
                ]
            )
        return predictions


ALL_BASELINES = (IdentitySolver, MostCommonTrainOutputSolver, RandomSymmetrySolver)


def build_all() -> list:
    """Fresh instances of every baseline, in a stable order."""
    return [factory() for factory in ALL_BASELINES]
