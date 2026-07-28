"""Shared fixtures.

Most tests run on synthetic tasks built in memory. That is deliberate: the task data is
not committed, so anything that needs it has to skip cleanly rather than fail, and the
official-metric tests are too important to depend on a download.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arc.task import DATASETS, SPLITS, Task, data_root, parse_task  # noqa: E402


def make_task(train: list[tuple], test: list[tuple], task_id: str = "deadbeef") -> Task:
    """Build a task from lists of ``(input, output)`` pairs of plain nested lists."""
    return parse_task(
        {
            "train": [{"input": i, "output": o} for i, o in train],
            "test": [{"input": i, "output": o} for i, o in test],
        },
        task_id=task_id,
        source="synthetic",
    )


@pytest.fixture
def single_test_task() -> Task:
    """One test input. Output is the input with every colour incremented, so no trivial
    solver -- identity, symmetry or copying a train output -- can accidentally hit it."""
    return make_task(
        train=[([[1, 2], [3, 4]], [[2, 3], [4, 5]]), ([[5, 6], [7, 8]], [[6, 7], [8, 9]])],
        test=[([[1, 1], [2, 2]], [[2, 2], [3, 3]])],
        task_id="00000001",
    )


@pytest.fixture
def multi_test_task() -> Task:
    """Two test inputs. The official rule requires *both* to be right."""
    return make_task(
        train=[([[1, 2], [3, 4]], [[2, 3], [4, 5]])],
        test=[
            ([[1, 1], [2, 2]], [[2, 2], [3, 3]]),
            ([[4, 4], [5, 5]], [[5, 5], [6, 6]]),
        ],
        task_id="00000002",
    )


@pytest.fixture
def reshaping_task() -> Task:
    """Output has a different shape from the input."""
    return make_task(
        train=[([[1, 2], [3, 4]], [[1, 2, 1, 2], [3, 4, 3, 4]])],
        test=[([[5, 6], [7, 8]], [[5, 6, 5, 6], [7, 8, 7, 8]])],
        task_id="00000003",
    )


def _data_available() -> bool:
    root = data_root()
    return all((root / d / s).is_dir() for d in DATASETS for s in SPLITS)


requires_data = pytest.mark.skipif(
    not _data_available(),
    reason="task data not present; see data/README.md",
)
