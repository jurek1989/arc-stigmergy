"""Loading and representing ARC tasks.

An ARC task is a handful of demonstration pairs (input grid -> output grid) plus one or
more test inputs whose outputs the solver has to produce. Grids are 2D arrays of integers
0..9, at most 30x30.

Design decisions worth knowing:

* Grids are ``numpy.ndarray`` of dtype ``uint8``, and they are made **read-only**. Tasks
  are cached per process, so a solver that mutated a grid in place would silently corrupt
  every later run. Read-only arrays turn that into an immediate exception.
* ``ndarray`` is not hashable and ``==`` on it is element-wise, which makes it unusable as
  a dict key. Use :func:`grid_key` wherever a hashable identity is needed (memoisation,
  deduplicating candidate programs, keying a primitive library).
* ``Pair`` and ``Task`` are frozen dataclasses with ``eq=False``. The generated ``__eq__``
  would compare ndarray fields with ``==`` and raise "truth value of an array is
  ambiguous". Compare tasks by ``task_id``, grids by :func:`grids_equal`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np

Grid = np.ndarray
"""A 2D, read-only ``uint8`` array with values in 0..9 and shape within 1..30 x 1..30."""

MAX_GRID_SIDE = 30
N_COLORS = 10
BACKGROUND_COLOR = 0

DATASETS = ("arc-agi-1", "arc-agi-2")
SPLITS = ("training", "evaluation")

LEAKED_TASK_IDS = frozenset(
    {"0934a4d8", "136b0064", "16b78196", "981571dc", "aa4ec2a5", "da515329"}
)
"""Tasks present in both ``arc-agi-1/evaluation`` and ``arc-agi-2/evaluation``.

They are the same tasks, only with their pairs reordered. Anything tuned on the ARC-AGI-1
evaluation set leaks into the ARC-AGI-2 public evaluation score through these six.
See ``data/README.md``.
"""


def data_root() -> Path:
    """Directory holding ``arc-agi-1/`` and ``arc-agi-2/``.

    Overridable with the ``ARC_DATA_ROOT`` environment variable; otherwise it is the
    ``data/`` directory next to the package.
    """
    env = os.environ.get("ARC_DATA_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "data"


def _as_grid(raw: list[list[int]], where: str) -> Grid:
    try:
        grid = np.asarray(raw, dtype=np.uint8)
    except ValueError as exc:
        # Ragged rows land here, and numpy's own message ("inhomogeneous shape after 1
        # dimensions") says nothing about which file is malformed.
        raise ValueError(f"{where}: not a rectangular 2D grid ({exc})") from exc
    if grid.ndim != 2 or grid.size == 0:
        raise ValueError(f"{where}: expected a non-empty 2D grid, got shape {grid.shape}")
    if not (1 <= grid.shape[0] <= MAX_GRID_SIDE and 1 <= grid.shape[1] <= MAX_GRID_SIDE):
        raise ValueError(f"{where}: shape {grid.shape} outside 1..{MAX_GRID_SIDE}")
    if grid.max(initial=0) >= N_COLORS:
        raise ValueError(f"{where}: colour {int(grid.max())} outside 0..{N_COLORS - 1}")
    grid.flags.writeable = False
    return grid


def grid_key(grid: Grid) -> bytes:
    """A hashable identity for a grid, usable as a dict key or set member."""
    return bytes(grid.shape) + grid.tobytes()


def grids_equal(a: Grid, b: Grid) -> bool:
    """Exact equality: same shape and same cell values."""
    return a.shape == b.shape and bool(np.array_equal(a, b))


@dataclass(frozen=True, eq=False)
class Pair:
    """One demonstration or test pair.

    ``output`` is ``None`` only for held-out test pairs. Every pair in the public datasets
    shipped in ``data/`` has its output, which is what lets the harness score offline.
    """

    input: Grid
    output: Grid | None = None


@dataclass(frozen=True, eq=False)
class Task:
    """A complete ARC task."""

    task_id: str
    train: tuple[Pair, ...]
    test: tuple[Pair, ...]
    source: str = ""
    """Where it came from, e.g. ``"arc-agi-2/evaluation"``. Informational only."""

    @property
    def n_test(self) -> int:
        return len(self.test)

    def __repr__(self) -> str:
        return (
            f"Task({self.task_id!r}, train={len(self.train)}, test={len(self.test)},"
            f" source={self.source!r})"
        )


def parse_task(payload: dict, task_id: str, source: str = "") -> Task:
    """Build a :class:`Task` from the raw JSON structure."""
    missing = {"train", "test"} - payload.keys()
    if missing:
        raise ValueError(f"{task_id}: missing key(s) {sorted(missing)}")

    def pairs(split: str) -> tuple[Pair, ...]:
        out = []
        for i, raw in enumerate(payload[split]):
            where = f"{task_id} {split}[{i}]"
            if "input" not in raw:
                raise ValueError(f"{where}: missing 'input'")
            output = raw.get("output")
            out.append(
                Pair(
                    input=_as_grid(raw["input"], f"{where}.input"),
                    output=_as_grid(output, f"{where}.output") if output is not None else None,
                )
            )
        return tuple(out)

    return Task(task_id=task_id, train=pairs("train"), test=pairs("test"), source=source)


def load_task(path: str | Path, source: str = "") -> Task:
    """Load a single task from a ``<task_id>.json`` file.

    Note that 11 files in ARC-AGI-1 carry an extra top-level ``"name"`` key; it is ignored
    rather than rejected.
    """
    path = Path(path)
    return parse_task(json.loads(path.read_text()), task_id=path.stem, source=source)


@lru_cache(maxsize=None)
def _load_split_cached(dataset: str, split: str) -> tuple[Task, ...]:
    directory = data_root() / dataset / split
    if not directory.is_dir():
        raise FileNotFoundError(
            f"{directory} does not exist. The task data is not committed; "
            f"see data/README.md for how to fetch it."
        )
    source = f"{dataset}/{split}"
    return tuple(load_task(p, source=source) for p in sorted(directory.glob("*.json")))


def load_split(dataset: str, split: str, *, drop_leaked: bool = False) -> list[Task]:
    """Load every task in ``<dataset>/<split>``, sorted by task id.

    Results are cached per process, so repeated calls are free. Tasks are immutable, so
    the cache cannot be corrupted by a solver.

    With ``drop_leaked=True`` the six tasks shared between ``arc-agi-1/evaluation`` and
    ``arc-agi-2/evaluation`` are excluded (see :data:`LEAKED_TASK_IDS`). This only has an
    effect on those two splits.
    """
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}, expected one of {DATASETS}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
    tasks = list(_load_split_cached(dataset, split))
    if drop_leaked:
        tasks = [t for t in tasks if t.task_id not in LEAKED_TASK_IDS]
    return tasks


def load_all() -> dict[str, list[Task]]:
    """Every split, keyed by ``"<dataset>/<split>"``."""
    return {f"{d}/{s}": load_split(d, s) for d in DATASETS for s in SPLITS}


def find_task(task_id: str, *, dataset: str | None = None, split: str | None = None) -> Task:
    """Look a task up by id, optionally restricted to one dataset or split.

    Most ids are ambiguous across the two datasets -- 767 of the 1000 ARC-AGI-2 training
    tasks also live somewhere in ARC-AGI-1 -- so a bare id usually needs a ``dataset`` to
    resolve. Within a single dataset ids are unique, since neither dataset shares ids
    between its own training and evaluation splits.

    Raises ``KeyError`` if nothing matches or if the id is still ambiguous.
    """
    # Look the file up by name rather than loading whole splits: this is what the
    # visualisation CLI calls, and loading 1120 tasks to render one of them is the
    # difference between an instant look and a ten-second wait.
    hits = []
    for d in DATASETS:
        for s in SPLITS:
            if (dataset is not None and d != dataset) or (split is not None and s != split):
                continue
            path = data_root() / d / s / f"{task_id}.json"
            if path.is_file():
                hits.append(load_task(path, source=f"{d}/{s}"))
    if not hits:
        where = f" in {dataset or 'any dataset'}/{split or 'any split'}"
        raise KeyError(f"no task with id {task_id!r}{where}")
    if len(hits) > 1:
        raise KeyError(
            f"task id {task_id!r} occurs in {len(hits)} splits "
            f"({', '.join(t.source for t in hits)}); pass dataset= or split= to disambiguate"
        )
    return hits[0]


def iter_pairs(task: Task) -> Iterator[Pair]:
    """All pairs of a task, train first then test."""
    yield from task.train
    yield from task.test
