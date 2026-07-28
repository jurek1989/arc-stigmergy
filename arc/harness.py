"""The evaluation harness: plug in a solver, get a scored, saved run.

Scoring follows the official ARC rules exactly -- two attempts per test input, pixel-perfect
match, and a task counts as solved only if *every* one of its test inputs is hit. On top of
that the harness records the partial measures from :mod:`arc.metrics`, the wall-clock time,
and the number of search steps the solver reports, because the official 0/1 score is far
too coarse to steer development with.

Solver contract
---------------

A solver is any object with a ``name`` and a ``solve`` method::

    class MySolver:
        name = "my-solver"

        def solve(self, task: Task) -> list[list[Grid]]:
            # one entry per test input, each entry 1 or 2 candidate output grids
            return [[grid_a, grid_b] for _ in task.test]

Optionally it may expose an integer ``steps`` attribute, which the harness reads *after*
each ``solve`` call and records. Baselines do not have to implement it; search-based
solvers should set it to however many candidates they examined.

The harness never trusts a solver. Exceptions are caught and recorded as a failed task,
and malformed predictions are repaired (and flagged) rather than crashing a run that may
be a few hundred tasks in.
"""

from __future__ import annotations

import csv
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable

import numpy as np

from .metrics import GridComparison, compare_grids, comparison_rank
from .task import MAX_GRID_SIDE, N_COLORS, Grid, Task, load_split

MAX_ATTEMPTS = 2
"""Attempts allowed per test input by the ARC rules."""

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


@runtime_checkable
class Solver(Protocol):
    """Structural type of anything the harness can evaluate."""

    name: str

    def solve(self, task: Task) -> list[list[Grid]]:  # pragma: no cover - protocol
        ...


@dataclass
class TestOutcome:
    """Result for a single test input of a task."""

    test_index: int
    n_attempts: int
    exact: bool
    """True if *any* attempt matched exactly."""
    best: GridComparison
    """Metrics of the best attempt, ranked by :func:`arc.metrics.comparison_rank`."""

    def as_dict(self) -> dict:
        return {
            "test_index": self.test_index,
            "n_attempts": self.n_attempts,
            "exact": self.exact,
            "best": self.best.as_dict(),
        }


@dataclass
class TaskResult:
    """Result for one task."""

    task_id: str
    source: str
    solved: bool
    """The official 0/1 outcome: every test input hit within two attempts."""
    n_test: int
    outcomes: list[TestOutcome]
    seconds: float
    steps: int | None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def n_exact(self) -> int:
        return sum(o.exact for o in self.outcomes)

    def mean(self, attribute: str) -> float | None:
        """Mean of a partial measure across this task's test inputs, skipping ``None``."""
        values = [getattr(o.best, attribute) for o in self.outcomes]
        values = [v for v in values if v is not None]
        return float(statistics.fmean(values)) if values else None

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "source": self.source,
            "solved": self.solved,
            "n_test": self.n_test,
            "n_exact": self.n_exact,
            "seconds": self.seconds,
            "steps": self.steps,
            "error": self.error,
            "warnings": self.warnings,
            "outcomes": [o.as_dict() for o in self.outcomes],
        }


@dataclass
class RunResult:
    """A complete evaluation run: metadata, per-task results, aggregate summary."""

    solver_name: str
    dataset: str
    split: str
    started_at: str
    tasks: list[TaskResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- aggregates -------------------------------------------------------------

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    @property
    def score(self) -> float:
        """The official metric: fraction of tasks fully solved."""
        return statistics.fmean(t.solved for t in self.tasks) if self.tasks else 0.0

    def _mean_over_tasks(self, attribute: str) -> float | None:
        values = [t.mean(attribute) for t in self.tasks]
        values = [v for v in values if v is not None]
        return float(statistics.fmean(values)) if values else None

    def summary(self) -> dict[str, Any]:
        """Everything worth printing or diffing against another run."""
        seconds = [t.seconds for t in self.tasks] or [0.0]
        steps = [t.steps for t in self.tasks if t.steps is not None]
        n_test_inputs = sum(t.n_test for t in self.tasks)
        n_exact_inputs = sum(t.n_exact for t in self.tasks)
        defined_cell_accuracy = sum(
            1 for t in self.tasks if t.mean("cell_accuracy") is not None
        )
        return {
            "solver": self.solver_name,
            "dataset": self.dataset,
            "split": self.split,
            "n_tasks": self.n_tasks,
            "score": self.score,
            "n_solved": sum(t.solved for t in self.tasks),
            "test_input_exact_rate": (
                n_exact_inputs / n_test_inputs if n_test_inputs else 0.0
            ),
            "shape_accuracy": self._mean_over_tasks("shape_correct"),
            "cell_accuracy": self._mean_over_tasks("cell_accuracy"),
            "cell_accuracy_n_tasks": defined_cell_accuracy,
            "cell_accuracy_padded": self._mean_over_tasks("cell_accuracy_padded"),
            "foreground_iou": self._mean_over_tasks("foreground_iou"),
            "color_histogram_distance": self._mean_over_tasks("color_histogram_distance"),
            "seconds_total": sum(seconds),
            "seconds_mean": float(statistics.fmean(seconds)),
            "seconds_max": max(seconds),
            "steps_total": sum(steps) if steps else None,
            "steps_mean": float(statistics.fmean(steps)) if steps else None,
            "n_errors": sum(1 for t in self.tasks if t.error),
            "n_warnings": sum(len(t.warnings) for t in self.tasks),
        }

    # -- persistence ------------------------------------------------------------

    def save(self, directory: str | Path = RESULTS_DIR) -> tuple[Path, Path]:
        """Write ``<run>.json`` (everything) and ``<run>.csv`` (one row per task).

        The JSON is the archive; the CSV is what you load into Polars to compare runs.
        Returns both paths.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stem = (
            f"{self.started_at.replace(':', '').replace('-', '')}"
            f"__{_slug(self.solver_name)}__{_slug(self.dataset)}_{_slug(self.split)}"
        )
        json_path = directory / f"{stem}.json"
        csv_path = directory / f"{stem}.csv"

        json_path.write_text(
            json.dumps(
                {
                    "summary": self.summary(),
                    "metadata": self.metadata,
                    "started_at": self.started_at,
                    "tasks": [t.as_dict() for t in self.tasks],
                },
                indent=2,
            )
        )

        fields = [
            "task_id", "source", "solved", "n_test", "n_exact", "seconds", "steps",
            "shape_correct", "cell_accuracy", "cell_accuracy_padded",
            "foreground_iou", "color_histogram_distance", "error", "n_warnings",
        ]
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for task in self.tasks:
                writer.writerow(
                    {
                        "task_id": task.task_id,
                        "source": task.source,
                        "solved": int(task.solved),
                        "n_test": task.n_test,
                        "n_exact": task.n_exact,
                        "seconds": round(task.seconds, 6),
                        "steps": task.steps if task.steps is not None else "",
                        "shape_correct": _fmt(task.mean("shape_correct")),
                        "cell_accuracy": _fmt(task.mean("cell_accuracy")),
                        "cell_accuracy_padded": _fmt(task.mean("cell_accuracy_padded")),
                        "foreground_iou": _fmt(task.mean("foreground_iou")),
                        "color_histogram_distance": _fmt(
                            task.mean("color_histogram_distance")
                        ),
                        "error": task.error or "",
                        "n_warnings": len(task.warnings),
                    }
                )
        return json_path, csv_path

    def solved_task_ids(self) -> list[str]:
        return [t.task_id for t in self.tasks if t.solved]

    def failed_task_ids(self) -> list[str]:
        return [t.task_id for t in self.tasks if not t.solved]


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in text)


# -- prediction sanitising -----------------------------------------------------


_FALLBACK_GRID = np.zeros((1, 1), dtype=np.uint8)


def _coerce_grid(candidate: Any) -> tuple[Grid, str | None]:
    """Turn whatever the solver returned into a valid grid, reporting any repair."""
    try:
        grid = np.asarray(candidate)
    except Exception as exc:  # noqa: BLE001 - the solver may return anything at all
        return _FALLBACK_GRID, f"attempt not array-like ({exc!r})"
    if grid.ndim != 2 or grid.size == 0:
        return _FALLBACK_GRID, f"attempt has shape {grid.shape}, expected a non-empty 2D grid"
    if not np.issubdtype(grid.dtype, np.integer):
        if not np.all(np.equal(np.mod(grid, 1), 0)):
            return _FALLBACK_GRID, f"attempt has non-integer dtype {grid.dtype}"
        grid = grid.astype(np.int64)
    if grid.min() < 0 or grid.max() >= N_COLORS:
        return _FALLBACK_GRID, f"attempt has colours outside 0..{N_COLORS - 1}"
    if grid.shape[0] > MAX_GRID_SIDE or grid.shape[1] > MAX_GRID_SIDE:
        return _FALLBACK_GRID, f"attempt has shape {grid.shape}, larger than {MAX_GRID_SIDE}"
    return grid.astype(np.uint8), None


def _normalise_predictions(
    raw: Any, task: Task, warnings: list[str]
) -> list[list[Grid]]:
    """Coerce a solver's return value into exactly ``n_test`` lists of <= 2 valid grids."""
    if not isinstance(raw, (list, tuple)):
        warnings.append(f"solver returned {type(raw).__name__}, expected a list")
        raw = []
    raw = list(raw)

    if len(raw) > task.n_test:
        warnings.append(f"solver returned {len(raw)} prediction lists for {task.n_test} test inputs; extra ignored")
        raw = raw[: task.n_test]
    while len(raw) < task.n_test:
        warnings.append(f"solver returned no prediction for test input {len(raw)}")
        raw.append([])

    normalised: list[list[Grid]] = []
    for index, attempts in enumerate(raw):
        if isinstance(attempts, np.ndarray) and attempts.ndim == 2:
            # A bare grid where a list of attempts was expected -- an easy mistake, and
            # unambiguous, so accept it rather than scoring a zero.
            attempts = [attempts]
        if not isinstance(attempts, (list, tuple)):
            warnings.append(f"test {index}: expected a list of attempts, got {type(attempts).__name__}")
            attempts = []
        attempts = list(attempts)
        if len(attempts) > MAX_ATTEMPTS:
            warnings.append(f"test {index}: {len(attempts)} attempts given, only the first {MAX_ATTEMPTS} count")
            attempts = attempts[:MAX_ATTEMPTS]
        grids = []
        for attempt in attempts:
            grid, problem = _coerce_grid(attempt)
            if problem:
                warnings.append(f"test {index}: {problem}")
            grids.append(grid)
        if not grids:
            grids = [_FALLBACK_GRID]
            warnings.append(f"test {index}: no usable attempt, scored against a 1x1 blank")
        normalised.append(grids)
    return normalised


# -- evaluation ----------------------------------------------------------------


def evaluate_task(solver: Solver, task: Task) -> TaskResult:
    """Run one task through a solver and score it."""
    warnings: list[str] = []
    error: str | None = None

    start = time.perf_counter()
    try:
        raw = solver.solve(task)
    except Exception as exc:  # noqa: BLE001 - a broken solver must not abort the run
        raw = []
        error = f"{type(exc).__name__}: {exc}"
    seconds = time.perf_counter() - start

    steps = getattr(solver, "steps", None)
    steps = int(steps) if isinstance(steps, (int, np.integer)) else None

    predictions = _normalise_predictions(raw, task, warnings)

    outcomes: list[TestOutcome] = []
    for index, (pair, attempts) in enumerate(zip(task.test, predictions)):
        if pair.output is None:
            raise ValueError(
                f"{task.task_id} test[{index}] has no ground-truth output; "
                f"this harness scores offline and cannot evaluate held-out pairs"
            )
        comparisons = [compare_grids(a, pair.output) for a in attempts]
        best = max(comparisons, key=comparison_rank)
        outcomes.append(
            TestOutcome(
                test_index=index,
                n_attempts=len(attempts),
                exact=any(c.exact_match for c in comparisons),
                best=best,
            )
        )

    return TaskResult(
        task_id=task.task_id,
        source=task.source,
        solved=bool(outcomes) and all(o.exact for o in outcomes),
        n_test=task.n_test,
        outcomes=outcomes,
        seconds=seconds,
        steps=steps,
        error=error,
        warnings=warnings,
    )


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:  # noqa: BLE001 - git may be absent; this is metadata, not logic
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def evaluate(
    solver: Solver,
    tasks: Iterable[Task] | None = None,
    *,
    dataset: str = "arc-agi-2",
    split: str = "evaluation",
    drop_leaked: bool = False,
    limit: int | None = None,
    progress: bool = False,
    extra_metadata: dict[str, Any] | None = None,
) -> RunResult:
    """Evaluate a solver over a set of tasks.

    Pass ``tasks`` explicitly, or leave it ``None`` to load ``dataset``/``split``. With
    ``drop_leaked=True`` the six tasks shared between the two evaluation sets are excluded
    (see :data:`arc.task.LEAKED_TASK_IDS`).
    """
    if tasks is None:
        tasks = load_split(dataset, split, drop_leaked=drop_leaked)
    tasks = list(tasks)
    if limit is not None:
        tasks = tasks[:limit]

    started_at = datetime.now().isoformat(timespec="seconds")
    results: list[TaskResult] = []
    for n, task in enumerate(tasks, start=1):
        results.append(evaluate_task(solver, task))
        if progress and (n % 25 == 0 or n == len(tasks)):
            solved = sum(r.solved for r in results)
            print(f"  {n}/{len(tasks)} tasks, {solved} solved", file=sys.stderr)

    metadata = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "drop_leaked": drop_leaked,
        "limit": limit,
        **(extra_metadata or {}),
    }
    return RunResult(
        solver_name=solver.name,
        dataset=dataset,
        split=split,
        started_at=started_at,
        tasks=results,
        metadata=metadata,
    )


def print_summary(run: RunResult, *, file=sys.stdout) -> None:
    """Human-readable one-block summary of a run."""
    s = run.summary()
    print(f"{s['solver']}  on  {s['dataset']}/{s['split']}  ({s['n_tasks']} tasks)", file=file)
    print(f"  score (official)        {s['score']:.4f}   ({s['n_solved']}/{s['n_tasks']} tasks)", file=file)
    print(f"  test inputs exact       {s['test_input_exact_rate']:.4f}", file=file)
    print(f"  shape accuracy          {_show(s['shape_accuracy'])}", file=file)
    print(f"  cell accuracy           {_show(s['cell_accuracy'])}   (defined for {s['cell_accuracy_n_tasks']} tasks)", file=file)
    print(f"  cell accuracy (padded)  {_show(s['cell_accuracy_padded'])}", file=file)
    print(f"  foreground IoU          {_show(s['foreground_iou'])}", file=file)
    print(f"  colour hist. distance   {_show(s['color_histogram_distance'])}", file=file)
    print(f"  time                    {s['seconds_total']:.2f}s total, {s['seconds_mean'] * 1000:.2f}ms/task, max {s['seconds_max'] * 1000:.1f}ms", file=file)
    if s["steps_total"] is not None:
        print(f"  search steps            {s['steps_total']} total, {s['steps_mean']:.1f}/task", file=file)
    if s["n_errors"] or s["n_warnings"]:
        print(f"  !! errors {s['n_errors']}, warnings {s['n_warnings']}", file=file)


def _show(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"
