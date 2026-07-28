"""ARC-AGI research scaffolding: data model, visualisation, evaluation harness.

Stage 0 of the project -- the measuring apparatus, built before any mechanism. See
``CLAUDE.md`` in the repository root for what this project is and is not.
"""

from .harness import (
    MAX_ATTEMPTS,
    RunResult,
    Solver,
    TaskResult,
    TestOutcome,
    evaluate,
    evaluate_task,
    print_summary,
)
from .metrics import GridComparison, compare_grids, diff_mask
from .task import (
    DATASETS,
    LEAKED_TASK_IDS,
    SPLITS,
    Grid,
    Pair,
    Task,
    find_task,
    grid_key,
    grids_equal,
    load_all,
    load_split,
    load_task,
)

__all__ = [
    "DATASETS",
    "SPLITS",
    "LEAKED_TASK_IDS",
    "MAX_ATTEMPTS",
    "Grid",
    "GridComparison",
    "Pair",
    "RunResult",
    "Solver",
    "Task",
    "TaskResult",
    "TestOutcome",
    "compare_grids",
    "diff_mask",
    "evaluate",
    "evaluate_task",
    "find_task",
    "grid_key",
    "grids_equal",
    "load_all",
    "load_split",
    "load_task",
    "print_summary",
]
