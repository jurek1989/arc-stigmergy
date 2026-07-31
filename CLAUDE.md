# ARC-AGI — project notes

Research exploration around the ARC-AGI benchmark. Not a product, not a competition entry.
The long-run question is whether a population of simple search agents sharing a single
external library of primitives can grow that library **only through compression**, and
whether that produces something resembling stigmergy in program space.

`ARC-KONTEKST.md` holds the founding discussion in full (in Polish). Read it before making
design decisions; this file only covers how the code is organised.

## Where the project is

**Stage 0: the measuring apparatus.** Complete. Data, loader, visualisation, evaluation
harness, trivial baselines.

The mechanism does not exist yet and is not to be built ahead of schedule. The stated
project risk is having five variants in a week and no way to tell which one works, so
measurement was built first, on purpose.

## Layout

```
arc/
  task.py       data model and loaders          (Task, Pair, Grid, load_split)
  metrics.py    grid comparison                 (compare_grids -> GridComparison)
  viz.py        matplotlib rendering            (plot_task, plot_tasks, plot_comparison)
  harness.py    evaluation harness              (evaluate -> RunResult)
  baselines.py  trivial solvers                 (identity, most-common-output, random D4)
scripts/
  run_baselines.py
tests/          pytest; the data-dependent ones skip when data/ is empty
data/           task JSON, downloaded, gitignored — see data/README.md
results/        run output, gitignored
```

## Running things

```bash
python scripts/run_baselines.py                       # all baselines on arc-agi-2/evaluation
python scripts/run_baselines.py --limit 20 --no-save  # smoke test
python -m arc.viz --task 0934a4d8 --out task.png
python -m arc.viz --split evaluation --limit 24 --out sheet.png
python -m pytest                                      # 101 tests, ~11s
python -m pytest tests/test_harness.py                # no data needed, instant
```

Everything must stay fast enough to run interactively. Loading the 120-task evaluation set
and running all three baselines over it takes about 3 seconds end to end. If a change makes
the loop take minutes, the change is wrong.

## The solver contract

```python
class MySolver:
    name = "my-solver"
    steps = 0                      # optional; read by the harness after each solve

    def solve(self, task: Task) -> list[list[Grid]]:
        ...                        # one entry per test input, each 1 or 2 candidate grids
```

`list[list[Grid]]` rather than `list[Grid]` because tasks can have several test inputs
(up to 4 in `arc-agi-2/training`), and ARC allows two attempts *per test input* while
requiring all of them to be right for the task to count. The nested shape also leaves room
for a future solver to amortise search across a task's test inputs.

The harness never trusts a solver: exceptions are caught and recorded, malformed
predictions are repaired and flagged. A broken solver produces a bad score, not a dead run.

## Compute

**Compute means: the number of candidate programs evaluated against a task's train pairs.**

This is the single definition for the whole project. It is the same quantity the harness
records per task as `TaskResult.steps`, and the same quantity `arc.dynamics` accumulates as
`EpochSnapshot.compute`. Wall-clock time is recorded too but is not the axis anything is
plotted against — it varies with the machine, and the question here is how much *search* a
result cost, not how fast the laptop was that evening.

Every module that touches the quantity points back here rather than restating it.

## Conventions

* **Grids are read-only `numpy.uint8` arrays.** Tasks are cached per process, so mutation
  would silently poison later runs; read-only arrays turn that into an exception at the
  point of the mistake.
* **`ndarray` is not hashable and `==` on it is element-wise.** Use `grid_key(grid)` for
  dict keys and sets, `grids_equal(a, b)` for equality. This matters more than it looks:
  the eventual library/MDL machinery needs hashable grid identity everywhere.
* `Task` and `Pair` are frozen dataclasses with `eq=False`, because a generated `__eq__`
  over ndarray fields raises.
* Colour 0 is background. This is a convention in ARC, not a guarantee, and
  `foreground_iou` depends on it.
* Explicit over clever. Names carry the meaning; comments explain *why*, not *what*.
* Code, comments, docstrings and commit messages in English.

## Metrics

The official score is binary and that is kept intact (`RunResult.score`). Everything else
exists because a binary signal is useless for diagnosis. The partial measures are reported
**separately, never blended**, because they fail differently:

| measure                    | catches                                             |
| -------------------------- | --------------------------------------------------- |
| `shape_correct`            | output dimensions — roughly half of ARC changes them |
| `cell_accuracy`            | cell agreement; `None` when shapes differ, not 0.0   |
| `cell_accuracy_padded`     | shape-tolerant fallback over the union rectangle     |
| `color_histogram_distance` | right shape, wrong colours                           |
| `foreground_iou`           | right colours, wrong position                        |

Baseline floors on `arc-agi-2/evaluation`, worth remembering before reading any partial
score as evidence of understanding: the identity solver scores **0.000** officially but
gets `shape_correct` **0.708** and `cell_accuracy` **0.810**. Most of a typical ARC output
is already sitting in its input. A partial metric only means something relative to these.

Note also that `random-symmetry` and `identity` have *identical* `color_histogram_distance`
(0.2184) — D4 permutes cells and preserves the colour histogram exactly. If that ever stops
holding, the metric is broken.

## Tests

The point of the suite is the official metric. Nothing else in the project can be checked
against an external reference, so if `evaluate` computed the score subtly wrongly — counted
a third attempt, or let a task pass with one of two test inputs right — every experiment
afterwards would be invalid and nothing would look amiss. Those rules are pinned on
synthetic tasks built in memory, so they hold even with `data/` empty.

The second half is the harness surviving broken solvers: exceptions, `None`, wrong list
lengths, out-of-range colours, oversized grids, in-place mutation of a cached task. Search
solvers fail in creative ways and a run must degrade to a bad score, not die at task 300.

The calibration floors quoted below are asserted in `tests/test_baselines.py`. They are
the reference every partial metric is read against, so they are not allowed to drift
silently — if a test fails there, update the numbers here in the same commit.

## Data

Two datasets, downloaded from public GitHub repositories, never committed. Provenance,
licence and re-fetch instructions are in `data/README.md`.

**ARC-AGI-1 is not an independent control set.** 767 of the 1000 `arc-agi-2/training`
tasks are ARC-AGI-1 tasks with their pairs reordered, and six tasks appear in *both*
evaluation sets. Pass `drop_leaked=True` to `load_split` / `evaluate` when that matters.
Details and the six ids in `data/README.md`.

## What this project does not do

* **No training on synthetic data.** Generating ARC-like tasks and fine-tuning on them
  works and is explicitly rejected here — it contradicts the point of the benchmark and is
  not what this exploration is about.
* **No external model APIs.** No network calls of any kind from project code. Data is
  fetched once, by hand, and that is the only time anything leaves the machine.
* **No heavy ML frameworks.** numpy, matplotlib, standard library. Nothing here trains a
  network.
* **No registration anywhere.** No Kaggle account, no ARC Prize API key. Everything runs
  offline against the public data.
* **No building ahead.** Do not add machinery for a stage that has not started. If a design
  choice is made to accommodate a future stage, say so explicitly and justify it.

## Working style

Ask before implementing anything with more than one sensible reading. Say so directly if a
decision looks wrong or if something is being designed for a problem that will not exist.
Summarise after each finished item, not after the whole stage. Do not add features that
were not asked for.
