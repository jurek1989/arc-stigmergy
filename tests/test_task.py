"""Data model and loading.

The invariants here are the ones the later stages will lean on hardest: grids that cannot
be mutated, and a grid identity that can be used as a dict key.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_task, requires_data

from arc.task import (
    DATASETS,
    LEAKED_TASK_IDS,
    SPLITS,
    find_task,
    grid_key,
    grids_equal,
    iter_pairs,
    load_split,
    load_task,
    parse_task,
)


# -- immutability ---------------------------------------------------------------


def test_grids_are_read_only(single_test_task):
    with pytest.raises(ValueError, match="read-only"):
        single_test_task.train[0].input[0, 0] = 9


def test_grids_are_uint8(single_test_task):
    for pair in iter_pairs(single_test_task):
        assert pair.input.dtype == np.uint8
        assert pair.output.dtype == np.uint8


def test_tasks_are_frozen(single_test_task):
    with pytest.raises(Exception):
        single_test_task.task_id = "changed"


# -- grid identity --------------------------------------------------------------


def test_grid_key_is_hashable_and_matches_equal_grids():
    a = np.asarray([[1, 2], [3, 4]], dtype=np.uint8)
    b = np.asarray([[1, 2], [3, 4]], dtype=np.uint8)
    assert grid_key(a) == grid_key(b)
    assert len({grid_key(a), grid_key(b)}) == 1


def test_grid_key_distinguishes_shape_from_content():
    """A 1x4 and a 4x1 grid hold the same bytes. Without the shape prefix they would
    collide, which would silently merge distinct entries in a primitive library."""
    row = np.asarray([[1, 2, 3, 4]], dtype=np.uint8)
    column = np.asarray([[1], [2], [3], [4]], dtype=np.uint8)
    assert row.tobytes() == column.tobytes()
    assert grid_key(row) != grid_key(column)


def test_grids_equal():
    a = np.asarray([[1, 2]], dtype=np.uint8)
    assert grids_equal(a, np.asarray([[1, 2]], dtype=np.uint8))
    assert not grids_equal(a, np.asarray([[1, 3]], dtype=np.uint8))
    assert not grids_equal(a, np.asarray([[1], [2]], dtype=np.uint8))


# -- parsing and validation -----------------------------------------------------


def test_extra_name_key_is_tolerated():
    """11 ARC-AGI-1 files carry a top-level "name"; it is ignored, not rejected."""
    task = parse_task(
        {"name": "whatever", "train": [{"input": [[1]], "output": [[2]]}],
         "test": [{"input": [[1]], "output": [[2]]}]},
        task_id="abcd1234",
    )
    assert task.task_id == "abcd1234"


@pytest.mark.parametrize(
    "payload, match",
    [
        ({"train": []}, "missing key"),
        ({"train": [{"input": [[1]], "output": [[10]]}], "test": []}, "colour"),
        ({"train": [{"input": [[1, 2], [3]], "output": [[1]]}], "test": []}, "not a rectangular"),
        ({"train": [{"input": [], "output": [[1]]}], "test": []}, "2D grid"),
        ({"train": [{"input": [[1] * 31], "output": [[1]]}], "test": []}, "outside 1..30"),
        ({"train": [{"output": [[1]]}], "test": []}, "missing 'input'"),
    ],
    ids=["missing-key", "bad-colour", "ragged", "empty", "too-wide", "no-input"],
)
def test_invalid_payloads_are_rejected(payload, match):
    with pytest.raises(ValueError, match=match):
        parse_task(payload, task_id="abcd1234")


def test_missing_output_is_allowed():
    """Held-out test pairs have no output. Parsing must accept that; the harness is what
    refuses to score it."""
    task = parse_task(
        {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[1]]}]},
        task_id="abcd1234",
    )
    assert task.test[0].output is None


def test_iter_pairs_yields_train_then_test(multi_test_task):
    pairs = list(iter_pairs(multi_test_task))
    assert len(pairs) == len(multi_test_task.train) + len(multi_test_task.test)
    assert pairs[0] is multi_test_task.train[0]
    assert pairs[-1] is multi_test_task.test[-1]


def test_load_task_reads_a_file(tmp_path):
    import json

    path = tmp_path / "cafe1234.json"
    path.write_text(json.dumps({"train": [{"input": [[1]], "output": [[2]]}],
                                "test": [{"input": [[3]], "output": [[4]]}]}))
    task = load_task(path, source="synthetic")
    assert task.task_id == "cafe1234"
    assert task.source == "synthetic"
    assert task.n_test == 1


def test_unknown_dataset_or_split_is_rejected():
    with pytest.raises(ValueError, match="unknown dataset"):
        load_split("arc-agi-9", "training")
    with pytest.raises(ValueError, match="unknown split"):
        load_split("arc-agi-2", "validation")


def test_task_repr_is_informative(single_test_task):
    assert "00000001" in repr(single_test_task)
    assert "train=2" in repr(single_test_task)


def test_make_task_helper_produces_distinct_objects():
    a = make_task([([[1]], [[2]])], [([[1]], [[2]])])
    b = make_task([([[1]], [[2]])], [([[1]], [[2]])])
    assert a is not b
    assert grids_equal(a.train[0].input, b.train[0].input)


# -- the real datasets ----------------------------------------------------------


@requires_data
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("split", SPLITS)
def test_every_split_loads(dataset, split):
    tasks = load_split(dataset, split)
    assert tasks
    assert len({t.task_id for t in tasks}) == len(tasks)
    assert all(t.source == f"{dataset}/{split}" for t in tasks)
    assert all(t.train and t.test for t in tasks)


@requires_data
def test_expected_task_counts():
    counts = {
        ("arc-agi-1", "training"): 400,
        ("arc-agi-1", "evaluation"): 400,
        ("arc-agi-2", "training"): 1000,
        ("arc-agi-2", "evaluation"): 120,
    }
    for (dataset, split), expected in counts.items():
        assert len(load_split(dataset, split)) == expected


@requires_data
def test_public_splits_have_test_outputs():
    """Everything shipped in data/ is scorable offline; nothing is held out."""
    for dataset in DATASETS:
        for split in SPLITS:
            for task in load_split(dataset, split):
                assert all(p.output is not None for p in task.test), task.task_id


@requires_data
def test_leaked_ids_really_are_in_both_evaluation_sets():
    """The six ids in LEAKED_TASK_IDS are load-bearing -- drop_leaked depends on them, and
    a stale list would silently stop excluding anything."""
    one = {t.task_id for t in load_split("arc-agi-1", "evaluation")}
    two = {t.task_id for t in load_split("arc-agi-2", "evaluation")}
    assert one & two == set(LEAKED_TASK_IDS)


@requires_data
def test_drop_leaked_removes_exactly_those_tasks():
    full = load_split("arc-agi-2", "evaluation")
    trimmed = load_split("arc-agi-2", "evaluation", drop_leaked=True)
    assert len(full) - len(trimmed) == len(LEAKED_TASK_IDS)
    assert not {t.task_id for t in trimmed} & LEAKED_TASK_IDS


@requires_data
def test_loading_is_cached_and_returns_the_same_objects():
    assert load_split("arc-agi-2", "evaluation")[0] is load_split("arc-agi-2", "evaluation")[0]


@requires_data
def test_find_task_needs_disambiguation_for_shared_ids():
    """Most ids exist in both datasets, so a bare lookup has to say so rather than pick."""
    with pytest.raises(KeyError, match="disambiguate"):
        find_task("00576224")
    assert find_task("00576224", dataset="arc-agi-2").source == "arc-agi-2/training"
    assert find_task("00576224", dataset="arc-agi-1").source == "arc-agi-1/evaluation"


@requires_data
def test_find_task_reports_a_missing_id():
    with pytest.raises(KeyError, match="no task with id"):
        find_task("ffffffff")


@requires_data
def test_find_task_agrees_with_load_split():
    task = load_split("arc-agi-2", "evaluation")[0]
    found = find_task(task.task_id, dataset="arc-agi-2", split="evaluation")
    assert found.task_id == task.task_id
    assert grids_equal(found.train[0].input, task.train[0].input)
