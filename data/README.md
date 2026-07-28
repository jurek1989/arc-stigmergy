# ARC task data

The task files themselves are **not committed** to this repository (see the top-level
`.gitignore`). This file records where they come from and how to get them back.

## Provenance

| Local path        | Source repository                                              | Branch   | Commit fetched                             | Fetched on |
| ----------------- | -------------------------------------------------------------- | -------- | ------------------------------------------ | ---------- |
| `data/arc-agi-1/` | [`fchollet/ARC-AGI`](https://github.com/fchollet/ARC-AGI)       | `master` | `399030444e0ab0cc8b4e199870fb20b863846f34` | 2026-07-28 |
| `data/arc-agi-2/` | [`arcprize/ARC-AGI-2`](https://github.com/arcprize/ARC-AGI-2)   | `main`   | `f3283f727488ad98fe575ea6a5ac981e4a188e49` | 2026-07-28 |

Only the `data/` subtree and the `LICENSE` of each repository were copied. No git history,
no notebooks, no apps.

## Layout

```
data/
├── arc-agi-1/
│   ├── LICENSE
│   ├── training/     400 tasks
│   └── evaluation/   400 tasks
└── arc-agi-2/
    ├── LICENSE
    ├── training/    1000 tasks
    └── evaluation/   120 tasks   (public eval set)
```

The private ARC-AGI-2 evaluation set (used for the competition leaderboard) is not public
and is not present here.

## File format

One JSON file per task, named `<task_id>.json` where `task_id` is an 8-character hex
string. Structure:

```json
{
  "train": [{"input": [[...]], "output": [[...]]}, ...],
  "test":  [{"input": [[...]], "output": [[...]]}, ...]
}
```

Grids are lists of rows of integers `0..9` (0 = background/black). Grid dimensions are
between 1×1 and 30×30. In these public sets, `test` pairs include their `output`.

Verified over all 1920 task files:

* every grid is rectangular, within 1..30 in both dimensions, all cells in `0..9`;
* all ten colours occur in every split;
* train pairs per task: 2–10; test pairs per task: 1–4.

One quirk: 11 files in ARC-AGI-1 carry an extra top-level `"name"` key
(9 in `evaluation`, 2 in `training`). It is ignorable, but a strict schema check will
trip on it.

## Overlap between the two datasets — read this before using ARC-AGI-1 as a control set

ARC-AGI-2's training set is largely a re-packaging of ARC-AGI-1. Measured by task id, with
task content compared as a multiset of `(input, output)` pairs:

| Comparison                              | Shared task ids | Content identical (up to pair ordering) |
| --------------------------------------- | --------------- | --------------------------------------- |
| arc-agi-1/training ∩ arc-agi-2/training | 391 of 400      | 391                                      |
| arc-agi-1/evaluation ∩ arc-agi-2/training | 376 of 400    | 375 (1 differs in some pairs)           |
| **arc-agi-1/evaluation ∩ arc-agi-2/evaluation** | **6**   | **6**                                    |
| arc-agi-1/training ∩ arc-agi-1/evaluation | 0             | —                                        |
| arc-agi-2/training ∩ arc-agi-2/evaluation | 0             | —                                        |

Consequences:

1. Only **233** of the 1000 ARC-AGI-2 training tasks are genuinely new relative to
   ARC-AGI-1. The rest are the same tasks, with `train`/`test` pairs reordered or
   re-split.
2. ARC-AGI-1 is **not** an independent control set with respect to ARC-AGI-2 training —
   it is ~96% subsumed by it.
3. These six task ids appear in **both** ARC-AGI-1 evaluation and ARC-AGI-2 public
   evaluation, and are the same tasks:
   `0934a4d8`, `136b0064`, `16b78196`, `981571dc`, `aa4ec2a5`, `da515329`.
   If ARC-AGI-1 evaluation is used for tuning, these six leak into the ARC-AGI-2 public
   eval score. Six of 120 is 5 percentage points — not negligible.

## Licence

Both source repositories are licensed under the **Apache License 2.0**; each repository's
`LICENSE` file is copied alongside its data. The datasets are © their respective authors
(François Chollet / the ARC Prize Foundation). This project uses them for research only
and redistributes nothing.

## Re-fetching

```bash
curl -sSL -o /tmp/arc2.tar.gz https://github.com/arcprize/ARC-AGI-2/archive/refs/heads/main.tar.gz
curl -sSL -o /tmp/arc1.tar.gz https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.tar.gz

mkdir -p data/arc-agi-1 data/arc-agi-2
tar -xzf /tmp/arc2.tar.gz -C /tmp ARC-AGI-2-main/data ARC-AGI-2-main/LICENSE
cp -r /tmp/ARC-AGI-2-main/data/training /tmp/ARC-AGI-2-main/data/evaluation data/arc-agi-2/
cp /tmp/ARC-AGI-2-main/LICENSE data/arc-agi-2/LICENSE

tar -xzf /tmp/arc1.tar.gz -C /tmp ARC-AGI-master/data ARC-AGI-master/LICENSE
cp -r /tmp/ARC-AGI-master/data/training /tmp/ARC-AGI-master/data/evaluation data/arc-agi-1/
cp /tmp/ARC-AGI-master/LICENSE data/arc-agi-1/LICENSE
```

Note that upstream `main`/`master` may have moved since the commits recorded above;
ARC-AGI-2 keeps a `changelog.md` documenting task fixes.
