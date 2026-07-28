#!/usr/bin/env python3
"""Run every baseline solver over a split and save the results.

    python scripts/run_baselines.py                          # arc-agi-2/evaluation
    python scripts/run_baselines.py --dataset arc-agi-1 --split training
    python scripts/run_baselines.py --limit 20 --no-save     # quick smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arc.baselines import build_all  # noqa: E402
from arc.harness import evaluate, print_summary  # noqa: E402
from arc.task import DATASETS, SPLITS  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="arc-agi-2", choices=DATASETS)
    parser.add_argument("--split", default="evaluation", choices=SPLITS)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N tasks")
    parser.add_argument(
        "--drop-leaked",
        action="store_true",
        help="exclude the six tasks shared between the two evaluation sets",
    )
    parser.add_argument("--no-save", action="store_true", help="print only, write nothing")
    args = parser.parse_args(argv)

    for solver in build_all():
        run = evaluate(
            solver,
            dataset=args.dataset,
            split=args.split,
            drop_leaked=args.drop_leaked,
            limit=args.limit,
        )
        print_summary(run)
        if not args.no_save:
            json_path, csv_path = run.save()
            print(f"  saved  {json_path.name}  {csv_path.name}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
