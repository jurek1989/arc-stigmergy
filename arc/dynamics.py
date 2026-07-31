"""Recording how a library behaves over a whole run.

`arc.harness` measures a single evaluation pass: given a solver, how many tasks fall. That
is the wrong shape of instrument for the question this project actually asks, which is
about a *trajectory* — what happens to a shared library of primitives as agents keep
adding to it and the compression criterion keeps pruning it.

The founding sketch proposed one curve, library size against tasks solved, read as:
flattening means stigmergy, linear growth means noise. That curve cannot tell the two
interesting failure modes apart. A library that stops growing while success also stops
rising is not convergence, it is **collapse** — the pheromone evaporated, every agent now
walks the same dead trail, and the system has quietly stopped exploring. Both cases show a
flat library. What separates them is whether *usage* stays spread out and whether newly
accepted solutions keep getting cheaper to describe.

So four signals are tracked against one shared axis:

1. cumulative distinct tasks solved — is it working at all
2. library size — is growth decelerating
3. entropy of primitive usage — is the library still being used broadly, or has it
   collapsed onto a handful of primitives
4. description length of newly accepted solutions — must fall if compression is real

**Compute is the shared axis.** See ``CLAUDE.md``, section "Compute": compute is the
number of candidate programs evaluated against a task's train pairs — the same quantity
the harness records as ``steps``.

This module contains no mechanism. It receives plain numbers and stores them. The future
search will build :class:`EpochSnapshot` objects itself; there is deliberately no abstract
interface here for a mechanism that does not exist yet.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

_LOG2_E = 1.0 / math.log(2.0)


@dataclass(frozen=True)
class EpochSnapshot:
    """The state of a run at the end of one epoch.

    All three of ``compute``, ``tasks_solved`` and the usage counts are *cumulative* over
    the run so far; ``new_description_lengths`` is the only per-epoch quantity, and it is
    empty in epochs where nothing was accepted.
    """

    epoch: int
    compute: int
    """Cumulative candidate programs evaluated against train pairs. Same quantity as
    ``TaskResult.steps`` in the harness; see CLAUDE.md, "Compute"."""

    tasks_solved: int
    """Cumulative count of *distinct* tasks solved at least once."""

    library_size: int
    """Number of primitives currently in the library. Not required to be monotonic:
    a compression criterion that also evicts primitives is the whole point, and a log that
    forbade shrinkage could not record evaporation."""

    usage_counts: Mapping[str, int]
    """How often each primitive appears across the currently retained solution corpus.

    May cover fewer primitives than ``library_size`` — a primitive can sit in the library
    unused. It may never cover *more*, which is checked."""

    new_description_lengths: tuple[float, ...] = ()
    """Description lengths, in bits, of the solutions accepted during this epoch."""

    def __post_init__(self) -> None:
        for name in ("epoch", "compute", "tasks_solved", "library_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")

        counts = dict(self.usage_counts)
        for key, count in counts.items():
            if not isinstance(count, int) or isinstance(count, bool):
                raise TypeError(f"usage count for {key!r} must be an int")
            if count < 0:
                raise ValueError(f"usage count for {key!r} is negative: {count}")
        if len(counts) > self.library_size:
            raise ValueError(
                f"epoch {self.epoch}: {len(counts)} primitives used but library_size is "
                f"{self.library_size}; usage cannot cover primitives that are not there"
            )

        lengths = tuple(float(x) for x in self.new_description_lengths)
        for length in lengths:
            if not math.isfinite(length) or length < 0:
                raise ValueError(f"epoch {self.epoch}: bad description length {length}")

        object.__setattr__(self, "usage_counts", MappingProxyType(counts))
        object.__setattr__(self, "new_description_lengths", lengths)

    # -- derived quantities -----------------------------------------------------

    @property
    def total_usage(self) -> int:
        """Total number of primitive occurrences in the retained corpus."""
        return sum(self.usage_counts.values())

    @property
    def n_distinct_used(self) -> int:
        """Primitives with a non-zero count. Bounded above by ``library_size``."""
        return sum(1 for count in self.usage_counts.values() if count > 0)

    @property
    def usage_entropy_bits(self) -> float:
        """Shannon entropy of the usage distribution, in bits (plug-in estimator).

        Zero when nothing is used or when everything funnels through one primitive, which
        is exactly the collapse signature. Bounded above by ``log2(n_distinct_used)``.

        This is the maximum-likelihood estimator and it is **biased downwards** on small
        corpora — see :attr:`usage_entropy_bits_miller_madow`.
        """
        total = self.total_usage
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in self.usage_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return entropy

    @property
    def usage_entropy_bits_miller_madow(self) -> float:
        """Miller–Madow bias-corrected entropy, in bits.

        The plug-in estimator understates entropy by roughly ``(K - 1) / (2N ln 2)`` bits,
        where ``K`` is the number of distinct primitives observed and ``N`` the total
        usage. Early in a run ``N`` is small while ``K`` grows, so the bias is largest
        exactly where it does the most damage: a healthy, broadly-used young library reads
        as if it were collapsing.

        Recorded alongside the plug-in value rather than replacing it. The gap between the
        two is itself the diagnostic — while it is wide, the corpus is too small to read
        the entropy panel at all.
        """
        total = self.total_usage
        if total == 0:
            return 0.0
        return self.usage_entropy_bits + (self.n_distinct_used - 1) * _LOG2_E / (2 * total)

    @property
    def normalized_usage_entropy(self) -> float | None:
        """Usage entropy as a fraction of the maximum the library could carry, in [0, 1].

        ``usage_entropy_bits / log2(library_size)``, and ``None`` when ``library_size < 2``
        because a one-primitive library has no spread to measure.

        Normalisation uses the plug-in value on purpose: it is bounded by
        ``log2(library_size)``, so the ratio stays in [0, 1]. The Miller–Madow correction
        carries no such bound and would produce ratios above 1 on small corpora.
        """
        if self.library_size < 2:
            return None
        return self.usage_entropy_bits / math.log2(self.library_size)

    @property
    def n_new_solutions(self) -> int:
        return len(self.new_description_lengths)

    @property
    def mean_new_description_length(self) -> float | None:
        """Mean description length of this epoch's accepted solutions, or ``None`` if
        none were accepted. ``None`` rather than 0.0: an epoch that accepted nothing is
        not an epoch that accepted something free."""
        if not self.new_description_lengths:
            return None
        return sum(self.new_description_lengths) / len(self.new_description_lengths)

    # -- serialisation ----------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "compute": self.compute,
            "tasks_solved": self.tasks_solved,
            "library_size": self.library_size,
            "usage_counts": dict(self.usage_counts),
            "new_description_lengths": list(self.new_description_lengths),
            "derived": {
                "total_usage": self.total_usage,
                "n_distinct_used": self.n_distinct_used,
                "usage_entropy_bits": self.usage_entropy_bits,
                "usage_entropy_bits_miller_madow": self.usage_entropy_bits_miller_madow,
                "normalized_usage_entropy": self.normalized_usage_entropy,
                "n_new_solutions": self.n_new_solutions,
                "mean_new_description_length": self.mean_new_description_length,
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EpochSnapshot":
        """Rebuild from :meth:`as_dict`. The ``derived`` block is ignored — it is written
        for whoever reads the JSON by hand, and recomputing it is cheaper than trusting it."""
        return cls(
            epoch=int(payload["epoch"]),
            compute=int(payload["compute"]),
            tasks_solved=int(payload["tasks_solved"]),
            library_size=int(payload["library_size"]),
            usage_counts={str(k): int(v) for k, v in payload["usage_counts"].items()},
            new_description_lengths=tuple(
                float(x) for x in payload.get("new_description_lengths", ())
            ),
        )


@dataclass
class DynamicsLog:
    """An ordered series of :class:`EpochSnapshot`, with validation on append.

    Naming mirrors ``RunResult`` so that both writers drop files into ``results/`` under
    the same stem convention, and a run's score file and its dynamics file sort next to
    each other.
    """

    name: str
    dataset: str = "arc-agi-2"
    split: str = "evaluation"
    started_at: str = ""
    snapshots: list[EpochSnapshot] = field(default_factory=list)
    n_tasks_total: int | None = None
    """Size of the task set, if known. Only used to annotate the success panel."""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.started_at:
            self.started_at = datetime.now().isoformat(timespec="seconds")
        if not self.metadata:
            self.metadata = {"python": sys.version.split()[0], "platform": platform.platform()}
        existing, self.snapshots = list(self.snapshots), []
        for snapshot in existing:
            self.append(snapshot)

    # -- building ---------------------------------------------------------------

    def append(self, snapshot: EpochSnapshot) -> None:
        """Add an epoch, checking it follows the previous one.

        Epoch index and cumulative compute must strictly increase — an epoch that burned
        no compute is not an epoch, and a repeated index means two writers are appending
        to the same log. Cumulative tasks solved must not decrease, since it counts
        distinct tasks ever solved. Library size is deliberately unconstrained.
        """
        if self.snapshots:
            previous = self.snapshots[-1]
            if snapshot.epoch <= previous.epoch:
                raise ValueError(
                    f"epoch must strictly increase: {snapshot.epoch} follows {previous.epoch}"
                )
            if snapshot.compute <= previous.compute:
                raise ValueError(
                    f"epoch {snapshot.epoch}: cumulative compute must strictly increase, "
                    f"{snapshot.compute} follows {previous.compute}"
                )
            if snapshot.tasks_solved < previous.tasks_solved:
                raise ValueError(
                    f"epoch {snapshot.epoch}: cumulative tasks solved fell from "
                    f"{previous.tasks_solved} to {snapshot.tasks_solved}"
                )
        self.snapshots.append(snapshot)

    def extend(self, snapshots: Iterable[EpochSnapshot]) -> None:
        for snapshot in snapshots:
            self.append(snapshot)

    def __len__(self) -> int:
        return len(self.snapshots)

    def __iter__(self):
        return iter(self.snapshots)

    def __getitem__(self, index: int) -> EpochSnapshot:
        return self.snapshots[index]

    @property
    def final(self) -> EpochSnapshot | None:
        return self.snapshots[-1] if self.snapshots else None

    # -- series, for plotting and for the regime statistics ----------------------

    def series(self, attribute: str) -> list[Any]:
        """Values of one snapshot attribute across the run, in epoch order."""
        return [getattr(snapshot, attribute) for snapshot in self.snapshots]

    def defined_series(self, attribute: str) -> tuple[list[int], list[float]]:
        """Cumulative compute and values for the epochs where ``attribute`` is not None.

        Used for the description-length panel, where epochs that accepted nothing have to
        be gaps rather than zeros.
        """
        x, y = [], []
        for snapshot in self.snapshots:
            value = getattr(snapshot, attribute)
            if value is not None:
                x.append(snapshot.compute)
                y.append(float(value))
        return x, y

    def summary(self) -> dict[str, Any]:
        """Aggregate description of the run, for the JSON header and for eyeballing."""
        final = self.final
        return {
            "name": self.name,
            "dataset": self.dataset,
            "split": self.split,
            "n_epochs": len(self.snapshots),
            "n_tasks_total": self.n_tasks_total,
            "final_compute": final.compute if final else 0,
            "final_tasks_solved": final.tasks_solved if final else 0,
            "final_library_size": final.library_size if final else 0,
            "final_usage_entropy_bits": final.usage_entropy_bits if final else None,
            "final_normalized_usage_entropy": (
                final.normalized_usage_entropy if final else None
            ),
            "library_growth_ratio": library_growth_ratio(self),
            "description_length_trend": description_length_trend(self),
            "total_new_solutions": sum(s.n_new_solutions for s in self.snapshots),
        }

    # -- persistence ------------------------------------------------------------

    def stem(self) -> str:
        """Filename stem, identical in shape to the one ``RunResult.save`` produces."""
        return (
            f"{self.started_at.replace(':', '').replace('-', '')}"
            f"__{_slug(self.name)}__{_slug(self.dataset)}_{_slug(self.split)}"
        )

    def save(self, directory: str | Path = RESULTS_DIR) -> tuple[Path, Path]:
        """Write ``<stem>.dynamics.json`` (everything) and ``<stem>.dynamics.csv``.

        Same division of labour as ``RunResult.save``: the JSON is the archive and the
        only thing :meth:`load` reads, the CSV is the flat per-epoch table to pull into
        Polars. The usage-count mapping only exists in the JSON — it does not fit a CSV
        cell, and the CSV carries the derived scalars computed from it instead.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / f"{self.stem()}.dynamics.json"
        csv_path = directory / f"{self.stem()}.dynamics.csv"

        json_path.write_text(
            json.dumps(
                {
                    "summary": self.summary(),
                    "name": self.name,
                    "dataset": self.dataset,
                    "split": self.split,
                    "started_at": self.started_at,
                    "n_tasks_total": self.n_tasks_total,
                    "metadata": self.metadata,
                    "snapshots": [s.as_dict() for s in self.snapshots],
                },
                indent=2,
            )
        )

        fields = [
            "epoch", "compute", "tasks_solved", "library_size",
            "total_usage", "n_distinct_used",
            "usage_entropy_bits", "usage_entropy_bits_miller_madow",
            "normalized_usage_entropy",
            "n_new_solutions", "mean_new_description_length",
        ]
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for s in self.snapshots:
                writer.writerow(
                    {
                        "epoch": s.epoch,
                        "compute": s.compute,
                        "tasks_solved": s.tasks_solved,
                        "library_size": s.library_size,
                        "total_usage": s.total_usage,
                        "n_distinct_used": s.n_distinct_used,
                        "usage_entropy_bits": f"{s.usage_entropy_bits:.6f}",
                        "usage_entropy_bits_miller_madow": (
                            f"{s.usage_entropy_bits_miller_madow:.6f}"
                        ),
                        "normalized_usage_entropy": _fmt(s.normalized_usage_entropy),
                        "n_new_solutions": s.n_new_solutions,
                        "mean_new_description_length": _fmt(s.mean_new_description_length),
                    }
                )
        return json_path, csv_path

    @classmethod
    def load(cls, path: str | Path) -> "DynamicsLog":
        """Read back a log written by :meth:`save`."""
        payload = json.loads(Path(path).read_text())
        return cls(
            name=payload["name"],
            dataset=payload.get("dataset", ""),
            split=payload.get("split", ""),
            started_at=payload.get("started_at", ""),
            snapshots=[EpochSnapshot.from_dict(s) for s in payload["snapshots"]],
            n_tasks_total=payload.get("n_tasks_total"),
            metadata=payload.get("metadata", {}),
        )


# -- regime statistics ----------------------------------------------------------
#
# These three numbers are what separate a healthy run from noise and from collapse. They
# live here rather than in the tests because they are part of the instrument: when the
# real mechanism runs, these are the numbers to read off it.


def library_growth_ratio(log: DynamicsLog, fraction: float = 0.25) -> float | None:
    """How much the library grew late in the run, relative to how much it grew early.

    Compares growth over the last ``fraction`` of epochs against the first ``fraction``.
    Around 1.0 means growth never slowed — the library is accumulating linearly, which is
    the noise signature. Well below 1.0 means deceleration. At or near 0.0 the library has
    stopped moving entirely, which on its own is ambiguous: convergence and collapse look
    identical here and only the entropy panel tells them apart.

    Negative values mean the library shrank over the late window — eviction outpaced
    acceptance, which is worth seeing rather than clamping away.

    ``None`` when the run is too short to split, or when the library did not grow at all
    early on and the ratio would be a division by zero.
    """
    sizes = log.series("library_size")
    window = max(1, int(len(sizes) * fraction))
    if len(sizes) < 2 * window + 1:
        return None
    early = sizes[window] - sizes[0]
    late = sizes[-1] - sizes[-1 - window]
    if early <= 0:
        return None
    return late / early


def description_length_trend(log: DynamicsLog) -> float | None:
    """Least-squares slope of mean new-solution description length against epoch index.

    Negative means newly accepted solutions are getting cheaper to describe, which is the
    only direct evidence that the compression criterion is doing anything. Slope is per
    epoch, not per unit compute: bits-per-candidate-evaluated is a number too small to
    read, and epochs are the unit accepted solutions arrive in.

    ``None`` when fewer than two epochs accepted anything.
    """
    xs: list[float] = []
    ys: list[float] = []
    for snapshot in log.snapshots:
        mean = snapshot.mean_new_description_length
        if mean is not None:
            xs.append(float(snapshot.epoch))
            ys.append(mean)
    if len(xs) < 2:
        return None
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    variance = sum((x - mean_x) ** 2 for x in xs)
    if variance == 0:
        return None
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return covariance / variance


def final_normalized_entropy(log: DynamicsLog) -> float | None:
    """Normalised usage entropy at the end of the run, or ``None`` if unavailable."""
    final = log.final
    return final.normalized_usage_entropy if final else None


# -- helpers --------------------------------------------------------------------


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def _slug(text: str) -> str:
    # Kept byte-identical to harness._slug on purpose, so both writers produce the same
    # stem shape. tests/test_dynamics.py pins the two against each other.
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in text)


def build_log(
    name: str,
    snapshots: Sequence[EpochSnapshot],
    **kwargs: Any,
) -> DynamicsLog:
    """Convenience constructor: a named log from an already-built sequence of epochs."""
    return DynamicsLog(name=name, snapshots=list(snapshots), **kwargs)
