"""Rendering ARC tasks and predictions with matplotlib.

Three views, in increasing zoom:

* :func:`plot_tasks` -- a contact sheet of many tasks, one train pair each, for browsing.
* :func:`plot_task` -- one task in full: every train pair, every test pair.
* :func:`plot_comparison` -- predicted vs expected with the disagreeing cells marked.

Every function returns a ``matplotlib.figure.Figure``; nothing is shown or saved unless
you ask. The module also has a small command line, see ``python -m arc.viz --help``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.figure import Figure

from .metrics import compare_grids, diff_mask
from .task import Grid, Task, find_task, load_split

ARC_COLORS = [
    "#000000",  # 0 black (background)
    "#0074D9",  # 1 blue
    "#FF4136",  # 2 red
    "#2ECC40",  # 3 green
    "#FFDC00",  # 4 yellow
    "#AAAAAA",  # 5 grey
    "#F012BE",  # 6 magenta
    "#FF851B",  # 7 orange
    "#7FDBFF",  # 8 azure
    "#870C25",  # 9 maroon
]
"""The ten ARC colours, in the palette used by the official task viewer."""

ARC_CMAP = ListedColormap(ARC_COLORS)
ARC_NORM = Normalize(vmin=-0.5, vmax=9.5)

_CELL_INCHES = 0.16
"""Rendered size of one grid cell. Grids are drawn to scale, so a 5x5 grid really does
look smaller than a 30x30 one -- which is information you want when eyeballing a task."""

_MIN_PANEL_INCHES = 0.9


def draw_grid(ax, grid: Grid, title: str | None = None, *, title_color: str = "black") -> None:
    """Draw one grid into an existing axes, with cell borders and no ticks."""
    grid = np.asarray(grid)
    ax.imshow(grid, cmap=ARC_CMAP, norm=ARC_NORM, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#404040", linewidth=0.4)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_edgecolor("#808080")
        spine.set_linewidth(0.8)
    if title:
        ax.set_title(title, fontsize=7, color=title_color, pad=3)


def _panel_size(grids: list[Grid]) -> tuple[float, float]:
    height = max((g.shape[0] for g in grids), default=1)
    width = max((g.shape[1] for g in grids), default=1)
    return (
        max(width * _CELL_INCHES, _MIN_PANEL_INCHES),
        max(height * _CELL_INCHES, _MIN_PANEL_INCHES),
    )


def plot_task(task: Task, *, show_test_output: bool = True) -> Figure:
    """Render a whole task: one column per pair, input on top, output below.

    Train pairs come first with black titles, test pairs after them in blue. Test outputs
    are drawn too (the public datasets ship them); pass ``show_test_output=False`` to hide
    them and look at the task the way a solver sees it.
    """
    columns = [("train", i, p) for i, p in enumerate(task.train)]
    columns += [("test", i, p) for i, p in enumerate(task.test)]

    all_grids = [p.input for _, _, p in columns]
    all_grids += [p.output for _, _, p in columns if p.output is not None]
    panel_w, panel_h = _panel_size(all_grids)

    fig, axes = plt.subplots(
        2,
        len(columns),
        figsize=(panel_w * len(columns) + 0.6, panel_h * 2 + 1.4),
        squeeze=False,
    )
    for col, (split, index, pair) in enumerate(columns):
        color = "black" if split == "train" else "#0074D9"
        draw_grid(axes[0][col], pair.input, f"{split} {index}\nin {pair.input.shape}", title_color=color)
        # Grids are drawn to scale, so a short grid floats inside its equal-sized subplot.
        # Anchoring the input row to the bottom and the output row to the top pulls each
        # pair together and keeps the output title clear of the input above it.
        axes[0][col].set_anchor("S")
        bottom = axes[1][col]
        bottom.set_anchor("N")
        if pair.output is not None and (split == "train" or show_test_output):
            draw_grid(bottom, pair.output, f"out {pair.output.shape}", title_color=color)
        else:
            bottom.set_title("?", fontsize=9, color=color)
            bottom.set_axis_off()

    fig.suptitle(f"{task.task_id}   ({task.source})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.6)
    return fig


def plot_tasks(tasks: list[Task], *, columns: int = 6, pair_index: int = 0) -> Figure:
    """Contact sheet: the chosen train pair of each task, laid out in a grid.

    Meant for scanning a set quickly -- which tasks a solver missed, what a split looks
    like overall. Each task contributes two small panels (input, output) side by side.
    """
    if not tasks:
        raise ValueError("nothing to plot")
    rows = -(-len(tasks) // columns)
    fig, axes = plt.subplots(
        rows,
        columns * 2,
        figsize=(columns * 2 * 1.15 + 0.4, rows * 1.5 + 0.5),
        squeeze=False,
    )
    for ax_row in axes:
        for ax in ax_row:
            ax.set_axis_off()

    for n, task in enumerate(tasks):
        row, col = divmod(n, columns)
        pair = task.train[min(pair_index, len(task.train) - 1)]
        for offset, (grid, label) in enumerate(
            ((pair.input, task.task_id), (pair.output, "→"))
        ):
            if grid is None:
                continue
            ax = axes[row][col * 2 + offset]
            ax.set_axis_on()
            draw_grid(ax, grid, label)
    fig.tight_layout()
    return fig


def plot_comparison(
    predicted: Grid,
    expected: Grid,
    *,
    title: str = "",
    task_input: Grid | None = None,
) -> Figure:
    """Predicted vs expected, with the disagreeing cells marked.

    Three panels (four if ``task_input`` is given): the prediction, the ground truth, and
    a diff map. Cells that differ are crossed out on the prediction and shown in red on
    the diff map. When the shapes disagree the comparison is made over the union
    rectangle, so area that one grid has and the other does not shows up as a difference.
    """
    predicted = np.asarray(predicted)
    expected = np.asarray(expected)
    comparison = compare_grids(predicted, expected)
    mask = diff_mask(predicted, expected)

    panels: list[tuple[str, Grid | None]] = []
    if task_input is not None:
        panels.append(("test input", np.asarray(task_input)))
    panels.append((f"predicted {predicted.shape}", predicted))
    panels.append((f"expected {expected.shape}", expected))
    panels.append((f"diff  {int(mask.sum())} cells", None))

    grids = [g for _, g in panels if g is not None] + [mask.astype(np.uint8)]
    panel_w, panel_h = _panel_size(grids)
    fig, axes = plt.subplots(
        1, len(panels), figsize=(panel_w * len(panels) + 0.6, panel_h + 1.2), squeeze=False
    )

    for ax, (label, grid) in zip(axes[0], panels):
        if grid is None:
            ax.imshow(mask, cmap=ListedColormap(["#202020", "#FF4136"]), vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_xticks(np.arange(-0.5, mask.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, mask.shape[0], 1), minor=True)
            ax.grid(which="minor", color="#404040", linewidth=0.4)
            ax.tick_params(which="both", bottom=False, left=False,
                           labelbottom=False, labelleft=False)
            ax.set_title(label, fontsize=7, pad=3)
        else:
            draw_grid(ax, grid, label)

    # Cross out the wrong cells on the prediction panel.
    prediction_ax = axes[0][1 if task_input is not None else 0]
    wrong = np.argwhere(mask[: predicted.shape[0], : predicted.shape[1]])
    if len(wrong) <= 400:  # beyond that the crosses are just a red smear
        prediction_ax.scatter(
            wrong[:, 1], wrong[:, 0], marker="x", s=14, c="#FF4136", linewidths=0.9
        )

    verdict = "EXACT MATCH" if comparison.exact_match else "MISMATCH"
    accuracy = (
        f"cells {comparison.cell_accuracy:.3f}"
        if comparison.cell_accuracy is not None
        else f"shape wrong, padded cells {comparison.cell_accuracy_padded:.3f}"
    )
    header = f"{title}  " if title else ""
    fig.suptitle(
        f"{header}{verdict}   {accuracy}   iou {comparison.foreground_iou:.3f}"
        f"   hist {comparison.color_histogram_distance:.3f}",
        fontsize=9,
        color="#2ECC40" if comparison.exact_match else "#FF4136",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def save(fig: Figure, path: str | Path, *, dpi: int = 160) -> Path:
    """Write a figure to disk and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render ARC tasks to a PNG.")
    parser.add_argument("--task", help="task id to render in full")
    parser.add_argument(
        "--dataset",
        default="arc-agi-2",
        help="dataset to search or sheet; most task ids exist in both datasets",
    )
    parser.add_argument("--split", help="render a contact sheet of this split")
    parser.add_argument("--limit", type=int, default=24, help="tasks on the contact sheet")
    parser.add_argument("--out", default="task.png", help="output PNG path")
    args = parser.parse_args(argv)

    mpl.use("Agg")
    if args.task:
        fig = plot_task(find_task(args.task, dataset=args.dataset, split=args.split))
    elif args.split:
        fig = plot_tasks(load_split(args.dataset, args.split)[: args.limit])
    else:
        parser.error("give either --task or --split")
    print(save(fig, args.out))


if __name__ == "__main__":
    main()
