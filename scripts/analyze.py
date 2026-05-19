"""Paired performance comparison: 1-vs-1 scatter plot + Wilcoxon signed-rank test.

Usage:
    python -m scripts.analyze results.csv \\
        --base-col base_ens_acc --new-col pruned_ens_acc --output fig
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def plot_one_vs_one(
    df: pd.DataFrame,
    base_col: str,
    new_col: str,
    xlabel: str = "Base",
    ylabel: str = "New",
    title: str = "",
    xy_min: float = 0.0,
    xy_max: float = 1.0,
    output: Path | None = None,
) -> float:
    """Render the scatter, return the Wilcoxon p-value."""
    base = df[base_col].to_numpy()
    new = df[new_col].to_numpy()
    stat, p_value = wilcoxon(base, new)

    wins = int((new > base).sum())
    ties = int((new == base).sum())
    losses = int((new < base).sum())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([xy_min, xy_max], [xy_min, xy_max], color="blue", linewidth=1)

    above = new > base
    same = new == base
    below = new < base

    ax.scatter(base[below], new[below], color="green", label=f"{xlabel} wins — {losses}")
    ax.scatter(base[same], new[same], color="orange", label=f"Equal — {ties}")
    ax.scatter(base[above], new[above], color="red", label=f"{ylabel} wins — {wins}")
    ax.scatter([], [], color="none", label=f"p-value: {p_value:.5f}")

    ax.set_xlim(xy_min, xy_max)
    ax.set_ylim(xy_min, xy_max)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    if title:
        ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right")
    fig.tight_layout()

    if output is not None:
        fig.savefig(f"{output}.pdf", bbox_inches="tight", pad_inches=0)
        fig.savefig(f"{output}.png", bbox_inches="tight", pad_inches=0, dpi=150)
        print(f"saved {output}.pdf / {output}.png")
    plt.show()
    return float(p_value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("--base-col", required=True)
    parser.add_argument("--new-col", required=True)
    parser.add_argument("--xlabel", default="Base")
    parser.add_argument("--ylabel", default="New")
    parser.add_argument("--title", default="")
    parser.add_argument("--xy-min", type=float, default=0.0)
    parser.add_argument("--xy-max", type=float, default=1.01)
    parser.add_argument("--output", type=Path, default=None, help="Output figure stem (no extension)")
    args = parser.parse_args()

    df = pd.read_csv(args.results_csv)
    p = plot_one_vs_one(
        df,
        base_col=args.base_col,
        new_col=args.new_col,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        title=args.title,
        xy_min=args.xy_min,
        xy_max=args.xy_max,
        output=args.output,
    )
    print(f"Wilcoxon signed-rank p-value: {p:.6f}")


if __name__ == "__main__":
    main()
