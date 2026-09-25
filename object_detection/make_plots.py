#!/usr/bin/env python3
"""Plot object-detection benchmark results: FPS vs power, and inferences per joule.

Each column in the results CSV (one run: platform + engine + quant) is
plotted as its own labeled series/point (scatter) or bar (bar chart).

Usage:
    python plot_fps_vs_power.py [path/to/results.csv] [-o scatter.png] [-b bar.png]
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Fixed-order categorical palette (dataviz skill reference palette, light mode).
CATEGORICAL_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def load_runs(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, index_col=0)
    df = raw.transpose()
    df.columns = [str(c).strip() for c in df.columns]

    # The platform name is baked into the header row (e.g. "Astra", "Astra.1",
    # "Raspi.1") since this CSV's first row is "Platform,<name>,<name>,..."
    # rather than a generic label. Strip pandas' de-dup suffix to recover it.
    df["Platform"] = [re.sub(r"\.\d+$", "", str(idx)) for idx in df.index]

    df["mean fps - inference only"] = pd.to_numeric(df["mean fps - inference only"], errors="coerce")
    df["mean power (w)"] = pd.to_numeric(df["mean power (w)"], errors="coerce")
    df["Inferences per Joule"] = pd.to_numeric(df["Inferences per Joule"], errors="coerce")
    df = df.dropna(subset=["mean fps - inference only", "mean power (w)"])

    df["label"] = (
        df["Platform"].astype(str).str.strip()
        + "_"
        + df["Engine"].astype(str).str.strip()
        + "_"
        + df["Model Quant"].astype(str).str.strip()
    )
    return df


def plot_scatter(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for i, (_, row) in enumerate(df.iterrows()):
        color = CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)]
        x = row["mean power (w)"]
        y = row["mean fps - inference only"]
        ax.scatter(
            x, y,
            s=110,
            color=color,
            edgecolor=SURFACE,
            linewidth=1.2,
            zorder=3,
            label=row["label"],
        )
        ax.annotate(
            row["label"],
            (x, y),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
            color=INK,
        )

    ax.set_xlabel("Mean Power (W)", fontsize=12, color=INK)
    ax.set_ylabel("Mean FPS", fontsize=12, color=INK)
    ax.set_title(
        "Mean FPS vs Mean Power by Platform / Engine / Quant",
        fontsize=13, color=INK, pad=14,
    )
    ax.margins(x=0.15, y=0.15)

    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=SECONDARY_INK)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=9,
        title="Platform_Engine_Quant",
        title_fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    print(f"Saved plot to {out_path}")


def plot_bar(df: pd.DataFrame, out_path: Path) -> None:
    df = df.dropna(subset=["Inferences per Joule"])
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    x_positions = range(len(df))
    for i, (x, (_, row)) in enumerate(zip(x_positions, df.iterrows())):
        color = CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)]
        value = row["Inferences per Joule"]
        ax.bar(
            x, value,
            color=color,
            edgecolor=SURFACE,
            linewidth=1.2,
            zorder=3,
            label=row["label"],
            width=0.7,
        )
        ax.annotate(
            f"{value:.2f}",
            (x, value),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=9,
            color=INK,
        )

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(df["label"], rotation=30, ha="right", fontsize=9, color=SECONDARY_INK)
    ax.set_xlabel("Platform / Engine / Quant", fontsize=12, color=INK)
    ax.set_ylabel("Inferences per Joule", fontsize=12, color=INK)
    ax.set_title(
        "Inferences per Joule by Platform / Engine / Quant",
        fontsize=13, color=INK, pad=14,
    )
    ax.margins(x=0.1)
    ax.set_ylim(0, df["Inferences per Joule"].max() * 1.15)

    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=SECONDARY_INK)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=9,
        title="Platform_Engine_Quant",
        title_fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="yolo_benchmarking_results_sept_7.csv",
        help="Path to the benchmark results CSV (default: yolo_benchmarking_results_sept_7.csv)",
    )
    parser.add_argument(
        "-o", "--output",
        default="fps_vs_power.png",
        help="Scatter plot output image path (default: fps_vs_power.png)",
    )
    parser.add_argument(
        "-b", "--bar-output",
        default="inferences_per_joule.png",
        help="Bar chart output image path (default: inferences_per_joule.png)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        sys.exit(f"CSV file not found: {csv_path}")

    df = load_runs(csv_path)
    plot_scatter(df, Path(args.output))
    plot_bar(df, Path(args.bar_output))


if __name__ == "__main__":
    main()
