"""Generates the two paper figures from the results CSVs.

Uses the validated categorical palette (light mode, adjacent-pair order:
blue, orange, aqua, yellow) documented in the dataviz skill's palette
reference rather than matplotlib's default color cycle.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_registry import results_suffix

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#dddddd"

SHORT_NAMES = {
    "Baseline (TF-IDF)": "TF-IDF",
    "SBERT (all-MiniLM-L6-v2)": "SBERT",
    "LLM (mxbai-embed-large, local)": "LLM",
    "Hybrid (TF-IDF + mxbai-embed-large)": "Hybrid",
}


def _clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors=TEXT_SECONDARY, length=0)


def plot_main_results(dataset="movielens"):
    suffix = results_suffix(dataset)
    df = pd.read_csv(RESULTS_DIR / f"results_table{suffix}.csv")
    df = df[df["Scorer"] == "lgbm"].copy()
    df["ShortName"] = df["Pipeline"].map(SHORT_NAMES)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    x = range(len(df))
    bars = ax.bar(
        x,
        df["NDCG@10_mean"],
        yerr=df["NDCG@10_ci95"],
        capsize=4,
        color=CATEGORICAL[: len(df)],
        width=0.55,
        zorder=3,
        error_kw=dict(ecolor=TEXT_SECONDARY, elinewidth=1.2),
    )
    for rect, mean in zip(bars, df["NDCG@10_mean"]):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.015,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=TEXT_PRIMARY,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["ShortName"], color=TEXT_PRIMARY)
    ax.set_ylabel("NDCG@10 (mean ± 95% CI, 5 seeds)", color=TEXT_PRIMARY)
    ax.set_ylim(0, max(df["NDCG@10_mean"]) * 1.2)
    ax.set_title("Ranking quality by item-feature pipeline (LGBMRanker)", color=TEXT_PRIMARY)
    _clean_axes(ax)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"main_results{suffix}.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_robustness(dataset="movielens"):
    suffix = results_suffix(dataset)
    df = pd.read_csv(RESULTS_DIR / f"results_table_robustness{suffix}.csv")
    df["ShortName"] = df["Pipeline"].map(SHORT_NAMES)

    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=200)
    x = list(range(len(df)))
    width = 0.35

    random_bars = ax.bar(
        [i - width / 2 for i in x],
        df["NDCG@10_random_mean"],
        yerr=df["NDCG@10_random_ci95"],
        width=width,
        color=CATEGORICAL[0],
        capsize=3,
        zorder=3,
        error_kw=dict(ecolor=TEXT_SECONDARY, elinewidth=1.0),
        label="Random split",
    )
    temporal_bars = ax.bar(
        [i + width / 2 for i in x],
        df["NDCG@10_temporal_mean"],
        yerr=df["NDCG@10_temporal_ci95"],
        width=width,
        color=CATEGORICAL[1],
        capsize=3,
        zorder=3,
        error_kw=dict(ecolor=TEXT_SECONDARY, elinewidth=1.0),
        label="Temporal split (shift)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["ShortName"], color=TEXT_PRIMARY)
    ax.set_ylabel("NDCG@10 (mean ± 95% CI, 3 seeds)", color=TEXT_PRIMARY)
    ax.set_title("Robustness under temporal distribution shift", color=TEXT_PRIMARY)
    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY)
    _clean_axes(ax)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"robustness{suffix}.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="movielens", choices=["movielens", "amazon", "yelp"]
    )
    args = parser.parse_args()
    plot_main_results(args.dataset)
    plot_robustness(args.dataset)
