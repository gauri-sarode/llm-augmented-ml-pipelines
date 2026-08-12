"""Corpus redundancy vs. embedding benefit: computes, for each dataset, the
average inter-item TF-IDF cosine similarity (a cheap proxy for how much
item text is lexically redundant/generic vs. distinctive) and pairs it with
the observed NDCG@10 gap between the best embedding pipeline and TF-IDF.

This corroborates -- in a different setting (embeddings as features for a
learned ranker, not retrieval; recommendation domains; a genuine LLM
embedding, not just standard dense retrievers) -- the redundancy-moderates-
dense-benefit mechanism formalized in RARE (Cho & Lee, ACL 2026): dense
methods' advantage over lexical ones shrinks as corpus lexical overlap
grows. It is NOT presented as a novel discovery.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_registry import get_dataset, results_suffix
from src.generate_figures import CATEGORICAL, TEXT_PRIMARY, _clean_axes

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"

DATASETS = ["movielens", "amazon", "yelp"]
LABELS = {"movielens": "MovieLens", "amazon": "Amazon Reviews", "yelp": "Yelp"}


def redundancy_score(texts, sample_n=500, seed=0):
    """Average pairwise TF-IDF cosine similarity across a sample of items.

    Higher = more lexically redundant/generic text across items (TF-IDF
    already discounts shared vocabulary via IDF, so there's little residual
    signal for a dense embedding to add). Lower = items are already lexically
    distinctive from each other.
    """
    rng = np.random.RandomState(seed)
    texts = list(texts)
    if len(texts) > sample_n:
        idx = rng.choice(len(texts), sample_n, replace=False)
        texts = [texts[i] for i in idx]
    vec = TfidfVectorizer(max_features=2000)
    X = vec.fit_transform(texts)
    sims = cosine_similarity(X)
    n = sims.shape[0]
    off_diag = sims[~np.eye(n, dtype=bool)]
    return float(off_diag.mean())


def ndcg_gap(dataset):
    """Best-embedding NDCG@10 minus TF-IDF NDCG@10 (LGBMRanker), or None if
    that dataset's results table isn't available (or is a stale pre-rerun
    file from the old 4-pipeline mxbai-only setup) yet."""
    suffix = results_suffix(dataset)
    path = RESULTS_DIR / f"results_table{suffix}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "LLM (phi4-mini, local)" not in df["Pipeline"].values:
        return None  # stale pre-rerun file, not the current 5-pipeline results
    df = df[df["Scorer"] == "lgbm"].set_index("Pipeline")
    tfidf = df.loc["Baseline (TF-IDF)", "NDCG@10_mean"]
    embed_pipelines = [p for p in df.index if p != "Baseline (TF-IDF)"]
    best_embed = df.loc[embed_pipelines, "NDCG@10_mean"].max()
    return float(best_embed - tfidf)


def build_table():
    rows = []
    for ds in DATASETS:
        load_fn, _, _ = get_dataset(ds)
        _, movies = load_fn()
        redundancy = redundancy_score(movies["text"])
        gap = ndcg_gap(ds)
        rows.append(
            {
                "Dataset": LABELS[ds],
                "RedundancyScore": redundancy,
                "AvgWordsPerItem": float(np.mean([len(t.split()) for t in movies["text"]])),
                "NDCGGapVsTFIDF": gap,
            }
        )
    return pd.DataFrame(rows)


def plot_redundancy(df):
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=200)
    for i, row in df.iterrows():
        if pd.isna(row["NDCGGapVsTFIDF"]):
            continue
        ax.scatter(
            row["RedundancyScore"],
            row["NDCGGapVsTFIDF"],
            s=100,
            color=CATEGORICAL[i % len(CATEGORICAL)],
            zorder=3,
            edgecolors="white",
            linewidths=0.8,
        )
        ax.annotate(
            row["Dataset"],
            (row["RedundancyScore"], row["NDCGGapVsTFIDF"]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
            color=TEXT_PRIMARY,
        )
    ax.axhline(0, color="#c3c2b7", linewidth=1, zorder=1)
    ax.set_xlabel("Corpus redundancy (avg. inter-item TF-IDF cosine sim.)", color=TEXT_PRIMARY)
    ax.set_ylabel("Best-embedding NDCG@10 $-$ TF-IDF NDCG@10", color=TEXT_PRIMARY)
    ax.set_title("Embedding benefit vs. corpus redundancy", color=TEXT_PRIMARY)
    _clean_axes(ax)
    ax.xaxis.grid(True, color="#dddddd", linewidth=0.8, zorder=0)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "redundancy_vs_gap.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    df = build_table()
    out_path = RESULTS_DIR / "redundancy_analysis.csv"
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved to {out_path}")
    plot_redundancy(df)


if __name__ == "__main__":
    main()
