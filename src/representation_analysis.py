"""Section VII analyses: embedding-space geometry, the mean-centering check
for raw phi4-mini, and the per-user history-length interaction.

    python -m src.representation_analysis geometry [--dataset yelp]
    python -m src.representation_analysis history [--dataset amazon]
    python -m src.representation_analysis centering     # one seed, ~10 min
    python -m src.representation_analysis affinity [--dataset yelp]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.candidates import POSITIVE_THRESHOLD, build_user_candidates
from src.dataset_registry import get_dataset
from src.embeddings import _load_cache, build_tfidf_features
from src.run_experiment import run_pipeline, run_tag
from src.metrics import evaluate_ranking
from src.user_features import affinity_features, popularity_stats, training_interactions, user_positive_map

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SBERT = "SBERT (all-MiniLM-L6-v2)"
CACHES = {
    "SBERT": "sbert_all-MiniLM-L6-v2",
    "mxbai": "llm_mxbai-embed-large",
    "phi4-mini (raw)": "truellm_phi4-mini",
    "Qwen3-Emb.": "trainedllm_Qwen3-Embedding-4B",
}


def geometry(dataset="movielens", sample=1000, seed=0):
    """Pairwise-cosine spread and effective dimensionality per embedding space."""
    load_fn, prefix, _ = get_dataset(dataset)
    _, movies = load_fn()
    ids = np.random.default_rng(seed).choice(
        movies["movieId"].to_numpy(), min(sample, len(movies)), replace=False
    )
    spaces = {"TF-IDF": build_tfidf_features(movies)[0]}
    spaces.update({name: _load_cache(prefix + cache) for name, cache in CACHES.items()})
    rows = []
    for name, fd in spaces.items():
        X = np.stack([fd[i] for i in ids]).astype(np.float64)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        off = (X @ X.T)[~np.eye(len(X), dtype=bool)]
        ev = np.clip(np.linalg.eigvalsh(np.cov(X.T)), 0, None)
        rows.append(
            {
                "Space": name,
                "Dim": X.shape[1],
                "MeanCos": off.mean(),
                "SDCos": off.std(),
                "EffDim": ev.sum() ** 2 / (ev**2).sum(),  # participation ratio
            }
        )
    return pd.DataFrame(rows)


def history_trend(dataset="movielens"):
    """Regress per-user paired NDCG differences (mean over seeds) on log #positives."""
    ratings, _ = get_dataset(dataset)[0]()
    npos = ratings[ratings["rating"] >= POSITIVE_THRESHOLD].groupby("userId").size()
    comparisons = {
        "Qwen3-SBERT": ("LLM embedder (Qwen3-Embedding-4B)", SBERT),
        "mxbai-SBERT": ("Large Embedding Model (mxbai-embed-large)", SBERT),
        "SBERT-TFIDF": (SBERT, "Baseline (TF-IDF)"),
    }
    rows = []
    for mode, neg in [("personalized_pop", "popularity"), ("personalized", "popularity"), ("personalized_pop", "uniform")]:
        per_user = pd.read_csv(RESULTS_DIR / f"per_user{run_tag(dataset, mode, neg)}.csv")
        for scorer in ("lgbm", "mlp"):
            u = per_user[per_user["Scorer"].isin([scorer, "-"])]
            u = u.groupby(["Pipeline", "userId"])["NDCG"].mean().unstack(0)
            x = np.log(npos.reindex(u.index).to_numpy())
            for label, (a, b) in comparisons.items():
                fit = stats.linregress(x, (u[a] - u[b]).to_numpy())
                rows.append(
                    {"Config": f"{mode}/{neg}", "Scorer": scorer, "Diff": label,
                     "SlopePerLogPos": fit.slope, "p": fit.pvalue}
                )
    return pd.DataFrame(rows)


def centering(seed=1):
    """phi4-mini raw vs corpus-mean-centered vs SBERT, one seed, hard negatives."""
    ratings, movies = get_dataset("movielens")[0]()
    raw = _load_cache(CACHES["phi4-mini (raw)"])
    ids = list(raw)
    X = np.stack([raw[i] for i in ids])
    centered = dict(zip(ids, X - X.mean(axis=0)))  # unsupervised: no labels used
    train_df, test_df = build_user_candidates(ratings, movies, seed=seed, negative_sampling="popularity")
    context = {
        "user_positives": user_positive_map(train_df),
        "pop_stats": popularity_stats(training_interactions(ratings, test_df)),
    }
    rows = []
    for name, fd in [("phi4-mini raw", raw), ("phi4-mini centered", centered), ("SBERT", _load_cache(CACHES["SBERT"]))]:
        result, _ = run_pipeline(train_df, test_df, (fd,), "lgbm", seed, "personalized_pop", context)
        rows.append({"Encoder": name, "NDCG@10": result["ndcg@10_mean"]})
    return pd.DataFrame(rows)


def affinity_only(dataset="movielens", seed=1):
    """Rank each user's test candidates by profile similarity alone (no scorer).

    Rank-based, so invariant to how compressed an encoder's cosine range is:
    it isolates whether the similarity *ordering* carries signal.
    """
    load_fn, prefix, _ = get_dataset(dataset)
    ratings, movies = load_fn()
    train_df, test_df = build_user_candidates(ratings, movies, seed=seed, negative_sampling="popularity")
    user_positives = user_positive_map(train_df)
    spaces = {"TF-IDF": build_tfidf_features(movies)[0]}
    spaces.update({name: _load_cache(prefix + cache) for name, cache in CACHES.items()})
    rows = []
    for name, fd in spaces.items():
        aff = affinity_features(test_df, user_positives, fd)
        for col, label in enumerate(("mean", "max")):
            result = evaluate_ranking(test_df, aff[:, col])
            rows.append({"Space": name, "Feature": label, "NDCG@10": result["ndcg@10_mean"]})
    return pd.DataFrame(rows).pivot(index="Space", columns="Feature", values="NDCG@10").reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis", choices=["geometry", "history", "centering", "affinity"])
    parser.add_argument("--dataset", default="movielens")
    args = parser.parse_args()
    if args.analysis == "geometry":
        out = geometry(args.dataset)
    elif args.analysis == "history":
        out = history_trend(args.dataset)
    elif args.analysis == "affinity":
        out = affinity_only(args.dataset)
    else:
        out = centering()
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
