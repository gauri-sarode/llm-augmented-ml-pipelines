"""Orchestrates the MVP experiment: run all pipelines across multiple seeds,
measure ranking quality (NDCG@10, Recall@10), stability across seeds, and
latency/cost, and write a results table.

For a given seed, all pipelines are evaluated on the *same* train/test
candidate split (controlled comparison) — only the item features differ.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.candidates import build_user_candidates
from src.dataset_registry import get_dataset, results_suffix
from src.embeddings import (
    build_llm_features,
    build_sbert_features,
    build_tfidf_features,
    features_for_ids,
    hybrid_features_for_ids,
)
from src.metrics import aggregate, evaluate_ranking
from src.pipeline import score_and_time, train_lgbm_ranker, train_mlp_scorer

SEEDS = [1, 2, 3, 4, 5]
K = 10
SCORERS = ["lgbm", "mlp"]
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def build_feature_sets(movies, cache_prefix=""):
    """Build item feature dicts for all pipelines once (shared across seeds)."""
    tfidf_dict, tfidf_latency = build_tfidf_features(movies)

    print("Building SBERT embeddings (all-MiniLM-L6-v2)...")
    sbert_dict, sbert_latency = build_sbert_features(movies, cache_prefix=cache_prefix)

    print("Building mxbai-embed-large embeddings via Ollama (cached after first run)...")
    llm_dict, llm_latency = build_llm_features(movies, cache_prefix=cache_prefix)

    return {
        "Baseline (TF-IDF)": {"features": (tfidf_dict,), "embed_latency": tfidf_latency},
        "SBERT (all-MiniLM-L6-v2)": {"features": (sbert_dict,), "embed_latency": sbert_latency},
        "LLM (mxbai-embed-large, local)": {"features": (llm_dict,), "embed_latency": llm_latency},
        "Hybrid (TF-IDF + mxbai-embed-large)": {
            "features": (tfidf_dict, llm_dict),
            "embed_latency": tfidf_latency + llm_latency,
        },
    }


def run_pipeline(train_df, test_df, feature_dicts, scorer, seed):
    if len(feature_dicts) == 1:
        X_train = features_for_ids(train_df["movieId"], feature_dicts[0])
        X_test = features_for_ids(test_df["movieId"], feature_dicts[0])
    else:
        X_train = hybrid_features_for_ids(train_df["movieId"], *feature_dicts)
        X_test = hybrid_features_for_ids(test_df["movieId"], *feature_dicts)

    if scorer == "lgbm":
        model = train_lgbm_ranker(train_df, X_train)
    elif scorer == "mlp":
        model = train_mlp_scorer(train_df, X_train, seed=seed)
    else:
        raise ValueError(f"Unknown scorer: {scorer}")

    scores, score_latency = score_and_time(model, X_test)
    result = evaluate_ranking(test_df, scores, k=K)
    return result[f"ndcg@{K}_mean"], result[f"recall@{K}_mean"], score_latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="movielens", choices=["movielens", "amazon", "yelp"]
    )
    args = parser.parse_args()

    load_fn, cache_prefix, label = get_dataset(args.dataset)
    print(f"Loading {label}...")
    ratings, movies = load_fn()
    print(f"{len(ratings)} ratings, {len(movies)} rated items")

    feature_sets = build_feature_sets(movies, cache_prefix=cache_prefix)
    per_pipeline = {
        (name, scorer): {"ndcgs": [], "recalls": [], "latencies": []}
        for name in feature_sets
        for scorer in SCORERS
    }

    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        train_df, test_df = build_user_candidates(ratings, movies, seed=seed)

        for name, spec in feature_sets.items():
            for scorer in SCORERS:
                ndcg, recall, score_latency = run_pipeline(
                    train_df, test_df, spec["features"], scorer, seed
                )
                r = per_pipeline[(name, scorer)]
                r["ndcgs"].append(ndcg)
                r["recalls"].append(recall)
                r["latencies"].append(score_latency)
                print(f"  {name} [{scorer}]: NDCG@{K}={ndcg:.4f}  Recall@{K}={recall:.4f}")

    rows = []
    for name, spec in feature_sets.items():
        for scorer in SCORERS:
            r = per_pipeline[(name, scorer)]
            ndcg_stats = aggregate(r["ndcgs"])
            recall_stats = aggregate(r["recalls"])
            rows.append(
                {
                    "Pipeline": name,
                    "Scorer": scorer,
                    f"NDCG@{K}_mean": ndcg_stats["mean"],
                    f"NDCG@{K}_std": ndcg_stats["std"],
                    f"NDCG@{K}_ci95": ndcg_stats["ci95"],
                    f"Recall@{K}_mean": recall_stats["mean"],
                    f"Recall@{K}_std": recall_stats["std"],
                    f"Recall@{K}_ci95": recall_stats["ci95"],
                    "EmbedLatency_ms_per_item": spec["embed_latency"] * 1000,
                    "ScoreLatency_ms_per_query": float(np.mean(r["latencies"])) * 1000,
                    "MarginalCost_USD": 0.0,  # all pipelines run locally, no hosted API calls
                }
            )

    results_table = pd.DataFrame(rows)
    results_path = RESULTS_DIR / f"results_table{results_suffix(args.dataset)}.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_table.to_csv(results_path, index=False)

    print("\n=== Results (stability across 5 seeds) ===")
    print(results_table.to_string(index=False))
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
