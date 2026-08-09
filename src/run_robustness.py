"""Robustness / distribution-shift analysis: compares each pipeline's NDCG@10
under a random per-user split vs. a temporal split (train on earlier
interactions, test on each user's most recent ones) using the LGBM scorer.

Reuses the feature-set builder from run_experiment.py so cached SBERT/LLM
embeddings aren't recomputed.
"""

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.candidates import build_user_candidates
from src.dataset_registry import get_dataset, results_suffix
from src.embeddings import features_for_ids, hybrid_features_for_ids
from src.metrics import aggregate, evaluate_ranking
from src.pipeline import score_and_time, train_lgbm_ranker
from src.run_experiment import K, build_feature_sets

SEEDS = [1, 2, 3]
SPLIT_METHODS = ["random", "temporal"]
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_pipeline(train_df, test_df, feature_dicts):
    if len(feature_dicts) == 1:
        X_train = features_for_ids(train_df["movieId"], feature_dicts[0])
        X_test = features_for_ids(test_df["movieId"], feature_dicts[0])
    else:
        X_train = hybrid_features_for_ids(train_df["movieId"], *feature_dicts)
        X_test = hybrid_features_for_ids(test_df["movieId"], *feature_dicts)

    model = train_lgbm_ranker(train_df, X_train)
    scores, _ = score_and_time(model, X_test)
    result = evaluate_ranking(test_df, scores, k=K)
    ndcg, recall = result[f"ndcg@{K}_mean"], result[f"recall@{K}_mean"]

    del X_train, X_test, model, scores, result
    gc.collect()
    return ndcg, recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="movielens", choices=["movielens", "amazon", "yelp"]
    )
    args = parser.parse_args()

    load_fn, cache_prefix, label = get_dataset(args.dataset)
    print(f"Loading {label}...")
    ratings, movies = load_fn()

    feature_sets = build_feature_sets(movies, cache_prefix=cache_prefix)
    per_combo = {
        (name, split): {"ndcgs": [], "recalls": []}
        for name in feature_sets
        for split in SPLIT_METHODS
    }

    for seed in SEEDS:
        for split_method in SPLIT_METHODS:
            print(f"\n=== Seed {seed}, split={split_method} ===")
            train_df, test_df = build_user_candidates(
                ratings, movies, seed=seed, split_method=split_method
            )
            for name, spec in feature_sets.items():
                ndcg, recall = run_pipeline(train_df, test_df, spec["features"])
                r = per_combo[(name, split_method)]
                r["ndcgs"].append(ndcg)
                r["recalls"].append(recall)
                print(f"  {name}: NDCG@{K}={ndcg:.4f}  Recall@{K}={recall:.4f}")

    rows = []
    for name in feature_sets:
        random_ndcg = aggregate(per_combo[(name, "random")]["ndcgs"])
        temporal_ndcg = aggregate(per_combo[(name, "temporal")]["ndcgs"])
        random_recall = aggregate(per_combo[(name, "random")]["recalls"])
        temporal_recall = aggregate(per_combo[(name, "temporal")]["recalls"])
        rows.append(
            {
                "Pipeline": name,
                "NDCG@10_random_mean": random_ndcg["mean"],
                "NDCG@10_random_ci95": random_ndcg["ci95"],
                "NDCG@10_temporal_mean": temporal_ndcg["mean"],
                "NDCG@10_temporal_ci95": temporal_ndcg["ci95"],
                "NDCG@10_drop": random_ndcg["mean"] - temporal_ndcg["mean"],
                "Recall@10_random_mean": random_recall["mean"],
                "Recall@10_temporal_mean": temporal_recall["mean"],
                "Recall@10_drop": random_recall["mean"] - temporal_recall["mean"],
            }
        )

    results_table = pd.DataFrame(rows)
    results_path = RESULTS_DIR / f"results_table_robustness{results_suffix(args.dataset)}.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_table.to_csv(results_path, index=False)

    print("\n=== Robustness: random vs. temporal split ===")
    print(results_table.to_string(index=False))
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
