"""Orchestrates the MVP experiment: run all pipelines across multiple seeds,
measure ranking quality (NDCG@10, Recall@10), stability across seeds, and
latency/cost, and write a results table.

For a given seed, all pipelines are evaluated on the *same* train/test
candidate split (controlled comparison) — only the item features differ.
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.candidates import build_user_candidates
from src.dataset_registry import get_dataset, results_suffix
from src.embeddings import (
    build_large_embedding_model_features,
    build_sbert_features,
    build_tfidf_features,
    build_true_llm_features,
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
    large_embed_dict, large_embed_latency = build_large_embedding_model_features(
        movies, cache_prefix=cache_prefix
    )

    print("Building phi4-mini embeddings via llama.cpp (mean-pooled hidden states)...")
    true_llm_dict, true_llm_latency = build_true_llm_features(movies, cache_prefix=cache_prefix)

    return {
        "Baseline (TF-IDF)": {"features": (tfidf_dict,), "embed_latency": tfidf_latency},
        "SBERT (all-MiniLM-L6-v2)": {"features": (sbert_dict,), "embed_latency": sbert_latency},
        "Large Embedding Model (mxbai-embed-large)": {
            "features": (large_embed_dict,),
            "embed_latency": large_embed_latency,
        },
        "LLM (phi4-mini, local)": {
            "features": (true_llm_dict,),
            "embed_latency": true_llm_latency,
        },
        "Hybrid (TF-IDF + phi4-mini)": {
            "features": (tfidf_dict, true_llm_dict),
            "embed_latency": tfidf_latency + true_llm_latency,
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
    ndcg, recall = result[f"ndcg@{K}_mean"], result[f"recall@{K}_mean"]

    # Explicit cleanup, matching run_robustness.py: this loop runs 50 fits
    # (5 seeds x 2 scorers x 5 pipelines) in one process, and the Hybrid
    # pipeline's 5072-dim arrays are large enough that letting them pile up
    # across iterations (rather than being freed as soon as each fit is
    # done) has caused severe swap thrashing in practice.
    del X_train, X_test, model, scores
    gc.collect()
    return ndcg, recall, score_latency


def _load_seed_checkpoint(checkpoint_path):
    if checkpoint_path.exists():
        return pd.read_csv(checkpoint_path)
    return pd.DataFrame(columns=["Seed", "Pipeline", "Scorer", "NDCG", "Recall", "ScoreLatency"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="movielens", choices=["movielens", "amazon", "yelp"]
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=len(SEEDS),
        help="Use only the first N of SEEDS -- a submission-deadline speed/rigor "
        "tradeoff for a specific dataset; note in the writeup which datasets used "
        "fewer than the full 5 seeds.",
    )
    args = parser.parse_args()
    active_seeds = SEEDS[: args.num_seeds]

    load_fn, cache_prefix, label = get_dataset(args.dataset)
    print(f"Loading {label}...")
    ratings, movies = load_fn()
    print(f"{len(ratings)} ratings, {len(movies)} rated items")

    feature_sets = build_feature_sets(movies, cache_prefix=cache_prefix)

    # Per-seed checkpointing: this loop runs 50 fits total and, before the
    # explicit gc.collect() fix, has crashed partway through from memory
    # pressure -- losing every seed computed so far, since results were only
    # ever written to disk once at the very end. Checkpoint each seed's rows
    # to disk as soon as it finishes, and skip any seed already checkpointed
    # on restart, so a crash loses at most one seed's work, not all of it.
    suffix = results_suffix(args.dataset)
    checkpoint_path = RESULTS_DIR / f"seed_checkpoint{suffix}.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_df = _load_seed_checkpoint(checkpoint_path)
    done_seeds = set(checkpoint_df["Seed"].unique()) if len(checkpoint_df) else set()

    for seed in active_seeds:
        if seed in done_seeds:
            print(f"\n=== Seed {seed} (skipped, already checkpointed) ===")
            continue
        print(f"\n=== Seed {seed} ===")
        train_df, test_df = build_user_candidates(ratings, movies, seed=seed)

        seed_rows = []
        for name, spec in feature_sets.items():
            for scorer in SCORERS:
                ndcg, recall, score_latency = run_pipeline(
                    train_df, test_df, spec["features"], scorer, seed
                )
                seed_rows.append(
                    {
                        "Seed": seed,
                        "Pipeline": name,
                        "Scorer": scorer,
                        "NDCG": ndcg,
                        "Recall": recall,
                        "ScoreLatency": score_latency,
                    }
                )
                print(f"  {name} [{scorer}]: NDCG@{K}={ndcg:.4f}  Recall@{K}={recall:.4f}")

        checkpoint_df = pd.concat([checkpoint_df, pd.DataFrame(seed_rows)], ignore_index=True)
        checkpoint_df.to_csv(checkpoint_path, index=False)

    per_pipeline = {
        (name, scorer): {
            "ndcgs": checkpoint_df[
                (checkpoint_df["Pipeline"] == name) & (checkpoint_df["Scorer"] == scorer)
            ]["NDCG"].tolist(),
            "recalls": checkpoint_df[
                (checkpoint_df["Pipeline"] == name) & (checkpoint_df["Scorer"] == scorer)
            ]["Recall"].tolist(),
            "latencies": checkpoint_df[
                (checkpoint_df["Pipeline"] == name) & (checkpoint_df["Scorer"] == scorer)
            ]["ScoreLatency"].tolist(),
        }
        for name in feature_sets
        for scorer in SCORERS
    }

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
    checkpoint_path.unlink(missing_ok=True)  # final results saved; stale checkpoint would confuse a future fresh run

    print(f"\n=== Results (stability across {len(active_seeds)} seeds) ===")
    print(results_table.to_string(index=False))
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
