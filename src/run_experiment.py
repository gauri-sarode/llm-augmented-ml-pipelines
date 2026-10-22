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

from src.baselines import popularity_scores, puresvd_scores
from src.candidates import build_user_candidates
from src.dataset_registry import get_dataset, results_suffix
from src.embeddings import (
    build_large_embedding_model_features,
    build_sbert_features,
    build_tfidf_features,
    build_trained_llm_embedder_features,
    build_true_llm_features,
)
from src.metrics import aggregate, evaluate_ranking
from src.pipeline import score_and_time, train_lgbm_ranker, train_mlp_scorer
from src.user_features import (
    FEATURE_MODES,
    build_scorer_features,
    popularity_stats,
    training_interactions,
    user_positive_map,
)

SEEDS = [1, 2, 3, 4, 5]
K = 10
SCORERS = ["lgbm", "mlp"]
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def build_feature_sets(movies, cache_prefix="", only=None):
    """Build item feature dicts for all pipelines once (shared across seeds).

    `only`: optional list of name substrings; builds just those pipelines.
    """
    if only is not None:
        return _build_subset(movies, cache_prefix, only)
    tfidf_dict, tfidf_latency = build_tfidf_features(movies)

    print("Building SBERT embeddings (all-MiniLM-L6-v2)...")
    sbert_dict, sbert_latency = build_sbert_features(movies, cache_prefix=cache_prefix)

    print("Building mxbai-embed-large embeddings via Ollama (cached after first run)...")
    large_embed_dict, large_embed_latency = build_large_embedding_model_features(
        movies, cache_prefix=cache_prefix
    )

    print("Building phi4-mini embeddings via llama.cpp (mean-pooled hidden states)...")
    true_llm_dict, true_llm_latency = build_true_llm_features(movies, cache_prefix=cache_prefix)

    print("Building Qwen3-Embedding-4B embeddings via llama.cpp (trained LLM embedder)...")
    trained_dict, trained_latency = build_trained_llm_embedder_features(
        movies, cache_prefix=cache_prefix
    )

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
        "LLM embedder (Qwen3-Embedding-4B)": {
            "features": (trained_dict,),
            "embed_latency": trained_latency,
        },
        "Hybrid (TF-IDF + phi4-mini)": {
            "features": (tfidf_dict, true_llm_dict),
            "embed_latency": tfidf_latency + true_llm_latency,
        },
    }


def _build_subset(movies, cache_prefix, only):
    builders = {
        "Baseline (TF-IDF)": lambda: build_tfidf_features(movies),
        "SBERT (all-MiniLM-L6-v2)": lambda: build_sbert_features(movies, cache_prefix=cache_prefix),
    }
    chosen = {n: b for n, b in builders.items() if any(o in n for o in only)}
    if not chosen:
        raise ValueError(f"--pipelines {only} matches none of {list(builders)}")
    out = {}
    for name, build in chosen.items():
        fd, latency = build()
        out[name] = {"features": (fd,), "embed_latency": latency}
    return out


def run_pipeline(train_df, test_df, feature_dicts, scorer, seed, mode, context):
    X_train = build_scorer_features(train_df, feature_dicts, mode, **context)
    X_test = build_scorer_features(test_df, feature_dicts, mode, **context)

    if scorer == "lgbm":
        model = train_lgbm_ranker(train_df, X_train)
    elif scorer == "mlp":
        model = train_mlp_scorer(train_df, X_train, seed=seed)
    else:
        raise ValueError(f"Unknown scorer: {scorer}")

    scores, score_latency = score_and_time(model, X_test)
    result = evaluate_ranking(test_df, scores, k=K)

    # Explicit cleanup: this loop runs up to 60 fits
    # (5 seeds x 2 scorers x 6 pipelines) in one process, and the Hybrid
    # pipeline's 5072-dim arrays are large enough that letting them pile up
    # across iterations (rather than being freed as soon as each fit is
    # done) has caused severe swap thrashing in practice.
    del X_train, X_test, model, scores
    gc.collect()
    return result, score_latency


def run_baselines(test_df, train_inter, pop_stats, seed):
    """Non-content reference baselines; scorer-independent, so Scorer="-"."""
    return {
        "Popularity": evaluate_ranking(test_df, popularity_scores(test_df, pop_stats), k=K),
        "PureSVD (MF)": evaluate_ranking(
            test_df, puresvd_scores(test_df, train_inter, seed=seed), k=K
        ),
    }


def _load_checkpoint(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["Seed", "Pipeline", "Scorer", "userId", "NDCG", "Recall"])


def _per_user_rows(seed, name, scorer, result):
    return pd.DataFrame(
        {
            "Seed": seed,
            "Pipeline": name,
            "Scorer": scorer,
            "userId": result["users"],
            "NDCG": result["ndcg_per_user"],
            "Recall": result["recall_per_user"],
        }
    )


def run_tag(dataset, mode, negatives, split="random"):
    """Filename suffix identifying one experimental configuration."""
    tag = results_suffix(dataset)
    if split != "random":
        tag += f"_split-{split}"
    if mode != "item":
        tag += f"_{mode}"
    if negatives != "uniform":
        tag += f"_neg-{negatives}"
    return tag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="movielens",
        choices=["movielens", "movielens-title", "movielens-genres", "amazon", "yelp"],
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=None,
        help="Restrict to these pipeline names (substring match), e.g. TF-IDF SBERT. "
        "Only the embeddings those pipelines need are built.",
    )
    parser.add_argument(
        "--features",
        default="personalized_pop",
        choices=FEATURE_MODES,
        help="Scorer input: item vectors only (ablation), + user-item affinity, "
        "or + affinity and training-only popularity features.",
    )
    parser.add_argument("--negatives", default="uniform", choices=["uniform", "popularity"])
    parser.add_argument(
        "--split",
        default="random",
        choices=["random", "temporal"],
        help="temporal holds out each user's most recent positives (distribution shift).",
    )
    parser.add_argument("--scorers", nargs="+", default=SCORERS, choices=SCORERS)
    parser.add_argument("--num-seeds", type=int, default=len(SEEDS))
    args = parser.parse_args()
    active_seeds = SEEDS[: args.num_seeds]

    load_fn, cache_prefix, label = get_dataset(args.dataset)
    print(f"Loading {label}...")
    ratings, movies = load_fn()
    print(f"{len(ratings)} ratings, {len(movies)} rated items")
    print(f"Features: {args.features}, negatives: {args.negatives}, split: {args.split}")

    feature_sets = build_feature_sets(movies, cache_prefix=cache_prefix, only=args.pipelines)

    # Per-seed checkpointing of per-user results: a crash loses at most one
    # seed's work, and the per-user rows feed the paired significance tests
    # (src/stats.py) as well as the seed-level aggregates below.
    tag = run_tag(args.dataset, args.features, args.negatives, args.split)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = RESULTS_DIR / f"per_user{tag}.csv"
    per_user = _load_checkpoint(checkpoint_path)
    done_seeds = set(per_user["Seed"].unique()) if len(per_user) else set()

    for seed in active_seeds:
        if seed in done_seeds:
            print(f"\n=== Seed {seed} (skipped, already checkpointed) ===")
            continue
        print(f"\n=== Seed {seed} ===")
        train_df, test_df = build_user_candidates(
            ratings,
            movies,
            seed=seed,
            negative_sampling=args.negatives,
            split_method=args.split,
        )
        train_inter = training_interactions(ratings, test_df)
        pop_stats = popularity_stats(train_inter)
        context = {"user_positives": user_positive_map(train_df), "pop_stats": pop_stats}

        frames = []
        for name, result in run_baselines(test_df, train_inter, pop_stats, seed).items():
            frames.append(_per_user_rows(seed, name, "-", result))
            print(f"  {name}: NDCG@{K}={result[f'ndcg@{K}_mean']:.4f}")
        for name, spec in feature_sets.items():
            for scorer in args.scorers:
                result, _ = run_pipeline(
                    train_df, test_df, spec["features"], scorer, seed, args.features, context
                )
                frames.append(_per_user_rows(seed, name, scorer, result))
                print(
                    f"  {name} [{scorer}]: NDCG@{K}={result[f'ndcg@{K}_mean']:.4f}  "
                    f"Recall@{K}={result[f'recall@{K}_mean']:.4f}"
                )

        per_user = pd.concat([per_user, *frames], ignore_index=True)
        per_user.to_csv(checkpoint_path, index=False)

    # Seed-level means, then mean +/- 95% CI across seeds (stability).
    seed_means = (
        per_user.groupby(["Pipeline", "Scorer", "Seed"], sort=False)[["NDCG", "Recall"]]
        .mean()
        .reset_index()
    )
    embed_latency = {name: spec["embed_latency"] for name, spec in feature_sets.items()}
    rows = []
    for (name, scorer), g in seed_means.groupby(["Pipeline", "Scorer"], sort=False):
        ndcg_stats, recall_stats = aggregate(g["NDCG"]), aggregate(g["Recall"])
        rows.append(
            {
                "Pipeline": name,
                "Scorer": scorer,
                "Seeds": len(g),
                f"NDCG@{K}_mean": ndcg_stats["mean"],
                f"NDCG@{K}_ci95": ndcg_stats["ci95"],
                f"Recall@{K}_mean": recall_stats["mean"],
                f"Recall@{K}_ci95": recall_stats["ci95"],
                "EmbedLatency_ms_per_item": embed_latency.get(name, 0.0) * 1000,
            }
        )
    results_table = pd.DataFrame(rows)
    results_path = RESULTS_DIR / f"results_table{tag}.csv"
    results_table.to_csv(results_path, index=False)

    print(f"\n=== Results ({len(active_seeds)} seeds) ===")
    print(results_table.to_string(index=False))
    print(f"\nSaved to {results_path}; per-user rows in {checkpoint_path}")


if __name__ == "__main__":
    main()
