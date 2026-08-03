"""Qualitative failure analysis: which users does the LLM pipeline help/hurt
most relative to the TF-IDF baseline, and what do their test-candidate movies
look like?

Uses seed=1 (random split) and the LGBM scorer, reusing cached embeddings.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.candidates import build_user_candidates
from src.data import load_movielens
from src.embeddings import build_llm_features, build_tfidf_features, features_for_ids
from src.metrics import ndcg_at_k
from src.pipeline import score_and_time, train_lgbm_ranker

SEED = 1
K = 10
NUM_EXAMPLES = 5
RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "failure_examples.csv"


def per_user_ndcg(test_df, scores):
    """Return {userId: ndcg_at_k} using the same groupby order as metrics.evaluate_ranking."""
    df = test_df.copy()
    df["score"] = np.asarray(scores, dtype=float)
    out = {}
    for user_id, group in df.groupby("userId", sort=False):
        out[user_id] = ndcg_at_k(group["score"].to_numpy(), group["relevance"].to_numpy(), K)
    return out


def describe_user_positives(test_df, movies, user_id):
    rows = test_df[(test_df["userId"] == user_id) & (test_df["relevance"] > 0)]
    titled = rows.merge(movies[["movieId", "title", "genres"]], on="movieId")
    return "; ".join(f"{t} [{g}]" for t, g in zip(titled["title"], titled["genres"]))


def main():
    print("Loading MovieLens (ml-latest-small)...")
    ratings, movies = load_movielens()

    tfidf_dict, _ = build_tfidf_features(movies)
    print("Loading mxbai-embed-large embeddings (from cache)...")
    llm_dict, _ = build_llm_features(movies)

    train_df, test_df = build_user_candidates(ratings, movies, seed=SEED)

    X_train_tfidf = features_for_ids(train_df["movieId"], tfidf_dict)
    X_test_tfidf = features_for_ids(test_df["movieId"], tfidf_dict)
    X_train_llm = features_for_ids(train_df["movieId"], llm_dict)
    X_test_llm = features_for_ids(test_df["movieId"], llm_dict)

    baseline_model = train_lgbm_ranker(train_df, X_train_tfidf)
    baseline_scores, _ = score_and_time(baseline_model, X_test_tfidf)

    llm_model = train_lgbm_ranker(train_df, X_train_llm)
    llm_scores, _ = score_and_time(llm_model, X_test_llm)

    baseline_ndcg = per_user_ndcg(test_df, baseline_scores)
    llm_ndcg = per_user_ndcg(test_df, llm_scores)

    diffs = pd.DataFrame(
        {
            "userId": list(baseline_ndcg.keys()),
            "baseline_ndcg": list(baseline_ndcg.values()),
            "llm_ndcg": [llm_ndcg[u] for u in baseline_ndcg],
        }
    )
    diffs["diff"] = diffs["llm_ndcg"] - diffs["baseline_ndcg"]

    regressions = diffs.sort_values("diff").head(NUM_EXAMPLES).copy()
    regressions["case_type"] = "llm_regression"
    gains = diffs.sort_values("diff", ascending=False).head(NUM_EXAMPLES).copy()
    gains["case_type"] = "llm_gain"

    examples = pd.concat([regressions, gains], ignore_index=True)
    examples["test_positive_movies"] = examples["userId"].apply(
        lambda u: describe_user_positives(test_df, movies, u)
    )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    examples.to_csv(RESULTS_PATH, index=False)

    print(f"\n=== Biggest LLM-pipeline regressions vs. TF-IDF baseline (seed={SEED}) ===")
    for _, row in regressions.iterrows():
        print(
            f"  user {row.userId}: baseline={row.baseline_ndcg:.3f} llm={row.llm_ndcg:.3f} "
            f"diff={row['diff']:+.3f}\n    positives: "
            f"{describe_user_positives(test_df, movies, row.userId)}"
        )

    print(f"\n=== Biggest LLM-pipeline gains vs. TF-IDF baseline (seed={SEED}) ===")
    for _, row in gains.iterrows():
        print(
            f"  user {row.userId}: baseline={row.baseline_ndcg:.3f} llm={row.llm_ndcg:.3f} "
            f"diff={row['diff']:+.3f}\n    positives: "
            f"{describe_user_positives(test_df, movies, row.userId)}"
        )

    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
