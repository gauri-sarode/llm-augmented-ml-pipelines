"""Fast, offline end-to-end smoke test: candidates -> embeddings -> pipeline -> metrics.

Uses synthetic ratings/movies and the TF-IDF feature path only (no network
calls, no SBERT/Ollama downloads) so this stays fast and hermetic. The
SBERT/LLM embedding paths are exercised separately by the full
run_experiment.py run against real data.
"""

import time

import numpy as np
import pandas as pd

from src.candidates import build_user_candidates
from src.embeddings import build_tfidf_features, features_for_ids
from src.metrics import evaluate_ranking
from src.pipeline import score_and_time, train_lgbm_ranker


def _make_synthetic_data(num_users=20, num_movies=50, seed=0):
    rng = np.random.default_rng(seed)

    movies = pd.DataFrame(
        {
            "movieId": np.arange(1, num_movies + 1),
            "title": [f"Movie {i}" for i in range(1, num_movies + 1)],
            "genres": ["Action Comedy" for _ in range(num_movies)],
        }
    )
    movies["text"] = movies["title"] + " " + movies["genres"]

    rows = []
    for user_id in range(1, num_users + 1):
        rated_movies = rng.choice(num_movies, size=15, replace=False) + 1
        ratings_for_user = rng.choice([2.0, 3.0, 4.0, 4.5, 5.0], size=15)
        for movie_id, rating in zip(rated_movies, ratings_for_user):
            rows.append((user_id, movie_id, rating))
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating"])
    return ratings, movies


def test_full_pipeline_smoke():
    start = time.time()

    ratings, movies = _make_synthetic_data()

    train_df, test_df = build_user_candidates(
        ratings, movies, seed=0, num_negatives_train=3, num_negatives_test=10
    )
    assert len(train_df) > 0
    assert len(test_df) > 0

    feature_dict, avg_latency = build_tfidf_features(movies, max_features=50)
    assert avg_latency == 0.0  # TF-IDF is not a timed model call

    X_train = features_for_ids(train_df["movieId"], feature_dict)
    X_test = features_for_ids(test_df["movieId"], feature_dict)

    model = train_lgbm_ranker(train_df, X_train, n_estimators=10, num_leaves=8)
    scores, score_latency = score_and_time(model, X_test)
    assert score_latency >= 0.0
    assert len(scores) == len(test_df)

    result = evaluate_ranking(test_df, scores, k=5)

    num_test_users = test_df["userId"].nunique()
    assert len(result["ndcg_per_user"]) == num_test_users
    assert len(result["recall_per_user"]) == num_test_users
    assert not np.isnan(result["ndcg_per_user"]).any()
    assert not np.isnan(result["recall_per_user"]).any()
    assert 0.0 <= result["ndcg@5_mean"] <= 1.0
    assert 0.0 <= result["recall@5_mean"] <= 1.0

    assert time.time() - start < 30, "smoke test should complete in well under 30s"
