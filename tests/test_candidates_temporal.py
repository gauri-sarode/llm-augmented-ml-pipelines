import numpy as np
import pandas as pd

from src.candidates import build_user_candidates


def _make_synthetic_data(num_users=10, num_movies=30, seed=0):
    rng = np.random.default_rng(seed)

    movies = pd.DataFrame({"movieId": np.arange(1, num_movies + 1)})
    movies["text"] = "placeholder"

    rows = []
    for user_id in range(1, num_users + 1):
        rated_movies = rng.choice(num_movies, size=10, replace=False) + 1
        ratings_for_user = rng.choice([2.0, 3.0, 4.0, 4.5, 5.0], size=10)
        # Strictly increasing timestamps per rating, in a random order relative
        # to the rating value, so "temporal" and "random" splits can disagree.
        timestamps = rng.permutation(np.arange(1_000_000, 1_000_010))
        for movie_id, rating, ts in zip(rated_movies, ratings_for_user, timestamps):
            rows.append((user_id, movie_id, rating, ts))
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    return ratings, movies


def test_temporal_split_holds_out_most_recent_per_user():
    ratings, movies = _make_synthetic_data()

    train_df, test_df = build_user_candidates(
        ratings, movies, seed=0, split_method="temporal", num_negatives_train=2, num_negatives_test=5
    )

    ratings_by_key = {
        (row.userId, row.movieId): row.timestamp for row in ratings.itertuples()
    }

    for user_id in train_df["userId"].unique():
        train_positive_ts = [
            ratings_by_key[(user_id, mid)]
            for mid, rel in zip(
                train_df.loc[train_df["userId"] == user_id, "movieId"],
                train_df.loc[train_df["userId"] == user_id, "relevance"],
            )
            if rel > 0
        ]
        test_positive_ts = [
            ratings_by_key[(user_id, mid)]
            for mid, rel in zip(
                test_df.loc[test_df["userId"] == user_id, "movieId"],
                test_df.loc[test_df["userId"] == user_id, "relevance"],
            )
            if rel > 0
        ]
        if not train_positive_ts or not test_positive_ts:
            continue
        # Every train positive must be strictly earlier than every test positive.
        assert max(train_positive_ts) < min(test_positive_ts), (
            f"user {user_id}: train positives are not strictly earlier than test positives"
        )


def test_temporal_split_is_deterministic_for_positive_positions():
    # The positive/negative split boundary shouldn't depend on the negative-sampling seed.
    ratings, movies = _make_synthetic_data()

    train_a, test_a = build_user_candidates(ratings, movies, seed=1, split_method="temporal")
    train_b, test_b = build_user_candidates(ratings, movies, seed=2, split_method="temporal")

    positives_a = set(
        zip(train_a.loc[train_a["relevance"] > 0, "userId"], train_a.loc[train_a["relevance"] > 0, "movieId"])
    )
    positives_b = set(
        zip(train_b.loc[train_b["relevance"] > 0, "userId"], train_b.loc[train_b["relevance"] > 0, "movieId"])
    )
    assert positives_a == positives_b
