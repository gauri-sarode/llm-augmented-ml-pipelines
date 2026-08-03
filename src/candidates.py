"""Per-user train/test split of positive interactions and negative candidate sampling."""

import numpy as np
import pandas as pd

POSITIVE_THRESHOLD = 4.0


def build_user_candidates(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    seed: int,
    test_frac: float = 0.2,
    num_negatives_train: int = 4,
    num_negatives_test: int = 50,
    min_positives: int = 2,
    split_method: str = "random",
):
    """Build per-user train/test candidate sets with graded relevance labels.

    For each user with >= min_positives positive interactions (rating >=
    POSITIVE_THRESHOLD), split their positives into train/test, then sample
    negative candidates (movies the user never rated) for both splits.

    Args:
        split_method: "random" (default) shuffles each user's positives before
            splitting. "temporal" sorts by the ratings `timestamp` column and
            holds out the user's most recent interactions for test — used for
            the distribution-shift robustness analysis. Negative sampling is
            seeded/random either way.

    Returns:
        train_df, test_df: DataFrame[userId, movieId, relevance] where
            relevance is the raw rating for positives and 0.0 for sampled
            negatives. Each user's rows in test_df form one ranked list for
            per-user NDCG/Recall evaluation.
    """
    if split_method not in ("random", "temporal"):
        raise ValueError(f"Unknown split_method: {split_method}")

    rng = np.random.default_rng(seed)
    all_movie_ids = movies["movieId"].to_numpy()

    positives = ratings[ratings["rating"] >= POSITIVE_THRESHOLD]
    train_rows = []
    test_rows = []

    for user_id, group in positives.groupby("userId"):
        if split_method == "temporal":
            group = group.sort_values("timestamp")
        pos_movie_ids = group["movieId"].to_numpy()
        pos_ratings = group["rating"].to_numpy()
        if len(pos_movie_ids) < min_positives:
            continue

        if split_method == "random":
            order = rng.permutation(len(pos_movie_ids))
            pos_movie_ids = pos_movie_ids[order]
            pos_ratings = pos_ratings[order]

        n_test = max(1, int(round(len(pos_movie_ids) * test_frac)))
        n_test = min(n_test, len(pos_movie_ids) - 1)  # keep >=1 positive for train

        if split_method == "temporal":
            # Earliest interactions train the model; most recent are held out
            # for test, simulating prediction under a temporal distribution shift.
            train_pos_ids, test_pos_ids = pos_movie_ids[:-n_test], pos_movie_ids[-n_test:]
            train_pos_ratings, test_pos_ratings = pos_ratings[:-n_test], pos_ratings[-n_test:]
        else:
            test_pos_ids, train_pos_ids = pos_movie_ids[:n_test], pos_movie_ids[n_test:]
            test_pos_ratings, train_pos_ratings = pos_ratings[:n_test], pos_ratings[n_test:]

        rated_ids = ratings.loc[ratings["userId"] == user_id, "movieId"].to_numpy()
        candidate_pool = np.setdiff1d(all_movie_ids, rated_ids, assume_unique=False)
        if len(candidate_pool) == 0:
            continue

        def sample_negatives(n):
            n = min(n, len(candidate_pool))
            return rng.choice(candidate_pool, size=n, replace=False)

        train_neg_ids = sample_negatives(num_negatives_train * max(1, len(train_pos_ids)))
        test_neg_ids = sample_negatives(num_negatives_test)

        for mid, r in zip(train_pos_ids, train_pos_ratings):
            train_rows.append((user_id, mid, r))
        for mid in train_neg_ids:
            train_rows.append((user_id, mid, 0.0))

        for mid, r in zip(test_pos_ids, test_pos_ratings):
            test_rows.append((user_id, mid, r))
        for mid in test_neg_ids:
            test_rows.append((user_id, mid, 0.0))

    train_df = pd.DataFrame(train_rows, columns=["userId", "movieId", "relevance"])
    test_df = pd.DataFrame(test_rows, columns=["userId", "movieId", "relevance"])
    return train_df, test_df
