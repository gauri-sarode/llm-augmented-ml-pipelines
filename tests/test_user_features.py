import numpy as np
import pandas as pd

from src.baselines import popularity_scores, puresvd_scores
from src.candidates import build_user_candidates
from src.user_features import (
    affinity_features,
    build_scorer_features,
    popularity_stats,
    training_interactions,
    user_positive_map,
)

FEATURES = {
    1: np.array([1.0, 0.0], dtype=np.float32),
    2: np.array([0.0, 1.0], dtype=np.float32),
    3: np.array([1.0, 1.0], dtype=np.float32),
    4: np.array([-1.0, 0.0], dtype=np.float32),
}


def test_affinity_leave_one_out_excludes_self():
    rows = pd.DataFrame({"userId": [7, 7, 7], "movieId": [1, 2, 4]})
    positives = {7: np.array([1, 2])}
    out = affinity_features(rows, positives, FEATURES)
    # Item 1 is a training positive: compared only against item 2 (cos 0), not itself.
    np.testing.assert_allclose(out[0], [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(out[1], [0.0, 0.0], atol=1e-6)
    # Item 4 is opposite item 1 (cos -1) and orthogonal to item 2 (cos 0).
    np.testing.assert_allclose(out[2], [-0.5, 0.0], atol=1e-6)


def test_affinity_single_positive_profile_is_empty_for_itself():
    rows = pd.DataFrame({"userId": [7, 7], "movieId": [3, 1]})
    out = affinity_features(rows, {7: np.array([3])}, FEATURES)
    np.testing.assert_allclose(out[0], [0.0, 0.0])
    np.testing.assert_allclose(out[1], [np.sqrt(0.5)] * 2, atol=1e-6)


def test_affinity_preserves_row_order_across_interleaved_users():
    rows = pd.DataFrame({"userId": [8, 7, 8], "movieId": [1, 1, 2]})
    out = affinity_features(rows, {7: np.array([2]), 8: np.array([1])}, FEATURES)
    np.testing.assert_allclose(out[:, 1], [0.0, 0.0, 0.0], atol=1e-6)  # row 0 is self-only
    rows = pd.DataFrame({"userId": [8, 7], "movieId": [3, 3]})
    out = affinity_features(rows, {7: np.array([2]), 8: np.array([4])}, FEATURES)
    np.testing.assert_allclose(out[:, 1], [-np.sqrt(0.5), np.sqrt(0.5)], atol=1e-6)


def test_training_interactions_drop_only_test_positives():
    ratings = pd.DataFrame(
        {"userId": [1, 1, 2], "movieId": [10, 11, 10], "rating": [5.0, 4.0, 3.0], "timestamp": 0}
    )
    test_df = pd.DataFrame({"userId": [1, 1], "movieId": [11, 99], "relevance": [4.0, 0.0]})
    train = training_interactions(ratings, test_df)
    assert set(zip(train["userId"], train["movieId"])) == {(1, 10), (2, 10)}
    stats = popularity_stats(train)
    assert np.isclose(stats.loc[10, "log_pop"], np.log1p(2))
    assert 11 not in stats.index


def _synthetic(num_users=30, num_movies=60, seed=0):
    rng = np.random.default_rng(seed)
    movies = pd.DataFrame({"movieId": np.arange(num_movies)})
    rows = []
    for u in range(num_users):
        for m in rng.choice(num_movies, size=12, replace=False):
            rows.append((u, m, float(rng.choice([2.0, 4.0, 5.0])), int(rng.integers(1e6))))
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    feats = {m: rng.normal(size=8).astype(np.float32) for m in movies["movieId"]}
    return ratings, movies, feats


def test_scorer_feature_shapes_and_baselines_run():
    ratings, movies, feats = _synthetic()
    for negatives in ("uniform", "popularity"):
        train_df, test_df = build_user_candidates(
            ratings, movies, seed=0, num_negatives_test=10, negative_sampling=negatives
        )
        train_inter = training_interactions(ratings, test_df)
        pop = popularity_stats(train_inter)
        ctx = {"user_positives": user_positive_map(train_df), "pop_stats": pop}
        assert build_scorer_features(test_df, (feats,), "item", **ctx).shape[1] == 8
        assert build_scorer_features(test_df, (feats,), "personalized", **ctx).shape[1] == 10
        X = build_scorer_features(test_df, (feats, feats), "personalized_pop", **ctx)
        assert X.shape == (len(test_df), 8 + 8 + 2 + 2 + 2)
        assert np.isfinite(X).all()
        assert popularity_scores(test_df, pop).shape == (len(test_df),)
        assert np.isfinite(puresvd_scores(test_df, train_inter, rank=5)).all()


def test_chunk_words_and_mean_pool():
    from src.embeddings import _mean_pool_chunks, chunk_words

    assert chunk_words("a b c d e", 2) == ["a b", "c d", "e"]
    assert chunk_words("", 2) == [""]
    pooled = _mean_pool_chunks(
        ["a b c", "d"], 2, lambda chunks: [[float(len(c))] for c in chunks]
    )
    np.testing.assert_allclose(pooled, [[2.0], [1.0]])  # mean(len("a b"), len("c")); len("d")


def test_affinity_handles_string_user_ids():
    rows = pd.DataFrame({"userId": ["b", "a", "b"], "movieId": [3, 3, 4]})
    out = affinity_features(rows, {"a": np.array([1]), "b": np.array([2])}, FEATURES)
    np.testing.assert_allclose(out[:, 1], [np.sqrt(0.5), np.sqrt(0.5), 0.0], atol=1e-6)
