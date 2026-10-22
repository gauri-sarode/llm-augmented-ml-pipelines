"""Personalized (user-item) features for the scoring model.

Feeding the scorer item-level features only (the "item" mode) makes the
ranker learn one global score per item and ranked every user's candidates in the
same order -- effectively a content-predicted popularity model, which cannot
show whether a richer item representation helps *personalized* ranking.

This module adds, per (user, candidate) row:

- Content-affinity features, computed separately in each pipeline's own
  embedding space: the mean and max cosine similarity between the candidate
  and the user's *training* positives. This is where item-representation
  quality can matter per user. For a training-positive row the item itself
  is excluded from its own user profile (leave-one-out), so the scorer never
  trains on a self-similarity of 1.0 that it would never see at test time.
- Optionally, popularity features (log interaction count and mean rating),
  computed from training interactions only -- every rating except the
  held-out test positives. These are identical across pipelines, so they
  shift every pipeline equally and leave the feature-source comparison
  controlled.
"""

import numpy as np
import pandas as pd

FEATURE_MODES = ("item", "personalized", "personalized_pop")


def training_interactions(ratings: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """All ratings except the held-out (user, item) test positives."""
    test_pos = test_df.loc[test_df["relevance"] > 0, ["userId", "movieId"]]
    merged = ratings.merge(test_pos, on=["userId", "movieId"], how="left", indicator=True)
    return merged.loc[merged["_merge"] == "left_only", ratings.columns]


def popularity_stats(train_interactions: pd.DataFrame) -> pd.DataFrame:
    """Per-item log1p(interaction count) and mean rating, from training interactions."""
    stats = train_interactions.groupby("movieId")["rating"].agg(["size", "mean"])
    return pd.DataFrame(
        {"log_pop": np.log1p(stats["size"]), "mean_rating": stats["mean"]}, index=stats.index
    )


def _normalized_matrix(feature_dict):
    ids = list(feature_dict.keys())
    mat = np.stack([feature_dict[i] for i in ids]).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat /= np.where(norms == 0, 1.0, norms)
    return {mid: i for i, mid in enumerate(ids)}, mat


def affinity_features(rows: pd.DataFrame, user_positives: dict, feature_dict) -> np.ndarray:
    """Mean/max cosine similarity of each row's item to its user's training positives.

    Args:
        rows: DataFrame[userId, movieId, ...]; output rows follow its order.
        user_positives: {userId: array of training-positive movieIds}.
        feature_dict: {movieId: vector} for one embedding space.

    Returns:
        (len(rows), 2) float32 array [mean_sim, max_sim]. A row whose user has
        no other training positive (leave-one-out leaves an empty profile)
        gets 0.0 for both.
    """
    index, mat = _normalized_matrix(feature_dict)
    out = np.zeros((len(rows), 2), dtype=np.float32)
    user_ids = rows["userId"].to_numpy()
    movie_ids = rows["movieId"].to_numpy()

    # Group rows by user via integer codes (ids may be strings, e.g. Amazon).
    codes = pd.factorize(user_ids)[0]
    order = np.argsort(codes, kind="stable")
    boundaries = np.flatnonzero(np.diff(codes[order])) + 1
    for row_idx in np.split(order, boundaries):
        if len(row_idx) == 0:
            continue
        pos_ids = user_positives.get(user_ids[row_idx[0]])
        if pos_ids is None or len(pos_ids) == 0:
            continue
        cand_ids = movie_ids[row_idx]
        sims = mat[[index[m] for m in cand_ids]] @ mat[[index[m] for m in pos_ids]].T
        # Leave-one-out: a candidate never counts as similar to itself.
        self_mask = cand_ids[:, None] == np.asarray(pos_ids)[None, :]
        counts = (~self_mask).sum(axis=1)
        valid = counts > 0
        local = np.zeros((len(row_idx), 2), dtype=np.float32)
        local[valid, 0] = np.where(self_mask, 0.0, sims)[valid].sum(axis=1) / counts[valid]
        local[valid, 1] = np.where(self_mask, -np.inf, sims)[valid].max(axis=1)
        out[row_idx] = local
    return out


def user_positive_map(train_df: pd.DataFrame) -> dict:
    """{userId: array of that user's training-positive movieIds}."""
    pos = train_df[train_df["relevance"] > 0]
    return {u: g["movieId"].to_numpy() for u, g in pos.groupby("userId", sort=False)}


def build_scorer_features(rows, feature_dicts, mode, user_positives=None, pop_stats=None):
    """Scorer input matrix for `rows` under a feature mode.

    - "item": the item vector(s) only (ablation; hybrid concatenates).
    - "personalized": item vector(s) + [mean_sim, max_sim] per embedding space.
    - "personalized_pop": the above + [log_pop, mean_rating] (training-only).
    """
    if mode not in FEATURE_MODES:
        raise ValueError(f"Unknown feature mode {mode!r}; choose from {FEATURE_MODES}")
    ids = rows["movieId"]
    parts = [np.stack([fd[m] for m in ids]).astype(np.float32) for fd in feature_dicts]
    if mode in ("personalized", "personalized_pop"):
        parts += [affinity_features(rows, user_positives, fd) for fd in feature_dicts]
    if mode == "personalized_pop":
        pop = pop_stats.reindex(ids.to_numpy())
        # Items with no training interaction: zero count, neutral mean rating.
        pop = pop.fillna({"log_pop": 0.0, "mean_rating": pop_stats["mean_rating"].mean()})
        parts.append(pop.to_numpy(dtype=np.float32))
    return np.concatenate(parts, axis=1)
