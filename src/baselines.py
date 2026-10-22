"""Non-content reference baselines: popularity and PureSVD matrix factorization.

Both are fit on training interactions only (every rating except the held-out
test positives, see user_features.training_interactions) and score the same
test candidate lists as the content pipelines, so their NDCG@10 is directly
comparable. They give the absolute numbers a reference point: how much of a
pipeline's ranking quality is just "rank popular items first" (popularity),
and how a standard collaborative model with no item text at all compares.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from src.candidates import POSITIVE_THRESHOLD

# Fixed a priori rather than tuned on the test lists; PureSVD is known to be
# fairly insensitive to rank in this range (Cremonesi et al., RecSys 2010).
SVD_RANK = 50


def popularity_scores(test_df: pd.DataFrame, pop_stats: pd.DataFrame) -> np.ndarray:
    """Score = log1p(training interaction count); unseen items score 0."""
    return pop_stats["log_pop"].reindex(test_df["movieId"].to_numpy()).fillna(0.0).to_numpy()


def puresvd_scores(
    test_df: pd.DataFrame, train_interactions: pd.DataFrame, rank: int = SVD_RANK, seed: int = 0
) -> np.ndarray:
    """PureSVD on the binarized (rating >= threshold) training user-item matrix.

    Score for (u, i) is the reconstructed matrix entry R V V^T [u, i]. Users
    or items absent from the training matrix score 0.
    """
    pos = train_interactions[train_interactions["rating"] >= POSITIVE_THRESHOLD]
    user_index = {u: k for k, u in enumerate(pos["userId"].unique())}
    item_index = {m: k for k, m in enumerate(pos["movieId"].unique())}
    R = csr_matrix(
        (
            np.ones(len(pos), dtype=np.float32),
            (pos["userId"].map(user_index).to_numpy(), pos["movieId"].map(item_index).to_numpy()),
        ),
        shape=(len(user_index), len(item_index)),
    )
    rank = min(rank, min(R.shape) - 1)
    svd = TruncatedSVD(n_components=rank, random_state=seed).fit(R)
    item_factors = svd.components_.T  # (n_items, rank)
    user_factors = R @ item_factors  # (n_users, rank)

    scores = np.zeros(len(test_df), dtype=np.float64)
    u_idx = test_df["userId"].map(user_index)
    i_idx = test_df["movieId"].map(item_index)
    known = u_idx.notna().to_numpy() & i_idx.notna().to_numpy()
    u = u_idx[known].astype(int).to_numpy()
    i = i_idx[known].astype(int).to_numpy()
    scores[known] = np.einsum("ij,ij->i", user_factors[u], item_factors[i])
    return scores
