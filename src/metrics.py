"""Per-user NDCG@K / Recall@K and aggregate statistics with a 95% CI.

Unlike a naive implementation that ranks the entire test set as a single
list, these functions operate on one user's candidate list at a time and are
meant to be averaged across users via `evaluate_ranking`.
"""

import numpy as np
import pandas as pd


def ndcg_at_k(scores, relevance, k=10):
    """NDCG@k for a single ranked list (one user's candidates).

    Args:
        scores: predicted scores, one per candidate item.
        relevance: graded relevance (e.g. rating, 0 if irrelevant), same
            length/order as scores.
    """
    scores = np.asarray(scores, dtype=float)
    relevance = np.asarray(relevance, dtype=float)
    k = min(k, len(scores))
    if k == 0:
        return 0.0

    order = np.argsort(-scores, kind="stable")[:k]
    gains = 2 ** relevance[order] - 1
    discounts = np.log2(np.arange(len(order)) + 2)
    dcg = np.sum(gains / discounts)

    ideal_order = np.argsort(-relevance, kind="stable")[:k]
    ideal_gains = 2 ** relevance[ideal_order] - 1
    ideal_dcg = np.sum(ideal_gains / discounts)

    if ideal_dcg == 0:
        return 0.0
    return float(dcg / ideal_dcg)


def recall_at_k(scores, relevance, k=10, relevance_threshold=0.0):
    """Recall@k: fraction of relevant items (relevance > threshold) retrieved in the top k."""
    scores = np.asarray(scores, dtype=float)
    relevance = np.asarray(relevance, dtype=float)
    is_relevant = relevance > relevance_threshold
    total_relevant = is_relevant.sum()
    if total_relevant == 0:
        return 0.0

    k = min(k, len(scores))
    order = np.argsort(-scores, kind="stable")[:k]
    retrieved_relevant = is_relevant[order].sum()
    return float(retrieved_relevant / total_relevant)


def aggregate(values):
    """mean/std/95% CI half-width for an array of per-user metric values."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean()) if n else 0.0
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    ci95 = float(1.96 * std / np.sqrt(n)) if n > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": ci95}


def evaluate_ranking(test_df: pd.DataFrame, scores, k: int = 10):
    """Compute per-user NDCG@k / Recall@k over test_df, aligned row-wise with scores.

    Args:
        test_df: DataFrame[userId, movieId, relevance]
        scores: predicted scores, same length/order as test_df rows

    Returns:
        dict with per-user arrays plus aggregate mean/std/95% CI for both metrics.
    """
    df = test_df.copy()
    df["score"] = np.asarray(scores, dtype=float)

    ndcgs, recalls = [], []
    for _, group in df.groupby("userId", sort=False):
        s, r = group["score"].to_numpy(), group["relevance"].to_numpy()
        ndcgs.append(ndcg_at_k(s, r, k))
        recalls.append(recall_at_k(s, r, k))

    ndcgs, recalls = np.array(ndcgs), np.array(recalls)
    result = {"ndcg_per_user": ndcgs, "recall_per_user": recalls}
    for stat, v in aggregate(ndcgs).items():
        result[f"ndcg@{k}_{stat}"] = v
    for stat, v in aggregate(recalls).items():
        result[f"recall@{k}_{stat}"] = v
    return result
