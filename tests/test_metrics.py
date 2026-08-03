import numpy as np
import pandas as pd
import pytest

from src.metrics import aggregate, evaluate_ranking, ndcg_at_k, recall_at_k


def test_ndcg_at_k_perfect_ranking_is_one():
    # Top-scored item is the only relevant one -> ideal order -> NDCG == 1.
    scores = [0.9, 0.1, 0.05]
    relevance = [5, 0, 0]
    assert ndcg_at_k(scores, relevance, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_worst_case_ranking():
    # Relevant item (relevance=5) ranked last out of 3 -> hand-computed NDCG.
    scores = [0.05, 0.1, 0.9]  # order by score desc: idx2, idx1, idx0
    relevance = [5, 0, 0]
    # dcg: only idx0 (relevance 5) contributes, at rank 3 -> gain=31, discount=log2(4)=2
    # idcg: idx0 at rank 1 -> gain=31, discount=log2(2)=1
    expected = (31 / 2) / (31 / 1)
    assert ndcg_at_k(scores, relevance, k=3) == pytest.approx(expected)


def test_ndcg_at_k_no_relevant_items_is_zero():
    assert ndcg_at_k([0.9, 0.5], [0, 0], k=2) == 0.0


def test_ndcg_at_k_truncates_to_k():
    # Only top-1 considered; relevant item is second by score -> NDCG should be 0.
    scores = [0.9, 0.1]
    relevance = [0, 5]
    assert ndcg_at_k(scores, relevance, k=1) == pytest.approx(0.0)


def test_recall_at_k_all_relevant_retrieved():
    scores = [0.9, 0.8, 0.1, 0.05]
    relevance = [5, 4, 0, 0]
    assert recall_at_k(scores, relevance, k=2) == pytest.approx(1.0)


def test_recall_at_k_partial_retrieval():
    # Top-2 by score are idx0 (relevant) and idx2 (irrelevant); 1 of 2 relevant retrieved.
    scores = [0.9, 0.1, 0.5, 0.05]
    relevance = [5, 4, 0, 0]
    assert recall_at_k(scores, relevance, k=2) == pytest.approx(0.5)


def test_recall_at_k_no_relevant_items_is_zero():
    assert recall_at_k([0.9, 0.5], [0, 0], k=2) == 0.0


def test_aggregate_known_values():
    result = aggregate([0.0, 1.0])
    assert result["mean"] == pytest.approx(0.5)
    assert result["std"] == pytest.approx(0.70710678)
    assert result["ci95"] == pytest.approx(0.98, abs=1e-2)


def test_aggregate_constant_values_zero_std():
    result = aggregate([1.0, 1.0, 1.0])
    assert result["mean"] == pytest.approx(1.0)
    assert result["std"] == pytest.approx(0.0)
    assert result["ci95"] == pytest.approx(0.0)


def test_evaluate_ranking_groups_per_user():
    # Two users, each with their own candidate list; verify per-user metrics
    # match direct ndcg_at_k/recall_at_k calls and aggregate is their mean.
    test_df = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [10, 11, 12, 20, 21, 22],
            "relevance": [5, 0, 0, 0, 4, 0],
        }
    )
    scores = [0.9, 0.1, 0.05, 0.9, 0.1, 0.05]  # user1: perfect; user2: worst-case

    result = evaluate_ranking(test_df, scores, k=3)

    expected_user1_ndcg = ndcg_at_k([0.9, 0.1, 0.05], [5, 0, 0], k=3)
    expected_user2_ndcg = ndcg_at_k([0.9, 0.1, 0.05], [0, 4, 0], k=3)

    assert result["ndcg_per_user"] == pytest.approx([expected_user1_ndcg, expected_user2_ndcg])
    assert result["ndcg@3_mean"] == pytest.approx(
        np.mean([expected_user1_ndcg, expected_user2_ndcg])
    )
    assert "recall@3_mean" in result
    assert "ndcg@3_ci95" in result
