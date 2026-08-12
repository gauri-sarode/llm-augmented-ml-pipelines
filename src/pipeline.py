"""LightGBM ranker and MLP scorer training/scoring, grouped by user."""

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _group_sizes(df: pd.DataFrame):
    """Row-count per user, in the order users first appear (must match row order)."""
    return df.groupby("userId", sort=False).size().to_numpy()


def train_lgbm_ranker(train_df: pd.DataFrame, X_train: np.ndarray, **lgbm_kwargs):
    """Train an LGBMRanker (lambdarank) grouped by user.

    Args:
        train_df: DataFrame[userId, movieId, relevance]; row order must match X_train's rows.
        X_train: feature matrix, one row per (userId, movieId) candidate.
    """
    params = dict(
        objective="lambdarank",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=64,
        verbosity=-1,
        # Capped rather than left at "use all cores": LightGBM's per-thread
        # histogram buffers scale with thread count x feature count. Past
        # thrash episodes traced to `llama-server` staying resident during
        # scoring and to a memory leak across the 50-fit run_experiment.py
        # loop (both now fixed) rather than to this thread count itself.
        # Bumped 4->6 (of 10 physical cores) under a submission deadline for
        # a straightforward speedup with no effect on results, now that the
        # actual leak/contention sources are fixed rather than papered over
        # with a lower thread count.
        num_threads=6,
    )
    params.update(lgbm_kwargs)

    model = lgb.LGBMRanker(**params)
    group = _group_sizes(train_df)
    # LGBMRanker's lambdarank objective requires integer labels (used as an
    # index into label_gain); MovieLens has half-star ratings like 4.5, so we
    # round for the training label while keeping the continuous relevance
    # (used for NDCG gain) unchanged in metrics.py.
    labels = train_df["relevance"].round().astype(int).to_numpy()
    model.fit(X_train, labels, group=group)
    return model


def train_mlp_scorer(train_df: pd.DataFrame, X_train: np.ndarray, seed: int = 0, **mlp_kwargs):
    """Train a scaled MLPRegressor pointwise scorer on continuous relevance.

    Unlike LGBMRanker, this is not listwise/grouped — it regresses each
    (userId, movieId) row's continuous relevance directly, and the resulting
    scores are used for per-user ranking downstream exactly like the LGBM
    ranker's scores (via score_and_time / metrics.evaluate_ranking).
    """
    params = dict(
        hidden_layer_sizes=(64, 32),
        max_iter=300,
        early_stopping=True,
        random_state=seed,
    )
    params.update(mlp_kwargs)

    # copy=False: scale in place rather than allocating a full duplicate of
    # X_train -- identical output, just avoids doubling peak memory for the
    # widest (Hybrid, 5072-dim) feature matrix, which has been the recurring
    # site of severe swap thrashing on this machine.
    model = Pipeline([("scaler", StandardScaler(copy=False)), ("mlp", MLPRegressor(**params))])
    model.fit(X_train, train_df["relevance"].to_numpy())
    return model


def score_and_time(model, X_test):
    """Predict scores for X_test. Returns (scores, avg_latency_sec_per_row)."""
    start = time.time()
    scores = model.predict(X_test)
    elapsed = time.time() - start
    avg_latency = elapsed / max(len(X_test), 1)
    return scores, avg_latency
