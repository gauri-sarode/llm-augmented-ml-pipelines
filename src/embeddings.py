"""Item embedding builders: TF-IDF, SBERT, and a local LLM (phi4-mini via Ollama).

Item features (title + genres embeddings) are content-based and independent
of the train/test split, so each embedding type is computed once per movieId
and cached to disk under data/cache/.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def _cache_paths(name):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}_ids.npy", CACHE_DIR / f"{name}_vectors.npy"


def _load_cache(name):
    ids_path, vecs_path = _cache_paths(name)
    if ids_path.exists() and vecs_path.exists():
        ids = np.load(ids_path)
        vecs = np.load(vecs_path)
        return dict(zip(ids.tolist(), vecs))
    return {}


def _save_cache(name, movie_id_to_vec):
    ids_path, vecs_path = _cache_paths(name)
    ids = np.array(list(movie_id_to_vec.keys()))
    vecs = np.stack([movie_id_to_vec[i] for i in ids])
    np.save(ids_path, ids)
    np.save(vecs_path, vecs)


def build_tfidf_features(movies: pd.DataFrame, max_features: int = 2000):
    """Fit TF-IDF over all item text. Returns ({movieId: dense vector}, avg_latency_sec=0)."""
    vectorizer = TfidfVectorizer(max_features=max_features)
    matrix = vectorizer.fit_transform(movies["text"]).toarray().astype(np.float32)
    return {mid: matrix[i] for i, mid in enumerate(movies["movieId"])}, 0.0


def build_sbert_features(
    movies: pd.DataFrame, model_name="all-MiniLM-L6-v2", use_cache=True, cache_prefix=""
):
    """Sentence-transformer embeddings, cached per movieId. Returns (dict, avg_latency_sec).

    `cache_prefix` namespaces the on-disk cache (e.g. "amazon_") so item ids
    from different datasets never collide in the same cache file.
    """
    cache_name = f"{cache_prefix}sbert_{model_name}"
    cached = _load_cache(cache_name) if use_cache else {}
    missing = [mid for mid in movies["movieId"] if mid not in cached]

    avg_latency = 0.0
    if missing:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        texts = movies.set_index("movieId").loc[missing, "text"].tolist()
        start = time.time()
        vectors = model.encode(texts, show_progress_bar=False)
        avg_latency = (time.time() - start) / len(missing)
        for mid, vec in zip(missing, vectors):
            cached[mid] = np.asarray(vec, dtype=np.float32)
        if use_cache:
            _save_cache(cache_name, cached)

    return {mid: cached[mid] for mid in movies["movieId"]}, avg_latency


def build_llm_features(
    movies: pd.DataFrame, model_name="mxbai-embed-large", use_cache=True, cache_prefix=""
):
    """Local LLM embeddings via Ollama's embed API, cached per movieId.

    Returns (dict, avg_latency_sec). Marginal API cost is $0 since this runs
    against a local Ollama server. Uses mxbai-embed-large (335M params) rather
    than a chat model like phi4-mini: Ollama's desktop-app-managed server only
    serves embeddings for models started in embedding mode, which chat models
    aren't by default. `cache_prefix` namespaces the on-disk cache (e.g.
    "amazon_") so item ids from different datasets never collide.
    """
    cache_name = f"{cache_prefix}llm_{model_name}"
    cached = _load_cache(cache_name) if use_cache else {}
    missing = [mid for mid in movies["movieId"] if mid not in cached]

    avg_latency = 0.0
    if missing:
        import ollama

        texts_by_id = movies.set_index("movieId")["text"]
        latencies = []
        for mid in missing:
            start = time.time()
            response = ollama.embed(model=model_name, input=texts_by_id[mid])
            latencies.append(time.time() - start)
            cached[mid] = np.asarray(response["embeddings"][0], dtype=np.float32)
        avg_latency = float(np.mean(latencies))
        if use_cache:
            _save_cache(cache_name, cached)

    return {mid: cached[mid] for mid in movies["movieId"]}, avg_latency


def features_for_ids(movie_ids, feature_dict):
    """Stack feature vectors for a list of movieIds, in order."""
    return np.stack([feature_dict[mid] for mid in movie_ids])


def hybrid_features_for_ids(movie_ids, *feature_dicts):
    """Concatenate multiple feature dicts' vectors for a list of movieIds, in order."""
    parts = [features_for_ids(movie_ids, fd) for fd in feature_dicts]
    return np.concatenate(parts, axis=1)
