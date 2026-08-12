"""Item embedding builders: TF-IDF, SBERT, a dedicated embedding model
(mxbai-embed-large via Ollama), and a genuine local LLM (phi4-mini, via
llama.cpp).

Item features (title + genres embeddings) are content-based and independent
of the train/test split, so each embedding type is computed once per movieId
and cached to disk under data/cache/.
"""

import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
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


def build_large_embedding_model_features(
    movies: pd.DataFrame, model_name="mxbai-embed-large", use_cache=True, cache_prefix=""
):
    """mxbai-embed-large embeddings via Ollama's embed API, cached per movieId.

    mxbai-embed-large is a dedicated 335M-parameter embedding model (BERT-style
    encoder) -- not an LLM. It's included as a second, honestly-labeled large
    embedding baseline alongside the true LLM arm (see build_true_llm_features).
    Returns (dict, avg_latency_sec). Marginal API cost is $0 since this runs
    against a local Ollama server. `cache_prefix` namespaces the on-disk cache
    (e.g. "amazon_") so item ids from different datasets never collide.
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


def _ollama_gguf_path(model_name):
    """Resolve the local GGUF blob path Ollama already downloaded for `model_name`.

    Shells out to `ollama show --modelfile` and parses the `FROM <path>` line,
    rather than hardcoding a blob hash, so this reproduces on any machine that
    has run `ollama pull {model_name}`.
    """
    result = subprocess.run(
        ["ollama", "show", model_name, "--modelfile"],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("FROM "):
            return line.removeprefix("FROM ").strip()
    raise RuntimeError(f"could not find a GGUF blob path for '{model_name}' in `ollama show`")


def build_true_llm_features(
    movies: pd.DataFrame,
    model_name="phi4-mini",
    server_url="http://127.0.0.1:11600",
    use_cache=True,
    cache_prefix="",
):
    """Mean-pooled hidden-state embeddings from a genuine local LLM (phi4-mini,
    3.8B-parameter decoder-only), cached per movieId.

    Ollama gates its /api/embed endpoint per-model capability, and phi4-mini
    (a chat model) isn't flagged for embeddings -- confirmed this holds even
    against a bare `ollama serve` instance, not just the desktop app. Instead
    this calls a llama.cpp `llama-server` (`brew install llama.cpp`) started
    directly against phi4-mini's GGUF blob with `--embeddings --pooling mean`,
    which bypasses Ollama's gate and mean-pools the model's last-layer hidden
    states into a single vector -- the standard way to repurpose a decoder-only
    LLM as an embedder (cf. SGPT, Muennighoff et al. 2022). Start the server
    with e.g.:

        llama-server -m "$(python -c 'from src.embeddings import _ollama_gguf_path; \
            print(_ollama_gguf_path("phi4-mini"))')" \
            --embeddings --pooling mean --port 11600 --host 127.0.0.1 \
            -c 4096 -np 1 -ub 1536 -b 1536

    Returns (dict, avg_latency_sec). Marginal cost is $0 (local inference).
    `cache_prefix` namespaces the on-disk cache (e.g. "amazon_") so item ids
    from different datasets never collide.
    """
    cache_name = f"{cache_prefix}truellm_{model_name}"
    cached = _load_cache(cache_name) if use_cache else {}
    missing = [mid for mid in movies["movieId"] if mid not in cached]

    avg_latency = 0.0
    if missing:
        texts_by_id = movies.set_index("movieId")["text"]
        latencies = []
        for i, mid in enumerate(missing):
            start = time.time()
            # Retry transient 503s: llama-server's /health can report ready
            # a moment before its single slot actually is, which otherwise
            # crashes the very first request of a run.
            for attempt in range(5):
                response = requests.post(
                    f"{server_url}/embedding", json={"content": texts_by_id[mid]}, timeout=120
                )
                if response.status_code != 503:
                    break
                time.sleep(2 * (attempt + 1))
            response.raise_for_status()
            latencies.append(time.time() - start)
            vec = response.json()[0]["embedding"][0]
            cached[mid] = np.asarray(vec, dtype=np.float32)
            # Print periodically (cheap) and checkpoint to disk periodically
            # (more expensive -- rewrites the whole cache): this loop can run
            # for hours on long-review datasets (Yelp), and both external
            # monitoring (stdout must keep growing, or a log-watching
            # supervisor will mistake real progress for a stall) and
            # crash/restart recovery (should resume near where it left off,
            # not from scratch) depend on this.
            if (i + 1) % 50 == 0:
                print(f"  ...{i + 1}/{len(missing)} phi4-mini embeddings done")
            if use_cache and (i + 1) % 200 == 0:
                _save_cache(cache_name, cached)
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
