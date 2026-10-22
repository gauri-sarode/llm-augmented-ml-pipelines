"""Like-for-like embedding latency benchmark for SBERT, mxbai-embed-large,
phi4-mini, and Qwen3-Embedding-4B.

Comparing SBERT's batched, whole-corpus `.encode()` against unbatched
single-request LLM calls inflates the gap by an order of magnitude. Here every
model embeds the same fixed item sample, under the same batch sizes, on the
same machine, after a warmup, with full-text coverage (chunk-and-pool for the
two length-limited encoders, exactly as the experiments do). Rounds rotate
model order so slow drift (thermal, background load) doesn't systematically
favor one model; we report the median over rounds of per-item latency.

Run only on an otherwise idle machine, with Ollama serving mxbai-embed-large
and two llama-servers, each with 4 parallel slots (-c 8192 -np 4 -ub 2048):
phi4-mini (--pooling mean) on :11600 and Qwen3-Embedding-4B (--pooling last)
on :11601.

Usage: python -m src.latency_benchmark --dataset movielens
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_registry import get_dataset, results_suffix
from src.embeddings import MXBAI_CHUNK_WORDS, SBERT_CHUNK_WORDS, _mean_pool_chunks

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SAMPLE_SIZE = 256
BATCH_SIZES = (1, 32)
ROUNDS = 3


def _sbert_encoder():
    import torch
    from sentence_transformers import SentenceTransformer

    # Same accelerator class as llama.cpp / Ollama (Metal on Apple silicon).
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    return lambda texts: _mean_pool_chunks(
        texts, SBERT_CHUNK_WORDS, lambda c: model.encode(c, batch_size=len(c), show_progress_bar=False)
    )


def _mxbai_encoder():
    import ollama

    return lambda texts: _mean_pool_chunks(
        texts, MXBAI_CHUNK_WORDS, lambda c: ollama.embed(model="mxbai-embed-large", input=c)["embeddings"]
    )


def _llama_server_encoder(server_url):
    """llama-server /v1/embeddings; a list input is spread over the server's
    parallel slots (-np 4), the server-side analogue of batched encoding."""

    def encode(texts):
        response = requests.post(f"{server_url}/v1/embeddings", json={"input": texts}, timeout=600)
        response.raise_for_status()
        return [d["embedding"] for d in response.json()["data"]]

    return encode


def benchmark(texts, encoders):
    rows = []
    names = list(encoders)
    for name in names:  # warmup: load weights, compile kernels
        encoders[name](texts[:4])
    for r in range(ROUNDS):
        order = names[r % len(names) :] + names[: r % len(names)]
        for batch_size in BATCH_SIZES:
            for name in order:
                start = time.perf_counter()
                for i in range(0, len(texts), batch_size):
                    encoders[name](texts[i : i + batch_size])
                per_item_ms = (time.perf_counter() - start) * 1000 / len(texts)
                rows.append({"Round": r, "Model": name, "BatchSize": batch_size, "ms_per_item": per_item_ms})
                print(f"  round {r} batch {batch_size:>2} {name}: {per_item_ms:.2f} ms/item", flush=True)
    raw = pd.DataFrame(rows)
    summary = raw.groupby(["Model", "BatchSize"])["ms_per_item"].median().unstack("BatchSize")
    summary.columns = [f"ms_per_item_batch{b}" for b in summary.columns]
    sbert = summary.loc["SBERT (all-MiniLM-L6-v2)"]
    for col in list(summary.columns):
        summary[col.replace("ms_per_item", "rel_to_sbert")] = summary[col] / sbert[col]
    return raw, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="movielens", choices=["movielens", "amazon", "yelp"])
    args = parser.parse_args()

    load_fn, _, _ = get_dataset(args.dataset)
    _, movies = load_fn()
    sample = movies.sample(n=min(SAMPLE_SIZE, len(movies)), random_state=0)["text"].tolist()

    encoders = {
        "SBERT (all-MiniLM-L6-v2)": _sbert_encoder(),
        "Large Embedding Model (mxbai-embed-large)": _mxbai_encoder(),
        "LLM (phi4-mini, local)": _llama_server_encoder("http://127.0.0.1:11600"),
        "LLM embedder (Qwen3-Embedding-4B)": _llama_server_encoder("http://127.0.0.1:11601"),
    }
    raw, summary = benchmark(sample, encoders)
    suffix = results_suffix(args.dataset)
    raw.to_csv(RESULTS_DIR / f"latency_raw{suffix}.csv", index=False)
    summary.to_csv(RESULTS_DIR / f"latency_summary{suffix}.csv")
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
