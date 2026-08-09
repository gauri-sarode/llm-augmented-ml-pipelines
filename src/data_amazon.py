"""Download, cache, and load an Amazon Reviews 2023 (McAuley Lab) category.

Unlike MovieLens, Amazon items have no curated short metadata field, so this
loader exercises the "long free-text review" regime the paper's dataset
section promises: each item's `text` feature is built from its own review
text (title + body, several reviews concatenated), not just a product title.

We use the McAuley Lab "5-core" interaction subset (users and items with
>= 5 interactions each) for ratings -- the standard, pre-filtered benchmark
protocol for this dataset family, analogous to how `ml-latest-small` is
itself already a curated small subset of full MovieLens -- then join in
review text (from the raw per-category review file) and product
title/category (from the raw per-category metadata file) for the items that
appear in that interaction subset.
"""

import json
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "amazon"
DEFAULT_CATEGORY = "All_Beauty"
HF_BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main"
MAX_REVIEWS_PER_ITEM = 10
MAX_CHARS_PER_REVIEW = 500


def _download(url: str, dest: Path):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=300, stream=True)
    response.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    tmp.rename(dest)


def _ensure_raw_files(category: str, data_dir: Path):
    ratings_path = data_dir / f"ratings_5core_{category}.csv"
    reviews_path = data_dir / f"review_{category}.jsonl"
    meta_path = data_dir / f"meta_{category}.jsonl"

    _download(f"{HF_BASE}/benchmark/5core/rating_only/{category}.csv", ratings_path)
    _download(f"{HF_BASE}/raw/review_categories/{category}.jsonl", reviews_path)
    _download(f"{HF_BASE}/raw/meta_categories/meta_{category}.jsonl", meta_path)
    return ratings_path, reviews_path, meta_path


def _build_item_text(reviews_path: Path, item_ids: set):
    """Concatenate up to MAX_REVIEWS_PER_ITEM review title+text per item."""
    snippets = {mid: [] for mid in item_ids}
    with open(reviews_path) as f:
        for line in f:
            d = json.loads(line)
            pid = d.get("parent_asin")
            if pid not in snippets or len(snippets[pid]) >= MAX_REVIEWS_PER_ITEM:
                continue
            title = (d.get("title") or "").strip()
            body = (d.get("text") or "").strip()
            snippet = f"{title} {body}".strip()[:MAX_CHARS_PER_REVIEW]
            if snippet:
                snippets[pid].append(snippet)
    return {mid: " ".join(parts) for mid, parts in snippets.items()}


def _build_item_meta(meta_path: Path, item_ids: set):
    """Look up product title + category for each item from the raw meta file."""
    titles, categories = {}, {}
    with open(meta_path) as f:
        for line in f:
            d = json.loads(line)
            pid = d.get("parent_asin")
            if pid not in item_ids:
                continue
            titles[pid] = (d.get("title") or pid).strip()
            cats = d.get("categories") or []
            categories[pid] = " ".join(cats) if cats else (d.get("main_category") or "")
    return titles, categories


def load_amazon(category: str = DEFAULT_CATEGORY, data_dir: Path = DATA_DIR):
    """Load an Amazon Reviews 2023 category as (ratings, movies) matching the
    MovieLens loader's schema.

    Returns:
        ratings: DataFrame[userId, movieId, rating, timestamp]
        movies: DataFrame[movieId, title, genres, text] where `text` is
            product title + category + concatenated review text
    """
    processed_ratings = data_dir / f"ratings_processed_{category}.csv"
    processed_movies = data_dir / f"movies_processed_{category}.csv"
    if processed_ratings.exists() and processed_movies.exists():
        return pd.read_csv(processed_ratings), pd.read_csv(processed_movies)

    ratings_path, reviews_path, meta_path = _ensure_raw_files(category, data_dir)

    ratings = pd.read_csv(ratings_path)
    ratings = ratings.rename(
        columns={"user_id": "userId", "parent_asin": "movieId", "rating": "rating"}
    )[["userId", "movieId", "rating", "timestamp"]]

    item_ids = set(ratings["movieId"].unique())
    review_text = _build_item_text(reviews_path, item_ids)
    titles, categories = _build_item_meta(meta_path, item_ids)

    rows = []
    for mid in sorted(item_ids):
        title = titles.get(mid, mid)
        genres = categories.get(mid, "")
        text = f"{title} {genres} {review_text.get(mid, '')}".strip()
        rows.append({"movieId": mid, "title": title, "genres": genres, "text": text})
    movies = pd.DataFrame(rows)

    data_dir.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(processed_ratings, index=False)
    movies.to_csv(processed_movies, index=False)
    return ratings, movies
