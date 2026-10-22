"""Load a city subset of the Yelp Open Dataset (yelp_academic_dataset_*.json,
inside yelp_dataset.tar) as (ratings, movies) matching the MovieLens schema.

The full dataset (150K+ businesses, ~7M reviews, ~5GB of review text) is too
large to run the full pipeline suite against locally, so -- mirroring
ml-latest-small already being a curated small subset of full MovieLens -- we
restrict to a single city. Boise, ID was picked because it lands in the same
"thousands of items" scale as the MovieLens item catalog without being one of
the dataset's largest metros (Philadelphia, Tucson, Tampa), which would make
the local LLM-embedding pass impractically slow.

Like `data_amazon.py`, each item's `text` feature is built from its own
review text (long free-text reviews), unlike MovieLens's short title+genre
metadata -- this is the dataset's point of contrast in the paper.
"""

import json
import tarfile
from pathlib import Path

import pandas as pd

from src.candidates import POSITIVE_THRESHOLD

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "yelp"
TAR_PATH = DATA_DIR / "yelp_dataset.tar"
DEFAULT_CITY = "Boise"
DEFAULT_STATE = "ID"
MIN_BUSINESS_REVIEWS = 5
MAX_REVIEWS_PER_ITEM = 10
MAX_CHARS_PER_REVIEW = 500


def _load_businesses(tf: tarfile.TarFile, city: str, state: str):
    f = tf.extractfile("yelp_academic_dataset_business.json")
    businesses = {}
    for line in f:
        d = json.loads(line)
        if (
            d.get("city") == city
            and d.get("state") == state
            and (d.get("review_count") or 0) >= MIN_BUSINESS_REVIEWS
        ):
            businesses[d["business_id"]] = {
                "name": d.get("name") or d["business_id"],
                "categories": d.get("categories") or "",
            }
    return businesses


def _load_reviews(tf: tarfile.TarFile, business_ids: set):
    """All reviews for the city's businesses, as rating rows plus raw text."""
    f = tf.extractfile("yelp_academic_dataset_review.json")
    rows = []
    for line in f:
        d = json.loads(line)
        if d.get("business_id") not in business_ids:
            continue
        text = (d.get("text") or "").strip()[:MAX_CHARS_PER_REVIEW]
        rows.append((d["user_id"], d["business_id"], d["stars"], d.get("date", ""), text))
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating", "date", "text"])


def _item_snippets(reviews: pd.DataFrame, min_positives: int = 2):
    """Up to MAX_REVIEWS_PER_ITEM review texts per business, excluding every
    review by an evaluable user (one with >= min_positives positive ratings,
    matching build_user_candidates). Item text therefore never contains a
    review whose rating is a held-out relevance label, under either split.
    """
    positives = reviews[reviews["rating"] >= POSITIVE_THRESHOLD].groupby("userId").size()
    evaluable = set(positives[positives >= min_positives].index)
    source = reviews[~reviews["userId"].isin(evaluable) & (reviews["text"] != "")]
    return source.groupby("movieId")["text"].apply(lambda t: t.iloc[:MAX_REVIEWS_PER_ITEM].tolist())


def load_yelp(
    city: str = DEFAULT_CITY, state: str = DEFAULT_STATE, data_dir: Path = DATA_DIR
):
    """Load a Yelp Open Dataset city subset as (ratings, movies).

    Returns:
        ratings: DataFrame[userId, movieId, rating, timestamp] -- timestamp is
            the review's `date` field converted to a Unix timestamp (seconds),
            used for the temporal-split robustness analysis exactly like
            MovieLens's rating timestamp.
        movies: DataFrame[movieId, title, genres, text] where `text` is
            business name + categories + concatenated review text
    """
    # "_v2": item text excludes evaluable users' reviews (earlier caches leaked them).
    processed_ratings = data_dir / f"ratings_processed_{city}_v2.csv"
    processed_movies = data_dir / f"movies_processed_{city}_v2.csv"
    if processed_ratings.exists() and processed_movies.exists():
        return pd.read_csv(processed_ratings), pd.read_csv(processed_movies)

    if not TAR_PATH.exists():
        raise FileNotFoundError(
            f"{TAR_PATH} not found -- download the Yelp Open Dataset tarball "
            "(requires registering at https://www.yelp.com/dataset) and place "
            "it there before running this loader."
        )

    tf = tarfile.open(TAR_PATH)
    businesses = _load_businesses(tf, city, state)
    business_ids = set(businesses.keys())
    reviews = _load_reviews(tf, business_ids)
    tf.close()
    review_snippets = _item_snippets(reviews)

    ratings = reviews[["userId", "movieId", "rating", "date"]].copy()
    ratings["timestamp"] = pd.to_datetime(ratings["date"]).astype("int64") // 10**9
    ratings = ratings[["userId", "movieId", "rating", "timestamp"]]

    rows = []
    for bid in sorted(business_ids):
        info = businesses[bid]
        review_text = " ".join(review_snippets.get(bid, []))
        text = f"{info['name']} {info['categories']} {review_text}".strip()
        rows.append(
            {"movieId": bid, "title": info["name"], "genres": info["categories"], "text": text}
        )
    movies = pd.DataFrame(rows)

    data_dir.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(processed_ratings, index=False)
    movies.to_csv(processed_movies, index=False)
    return ratings, movies
