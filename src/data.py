"""Download, cache, and load the MovieLens ml-latest-small dataset."""

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _ensure_downloaded(data_dir: Path = DATA_DIR) -> Path:
    """Download and extract ml-latest-small into data_dir if not already present."""
    extracted_dir = data_dir / "ml-latest-small"
    if (extracted_dir / "ratings.csv").exists() and (extracted_dir / "movies.csv").exists():
        return extracted_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(MOVIELENS_URL, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        zf.extractall(data_dir)

    if not (extracted_dir / "ratings.csv").exists():
        raise FileNotFoundError(f"Expected ratings.csv under {extracted_dir} after extraction")
    return extracted_dir


ITEM_TEXT_VARIANTS = ("title_genres", "title", "genres")


def load_movielens(data_dir: Path = DATA_DIR, item_text: str = "title_genres"):
    """Load ratings and movies, and build an item text corpus.

    `item_text` selects the text construction ("title_genres" is the main
    setting). The "title" and "genres" variants hold users, items, and labels
    fixed while moving corpus redundancy (genres-only text draws from ~20
    tokens, so items are highly redundant), giving a within-dataset test of
    the redundancy diagnostic.

    Returns:
        ratings: DataFrame[userId, movieId, rating, timestamp]
        movies: DataFrame[movieId, title, genres, text]  where `text` is the
            concatenated title + genres string used as the item's text feature
    """
    extracted_dir = _ensure_downloaded(data_dir)

    ratings = pd.read_csv(extracted_dir / "ratings.csv")
    movies = pd.read_csv(extracted_dir / "movies.csv")

    movies = movies.copy()
    movies["genres_text"] = movies["genres"].str.replace("|", " ", regex=False)
    if item_text not in ITEM_TEXT_VARIANTS:
        raise ValueError(f"Unknown item_text {item_text!r}; choose from {ITEM_TEXT_VARIANTS}")
    movies["text"] = {
        "title_genres": movies["title"] + " " + movies["genres_text"],
        "title": movies["title"],
        "genres": movies["genres_text"],
    }[item_text]

    # Keep only movies that actually have ratings, so candidate items always
    # have at least some interaction signal in the dataset.
    rated_movie_ids = set(ratings["movieId"].unique())
    movies = movies[movies["movieId"].isin(rated_movie_ids)].reset_index(drop=True)

    return ratings, movies[["movieId", "title", "genres", "text"]]
