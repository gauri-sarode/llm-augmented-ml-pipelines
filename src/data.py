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


def load_movielens(data_dir: Path = DATA_DIR):
    """Load ratings and movies, and build an item text corpus (title + genres).

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
    movies["text"] = movies["title"] + " " + movies["genres_text"]

    # Keep only movies that actually have ratings, so candidate items always
    # have at least some interaction signal in the dataset.
    rated_movie_ids = set(ratings["movieId"].unique())
    movies = movies[movies["movieId"].isin(rated_movie_ids)].reset_index(drop=True)

    return ratings, movies[["movieId", "title", "genres", "text"]]
