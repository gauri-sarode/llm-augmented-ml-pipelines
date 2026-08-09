"""Maps a --dataset CLI value to its loader, cache namespace, and label.

Modules are imported lazily inside get_dataset() so that, e.g., running the
movielens experiment doesn't require the Yelp tarball to be present.
"""

import importlib

_DATASETS = {
    "movielens": {
        "module": "src.data",
        "loader": "load_movielens",
        "cache_prefix": "",
        "label": "MovieLens (ml-latest-small)",
    },
    "amazon": {
        "module": "src.data_amazon",
        "loader": "load_amazon",
        "cache_prefix": "amazon_",
        "label": "Amazon Reviews (All_Beauty)",
    },
    "yelp": {
        "module": "src.data_yelp",
        "loader": "load_yelp",
        "cache_prefix": "yelp_",
        "label": "Yelp Open Dataset (Boise, ID)",
    },
}


def get_dataset(name: str):
    """Returns (loader_fn, cache_prefix, label) for a dataset name."""
    if name not in _DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {list(_DATASETS)}")
    spec = _DATASETS[name]
    module = importlib.import_module(spec["module"])
    loader = getattr(module, spec["loader"])
    return loader, spec["cache_prefix"], spec["label"]


def results_suffix(name: str) -> str:
    """Filename suffix for a dataset's results/figures ("" for movielens, the
    original default, to keep existing filenames/tests unchanged)."""
    return "" if name == "movielens" else f"_{name}"
