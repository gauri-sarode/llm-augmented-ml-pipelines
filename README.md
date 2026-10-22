# Evaluating LLM-Augmented ML Pipelines: Performance, Stability, and Cost

An empirical study of when LLM-based item features help, hurt, or are
unnecessary inside a recommendation-ranking ML pipeline — measured on
ranking quality, stability across seeds, robustness under distribution
shift, and latency/cost, not just a single accuracy number.

## Task

Three datasets, each cast as implicit-feedback, per-user ranking:
MovieLens (`ml-latest-small`, short title+genre item text, 610 users),
Amazon Reviews (`All_Beauty` 5-core, long free-form review text, 253
users), and Yelp (Boise, ID subset, long free-form review text, 10,537
users).

MovieLens specifics:
- Positive interaction: rating >= 4.0
- Per user: hold out a slice of positives for test, sample negative candidates
  from unrated movies (random split for main results; a temporal split for
  the robustness analysis)
- Item features come from embedding `title + genres`
- Two scoring models (`LGBMRanker` and an MLP), each grouped/evaluated per
  user
- Metrics (NDCG@10, Recall@10) are computed **per user** and averaged with a
  95% CI across users (and across seeds, for stability)

## Pipelines

| Pipeline | Item features |
|---|---|
| Baseline (TF-IDF) | classical TF-IDF over title+genres |
| SBERT | `all-MiniLM-L6-v2` local sentence embeddings (22M params, small, not an LLM) |
| Large Embedding Model (mxbai-embed-large) | local dedicated embedding model (335M params) via Ollama's `/api/embed`, $0 marginal cost — still not an LLM, kept as a second large-embedding data point |
| LLM (phi4-mini) | mean-pooled last-hidden-layer embeddings from a genuine 3.8B-parameter decoder-only LLM, served locally via `llama.cpp`, $0 marginal cost |
| Hybrid | TF-IDF + phi4-mini concatenated |

Each is scored by both `LGBMRanker` (lambdarank) and a scikit-learn MLP, to
check conclusions aren't an artifact of one scorer's inductive biases.

We initially tried `phi4-mini` (already pulled locally via Ollama) for the
LLM pipeline directly through Ollama's `/api/embed`, but Ollama gates that
endpoint per model capability and `phi4-mini`'s manifest isn't flagged for
embeddings — confirmed this holds against a bare `ollama serve` instance too,
not just the desktop app. Instead we point `llama.cpp`'s own server
(`llama-server --embeddings --pooling mean`) directly at the same GGUF blob
Ollama already downloaded, which has no such gate and mean-pools the model's
hidden states into a single embedding vector — the standard way to repurpose
a decoder-only LLM as an embedder (SGPT, Muennighoff et al. 2022). This is
the genuine LLM arm; `mxbai-embed-large`, a dedicated (non-LLM) embedding
model, is kept alongside it as a second large-embedding-model data point.

## Setup

```bash
pip install -r requirements.txt
brew install libomp                # required by lightgbm on macOS
brew install llama.cpp             # provides `llama-server`, for the phi4-mini LLM pipeline
ollama pull mxbai-embed-large      # only needed if not already present
ollama pull phi4-mini               # only needed if not already present

# start the LLM embedding server before running any experiment script
# (leave running in a separate terminal; blob path is resolved automatically)
llama-server \
  -m "$(python -c 'from src.embeddings import _ollama_gguf_path; print(_ollama_gguf_path("phi4-mini"))')" \
  --embeddings --pooling mean --port 11600 --host 127.0.0.1 \
  -c 4096 -np 1 -ub 1536 -b 1536
```

## Run

```bash
pytest tests/                     # unit + smoke tests

# --dataset: movielens (default), movielens-title, movielens-genres, amazon, yelp
# --features: item | personalized | personalized_pop (default)
# --negatives: uniform (default) | popularity     --split: random (default) | temporal
python -m src.run_experiment --dataset movielens --negatives popularity     # main configuration
python -m src.run_experiment --dataset movielens --features item --negatives popularity   # item-only ablation
python -m src.run_experiment --dataset movielens --negatives popularity --split temporal --scorers lgbm
python -m src.run_experiment --dataset movielens-genres --negatives popularity --scorers lgbm --pipelines TF-IDF SBERT

python -m src.stats results/per_user_personalized_pop_neg-popularity.csv   # paired t / TOST tests
python -m src.make_tables                       # paper/tables/*.tex from results CSVs
python -m src.representation_analysis geometry --dataset yelp  # §VII embedding geometry
python -m src.representation_analysis affinity --dataset yelp  # §VII affinity-only ranking (also: history, centering)
python -m src.latency_benchmark --dataset movielens   # like-for-like latency; run on an idle machine
```

Amazon Reviews and Yelp download their raw data on first run
(`src/data_amazon.py`, `src/data_yelp.py`); Yelp additionally requires
placing `yelp_dataset.tar` (direct download: Yelp-JSON.zip from
https://business.yelp.com/data/resources/open-dataset/, or from registering at
[yelp.com/dataset](https://www.yelp.com/dataset)) under `data/yelp/`
first, since it isn't otherwise downloadable without agreeing to Yelp's
terms.

Results land in `results/*.csv` (suffixed `_amazon`/`_yelp` for those
datasets; the movielens run keeps the original unsuffixed filenames). On
a memory-constrained machine, LightGBM's
thread count is capped (`src/pipeline.py`) specifically to avoid swap
thrashing when running alongside other apps — see `num_threads` there if
running on a dedicated/high-memory machine where you'd rather let it use all
cores.

## Results

Every number in the paper comes from the per-user CSVs in `results/`:
`python -m src.stats` runs the paired tests and `python -m src.make_tables`
writes the LaTeX tables in `paper/tables/`. Headline results (popularity-matched
negatives, personalized scorer, LGBMRanker, NDCG@10, 5 seeds):

| Pipeline | MovieLens | Amazon | Yelp |
|---|---|---|---|
| Popularity | 0.379 | 0.090 | 0.133 |
| PureSVD | 0.572 | 0.151 | 0.151 |
| TF-IDF | 0.576 | 0.170 | 0.179 |
| SBERT (22M) | 0.589 | 0.175 | 0.192 |
| mxbai (335M) | 0.583 | 0.178 | 0.195 |
| phi4-mini raw (3.8B) | 0.486 | 0.160 | 0.194 |
| Qwen3-Embedding (4B) | 0.587 | 0.185 | 0.198 |
| Hybrid | 0.579 | 0.172 | 0.199 |

## Repo layout

```
src/
  data.py                — download/cache MovieLens, build item text corpus
  data_amazon.py          — download/cache Amazon Reviews (All_Beauty 5-core), build item text corpus
  data_yelp.py              — load a Yelp Open Dataset city subset from yelp_dataset.tar, build item text corpus
  dataset_registry.py        — maps --dataset name to loader, cache namespace, results-filename suffix
  candidates.py                — per-user train/test split (random or temporal) + negative sampling
  embeddings.py                  — TF-IDF, SBERT, mxbai (Ollama), phi4-mini (llama.cpp) features, disk-cached (namespaced per dataset)
  metrics.py                       — per-user NDCG@K / Recall@K + aggregate stats
  pipeline.py                        — LGBMRanker + MLP scorers
  user_features.py                     — leave-one-out user-item affinity + training-only popularity features
  baselines.py                          — popularity and PureSVD reference baselines
  run_experiment.py                    — pipelines x scorers x seeds for one configuration; per-user output
  stats.py                              — paired t-test / Wilcoxon / TOST, Holm-corrected
  make_tables.py                         — paper tables from results CSVs
  representation_analysis.py             — embedding geometry, affinity-only ranking, centering, user-history check
  latency_benchmark.py                   — like-for-like encoder latency benchmark
  generate_figures.py                     — figures
  quality_cost_analysis.py                 — clean, uncached quality-vs-latency benchmark + Pareto plot
  redundancy_analysis.py                    — corpus lexical redundancy vs. embedding benefit
tests/                                       — pytest unit + smoke tests
results/                                     — output CSVs and LaTeX table snippets
```

## Status

Complete for MovieLens, Amazon Reviews, and Yelp: five feature/negative/split
configurations, two scorers, five seeds, paired significance and equivalence
tests, representation analyses, and a like-for-like latency benchmark.
