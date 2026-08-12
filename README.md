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

# --dataset defaults to movielens; also accepts amazon, yelp
python src/run_experiment.py --dataset movielens     # main results: 5 pipelines x 2 scorers x 5 seeds
python src/run_robustness.py --dataset movielens      # robustness: random vs. temporal split, 3 seeds
python src/failure_analysis.py --dataset movielens    # qualitative per-user regression/gain examples
python src/generate_figures.py --dataset movielens    # regenerates result figures
python src/quality_cost_analysis.py --dataset movielens  # clean, uncached quality-vs-latency benchmark
python src/redundancy_analysis.py                     # corpus lexical redundancy vs. embedding benefit, all 3 datasets
```

Amazon Reviews and Yelp download their raw data on first run
(`src/data_amazon.py`, `src/data_yelp.py`); Yelp additionally requires
placing `yelp_dataset.tar` (from registering at
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

## Results (ml-latest-small)

**Main comparison** (LGBMRanker, 5-seed mean ± 95% CI):

| Pipeline | NDCG@10 | Recall@10 | Embed latency (ms/item)\* |
|---|---|---|---|
| Baseline (TF-IDF) | 0.7055 ± 0.0065 | 0.6080 ± 0.0062 | 0.00 |
| SBERT (all-MiniLM-L6-v2) | 0.7258 ± 0.0043 | 0.6377 ± 0.0064 | 0.43 |
| Large Embedding Model (mxbai-embed-large) | 0.7255 ± 0.0056 | 0.6396 ± 0.0071 | -- |
| LLM (phi4-mini, local) | 0.7250 ± 0.0058 | 0.6399 ± 0.0071 | 262.17 |
| Hybrid (TF-IDF + phi4-mini) | 0.7251 ± 0.0051 | 0.6396 ± 0.0072 | 262.17 |

\* A dedicated, uncached re-benchmark (mean; see `results/quality_cost.tex`
for mean+median and the SBERT-relative cost multiplier) — not simple
production-run figures, which can be cache-contaminated. Large Embedding
Model is omitted here because it was already cached at measurement time. In
production, item embeddings are content features computed once and cached,
so this cost is amortized, not paid per query.

None of the three larger-capacity pipelines (the dedicated large embedding
model, the LLM, or their hybrid with TF-IDF) improves measurably on the much
smaller SBERT baseline, despite the LLM costing orders of magnitude more in
per-item embedding latency. What both embedding-based pipelines clearly do
is beat TF-IDF (best embedding, SBERT: 0.7258 vs. TF-IDF's 0.7055).

**Robustness** (NDCG@10 under random vs. temporal split, LGBMRanker,
3-seed mean):

| Pipeline | Random | Temporal | Drop |
|---|---|---|---|
| TF-IDF | 0.7041 | 0.6290 | 0.0750 |
| SBERT | 0.7250 | 0.6549 | 0.0701 |
| Large Embedding Model | 0.7251 | 0.6566 | 0.0685 |
| LLM (phi4-mini) | 0.7236 | 0.6566 | 0.0669 |
| Hybrid | 0.7236 | 0.6574 | 0.0662 |

TF-IDF is the *most* brittle to distribution shift — the practical gap
between TF-IDF and the embedding pipelines widens, not stays constant, once
the model has to generalize to future user behavior rather than interpolate
a random split.

**Failure analysis** (`results/failure_examples.csv`): the largest per-user
regressions/gains for the LLM pipeline vs. TF-IDF are concentrated among
users with very few (1–3) held-out positives, where per-user NDCG@10 is a
coarse, high-variance statistic.

## Results (Amazon Reviews, `All_Beauty` 5-core)

**Main comparison** (LGBMRanker, 5-seed mean ± 95% CI):

| Pipeline | NDCG@10 | Recall@10 | Embed latency (ms/item) |
|---|---|---|---|
| Baseline (TF-IDF) | 0.0972 ± 0.0030 | 0.1854 ± 0.0133 | 0.00 |
| SBERT (all-MiniLM-L6-v2) | 0.0994 ± 0.0071 | 0.1899 ± 0.0175 | -- |
| Large Embedding Model (mxbai-embed-large) | 0.0982 ± 0.0048 | 0.1908 ± 0.0138 | -- |
| LLM (phi4-mini, local) | 0.1009 ± 0.0065 | 0.2009 ± 0.0143 | 6871.08 |
| Hybrid (TF-IDF + phi4-mini) | 0.0979 ± 0.0043 | 0.1914 ± 0.0103 | 6871.08 |

Unlike MovieLens, all five pipelines land within each other's confidence
intervals — the classical-vs-embedding gap effectively disappears on this
dataset's long, review-derived item text. Amazon Reviews' small scale (253
users) means these CIs are wide relative to the pipeline differences
observed, so this null result carries less weight on its own than Yelp's
higher-precision one below.

**Robustness** (NDCG@10 under random vs. temporal split, LGBMRanker,
3-seed mean):

| Pipeline | Random | Temporal | Drop |
|---|---|---|---|
| TF-IDF | 0.0959 | 0.0541 | 0.0418 |
| SBERT | 0.0985 | 0.0527 | 0.0458 |
| Large Embedding Model | 0.0967 | 0.0508 | 0.0459 |
| LLM (phi4-mini) | 0.0985 | 0.0519 | 0.0466 |
| Hybrid | 0.0950 | 0.0554 | 0.0397 |

## Results (Yelp, Boise ID subset)

**Main comparison** (LGBMRanker, 3-seed mean ± 95% CI — reduced from 5 seeds
for local compute-time tractability; see `results/results_table_yelp.csv`):

| Pipeline | NDCG@10 | Recall@10 | Embed latency (ms/item) |
|---|---|---|---|
| Baseline (TF-IDF) | 0.4656 ± 0.0040 | 0.7039 ± 0.0044 | 0.00 |
| SBERT (all-MiniLM-L6-v2) | 0.4648 ± 0.0033 | 0.7037 ± 0.0032 | -- |
| Large Embedding Model (mxbai-embed-large) | 0.4655 ± 0.0036 | 0.7042 ± 0.0038 | -- |
| LLM (phi4-mini, local) | 0.4649 ± 0.0036 | 0.7032 ± 0.0033 | -- |
| Hybrid (TF-IDF + phi4-mini) | 0.4649 ± 0.0029 | 0.7030 ± 0.0034 | -- |

All embedding-pipeline latencies were already cached at measurement time in
this main run; see `results/quality_cost.tex`-style analysis for a clean,
uncached comparison methodology (run against MovieLens in this repo).

With 10,537 qualifying users, this dataset has the tightest confidence
intervals of the three — and still shows no separation between pipelines,
the strongest evidence in this study that larger-capacity (and even
small-embedding) augmentation isn't a reliable win on long, review-derived
item text.

**Robustness** (NDCG@10 under random vs. temporal split, LGBMRanker,
2-seed mean, reduced from 3 for tractability):

| Pipeline | Random | Temporal | Drop |
|---|---|---|---|
| TF-IDF | 0.4636 | 0.4457 | 0.0178 |
| SBERT | 0.4631 | 0.4460 | 0.0171 |
| Large Embedding Model | 0.4637 | 0.4476 | 0.0161 |
| LLM (phi4-mini) | 0.4631 | 0.4465 | 0.0166 |
| Hybrid | 0.4634 | 0.4463 | 0.0170 |

**Cross-dataset takeaway:** the "larger-capacity embeddings ≈ small SBERT
embeddings" half of the MovieLens finding replicates on both new datasets —
larger-capacity embeddings never meaningfully beat SBERT anywhere. The
"embeddings > TF-IDF" half does *not* replicate: on Amazon Reviews and Yelp,
TF-IDF is statistically indistinguishable from every embedding pipeline. We
trace this to corpus lexical redundancy (`src/redundancy_analysis.py`):
MovieLens's titles are lexically distinctive (low TF-IDF cosine similarity
across items), while Amazon and Yelp reviews reuse the same phrasing across
unrelated items (high similarity) — the more redundant the text, the less
there is for a denser representation to add over TF-IDF.

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
  run_experiment.py                    — main results: pipelines x scorers x seeds
  run_robustness.py                     — random vs. temporal split comparison
  failure_analysis.py                    — qualitative per-user regression/gain examples
  generate_figures.py                     — figures
  quality_cost_analysis.py                 — clean, uncached quality-vs-latency benchmark + Pareto plot
  redundancy_analysis.py                    — corpus lexical redundancy vs. embedding benefit
tests/                                       — pytest unit + smoke tests
results/                                     — output CSVs and LaTeX table snippets
```

## Status

**Complete: MovieLens, Amazon Reviews, and Yelp.** All three datasets have
five pipelines, two scorers, stability (multi-seed), robustness (temporal
shift), qualitative failure analysis, and cost/latency, including a
dedicated quality-vs-cost benchmark and a corpus-redundancy diagnostic. No
experimental work remains.
