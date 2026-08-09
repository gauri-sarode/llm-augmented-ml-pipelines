# Evaluating LLM-Augmented ML Pipelines: Performance, Stability, and Cost

An empirical study of when LLM-based item features help, hurt, or are
unnecessary inside a recommendation-ranking ML pipeline — measured on
ranking quality, stability across seeds, robustness under distribution
shift, and latency/cost, not just a single accuracy number.

**Paper draft (IEEE format):** [`paper/main.pdf`](paper/main.pdf) — complete
results for all three datasets (MovieLens, Amazon Reviews, Yelp); only the
author name/affiliation placeholders remain (see [Status](#status) below).

## Task

Three datasets, each cast as implicit-feedback, per-user ranking:
MovieLens (`ml-latest-small`, short title+genre item text), Amazon
Reviews (`All_Beauty` 5-core, long free-form review text), and Yelp
(Boise, ID subset, long free-form review text). See
[`paper/sections/05_experimental_setup.tex`](paper/sections/05_experimental_setup.tex)
for exact dataset sizes and construction.

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
| LLM (mxbai-embed-large) | local LLM-scale embeddings (335M params) via Ollama's `/api/embed`, $0 marginal cost |
| Hybrid | TF-IDF + mxbai-embed-large concatenated |

Each is scored by both `LGBMRanker` (lambdarank) and a scikit-learn MLP, to
check conclusions aren't an artifact of one scorer's inductive biases.

We initially tried `phi4-mini` (already pulled locally) for the LLM pipeline,
but Ollama's desktop-app-managed server only serves embeddings for models
started in embedding mode, which chat models like phi4-mini aren't by
default. `mxbai-embed-large` is purpose-built for embeddings, works cleanly
with Ollama's embed API, and is still ~15x larger than the SBERT baseline.

## Setup

```bash
pip install -r requirements.txt
brew install libomp                # required by lightgbm on macOS
brew install texlive                # only needed to compile paper/main.tex
ollama pull mxbai-embed-large      # only needed if not already present
```

## Run

```bash
pytest tests/                     # unit + smoke tests

# --dataset defaults to movielens; also accepts amazon, yelp
python src/run_experiment.py --dataset movielens   # main results: 4 pipelines x 2 scorers x 5 seeds
python src/run_robustness.py --dataset movielens    # robustness: random vs. temporal split, 3 seeds
python src/failure_analysis.py --dataset movielens  # qualitative per-user regression/gain examples
python src/generate_figures.py --dataset movielens  # paper figures -> paper/figures/*.png

cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
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
thread count is capped at 4 (`src/pipeline.py`) specifically to avoid swap
thrashing when running alongside other apps — see `num_threads` there if
running on a dedicated/high-memory machine where you'd rather let it use all
cores.

## Results (ml-latest-small)

**Main comparison** (LGBMRanker, 5-seed mean ± 95% CI):

| Pipeline | NDCG@10 | Recall@10 | Embed latency (ms/item)\* |
|---|---|---|---|
| Baseline (TF-IDF) | 0.7055 ± 0.0065 | 0.6080 ± 0.0062 | 0.00 |
| SBERT (all-MiniLM-L6-v2) | 0.7258 ± 0.0043 | 0.6377 ± 0.0064 | 0.91 |
| LLM (mxbai-embed-large, local) | 0.7255 ± 0.0056 | 0.6396 ± 0.0071 | 30.25 |
| Hybrid (TF-IDF + mxbai-embed-large) | 0.7256 ± 0.0053 | 0.6406 ± 0.0073 | 30.25 |

\* Wall-clock cost of a fresh (uncached) embedding call; in production these
are content features computed once and cached, so this cost is amortized,
not paid per query.

Both embedding-based pipelines clearly beat TF-IDF, but the 335M-param LLM
embedding buys essentially nothing over the 22M-param SBERT baseline —
despite costing ~33x more in per-item embedding latency. Hybrid doesn't
meaningfully beat SBERT or LLM alone either.

**Scorer comparison** (NDCG@10, 5-seed mean) — the picture is more nuanced
than the table above alone suggests:

| Pipeline | LGBMRanker | MLP |
|---|---|---|
| TF-IDF | 0.7055 | 0.7218 |
| SBERT | 0.7258 | 0.7223 |
| LLM | 0.7255 | 0.7265 |
| Hybrid | 0.7256 | 0.7250 |

With an MLP scorer, TF-IDF nearly closes the gap with the embedding
pipelines — some of the classical-vs-embedding gap seen with LGBMRanker
reflects how that specific scorer handles a sparse TF-IDF vector, not a
fundamental TF-IDF limitation. What's robust across *both* scorers: the
LLM-scale embedding never meaningfully beats SBERT.

**Robustness** (NDCG@10 under random vs. temporal split, LGBMRanker,
3-seed mean):

| Pipeline | Random | Temporal | Drop |
|---|---|---|---|
| TF-IDF | 0.7041 | 0.6290 | 0.0750 |
| SBERT | 0.7250 | 0.6549 | 0.0701 |
| LLM | 0.7251 | 0.6566 | 0.0685 |
| Hybrid | 0.7247 | 0.6564 | 0.0684 |

TF-IDF is the *most* brittle to distribution shift — the practical gap
between TF-IDF and the embedding pipelines widens, not stays constant, once
the model has to generalize to future user behavior rather than interpolate
a random split.

**Failure analysis** (`results/failure_examples.csv`): the largest per-user
regressions/gains for the LLM pipeline vs. TF-IDF are concentrated among
users with very few (1–3) held-out positives, where per-user NDCG@10 is a
coarse, high-variance statistic — see `paper/sections/07_failure_analysis.tex`
for the full qualitative breakdown.

## Results (Amazon Reviews, `All_Beauty` 5-core)

**Main comparison** (LGBMRanker, 5-seed mean ± 95% CI):

| Pipeline | NDCG@10 | Recall@10 | Embed latency (ms/item) |
|---|---|---|---|
| Baseline (TF-IDF) | 0.0972 ± 0.0030 | 0.1854 ± 0.0133 | 0.00 |
| SBERT (all-MiniLM-L6-v2) | 0.0994 ± 0.0071 | 0.1899 ± 0.0175 | 11.72 |
| LLM (mxbai-embed-large, local) | 0.0982 ± 0.0048 | 0.1908 ± 0.0138 | 765.52 |
| Hybrid (TF-IDF + mxbai-embed-large) | 0.0975 ± 0.0068 | 0.1862 ± 0.0104 | 765.52 |

Unlike MovieLens, all four pipelines land within each other's confidence
intervals — the classical-vs-embedding gap effectively disappears on
this dataset's long, review-derived item text. Embedding latency for the
LLM/hybrid pipelines jumps to 765.52 ms/item (25x higher than
MovieLens's 30.25 ms/item) purely because the input text is much longer.

## Results (Yelp, Boise ID subset)

**Main comparison** (LGBMRanker, 5-seed mean ± 95% CI):

| Pipeline | NDCG@10 | Recall@10 | Embed latency (ms/item) |
|---|---|---|---|
| Baseline (TF-IDF) | 0.4672 ± 0.0032 | 0.7049 ± 0.0040 | 0.00 |
| SBERT (all-MiniLM-L6-v2) | 0.4665 ± 0.0029 | 0.7040 ± 0.0024 | 24.43 |
| LLM (mxbai-embed-large, local) | 0.4668 ± 0.0026 | 0.7046 ± 0.0028 | 914.15 |
| Hybrid (TF-IDF + mxbai-embed-large) | 0.4661 ± 0.0029 | 0.7032 ± 0.0033 | 914.15 |

With 10,537 qualifying users, this dataset has the tightest confidence
intervals of the three — and still shows no separation between
pipelines, the strongest evidence in this study that LLM-scale (and
even small-embedding) augmentation isn't a reliable win on long,
review-derived item text.

**Cross-dataset takeaway:** the "LLM-scale embeddings ≈ small SBERT
embeddings" half of the MovieLens finding replicates on both new
datasets — LLM-scale embeddings never meaningfully beat SBERT anywhere.
The "embeddings > TF-IDF" half does *not* replicate: on Amazon Reviews
and Yelp, TF-IDF is statistically indistinguishable from every embedding
pipeline. Full narrative, per-dataset robustness/failure-analysis
results, and all figures are in
[`paper/sections/06_results.tex`](paper/sections/06_results.tex)
(§Cross-Dataset Synthesis) and [`paper/main.pdf`](paper/main.pdf).

## Repo layout

```
src/
  data.py               — download/cache MovieLens, build item text corpus
  data_amazon.py          — download/cache Amazon Reviews (All_Beauty 5-core), build item text corpus
  data_yelp.py              — load a Yelp Open Dataset city subset from yelp_dataset.tar, build item text corpus
  dataset_registry.py        — maps --dataset name to loader, cache namespace, results-filename suffix
  candidates.py                — per-user train/test split (random or temporal) + negative sampling
  embeddings.py                  — TF-IDF, SBERT, LLM (Ollama) features, disk-cached (namespaced per dataset)
  metrics.py                       — per-user NDCG@K / Recall@K + aggregate stats
  pipeline.py                        — LGBMRanker + MLP scorers
  run_experiment.py                    — main results: pipelines x scorers x seeds
  run_robustness.py                     — random vs. temporal split comparison
  failure_analysis.py                    — qualitative per-user regression/gain examples
  generate_figures.py                     — paper figures
tests/                                    — pytest unit + smoke tests
results/                                  — output CSVs
paper/                                    — IEEE LaTeX source + compiled main.pdf
```

## Status

**Complete: MovieLens, Amazon Reviews, and Yelp.** All three datasets
have four pipelines, two scorers, stability (5 seeds), robustness
(temporal shift), qualitative failure analysis, and cost/latency —
matching the full paper outline. Cross-dataset synthesis is in
[`paper/sections/06_results.tex`](paper/sections/06_results.tex).

**Open for the coauthor:** author name/affiliation only
(`\coauthortodo{}` markers in `paper/main.tex`, render in red in the
PDF) — no experimental work remains.
