"""Quality-cost analysis: turns the raw results table into a Pareto-style
comparison of ranking quality vs. embedding latency, relative to SBERT (the
cheapest non-trivial embedding baseline).

For each pipeline (LGBMRanker rows only, matching the other figures):
  - Relative latency: embed latency / SBERT's embed latency
  - Delta NDCG@10 vs SBERT (percentage points)
  - Efficiency: delta NDCG@10 per additional ms of embed latency
  - Significance: whether the pipeline's NDCG@10 95% CI overlaps SBERT's
    (non-overlap is the bar already used elsewhere in this paper for
    "statistically distinguishable")

TF-IDF has ~0 embed latency by construction (no model inference), so its
relative-latency and efficiency figures aren't meaningful in the same way as
the embedding-model comparisons; they're left as "--" rather than forced
into a division that doesn't mean anything.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_registry import get_dataset, results_suffix
from src.embeddings import build_sbert_features, build_true_llm_features
from src.generate_figures import CATEGORICAL, TEXT_PRIMARY, TEXT_SECONDARY, _clean_axes

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"

# The headline comparison this table exists for: SBERT (cheap embedding
# baseline) vs. the genuine LLM (phi4-mini). mxbai isn't re-benchmarked here
# -- it's kept in the results table elsewhere, but this analysis is scoped to
# the SBERT-vs-real-LLM cost/quality question, not mxbai's.
PIPELINES = [
    "Baseline (TF-IDF)",
    "SBERT (all-MiniLM-L6-v2)",
    "LLM (phi4-mini, local)",
    "Hybrid (TF-IDF + phi4-mini)",
]


def fresh_sbert_latency_ms(dataset):
    """Real (uncached) per-item SBERT latency, needed as the cost baseline.

    The results table's EmbedLatency column is 0 whenever SBERT's embeddings
    were already cached before that run -- the normal case by the time this
    analysis runs, since they're built once early on and reused. Benchmarked
    on the FULL item set, not a sample: `.encode()` batches internally, so
    its per-item amortized cost depends on batch size (a 20-item sample
    measured ~25x higher per-item latency than the full-dataset batch does).
    phi4-mini isn't re-benchmarked here since its EmbedLatency figure is
    already genuinely fresh from the main results run.
    """
    load_fn, cache_prefix, _ = get_dataset(dataset)
    _, movies = load_fn()
    _, sbert_latency = build_sbert_features(movies, use_cache=False, cache_prefix=cache_prefix)
    return sbert_latency * 1000  # sec -> ms


def fresh_llm_latency_ms(dataset, server_url="http://127.0.0.1:11600"):
    """Real (uncached) per-item phi4-mini latency, mean AND median.

    The production run's EmbedLatency figure for phi4-mini is a simple mean
    over per-item timings recorded live during a multi-hour unattended run --
    which we've since confirmed was intermittently disrupted (a supervisor
    restart landed mid-embedding-generation for at least one dataset, and
    even "successful" runs completed several times slower than a clean
    benchmark predicted, most likely OS-level throttling during idle periods
    that a persistent `caffeinate` now prevents). A few multi-minute outlier
    requests are enough to badly inflate a mean. Re-benchmarks a clean,
    uncached pass here (mean and median, so the writeup can report -- and
    readers can sanity-check -- a statistic that isn't outlier-dominated) now
    that caffeinate is active and the server is otherwise idle.
    """
    import time

    import numpy as np
    import requests

    load_fn, _, _ = get_dataset(dataset)
    _, movies = load_fn()
    texts = movies["text"].tolist()

    latencies = []
    for text in texts:
        start = time.time()
        # Same 503 retry as the production embedding builder: llama-server's
        # /health (and, we've now confirmed, a request right after it) can
        # report ready a moment before the model is actually loaded.
        for attempt in range(5):
            response = requests.post(f"{server_url}/embedding", json={"content": text}, timeout=120)
            if response.status_code != 503:
                break
            time.sleep(2 * (attempt + 1))
        response.raise_for_status()
        latencies.append(time.time() - start)

    latencies_ms = np.asarray(latencies) * 1000
    return float(np.mean(latencies_ms)), float(np.median(latencies_ms))


def build_quality_cost_table(dataset="movielens", fresh_llm_latency=True, llm_server_url="http://127.0.0.1:11600"):
    suffix = results_suffix(dataset)
    df = pd.read_csv(RESULTS_DIR / f"results_table{suffix}.csv")
    df = df[df["Scorer"] == "lgbm"].set_index("Pipeline").loc[PIPELINES]

    print("Benchmarking fresh (uncached) SBERT latency...")
    df.loc["SBERT (all-MiniLM-L6-v2)", "EmbedLatency_ms_per_item"] = fresh_sbert_latency_ms(dataset)

    llm_median_ms = None
    if fresh_llm_latency:
        print(f"Benchmarking fresh (uncached) phi4-mini latency (mean + median) via {llm_server_url}...")
        llm_mean_ms, llm_median_ms = fresh_llm_latency_ms(dataset, server_url=llm_server_url)
        df.loc["LLM (phi4-mini, local)", "EmbedLatency_ms_per_item"] = llm_mean_ms
        df.loc["Hybrid (TF-IDF + phi4-mini)", "EmbedLatency_ms_per_item"] = llm_mean_ms

    sbert_row = df.loc["SBERT (all-MiniLM-L6-v2)"]
    sbert_ndcg = sbert_row["NDCG@10_mean"]
    sbert_ci = sbert_row["NDCG@10_ci95"]
    sbert_latency = sbert_row["EmbedLatency_ms_per_item"]

    rows = []
    for pipeline, row in df.iterrows():
        ndcg = row["NDCG@10_mean"]
        ci = row["NDCG@10_ci95"]
        latency = row["EmbedLatency_ms_per_item"]
        delta_ndcg = ndcg - sbert_ndcg
        # Non-overlap of the two 95% CIs around the *difference* -- if the
        # pipelines' own CIs don't overlap, the difference is significant at
        # roughly the 95% level (a conservative version of a two-sample test,
        # consistent with the "statistically indistinguishable" language used
        # elsewhere in this paper).
        significant = pipeline != "SBERT (all-MiniLM-L6-v2)" and (
            abs(delta_ndcg) > (ci + sbert_ci)
        )

        if latency <= 1e-9 or sbert_latency <= 1e-9:
            rel_latency = None
            efficiency = None
        else:
            rel_latency = latency / sbert_latency
            delta_latency = latency - sbert_latency
            efficiency = delta_ndcg / delta_latency if delta_latency != 0 else None

        rows.append(
            {
                "Pipeline": pipeline,
                "NDCG@10": ndcg,
                "Recall@10": row["Recall@10_mean"],
                "Latency_ms_per_item": latency,
                "MedianLatency_ms_per_item": (
                    llm_median_ms if pipeline in ("LLM (phi4-mini, local)", "Hybrid (TF-IDF + phi4-mini)") else None
                ),
                "RelativeLatency_vs_SBERT": rel_latency,
                "DeltaNDCG_vs_SBERT": delta_ndcg,
                "Significant_vs_SBERT": significant,
                "EfficiencyDeltaNDCGperMs": efficiency,
            }
        )

    return pd.DataFrame(rows)


SHORT_NAMES = {
    "Baseline (TF-IDF)": "TF-IDF",
    "SBERT (all-MiniLM-L6-v2)": "SBERT",
    "LLM (phi4-mini, local)": "phi4-mini (LLM)",
    "Hybrid (TF-IDF + phi4-mini)": "Hybrid",
}


def format_latex_table(qc_df, dataset_label):
    # Single-column IEEEtran width can't fit 7 verbose columns at normal
    # size without pushing the last 1-2 columns off the page margin
    # (silently -- pdflatex only warns "Overfull \hbox", it doesn't error).
    # \scriptsize + tight \tabcolsep + a merged Sig./asterisk column keeps
    # every column on the visible page; verified by rendering to PNG and
    # inspecting, not just checking the page count.
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Quality-cost comparison on "
        + dataset_label
        + r" (LGBMRanker). Latency is a clean, uncached re-benchmark (mean"
        + r" and, for phi4-mini, median -- robust to the rare multi-minute"
        + r" outlier request an unattended multi-hour run can pick up)."
        + r" Relative latency and efficiency are computed against SBERT,"
        + r" the cheapest embedding baseline; TF-IDF has no embedding-model"
        + r" latency to compare.}"
    )
    lines.append(r"\label{tab:quality-cost-" + dataset_label.lower().replace(" ", "-") + "}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Pipeline & NDCG@10 & Mean (ms) & Med.\ (ms) & Rel.\ lat.\ & $\Delta$NDCG \\")
    lines.append(r"\midrule")
    any_significant = False
    for _, row in qc_df.iterrows():
        name = SHORT_NAMES.get(row["Pipeline"], row["Pipeline"])
        rel = "--" if pd.isna(row["RelativeLatency_vs_SBERT"]) else f"${row['RelativeLatency_vs_SBERT']:.1f}\\times$"
        median = "--" if pd.isna(row["MedianLatency_ms_per_item"]) else f"{row['MedianLatency_ms_per_item']:.2f}"
        significant = bool(row["Significant_vs_SBERT"])
        any_significant = any_significant or significant
        star = "^{*}" if significant else ""
        delta = f"${row['DeltaNDCG_vs_SBERT']:+.4f}{star}$"
        lines.append(
            f"{name} & {row['NDCG@10']:.4f} & {row['Latency_ms_per_item']:.2f} & {median} & "
            f"{rel} & {delta} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{2pt}")
    if any_significant:
        lines.append(
            r"{\scriptsize $^{*}$significant: 95\% CI (across seeds) vs.\ SBERT does not overlap.\par}"
        )
    else:
        lines.append(
            r"{\scriptsize None of the above $\Delta$NDCG values are significant"
            r" (95\% CI vs.\ SBERT overlaps).\par}"
        )
    lines.append(r"\end{table}")
    return "\n".join(lines)


def plot_quality_cost(qc_df, dataset):
    """Quality-vs-cost scatter (log-x latency), the Pareto-frontier view.

    TF-IDF is excluded: its ~0ms latency has no meaningful position on a log
    axis, and it's already covered by the main results bar chart elsewhere.
    This plot is specifically about the embedding-model cost/quality
    tradeoff (SBERT vs. phi4-mini vs. Hybrid).
    """
    plot_df = qc_df[qc_df["Pipeline"] != "Baseline (TF-IDF)"].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=200)
    for i, row in plot_df.iterrows():
        name = SHORT_NAMES.get(row["Pipeline"], row["Pipeline"])
        color = CATEGORICAL[i % len(CATEGORICAL)]
        ax.scatter(
            row["Latency_ms_per_item"],
            row["NDCG@10"],
            s=90,
            color=color,
            zorder=3,
            edgecolors="white",
            linewidths=0.8,
        )
        ax.annotate(
            name,
            (row["Latency_ms_per_item"], row["NDCG@10"]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
            color=TEXT_PRIMARY,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Embedding latency (ms/item, log scale)", color=TEXT_PRIMARY)
    ax.set_ylabel("NDCG@10", color=TEXT_PRIMARY)
    labels = {"movielens": "MovieLens", "amazon": "Amazon Reviews", "yelp": "Yelp"}
    ax.set_title(f"Quality vs. cost: {labels[dataset]}", color=TEXT_PRIMARY)
    _clean_axes(ax)
    ax.xaxis.grid(True, color="#dddddd", linewidth=0.8, zorder=0)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    suffix = results_suffix(dataset)
    out_path = FIGURES_DIR / f"quality_cost{suffix}.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="movielens", choices=["movielens", "amazon", "yelp"]
    )
    parser.add_argument("--llm-server-url", default="http://127.0.0.1:11600")
    parser.add_argument(
        "--latency-only",
        action="store_true",
        help="Only benchmark fresh phi4-mini latency (mean/median) and save it, without "
        "assembling the full quality-cost table -- use when results_table_<dataset>.csv "
        "doesn't exist yet (e.g. a still-running dataset). Run again without this flag "
        "later to build the full table once results are ready.",
    )
    args = parser.parse_args()

    if args.latency_only:
        print(f"Benchmarking fresh (uncached) phi4-mini latency for {args.dataset} via {args.llm_server_url}...")
        mean_ms, median_ms = fresh_llm_latency_ms(args.dataset, server_url=args.llm_server_url)
        suffix = results_suffix(args.dataset)
        out_path = RESULTS_DIR / f"llm_latency_only{suffix}.csv"
        pd.DataFrame([{"MeanLatency_ms": mean_ms, "MedianLatency_ms": median_ms}]).to_csv(
            out_path, index=False
        )
        print(f"Mean: {mean_ms:.2f} ms, Median: {median_ms:.2f} ms")
        print(f"Saved to {out_path}")
        return

    qc_df = build_quality_cost_table(args.dataset, llm_server_url=args.llm_server_url)
    suffix = results_suffix(args.dataset)
    out_path = RESULTS_DIR / f"quality_cost{suffix}.csv"
    qc_df.to_csv(out_path, index=False)
    print(qc_df.to_string(index=False))
    print(f"\nSaved to {out_path}")

    labels = {"movielens": "MovieLens", "amazon": "Amazon Reviews", "yelp": "Yelp"}
    latex = format_latex_table(qc_df, labels[args.dataset])
    tex_path = RESULTS_DIR / f"quality_cost{suffix}.tex"
    tex_path.write_text(latex + "\n")
    print(f"Saved LaTeX table to {tex_path}")

    plot_quality_cost(qc_df, args.dataset)


if __name__ == "__main__":
    main()
