"""Generate the paper's LaTeX result tables from results/*.csv.

Every number in the results section comes from here, so the tables can
be regenerated after any rerun:

    python -m src.make_tables          # writes paper/tables/*.tex
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.run_experiment import run_tag
from src.stats import pairwise_tests

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TABLES_DIR = Path(__file__).resolve().parent.parent / "paper" / "tables"

SHORT = {
    "Popularity": "Popularity",
    "PureSVD (MF)": "PureSVD",
    "Baseline (TF-IDF)": "TF-IDF",
    "SBERT (all-MiniLM-L6-v2)": "SBERT (22M)",
    "Large Embedding Model (mxbai-embed-large)": "mxbai (335M)",
    "LLM (phi4-mini, local)": "phi4-mini raw (3.8B)",
    "LLM embedder (Qwen3-Embedding-4B)": "Qwen3-Emb.\\ (4B)",
    "Hybrid (TF-IDF + phi4-mini)": "Hybrid",
}
ORDER = list(SHORT)


def _load(dataset, mode, negatives, split="random"):
    tag = run_tag(dataset, mode, negatives, split)
    return pd.read_csv(RESULTS_DIR / f"results_table{tag}.csv")


def _cell(row):
    return f"{row['NDCG@10_mean']:.3f}$\\pm${row['NDCG@10_ci95']:.3f}"


def main_table(dataset, scorer="lgbm"):
    """NDCG@10 (mean +/- 95% CI over 5 seeds) across the feature/negative configs."""
    configs = [
        ("item", "popularity", "Item only"),
        ("personalized", "popularity", "+Affinity"),
        ("personalized_pop", "popularity", "+Aff.+Pop."),
        ("personalized_pop", "uniform", "Uniform neg."),
    ]
    cols = {}
    for mode, neg, label in configs:
        df = _load(dataset, mode, neg)
        df = df[df["Scorer"].isin([scorer, "-"])].set_index("Pipeline")
        cols[label] = df.apply(_cell, axis=1)
    table = pd.DataFrame(cols).reindex(ORDER).dropna(how="all")
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        " & \\multicolumn{3}{c}{Popularity-matched negatives} & \\\\",
        "\\cmidrule(lr){2-4}",
        "Pipeline & " + " & ".join(table.columns) + " \\\\",
        "\\midrule",
    ]
    for name, row in table.iterrows():
        if name == "Baseline (TF-IDF)":
            lines.append("\\midrule")
        lines.append(SHORT[name] + " & " + " & ".join(row.fillna("--")) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


DATASET_LABELS = {"movielens": "MovieLens", "amazon": "Amazon", "yelp": "Yelp"}


def review_table(datasets=("amazon", "yelp"), scorer="lgbm"):
    """Amazon and Yelp side by side: item-only, full personalized, uniform negatives."""
    configs = [
        ("item", "popularity", "Item only"),
        ("personalized_pop", "popularity", "+Aff.+Pop."),
        ("personalized_pop", "uniform", "Uniform"),
    ]
    blocks = []
    for dataset in datasets:
        for mode, neg, label in configs:
            df = _load(dataset, mode, neg)
            df = df[df["Scorer"].isin([scorer, "-"])].set_index("Pipeline")
            blocks.append(df.apply(_cell, axis=1).rename((dataset, label)))
    table = pd.concat(blocks, axis=1).reindex(ORDER).dropna(how="all")
    n = len(configs)
    header = " & ".join(
        f"\\multicolumn{{{n}}}{{c}}{{{DATASET_LABELS[d]} ({_users(d)} users)}}" for d in datasets
    )
    rules = "".join(f"\\cmidrule(lr){{{2 + i * n}-{1 + (i + 1) * n}}}" for i in range(len(datasets)))
    lines = [
        "\\begin{tabular}{l" + "c" * (n * len(datasets)) + "}",
        "\\toprule",
        f" & {header} \\\\",
        rules,
        "Pipeline & " + " & ".join(label for _, _, label in configs * len(datasets)) + " \\\\",
        "\\midrule",
    ]
    for name, row in table.iterrows():
        if name == "Baseline (TF-IDF)":
            lines.append("\\midrule")
        lines.append(SHORT[name] + " & " + " & ".join(row.fillna("--")) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def _users(dataset):
    tag = run_tag(dataset, "personalized_pop", "popularity")
    return f"{pd.read_csv(RESULTS_DIR / f'per_user{tag}.csv')['userId'].nunique():,}"


def _delta_vs_sbert(tests, other):
    sbert = "SBERT (all-MiniLM-L6-v2)"
    row = tests[((tests.A == sbert) & (tests.B == other)) | ((tests.A == other) & (tests.B == sbert))]
    r = row.iloc[0]
    delta = r["MeanDiff"] if r["A"] == other else -r["MeanDiff"]  # other - SBERT
    mark = ""
    if r["Verdict"].startswith("equivalent"):
        mark = "$^{\\equiv}$"
    elif r["p_ttest_holm"] < 0.05:
        mark = "$^{*}$"
    return f"{delta:+.3f}{mark}"


def cross_dataset_paired(datasets=("movielens", "amazon", "yelp"), scorers=("lgbm", "mlp")):
    """Every encoder vs SBERT on every dataset and scorer (hard negatives,
    personalized scorer). * = Holm p < 0.05; = = TOST-equivalent at +/-0.01."""
    tests = {
        d: pairwise_tests(pd.read_csv(RESULTS_DIR / f"per_user{run_tag(d, 'personalized_pop', 'popularity')}.csv"))
        for d in datasets
    }
    others = [o for o in ORDER[2:] if o != "SBERT (all-MiniLM-L6-v2)"]
    header = " & ".join(f"\\multicolumn{{{len(scorers)}}}{{c}}{{{DATASET_LABELS[d]}}}" for d in datasets)
    rules = "".join(
        f"\\cmidrule(lr){{{2 + i * len(scorers)}-{1 + (i + 1) * len(scorers)}}}" for i in range(len(datasets))
    )
    scorer_names = {"lgbm": "LGBM", "mlp": "MLP"}
    lines = [
        "\\begin{tabular}{l" + "r" * (len(scorers) * len(datasets)) + "}",
        "\\toprule",
        f"$-$ SBERT & {header} \\\\",
        rules,
        " & " + " & ".join(scorer_names[s] for s in scorers * len(datasets)) + " \\\\",
        "\\midrule",
    ]
    for other in others:
        cells = [
            _delta_vs_sbert(tests[d][tests[d]["Scorer"] == s], other) for d in datasets for s in scorers
        ]
        lines.append(SHORT[other] + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / "main.tex").write_text(main_table("movielens") + "\n")
    (TABLES_DIR / "main_reviews.tex").write_text(review_table() + "\n")
    (TABLES_DIR / "paired_all.tex").write_text(cross_dataset_paired() + "\n")
    print("wrote main.tex, main_reviews.tex, paired_all.tex")


if __name__ == "__main__":
    main()
