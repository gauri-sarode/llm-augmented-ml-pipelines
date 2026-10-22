"""Paired significance and equivalence tests between pipelines.

Every pipeline is evaluated on the same users, splits, and candidate lists
per seed, so comparisons are paired. For each pipeline pair (same scorer) we
average each user's NDCG@10 over seeds, take per-user differences, and run:

- a paired t-test and a Wilcoxon signed-rank test for any difference
  (Holm-corrected across all pairs within a scorer), and
- TOST equivalence (two one-sided paired t-tests) against a margin of
  +/- EQUIV_MARGIN NDCG@10, so "no separation" claims are backed by a test
  of equivalence rather than overlapping confidence intervals.

Usage: python -m src.stats results/per_user<tag>.csv
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Smallest NDCG@10 difference treated as practically meaningful. Fixed before
# looking at the main results: 0.01 is half of the MovieLens TF-IDF-vs-SBERT gap in pilot runs.
EQUIV_MARGIN = 0.01


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def tost_p(diffs, margin):
    """p-value of TOST equivalence: max of the two one-sided paired t-tests."""
    lower = stats.ttest_1samp(diffs, -margin, alternative="greater").pvalue
    upper = stats.ttest_1samp(diffs, margin, alternative="less").pvalue
    return max(lower, upper)


def pairwise_tests(per_user: pd.DataFrame, margin: float = EQUIV_MARGIN) -> pd.DataFrame:
    user_means = per_user.groupby(["Pipeline", "Scorer", "userId"])["NDCG"].mean()
    rows = []
    for scorer in [s for s in per_user["Scorer"].unique() if s != "-"]:
        # Baselines (Scorer "-") are compared against every scorer's pipelines.
        pipes = [
            (p, s)
            for p, s in per_user[["Pipeline", "Scorer"]].drop_duplicates().itertuples(index=False)
            if s in (scorer, "-")
        ]
        block = []
        for (a, sa), (b, sb) in itertools.combinations(pipes, 2):
            if sa == sb == "-":
                continue  # baseline-vs-baseline isn't a feature-source comparison
            x, y = user_means.loc[(a, sa)], user_means.loc[(b, sb)]
            x, y = x.align(y, join="inner")
            d = (x - y).to_numpy()
            block.append(
                {
                    "Scorer": scorer,
                    "A": a,
                    "B": b,
                    "Users": len(d),
                    "MeanDiff": d.mean(),
                    "CI95": 1.96 * d.std(ddof=1) / np.sqrt(len(d)),
                    "p_ttest": stats.ttest_1samp(d, 0.0).pvalue,
                    "p_wilcoxon": stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0,
                    "p_tost": tost_p(d, margin),
                }
            )
        block = pd.DataFrame(block)
        block["p_ttest_holm"] = holm(block["p_ttest"])
        block["p_tost_holm"] = holm(block["p_tost"])
        rows.append(block)
    out = pd.concat(rows, ignore_index=True)
    # A difference can be both detectable (t-test) and inside the margin
    # (TOST) when power is high; report that as negligible, not "different".
    diff, equiv = out["p_ttest_holm"] < 0.05, out["p_tost_holm"] < 0.05
    out["Verdict"] = np.select(
        [diff & equiv, diff, equiv],
        [f"negligible (<{margin})", "different", f"equivalent (+/-{margin})"],
        default="inconclusive",
    )
    return out


def main():
    path = Path(sys.argv[1])
    out = pairwise_tests(pd.read_csv(path))
    out_path = path.with_name(path.name.replace("per_user", "paired_tests"))
    out.to_csv(out_path, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(out[["Scorer", "A", "B", "MeanDiff", "CI95", "p_ttest_holm", "p_tost_holm", "Verdict"]])
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
