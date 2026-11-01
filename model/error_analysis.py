"""
Error analysis on the validation split: where does the neural model beat
logistic regression, and where not?

Breaks the per-chain long-range top-L/5 precision (per_chain_val.csv) down by
  * chain length         (50-99, 100-199, 200-299, 300-512 residues)
  * family size          (number of chains sharing the chain's MMseqs2 cluster
                          in backbone_index.tsv: singleton, 2-5, 6-20, 21+)
For every group: mean precision with a 95% bootstrap CI for each method, and
the paired difference neural - logreg with its 95% CI (common.bootstrap_ci on
per-chain differences, as in common.write_table).

As a robustness check it also reports the overall neural - logreg difference
with a cluster bootstrap (resampling whole clusters), because chains from the
same family are not independent and the chain bootstrap can be too optimistic.

Run from the repo root (after model/evaluate.py has added the 'neural' column):
    python model/error_analysis.py

Writes to results/progress_report/error_analysis/:
    error_analysis_table.md / .csv
    fig_precision_by_group.png      precision per group and method (95% CI)
    fig_neural_minus_logreg.png     paired difference per group (95% CI)
Validation split only. Nothing here loads the test split.
"""
import csv
import os
from collections import Counter

from common import DATA_DIR, RESULTS_DIR, bootstrap_ci

import numpy as np

OUT_DIR = os.path.join(RESULTS_DIR, "error_analysis")
METHODS = [("separation", "Sequence separation"), ("logreg", "Logistic regression"),
           ("neural", "Neural distogram model")]
LENGTH_GROUPS = [("50–99", 0, 100), ("100–199", 100, 200), ("200–299", 200, 300),
                 ("300–512", 300, 10**6)]
FAMILY_GROUPS = [("1 (singleton)", 1, 2), ("2–5", 2, 6), ("6–20", 6, 21), ("21+", 21, 10**9)]


def cluster_bootstrap_ci(diff, clusters, n_boot=2000, seed=0):
    """Mean of per-chain differences and 95% CI, resampling whole clusters."""
    ids, inv = np.unique(clusters, return_inverse=True)
    s = np.bincount(inv, weights=diff, minlength=len(ids))
    n = np.bincount(inv, minlength=len(ids)).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ids), (n_boot, len(ids)))
    means = s[pick].sum(1) / n[pick].sum(1)
    return diff.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(os.path.join(RESULTS_DIR, "per_chain_val.csv"))))
    if "neural" not in rows[0]:
        raise SystemExit("per_chain_val.csv has no 'neural' column. Run python model/evaluate.py first.")
    index = {r["entity_id"]: r for r in csv.DictReader(
        open(os.path.join(DATA_DIR, "backbone_index.tsv")), delimiter="\t")}
    cluster = np.array([index[r["entity_id"]]["cluster_id"] for r in rows])
    assert all(index[r["entity_id"]]["split"] == "val" for r in rows), "non-val chain found"
    fam_size_of = Counter(r["cluster_id"] for r in index.values())
    family = np.array([fam_size_of[c] for c in cluster])
    length = np.array([int(r["length"]) for r in rows])
    prec = {m: np.array([float(r[m]) for r in rows]) for m, _ in METHODS}
    diff_all = prec["neural"] - prec["logreg"]

    groups = [("All chains", np.ones(len(rows), bool))]
    groups += [(f"Length {lab}", (length >= lo) & (length < hi)) for lab, lo, hi in LENGTH_GROUPS]
    groups += [(f"Family size {lab}", (family >= lo) & (family < hi)) for lab, lo, hi in FAMILY_GROUPS]

    table = []
    for name, sel in groups:
        rec = {"group": name, "chains": int(sel.sum()), "clusters": len(set(cluster[sel]))}
        for m, _ in METHODS:
            rec[m] = bootstrap_ci(prec[m][sel])
        rec["delta"] = bootstrap_ci(diff_all[sel])
        table.append(rec)

    # ---- tables ----
    os.makedirs(OUT_DIR, exist_ok=True)
    pct = lambda t: f"{100*t[0]:.2f}% ({100*t[1]:.2f}–{100*t[2]:.2f})"
    dpp = lambda t: f"{100*t[0]:+.2f} pp ({100*t[1]:+.2f} to {100*t[2]:+.2f})"
    verdict = lambda t: "neural better" if t[1] > 0 else ("logreg better" if t[2] < 0 else "no clear difference")
    lines = ["| Group | Chains (clusters) | Sequence separation | Logistic regression | "
             "Neural | Neural − logreg (paired 95% CI) | Verdict |", "|---|---|---|---|---|---|---|"]
    for r in table:
        lines.append(f"| {r['group']} | {r['chains']:,} ({r['clusters']:,}) | {pct(r['separation'])} | "
                     f"{pct(r['logreg'])} | {pct(r['neural'])} | {dpp(r['delta'])} | {verdict(r['delta'])} |")
    cm, clo, chi = cluster_bootstrap_ci(diff_all, cluster)
    lines += ["", "Long-range top-L/5 precision on validation chains, mean with 95% CI from 2,000 bootstrap "
              "resamples of chains; Δ is the paired per-chain difference. "
              "Family size = chains in the same MMseqs2 cluster (30% identity) in backbone_index.tsv; "
              "all clusters sit entirely inside the validation split, so no validation family appears in training.",
              "", f"Robustness: overall Δ with a cluster bootstrap (resampling {len(set(cluster)):,} clusters): "
              f"{100*cm:+.2f} pp ({100*clo:+.2f} to {100*chi:+.2f})."]
    text = "\n".join(lines)
    open(os.path.join(OUT_DIR, "error_analysis_table.md"), "w").write(text + "\n")
    with open(os.path.join(OUT_DIR, "error_analysis_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "chains", "clusters"] + [f"{m}_{k}" for m, _ in METHODS for k in ("mean", "lo", "hi")]
                   + ["delta_mean", "delta_lo", "delta_hi"])
        for r in table:
            w.writerow([r["group"], r["chains"], r["clusters"]]
                       + [round(100 * v, 3) for m, _ in METHODS for v in r[m]]
                       + [round(100 * v, 3) for v in r["delta"]])
    print(text)

    # ---- figures ----
    SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    SHADES = {"separation": "#b7d3f5", "logreg": "#5a9de3", "neural": "#0d366b"}
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
                         "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
                         "axes.titlelocation": "left"})

    # 1. precision per group and method
    panels = [("By chain length (residues)", table[1:5]), ("By family size (chains per cluster)", table[5:9])]
    top = max(100 * r[m][2] for r in table for m, _ in METHODS) * 1.12
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    w = 0.26
    for ax, (title, recs) in zip(axes, panels):
        x = np.arange(len(recs))
        for k, (m, label) in enumerate(METHODS):
            mean = np.array([100 * r[m][0] for r in recs])
            err = np.array([[100 * (r[m][0] - r[m][1]) for r in recs], [100 * (r[m][2] - r[m][0]) for r in recs]])
            ax.bar(x + (k - 1) * w, mean, w, color=SHADES[m], label=label,
                   yerr=err, error_kw={"ecolor": INK2, "capsize": 2, "lw": 1})
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['group'].split(' ', 1)[1] if r['group'].startswith('Length') else r['group'][12:]}\n(n={r['chains']:,})"
                            for r in recs], fontsize=9)
        ax.set_title(title, fontsize=11); ax.grid(axis="x", visible=False); ax.set_ylim(0, top)
    axes[0].set_ylabel("Long-range top-L/5 precision (%), 95% CI")
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(os.path.join(OUT_DIR, "fig_precision_by_group.png"), dpi=200)
    plt.close(fig)

    # 2. paired difference neural - logreg
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = [r["group"] for r in table]
    y = np.arange(len(table))[::-1]
    for yy, r in zip(y, table):
        m, lo, hi = (100 * v for v in r["delta"])
        ax.errorbar(m, yy, xerr=[[m - lo], [hi - m]], fmt="o", color=SHADES["neural"],
                    ecolor=SHADES["logreg"], elinewidth=2, capsize=3, ms=5)
        ax.text(hi + 0.15, yy, f"{m:+.2f} pp", va="center", fontsize=9, color=INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Neural − logistic regression, long-range top-L/5 precision (pp), paired 95% CI")
    ax.set_title("Where the neural model beats logistic regression")
    ax.set_xlim(min(0, min(100 * r["delta"][1] for r in table)) - 0.5,
                max(100 * r["delta"][2] for r in table) + 1.6)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig_neural_minus_logreg.png"), dpi=200)
    plt.close(fig)
    print(f"\nSaved table and figures in {OUT_DIR}")


if __name__ == "__main__":
    main()
