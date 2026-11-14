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
    python model/error_analysis.py            # version 1: sequence separation, logreg, neural
    python model/error_analysis.py --v2       # version 2: adds gbt and the long / medium neural models

Writes to results/progress_report/error_analysis/ (version 2 files carry the suffix _v2, so the
version 1 files are never overwritten):
    error_analysis_table.md / .csv
    fig_precision_by_group.png      precision per group and method (95% CI)
    fig_neural_minus_logreg.png     paired difference per group (95% CI)
Version 2 compares every method with logistic regression (paired, per group and overall, with a
cluster bootstrap robustness check for each) and needs the columns gbt, neural_long, neural_medium
and neural_medium_long in per_chain_val.csv.
Validation split only. Nothing here loads the test split.
"""
import argparse
import csv
import os
from collections import Counter

from common import DATA_DIR, RESULTS_DIR, bootstrap_ci

import numpy as np

OUT_DIR = os.path.join(RESULTS_DIR, "error_analysis")
METHODS = [("separation", "Sequence separation"), ("logreg", "Logistic regression"),
           ("neural", "Neural distogram model")]
COMPARE = ["neural"]                    # methods compared with logistic regression (paired)
METHODS_V2 = [("separation", "Sequence separation"), ("logreg", "Logistic regression"),
              ("gbt", "Gradient-boosted trees"), ("neural", "Neural (small)"),
              ("neural_long", "Neural (small, long)"), ("neural_medium", "Neural (medium)"),
              ("neural_medium_long", "Neural (medium, long)")]
COMPARE_V2 = ["gbt", "neural", "neural_long", "neural_medium", "neural_medium_long"]
COLOURS_V2 = {"separation": "#c9ccd1", "logreg": "#b7d3f5", "gbt": "#e0a458", "neural": "#7fb0ea",
              "neural_long": "#3f86d4", "neural_medium": "#1f5aa6", "neural_medium_long": "#0d366b"}
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", action="store_true",
                    help="version 2: also gbt and the long/medium neural models; outputs get the suffix _v2")
    args = ap.parse_args()
    methods, compare, suffix = (METHODS_V2, COMPARE_V2, "_v2") if args.v2 else (METHODS, COMPARE, "")
    label_of = dict(methods)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(os.path.join(RESULTS_DIR, "per_chain_val.csv"))))
    missing = [m for m, _ in methods if m not in rows[0]]
    if missing:
        raise SystemExit(f"per_chain_val.csv has no column(s) {missing}. Run python model/evaluate.py first.")
    index = {r["entity_id"]: r for r in csv.DictReader(
        open(os.path.join(DATA_DIR, "backbone_index.tsv")), delimiter="\t")}
    cluster = np.array([index[r["entity_id"]]["cluster_id"] for r in rows])
    assert all(index[r["entity_id"]]["split"] == "val" for r in rows), "non-val chain found"
    fam_size_of = Counter(r["cluster_id"] for r in index.values())
    family = np.array([fam_size_of[c] for c in cluster])
    length = np.array([int(r["length"]) for r in rows])
    prec = {m: np.array([float(r[m]) for r in rows]) for m, _ in methods}
    diff_all = prec["neural"] - prec["logreg"]
    diffs = {m: prec[m] - prec["logreg"] for m in compare}

    groups = [("All chains", np.ones(len(rows), bool))]
    groups += [(f"Length {lab}", (length >= lo) & (length < hi)) for lab, lo, hi in LENGTH_GROUPS]
    groups += [(f"Family size {lab}", (family >= lo) & (family < hi)) for lab, lo, hi in FAMILY_GROUPS]

    table = []
    for name, sel in groups:
        rec = {"group": name, "chains": int(sel.sum()), "clusters": len(set(cluster[sel]))}
        for m, _ in methods:
            rec[m] = bootstrap_ci(prec[m][sel])
        rec["delta"] = bootstrap_ci(diff_all[sel])
        for m in compare:
            rec["delta_" + m] = bootstrap_ci(diffs[m][sel])
        table.append(rec)

    # ---- tables ----
    os.makedirs(OUT_DIR, exist_ok=True)
    pct = lambda t: f"{100*t[0]:.2f}% ({100*t[1]:.2f}–{100*t[2]:.2f})"
    dpp = lambda t: f"{100*t[0]:+.2f} pp ({100*t[1]:+.2f} to {100*t[2]:+.2f})"
    verdict = lambda t: "neural better" if t[1] > 0 else ("logreg better" if t[2] < 0 else "no clear difference")
    if not args.v2:
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
    else:
        bold = lambda t, txt: f"**{txt}**" if (t[1] > 0 or t[2] < 0) else txt
        lines = ["| Group | Chains (clusters) | " + " | ".join(label_of[m] for m, _ in methods) + " |",
                 "|---|---|" + "---|" * len(methods)]
        for r in table:
            lines.append(f"| {r['group']} | {r['chains']:,} ({r['clusters']:,}) | "
                         + " | ".join(pct(r[m]) for m, _ in methods) + " |")
        lines += ["", "Paired difference from logistic regression (percentage points, 95% CI); "
                  "bold = the CI excludes 0.", "",
                  "| Group | " + " | ".join(f"{label_of[m]} − logreg" for m in compare) + " |",
                  "|---|" + "---|" * len(compare)]
        for r in table:
            lines.append(f"| {r['group']} | " + " | ".join(bold(r["delta_" + m], dpp(r["delta_" + m]))
                                                            for m in compare) + " |")
        lines += ["", "Long-range top-L/5 precision on validation chains, mean with 95% CI from 2,000 bootstrap "
                  "resamples of chains; differences are paired per chain. "
                  "Family size = chains in the same MMseqs2 cluster (30% identity) in backbone_index.tsv; "
                  "all clusters sit entirely inside the validation split, so no validation family appears in training.",
                  "", "Robustness: overall difference from logistic regression with a cluster bootstrap "
                  f"(resampling {len(set(cluster)):,} clusters):"]
        for m in compare:
            cm, clo, chi = cluster_bootstrap_ci(diffs[m], cluster)
            lines.append(f"- {label_of[m]}: {100*cm:+.2f} pp ({100*clo:+.2f} to {100*chi:+.2f})")
        text = "\n".join(lines)
        open(os.path.join(OUT_DIR, f"error_analysis_table{suffix}.md"), "w").write(text + "\n")
        with open(os.path.join(OUT_DIR, f"error_analysis_table{suffix}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["group", "chains", "clusters"] + [f"{m}_{k}" for m, _ in methods for k in ("mean", "lo", "hi")]
                       + [f"delta_{m}_{k}" for m in compare for k in ("mean", "lo", "hi")])
            for r in table:
                w.writerow([r["group"], r["chains"], r["clusters"]]
                           + [round(100 * v, 3) for m, _ in methods for v in r[m]]
                           + [round(100 * v, 3) for m in compare for v in r["delta_" + m]])
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
    top = max(100 * r[m][2] for r in table for m, _ in methods) * 1.12
    shades = COLOURS_V2 if args.v2 else SHADES
    nm = len(methods)
    fig, axes = plt.subplots(1, 2, figsize=(10 if not args.v2 else 12, 3.8 if not args.v2 else 4.4), sharey=True)
    w = 0.26 if not args.v2 else 0.8 / nm
    for ax, (title, recs) in zip(axes, panels):
        x = np.arange(len(recs))
        for k, (m, label) in enumerate(methods):
            mean = np.array([100 * r[m][0] for r in recs])
            err = np.array([[100 * (r[m][0] - r[m][1]) for r in recs], [100 * (r[m][2] - r[m][0]) for r in recs]])
            ax.bar(x + (k - (nm - 1) / 2) * w, mean, w, color=shades[m], label=label,
                   yerr=err, error_kw={"ecolor": INK2, "capsize": 2, "lw": 1})
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['group'].split(' ', 1)[1] if r['group'].startswith('Length') else r['group'][12:]}\n(n={r['chains']:,})"
                            for r in recs], fontsize=9)
        ax.set_title(title, fontsize=11); ax.grid(axis="x", visible=False); ax.set_ylim(0, top)
    axes[0].set_ylabel("Long-range top-L/5 precision (%), 95% CI")
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=3 if not args.v2 else 4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07 if not args.v2 else 0.12, 1, 1))
    fig.savefig(os.path.join(OUT_DIR, f"fig_precision_by_group{suffix}.png"), dpi=200)
    plt.close(fig)

    if not args.v2:
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
    else:
        fig, ax = plt.subplots(figsize=(8.5, 6.6))
        y = np.arange(len(table))[::-1].astype(float)
        step = 0.7 / len(compare)
        for k, m in enumerate(compare):
            off = (k - (len(compare) - 1) / 2) * step
            for yy, r in zip(y, table):
                mm, lo, hi = (100 * v for v in r["delta_" + m])
                ax.errorbar(mm, yy + off, xerr=[[mm - lo], [hi - mm]], fmt="o", color=COLOURS_V2[m],
                            ecolor=COLOURS_V2[m], elinewidth=1.6, capsize=2, ms=4,
                            label=label_of[m] if yy == y[0] else None)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_yticks(y); ax.set_yticklabels([r["group"] for r in table]); ax.grid(axis="y", visible=False)
        ax.set_xlabel("Method − logistic regression, long-range top-L/5 precision (pp), paired 95% CI")
        ax.set_title("Where each method beats logistic regression")
        handles, labs = ax.get_legend_handles_labels()
        fig.legend(handles, labs, loc="lower center", ncol=3, frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(os.path.join(OUT_DIR, f"fig_method_minus_logreg{suffix}.png"), dpi=200)
        plt.close(fig)
    print(f"\nSaved table and figures in {OUT_DIR}")


if __name__ == "__main__":
    main()
