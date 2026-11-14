"""
Summarise the 3D-structure scoring of several models on the SAME validation proteins.

Reads results/structure_3d/<run_name>/summary.csv (written by model/structure_3d.py) and reports
  1. per model: mean / median RMSD, TM-score, lDDT, confidence-vs-lDDT correlation, oracle check
  2. paired per-protein differences between models, with 2,000-resample bootstrap 95% CIs
  3. length dependence: rank correlation of each score with protein length (bootstrap CI), and
     mean score for short / medium / long thirds

Run from the repo root (validation proteins only):
    python model/analyze_structure_3d.py | tee results/structure_3d/analysis.txt
    python model/analyze_structure_3d.py --runs medium_long long medium
"""
import argparse
import csv
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_BOOT = 2000
SCORES = [("rmsd_pred", "RMSD (A, lower is better)"), ("tm_pred", "TM-score (higher is better)"),
          ("lddt_pred", "lDDT (higher is better)")]
ORACLE_FAIL = 3.0       # oracle RMSD above this (A) = the reconstruction itself failed


def load(run):
    path = os.path.join(ROOT, "results", "structure_3d", run, "summary.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {r["entity_id"]: {k: (float(v) if k != "entity_id" and k != "mirror_pred" else v)
                             for k, v in r.items()} for r in rows}


def ranks(x):
    """average ranks (handles ties)"""
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):
        idx = x == v
        if idx.sum() > 1:
            r[idx] = r[idx].mean()
    return r


def spearman(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def boot_ci(stat, n, rng):
    """percentile bootstrap over proteins; stat(idx) -> number"""
    vals = np.array([stat(rng.integers(0, n, n)) for _ in range(N_BOOT)])
    return np.nanpercentile(vals, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["medium_long", "long", "medium"])
    args = ap.parse_args()
    data = {r: load(r) for r in args.runs}
    ids = sorted(set.intersection(*[set(d) for d in data.values()]))
    for r in args.runs:
        extra = set(data[r]) - set(ids)
        if extra:
            print(f"note: {r} has {len(extra)} proteins the other runs do not; they are ignored")
    n = len(ids)
    print(f"{n} validation proteins scored by every model: {', '.join(args.runs)}")
    print("(same proteins for every model, so differences below are paired)\n")
    col = {r: {k: np.array([data[r][i][k] for i in ids]) for k in data[r][ids[0]] if k not in ("entity_id", "mirror_pred")}
           for r in args.runs}
    length = col[args.runs[0]]["length"]
    rng = np.random.default_rng(0)

    print("1. PER MODEL (mean / median over proteins)")
    for r in args.runs:
        c = col[r]
        print(f"  {r}:")
        for k, label in SCORES:
            print(f"    {label:32s} mean {c[k].mean():6.3f}   median {np.median(c[k]):6.3f}")
        print(f"    {'confidence vs lDDT (Spearman)':32s} mean {np.nanmean(c['confidence_vs_lddt_spearman']):+6.3f}   "
              f"median {np.nanmedian(c['confidence_vs_lddt_spearman']):+6.3f}")
        print(f"    proteins with TM >= 0.5: {int((c['tm_pred'] >= 0.5).sum())} / {n}")
    print("\n  Oracle (same reconstruction on the TRUE distances; same for every model):")
    o = col[args.runs[0]]
    bad = [i for i, v in zip(ids, o["rmsd_oracle"]) if v > ORACLE_FAIL]
    print(f"    mean RMSD {o['rmsd_oracle'].mean():.2f} A, median {np.median(o['rmsd_oracle']):.2f} A; "
          f"mean lDDT {o['lddt_oracle'].mean():.3f}")
    print(f"    reconstruction failures (oracle RMSD > {ORACLE_FAIL} A): {len(bad)} {bad}")

    print(f"\n2. PAIRED DIFFERENCES (A minus B, mean over proteins, {N_BOOT}-resample bootstrap 95% CI)")
    for ia in range(len(args.runs)):
        for ib in range(ia + 1, len(args.runs)):
            a, b = args.runs[ia], args.runs[ib]
            print(f"  {a} - {b}:")
            for k, label in SCORES:
                d = col[a][k] - col[b][k]
                lo, hi = boot_ci(lambda idx: d[idx].mean(), n, rng)
                wins = int((d < 0).sum()) if k == "rmsd_pred" else int((d > 0).sum())
                sig = "CI excludes 0" if (lo > 0 or hi < 0) else "CI includes 0"
                print(f"    {label:32s} {d.mean():+7.3f}  [{lo:+.3f}, {hi:+.3f}]  ({sig}); {a} better on {wins}/{n}")

    print("\n3. LENGTH DEPENDENCE")
    print(f"  protein lengths: min {int(length.min())}, median {int(np.median(length))}, max {int(length.max())}")
    thirds = np.array_split(np.argsort(length, kind="stable"), 3)
    names = ["shortest third", "middle third", "longest third"]
    for r in args.runs:
        print(f"  {r}: rank correlation of score with length (bootstrap 95% CI)")
        for k, label in SCORES:
            x = col[r][k]
            rho = spearman(length, x)
            lo, hi = boot_ci(lambda idx: spearman(length[idx], x[idx]), n, rng)
            print(f"    {label:32s} rho {rho:+.2f}  [{lo:+.2f}, {hi:+.2f}]")
        for nm, idx in zip(names, thirds):
            print(f"    {nm:15s} (L {int(length[idx].min())}-{int(length[idx].max())}, n={len(idx)}): "
                  f"RMSD {col[r]['rmsd_pred'][idx].mean():5.1f} A, TM {col[r]['tm_pred'][idx].mean():.3f}, "
                  f"lDDT {col[r]['lddt_pred'][idx].mean():.3f}")
    print("\nCaveat: lengths are limited to 50-300 residues (structure_3d.py --max_len), so this says "
          "nothing about longer proteins. Spearman rho near +-1 = strong monotone trend.")


if __name__ == "__main__":
    main()
