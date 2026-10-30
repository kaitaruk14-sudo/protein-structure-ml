"""
Paired bootstrap comparison of two methods from per_chain_val.csv.

write_table() in common.py only compares every method against sequence
separation. This script gives the comparison that matters for the project's
success criterion: neural model vs logistic regression,

    Delta = P_neural - P_logreg   (per-chain difference, paired by chain)

with a 95% CI from 2,000 bootstrap resamples of chains (common.bootstrap_ci).
The criterion is met if the lower bound of the CI is strictly above 0.

Run from the repo root (after model/evaluate.py):
    python model/compare_methods.py                      # neural vs logreg
    python model/compare_methods.py neural separation    # any two columns

Writes results/progress_report/<a>_vs_<b>.md. Uses the validation split only.
"""
import csv
import os
import sys

from common import RESULTS_DIR, bootstrap_ci

import numpy as np

LABELS = {"random": "Random", "separation": "Sequence separation",
          "logreg": "Logistic regression", "neural": "Neural distogram model"}


def main():
    a = sys.argv[1] if len(sys.argv) > 1 else "neural"
    b = sys.argv[2] if len(sys.argv) > 2 else "logreg"
    path = os.path.join(RESULTS_DIR, "per_chain_val.csv")
    rows = list(csv.DictReader(open(path)))
    if a not in rows[0] or b not in rows[0]:
        raise SystemExit(f"per_chain_val.csv has columns {list(rows[0])}; need '{a}' and '{b}'.")
    va = np.array([float(r[a]) for r in rows])
    vb = np.array([float(r[b]) for r in rows])
    n = len(rows)

    ma, loa, hia = bootstrap_ci(va)
    mb, lob, hib = bootstrap_ci(vb)
    dm, dlo, dhi = bootstrap_ci(va - vb)
    better, worse = int((va > vb).sum()), int((va < vb).sum())
    tie = n - better - worse
    ok = dlo > 0

    text = "\n".join([
        f"| Method | Long-range top-L/5 precision (val) | 95% CI |",
        "|---|---|---|",
        f"| {LABELS.get(a, a)} | {100*ma:.2f}% | {100*loa:.2f}% – {100*hia:.2f}% |",
        f"| {LABELS.get(b, b)} | {100*mb:.2f}% | {100*lob:.2f}% – {100*hib:.2f}% |",
        "",
        f"Paired difference (Delta = {a} − {b}): {100*dm:+.2f} pp, "
        f"95% CI {100*dlo:+.2f} to {100*dhi:+.2f} pp "
        f"({n:,} validation chains, 2,000 bootstrap resamples of chains).",
        f"Chains where {a} is better / worse / equal: {better:,} / {worse:,} / {tie:,}.",
        f"Lower bound of the CI strictly above 0: {'YES' if ok else 'NO'}.",
    ])
    print(text)
    out = os.path.join(RESULTS_DIR, f"{a}_vs_{b}.md")
    with open(out, "w") as f:
        f.write(text + "\n")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
