"""
The final evaluation: every frozen method, scored once on the TEST split.

Nothing is trained or tuned here. The logistic regression baseline is the only method that
is refitted (on the TRAIN split, exactly as in baselines.py: same seed, same chains, so it
is the same model); the tree models and neural models are loaded from model/checkpoints/.

Two modes, because the test set may be used only once:

  DRY RUN on validation chains. Reruns exactly the same code on the first N validation
  chains and checks every column against results/progress_report/per_chain_val.csv. It
  must say PASS before the real run, otherwise this script does not reproduce the
  validation pipeline. Nothing of the test split is read.
      python model/test_evaluation.py --split val --limit 150

  THE REAL RUN (once). Needs --confirm-final and refuses to run again if its output exists.
      python model/test_evaluation.py --split test --confirm-final

Evaluation conventions are the same as everywhere: chains up to length 512, contact = Cb-Cb
< 8 A, long-range |i-j| >= 24, top-L/5 precision, 2,000-resample bootstrap CIs, paired
differences. The random tie-breaking streams are reproduced exactly as in baselines.py,
tree_models.py and evaluate.py, which is what makes the dry run match to four decimals.

Writes (test): results/test_final/per_chain_test.csv, results_table_test.md, test_summary.json
Writes (dry run): tmp/test_dry_run/ (git-ignored)
"""
import argparse
import csv
import json
import os
import random
import time

from common import (CHECKPOINT_DIR, CONTACT_CUTOFF, EVAL_MAX_LEN, RESULTS_DIR, ROOT, bootstrap_ci,
                    distance_matrix, load_chain, load_index, long_range_pairs,
                    top_l5_precision)  # common first: macOS OpenMP fix

import numpy as np

import baselines
import tree_models

SEED = 0
# paired comparisons written under the main table (only those whose columns exist)
PAIRS = [("neural_large", "neural_medium_long"), ("neural_large", "logreg_rich"),
         ("neural_large", "gbt"), ("neural_large", "logreg"), ("neural_medium_long", "logreg_rich"),
         ("logreg_rich", "logreg")]
LABELS = {"random": "Random", "separation": "Sequence separation", "logreg": "Logistic regression",
          "logreg_rich": "Logistic regression, rich features",
          "gbt": "Gradient-boosted trees, rich features", "neural": "Neural distogram model (small)"}


def label(col):
    if col in LABELS:
        return LABELS[col]
    return "Neural distogram model (" + col[len("neural_"):].replace("_", " ") + ")"


def column_of(run_name):
    return "neural" if run_name == "basic" else f"neural_{run_name}"


def score_classical(rows, limit):
    """random, separation, logreg, logreg_rich, gbt -> dict of lists, plus the chains used."""
    import joblib
    # logistic regression: refit on TRAIN exactly as baselines.py does (same seed and order)
    random.seed(baselines.SEED)
    rng = np.random.default_rng(baselines.SEED)
    train_rows = load_index("train")
    print(f"Refitting the logistic regression on {len(train_rows):,} train chains "
          "(same recipe as baselines.py)...")
    f_train, y_train, _ = baselines.build_training_pairs(train_rows, rng)
    lr = baselines.LogisticRegression()
    lr.fit(f_train, y_train)

    trees = {}
    for name in ("logreg_rich", "gbt"):          # dict order = order used in tree_models.py
        path = os.path.join(CHECKPOINT_DIR, f"{name}_best.joblib")
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}")
        trees[name] = joblib.load(path)

    eval_rng = np.random.default_rng(baselines.SEED)    # random / separation / logreg stream
    tree_rng = np.random.default_rng(tree_models.SEED)  # logreg_rich / gbt stream
    out = {c: [] for c in ("random", "separation", "logreg", "logreg_rich", "gbt")}
    used, lengths = [], []
    n_pairs = n_contacts = 0
    t0 = time.time()
    for k, r in enumerate(rows):
        if limit and len(used) >= limit:
            break
        tokens, coords, mask = load_chain(r["entity_id"])
        i, j = long_range_pairs(mask)
        if len(i) == 0:
            continue
        L = len(tokens)
        true = distance_matrix(coords)[i, j] < CONTACT_CUTOFF
        n_pairs += len(true); n_contacts += int(true.sum())
        out["random"].append(top_l5_precision(eval_rng.random(len(i)), true, L, eval_rng))
        out["separation"].append(top_l5_precision(-(j - i).astype(float), true, L, eval_rng))
        out["logreg"].append(top_l5_precision(lr.logits(baselines.pair_features(tokens, i, j)),
                                              true, L, eval_rng))
        X = tree_models.pair_features(tokens, i, j)
        for name, model in trees.items():
            out[name].append(top_l5_precision(model.predict_proba(X)[:, 1], true, L, tree_rng))
        used.append(r); lengths.append(L)
        if len(used) % 500 == 0:
            print(f"  {len(used):,} chains ({time.time() - t0:.0f}s)")
    return out, used, lengths, n_pairs, n_contacts


def score_neural(run_names, used, device):
    from evaluate import evaluate_chains
    from model import count_parameters, load_model
    out, params = {}, {}
    for name in run_names:
        path = os.path.join(CHECKPOINT_DIR, f"{name}_best.pt")
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}")
        model, _ = load_model(path, device)
        params[column_of(name)] = count_parameters(model)
        t0 = time.time()
        print(f"Scoring neural model '{name}' ({params[column_of(name)]:,} parameters) "
              f"on {len(used):,} chains...")
        _, prec, _, _ = evaluate_chains(model, used, device)
        assert len(prec) == len(used), "chain count mismatch for " + name
        out[column_of(name)] = prec
        print(f"  done in {(time.time() - t0) / 60:.1f} min")
        del model
    return out, params


def write_table(per_chain, columns, split, out_path):
    lines = [f"| Method | Long-range top-L/5 precision ({split}) | 95% CI | "
             "Difference vs sequence separation (95% CI) |", "|---|---|---|---|"]
    base = np.asarray(per_chain["separation"])
    for c in columns:
        v = np.asarray(per_chain[c])
        m, lo, hi = bootstrap_ci(v)
        diff = ""
        if c != "separation":
            dm, dlo, dhi = bootstrap_ci(v - base)
            diff = f"{100*dm:+.2f} pp ({100*dlo:+.2f} to {100*dhi:+.2f})"
        lines.append(f"| {label(c)} | {100*m:.2f}% | {100*lo:.2f}% – {100*hi:.2f}% | {diff} |")
    n = len(base)
    lines += ["", f"Evaluated on {n:,} {split} chains (length ≤ {EVAL_MAX_LEN}). Contact = Cβ–Cβ < "
              f"{CONTACT_CUTOFF:.0f} Å; long-range = |i−j| ≥ 24. CIs: 2,000 bootstrap resamples of chains; "
              "pp = percentage points.", "",
              "Paired differences (A − B), 95% CI from 2,000 bootstrap resamples of chains:", "",
              "| A − B | Difference | Chains A better / worse / equal |", "|---|---|---|"]
    for a, b in PAIRS:
        if a in per_chain and b in per_chain:
            d = np.asarray(per_chain[a]) - np.asarray(per_chain[b])
            dm, dlo, dhi = bootstrap_ci(d)
            lines.append(f"| {label(a)} − {label(b)} | {100*dm:+.2f} pp ({100*dlo:+.2f} to {100*dhi:+.2f}) | "
                         f"{int((d > 0).sum()):,} / {int((d < 0).sum()):,} / {int((d == 0).sum()):,} |")
    text = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print("\n" + text)


def dry_run_check(rows_out, columns):
    """Compare every column with results/progress_report/per_chain_val.csv (first N rows)."""
    ref_path = os.path.join(RESULTS_DIR, "per_chain_val.csv")
    ref = list(csv.DictReader(open(ref_path)))
    ok = True
    print("\nDRY-RUN CHECK against per_chain_val.csv (first "
          f"{len(rows_out)} chains, formatted to 4 decimals):")
    if [r["entity_id"] for r in rows_out] != [r["entity_id"] for r in ref[:len(rows_out)]]:
        print("  FAIL: chain ids / order differ from per_chain_val.csv")
        return False
    for c in columns:
        if c not in ref[0]:
            print(f"  skipped {c}: not in per_chain_val.csv yet")
            continue
        bad = [k for k, r in enumerate(rows_out) if r[c] != ref[k][c]]
        print(f"  {c:20s} {'PASS' if not bad else f'FAIL ({len(bad)} of {len(rows_out)} differ, first: chain {rows_out[bad[0]]['entity_id']}: {rows_out[bad[0]][c]} vs {ref[bad[0]][c]})'}")
        ok &= not bad
    print("\nDRY RUN " + ("PASSED: this script reproduces the validation results."
                          if ok else "FAILED: do NOT run on test until this is fixed."))
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--limit", type=int, default=0, help="dry run only: first N chains of the split")
    p.add_argument("--confirm-final", action="store_true",
                   help="required for --split test: the one and only test-set evaluation")
    p.add_argument("--neural", nargs="*", default=["medium_long", "large"],
                   help="neural checkpoints to score, model/checkpoints/<name>_best.pt (default: medium_long large)")
    p.add_argument("--rerun", action="store_true", help="allow overwriting an existing test result (do not)")
    args = p.parse_args()

    if args.split == "test":
        if not args.confirm_final:
            raise SystemExit("The test set is used once. Add --confirm-final when the models are frozen "
                             "and the dry run (--split val --limit 150) has passed.")
        if args.limit:
            raise SystemExit("--limit is only for the validation dry run.")
        out_dir = os.path.join(ROOT, "results", "test_final")
        if os.path.exists(os.path.join(out_dir, "per_chain_test.csv")) and not args.rerun:
            raise SystemExit(f"{out_dir}/per_chain_test.csv already exists: the test set has been evaluated. "
                             "Not running again.")
    else:
        out_dir = os.path.join(ROOT, "tmp", "test_dry_run")
    os.makedirs(out_dir, exist_ok=True)

    from train import get_device
    device = get_device()
    rows = load_index(args.split, max_len=EVAL_MAX_LEN)
    print(f"Split: {args.split}. {len(rows):,} chains after the filters (length <= {EVAL_MAX_LEN})."
          + (f" Dry run on the first {args.limit}." if args.limit else ""))
    t0 = time.time()
    classical, used, lengths, n_pairs, n_contacts = score_classical(rows, args.limit)
    neural, params = score_neural(args.neural, used, device)

    columns = ["random", "separation", "logreg", "logreg_rich", "gbt"] + list(neural)
    per_chain = {**classical, **neural}
    rows_out = []
    for k, r in enumerate(used):
        row = {"entity_id": r["entity_id"], "length": lengths[k]}
        row.update({c: f"{per_chain[c][k]:.4f}" for c in columns})
        rows_out.append(row)
    csv_path = os.path.join(out_dir, f"per_chain_{args.split}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)
    print(f"\nPer-chain results: {csv_path}")

    if args.split == "val":
        dry_run_check(rows_out, columns)
        return

    write_table({c: [float(r[c]) for r in rows_out] for c in columns}, columns, "test",
                os.path.join(out_dir, "results_table_test.md"))
    with open(os.path.join(out_dir, "test_summary.json"), "w") as f:
        json.dump({"n_test_chains": len(used), "test_long_range_pairs": int(n_pairs),
                   "test_long_range_contact_rate_pct": round(100 * n_contacts / n_pairs, 3),
                   "neural_models": args.neural, "parameters": params,
                   "minutes": round((time.time() - t0) / 60, 1)}, f, indent=2)
    print(f"\nTest evaluation done in {(time.time() - t0) / 60:.1f} min. Files in {out_dir}")


if __name__ == "__main__":
    main()
