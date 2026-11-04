"""
Alternative models: is it the FEATURES or the MODEL?

Two models trained on the same, richer pair features:
  1. logreg_rich - logistic regression (a linear model)
  2. gbt         - gradient-boosted decision trees
                   (scikit-learn HistGradientBoostingClassifier, a non-linear model)

Compared with the original logistic regression (amino-acid pair + separation only):
  logreg  -> logreg_rich : what the richer features add
  logreg_rich -> gbt     : what a more flexible model adds on the same features

Features for a long-range residue pair (i, j), i < j:
  * amino-acid type of i and of j
  * sequence separation (and its log), relative position of i and j in the chain,
    chain length
  * Kyte-Doolittle hydrophobicity and charge of i and of j
  * the same averaged over a window of 3 residues on each side of i and of j,
    plus the fraction of helix-breaking residues (P, G) in each window
  * both-cysteine flag (possible disulfide bond), product of the two charges

Run from the repo root:
    python model/tree_models.py train      # Kai: builds features, fits both models
    python model/tree_models.py evaluate   # Bhuvika: scores them on the validation set

train writes  model/checkpoints/logreg_rich_best.joblib and gbt_best.joblib
evaluate adds 'logreg_rich' and 'gbt' columns to results/progress_report/per_chain_val.csv
and rewrites results_table.md. Validation split only; nothing here touches the test split.
"""
from common import (AA_TO_INT, CHECKPOINT_DIR, CONTACT_CUTOFF, EVAL_MAX_LEN, RESULTS_DIR,
                    distance_matrix, load_chain, load_index, long_range_pairs,
                    top_l5_precision, write_table)  # common first: macOS OpenMP fix

import csv
import os
import random
import sys
import time

import numpy as np

SEED = 0
PAIRS_PER_TRAIN_CHAIN = 200   # same sampling as baselines.py
WINDOW = 3                    # residues on each side

# Kyte-Doolittle hydrophobicity, indexed like common.AMINO_ACIDS
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
      "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
      "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
CHARGE = {"K": 1.0, "R": 1.0, "D": -1.0, "E": -1.0}
HYD = np.zeros(20); CHG = np.zeros(20); BREAKER = np.zeros(20)
for aa, k in AA_TO_INT.items():
    HYD[k] = KD[aa]
    CHG[k] = CHARGE.get(aa, 0.0)
    BREAKER[k] = 1.0 if aa in "PG" else 0.0
CYS = AA_TO_INT["C"]

NUMERIC = ["log_sep", "sep", "rel_i", "rel_j", "log_len",
           "hyd_i", "hyd_j", "chg_i", "chg_j",
           "hyd_win_i", "hyd_win_j", "chg_win_i", "chg_win_j", "brk_win_i", "brk_win_j",
           "both_cys", "chg_product"]
CATEGORICAL = ["aa_i", "aa_j"]
FEATURES = CATEGORICAL + NUMERIC


def window_mean(values):
    """Mean of values over residues k-WINDOW .. k+WINDOW (clipped at the chain ends)."""
    L = len(values)
    c = np.concatenate([[0.0], np.cumsum(values)])
    lo = np.clip(np.arange(L) - WINDOW, 0, L)
    hi = np.clip(np.arange(L) + WINDOW + 1, 0, L)
    return (c[hi] - c[lo]) / (hi - lo)


def pair_features(tokens, i, j):
    """Feature matrix [n_pairs, len(FEATURES)] for pairs (i, j) of one chain."""
    L = len(tokens)
    hyd, chg, brk = HYD[tokens], CHG[tokens], BREAKER[tokens]
    hw, cw, bw = window_mean(hyd), window_mean(chg), window_mean(brk)
    sep = (j - i).astype(float)
    cols = {
        "aa_i": tokens[i], "aa_j": tokens[j],
        "log_sep": np.log(sep), "sep": sep,
        "rel_i": i / L, "rel_j": j / L, "log_len": np.full(len(i), np.log(L)),
        "hyd_i": hyd[i], "hyd_j": hyd[j], "chg_i": chg[i], "chg_j": chg[j],
        "hyd_win_i": hw[i], "hyd_win_j": hw[j], "chg_win_i": cw[i], "chg_win_j": cw[j],
        "brk_win_i": bw[i], "brk_win_j": bw[j],
        "both_cys": ((tokens[i] == CYS) & (tokens[j] == CYS)).astype(float),
        "chg_product": chg[i] * chg[j],
    }
    return np.column_stack([cols[f] for f in FEATURES]).astype(np.float32)


def build_training_set(rng):
    """One random chain per training cluster, up to 200 random long-range pairs each."""
    by_cluster = {}
    for r in load_index("train"):
        by_cluster.setdefault(r["cluster_id"], []).append(r)
    X, y = [], []
    for members in by_cluster.values():
        r = random.choice(members)
        tokens, coords, mask = load_chain(r["entity_id"])
        i, j = long_range_pairs(mask)
        if len(i) == 0:
            continue
        pick = rng.choice(len(i), min(PAIRS_PER_TRAIN_CHAIN, len(i)), replace=False)
        i, j = i[pick], j[pick]
        X.append(pair_features(tokens, i, j))
        y.append(distance_matrix(coords)[i, j] < CONTACT_CUTOFF)
    return np.concatenate(X), np.concatenate(y).astype(int), len(by_cluster)


def make_models():
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    cat_idx = [FEATURES.index(f) for f in CATEGORICAL]
    num_idx = [FEATURES.index(f) for f in NUMERIC]
    logreg_rich = Pipeline([
        ("prep", ColumnTransformer([
            ("aa", OneHotEncoder(categories=[list(range(20))] * 2, handle_unknown="ignore"), cat_idx),
            ("num", StandardScaler(), num_idx)])),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])
    gbt = HistGradientBoostingClassifier(
        categorical_features=cat_idx, max_iter=400, learning_rate=0.1, max_leaf_nodes=31,
        min_samples_leaf=100, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,  # held out from TRAIN
        random_state=SEED)
    return {"logreg_rich": logreg_rich, "gbt": gbt}


def train():
    import joblib
    random.seed(SEED)
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    print("Building training pairs (one chain per training cluster, 200 pairs each)...")
    X, y, n_clusters = build_training_set(rng)
    print(f"  {len(y):,} pairs from {n_clusters:,} clusters, {100*y.mean():.2f}% contacts, "
          f"{X.shape[1]} features ({time.time()-t0:.0f}s)")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    for name, model in make_models().items():
        t = time.time()
        print(f"Training {name}...")
        model.fit(X, y)
        if name == "gbt":
            print(f"  boosting rounds used: {model.n_iter_} (early stopping on 10% of TRAIN pairs)")
        path = os.path.join(CHECKPOINT_DIR, f"{name}_best.joblib")
        joblib.dump(model, path, compress=3)
        print(f"  saved {path} ({os.path.getsize(path)/1e6:.1f} MB, {time.time()-t:.0f}s)")
    print(f"Done in {(time.time()-t0)/60:.1f} min. Next (Bhuvika): python model/tree_models.py evaluate")


def evaluate():
    import joblib
    models = {}
    for name in ("logreg_rich", "gbt"):
        path = os.path.join(CHECKPOINT_DIR, f"{name}_best.joblib")
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}. Run: python model/tree_models.py train")
        models[name] = joblib.load(path)

    csv_path = os.path.join(RESULTS_DIR, "per_chain_val.csv")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    print(f"Evaluating on the {len(rows):,} validation chains in per_chain_val.csv...")
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    scores = {name: [] for name in models}
    for k, r in enumerate(rows):
        tokens, coords, mask = load_chain(r["entity_id"])
        i, j = long_range_pairs(mask)
        L = len(tokens)
        true = distance_matrix(coords)[i, j] < CONTACT_CUTOFF   # same labels as all other methods
        X = pair_features(tokens, i, j)
        for name, model in models.items():
            p = model.predict_proba(X)[:, 1]
            scores[name].append(top_l5_precision(p, true, L, rng))
        if (k + 1) % 500 == 0:
            print(f"  {k+1:,}/{len(rows):,} chains ({time.time()-t0:.0f}s)")

    for name in models:
        for r, s in zip(rows, scores[name]):
            r[name] = f"{s:.4f}"
    fields = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    methods = [("random", "Random"), ("separation", "Sequence separation"),
               ("logreg", "Logistic regression"), ("logreg_rich", "Logistic regression, rich features"),
               ("gbt", "Gradient-boosted trees, rich features"), ("neural", "Neural distogram model")]
    per_chain = {m: [float(r[m]) for r in rows] for m, _ in methods if m in rows[0]}
    print()
    write_table(per_chain, [m for m in methods if m[0] in per_chain])


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "train":
        train()
    elif mode == "evaluate":
        evaluate()
    else:
        raise SystemExit("Usage: python model/tree_models.py train|evaluate")
