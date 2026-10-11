"""
Three simple baselines for long-range contact prediction, all scored on the
validation set with the same metric as the neural model.

  1. Random              - score every pair at random (= the base rate)
  2. Sequence separation - residues closer along the chain are more likely to
                           touch, so rank pairs by smallest |i - j|
  3. Logistic regression - learns from the training set using only:
                             * which two amino acids form the pair (210 types)
                             * how far apart they are along the chain
                           Implemented from scratch with numpy (gradient descent
                           on the log-loss), so no extra libraries are needed.

Run from ~/machine_learning:   python model/baselines.py
Writes results/preliminary/: per_chain_val.csv, results_table.md,
                             logreg_weights.json, baselines_summary.json
"""
import json
import os
import random
import time

import numpy as np

from common import (AMINO_ACIDS, CONTACT_CUTOFF, EVAL_MAX_LEN, LONG_RANGE, RESULTS_DIR,
                    distance_matrix, load_chain, load_index, long_range_pairs,
                    top_l5_precision, write_table)

SEED = 0
PAIRS_PER_TRAIN_CHAIN = 200
SEP_BIN_EDGES = np.array([24, 32, 48, 64, 96, 128, 192, 256, 384])  # 9 bins

# unordered amino-acid pair -> index 0..209
PAIR_INDEX = np.zeros((20, 20), dtype=np.int64)
_k = 0
for a in range(20):
    for b in range(a, 20):
        PAIR_INDEX[a, b] = PAIR_INDEX[b, a] = _k
        _k += 1
N_PAIR_TYPES = _k  # 210


def pair_features(tokens, i, j):
    """Features of pairs (i, j) as integer indices + one number."""
    sep = j - i
    return {
        "pair": PAIR_INDEX[tokens[i], tokens[j]],               # 0..209
        "sep_bin": np.searchsorted(SEP_BIN_EDGES, sep, "right") - 1,  # 0..8
        "log_sep": np.log(sep).astype(np.float64),
    }


class LogisticRegression:
    """
    p(contact) = sigmoid( w_pair[pair type] + w_sep[separation bin]
                          + w_log * standardised log(separation) + b )
    This is ordinary logistic regression on one-hot features; storing the
    one-hot features as indices just avoids building a huge 0/1 matrix.
    """

    def __init__(self):
        self.w_pair = np.zeros(N_PAIR_TYPES)
        self.w_sep = np.zeros(len(SEP_BIN_EDGES))
        self.w_log = 0.0
        self.b = 0.0
        self.mu, self.sd = 0.0, 1.0

    def n_params(self):
        return len(self.w_pair) + len(self.w_sep) + 2

    def logits(self, f):
        x = (f["log_sep"] - self.mu) / self.sd
        return self.w_pair[f["pair"]] + self.w_sep[f["sep_bin"]] + self.w_log * x + self.b

    def fit(self, f, y, steps=400, lr=0.05, l2=1e-4):
        self.mu, self.sd = f["log_sep"].mean(), f["log_sep"].std()
        x = (f["log_sep"] - self.mu) / self.sd
        n = len(y)
        # Adam optimiser state for each parameter group
        params = ["w_pair", "w_sep", "w_log", "b"]
        m = {p: np.zeros_like(np.asarray(getattr(self, p), dtype=float)) for p in params}
        v = {p: np.zeros_like(np.asarray(getattr(self, p), dtype=float)) for p in params}
        for t in range(1, steps + 1):
            p = 1 / (1 + np.exp(-self.logits(f)))
            dz = (p - y) / n                                      # d(log-loss)/d(logit)
            grads = {
                "w_pair": np.bincount(f["pair"], dz, N_PAIR_TYPES) + l2 * self.w_pair,
                "w_sep": np.bincount(f["sep_bin"], dz, len(SEP_BIN_EDGES)) + l2 * self.w_sep,
                "w_log": np.sum(dz * x),
                "b": np.sum(dz),
            }
            for name in params:
                g = grads[name]
                m[name] = 0.9 * m[name] + 0.1 * g
                v[name] = 0.999 * v[name] + 0.001 * g * g
                mhat = m[name] / (1 - 0.9 ** t)
                vhat = v[name] / (1 - 0.999 ** t)
                setattr(self, name, getattr(self, name) - lr * mhat / (np.sqrt(vhat) + 1e-8))
            if t % 100 == 0 or t == 1:
                eps = 1e-9
                loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
                print(f"  step {t:4d}  log-loss {loss:.4f}")


def build_training_pairs(rows, rng):
    """One random chain per cluster, up to 200 random long-range pairs from each."""
    by_cluster = {}
    for r in rows:
        by_cluster.setdefault(r["cluster_id"], []).append(r)
    feats, labels = [], []
    for members in by_cluster.values():
        r = random.choice(members)
        tokens, coords, mask = load_chain(r["entity_id"])
        i, j = long_range_pairs(mask)
        if len(i) == 0:
            continue
        pick = rng.choice(len(i), min(PAIRS_PER_TRAIN_CHAIN, len(i)), replace=False)
        i, j = i[pick], j[pick]
        d = distance_matrix(coords)[i, j]
        feats.append(pair_features(tokens, i, j))
        labels.append((d < CONTACT_CUTOFF).astype(np.float64))
    f = {k: np.concatenate([x[k] for x in feats]) for k in feats[0]}
    return f, np.concatenate(labels), len(by_cluster)


def main():
    random.seed(SEED)
    rng = np.random.default_rng(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    start = time.time()

    # ---- train logistic regression ----
    train_rows = load_index("train")
    print(f"Building training pairs from {len(train_rows):,} train chains...")
    f_train, y_train, n_clusters = build_training_pairs(train_rows, rng)
    print(f"  {len(y_train):,} long-range pairs from {n_clusters:,} clusters, "
          f"{100*y_train.mean():.2f}% are contacts")
    model = LogisticRegression()
    print(f"Training logistic regression ({model.n_params()} parameters)...")
    model.fit(f_train, y_train)

    # ---- evaluate everything on validation ----
    val_rows = load_index("val", max_len=EVAL_MAX_LEN)
    print(f"Evaluating on {len(val_rows):,} validation chains (length <= {EVAL_MAX_LEN})...")
    eval_rng = np.random.default_rng(SEED)
    results = {"random": [], "separation": [], "logreg": []}
    ids, lengths = [], []
    n_pairs = n_contacts = 0
    for r in val_rows:
        tokens, coords, mask = load_chain(r["entity_id"])
        i, j = long_range_pairs(mask)
        if len(i) == 0:
            continue
        L = len(tokens)
        true = distance_matrix(coords)[i, j] < CONTACT_CUTOFF
        n_pairs += len(true); n_contacts += true.sum()
        results["random"].append(top_l5_precision(eval_rng.random(len(i)), true, L, eval_rng))
        results["separation"].append(top_l5_precision(-(j - i).astype(float), true, L, eval_rng))
        results["logreg"].append(top_l5_precision(model.logits(pair_features(tokens, i, j)),
                                                  true, L, eval_rng))
        ids.append(r["entity_id"]); lengths.append(L)

    with open(os.path.join(RESULTS_DIR, "per_chain_val.csv"), "w") as f:
        f.write("entity_id,length,random,separation,logreg\n")
        for k, eid in enumerate(ids):
            f.write(f"{eid},{lengths[k]},{results['random'][k]:.4f},"
                    f"{results['separation'][k]:.4f},{results['logreg'][k]:.4f}\n")

    # most and least contact-prone amino-acid pairs, for interpretation
    names = [None] * N_PAIR_TYPES
    for a in range(20):
        for b in range(a, 20):
            names[PAIR_INDEX[a, b]] = AMINO_ACIDS[a] + AMINO_ACIDS[b]
    order = np.argsort(model.w_pair)
    with open(os.path.join(RESULTS_DIR, "logreg_weights.json"), "w") as f:
        json.dump({
            "most_contact_prone_pairs": [[names[k], round(float(model.w_pair[k]), 3)] for k in order[::-1][:15]],
            "least_contact_prone_pairs": [[names[k], round(float(model.w_pair[k]), 3)] for k in order[:15]],
            "separation_bin_weights": dict(zip([str(e) for e in SEP_BIN_EDGES], np.round(model.w_sep, 3).tolist())),
            "log_separation_weight": round(float(model.w_log), 3),
            "n_parameters": model.n_params(),
        }, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "baselines_summary.json"), "w") as f:
        json.dump({"n_val_chains": len(ids),
                   "val_long_range_pairs": int(n_pairs),
                   "val_long_range_contact_rate_pct": round(100 * n_contacts / n_pairs, 3),
                   "logreg_training_pairs": int(len(y_train)),
                   "logreg_training_contact_rate_pct": round(100 * float(y_train.mean()), 3),
                   "logreg_parameters": model.n_params()}, f, indent=2)

    print(f"\nLong-range contact rate on val (all pairs): {100*n_contacts/n_pairs:.2f}%\n")
    write_table(results, [("random", "Random"), ("separation", "Sequence separation"),
                          ("logreg", "Logistic regression")])
    print(f"\nDone in {(time.time()-start)/60:.1f} min. Files in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
