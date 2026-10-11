"""
Shared helpers used by every script in model/:
paths, loading chains, contact maps, the evaluation metric, and bootstrap CIs.
"""
import os

# macOS + conda: two copies of the OpenMP library (one from PyTorch, one from
# the conda mmseqs2 install) crash Python unless this is set before numpy/torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import csv
import numpy as np

# ---------- paths (all relative to the repo root, ~/machine_learning) ----------
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(MODEL_DIR)
DATA_DIR = os.environ.get("PROTEIN_DATA_DIR", os.path.join(ROOT, "data", "protein_structures"))
RESULTS_DIR = os.environ.get("PROTEIN_RESULTS_DIR", os.path.join(ROOT, "results", "preliminary"))
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")

# ---------- settings shared by all methods ----------
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INT = {a: i for i, a in enumerate(AMINO_ACIDS)}
MIN_OBSERVED = 30        # chain needs at least 30 residues with coordinates
MIN_FRAC_OBSERVED = 0.5  # ...and at least half of the chain observed
EVAL_MAX_LEN = 512       # evaluate on chains up to this length (89% of val)
CONTACT_CUTOFF = 8.0     # Å between C-beta atoms
LONG_RANGE = 24          # |i - j| >= 24 counts as long-range


def load_index(split, max_len=None, data_dir=DATA_DIR):
    """Rows of backbone_index.tsv for one split, after the quality filters."""
    path = os.path.join(data_dir, "backbone_index.tsv")
    with open(path) as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t")
                if r["split"] == split
                and int(r["n_observed"]) >= MIN_OBSERVED
                and float(r["frac_observed"]) >= MIN_FRAC_OBSERVED
                and (max_len is None or int(r["length"]) <= max_len)]
    return [r for r in rows if os.path.exists(chain_path(r["entity_id"], data_dir))]


def chain_path(entity_id, data_dir=DATA_DIR):
    return os.path.join(data_dir, "backbone", f"{entity_id}.npz")


def load_chain(entity_id, data_dir=DATA_DIR):
    """-> tokens [L] ints 0-19, coords [L,3,3] (N, CA, C), mask [L] bool."""
    d = np.load(chain_path(entity_id, data_dir))
    tokens = np.array([AA_TO_INT[a] for a in str(d["seq"])], dtype=np.int64)
    return tokens, d["coords"], d["mask"]


def virtual_cbeta(coords):
    """C-beta position calculated from the backbone (as in AlphaFold/trRosetta)."""
    n, ca, c = coords[:, 0], coords[:, 1], coords[:, 2]
    b = ca - n
    c_ = c - ca
    a = np.cross(b, c_)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c_ + ca


def distance_matrix(coords):
    cb = virtual_cbeta(np.nan_to_num(coords))
    return np.linalg.norm(cb[:, None, :] - cb[None, :, :], axis=-1)


def long_range_pairs(mask, min_sep=LONG_RANGE):
    """Indices (i, j), i < j, of long-range pairs where both residues are observed."""
    L = len(mask)
    i, j = np.triu_indices(L, k=min_sep)
    keep = mask[i] & mask[j]
    return i[keep], j[keep]


def top_l5_precision(scores, true_contacts, L, rng=None):
    """
    The evaluation metric.
    scores, true_contacts: 1-D arrays over the chain's long-range pairs.
    Take the L/5 pairs with the highest score; return the fraction that are
    real contacts. L is the full chain length. Ties are broken randomly.
    """
    if len(scores) == 0:
        return None
    k = max(1, L // 5)
    k = min(k, len(scores))
    rng = rng or np.random.default_rng(0)
    order = np.lexsort((rng.random(len(scores)), -scores))  # by score, random tiebreak
    return float(true_contacts[order[:k]].mean())


def bootstrap_ci(values, n_boot=2000, seed=0):
    """Mean and 95% CI over chains (resampling chains with replacement)."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    return v.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def write_table(per_chain, methods, out_dir=RESULTS_DIR, baseline="separation"):
    """
    per_chain: dict method -> list of per-chain precisions (same chains, same order).
    Writes results_table.md and prints it.
    """
    lines = ["| Method | Long-range top-L/5 precision (val) | 95% CI | "
             f"Difference vs {baseline} (95% CI) |", "|---|---|---|---|"]
    base = np.asarray(per_chain[baseline]) if baseline in per_chain else None
    for name, label in methods:
        if name not in per_chain:
            continue
        v = np.asarray(per_chain[name])
        m, lo, hi = bootstrap_ci(v)
        diff = ""
        if base is not None and name != baseline:
            dm, dlo, dhi = bootstrap_ci(v - base)
            diff = f"{100*dm:+.2f} pp ({100*dlo:+.2f} to {100*dhi:+.2f})"
        lines.append(f"| {label} | {100*m:.2f}% | {100*lo:.2f}% – {100*hi:.2f}% | {diff} |")
    n = len(next(iter(per_chain.values())))
    lines.append("")
    lines.append(f"Evaluated on {n:,} validation chains (length ≤ {EVAL_MAX_LEN}). "
                 f"Contact = Cβ–Cβ < {CONTACT_CUTOFF:.0f} Å; long-range = |i−j| ≥ {LONG_RANGE}. "
                 "CIs: 2,000 bootstrap resamples of chains; pp = percentage points.")
    text = "\n".join(lines)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results_table.md"), "w") as f:
        f.write(text + "\n")
    print(text)
    return text
