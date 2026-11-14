"""
Task B sanity checks for the validation results. Validation split only; nothing here loads test chains.

  1. no cluster_id / entity_id / identical sequence appears in more than one split
  2. per_chain_val.csv covers exactly the 3,992 val chains that pass the filters, in baseline order
  3. every method column has valid values, and precision * k is a whole number (k = min(max(1, L//5), pairs))
  4. independent hand recomputation (own loops, own C-beta code) of sequence-separation and
     neural precision for two chains, compared with the CSV

Every method column in per_chain_val.csv is checked (random, separation, logreg, logreg_rich, gbt and
every neural / neural_<run> column); each neural column is also recomputed by hand from its own checkpoint.

Run from the repo root after model/evaluate.py:
    python model/sanity_checks.py [baseline_git_ref]     (default ref: main)
    python model/sanity_checks.py | tee results/progress_report/sanity_checks.txt
"""
import csv
import io
import os
import subprocess
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_REF = sys.argv[1] if len(sys.argv) > 1 else "main"
sys.path.insert(0, os.path.join(REPO, "model"))

import common  # noqa: E402,F401  must be imported before numpy / torch (OpenMP fix, e.g. 'OMP: Error #15' on macOS)
import numpy as np  # noqa: E402
RES = os.path.join(REPO, "results", "progress_report")
DATA = os.path.join(REPO, "data", "protein_structures")
RES = os.path.join(REPO, "results", "progress_report")

idx = list(csv.DictReader(open(os.path.join(DATA, "backbone_index.tsv")), delimiter="\t"))
spl = list(csv.DictReader(open(os.path.join(DATA, "dataset_split.tsv")), delimiter="\t"))

print(f"backbone_index.tsv rows: {len(idx):,}; dataset_split.tsv rows: {len(spl):,}")

# 1a. no cluster in more than one split
for name, rows in (("backbone_index", idx), ("dataset_split", spl)):
    c2s = defaultdict(set)
    for r in rows:
        c2s[r["cluster_id"]].add(r["split"])
    bad = {c: s for c, s in c2s.items() if len(s) > 1}
    counts = defaultdict(int)
    for r in rows:
        counts[r["split"]] += 1
    print(f"[{name}] clusters: {len(c2s):,}; clusters in >1 split: {len(bad)}; rows per split: {dict(counts)}")

# 1b. no chain in two splits, ids unique, and both files agree
e2s = defaultdict(set)
for r in idx:
    e2s[r["entity_id"]].add(r["split"])
dup = [e for e, s in e2s.items() if len(s) > 1]
print(f"[backbone_index] entity_ids: {len(e2s):,} unique of {len(idx):,} rows; entities in >1 split: {len(dup)}")
split_by_id = {r["entity_id"]: r["split"] for r in spl}
disagree = [r["entity_id"] for r in idx if r["entity_id"] in split_by_id
            and split_by_id[r["entity_id"]] != r["split"]]
print(f"entity split assignments that differ between the two files: {len(disagree)}")
cl_idx = {r["entity_id"]: r["cluster_id"] for r in idx}
cl_spl = {r["entity_id"]: r["cluster_id"] for r in spl}
print(f"entity cluster assignments that differ between the two files: "
      f"{sum(1 for e in cl_idx if e in cl_spl and cl_idx[e] != cl_spl[e])}")

# identical sequences must not straddle splits
seq2s = defaultdict(set)
for r in spl:
    seq2s[r["sequence"]].add(r["split"])
print(f"identical sequences appearing in >1 split: {sum(1 for s in seq2s.values() if len(s) > 1)}")

# 2. per_chain_val.csv coverage
val_ids = [r["entity_id"] for r in csv.DictReader(open(os.path.join(RES, "per_chain_val.csv")))]
print(f"per_chain_val.csv chains: {len(val_ids):,}; unique: {len(set(val_ids)):,}")
print(f"all of them are 'val' in backbone_index: {all(split_by_id.get(e) == 'val' and cl_idx.get(e) for e in val_ids)} "
      f"(splits found: {sorted({next(r['split'] for r in idx if r['entity_id']==e) for e in val_ids[:50]})} in first 50)")
by_id = {r["entity_id"]: r for r in idx}
splits_of_val_ids = {by_id[e]["split"] for e in val_ids}
print(f"set of splits among per_chain_val.csv chains (backbone_index): {splits_of_val_ids}")

# expected eval set from the stated filters
exp = [r["entity_id"] for r in idx if r["split"] == "val" and int(r["n_observed"]) >= 30
       and float(r["frac_observed"]) >= 0.5 and int(r["length"]) <= 512
       and os.path.exists(os.path.join(DATA, "backbone", r["entity_id"] + ".npz"))]
print(f"val rows passing filters (>=30 observed, >=50%, len<=512): {len(exp):,}")
print(f"CSV ids == filtered val ids as sets: {set(exp) == set(val_ids)}; same order: {exp == val_ids}")


before = list(csv.DictReader(io.StringIO(subprocess.run(
    ["git", "show", f"{BASELINE_REF}:results/progress_report/per_chain_val.csv"],
    cwd=REPO, capture_output=True, text=True, check=True).stdout)))
after = list(csv.DictReader(open(os.path.join(RES, "per_chain_val.csv"))))
print(f"(a) rows before/after: {len(before):,}/{len(after):,}")
print(f"    same ids in same order: {[r['entity_id'] for r in before] == [r['entity_id'] for r in after]}")
kept_cols = [c for c in before[0] if c != "entity_id"]
print(f"    columns of {BASELINE_REF} still unchanged ({', '.join(kept_cols)}): "
      f"{all(b[c] == a[c] for b, a in zip(before, after) for c in kept_cols)}")
print(f"    columns now: {list(after[0])}")

# (b)
bad_range = bad_int = 0
meta = {}
method_cols = [c for c in after[0] if c not in ("entity_id", "length")]
print(f"(b) checking columns: {method_cols}")
for r in after:
    d = np.load(os.path.join(DATA, "backbone", r["entity_id"] + ".npz"))
    mask = d["mask"].astype(bool)
    L = len(mask)
    assert L == int(r["length"]), r["entity_id"]
    obs = np.where(mask)[0]
    # number of observed pairs with j - i >= 24
    n_pairs = sum(int((obs[a + 1:] - obs[a] >= 24).sum()) for a in range(len(obs)))
    k = min(max(1, L // 5), n_pairs)
    meta[r["entity_id"]] = (L, k, n_pairs)
    for m in method_cols:
        v = float(r[m])
        if not (0 <= v <= 1) or np.isnan(v):
            bad_range += 1
        x = v * k
        if abs(x - round(x)) > 0.01 + 5e-5 * k:
            bad_int += 1
print(f"(b) values outside [0,1] or NaN: {bad_range}; precision*k not a whole number: {bad_int}")

# (c) independent recomputation
def cbeta(coords):
    n, ca, c = coords[:, 0], coords[:, 1], coords[:, 2]
    b = ca - n
    cc = c - ca
    a = np.cross(b, cc)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * cc + ca


def pairs_and_labels(eid):
    d = np.load(os.path.join(DATA, "backbone", eid + ".npz"))
    coords, mask = d["coords"].astype(float), d["mask"].astype(bool)
    cb = cbeta(np.nan_to_num(coords))
    out = []
    for i in range(len(mask)):
        if not mask[i]:
            continue
        for j in range(i + 24, len(mask)):
            if mask[j]:
                out.append((i, j, float(np.sqrt(((cb[i] - cb[j]) ** 2).sum())) < 8.0))
    return out, len(mask)


by_len = sorted(after, key=lambda r: abs(int(r["length"]) - 150))
picks = [by_len[0], by_len[1]] if by_len[0]["entity_id"] != by_len[1]["entity_id"] else by_len[:2]
nz=[r for r in after if float(r["separation"])>0 and float(r["neural"])>0 and 80<=int(r["length"])<=300]
picks = [nz[0], nz[len(nz)//2]]

import torch
from common import load_chain
from model import load_model
from dataset import BIN_EDGES


def checkpoint_for(col):
    """column 'neural' -> basic_best.pt; column 'neural_<run>' -> <run>_best.pt"""
    run = "basic" if col == "neural" else col[len("neural_"):]
    return os.path.join(REPO, "model", "checkpoints", f"{run}_best.pt")


neural_cols = [c for c in after[0] if c == "neural" or c.startswith("neural_")]
nets = {}
for col in neural_cols:
    path = checkpoint_for(col)
    if os.path.exists(path):
        net, _ = load_model(path, "cpu")
        net.eval()
        nets[col] = net
    else:
        print(f"(c) no checkpoint at {path}: the hand check for column {col} is skipped")
print(f"(c) neural columns hand-checked: {list(nets)}")

for r in picks:
    eid = r["entity_id"]
    pl, L = pairs_and_labels(eid)
    k = min(max(1, L // 5), len(pl))
    # separation: smallest |i-j| first; ties at the cut-off can go either way
    pl_sorted = sorted(pl, key=lambda t: t[1] - t[0])
    cut = pl_sorted[k - 1][1] - pl_sorted[k - 1][0]
    sure = [t for t in pl_sorted if t[1] - t[0] < cut]
    ties = [t for t in pl_sorted if t[1] - t[0] == cut]
    need = k - len(sure)
    base_hits = sum(t[2] for t in sure)
    n_pos_ties = sum(t[2] for t in ties)
    lo = (base_hits + max(0, need - (len(ties) - n_pos_ties))) / k
    hi = (base_hits + min(need, n_pos_ties)) / k
    csv_sep = float(r["separation"])
    ok_sep = lo - 6e-5 <= csv_sep <= hi + 6e-5
    labels = np.array([x[2] for x in pl])
    print(f"(c) {eid}: L={L}, long-range pairs={len(pl)}, k={k}, contacts={int(labels.sum())}")
    print(f"    separation: hand range [{lo:.4f}, {hi:.4f}] vs CSV {csv_sep:.4f} -> {'MATCH' if ok_sep else 'MISMATCH'}")
    # neural: own top-k from each model's P(d < 8 A) score
    tokens, _, _ = load_chain(eid)
    for col, net in nets.items():
        with torch.no_grad():
            t = torch.tensor(tokens)[None]
            sm = torch.ones_like(t, dtype=torch.bool)
            logits = net(t, sm)
        n_contact_bins = int((BIN_EDGES <= 8.0).sum())
        prob = logits.softmax(1)[0, :n_contact_bins].sum(0).numpy()
        scores = np.array([prob[i, j] for i, j, _ in pl])
        order = np.argsort(-scores, kind="stable")
        neural_hand = labels[order[:k]].mean()
        print(f"    {col}: hand {neural_hand:.4f} vs CSV {float(r[col]):.4f} -> "
              f"{'MATCH' if abs(neural_hand - float(r[col])) < 6e-5 else 'MISMATCH'}")
