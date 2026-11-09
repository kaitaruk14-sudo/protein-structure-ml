"""
From predicted distances to a 3D structure.

For each protein:
  1. The trained neural model predicts a distogram: for every residue pair, a probability
     for each distance bin (2-22 Å, plus "< 2 Å" and "> 22 Å").
  2. Distances -> 3D: turn the distogram into an expected distance for every pair the model
     is confident about (pairs it thinks are within 22 Å), fill in the rest with shortest
     paths, get a starting 3D shape with classical multidimensional scaling (MDS), then
     refine the coordinates with gradient descent so their pairwise distances match the
     predicted ones (the AlphaFold 1 idea, simplified). One point per residue: the C-beta
     atom, the same atom the distances are defined on.
  3. Compare with the real structure: superimpose (Kabsch) and report RMSD in Å and an
     approximate TM-score (0-1, higher is better; computed on the RMSD superposition, so it
     is a lower bound of the true TM-score). Distances can't tell a structure from its
     mirror image, so the mirror is also tried and the better one kept.
  4. As a check of step 2 alone, the same reconstruction is run on the TRUE distances
     ("oracle"): if that gives a low RMSD, any error in the prediction comes from the model,
     not from the reconstruction.

Outputs per protein in results/structure_3d/<run_name>/:
  <id>_pred.pdb, <id>_true.pdb   open in Mol* (molstar.org/viewer), PyMOL or ChimeraX
  <id>.png                       predicted structure superimposed on the real one
and summary.csv with RMSD / TM-score for every protein.

Run from the repo root (validation proteins only; the test split is not used):
    python model/structure_3d.py                      # 5 random val proteins, model 'long'
    python model/structure_3d.py --n 20
    python model/structure_3d.py --entity 1E41_1 --entity 2GRO_1
    python model/structure_3d.py --run_name basic     # use the original model instead
"""
import argparse
import csv
import os
import random
import time

from common import (AMINO_ACIDS, CHECKPOINT_DIR, ROOT, load_chain, load_index,
                    virtual_cbeta)  # common first: macOS OpenMP fix

import numpy as np
import torch

from dataset import BIN_EDGES

THREE_LETTER = dict(zip("ACDEFGHIKLMNPQRSTVWY",
                        ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
                         "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"]))
BIN_CENTRES = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2      # centres of bins 1..62 (2-22 Å)
FAR = 22.0                                              # beyond the last bin edge


# ---------------------------------------------------------------- 1. distogram
@torch.no_grad()
def predict_distogram(model, tokens, device):
    """-> probs [64, L, L] as numpy."""
    L = len(tokens)
    x = torch.from_numpy(tokens)[None].to(device)
    logits = model(x, torch.ones(1, L, dtype=torch.bool, device=device))
    return logits.softmax(1)[0].float().cpu().numpy()


def distances_from_distogram(probs, min_conf=0.5):
    """
    Expected distance and confidence for every pair.
      p_in  = P(2 Å <= d <= 22 Å);  d_exp = mean distance given it's in that range
      p_far = P(d > 22 Å)
    Pairs with p_in > min_conf become distance restraints (weight p_in);
    pairs with p_far > min_conf become "keep at least 22 Å apart" restraints.
    """
    inner = probs[1:63]
    p_in = inner.sum(0)
    d_exp = (inner * BIN_CENTRES[:, None, None]).sum(0) / np.clip(p_in, 1e-6, None)
    p_far = probs[63]
    w = np.where(p_in > min_conf, p_in, 0.0)
    far = p_far > min_conf
    np.fill_diagonal(w, 0); np.fill_diagonal(far, False)
    return d_exp, w, far


def distances_from_truth(true_d, mask):
    """Oracle restraints: the real distances, but only the information a perfect
    distogram would give (exact within 22 Å, 'far' beyond)."""
    obs = mask[:, None] & mask[None, :]
    w = np.where(obs & (true_d <= FAR), 1.0, 0.0)
    far = obs & (true_d > FAR)
    np.fill_diagonal(w, 0); np.fill_diagonal(far, False)
    return true_d.copy(), w, far


# ---------------------------------------------------------------- 2. distances -> 3D
def shortest_path_fill(d, w):
    """Complete the distance matrix: unknown pairs get the shortest path through known ones."""
    L = len(d)
    D = np.where(w > 0, d, np.inf)
    np.fill_diagonal(D, 0)
    for k in range(L):                                  # Floyd-Warshall, vectorised
        D = np.minimum(D, D[:, k:k + 1] + D[k:k + 1, :])
    finite = np.isfinite(D)
    if not finite.all():                                # disconnected parts: put them far apart
        D[~finite] = D[finite].max() * 1.5 if finite.any() else 30.0
    return D


def classical_mds(D, dim=3):
    L = len(D)
    J = np.eye(L) - np.ones((L, L)) / L
    B = -0.5 * J @ (D ** 2) @ J
    vals, vecs = np.linalg.eigh(B)
    idx = np.argsort(vals)[::-1][:dim]
    return vecs[:, idx] * np.sqrt(np.clip(vals[idx], 1e-6, None))


def reconstruct(d, w, far, steps=1500, seed=0):
    """Coordinates [L, 3] whose distances match d on weighted pairs."""
    torch.manual_seed(seed)
    X0 = classical_mds(shortest_path_fill(d, w))
    X = torch.tensor(X0, dtype=torch.float32, requires_grad=True)
    D_t = torch.tensor(d, dtype=torch.float32)
    W_t = torch.tensor(w, dtype=torch.float32)
    F_t = torch.tensor(far, dtype=torch.float32)
    L = len(d)
    offdiag = 1 - torch.eye(L)
    opt = torch.optim.Adam([X], lr=0.05)
    for step in range(steps):
        diff = X[:, None, :] - X[None, :, :]
        dist = torch.sqrt((diff ** 2).sum(-1) + 1e-8)
        fit = (W_t * (dist - D_t) ** 2).sum() / W_t.sum().clamp(min=1)
        keep_far = (F_t * torch.relu(FAR - dist) ** 2).sum() / F_t.sum().clamp(min=1)
        clash = (offdiag * torch.relu(3.5 - dist) ** 2).sum() / offdiag.sum()
        loss = fit + 0.1 * keep_far + clash
        opt.zero_grad(); loss.backward(); opt.step()
    return X.detach().numpy().astype(np.float64)


# ---------------------------------------------------------------- 3. comparison
def kabsch(P, Q):
    """Rotate/translate P onto Q (both [n, 3]); returns aligned P."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    s = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, s]) @ Vt
    return Pc @ R + Q.mean(0)


def compare(pred, true, mask):
    """RMSD (Å) and approximate TM-score over observed residues; tries the mirror image too."""
    t = true[mask]
    n = len(t)
    d0 = max(1.24 * (n - 15) ** (1 / 3) - 1.8, 0.5) if n > 15 else 0.5
    best = None
    for mirror in (False, True):
        p = pred[mask] * (np.array([-1, 1, 1]) if mirror else 1)
        a = kabsch(p, t)
        dev = np.linalg.norm(a - t, axis=1)
        res = {"rmsd": float(np.sqrt((dev ** 2).mean())),
               "tm": float((1 / (1 + (dev / d0) ** 2)).mean()), "mirror": mirror}
        if best is None or res["rmsd"] < best["rmsd"]:
            best = res
    return best


def aligned_full(pred, true, mask, mirror):
    """Whole predicted chain superimposed on the real one (fit on observed residues)."""
    p = pred * (np.array([-1, 1, 1]) if mirror else 1)
    Pc_mean, Q_mean = p[mask].mean(0), true[mask].mean(0)
    Pc, Qc = p[mask] - Pc_mean, true[mask] - Q_mean
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    s = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, s]) @ Vt
    return (p - Pc_mean) @ R + Q_mean


# ---------------------------------------------------------------- 4. output
def write_pdb(path, seq, coords, mask=None, bfactor=None, title=""):
    """One C-beta atom per residue (glycine gets the virtual C-beta)."""
    lines = [f"REMARK   1 {title}"[:80],
             "REMARK   2 One CB atom per residue (virtual CB from the backbone for glycine)."]
    serial = 0
    for k, aa in enumerate(seq):
        if mask is not None and not mask[k]:
            continue
        serial += 1
        x, y, z = coords[k]
        b = 0.0 if bfactor is None else float(bfactor[k])
        lines.append(f"ATOM  {serial:5d}  CB  {THREE_LETTER[aa]:3s} A{k + 1:4d}    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{b:6.2f}           C")
    lines.append("END")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def plot_structures(path, pred, true, mask, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    t = np.where(mask[:, None], true, np.nan)            # gaps where residues are missing
    ax.plot(*t.T, color="#9a9893", lw=2.2, label="real structure")
    ax.plot(*pred.T, color="#2a78d6", lw=1.6, label="predicted")
    ax.scatter(*pred[0], color="#2a78d6", s=30); ax.scatter(*pred[-1], color="#2a78d6", s=30, marker="s")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", frameon=False)
    ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)"); ax.set_zlabel("z (Å)")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", default="long", help="which neural model: checkpoints/<run_name>_best.pt")
    p.add_argument("--entity", action="append", help="validation chain id(s), e.g. 1E41_1")
    p.add_argument("--n", type=int, default=5, help="number of random validation chains")
    p.add_argument("--max_len", type=int, default=300)
    p.add_argument("--steps", type=int, default=1500, help="gradient-descent steps per protein")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from model import BasicDistogramNet
    from train import get_device
    device = get_device()
    ck_path = os.path.join(CHECKPOINT_DIR, f"{args.run_name}_best.pt")
    model = BasicDistogramNet().to(device)
    model.load_state_dict(torch.load(ck_path, map_location=device)["model"])
    model.eval()

    val = {r["entity_id"]: r for r in load_index("val")}
    if args.entity:
        missing = [e for e in args.entity if e not in val]
        if missing:
            raise SystemExit(f"Not validation chains (or filtered out): {missing}")
        rows = [val[e] for e in args.entity]
    else:
        pool = [r for r in val.values() if 50 <= int(r["length"]) <= args.max_len
                and float(r["frac_observed"]) >= 0.9]
        rows = random.Random(args.seed).sample(sorted(pool, key=lambda r: r["entity_id"]),
                                               min(args.n, len(pool)))

    out_dir = os.path.join(ROOT, "results", "structure_3d", args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Model {args.run_name} on {device}; {len(rows)} validation proteins -> {out_dir}\n")
    print(f"{'protein':>9} {'L':>4} | {'RMSD pred':>9} {'TM pred':>7} | {'RMSD oracle':>11} {'TM oracle':>9} | time")

    summary = []
    for r in rows:
        t0 = time.time()
        eid = r["entity_id"]
        tokens, coords, mask = load_chain(eid)
        seq = "".join(AMINO_ACIDS[t] for t in tokens)
        true_cb = virtual_cbeta(np.nan_to_num(coords))
        true_d = np.linalg.norm(true_cb[:, None] - true_cb[None], axis=-1)

        probs = predict_distogram(model, tokens, device)
        d, w, far = distances_from_distogram(probs)
        pred = reconstruct(d, w, far, steps=args.steps, seed=args.seed)
        res = compare(pred, true_cb, mask)

        od, ow, ofar = distances_from_truth(true_d, mask)
        oracle = reconstruct(od, ow, ofar, steps=args.steps, seed=args.seed)
        ores = compare(oracle, true_cb, mask)

        aligned = aligned_full(pred, true_cb, mask, res["mirror"])
        confidence = 100 * np.where(w > 0, w, 0).max(1)       # per-residue, shown as B-factor
        write_pdb(os.path.join(out_dir, f"{eid}_pred.pdb"), seq, aligned, bfactor=confidence,
                  title=f"{eid} predicted by {args.run_name}; RMSD {res['rmsd']:.2f} A vs real")
        write_pdb(os.path.join(out_dir, f"{eid}_true.pdb"), seq, true_cb, mask=mask,
                  title=f"{eid} experimental structure (C-beta)")
        plot_structures(os.path.join(out_dir, f"{eid}.png"), aligned, true_cb, mask,
                        f"{eid} ({len(seq)} residues): RMSD {res['rmsd']:.1f} Å, "
                        f"TM ≈ {res['tm']:.2f}\n(oracle with true distances: RMSD {ores['rmsd']:.1f} Å)")
        summary.append({"entity_id": eid, "length": len(seq), "n_observed": int(mask.sum()),
                        "rmsd_pred": round(res["rmsd"], 3), "tm_pred": round(res["tm"], 3),
                        "mirror_pred": res["mirror"],
                        "rmsd_oracle": round(ores["rmsd"], 3), "tm_oracle": round(ores["tm"], 3)})
        print(f"{eid:>9} {len(seq):4d} | {res['rmsd']:9.2f} {res['tm']:7.3f} | "
              f"{ores['rmsd']:11.2f} {ores['tm']:9.3f} | {time.time() - t0:.0f}s")

    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        wr.writeheader(); wr.writerows(summary)
    m = lambda k: np.mean([s[k] for s in summary])
    print(f"\nMean over {len(summary)} proteins: predicted RMSD {m('rmsd_pred'):.2f} Å, TM ≈ {m('tm_pred'):.3f} | "
          f"oracle RMSD {m('rmsd_oracle'):.2f} Å, TM ≈ {m('tm_oracle'):.3f}")
    print(f"Open the .pdb files at https://molstar.org/viewer (Open Files), or look at the .png plots.")


if __name__ == "__main__":
    main()
