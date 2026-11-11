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
  3. Compare with the real structure: superimpose (Kabsch) and report RMSD in Å, lDDT
     (local accuracy, no superposition needed) and an approximate TM-score (0-1, higher is better; computed on the RMSD superposition, so it
     is a lower bound of the true TM-score). Distances can't tell a structure from its
     mirror image, so the mirror is also tried and the better one kept.
  4. As a check of step 2 alone, the same reconstruction is run on the TRUE distances
     ("oracle"): if that gives a low RMSD, any error in the prediction comes from the model,
     not from the reconstruction.

Outputs in results/structure_3d/<run_name>/:
  index.html                     overview: scores for every protein, links to everything
  <id>.html                      interactive 3D viewer (rotate/zoom; overlay, predicted or real;
                                 colour by chain position or by confidence). Needs internet once
                                 to load the 3Dmol.js viewer library.
  <id>.png                       real / predicted / overlay, same viewing angle
  <id>_maps.png                  predicted vs real distance map
  <id>_pred.pdb, <id>_true.pdb   also open in Mol* (molstar.org/viewer), PyMOL or ChimeraX
  summary.csv                    RMSD, TM-score, lDDT, confidence, confidence-vs-lDDT per protein

Per-residue confidence (0-100, stored in the PDB B-factor column): how sharply the model commits
to the distances between that residue and its neighbours (see residue_confidence).

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


# ---------------------------------------------------------------- 4. confidence
def residue_confidence(probs, d_exp, min_sep=3, near=15.0, tol=2.0):
    """
    Per-residue confidence, 0-100 (like AlphaFold's pLDDT in spirit, much simpler).
    For residue i, look at its pairs (i, j) with |i - j| >= 3 that the model places within
    15 Å; for each, take the probability mass within ±2 Å of the predicted distance
    (how sharply the model commits to that distance). Average those -> x 100.
    """
    inner = probs[1:63]                                             # [62, L, L]
    sharp = (inner * (np.abs(BIN_CENTRES[:, None, None] - d_exp[None]) <= tol)).sum(0)
    L = len(d_exp)
    idx = np.arange(L)
    use = (np.abs(idx[:, None] - idx[None, :]) >= min_sep) & (inner.sum(0) > 0.5) & (d_exp < near)
    conf = np.where(use, sharp, 0).sum(1) / np.clip(use.sum(1), 1, None)
    return 100 * conf


def lddt(pred, true, mask, cutoff=15.0, thresholds=(0.5, 1.0, 2.0, 4.0)):
    """
    lDDT (local Distance Difference Test, the local score AlphaFold uses), on C-beta atoms.
    For every pair of residues within 15 Å in the real structure, check whether the predicted
    distance is within 0.5 / 1 / 2 / 4 Å of the real one; average the four checks.
    No superposition needed, so it measures local correctness. Returns (per-residue 0-1, overall).
    """
    dt = np.linalg.norm(true[:, None] - true[None], axis=-1)
    dp = np.linalg.norm(pred[:, None] - pred[None], axis=-1)
    use = mask[:, None] & mask[None, :] & (dt < cutoff) & ~np.eye(len(true), dtype=bool)
    diff = np.abs(dp - dt)
    score = np.mean([diff < t for t in thresholds], axis=0)
    per_res = np.where(use, score, 0).sum(1) / np.clip(use.sum(1), 1, None)
    overall = float(np.where(use, score, 0).sum() / max(use.sum(), 1))
    return per_res, overall


def spearman(a, b):
    """Rank correlation (no scipy needed)."""
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")


# ---------------------------------------------------------------- 5. output files
def pdb_text(seq, coords, mask=None, bfactor=None, title=""):
    """
    PDB text: one C-beta atom per residue (virtual C-beta for glycine), with CONECT
    records between consecutive residues so viewers draw the chain as a connected trace.
    The B-factor column holds the per-residue confidence (predictions) or 0 (real).
    """
    lines = [f"TITLE     {title}"[:80],
             "REMARK   2 One CB atom per residue (virtual CB from the backbone for glycine).",
             "REMARK   3 B-factor column = per-residue confidence 0-100 (predicted structures)."]
    serial_of = {}
    for k, aa in enumerate(seq):
        if mask is not None and not mask[k]:
            continue
        serial = len(serial_of) + 1
        serial_of[k] = serial
        x, y, z = coords[k]
        b = 0.0 if bfactor is None else float(bfactor[k])
        lines.append(f"ATOM  {serial:5d}  CB  {THREE_LETTER[aa]:3s} A{k + 1:4d}    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{b:6.2f}           C")
    lines.append("TER")
    for k in serial_of:
        if k + 1 in serial_of:                                      # bond to the next residue
            lines.append(f"CONECT{serial_of[k]:5d}{serial_of[k + 1]:5d}")
    lines.append("END")
    return "\n".join(lines) + "\n"


def _style_axes(ax, pts):
    """Equal scale on all axes, no grey panes, light grid."""
    pts = pts[np.isfinite(pts).all(1)]
    centre = (pts.max(0) + pts.min(0)) / 2
    half = (pts.max(0) - pts.min(0)).max() / 2 + 2
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(centre[2] - half, centre[2] + half)
    ax.set_box_aspect((1, 1, 1))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#e4e3df")
        axis._axinfo["grid"]["color"] = "#ecebe7"
    ax.tick_params(labelsize=7, colors="#52514e")


def _rainbow_trace(ax, xyz, mask=None, lw=2.0):
    """Chain drawn segment by segment, coloured from start (blue) to end (red)."""
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("turbo")
    L = len(xyz)
    for k in range(L - 1):
        if mask is not None and not (mask[k] and mask[k + 1]):
            continue
        ax.plot(*xyz[k:k + 2].T, color=cmap(0.08 + 0.84 * k / max(L - 2, 1)), lw=lw,
                solid_capstyle="round")


def plot_structures(path, pred, true, mask, title, elev=20, azim=-60):
    """Three panels with the same view: real (rainbow), predicted (rainbow), overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    both = np.vstack([pred, true[mask]])
    fig = plt.figure(figsize=(13, 4.8), facecolor="#fcfcfb")
    panels = [("Real structure", "real"), ("Predicted", "pred"), ("Overlay", "both")]
    for n, (name, what) in enumerate(panels, 1):
        ax = fig.add_subplot(1, 3, n, projection="3d", facecolor="#fcfcfb")
        if what == "real":
            _rainbow_trace(ax, true, mask)
        elif what == "pred":
            _rainbow_trace(ax, pred)
        else:
            t = np.where(mask[:, None], true, np.nan)
            ax.plot(*t.T, color="#9a9893", lw=2.4, label="real")
            ax.plot(*pred.T, color="#2a78d6", lw=1.6, label="predicted")
            ax.legend(loc="upper left", frameon=False, fontsize=8)
        _style_axes(ax, both)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(name, fontsize=11, color="#0b0b0b")
    fig.suptitle(title, fontsize=11, color="#0b0b0b")
    fig.text(0.5, 0.02, "Colour runs from the start of the chain (blue) to the end (red). "
             "Distances are in Å.", ha="center", fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor()); plt.close(fig)


def plot_distance_maps(path, d_exp, p_in, true_d, mask, title):
    """Predicted vs real C-beta distance map (pairs predicted beyond 22 Å shown as 22+)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("d", ["#0d366b", "#256abf", "#6da7ec", "#cde2fb", "#fcfcfb"])
    pred_map = np.where(p_in > 0.5, d_exp, FAR)
    np.fill_diagonal(pred_map, 0.0)                     # a residue's distance to itself
    real_map = np.where(mask[:, None] & mask[None, :], np.minimum(true_d, FAR), np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), facecolor="#fcfcfb")
    for ax, m, name in zip(axes, (pred_map, real_map), ("Predicted distances", "Real distances")):
        im = ax.imshow(m, cmap=cmap, vmin=2, vmax=FAR, interpolation="nearest")
        ax.set_title(name, fontsize=11); ax.set_xlabel("Residue j"); ax.set_ylabel("Residue i")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("Cβ–Cβ distance (Å, capped at 22)")
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight"); plt.close(fig)


HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__ID__ 3D</title>
<script src="https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js"></script>
<style>
 body{margin:0;font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;background:#fcfcfb;color:#0b0b0b}
 header{padding:14px 18px 6px} h1{font-size:18px;margin:0 0 4px} p{margin:2px 0;color:#52514e;font-size:13px}
 .bar{display:flex;gap:6px;flex-wrap:wrap;padding:8px 18px}
 button{border:1px solid #d6d5d0;background:#fff;border-radius:6px;padding:6px 10px;font-size:13px;cursor:pointer}
 button.on{background:#0d366b;color:#fff;border-color:#0d366b}
 #v{position:relative;width:calc(100vw - 36px);height:72vh;margin:0 18px;border:1px solid #e4e3df;border-radius:8px;background:#fff}
 .legend{padding:8px 18px;font-size:12px;color:#52514e}
 .sw{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:-2px;margin:0 4px 0 10px}
</style></head><body>
<header><h1>__ID__ (__L__ residues)</h1>
<p>Model: __RUN__ &middot; RMSD __RMSD__ &Aring; &middot; TM-score &asymp; __TM__ &middot; oracle RMSD __ORMSD__ &Aring;</p>
<p>Drag to rotate, scroll to zoom. One point per residue (C&beta; atom).</p></header>
<div class="bar">
 <button id="b-both" class="on" onclick="show('both')">Overlay</button>
 <button id="b-pred" onclick="show('pred')">Predicted</button>
 <button id="b-real" onclick="show('real')">Real</button>
 <span style="width:12px"></span>
 <button id="c-chain" class="on" onclick="colour('chain')">Colour: start &rarr; end</button>
 <button id="c-conf" onclick="colour('conf')">Colour: confidence</button>
</div>
<div id="v"></div>
<div class="legend" id="lg"></div>
<script>
const PRED = `__PRED__`;
const REAL = `__REAL__`;
const viewer = $3Dmol.createViewer("v", {backgroundColor: "white"});
const mPred = viewer.addModel(PRED, "pdb");
const mReal = viewer.addModel(REAL, "pdb");
let mode = "both", scheme = "chain";
const L = __L__;
function rainbow(a){ const t = (a.resi - 1) / Math.max(L - 1, 1);
  const c = [[48,18,59],[70,107,227],[40,187,236],[50,241,151],[164,252,60],[237,208,58],[251,128,34],[213,51,6],[122,4,3]];
  const x = t * (c.length - 1), i = Math.min(Math.floor(x), c.length - 2), f = x - i;
  const m = c[i].map((v, k) => Math.round(v + f * (c[i + 1][k] - v)));
  return "rgb(" + m.join(",") + ")"; }
function conf(a){ const b = Math.max(0, Math.min(100, a.b)) / 100;
  const lo = [214,84,52], mid = [240,208,88], hi = [42,120,214];
  const m = b < 0.5 ? lo.map((v,k)=>Math.round(v + (b/0.5)*(mid[k]-v))) : mid.map((v,k)=>Math.round(v + ((b-0.5)/0.5)*(hi[k]-v)));
  return "rgb(" + m.join(",") + ")"; }
function render(){
  viewer.setStyle({}, {});
  const predCol = scheme === "conf" ? {colorfunc: conf} : {colorfunc: rainbow};
  if (mode !== "real") viewer.setStyle({model: mPred}, {stick: Object.assign({radius: 0.35}, predCol), sphere: Object.assign({scale: 0.32}, predCol)});
  if (mode !== "pred") {
    const realCol = mode === "both" ? {color: "#9a9893"} : {colorfunc: rainbow};
    viewer.setStyle({model: mReal}, {stick: Object.assign({radius: mode === "both" ? 0.5 : 0.35, opacity: mode === "both" ? 0.55 : 1}, realCol)});
  }
  viewer.render();
  document.getElementById("lg").innerHTML = mode === "both"
    ? '<span class="sw" style="background:#9a9893"></span>real structure (grey) <span class="sw" style="background:#2a78d6"></span>prediction (' + (scheme === "conf" ? "confidence colours" : "start &rarr; end colours") + ')'
    : (scheme === "conf" && mode === "pred" ? '<span class="sw" style="background:rgb(214,84,52)"></span>low confidence <span class="sw" style="background:rgb(240,208,88)"></span>medium <span class="sw" style="background:rgb(42,120,214)"></span>high'
       : '<span class="sw" style="background:rgb(70,107,227)"></span>chain start &rarr; <span class="sw" style="background:rgb(213,51,6)"></span>chain end');
}
function show(m){ mode = m; for (const k of ["both","pred","real"]) document.getElementById("b-"+k).classList.toggle("on", k === m); render(); viewer.zoomTo(); viewer.render(); }
function colour(c){ scheme = c; for (const k of ["chain","conf"]) document.getElementById("c-"+k).classList.toggle("on", k === c); render(); }
render(); viewer.zoomTo(); viewer.render();
</script></body></html>
"""


def write_html(path, eid, run_name, L, res, ores, pred_pdb, real_pdb):
    page = (HTML_TEMPLATE.replace("__ID__", eid).replace("__RUN__", run_name).replace("__L__", str(L))
            .replace("__RMSD__", f"{res['rmsd']:.1f}").replace("__TM__", f"{res['tm']:.2f}")
            .replace("__ORMSD__", f"{ores['rmsd']:.1f}")
            .replace("__PRED__", pred_pdb).replace("__REAL__", real_pdb))
    with open(path, "w") as f:
        f.write(page)


def write_index(path, run_name, summary):
    rows = "\n".join(
        f"<tr><td><a href='{s['entity_id']}.html'>{s['entity_id']}</a></td><td>{s['length']}</td>"
        f"<td>{s['rmsd_pred']:.1f}</td><td>{s['tm_pred']:.2f}</td><td>{s['lddt_pred']:.2f}</td>"
        f"<td>{s['mean_confidence']:.0f}</td><td>{s['confidence_vs_lddt_spearman']:+.2f}</td><td>{s['rmsd_oracle']:.1f}</td>"
        f"<td><a href='{s['entity_id']}.png'>plots</a> &middot; <a href='{s['entity_id']}_maps.png'>maps</a></td></tr>"
        for s in summary)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>3D structures: {run_name}</title>
<style>body{{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;background:#fcfcfb;color:#0b0b0b;margin:24px}}
table{{border-collapse:collapse;font-size:14px}} th,td{{padding:6px 12px;border-bottom:1px solid #e4e3df;text-align:right}}
th:first-child,td:first-child{{text-align:left}} th{{color:#52514e;font-weight:600}} a{{color:#1c5cab}}
p{{color:#52514e;font-size:13px;max-width:760px}}</style></head><body>
<h2>Predicted 3D structures: model “{run_name}”</h2>
<p>Validation proteins only. RMSD in Å (lower is better). TM-score 0–1 (higher is better; approximate).
lDDT 0–1 = local accuracy (are distances to nearby residues right?), the score AlphaFold's confidence is based on.
Confidence = mean per-residue confidence 0–100. “Conf. vs lDDT” = rank correlation between a residue's confidence
and its actual local accuracy (positive means the model knows where it is right).
Oracle = the same 3D reconstruction run on the true distances.</p>
<table><tr><th>Protein</th><th>Length</th><th>RMSD</th><th>TM</th><th>lDDT</th><th>Confidence</th><th>Conf. vs lDDT</th><th>Oracle RMSD</th><th>Images</th></tr>
{rows}</table></body></html>"""
    with open(path, "w") as f:
        f.write(page)


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

    from model import load_model
    from train import get_device
    device = get_device()
    ck_path = os.path.join(CHECKPOINT_DIR, f"{args.run_name}_best.pt")
    model, _ = load_model(ck_path, device)
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
    print(f"{'protein':>9} {'L':>4} | {'RMSD pred':>9} {'TM pred':>7} {'lDDT':>5} | {'RMSD oracle':>11} {'TM oracle':>9} | time")

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
        p_in = probs[1:63].sum(0)
        confidence = residue_confidence(probs, d)
        res_lddt, overall_lddt = lddt(pred, true_cb, mask)
        _, oracle_lddt = lddt(oracle, true_cb, mask)
        conf_lddt = spearman(confidence[mask], res_lddt[mask])
        pred_pdb = pdb_text(seq, aligned, bfactor=confidence,
                            title=f"{eid} predicted by {args.run_name}; RMSD {res['rmsd']:.2f} A")
        real_pdb = pdb_text(seq, true_cb, mask=mask, title=f"{eid} experimental structure (C-beta)")
        for name, text in ((f"{eid}_pred.pdb", pred_pdb), (f"{eid}_true.pdb", real_pdb)):
            with open(os.path.join(out_dir, name), "w") as f:
                f.write(text)
        title = (f"{eid} ({len(seq)} residues), model {args.run_name}: RMSD {res['rmsd']:.1f} Å, "
                 f"TM ≈ {res['tm']:.2f}  (oracle with true distances: {ores['rmsd']:.1f} Å)")
        plot_structures(os.path.join(out_dir, f"{eid}.png"), aligned, true_cb, mask, title)
        plot_distance_maps(os.path.join(out_dir, f"{eid}_maps.png"), d, p_in, true_d, mask,
                           f"{eid}: predicted vs real Cβ distances")
        write_html(os.path.join(out_dir, f"{eid}.html"), eid, args.run_name, len(seq), res, ores,
                   pred_pdb, real_pdb)
        summary.append({"entity_id": eid, "length": len(seq), "n_observed": int(mask.sum()),
                        "rmsd_pred": round(res["rmsd"], 3), "tm_pred": round(res["tm"], 3),
                        "mirror_pred": res["mirror"],
                        "lddt_pred": round(overall_lddt, 3), "lddt_oracle": round(oracle_lddt, 3),
                        "mean_confidence": round(float(confidence.mean()), 1),
                        "confidence_vs_lddt_spearman": round(conf_lddt, 3),
                        "rmsd_oracle": round(ores["rmsd"], 3), "tm_oracle": round(ores["tm"], 3)})
        print(f"{eid:>9} {len(seq):4d} | {res['rmsd']:9.2f} {res['tm']:7.3f} {overall_lddt:5.2f} | "
              f"{ores['rmsd']:11.2f} {ores['tm']:9.3f} | {time.time() - t0:.0f}s")

    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        wr.writeheader(); wr.writerows(summary)
    m = lambda k: np.mean([s[k] for s in summary])
    print(f"\nMean over {len(summary)} proteins: predicted RMSD {m('rmsd_pred'):.2f} Å, TM ≈ {m('tm_pred'):.3f} | "
          f"oracle RMSD {m('rmsd_oracle'):.2f} Å, TM ≈ {m('tm_oracle'):.3f}")
    write_index(os.path.join(out_dir, "index.html"), args.run_name, summary)
    print(f"Mean lDDT (local accuracy, 0-1): predicted {m('lddt_pred'):.3f} | oracle {m('lddt_oracle'):.3f}")
    print(f"Confidence vs per-residue lDDT, rank correlation: "
          f"{np.nanmean([x['confidence_vs_lddt_spearman'] for x in summary]):+.2f} "
          "(positive = residues the model is confident about really are more accurate)")
    print(f"\nOpen {os.path.join(out_dir, 'index.html')} in a browser for the interactive 3D views.")


if __name__ == "__main__":
    main()
