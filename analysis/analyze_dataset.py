"""
Dataset analysis for the project proposal.
Run from ~/machine_learning/analysis:   python analyze_dataset.py
Writes figures (PNG) and summary.md into this folder.
"""
import csv
import os
import random
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

DATA = "../data/protein_structures"
OUT = "."
random.seed(0)

# ---------- style (light, print-friendly) ----------
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; GRID = "#e4e3df"
BLUE = "#2a78d6"; ORANGE = "#eb6834"; AQUA = "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "figure.dpi": 150,
})

def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=200)
    plt.close(fig)

def count_lines(path):
    with open(path) as f:
        return sum(1 for l in f if l.strip())

# ---------- load ----------
index = list(csv.DictReader(open(f"{DATA}/backbone_index.tsv"), delimiter="\t"))
split_rows = list(csv.DictReader(open(f"{DATA}/dataset_split.tsv"), delimiter="\t"))
all_chains = [l.rstrip("\n").split("\t") for l in open(f"{DATA}/dataset_entities.txt")]

n_entries = count_lines(f"{DATA}/human_pdb_ids.txt")
n_entities = count_lines(f"{DATA}/human_protein_entities_v2.txt")
n_len_filtered = len(all_chains)
n_unique = len(split_rows)
n_coords = len(index)
usable = [r for r in index if int(r["n_observed"]) >= 30 and float(r["frac_observed"]) >= 0.5]
n_usable = len(usable)
n_clusters = len({r["cluster_id"] for r in split_rows})

S = {}  # summary numbers

# ---------- 1. pipeline funnel ----------
stages = [
    ("PDB entries with a human chain", n_entries),
    ("Human protein chains (experimental)", n_entities),
    ("Length 50–2000 residues", n_len_filtered),
    ("Unique sequences", n_unique),
    ("With 3D backbone downloaded", n_coords),
    ("Usable for training (≥50% observed)", n_usable),
    ("Sequence clusters (30% identity)", n_clusters),
]
fig, ax = plt.subplots(figsize=(8, 3.6))
labels = [s[0] for s in stages][::-1]; vals = [s[1] for s in stages][::-1]
bars = ax.barh(labels, vals, color=BLUE, height=0.6, edgecolor=SURFACE, linewidth=2)
for b, v in zip(bars, vals):
    ax.text(b.get_width() + max(vals) * 0.01, b.get_y() + b.get_height() / 2,
            f"{v:,}", va="center", color=INK, fontsize=9)
ax.set_xlim(0, max(vals) * 1.15); ax.grid(axis="y", visible=False)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x else "0"))
ax.set_xlabel("Count")
ax.set_title("Data pipeline: from the PDB to a training set")
save(fig, "fig1_pipeline.png")

# ---------- 2. redundancy ----------
seq_counts = Counter(c[2] for c in all_chains)
copies = np.array(list(seq_counts.values()))
cl_sizes = np.array(list(Counter(r["cluster_id"] for r in split_rows).values()))
S["max_copies"] = int(copies.max())
S["seqs_seen_once_pct"] = 100 * (copies == 1).mean()
S["singleton_clusters_pct"] = 100 * (cl_sizes == 1).mean()
top = np.sort(cl_sizes)[::-1]
k = max(1, len(top) // 100)
S["top1pct_clusters_share"] = 100 * top[:k].sum() / top.sum()
S["largest_cluster"] = int(top[0])

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, data, title, xl in [
    (axes[0], copies, "Copies of each sequence in the PDB", "PDB chains with identical sequence"),
    (axes[1], cl_sizes, "Unique sequences per cluster", "Cluster size (sequences)"),
]:
    bins = np.unique(np.logspace(0, np.log10(data.max() + 1), 30).astype(int))
    ax.hist(data, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(title, fontsize=11); ax.set_xlabel(xl); ax.set_ylabel("Count")
fig.suptitle("Heavy redundancy: why the split is done by cluster", x=0.02, ha="left",
             fontweight="bold", fontsize=12)
save(fig, "fig2_redundancy.png")

# ---------- 3. splits ----------
split_counts = Counter(r["split"] for r in usable)
split_clusters = {s: len({r["cluster_id"] for r in usable if r["split"] == s})
                  for s in ("train", "val", "test")}

# ---------- 4. lengths ----------
lengths = np.array([int(r["length"]) for r in usable])
S["len_median"] = int(np.median(lengths)); S["len_p90"] = int(np.percentile(lengths, 90))
S["len_over_128_pct"] = 100 * (lengths > 128).mean()
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.hist(lengths, bins=np.arange(50, 1250, 25), color=BLUE, edgecolor=SURFACE, linewidth=1)
ax.axvline(128, color=INK2, lw=1.2, ls="--", label="training crop (128 residues)")
ax.axvline(S["len_median"], color=ORANGE, lw=1.5, label=f"median ({S['len_median']} residues)")
ax.legend(frameon=False, loc="upper right", labelcolor=INK2)
ax.set_xlabel("Chain length (residues, up to 1,250 shown)"); ax.set_ylabel("Chains")
ax.set_title("Chain length distribution")
save(fig, "fig3_lengths.png")

# ---------- 5. observed fraction ----------
frac = np.array([float(r["frac_observed"]) for r in index])
S["frac_median"] = float(np.median(frac))
S["fully_observed_pct"] = 100 * (frac >= 0.999).mean()
S["below_half_pct"] = 100 * (frac < 0.5).mean()
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.hist(frac, bins=np.linspace(0, 1, 41), color=BLUE, edgecolor=SURFACE, linewidth=1)
ax.axvline(0.5, color=INK2, lw=1.2, ls="--", label="chains left of this line are excluded")
ax.legend(frameon=False, loc="upper left", labelcolor=INK2)
ax.set_xlabel("Fraction of residues with resolved backbone coordinates"); ax.set_ylabel("Chains")
ax.set_title("Missing residues in experimental structures")
save(fig, "fig4_observed_fraction.png")

# ---------- 6. contacts vs sequence separation (sample) ----------
def cbeta(c):
    n, ca, cc = c[:, 0], c[:, 1], c[:, 2]
    b = ca - n; c_ = cc - ca; a = np.cross(b, c_)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c_ + ca

sample = random.sample(usable, 3000)
max_sep = 200
contact_hits = np.zeros(max_sep + 1); pair_tot = np.zeros(max_sep + 1)
range_hits = {"short (6–11)": [0, 0], "medium (12–23)": [0, 0], "long (≥24)": [0, 0]}
all_d = []
for r in sample:
    d = np.load(f"{DATA}/backbone/{r['entity_id']}.npz")
    c, m = d["coords"], d["mask"]
    idx = np.flatnonzero(m)
    if len(idx) < 30:
        continue
    cb = cbeta(c[idx])
    D = np.linalg.norm(cb[:, None] - cb[None], axis=-1)
    i, j = np.triu_indices(len(idx), 1)
    sep = idx[j] - idx[i]; dist = D[i, j]; con = dist < 8
    keep = sep <= max_sep
    np.add.at(contact_hits, sep[keep], con[keep]); np.add.at(pair_tot, sep[keep], 1)
    for name, lo, hi in [("short (6–11)", 6, 11), ("medium (12–23)", 12, 23), ("long (≥24)", 24, 10**6)]:
        sel = (sep >= lo) & (sep <= hi)
        range_hits[name][0] += con[sel].sum(); range_hits[name][1] += sel.sum()
    all_d.append(dist[sep >= 6][::7])
for k_, (h, t) in range_hits.items():
    S[f"contact_rate_{k_}"] = 100 * h / max(t, 1)

fig, ax = plt.subplots(figsize=(8, 3.4))
seps = np.arange(1, max_sep + 1)
rate = 100 * contact_hits[1:] / np.maximum(pair_tot[1:], 1)
ax.plot(seps, rate, color=BLUE, lw=2)
ax.axvline(24, color=INK2, lw=1.2, ls="--")
ax.text(26, rate.max() * 0.6, "long-range (|i−j| ≥ 24)", color=INK2, fontsize=9)
ax.set_yscale("log"); ax.set_xlabel("Sequence separation |i − j|")
ax.set_ylabel("% of pairs in contact (< 8 Å)")
ax.set_title("Contacts get rare fast with sequence separation")
save(fig, "fig5_contacts_vs_separation.png")

# distance distribution
all_d = np.concatenate(all_d)
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.hist(all_d[all_d < 60], bins=np.arange(0, 60.5, 1), color=BLUE, edgecolor=SURFACE, linewidth=1)
ax.axvspan(2, 22, color=AQUA, alpha=0.12, lw=0, label="range the model predicts (2–22 Å, 64 bins)")
ax.axvline(8, color=INK2, lw=1.2, ls="--", label="contact cutoff (8 Å)")
ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
ax.legend(frameon=False, loc="upper right", labelcolor=INK2)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x else "0"))
ax.set_xlabel("Cβ–Cβ distance (Å), pairs with |i − j| ≥ 6"); ax.set_ylabel("Pairs (subsampled)")
ax.set_title("Residue-pair distance distribution")
save(fig, "fig6_distance_distribution.png")
S["pairs_within_22A_pct"] = 100 * (all_d <= 22).mean()

# ---------- 7. example distance map ----------
cands = [r for r in usable if r["split"] == "test" and 100 <= int(r["length"]) <= 160
         and float(r["frac_observed"]) >= 0.98]
ex = random.choice(cands)
d = np.load(f"{DATA}/backbone/{ex['entity_id']}.npz")
cb = cbeta(np.nan_to_num(d["coords"]))
D = np.linalg.norm(cb[:, None] - cb[None], axis=-1)
D[~(d["mask"][:, None] & d["mask"][None])] = np.nan
from matplotlib.colors import LinearSegmentedColormap
blues = LinearSegmentedColormap.from_list("b", ["#0d366b", "#256abf", "#6da7ec", "#cde2fb", SURFACE])
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
im = axes[0].imshow(np.clip(D, 0, 30), cmap=blues, vmin=0, vmax=30, interpolation="nearest")
axes[0].set_title("Cβ distance map (target)", fontsize=11)
cb_ = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04); cb_.set_label("Å (capped at 30)")
axes[1].imshow(np.where(np.isnan(D), np.nan, (D < 8).astype(float)), cmap=LinearSegmentedColormap.from_list("c", [SURFACE, "#184f95"]), interpolation="nearest")
axes[1].set_title("Contact map (< 8 Å)", fontsize=11)
for a in axes:
    a.grid(False); a.set_xlabel("Residue j"); a.set_ylabel("Residue i")
fig.suptitle(f"What the model learns to predict: test chain {ex['entity_id']} ({ex['length']} residues)",
             x=0.02, ha="left", fontweight="bold", fontsize=12)
save(fig, "fig7_example_distance_map.png")

# ---------- 8. amino-acid composition ----------
aa = "ACDEFGHIKLMNPQRSTVWY"
comp = Counter()
for r in split_rows:
    comp.update(r["sequence"])
tot = sum(comp[a] for a in aa)
order = sorted(aa, key=lambda a: -comp[a])
fig, ax = plt.subplots(figsize=(8, 3.0))
ax.bar(order, [100 * comp[a] / tot for a in order], color=BLUE, width=0.7, edgecolor=SURFACE, linewidth=2)
ax.set_ylabel("% of residues"); ax.grid(axis="x", visible=False)
ax.set_title("Amino-acid composition (unique sequences)")
save(fig, "fig8_aa_composition.png")

# ---------- summary ----------
with open(os.path.join(OUT, "summary.md"), "w") as f:
    f.write("# Dataset summary\n\n## Pipeline\n\n| Stage | Count |\n|---|---|\n")
    for name, v in stages:
        f.write(f"| {name} | {v:,} |\n")
    f.write("\n## Train / validation / test (usable chains, split by 30%-identity cluster)\n\n"
            "| Split | Chains | Clusters |\n|---|---|---|\n")
    for s in ("train", "val", "test"):
        f.write(f"| {s} | {split_counts[s]:,} | {split_clusters[s]:,} |\n")
    f.write(f"""
## Key numbers

- The most-repeated sequence appears in **{S['max_copies']:,}** PDB chains; {S['seqs_seen_once_pct']:.0f}% of unique sequences appear only once, the rest are repeated.
- Largest cluster: {S['largest_cluster']:,} sequences. The top 1% of clusters hold {S['top1pct_clusters_share']:.0f}% of all unique sequences; {S['singleton_clusters_pct']:.0f}% of clusters are a single sequence.
- Chain length: median {S['len_median']}, 90th percentile {S['len_p90']}; {S['len_over_128_pct']:.0f}% of chains are longer than the 128-residue training crop.
- Resolved coordinates: median {100*S['frac_median']:.0f}% of a chain; {S['fully_observed_pct']:.0f}% of chains fully observed; {S['below_half_pct']:.1f}% under 50% (excluded).
- Contact rates (Cβ < 8 Å, from 3,000 sampled chains): short-range {S['contact_rate_short (6–11)']:.1f}%, medium-range {S['contact_rate_medium (12–23)']:.1f}%, long-range {S['contact_rate_long (≥24)']:.2f}%.
- {S['pairs_within_22A_pct']:.0f}% of residue pairs (|i−j| ≥ 6) are within the 22 Å predicted range.
- Example map: test chain {ex['entity_id']}.

## Limitations to mention
- Human proteins only; no MSAs (single-sequence input).
- Experimental method and resolution not yet recorded per chain (X-ray, cryo-EM and NMR are mixed).
- Virtual Cβ is computed from backbone N/CA/C rather than read from the files.
""")
print(open(os.path.join(OUT, "summary.md")).read())
