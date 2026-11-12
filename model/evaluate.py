"""
Evaluate the trained neural model on the validation set, add it to the
baseline results, and make the figures for the preliminary-results section.

Run from ~/machine_learning (after baselines.py and train.py):
    python model/evaluate.py                      # the basic model (column 'neural')
    python model/evaluate.py --run_name long      # model/checkpoints/long_best.pt (column 'neural_long')

Writes to results/progress_report/:
    results_table.md            Table: all methods present in per_chain_val.csv
    per_chain_val.csv           per-chain precision for every method; only this run's
                                column is added or replaced, all other columns are kept
    model_summary.json          parameter count, training time, etc.
    fig_results.png             bar chart of the table
    fig_learning_curve.png      loss and precision over training
    fig_example_predictions.png true vs predicted contacts for 3 proteins
For a run other than 'basic' the last four files get a suffix instead of being
overwritten (model_summary_long.json, fig_results_long.png, ...), and the
learning curve is read from train_log_<run>.csv.
"""
import argparse
import csv
import json
import os

from common import (CHECKPOINT_DIR, CONTACT_CUTOFF, EVAL_MAX_LEN, RESULTS_DIR, distance_matrix,
                    load_chain, load_index, long_range_pairs, top_l5_precision, write_table,
                    bootstrap_ci)

import numpy as np
import torch
import torch.nn.functional as F

from dataset import CONTACT_BINS, ProteinDataset, collate

METHODS = [("random", "Random"), ("separation", "Sequence separation"),
           ("logreg", "Logistic regression"), ("neural", "Neural distogram model")]
# every column that can appear in per_chain_val.csv, in table order
ALL_METHODS = [("random", "Random"), ("separation", "Sequence separation"),
               ("logreg", "Logistic regression"),
               ("logreg_rich", "Logistic regression, rich features"),
               ("gbt", "Gradient-boosted trees, rich features"),
               ("neural", "Neural distogram model")]
ID_COLS = ("entity_id", "length")


def add_column(base, column, values):
    """
    Return the per-chain rows with `column` added (or replaced). Every other column keeps
    its original text, and the chain ids and their order are unchanged.
    """
    assert len(values) == len(base), "chain lists don't match the baselines"
    out = []
    for r, v in zip(base, values):
        r = dict(r)
        r[column] = f"{v:.4f}"
        out.append(r)
    assert [r["entity_id"] for r in out] == [r["entity_id"] for r in base]
    return out


def write_per_chain(path, rows):
    """Write per_chain_val.csv from dict rows, keeping the existing column order."""
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)    # same line endings as tree_models.py wrote
        w.writeheader()
        w.writerows(rows)


@torch.no_grad()
def evaluate_chains(model, rows, device, return_maps_for=()):
    """
    Full-length chains one at a time -> mean val loss, per-chain precision, maps.

    The true contacts are computed from the actual C-beta distances (< 8 Å, exactly as in
    baselines.py), so every method is scored against identical labels. The label implied
    by the distogram bins (b["bins"] < CONTACT_BINS) is d < 7.81 Å, because the 63 bin
    edges are spaced 0.32 Å apart and no edge falls on 8 Å; that variant is kept only as a
    cross-check (precisions_bins, returned as the 4th value).
    """
    model.eval()
    ds = ProteinDataset.__new__(ProteinDataset)        # reuse loading code, no cropping
    ds.rows, ds.crop_size, ds.random_crop = rows, None, False
    rng = np.random.default_rng(0)
    rng_bins = np.random.default_rng(0)
    losses, precisions, precisions_bins, maps = [], [], [], {}
    for k in range(len(rows)):
        b = collate([ds[k]])
        tokens, seq_mask = b["tokens"].to(device), b["seq_mask"].to(device)
        logits = model(tokens, seq_mask)
        bins, pm = b["bins"].to(device), b["pair_mask"].to(device)
        loss = F.cross_entropy(logits, bins, reduction="none")
        losses.append(((loss * pm).sum() / pm.sum().clamp(min=1)).item())
        # score = P(Cβ distance falls in a bin that lies below 8 Å); used only to rank pairs
        prob = logits.softmax(1)[0, :CONTACT_BINS].sum(0).float().cpu().numpy()
        _, coords, _ = load_chain(rows[k]["entity_id"])
        true = distance_matrix(coords) < CONTACT_CUTOFF                  # same labels as baselines
        true_bins = (b["bins"][0] < CONTACT_BINS).numpy()                # d < 7.81 Å (cross-check)
        mask = b["mask"][0].numpy()
        i, j = long_range_pairs(mask)
        L = len(mask)
        precisions.append(top_l5_precision(prob[i, j], true[i, j], L, rng) if len(i) else None)
        precisions_bins.append(top_l5_precision(prob[i, j], true_bins[i, j], L, rng_bins) if len(i) else None)
        if rows[k]["entity_id"] in return_maps_for:
            maps[rows[k]["entity_id"]] = (prob, true, mask)
    model.train()
    keep = [p for p in precisions if p is not None]
    keep_bins = [p for p in precisions_bins if p is not None]
    return float(np.mean(losses)), keep, maps, keep_bins


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--run_name", default="basic",
                   help="which model to evaluate: model/checkpoints/<run_name>_best.pt "
                        "(default basic; its results go in column 'neural', any other run "
                        "goes in column 'neural_<run_name>' with suffixed output files)")
    args = p.parse_args()
    name = args.run_name + ("_quick" if args.quick else "")
    column = "neural" if args.run_name == "basic" else f"neural_{args.run_name}"
    suffix = "" if args.run_name == "basic" else f"_{args.run_name}"
    # quick test results go to their own folder so the real ones are never overwritten
    out_dir = os.path.join(RESULTS_DIR, "quick_test") if args.quick else RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from train import get_device
    from model import count_parameters, load_model

    device = get_device()
    ck_path = os.path.join(CHECKPOINT_DIR, f"{name}_best.pt")
    if not os.path.exists(ck_path):
        raise SystemExit(f"No checkpoint at {ck_path}. Run python model/train.py "
                         f"--run_name {args.run_name} first.")
    model, ck = load_model(ck_path, device)   # rebuilds the architecture stored in the checkpoint
    n_params = count_parameters(model)

    # same chains, same order as baselines.py
    base_path = os.path.join(RESULTS_DIR, "per_chain_val.csv")
    if not os.path.exists(base_path):
        raise SystemExit("Run python model/baselines.py first.")
    base = list(csv.DictReader(open(base_path)))
    by_id = {r["entity_id"]: r for r in load_index("val", max_len=EVAL_MAX_LEN)}
    rows = [by_id[r["entity_id"]] for r in base]
    if args.quick:
        rows, base = rows[:20], base[:20]

    # pick 3 example proteins of different sizes for the figure
    lengths = np.array([int(r["length"]) for r in rows])
    examples = []
    for target in (100, 180, 260):
        examples.append(rows[int(np.argmin(np.abs(lengths - target)))]["entity_id"])

    print(f"Evaluating neural model '{args.run_name}' ({n_params:,} parameters) on {len(rows):,} val chains...")
    val_loss, prec, maps, prec_bins = evaluate_chains(model, rows, device, return_maps_for=set(examples))
    assert len(prec) == len(base), "chain lists don't match the baselines"

    # keep every existing column; only this run's column is added or replaced
    new_rows = add_column(base, column, prec)
    write_per_chain(os.path.join(out_dir, "per_chain_val.csv"), new_rows)
    methods = [m for m in ALL_METHODS if m[0] in new_rows[0]]
    if column != "neural":
        methods.append((column, f"Neural distogram model ({args.run_name} training)"))
    per_chain = {m: [float(r[m]) for r in new_rows] for m, _ in methods}
    print()
    write_table(per_chain, methods, out_dir=out_dir)

    log_name = "train_log.csv" if args.run_name == "basic" else f"train_log_{args.run_name}.csv"
    log_path = os.path.join(out_dir, log_name)
    log = list(csv.DictReader(open(log_path))) if os.path.exists(log_path) else []
    with open(os.path.join(out_dir, f"model_summary{suffix}.json"), "w") as f:
        json.dump({"parameters": n_params, "best_epoch": ck["epoch"],
                   "training_minutes": round(ck["minutes"], 1),
                   "device": str(device), "val_loss_full_chains": round(val_loss, 4),
                   "neural_precision_with_bin_labels_d_lt_7.81A": round(float(np.mean(prec_bins)), 4),
                   "training_settings": ck["args"]}, f, indent=2)

    # ---------------- figures ----------------
    SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
                         "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
                         "axes.titlelocation": "left"})

    # 1. results bar chart with 95% CIs
    # the basic run keeps the original four bars; another run shows every method present
    fig_methods = METHODS if args.run_name == "basic" else methods
    fig, ax = plt.subplots(figsize=(7.5, max(3.2, 0.5 * len(fig_methods) + 0.9)))
    labels, means, errs = [], [], []
    for m, label in fig_methods:
        mean, lo, hi = bootstrap_ci(per_chain[m])
        labels.append(label); means.append(100 * mean)
        errs.append([100 * (mean - lo), 100 * (hi - mean)])
    y = np.arange(len(labels))[::-1]
    ax.barh(y, means, color=BLUE, height=0.55, xerr=np.array(errs).T,
            error_kw={"ecolor": INK2, "capsize": 3, "lw": 1})
    for yy, mval, e in zip(y, means, errs):
        ax.text(mval + e[1] + 0.3, yy, f"{mval:.1f}%", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(m + e[1] for m, e in zip(means, errs)) * 1.18)
    ax.set_xlabel("Long-range top-L/5 contact precision on validation (%), 95% CI")
    ax.set_title("Preliminary results")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"fig_results{suffix}.png"), dpi=200)
    plt.close(fig)

    # 2. learning curve
    if log:
        ep = [int(r["epoch"]) for r in log]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        axes[0].plot(ep, [float(r["train_loss"]) for r in log], color=BLUE, lw=2, marker="o", ms=4, label="train (128-residue crops)")
        axes[0].plot(ep, [float(r["val_loss"]) for r in log], color=ORANGE, lw=2, marker="o", ms=4, label="validation (full chains)")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-entropy loss")
        axes[0].set_title("Distogram loss", fontsize=11); axes[0].legend(frameon=False)
        axes[1].plot(ep, [100 * float(r["val_precision"]) for r in log], color=BLUE, lw=2, marker="o", ms=4)
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Precision (%)")
        axes[1].set_title("Val long-range top-L/5 precision", fontsize=11)
        for a in axes:
            a.xaxis.get_major_locator().set_params(integer=True)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"fig_learning_curve{suffix}.png"), dpi=200)
        plt.close(fig)

    # 3. true vs predicted contacts
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("b", [SURFACE, "#86b6ef", "#1c5cab", "#0d366b"])
    fig, axes = plt.subplots(1, len(examples), figsize=(4 * len(examples), 4.2))
    for a, eid in zip(np.atleast_1d(axes), examples):
        prob, true, mask = maps[eid]
        L = len(mask)
        img = np.full((L, L), np.nan)
        iu = np.triu_indices(L, 1); il = np.tril_indices(L, -1)
        img[iu] = prob[iu]                        # upper triangle: prediction
        img[il] = true[il].astype(float)          # lower triangle: truth
        img[~(mask[:, None] & mask[None, :])] = np.nan
        a.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        a.set_title(f"{eid} ({L} residues)", fontsize=10); a.grid(False)
        a.set_xlabel("Residue j"); a.set_ylabel("Residue i")
    fig.suptitle("Contacts: model prediction (upper triangle) vs experimental truth (lower triangle)",
                 x=0.02, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(out_dir, f"fig_example_predictions{suffix}.png"), dpi=200)
    plt.close(fig)
    print(f"\nModel parameters: {n_params:,}")
    print(f"Figures and tables saved in {out_dir}")


if __name__ == "__main__":
    main()
