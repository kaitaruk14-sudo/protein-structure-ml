"""
Train BasicDistogramNet.

Run from ~/machine_learning:
    python model/train.py --quick       # 1-2 minute check that everything works
    python model/train.py               # real run (stops by itself after --max_minutes)

Ctrl+C is safe: re-running the same command continues from the last checkpoint.
Writes: model/checkpoints/basic_{best,last}.pt   (not uploaded to GitHub)
        results/progress_report/train_log.csv
"""
import argparse
import math
import os
import random
import time

from common import CHECKPOINT_DIR, EVAL_MAX_LEN, RESULTS_DIR  # sets the macOS OpenMP fix first

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import ProteinDataset, collate
from evaluate import evaluate_chains
from model import BasicDistogramNet, count_parameters


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():   # Apple Silicon GPU
        return torch.device("mps")
    return torch.device("cpu")


def masked_cross_entropy(logits, bins, pair_mask):
    loss = F.cross_entropy(logits, bins, reduction="none")
    m = pair_mask.float()
    return (loss * m).sum() / m.sum().clamp(min=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--crop", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_minutes", type=float, default=75,
                   help="stop training after this many minutes")
    p.add_argument("--val_chains", type=int, default=300,
                   help="validation chains checked after each epoch")
    p.add_argument("--quick", action="store_true", help="tiny smoke test")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.set_num_threads(1)   # see the OpenMP note in common.py
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    name = "basic_quick" if args.quick else "basic"
    if args.quick:
        args.epochs, args.val_chains = 1, 10
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = get_device()
    train_ds = ProteinDataset("train", crop_size=args.crop, random_crop=True,
                              max_items=200 if args.quick else None)
    val_rows = ProteinDataset("val", max_len=EVAL_MAX_LEN).rows
    val_rows = random.Random(0).sample(val_rows, min(args.val_chains, len(val_rows)))

    samples_per_epoch = 64 if args.quick else train_ds.n_clusters
    sampler = WeightedRandomSampler(train_ds.weights, samples_per_epoch, replacement=True)
    loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                        collate_fn=collate, num_workers=0)

    model = BasicDistogramNet().to(device)
    print(f"Device: {device}")
    print(f"Train: {len(train_ds):,} chains in {train_ds.n_clusters:,} clusters "
          f"({samples_per_epoch:,} samples per epoch) | val check: {len(val_rows)} chains")
    print(f"Model parameters: {count_parameters(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(samples_per_epoch / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, min(200, total_steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))

    last_path = os.path.join(CHECKPOINT_DIR, f"{name}_last.pt")
    best_path = os.path.join(CHECKPOINT_DIR, f"{name}_best.pt")
    log_dir = os.path.join(RESULTS_DIR, "quick_test") if args.quick else RESULTS_DIR
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train_log.csv")
    start_epoch, best_prec, minutes_done = 0, -1.0, 0.0
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best_prec, minutes_done = ck["epoch"], ck["best_prec"], ck["minutes"]
        print(f"Resuming after epoch {start_epoch} ({minutes_done:.0f} min done)")
    elif not os.path.exists(log_path) or args.quick:
        with open(log_path, "w") as f:
            f.write("epoch,step,minutes,train_loss,val_loss,val_precision\n")

    def save(path, epoch, minutes):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch, "best_prec": best_prec,
                    "minutes": minutes, "args": vars(args)}, path)

    t0 = time.time() - minutes_done * 60
    out_of_time = False
    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            running, n, t_epoch = 0.0, 0, time.time()
            for step, batch in enumerate(loader, 1):
                tokens = batch["tokens"].to(device)
                seq_mask = batch["seq_mask"].to(device)
                logits = model(tokens, seq_mask)
                loss = masked_cross_entropy(logits, batch["bins"].to(device),
                                            batch["pair_mask"].to(device))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()
                running += loss.item(); n += 1
                if step % 50 == 0 or (args.quick and step % 2 == 0):
                    per = (time.time() - t_epoch) / step
                    print(f"  epoch {epoch+1} step {step}/{steps_per_epoch}  loss {running/n:.3f}  "
                          f"{per:.2f}s/step  (~{per*steps_per_epoch/60:.0f} min per epoch, "
                          f"{(time.time()-t0)/60:.0f}/{args.max_minutes:.0f} min used)")
                if (time.time() - t0) / 60 > args.max_minutes:
                    out_of_time = True
                    break

            val_loss, precisions, _ = evaluate_chains(model, val_rows, device)
            val_prec = float(np.mean(precisions))
            minutes = (time.time() - t0) / 60
            print(f"Epoch {epoch+1}: train loss {running/max(n,1):.3f} | val loss {val_loss:.3f} | "
                  f"val long-range top-L/5 precision {100*val_prec:.2f}% | {minutes:.0f} min")
            with open(log_path, "a") as f:
                f.write(f"{epoch+1},{n},{minutes:.1f},{running/max(n,1):.4f},{val_loss:.4f},{val_prec:.4f}\n")
            if val_prec > best_prec:
                best_prec = val_prec
                save(best_path, epoch + 1, minutes)
                print("  new best -> saved")
            save(last_path, epoch + 1, minutes)
            if out_of_time:
                print(f"Reached the {args.max_minutes:.0f}-minute limit, stopping.")
                break
    except KeyboardInterrupt:
        print("\nStopped. Re-run the same command to continue from the last finished epoch.")

    print(f"Best val precision during training: {100*best_prec:.2f}%")
    print("Next: python model/evaluate.py" + (" --quick" if args.quick else ""))


if __name__ == "__main__":
    main()
