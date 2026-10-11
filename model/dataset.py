"""
PyTorch Dataset for the neural model.

Each example: the amino-acid sequence (input) and, for every pair of residues,
which of 64 distance bins (2-22 Å) their C-beta atoms fall in (target).
Long chains are cut to a random 128-residue window during training.
"""
import random
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

from common import distance_matrix, load_chain, load_index

PAD = 20                      # padding token (amino acids are 0-19)
N_BINS = 64
BIN_EDGES = np.linspace(2.0, 22.0, N_BINS - 1)   # 63 edges -> 64 bins
CONTACT_BINS = int((BIN_EDGES <= 8.0).sum())      # bins that mean "< 8 Å"


class ProteinDataset(Dataset):
    def __init__(self, split, crop_size=128, random_crop=True, max_len=None, max_items=None):
        self.rows = load_index(split, max_len=max_len)
        if max_items:
            self.rows = self.rows[:max_items]
        self.crop_size = crop_size
        self.random_crop = random_crop
        # sample weight 1 / cluster size -> every cluster seen about equally often
        sizes = Counter(r["cluster_id"] for r in self.rows)
        self.weights = [1.0 / sizes[r["cluster_id"]] for r in self.rows]
        self.n_clusters = len(sizes)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        entity_id = self.rows[idx]["entity_id"]
        tokens, coords, mask = load_chain(entity_id)
        L = len(tokens)
        if self.crop_size and L > self.crop_size:
            if self.random_crop:   # window centred on a random observed residue
                centre = random.choice(np.flatnonzero(mask))
                start = int(np.clip(centre - self.crop_size // 2, 0, L - self.crop_size))
            else:
                start = 0
            sl = slice(start, start + self.crop_size)
            tokens, coords, mask = tokens[sl], coords[sl], mask[sl]
        bins = np.digitize(distance_matrix(coords), BIN_EDGES)
        return {"id": entity_id,
                "tokens": torch.from_numpy(tokens),
                "bins": torch.from_numpy(bins.astype(np.int64)),
                "pair_mask": torch.from_numpy(mask[:, None] & mask[None, :]),
                "mask": torch.from_numpy(mask)}


def collate(batch):
    """Pad all chains in a batch to the same length."""
    L = max(len(b["tokens"]) for b in batch)
    B = len(batch)
    out = {"ids": [b["id"] for b in batch],
           "tokens": torch.full((B, L), PAD, dtype=torch.long),
           "bins": torch.zeros((B, L, L), dtype=torch.long),
           "pair_mask": torch.zeros((B, L, L), dtype=torch.bool),
           "mask": torch.zeros((B, L), dtype=torch.bool),
           "seq_mask": torch.zeros((B, L), dtype=torch.bool)}
    for k, b in enumerate(batch):
        n = len(b["tokens"])
        out["tokens"][k, :n] = b["tokens"]
        out["bins"][k, :n, :n] = b["bins"]
        out["pair_mask"][k, :n, :n] = b["pair_mask"]
        out["mask"][k, :n] = b["mask"]
        out["seq_mask"][k, :n] = True
    return out
