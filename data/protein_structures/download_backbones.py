"""
Step 4: download the real 3D structure for every chain in dataset_split.tsv
and save its backbone atoms (N, CA, C) as a small .npz file.

Uses the RCSB ModelServer, which returns ONLY the atoms of the chain we ask
for (a few KB) instead of the whole PDB entry (often MBs).

Each saved file backbone/<entity_id>.npz contains:
  seq    : the full sequence (string)
  coords : float32 array [L, 3, 3]  -> residue, atom (N, CA, C), xyz in Å
  mask   : bool array    [L]        -> True where the residue was actually
                                       seen in the experiment (missing loops
                                       etc. are False, coords are NaN there)

Safe to stop with Ctrl+C and re-run: finished chains are skipped.
At the end it writes backbone_index.tsv with one row per chain.
"""
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

SPLIT_FILE = "dataset_split.tsv"
OUT_DIR = "backbone"
INDEX_FILE = "backbone_index.tsv"
FAILED_FILE = "backbone_failed.txt"
WORKERS = 32
BACKBONE = {"N": 0, "CA": 1, "C": 2}
URL = ("https://models.rcsb.org/v1/{entry}/atoms"
       "?label_entity_id={entity}&model_nums=1&encoding=cif&copy_all_categories=false")


def parse_backbone(cif_text, seq_len):
    """Pull N/CA/C coordinates out of an mmCIF atom_site table.

    label_seq_id is the 1-based position in the full sequence, so residues
    that weren't resolved in the experiment simply never show up -> masked.
    If the entity has several copies (chains A, B, ...) we keep the one with
    the most residues observed.
    """
    lines = cif_text.splitlines()
    cols, rows, in_loop = [], [], False
    for line in lines:
        if line.startswith("_atom_site."):
            cols.append(line.split(".", 1)[1].strip())
            in_loop = True
        elif in_loop:
            if not line or line.startswith(("#", "loop_", "_")):
                if rows:
                    break
                continue
            rows.append(line.split())
    if not rows:
        return None

    c = {name: i for i, name in enumerate(cols)}
    need = ["label_atom_id", "label_seq_id", "label_asym_id",
            "Cartn_x", "Cartn_y", "Cartn_z"]
    if any(n not in c for n in need):
        return None
    alt_col = c.get("label_alt_id")

    chains = {}
    for r in rows:
        if len(r) != len(cols):
            continue
        atom = r[c["label_atom_id"]].strip('"')
        if atom not in BACKBONE:
            continue
        if alt_col is not None and r[alt_col] not in (".", "A", "?"):
            continue  # keep only the first alternate conformation
        seq_id = r[c["label_seq_id"]]
        if seq_id in (".", "?"):
            continue
        pos = int(seq_id) - 1
        if not 0 <= pos < seq_len:
            continue
        asym = r[c["label_asym_id"]]
        if asym not in chains:
            chains[asym] = np.full((seq_len, 3, 3), np.nan, dtype=np.float32)
        a = BACKBONE[atom]
        if np.isnan(chains[asym][pos, a, 0]):
            chains[asym][pos, a] = [float(r[c["Cartn_x"]]),
                                    float(r[c["Cartn_y"]]),
                                    float(r[c["Cartn_z"]])]
    if not chains:
        return None

    best = max(chains.values(), key=lambda x: (~np.isnan(x).any(axis=(1, 2))).sum())
    mask = ~np.isnan(best).any(axis=(1, 2))   # residue needs all of N, CA, C
    return best, mask


def fetch_one(session, row):
    entity_id, seq = row["entity_id"], row["sequence"]
    out_path = os.path.join(OUT_DIR, f"{entity_id}.npz")
    if os.path.exists(out_path):
        return entity_id, "skip"
    entry, entity = entity_id.split("_")
    for attempt in range(3):
        try:
            resp = session.get(URL.format(entry=entry, entity=entity), timeout=60)
            if resp.status_code == 200:
                break
            if resp.status_code == 404:
                return entity_id, "fail: 404"
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    else:
        return entity_id, "fail: network"

    parsed = parse_backbone(resp.text, len(seq))
    if parsed is None:
        return entity_id, "fail: no backbone atoms"
    coords, mask = parsed
    np.savez_compressed(out_path, seq=seq, coords=coords, mask=mask)
    return entity_id, "ok"


def write_index(rows):
    with open(INDEX_FILE, "w") as f:
        f.write("entity_id\tsplit\tcluster_id\tlength\tn_observed\tfrac_observed\n")
        for row in rows:
            path = os.path.join(OUT_DIR, f"{row['entity_id']}.npz")
            if not os.path.exists(path):
                continue
            mask = np.load(path)["mask"]
            n = int(mask.sum())
            f.write(f"{row['entity_id']}\t{row['split']}\t{row['cluster_id']}\t"
                    f"{len(mask)}\t{n}\t{n/len(mask):.3f}\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(SPLIT_FILE) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    print(f"{len(rows)} chains to fetch ({WORKERS} at a time)")

    session = requests.Session()
    counts = {"ok": 0, "skip": 0, "fail": 0}
    failed = []
    start = time.time()
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = [pool.submit(fetch_one, session, r) for r in rows]
        for i, fut in enumerate(as_completed(futures), 1):
            entity_id, status = fut.result()
            key = status.split(":")[0]
            counts[key] += 1
            if key == "fail":
                failed.append(f"{entity_id}\t{status}")
            if i % 500 == 0 or i == len(rows):
                rate = i / (time.time() - start)
                left = (len(rows) - i) / rate / 60
                print(f"{i}/{len(rows)}  ok={counts['ok']} skipped={counts['skip']} "
                      f"failed={counts['fail']}  ~{left:.0f} min left")

    with open(FAILED_FILE, "w") as f:
        f.write("\n".join(failed) + ("\n" if failed else ""))

    print("Building index...")
    write_index(rows)
    print(f"\nDone. Saved {counts['ok'] + counts['skip']} chains to {OUT_DIR}/")
    print(f"Failed: {counts['fail']} (listed in {FAILED_FILE})")
    print(f"Index: {INDEX_FILE}")


if __name__ == "__main__":
    main()
