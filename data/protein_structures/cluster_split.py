"""
Step 3: group similar proteins and split into train / val / test BY CLUSTER.

Why: the same protein appears in hundreds of PDB entries (one sequence shows
up 938 times). If copies land in both train and test, the test score is fake.
So we cluster sequences at 30% identity (standard for structure prediction)
and put whole clusters into one split.

Needs MMseqs2:  conda install -c conda-forge -c bioconda mmseqs2
           or:  brew install mmseqs2

Input : dataset_entities.txt   (entity_id \t entry_id \t sequence)
Output: dataset_split.tsv      (one row per UNIQUE sequence:
                                entity_id, entry_id, sequence, cluster_id, split)
"""
import os
import random
import shutil
import subprocess
import sys
from collections import defaultdict

INPUT_FILE = "dataset_entities.txt"
OUTPUT_FILE = "dataset_split.tsv"
WORK_DIR = "mmseqs_work"
MIN_IDENTITY = 0.3
COVERAGE = 0.8
SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def load_unique_sequences(path):
    """Keep one chain per unique sequence (the first one seen)."""
    by_seq = {}
    for line in open(path):
        entity_id, entry_id, seq = line.rstrip("\n").split("\t")
        # drop sequences with non-standard letters (X, U, etc.) - only ~400
        if set(seq) - STANDARD_AA:
            continue
        if seq not in by_seq:
            by_seq[seq] = (entity_id, entry_id)
    return by_seq


def run_mmseqs(by_seq):
    if shutil.which("mmseqs") is None:
        sys.exit("mmseqs not found. Install it with:\n"
                 "  conda install -c conda-forge -c bioconda mmseqs2\n"
                 "or\n  brew install mmseqs2")

    os.makedirs(WORK_DIR, exist_ok=True)
    fasta = os.path.join(WORK_DIR, "unique.fasta")
    with open(fasta, "w") as f:
        for seq, (entity_id, _) in by_seq.items():
            f.write(f">{entity_id}\n{seq}\n")

    prefix = os.path.join(WORK_DIR, "clu")
    subprocess.run(
        ["mmseqs", "easy-cluster", fasta, prefix, os.path.join(WORK_DIR, "tmp"),
         "--min-seq-id", str(MIN_IDENTITY), "-c", str(COVERAGE),
         "--cov-mode", "0", "-v", "1"],
        check=True,
    )
    # clu_cluster.tsv: representative \t member
    clusters = {}
    for line in open(prefix + "_cluster.tsv"):
        rep, member = line.split()
        clusters[member] = rep
    return clusters


def assign_splits(cluster_of):
    members = defaultdict(list)
    for entity_id, rep in cluster_of.items():
        members[rep].append(entity_id)

    reps = sorted(members)
    random.Random(SEED).shuffle(reps)

    # fill val and test by number of SEQUENCES, rest goes to train
    total = len(cluster_of)
    split_of_cluster = {}
    counts = {"val": 0, "test": 0}
    for rep in reps:
        size = len(members[rep])
        if counts["test"] < SPLIT_FRACTIONS["test"] * total:
            split_of_cluster[rep] = "test"; counts["test"] += size
        elif counts["val"] < SPLIT_FRACTIONS["val"] * total:
            split_of_cluster[rep] = "val"; counts["val"] += size
        else:
            split_of_cluster[rep] = "train"
    return split_of_cluster


def main():
    by_seq = load_unique_sequences(INPUT_FILE)
    print(f"Unique sequences (standard amino acids only): {len(by_seq)}")

    cluster_of = run_mmseqs(by_seq)
    split_of_cluster = assign_splits(cluster_of)

    n_split = defaultdict(int)
    n_clusters = defaultdict(int)
    for split in split_of_cluster.values():
        n_clusters[split] += 1

    with open(OUTPUT_FILE, "w") as f:
        f.write("entity_id\tentry_id\tsequence\tcluster_id\tsplit\n")
        for seq, (entity_id, entry_id) in by_seq.items():
            rep = cluster_of[entity_id]
            split = split_of_cluster[rep]
            n_split[split] += 1
            f.write(f"{entity_id}\t{entry_id}\t{seq}\t{rep}\t{split}\n")

    print(f"Clusters at {int(MIN_IDENTITY*100)}% identity: {len(split_of_cluster)}")
    for s in ("train", "val", "test"):
        print(f"  {s:5s}: {n_split[s]:6d} sequences in {n_clusters[s]:5d} clusters")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
