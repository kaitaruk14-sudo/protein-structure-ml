python - <<'PY'
import gzip
import glob
from collections import Counter

counts = Counter()

files = glob.glob("*.gz")

for i, filename in enumerate(files, 1):
    try:
        with gzip.open(filename, "rt", errors="ignore") as f:
            header = ""
            for _ in range(30):
                line = f.readline()
                if not line:
                    break
                header += line.upper()

        if "ALPHAFOLD" in header:
            counts["AlphaFold"] += 1
        elif "EXPERIMENT" in header:
            counts["Experimental"] += 1
        elif "X-RAY" in header:
            counts["X-ray"] += 1
        elif "NMR" in header:
            counts["NMR"] += 1
        else:
            counts["Other/unknown"] += 1

    except Exception:
        counts["Could not read"] += 1

    if i % 1000 == 0:
        print(f"Scanned {i}/{len(files)}")

print("\nRESULTS:")
for category, count in counts.items():
    print(f"{category}: {count}")
PY