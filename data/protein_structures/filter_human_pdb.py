import requests
import time

INPUT_FILE = "human_pdb_ids.txt"
OUTPUT_FILE = "human_protein_entities.txt"

# Read PDB IDs
with open(INPUT_FILE, "r") as f:
    pdb_ids = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(pdb_ids)} PDB entries")

# RCSB batch endpoint
url = "https://data.rcsb.org/graphql"

# We'll process PDB IDs in batches
batch_size = 100

protein_entities = []

query = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    polymer_entities {
      rcsb_id
      entity_poly {
        type
      }
    }
  }
}
"""

for start in range(0, len(pdb_ids), batch_size):

    batch = pdb_ids[start:start + batch_size]

    response = requests.post(
        url,
        json={
            "query": query,
            "variables": {
                "ids": batch
            }
        }
    )

    if response.status_code != 200:
        print("Request failed:")
        print(response.text)
        continue

    data = response.json()

    if "errors" in data:
        print("GraphQL error:")
        print(data["errors"])
        continue

    entries = data["data"]["entries"]

    for entry in entries:
        if entry is None:
            continue

        for entity in entry["polymer_entities"] or []:
            if entity is None:
                continue

            if entity["entity_poly"]["type"] == "polypeptide(L)":
                protein_entities.append(entity["rcsb_id"])

    processed = min(start + batch_size, len(pdb_ids))
    print(f"Processed {processed}/{len(pdb_ids)}")

    # Be polite to the RCSB server
    time.sleep(0.1)

# Remove duplicates
protein_entities = sorted(set(protein_entities))

with open(OUTPUT_FILE, "w") as f:
    for entity_id in protein_entities:
        f.write(entity_id + "\n")

print()
print("Finished!")
print(f"Found {len(protein_entities)} human protein entities")
print(f"Saved to {OUTPUT_FILE}")