import requests
import time

INPUT_FILE = "human_protein_entities_v2.txt"
OUTPUT_FILE = "dataset_entities.txt"

# Read protein entity IDs
with open(INPUT_FILE, "r") as f:
    entity_ids = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(entity_ids)} protein entities")

# RCSB GraphQL endpoint
url = "https://data.rcsb.org/graphql"

query = """
query($ids: [String!]!) {
    polymer_entities(entity_ids: $ids) {
        rcsb_id

        entity_poly {
            type
            pdbx_seq_one_letter_code_can
        }

        rcsb_polymer_entity {
            pdbx_description
        }

        rcsb_polymer_entity_container_identifiers {
            entry_id
        }
    }
}
"""

dataset = []

batch_size = 100

for start in range(0, len(entity_ids), batch_size):

    batch = entity_ids[start:start + batch_size]

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

    entities = data["data"]["polymer_entities"]

    for entity in entities:

        if entity is None:
            continue

        sequence = entity["entity_poly"]["pdbx_seq_one_letter_code_can"]

        # Remove whitespace from sequence
        sequence = "".join(sequence.split())

        # Keep proteins between 50 and 2000 amino acids
        if len(sequence) < 50:
            continue

        if len(sequence) > 2000:
            continue

        dataset.append({
            "entity_id": entity["rcsb_id"],
            "entry_id": entity[
                "rcsb_polymer_entity_container_identifiers"
            ]["entry_id"],
            "sequence": sequence
        })

    processed = min(start + batch_size, len(entity_ids))

    print(f"Processed {processed}/{len(entity_ids)}")

    time.sleep(0.1)


# Remove duplicate entity IDs
unique_dataset = {}

for item in dataset:
    unique_dataset[item["entity_id"]] = item

dataset = list(unique_dataset.values())


# Save
with open(OUTPUT_FILE, "w") as f:

    for item in dataset:

        f.write(
            f"{item['entity_id']}\t"
            f"{item['entry_id']}\t"
            f"{item['sequence']}\n"
        )


print()
print("Finished!")
print(f"Original entities: {len(entity_ids)}")
print(f"After sequence filtering: {len(dataset)}")
print(f"Saved to {OUTPUT_FILE}")