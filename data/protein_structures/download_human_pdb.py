import requests

url = "https://search.rcsb.org/rcsbsearch/v2/query"

query = {
    "query": {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": "rcsb_entity_source_organism.taxonomy_lineage.name",
            "operator": "exact_match",
            "value": "Homo sapiens"
        }
    },
    "return_type": "entry",
    "request_options": {
        "return_all_hits": True
    }
}

response = requests.post(url, json=query)

if response.status_code != 200:
    print("Something went wrong:")
    print(response.text)
    exit()

data = response.json()

pdb_ids = [
    result["identifier"]
    for result in data["result_set"]
]

print(f"Found {len(pdb_ids)} human PDB entries")

with open("human_pdb_ids.txt", "w") as f:
    for pdb_id in pdb_ids:
        f.write(pdb_id + "\n")