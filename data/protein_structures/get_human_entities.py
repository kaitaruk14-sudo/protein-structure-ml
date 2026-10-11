"""
Get every HUMAN protein chain (polymer entity) in the PDB, in one search call.

Replaces download_human_pdb.py + filter_human_pdb.py, which had a bug:
they grabbed every ENTRY containing any human chain, then kept ALL protein
chains in those entries -- so mouse antibodies, viral proteins, bacterial
partners etc. bound to a human protein were included as "human".

This asks RCSB for entities whose OWN source organism is Homo sapiens,
and only experimentally determined structures (no computed models).
"""
import requests

URL = "https://search.rcsb.org/rcsbsearch/v2/query"
OUTPUT_FILE = "human_protein_entities_v2.txt"

query = {
    "query": {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entity_source_organism.scientific_name",
                    "operator": "exact_match",
                    "value": "Homo sapiens",
                },
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "entity_poly.rcsb_entity_polymer_type",
                    "operator": "exact_match",
                    "value": "Protein",
                },
            },
        ],
    },
    "return_type": "polymer_entity",
    "request_options": {
        "return_all_hits": True,
        "results_content_type": ["experimental"],
    },
}

response = requests.post(URL, json=query, timeout=300)
response.raise_for_status()
ids = sorted({r["identifier"] for r in response.json()["result_set"]})

with open(OUTPUT_FILE, "w") as f:
    f.write("\n".join(ids) + "\n")

print(f"Found {len(ids)} human protein entities -> {OUTPUT_FILE}")

# Compare with the old (buggy) list if it's there
try:
    old = set(open("human_protein_entities.txt").read().split())
    print(f"Old list had {len(old)}; {len(old - set(ids))} of those were NOT human chains")
except FileNotFoundError:
    pass
