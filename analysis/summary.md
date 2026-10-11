# Dataset summary

## Pipeline

| Stage | Count |
|---|---|
| PDB entries with a human chain | 83,569 |
| Human protein chains (experimental) | 155,985 |
| Length 50–2000 residues | 145,993 |
| Unique sequences | 46,613 |
| With 3D backbone downloaded | 46,607 |
| Usable for training (≥50% observed) | 44,332 |
| Sequence clusters (30% identity) | 9,693 |

## Train / validation / test (usable chains, split by 30%-identity cluster)

| Split | Chains | Clusters |
|---|---|---|
| train | 35,475 | 6,520 |
| val | 4,486 | 1,098 |
| test | 4,371 | 1,127 |

## Key numbers

- The most-repeated sequence appears in **938** PDB chains; 65% of unique sequences appear only once, the rest are repeated.
- Largest cluster: 2,831 sequences. The top 1% of clusters hold 36% of all unique sequences; 50% of clusters are a single sequence.
- Chain length: median 225, 90th percentile 517; 79% of chains are longer than the 128-residue training crop.
- Resolved coordinates: median 95% of a chain; 23% of chains fully observed; 4.9% under 50% (excluded).
- Contact rates (Cβ < 8 Å, from 3,000 sampled chains): short-range 5.3%, medium-range 3.8%, long-range 0.83%.
- 26% of residue pairs (|i−j| ≥ 6) are within the 22 Å predicted range.
- Example map: test chain 2VPK_1.

## Limitations to mention
- Human proteins only; no MSAs (single-sequence input).
- Experimental method and resolution not yet recorded per chain (X-ray, cryo-EM and NMR are mixed).
- Virtual Cβ is computed from backbone N/CA/C rather than read from the files.
