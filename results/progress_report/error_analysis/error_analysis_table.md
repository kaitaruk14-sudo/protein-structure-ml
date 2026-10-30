| Group | Chains (clusters) | Sequence separation | Logistic regression | Neural | Neural − logreg (paired 95% CI) | Verdict |
|---|---|---|---|---|---|---|
| All chains | 3,992 (960) | 3.48% (3.33–3.62) | 8.77% (8.52–9.02) | 13.53% (13.18–13.89) | +4.76 pp (+4.42 to +5.10) | neural better |
| Length 50–99 | 614 (217) | 4.31% (3.88–4.74) | 13.03% (11.93–14.04) | 15.24% (14.10–16.46) | +2.21 pp (+1.27 to +3.19) | neural better |
| Length 100–199 | 1,397 (357) | 3.40% (3.21–3.58) | 8.27% (7.94–8.64) | 16.63% (16.01–17.29) | +8.36 pp (+7.71 to +9.01) | neural better |
| Length 200–299 | 701 (210) | 3.71% (3.30–4.15) | 9.32% (8.77–9.88) | 13.57% (12.92–14.32) | +4.25 pp (+3.54 to +5.02) | neural better |
| Length 300–512 | 1,280 (262) | 3.04% (2.80–3.30) | 6.97% (6.65–7.33) | 9.31% (8.98–9.65) | +2.33 pp (+1.98 to +2.70) | neural better |
| Family size 1 (singleton) | 430 (430) | 3.09% (2.65–3.58) | 8.48% (7.60–9.40) | 13.42% (12.37–14.51) | +4.94 pp (+3.84 to +5.98) | neural better |
| Family size 2–5 | 1,067 (369) | 3.50% (3.24–3.78) | 8.89% (8.37–9.43) | 14.12% (13.48–14.80) | +5.23 pp (+4.59 to +5.83) | neural better |
| Family size 6–20 | 1,148 (129) | 4.02% (3.68–4.38) | 8.60% (8.12–9.08) | 14.21% (13.58–14.83) | +5.61 pp (+4.98 to +6.24) | neural better |
| Family size 21+ | 1,347 (32) | 3.13% (2.95–3.31) | 8.91% (8.53–9.30) | 12.52% (11.93–13.09) | +3.61 pp (+3.02 to +4.19) | neural better |

Long-range top-L/5 precision on validation chains, mean with 95% CI from 2,000 bootstrap resamples of chains; Δ is the paired per-chain difference. Family size = chains in the same MMseqs2 cluster (30% identity) in backbone_index.tsv; all clusters sit entirely inside the validation split, so no validation family appears in training.

Robustness: overall Δ with a cluster bootstrap (resampling 960 clusters): +4.76 pp (+3.58 to +6.08).
