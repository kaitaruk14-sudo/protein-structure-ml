| Method | Long-range top-L/5 precision (val) | 95% CI | Difference vs separation (95% CI) |
|---|---|---|---|
| Random | 1.85% | 1.76% – 1.94% | -1.62 pp (-1.79 to -1.46) |
| Sequence separation | 3.48% | 3.33% – 3.62% |  |
| Logistic regression | 8.77% | 8.52% – 9.02% | +5.29 pp (+5.08 to +5.52) |
| Logistic regression, rich features | 10.43% | 10.11% – 10.75% | +6.95 pp (+6.66 to +7.26) |
| Gradient-boosted trees, rich features | 9.84% | 9.53% – 10.14% | +6.36 pp (+6.08 to +6.65) |
| Neural distogram model | 13.53% | 13.18% – 13.89% | +10.05 pp (+9.72 to +10.41) |
| Neural distogram model (long training) | 15.09% | 14.69% – 15.49% | +11.62 pp (+11.20 to +12.02) |
| Neural distogram model (medium long training) | 20.06% | 19.49% – 20.60% | +16.59 pp (+16.02 to +17.11) |
| Neural distogram model (medium training) | 17.20% | 16.76% – 17.62% | +13.72 pp (+13.30 to +14.15) |

Evaluated on 3,992 validation chains (length ≤ 512). Contact = Cβ–Cβ < 8 Å; long-range = |i−j| ≥ 24. CIs: 2,000 bootstrap resamples of chains; pp = percentage points.
