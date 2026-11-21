| Method | Long-range top-L/5 precision (test) | 95% CI | Difference vs sequence separation (95% CI) |
|---|---|---|---|
| Random | 1.67% | 1.59% – 1.76% | -1.93 pp (-2.10 to -1.76) |
| Sequence separation | 3.60% | 3.44% – 3.76% |  |
| Logistic regression | 8.41% | 8.17% – 8.66% | +4.81 pp (+4.59 to +5.05) |
| Logistic regression, rich features | 9.15% | 8.87% – 9.42% | +5.54 pp (+5.26 to +5.82) |
| Gradient-boosted trees, rich features | 8.48% | 8.22% – 8.75% | +4.88 pp (+4.61 to +5.15) |
| Neural distogram model (small) | 12.69% | 12.37% – 13.01% | +9.09 pp (+8.76 to +9.42) |
| Neural distogram model (long) | 15.38% | 14.95% – 15.81% | +11.78 pp (+11.34 to +12.21) |
| Neural distogram model (medium) | 17.14% | 16.69% – 17.58% | +13.54 pp (+13.09 to +13.99) |
| Neural distogram model (medium long) | 21.53% | 20.89% – 22.12% | +17.93 pp (+17.31 to +18.51) |
| Neural distogram model (large) | 21.71% | 21.12% – 22.28% | +18.11 pp (+17.51 to +18.69) |

Evaluated on 3,714 test chains (length ≤ 512). Contact = Cβ–Cβ < 8 Å; long-range = |i−j| ≥ 24. CIs: 2,000 bootstrap resamples of chains; pp = percentage points.

Paired differences (A − B), 95% CI from 2,000 bootstrap resamples of chains:

| A − B | Difference | Chains A better / worse / equal |
|---|---|---|
| Neural distogram model (large) − Neural distogram model (medium long) | +0.18 pp (-0.12 to +0.48) | 1,566 / 1,367 / 781 |
| Neural distogram model (large) − Logistic regression, rich features | +12.56 pp (+12.01 to +13.12) | 2,624 / 613 / 477 |
| Neural distogram model (large) − Gradient-boosted trees, rich features | +13.23 pp (+12.67 to +13.78) | 2,698 / 560 / 456 |
| Neural distogram model (large) − Logistic regression | +13.30 pp (+12.72 to +13.88) | 2,607 / 650 / 457 |
| Neural distogram model (medium long) − Logistic regression, rich features | +12.38 pp (+11.82 to +12.92) | 2,606 / 614 / 494 |
| Logistic regression, rich features − Logistic regression | +0.73 pp (+0.52 to +0.96) | 1,458 / 1,289 / 967 |
