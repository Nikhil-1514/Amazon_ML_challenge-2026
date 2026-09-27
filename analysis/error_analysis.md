# Error Analysis & Post-Processing Lessons

## 1. Post-Processing Rule Experiments

To improve the V1 prediction set without re-running 4-hour feature extraction, several post-processing transformation rules were tested on held-out validation data:

| Rule Tested | Validation Macro $F_{0.5}$ | $\Delta$ vs V1 Baseline | Outcome |
|---|---|---|---|
| **V1 Baseline (th=0.92, uncapped)** | **79.36%** | **0.00%** | **BASELINE (RETAINED)** |
| **V2 Top-5 Match Cap** | 79.30% | -0.06% | Rejected (LB drop from 0.758 to 0.757) |
| **Top-4 Match Cap** | 79.01% | -0.35% | Rejected |
| **Top-3 Match Cap** | 77.90% | -1.47% | Rejected |
| **Top-1 Match Cap** | 61.74% | -17.62% | Rejected |
| **Country-Specific Caps (France / India)** | 77.96% - 79.36% | -1.40% to 0.00% | Rejected |
| **Name Token / Core-Name Filter** | 78.99% | -0.37% | Rejected |

---

## 2. Key Takeaway & Lesson
High-confidence predictions at threshold `0.92` exhibit **89.14% precision**. Truncating candidate matches beyond Top-5 removes 525 True Positives while eliminating only 105 False Positives because true 1-to-many business branch relationships exist in ground truth. Preserving uncapped high-precision matches yielded the best performance.
