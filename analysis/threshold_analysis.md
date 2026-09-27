# Threshold & Macro F0.5 Optimization Analysis

## 1. Decision Threshold Grid Sweep (Held-Out Validation, 20,000 S1 Entities)

| Decision Threshold | Macro $F_{0.5}$ | Precision | Recall | Singleton $F_{0.5}$ | Non-Singleton $F_{0.5}$ | Total Predictions |
|---|---|---|---|---|---|---|
| **0.80** | 79.48% | 88.49% | 64.06% | 87.96% | 78.98% | 45,344 |
| **0.82** | 79.55% | 88.67% | 63.94% | 88.94% | 79.00% | 45,007 |
| **0.84** | 79.65% | 88.89% | 63.81% | 90.37% | 79.01% | 44,660 |
| **0.85 (Validation Optimal)** | **79.66%** | **88.94%** | **63.74%** | **91.17%** | **78.97%** | **44,495** |
| **0.87** | 79.65% | 89.06% | 63.50% | 91.70% | 78.93% | 44,105 |
| **0.89** | 79.63% | 89.18% | 63.24% | 93.04% | 78.83% | 43,687 |
| **0.90** | 79.56% | 89.17% | 63.07% | 93.22% | 78.74% | 43,479 |
| **0.92 (Final Submission V1)** | **79.36%** | **89.14%** | **62.64%** | **94.20%** | **78.48%** | **42,937** |

---

## 2. Engineering & Competition Time Budget Insight
While threshold `0.85` achieved the highest held-out validation score (**79.66%**), model probabilities were calculated dynamically in-memory and not saved to disk during the original run. Re-extracting feature matrices across 1.73M test entities at threshold 0.85 required **~4.06 hours**, exceeding the competition submission deadline budget. Consequently, **V1 at threshold 0.92 (Leaderboard 0.758)** was safely retained as the final submission.
