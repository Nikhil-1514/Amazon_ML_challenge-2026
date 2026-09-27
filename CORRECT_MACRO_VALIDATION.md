# CORRECT MACRO F0.5 VALIDATION REPORT

**Dataset:** Amazon ML Challenge 2026 — Business Entity Resolution  
**Evaluated Held-Out Entities:** `20,000` S1 entities (Rows 10,000 to 30,000 of `train_source1.tsv`)  
**Singletons (0 Matches):** `1,121` (`5.61%`)  
**Matched S1 Entities ($\ge 1$ Matches):** `18,879` (`94.39%`)  
**Evaluation Metric:** True Entity-Level **MACRO $F_{0.5}$** (Averaged across ALL held-out S1 entities)  

---

## 1. Executive Summary & Key Findings

### A. The Micro-vs-Macro Discrepancy Resolved
- **Previous Micro Pair $F_0.5$ Score:** `97.40%` (calculated on sampled candidate pairs, ignoring singletons and macro entity averaging).
- **True Macro $F_0.5$ Score at Original Threshold (0.92, Uncapped):** **`79.44%`**
- **Optimal Uncapped Threshold:** `0.85` with True Macro $F_0.5 = $ **`79.74%`**
- **Optimal Overall Strategy:** `Uncapped (Raw)` at Threshold `0.85` with True Macro $F_0.5 = $ **`79.74%`**

> [!IMPORTANT]
> **Answer to Key Question:**
> **YES.** True Macro $F_{0.5}$ validation evaluates the exact competition metric semantics and reveals that the real internal macro performance of the default threshold (0.92) is ~**`0.78 - 0.80`**, which directly mirrors the live leaderboard performance (`0.758` for V1 / `0.757` for V2). This confirms that Macro $F_{0.5}$ gives us a reliable, non-blind ground truth for model and post-processing optimization.

---

## 2. Evaluation Strategy Breakdown: Uncapped (Raw Model Predictions)

Evaluation of original model predictions across decision thresholds without post-processing caps:

| Threshold | Macro Prec (%) | Macro Rec (%) | **Macro F0.5 (%)** | F0.5 = 1 (%) | F0.5 = 0 (%) | Singleton F0.5 (%) | Non-Singleton F0.5 (%) | Avg Pred Matches / S1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `0.70` | 87.86% | 64.53% | **79.24%** | 28.33% | 9.04% | 84.12% | 78.95% | 2.329 |
| `0.75` | 88.27% | 64.36% | **79.45%** | 28.68% | 9.10% | 86.08% | 79.05% | 2.296 |
| `0.80` | 88.69% | 64.09% | **79.63%** | 28.74% | 9.14% | 88.14% | 79.12% | 2.261 |
| `0.85` **(Best Raw)** | 89.06% | 63.72% | **79.74%** | 28.73% | 9.23% | 90.90% | 79.07% | 2.221 |
| `0.90` | 89.30% | 63.09% | **79.65%** | 28.52% | 9.49% | 93.49% | 78.82% | 2.171 |
| `0.92` | 89.24% | 62.66% | **79.44%** | 28.34% | 9.72% | 94.38% | 78.55% | 2.145 |
| `0.95` | 89.00% | 61.56% | **78.82%** | 27.68% | 10.29% | 95.63% | 77.82% | 2.089 |
| `0.97` | 88.42% | 60.12% | **77.84%** | 26.66% | 11.05% | 96.79% | 76.71% | 2.025 |
| `0.99` | 86.04% | 55.75% | **74.48%** | 23.03% | 13.74% | 98.31% | 73.07% | 1.851 |

---

## 3. Post-Processing & Cap Strategy Comparison

Impact of match-count capping (Max 5, Max 3, Max 2, Max 1) across thresholds:


### Strategy: `Max 5`
| Threshold | Macro Prec (%) | Macro Rec (%) | **Macro F0.5 (%)** | F0.5 = 1 (%) | F0.5 = 0 (%) | Singleton F0.5 (%) | Non-Singleton F0.5 (%) | Avg Pred Matches / S1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `0.70` | 88.06% | 64.09% | **79.29%** | 27.36% | 9.04% | 84.12% | 79.00% | 2.275 |
| `0.75` | 88.42% | 63.93% | **79.46%** | 27.65% | 9.11% | 86.08% | 79.07% | 2.249 |
| `0.80` | 88.82% | 63.66% | **79.62%** | 27.71% | 9.14% | 88.14% | 79.12% | 2.218 |
| `0.85` | 89.15% | 63.31% | **79.71%** | 27.69% | 9.24% | 90.90% | 79.04% | 2.183 |
| `0.90` | 89.36% | 62.70% | **79.59%** | 27.48% | 9.49% | 93.49% | 78.77% | 2.137 |
| `0.92` | 89.29% | 62.28% | **79.38%** | 27.31% | 9.73% | 94.38% | 78.49% | 2.113 |
| `0.95` | 89.02% | 61.21% | **78.75%** | 26.71% | 10.29% | 95.63% | 77.75% | 2.061 |
| `0.97` | 88.44% | 59.80% | **77.76%** | 25.76% | 11.05% | 96.79% | 76.63% | 2.001 |
| `0.99` | 86.04% | 55.52% | **74.42%** | 22.37% | 13.74% | 98.31% | 73.00% | 1.834 |

### Strategy: `Max 3`
| Threshold | Macro Prec (%) | Macro Rec (%) | **Macro F0.5 (%)** | F0.5 = 1 (%) | F0.5 = 0 (%) | Singleton F0.5 (%) | Non-Singleton F0.5 (%) | Avg Pred Matches / S1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `0.70` | 88.83% | 59.08% | **78.14%** | 21.38% | 9.04% | 84.12% | 77.78% | 1.970 |
| `0.75` | 89.06% | 58.97% | **78.23%** | 21.55% | 9.11% | 86.08% | 77.76% | 1.953 |
| `0.80` | 89.34% | 58.80% | **78.32%** | 21.58% | 9.14% | 88.14% | 77.73% | 1.932 |
| `0.85` | 89.55% | 58.58% | **78.35%** | 21.65% | 9.24% | 90.90% | 77.60% | 1.909 |
| `0.90` | 89.62% | 58.13% | **78.18%** | 21.54% | 9.49% | 93.49% | 77.27% | 1.879 |
| `0.92` | 89.50% | 57.79% | **77.95%** | 21.44% | 9.73% | 94.38% | 76.98% | 1.861 |
| `0.95` | 89.16% | 56.93% | **77.33%** | 21.09% | 10.29% | 95.63% | 76.25% | 1.825 |
| `0.97` | 88.52% | 55.80% | **76.40%** | 20.45% | 11.05% | 96.79% | 75.19% | 1.781 |
| `0.99` | 86.07% | 52.28% | **73.26%** | 18.20% | 13.74% | 98.31% | 71.77% | 1.657 |

### Strategy: `Max 2`
| Threshold | Macro Prec (%) | Macro Rec (%) | **Macro F0.5 (%)** | F0.5 = 1 (%) | F0.5 = 0 (%) | Singleton F0.5 (%) | Non-Singleton F0.5 (%) | Avg Pred Matches / S1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `0.70` | 89.72% | 50.14% | **74.56%** | 14.90% | 9.04% | 84.12% | 73.99% | 1.551 |
| `0.75` | 89.80% | 50.09% | **74.57%** | 14.99% | 9.11% | 86.08% | 73.89% | 1.541 |
| `0.80` | 89.94% | 50.03% | **74.62%** | 15.07% | 9.15% | 88.14% | 73.81% | 1.529 |
| `0.85` | 90.01% | 49.95% | **74.59%** | 15.21% | 9.25% | 90.90% | 73.63% | 1.517 |
| `0.90` | 89.95% | 49.72% | **74.43%** | 15.23% | 9.50% | 93.49% | 73.29% | 1.499 |
| `0.92` | 89.78% | 49.50% | **74.22%** | 15.20% | 9.74% | 94.38% | 73.02% | 1.489 |
| `0.95` | 89.35% | 48.91% | **73.65%** | 15.02% | 10.30% | 95.63% | 72.35% | 1.466 |
| `0.97` | 88.66% | 48.15% | **72.84%** | 14.67% | 11.06% | 96.79% | 71.42% | 1.440 |
| `0.99` | 86.12% | 45.65% | **70.08%** | 13.48% | 13.74% | 98.31% | 68.40% | 1.359 |

### Strategy: `Max 1`
| Threshold | Macro Prec (%) | Macro Rec (%) | **Macro F0.5 (%)** | F0.5 = 1 (%) | F0.5 = 0 (%) | Singleton F0.5 (%) | Non-Singleton F0.5 (%) | Avg Pred Matches / S1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `0.70` | 90.75% | 32.41% | **62.11%** | 8.12% | 9.25% | 84.12% | 60.80% | 0.879 |
| `0.75` | 90.70% | 32.45% | **62.09%** | 8.21% | 9.30% | 86.08% | 60.67% | 0.875 |
| `0.80` | 90.67% | 32.51% | **62.10%** | 8.31% | 9.32% | 88.14% | 60.56% | 0.871 |
| `0.85` | 90.59% | 32.56% | **62.08%** | 8.43% | 9.41% | 90.90% | 60.37% | 0.867 |
| `0.90` | 90.36% | 32.56% | **61.96%** | 8.54% | 9.63% | 93.49% | 60.09% | 0.860 |
| `0.92` | 90.13% | 32.49% | **61.80%** | 8.57% | 9.87% | 94.38% | 59.86% | 0.856 |
| `0.95` | 89.60% | 32.31% | **61.42%** | 8.56% | 10.40% | 95.63% | 59.39% | 0.848 |
| `0.97` | 88.85% | 32.03% | **60.89%** | 8.54% | 11.15% | 96.79% | 58.75% | 0.839 |
| `0.99` | 86.19% | 31.04% | **59.00%** | 8.36% | 13.81% | 98.31% | 56.66% | 0.809 |

---

## 4. Predicted Match Count Distribution at Key Thresholds (Uncapped)

Distribution of predicted matches per S1 entity for held-out evaluation set:

| Threshold | 0 Matches | 1 Match | 2 Matches | 3-5 Matches | 6-10 Matches | >10 Matches | Total Evaluated S1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `0.70` | 2,414 | 4,144 | 5,072 | 7,736 | 617 | 17 | 20,000 |
| `0.75` | 2,494 | 4,191 | 5,085 | 7,634 | 582 | 14 | 20,000 |
| `0.80` | 2,575 | 4,262 | 5,114 | 7,489 | 548 | 12 | 20,000 |
| `0.85` | 2,670 | 4,323 | 5,160 | 7,333 | 507 | 7 | 20,000 |
| `0.90` | 2,797 | 4,422 | 5,191 | 7,120 | 466 | 4 | 20,000 |
| `0.92` | 2,882 | 4,461 | 5,203 | 7,005 | 447 | 2 | 20,000 |
| `0.95` | 3,037 | 4,609 | 5,178 | 6,779 | 395 | 2 | 20,000 |
| `0.97` | 3,228 | 4,754 | 5,183 | 6,479 | 354 | 2 | 20,000 |
| `0.99` | 3,813 | 5,188 | 5,050 | 5,690 | 259 | 0 | 20,000 |

---

## 5. Detailed Metric Definitions & Validation Protocol

1. **Held-Out Isolation**: Evaluation S1 entities were selected strictly from rows 10,001 to 30,000 of `train_source1.tsv`. The LightGBM model was trained strictly on rows 1 to 10,000, ensuring 0 data leakage.
2. **Exact Competition Metric Formula**:  
   - For each S1 entity $i$:

     - If $|GT_i| == 0$ and $|Pred_i| == 0 \implies P_i = 1, R_i = 1, F_{0.5}(i) = 1.0$.

     - If $|GT_i| == 0$ and $|Pred_i| > 0 \implies P_i = 0, R_i = 0, F_{0.5}(i) = 0.0$.

     - If $|GT_i| > 0$ and $|Pred_i| == 0 \implies P_i = 0, R_i = 0, F_{0.5}(i) = 0.0$.

     - If $|GT_i| > 0$ and $|Pred_i| > 0 \implies P_i = \frac{TP_i}{|Pred_i|}, R_i = \frac{TP_i}{|GT_i|}, F_{0.5}(i) = \frac{1.25 P_i R_i}{0.25 P_i + R_i}$.

   - Macro $F_{0.5} = \frac{1}{N} \sum_{i=1}^N F_{0.5}(i)$.