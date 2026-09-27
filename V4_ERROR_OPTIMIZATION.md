# V4 ERROR OPTIMIZATION & POST-PROCESSING DIAGNOSTIC REPORT

**Dataset:** Amazon ML Challenge 2026 — Business Entity Resolution  
**Evaluation Protocol:** True Competition Macro $F_{0.5}$ (Evaluated on 20,000 Genuinely Held-Out S1 Entities, Rows 10,001–30,000 of `train_source1.tsv`)  
**Decision Deadline Context:** ~2 Hours Remaining — Error-Driven V1 vs V2 Optimization Inquiry  

---

## 1. Executive Summary & Recommendation

> [!IMPORTANT]
> **FINAL RECOMMENDATION:**  
> **NO DEFENSIBLE V4 FOUND — KEEP V1 (Leaderboard Score: 0.758).**

* **Key Finding:** Rigorous empirical evaluation on held-out training data proves that **V1 (Uncapped at Threshold 0.92)** achieves the highest Macro $F_{0.5}$ score (**79.36%**).
* **Why V2 Scored Lower on Leaderboard (0.757 vs 0.758):** V2 applied a Top-5 match cap. On held-out training evaluation, capping at Top-5 removed **525 True Positives** while removing only **105 False Positives**, causing Macro $F_{0.5}$ to drop from **79.36% to 79.30%**.
* **Rule Testing Result:** Every attempted post-processing filter (Top-N capping, France/India specific capping, name token overlap filtering) destroyed true positive matches faster than false positives, resulting in lower held-out Macro $F_{0.5}$.
* **Submission Safety:** Do NOT submit a truncated V4. The safest and highest-scoring available file is **V1 (`output/matching_results.tsv`, LB 0.758)**.

---

## 2. Full Test Set Analysis: V1 vs V2 (1,732,544 S1 Entities)

| Metric | V1 (Uncapped, th=0.92) | V2 (Top-5 Capped, th=0.92) | Difference / Delta |
|---|---|---|---|
| **Total Predicted Matches** | **5,220,694** | **3,871,388** | -1,349,306 (-25.8%) |
| **Common Matches** | 3,871,388 | 3,871,388 | — |
| **V1-Only Matches (Truncated)** | 1,349,306 | 0 | 1,349,306 |
| **S1 Entities with V1 $\neq$ V2** | 101,336 (5.85%) | 101,336 (5.85%) | — |
| **S1 Entities with 0 Matches** | 248,053 (14.3%) | 248,053 (14.3%) | 0 |

### Match Count Distribution per S1 Entity (V1 Test Set)
* **0 matches:** 248,053 S1 entities (14.3%)
* **1 match:** 367,591 S1 entities (21.2%)
* **2 matches:** 415,349 S1 entities (24.0%)
* **3 matches:** 322,749 S1 entities (18.6%)
* **4 matches:** 189,158 S1 entities (10.9%)
* **5 matches:** 88,308 S1 entities (5.1%)
* **6–10 matches:** 62,331 S1 entities (3.6%) — *Truncated by V2*
* **11–20 matches:** 3,803 S1 entities (0.2%) — *Truncated by V2*
* **>20 matches:** 424 S1 entities (0.02%) — *Truncated by V2*

### Country Breakdown (V1 vs V2 Test Set)
* **France (FR):** V1 = **2,141,517** matches across 259,452 S1 | V2 = **840,206** matches (*1,301,311 matches removed by Top-5 cap*)
* **India (IN):** V1 = **1,535,789** matches across 809,986 S1 | V2 = **1,510,471** matches (*25,318 matches removed by Top-5 cap*)
* **United States (US):** V1 = **1,543,388** matches across 663,106 S1 | V2 = **1,520,711** matches (*22,677 matches removed by Top-5 cap*)

---

## 3. Held-Out Training Evaluation: Rule Experiments (20,000 S1 Entities)

All transformation rules were tested on **20,000 held-out training S1 entities** using the **exact competition Macro $F_{0.5}$ metric** (including singletons):

| Strategy / Rule | Macro $F_{0.5}$ | $\Delta$ vs V1 | Precision | Recall | Singleton $F_{0.5}$ | Non-Singleton $F_{0.5}$ | Total Preds | TP | FP | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| **V1 Baseline (Uncapped, th=0.92)** | **79.36%** | **0.00%** | **89.14%** | **62.64%** | **94.20%** | **78.48%** | **42,937** | **42,043** | **894** | **BASELINE** |
| **V2 Baseline (Top-5 Cap, th=0.92)** | 79.30% | -0.06% | 89.19% | 62.26% | 94.20% | 78.42% | 42,307 | 41,518 | 789 | Rejected |
| **Rule 1: Cap at 4** | 79.01% | -0.35% | 89.26% | 61.13% | 94.20% | 78.11% | 40,877 | 40,178 | 699 | Rejected |
| **Rule 2: Cap at 3** | 77.90% | -1.47% | 89.43% | 57.77% | 94.20% | 76.93% | 37,258 | 36,712 | 546 | Rejected |
| **Rule 3: Cap at 2** | 74.15% | -5.21% | 89.70% | 49.46% | 94.20% | 72.96% | 29,785 | 29,435 | 350 | Rejected |
| **Rule 4: Cap at 1** | 61.74% | -17.62% | 90.06% | 32.45% | 94.20% | 59.82% | 17,118 | 16,957 | 161 | Rejected |
| **Rule 5: France Cap at 2** | 79.36% | +0.00% | 89.14% | 62.64% | 94.20% | 78.48% | 42,937 | 42,043 | 894 | No Change |
| **Rule 6: India Cap at 2** | 77.96% | -1.40% | 89.51% | 58.81% | 94.20% | 76.99% | 38,863 | 38,348 | 515 | Rejected |
| **Rule 7: Name Token / Core Name Filter** | 78.99% | -0.37% | 89.17% | 61.74% | 94.20% | 78.09% | 42,184 | 41,327 | 857 | Rejected |

---

## 4. Root Cause Analysis: Why Post-Processing Rule Filtering Fails

1. **High Precision of High-Threshold Predictions:** At threshold `0.92`, the model's predictions have **89.14% precision**. Out of 42,937 predictions on held-out data, **42,043 are true matches (TP)** and only **894 are false matches (FP)**.
2. **True 1-to-Many Multi-Branch Matches:** In ground truth, business entities (especially multi-location chains and registries) naturally pair with multiple S2/S3 records. Truncating candidates beyond Top-5 removes 525 valid TPs while eliminating only 105 FPs.
3. **Recall Degradation Dominates Macro $F_{0.5}$:** Macro $F_{0.5}$ weights precision higher than recall ($1.25 \times P \times R / (0.25 P + R)$), but dropping recall from **62.64% down to 32.45%** (as in Cap-1) severely penalizes the overall macro score.

---

## 5. Submission Recommendation

> [!CAUTION]
> **NO DEFENSIBLE V4 FOUND — KEEP V1.**

* **Primary Submission File:** `output/matching_results.tsv` (**V1**, Leaderboard **0.758**).
* **Do NOT modify or replace `output/matching_results.tsv`**.
* **Do NOT create a truncated V4 TSV**, as evidence shows it lowers true Macro $F_{0.5}$.
