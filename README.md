# Amazon ML Challenge 2026 — Business Entity Resolution

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)

Scalable, high-precision Business Entity Resolution pipeline developed for the **Amazon ML Challenge 2026**. The system matches 1.73M reference entities against ~10M records across multiple heterogeneous data sources under a precision-heavy entity-level **Macro $F_{0.5}$** metric.

---

## 1. Problem Description

Given three distinct datasets (**Source 1** as the reference set, **Source 2**, and **Source 3**), the task is to resolve every reference entity in Source 1 to its corresponding matching entities in Source 2 and Source 3.

Key Challenges:
- **Noisy, Heterogeneous Text:** Business names contain legal suffixes (Inc, Corp, Pvt Ltd, SAS), abbreviations, and noise. Addresses suffer from spelling variations and differing locality formats.
- **1-to-Many Relationships:** A single reference entity in Source 1 may match zero, one, or multiple branch/registry records across Source 2 and Source 3.
- **Extreme Scale:** Matching 1.73M test entities against 10M records requires candidate search space reduction from $\sim 1.73 \times 10^{13}$ possible pairs down to a manageable candidate set without losing true matches.
- **Precision-Heavy Macro Metric:** Evaluation is scored via entity-level **Macro $F_{0.5}$**, heavily weighting precision while including singletons (entities with 0 true matches).

---

## 2. Dataset Scale

*Datasets profiled across 24,229,173 total records:*

| Subset | Source 1 (Reference) | Source 2 | Source 3 | Total Records |
|---|---|---|---|---|
| **Training Set** | 2,206,821 | 5,034,616 | 5,285,603 | **12,527,040** |
| **Test Set** | 1,732,544 | 4,887,273 | 5,082,316 | **11,702,133** |
| **Total Profiled** | **3,939,365** | **9,921,889** | **10,367,919** | **24,229,173** |

*Note: Raw competition datasets are not hosted in this public portfolio repository due to size restrictions.*

---

## 3. System Architecture

```
Raw Sources (S1, S2, S3)
   ↓
Text Normalization & Legal Suffix Stripping
   ↓
Layered Composite Candidate Blocking
   ↓
47-Feature Pairwise Vectorization
   ↓
LightGBM Gradient Boosted Matching
   ↓
Precision-Calibrated Thresholding (th=0.92)
   ↓
Entity-Level Match Formatting
   ↓
Official Submission Validation
```

---

## 4. Candidate Generation (Blocking)

To overcome quadratic pair complexity, a layered composite blocking strategy (**`Var5_Layered_Composite_Optimal`**) was deployed using multi-channel inverted indexes:

1. **Exact Legal Core-Name Indexing:** Strips suffixes (`Inc`, `Corp`, `LLC`, `Pvt Ltd`) to match core names within country.
2. **Postal Code (PIN) Indexing:** Matches 6-digit postal codes within country.
3. **Frequent Token + Street Numeric Composite Indexing:** Matches first name token paired with extracted street numbers.

**Blocking Performance (Evaluated on Training Data):**
- **Pair Recall:** `97.68%`
- **S1 Entity Coverage:** `93.25%`
- **Mean Candidates per S1:** `9,996.4`
- **Median Candidates per S1:** `9,166`
- **95th Percentile (P95):** `22,036`

---

## 5. Machine Learning & Feature Engineering

Pairs pass through a 47-feature vectorizer computing string and token similarities:
- **Name Features (15):** Jaro-Winkler, Levenshtein, 3-gram/4-gram Jaccard, token containment, set similarity.
- **Address Features (12):** Token Jaccard, numeric street overlap, locality token overlap.
- **Postal & Country Features (5):** Exact postal match, 3-digit prefix match, country match mask.
- **Cross-Field Interactions & Blocking Evidence (15):** Cross-similarity products, channel indicator masks.

**Models Evaluated:**
- Logistic Regression
- HistGradientBoosting
- Random Forest
- XGBoost
- **LightGBM (Selected Final Classifier)**

---

## 6. Important Evaluation Lesson: Micro vs Macro $F_{0.5}$

During early development, pair-level micro evaluation achieved **~97% Micro $F_{0.5}$**. However, the competition metric is **Entity-Level Macro $F_{0.5}$**, calculated by averaging $F_{0.5}$ across all 1.73M individual S1 entities (including singletons).

True Macro $F_{0.5}$ validation revealed that pairwise accuracy overestimates entity-level performance. Building a competition-compliant Macro $F_{0.5}$ validation engine aligned internal validation (79.36%) directly with live leaderboard feedback (0.758).

---

## 7. Final Submitted System & Results

### Final Submission (V1):
- **Classifier:** LightGBM
- **Decision Threshold:** `0.92` (Uncapped)
- **Live Leaderboard Score:** **`0.758`**
- **Test Match Output Breakdown:**
  - **S1 Entities with Matches:** `1,484,491`
  - **Singletons (0 Matches):** `248,053`
  - **Total Matches Generated:** `5,220,694`
  - **Source 2 Matches:** `2,471,232`
  - **Source 3 Matches:** `2,749,462`

### V2 Cap Experiment Lesson:
In V2, a Top-5 match cap per S1 was evaluated (`3,871,388` total matches). The leaderboard score dropped from **`0.758` down to `0.757`**. Validation confirmed that capping predictions removed **525 True Positives** while eliminating only **105 False Positives** because legitimate 1-to-many branch relationships exist in ground truth.

### V3/V4 Engineering Lessons:
- **Threshold 0.85 Optimization:** Grid sweeps identified threshold `0.85` as optimal on held-out validation (**79.66% Macro $F_{0.5}$**). However, generating full test feature extraction at threshold 0.85 required **~4.06 hours**, exceeding deadline limits.
- **Deterministic Post-Processing (V4):** Post-processing filtering rules degraded recall faster than precision. V1 (uncapped at threshold 0.92) was retained as the proven final submission.

---

## 8. Engineering & Implementation Highlights

- **Scale Management:** Efficient streaming and memory-cached indexing of 24M+ records in Python.
- **Validation Discipline:** 100% strict adherence to official `validate_submission.py` checks.
- **Zero Data Leakage:** Isolated 20,000 held-out entity validation split.

---

## 9. Disclaimer

*This repository is intended to demonstrate solution methodology and code architecture. Competition datasets and raw output databases are omitted due to storage and data usage policies. Synthetic example inputs are provided in `examples/sample_output.tsv`.*
