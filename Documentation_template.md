# ML Challenge 2026: Business Entity Resolution Solution Documentation

**Team Name:** Antigravity AI  
**Submission Date:** September 27, 2026  

---

## 1. Executive Summary
We present a high-precision, scalable dual-stage pipeline for multi-source Business Entity Resolution under the competition's exact Macro $F_{0.5}$ metric. The architecture combines high-coverage blocking channels (exact legal core-name indexing, postal code lookup, and numeric-token composites) with a 47-dimensional gradient-boosted decision model (LightGBM) calibrated specifically to optimize Macro $F_{0.5}$ across singletons and multi-branch entities.

---

## 2. Methodology

### 2.1 Problem Analysis
* **Noise Patterns & Country-Specific Variations:** Entity records from US, India, and France display extreme address string heterogeneity, varying legal suffix conventions (Inc., Corp., Pvt. Ltd., SAS), and regional postal formats.
* **1-to-Many Multi-Branch Matches:** Ground truth analysis revealed that business entities naturally form 1-to-many relationships across Source 2 and Source 3. Truncating matches prematurely degrades macro recall severely.

### 2.2 Solution Strategy
**Approach Type:** High-Precision Blocking Channels + 47-Feature LightGBM Entity Matcher  
**Core Innovation:** Challenge-Compliant Entity-Level Macro $F_{0.5}$ Validation Engine and multi-channel candidate generation guaranteeing high candidate recall with bounded pair complexity.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** Exact legal core-name indexing, country-aware postal PIN code matching, and frequent token + numeric composite indexing.
- **Candidate pairs generated:** ~25 million candidate pairs for 1,732,544 test S1 entities.
- **Recall Guarantee:** Multi-channel blocking channels cover over 98% of true matches in validation ground truth.

---

## 4. Matching Model

**Features Used (47 Features Total):**
- **Name Features (15):** Exact name match, exact legal core-name match, Jaro-Winkler similarity, Levenshtein character distance, 3-gram & 4-gram character Jaccard, token Jaccard, token containment, token set similarity, relative length difference.
- **Address Features (12):** Token Jaccard, token overlap coefficient, 3-gram & 4-gram character Jaccard, Jaro-Winkler similarity, Levenshtein distance, address numeric overlap count, locality token overlap.
- **Postal Features (4):** Exact postal match, 3-digit prefix match, availability mask, length equality.
- **Country Feature (1):** Exact country match (US, India, France, Other).
- **Cross-Field Interactions (6):** Name $\times$ Address exact, high-confidence cross-mask, Name JW $\times$ Address JW product, sum, Postal $\times$ Name JW, Address numeric $\times$ Name JW.
- **Blocking Evidence Features (7):** Channel indicator masks and total channel count.
- **Source Identifiers (2):** S2 / S3 target indicator flags.

**Model Type:** LightGBM Gradient Boosted Decision Trees  
**Threshold Selection Method:** Direct Macro $F_{0.5}$ Grid Optimization on 20,000 held-out validation entities.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score (Validation):** **79.36%** (Evaluated across 20,000 held-out S1 entities including singletons).
- **Live Leaderboard Score:** **0.758** (Primary Submission V1).
- **False Positive Analysis:** Unrestricted candidate additions without model scoring inflate false positives due to generic business words (e.g. "Hotel", "Store"). High-threshold LightGBM scoring maintains 89.14% precision.

---

## 6. Conclusion
The dual-stage blocking + 47-feature LightGBM architecture delivers robust, scalable entity resolution across 1.73M entities in full compliance with competition rules and validation semantics.

---

## Appendix

### A. Code Artefacts
* All source code is packaged under `code/business_entity_resolution/src/` with a standard `requirements.txt` and `README.md`.
* Entry points: `src/train_matcher.py` (training), `src/fast_v3_generator.py` (inference), `utils/validate_submission.py` (validator).
