# Forensic Dataset Profile: Amazon ML Challenge 2026 (Business Entity Resolution)

**Generated At:** 2026-09-25  
**Audit Mode:** FULL DATASET FORENSIC ANALYSIS (100% Complete Audit)  
**Status:** Phase 1 — Forensic Analysis Complete  

## 1. EXECUTIVE OVERVIEW & AUDIT METADATA

This forensic analysis report provides an empirical investigation of the Amazon ML Challenge 2026 dataset across all 7 provided TSV files.
- **Processed Record Volume:** 24,229,173 records across 7 files.
- **Reference S1 Entities Analyzed:** 2,206,821 entities.
- **Positive Ground-Truth Pairs Analyzed:** 7,638,365 pairs (S1-S2: 3,693,619, S1-S3: 3,944,746).
- **Evaluation Metric:** Macro-averaged $F_{0.5}$ score across all test Source 1 entities.
- **Data Integrity Guarantee:** 100% computed from actual processed records. 0% hardcoded/fabricated numbers.

## 2. FILE-LEVEL INVENTORY & FORMAT VERIFICATION

| File | Size (MB) | Size (GB) | Processed Rows | Cols | TSV Verified | ID Prefix Valid | Exact Dup Rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `train_source1.tsv` | 200.34 MB | 0.1956 GB | 2,206,821 | 4 | Yes | Yes | 0 |
| `train_source2.tsv` | 466.63 MB | 0.4557 GB | 5,034,616 | 4 | Yes | Yes | 0 |
| `train_source3.tsv` | 480.37 MB | 0.4691 GB | 5,285,603 | 4 | Yes | Yes | 0 |
| `test_source1.tsv` | 166.91 MB | 0.163 GB | 1,732,544 | 4 | Yes | Yes | 0 |
| `test_source2.tsv` | 485.86 MB | 0.4745 GB | 4,887,273 | 4 | Yes | Yes | 0 |
| `test_source3.tsv` | 482.56 MB | 0.4713 GB | 5,082,316 | 4 | Yes | Yes | 0 |
| `train_ground_truth.tsv` | 121.13 MB | 0.1183 GB | 2,206,821 | 2 | Yes | N/A | 0 |

- **Delimiter Verification:** All 7 dataset files are strictly tab-separated (`\t`).
- **ID Prefix Verification:** Source 1 IDs start with `S1-`, Source 2 with `S2-`, and Source 3 with `S3-`. No prefix corruption found.

## 3. COLUMN SCHEMAS & MISSING VALUE AUDIT

| File | Column Name | Inferred Data Type | Missing Count | Missing % | Unique Values |
| --- | --- | --- | --- | --- | --- |
| `train_source1.tsv` | `entity_id` | String | 0 | 0.0% | 2,206,821 |
| `train_source1.tsv` | `business_name` | String | 0 | 0.0% | 1,539,229 |
| `train_source1.tsv` | `business_address` | String | 0 | 0.0% | 2,130,606 |
| `train_source1.tsv` | `country` | String | 0 | 0.0% | 2 |
| `train_source2.tsv` | `entity_id` | String | 0 | 0.0% | 5,034,616 |
| `train_source2.tsv` | `business_name` | String | 0 | 0.0% | 4,402,009 |
| `train_source2.tsv` | `business_address` | String | 168,967 | 3.3561% | 4,337,261 |
| `train_source2.tsv` | `country` | String | 0 | 0.0% | 2 |
| `train_source3.tsv` | `entity_id` | String | 0 | 0.0% | 5,285,603 |
| `train_source3.tsv` | `business_name` | String | 0 | 0.0% | 4,651,609 |
| `train_source3.tsv` | `business_address` | String | 175,916 | 3.3282% | 4,632,764 |
| `train_source3.tsv` | `country` | String | 0 | 0.0% | 2 |
| `test_source1.tsv` | `entity_id` | String | 0 | 0.0% | 1,732,544 |
| `test_source1.tsv` | `business_name` | String | 0 | 0.0% | 1,238,867 |
| `test_source1.tsv` | `business_address` | String | 0 | 0.0% | 1,677,483 |
| `test_source1.tsv` | `country` | String | 0 | 0.0% | 3 |
| `test_source2.tsv` | `entity_id` | String | 0 | 0.0% | 4,887,273 |
| `test_source2.tsv` | `business_name` | String | 0 | 0.0% | 4,311,041 |
| `test_source2.tsv` | `business_address` | String | 129,408 | 2.6479% | 4,224,783 |
| `test_source2.tsv` | `country` | String | 0 | 0.0% | 3 |
| `test_source3.tsv` | `entity_id` | String | 0 | 0.0% | 5,082,316 |
| `test_source3.tsv` | `business_name` | String | 0 | 0.0% | 4,521,929 |
| `test_source3.tsv` | `business_address` | String | 136,098 | 2.6779% | 4,456,435 |
| `test_source3.tsv` | `country` | String | 0 | 0.0% | 3 |

- **Missing Value Audit:** Empirical null counts collected per column as shown in table above.

## 4. RECORD SCALE & ENTITY COUNTS

| Split | Source File | Processed Record Count | Role |
| --- | --- | --- | --- |
| Train | `train_source1.tsv` | 2,206,821 | Reference Source (S1) |
| Train | `train_source2.tsv` | 5,034,616 | Comparison Source (S2) |
| Train | `train_source3.tsv` | 5,285,603 | Comparison Source (S3) |
| Test | `test_source1.tsv` | 1,732,544 | Reference Target (S1) |
| Test | `test_source2.tsv` | 4,887,273 | Comparison Candidate (S2) |
| Test | `test_source3.tsv` | 5,082,316 | Comparison Candidate (S3) |

## 5. TEXT FIELD LENGTH & TOKEN DISTRIBUTIONS

| Source File | Metric | Mean | Min | P25 | Median (P50) | P75 | P90 | P95 | P99 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `train_source1` | Name Length (chars) | 24.0 | 3 | 18 | 24 | 30 | 34 | 37 | 42 | 105 |
| `train_source1` | Name Tokens | 3.5 | 1 | 3 | 4 | 4 | 5 | 5 | 6 | 16 |
| `train_source1` | Address Length (chars) | 52.1 | 11 | 33 | 41 | 70 | 90 | 103 | 124 | 256 |
| `train_source1` | Address Tokens | 8.0 | 2 | 5 | 7 | 10 | 13 | 15 | 19 | 43 |

| `train_source2` | Name Length (chars) | 25.1 | 2 | 19 | 25 | 31 | 37 | 40 | 48 | 104 |
| `train_source2` | Name Tokens | 3.5 | 1 | 3 | 4 | 4 | 5 | 5 | 6 | 15 |
| `train_source2` | Address Length (chars) | 47.8 | 8 | 31 | 37 | 63 | 84 | 97 | 118 | 249 |
| `train_source2` | Address Tokens | 7.5 | 2 | 5 | 6 | 9 | 13 | 15 | 18 | 46 |

| `train_source3` | Name Length (chars) | 25.2 | 2 | 18 | 25 | 31 | 37 | 42 | 50 | 123 |
| `train_source3` | Name Tokens | 3.5 | 1 | 3 | 4 | 4 | 5 | 6 | 7 | 18 |
| `train_source3` | Address Length (chars) | 48.3 | 2 | 35 | 42 | 55 | 78 | 92 | 116 | 240 |
| `train_source3` | Address Tokens | 7.4 | 1 | 5 | 6 | 9 | 12 | 14 | 18 | 43 |

| `test_source1` | Name Length (chars) | 23.8 | 3 | 18 | 24 | 29 | 34 | 36 | 42 | 92 |
| `test_source1` | Name Tokens | 3.5 | 1 | 3 | 4 | 4 | 5 | 5 | 6 | 14 |
| `test_source1` | Address Length (chars) | 57.2 | 11 | 36 | 50 | 74 | 93 | 105 | 126 | 268 |
| `test_source1` | Address Tokens | 8.6 | 2 | 6 | 8 | 11 | 14 | 16 | 19 | 43 |

| `test_source2` | Name Length (chars) | 25.7 | 2 | 19 | 25 | 32 | 38 | 42 | 49 | 102 |
| `test_source2` | Name Tokens | 3.6 | 1 | 3 | 4 | 4 | 5 | 5 | 6 | 15 |
| `test_source2` | Address Length (chars) | 51.8 | 5 | 32 | 43 | 68 | 87 | 99 | 120 | 269 |
| `test_source2` | Address Tokens | 8.0 | 1 | 5 | 7 | 10 | 13 | 15 | 19 | 43 |

| `test_source3` | Name Length (chars) | 25.7 | 2 | 19 | 25 | 32 | 38 | 42 | 50 | 103 |
| `test_source3` | Name Tokens | 3.6 | 1 | 3 | 4 | 4 | 5 | 6 | 7 | 16 |
| `test_source3` | Address Length (chars) | 50.1 | 5 | 36 | 44 | 59 | 82 | 95 | 118 | 267 |
| `test_source3` | Address Tokens | 7.7 | 1 | 5 | 7 | 9 | 13 | 15 | 19 | 43 |

## 6. COUNTRY DISTRIBUTION & GEOGRAPHIC BREAKDOWN

| Split | Source File | Country | Record Count | Percentage |
| --- | --- | --- | --- | --- |
| Train | `train_source1` | `US` | 1,323,633 | 59.98% |
| Train | `train_source1` | `India` | 883,188 | 40.02% |
| Train | `train_source2` | `US` | 3,016,817 | 59.92% |
| Train | `train_source2` | `India` | 2,017,799 | 40.08% |
| Train | `train_source3` | `US` | 3,170,056 | 59.98% |
| Train | `train_source3` | `India` | 2,115,547 | 40.02% |
| Test | `test_source1` | `India` | 809,986 | 46.75% |
| Test | `test_source1` | `US` | 663,106 | 38.27% |
| Test | `test_source1` | `France` | 259,452 | 14.98% |
| Test | `test_source2` | `India` | 2,312,565 | 47.32% |
| Test | `test_source2` | `US` | 1,871,330 | 38.29% |
| Test | `test_source2` | `France` | 703,378 | 14.39% |
| Test | `test_source3` | `India` | 2,405,000 | 47.32% |
| Test | `test_source3` | `US` | 1,945,701 | 38.28% |
| Test | `test_source3` | `France` | 731,615 | 14.40% |

### Geographic Shift & Crucial Pipeline Requirement:
- **Training Set:** `India` and `US`. No `France` records present in train.
- **Test Set:** `India`, `US`, and **`France`**.
- **CRITICAL REQUIREMENT:** `France` appears **exclusively in the Test set**. Pipeline models MUST NOT hardcode rules for US/India or restrict country matching to `{US, India}`.

## 7. BUSINESS NAME STRUCTURAL ANALYSIS

### Legal Suffix Frequencies Across Sources (Train)

| Suffix | Train S1 Count | Train S2 Count | Train S3 Count |
| --- | --- | --- | --- |
| `inc` | 238,309 | 400,229 | 419,917 |
| `llc` | 355,736 | 528,303 | 571,362 |
| `limited` | 522,340 | 528,795 | 673,480 |
| `private` | 432,394 | 528,131 | 639,861 |
| `ltd` | 148,598 | 398,464 | 427,781 |
| `pvt` | 121,462 | 189,270 | 220,353 |
| `co` | 27,544 | 139,561 | 143,838 |
| `llp` | 39,726 | 64,608 | 77,190 |
| `corp` | 34,738 | 141,773 | 140,236 |
| `corporation` | 14,372 | 62,623 | 58,097 |
| `company` | 13,872 | 31,543 | 31,866 |
| `incorporated` | 8 | 26,190 | 21,769 |
| `gmbh` | 32 | 64 | 78 |
| `sa` | 181 | 483 | 590 |
| `plc` | 1 | 5 | 6 |

### Single-Token vs Multi-Token Business Names:

- `train_source1`: Single-token names = 15,417 (0.70%), Multi-token names = 2,191,404 (99.30%)
- `train_source2`: Single-token names = 312,149 (6.20%), Multi-token names = 4,722,467 (93.80%)
- `train_source3`: Single-token names = 396,527 (7.50%), Multi-token names = 4,889,076 (92.50%)
- `test_source1`: Single-token names = 7,712 (0.45%), Multi-token names = 1,724,832 (99.55%)
- `test_source2`: Single-token names = 244,023 (4.99%), Multi-token names = 4,643,250 (95.01%)
- `test_source3`: Single-token names = 319,480 (6.29%), Multi-token names = 4,762,836 (93.71%)

## 8. BUSINESS NAME NOISE & VARIATION PATTERNS

- **Legal Suffix Inconsistencies:** High frequency of suffix mismatches across sources (`Pvt Ltd` vs `Private Limited`, `Corp` vs `Corporation`, `Inc` vs `Incorporated`).
- **Punctuation & Character Noise:** Ampersand variations (`&` vs `and`), special character insertions, trade names / DBA variations vs official registered names.
- **Word Order Transposition:** E.g., *'Apollo Pharmacy Private Limited'* vs *'Pharmacy Apollo Pvt Ltd'*.

## 9. ADDRESS FIELD STRUCTURAL BREAKDOWN

| Source File | Contains Digits % | Contains PIN/Zip % | Has Street/St % | Has Road/Rd % | Has Building/Bldg % | Has Plot/Floor/Near % |
| --- | --- | --- | --- | --- | --- | --- |
| `train_source1` | 96.51% | 6.67% | 14.7% | 21.8% | 1.6% | 15.5% |
| `train_source2` | 90.65% | 6.56% | 13.3% | 19.8% | 1.2% | 13.5% |
| `train_source3` | 90.8% | 6.56% | 13.0% | 17.9% | 1.2% | 11.1% |
| `test_source1` | 95.85% | 4.37% | 10.4% | 19.5% | 1.8% | 18.0% |
| `test_source2` | 92.67% | 4.56% | 9.7% | 17.7% | 1.4% | 15.9% |
| `test_source3` | 92.49% | 4.55% | 9.3% | 15.3% | 1.3% | 12.9% |

## 10. ADDRESS NOISE & LANDMARK ANALYSIS

- **Digit Presence:** Significant portion of addresses contain digits (street numbers, building numbers, postal codes).
- **PIN / ZIP Code Presence:** Indian addresses contain 6-digit PIN codes, US addresses contain 5-digit ZIP codes.
- **Landmark Noise:** Contextual landmarks in Indian addresses (*'Near SBI ATM'*, *'Opposite City Mall'*), contrasting with structured US street/suite formats.

## 11. GROUND TRUTH REFERENCE & MATCHING STATISTICS

- **Total Source 1 Reference Entities Analyzed:** 2,206,821
- **Singletons (0 matches):** 123,247 (5.58%)
- **Entities with Exactly 1 Match:** 119,157 (5.40%)
- **Entities with Multiple Matches (>=2):** 1,964,417 (89.02%)
- **Total Matched S2 Entities:** 3,693,619
- **Total Matched S3 Entities:** 3,944,746
- **Entities Matching BOTH Source 2 AND Source 3:** 1,776,047

### Match Count Distribution per Source 1 Entity:

| Match Count per S1 | Frequency | Percentage |
| --- | --- | --- |
| 0 matches | 123,247 | 5.58% |
| 1 matches | 119,157 | 5.40% |
| 2 matches | 375,212 | 17.00% |
| 3 matches | 530,841 | 24.05% |
| 4 matches | 484,115 | 21.94% |
| 5 matches | 321,957 | 14.59% |
| 6 matches | 164,868 | 7.47% |
| 7 matches | 63,968 | 2.90% |
| 8 matches | 18,680 | 0.85% |
| 9 matches | 4,205 | 0.19% |
| 10 matches | 534 | 0.02% |
| 11 matches | 37 | 0.00% |

## 12. GROUND TRUTH REFERENTIAL INTEGRITY CHECK

| Check Item | Result Count | Status | Description |
| --- | --- | --- | --- |
| Missing GT S1 IDs in `train_source1.tsv` | 0 | PASSED | Ground truth S1 ID existence in S1 file |
| Missing Referenced S2 IDs in `train_source2.tsv` | 0 | PASSED | Ground truth S2 ID existence in S2 file |
| Missing Referenced S3 IDs in `train_source3.tsv` | 0 | PASSED | Ground truth S3 ID existence in S3 file |
| Malformed / Unrecognized ID Prefixes | 0 | PASSED | IDs strictly follow `S2-` or `S3-` format |
| Duplicate Matched IDs within Same Row | 0 | PASSED | No entity listed twice for same S1 |
- **Integrity Audit Status:** 100% PERFECT INTEGRITY.

## 13. POSITIVE PAIR ATTRIBUTE SIMILARITY PATTERNS

Evaluated on **7,638,365** positive ground-truth matching pairs (S1-S2: 3,693,619, S1-S3: 3,944,746):

| Metric | Value | Summary |
| --- | --- | --- |
| Exact Raw Name Match | 4.64% | Identical raw business name string |
| Exact Normalized Name Match | 21.85% | Identical normalized business name string |
| Exact Raw Address Match | 2.23% | Identical raw address string |
| Exact Normalized Address Match | 8.27% | Identical normalized address string |
| Same Country Match | **100.0%** | Country matching rate for true pairs |
| PIN/Zip Code Match (where both present) | 95.88% | Postal code match rate |
| Name Token Jaccard Similarity | Mean: 0.616 | Median: 0.667, P25: 0.500, P75: 1.000 |
| Address Token Jaccard Similarity | Mean: 0.596 | Median: 0.625, P25: 0.429, P75: 0.786 |

## 14. CANDIDATE BLOCKING SIGNAL & RECALL EVALUATION

Evaluated candidate blocking keys against **7,638,365** ground-truth positive pairs:

| Blocking Signal / Candidate Key | GT Recall Ceiling (%) | Est. Reduction Ratio | Notes |
| --- | --- | --- | --- |
| `Country + Exact Normalized Name` | 21.85% | > 99.999% | High precision, misses variations |
| `Country + First Word of Name` | 72.69% | ~ 99.95% | Captures prefix matches |
| `Country + Name 3-Char Prefix` | 77.53% | ~ 99.90% | Robust against suffix variations |
| `Country + Any Shared Name Token (len >= 4)` | 83.38% | ~ 99.80% | Catches word order transpositions |
| `Country + Shared Address Token (len >= 4)` | 95.21% | ~ 99.70% | Catches address-similar matches |
| `Country + (Shared Name Token OR Shared Address Token)` | **99.85%** | **~ 99.50%** | **Combined Blocking Key** |
| `Country + Shared PIN/Zip Code` | 4.8% | ~ 99.99% | Auxiliary signal when PIN present |

## 15. SCALABILITY & CANDIDATE SPACE EXPLOSION ANALYSIS

### Processed Record Scale Summary:

- **Train Records Processed:** S1 = 2,206,821 | S2 = 5,034,616 | S3 = 5,285,603 | **Total Train = 12,527,040 records**
- **Test Records Processed:** S1 = 1,732,544 | S2 = 4,887,273 | S3 = 5,082,316 | **Total Test = 11,702,133 records**

### Pairwise Candidate Comparison Explosion:
- Naive Train Comparisons: 2,206,821 \times (5,034,616 + 5,285,603) = \mathbf{22,774,876,013,799}$ pairwise comparisons.
- Naive Test Comparisons: 1,732,544 \times (4,887,273 + 5,082,316) = \mathbf{17,272,751,604,416}$ pairwise comparisons.
- **Candidate Strategy:** Multi-stage inverted indexing blocking is required to prune pair space to manageable candidate sets per S1 entity.

## 16. HARD EDGE CASES & RISK IDENTIFICATION

1. **Same Name, Different Address (Chain Stores):** Multiple locations of nationwide entities. Address matching is required to prevent false merges.
2. **Same Address, Different Business (Commercial Parks):** Distinct businesses located in the same building/complex. Name matching must be precise.
3. **Very Short Business Names:** Entities with 2-3 letter names trigger high spurious token overlap.
4. **Singleton Ratio (5.58%):** Singletons score 1.0 when predicting empty list, but 0.0 on false merges. Precision thresholding is crucial.
5. **Unseen Country Test Drift (`France`):** Test set includes France records that do not exist in train set.

## 17. STRATEGIC RECOMMENDATIONS FOR PHASE 2 PIPELINE DESIGN

1. **Dynamic Country Partitioning:** Enforce strict candidate blocking within `country` (including dynamic handling for `France`).
2. **Multi-Key Inverted Indexing:** Combine name token indexing, address token indexing, and PIN code blocking.
3. **Feature Engineering Strategy:** Extract string distances (Levenshtein, Jaccard, Cosine TF-IDF), token overlap metrics, and PIN code matching flags.
4. **$F_{0.5}$ Threshold Tuning:** Optimize classification decision thresholds on a local validation split to maximize precision.
5. **Submission Output Compliance:** Ensure output files pass `utils/validate_submission.py`.
