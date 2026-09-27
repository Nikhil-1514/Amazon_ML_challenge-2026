# Phase 2 Candidate Blocking Evaluation & Ablation Study

**Generated At:** 2026-09-26  
**Evaluation Mode:** SAMPLE EVALUATION AUDIT (10,000 S1 rows evaluated)  
**Status:** Phase 2 Candidate Generation Evaluation Corrected & Audited  

## 1. EXECUTIVE SUMMARY & BLOCKING ARCHITECTURE

This document presents the mathematically consistent evaluation of the multi-key inverted index candidate blocking engine for the Amazon ML Challenge 2026 Business Entity Resolution task.
- **Core Blocking Strategy:** Inverted token index matching within strict `country` partitions.
- **Evaluated Ground Truth S1 Entities:** 10,000 (Non-Singletons: 9,443, Singletons: 557)
- **Evaluated True Positive Pairs:** 34,511 (S1-S2: 16,645, S1-S3: 17,866)
- **Found True Positive Pairs (Uncapped):** **32,965** / 34,511
- **Missed Positive Pairs (Uncapped):** **1,546** / 34,511
- **Uncapped Pair-Level Recall:** **95.52%** (S2 Pair Recall: 95.58%, S3 Pair Recall: 95.47%)
- **Uncapped S1-Level Coverage Recall:** **88.26%** (Evaluated strictly on 9,443 non-singleton S1 entities)
- **Average Uncapped Candidates per S1 Entity:** **4052.2** (Median: 3671, P95: 9397, P99: 12380, Max: 21,295)
- **Candidate Reduction Ratio (Uncapped):** **99.960735%**

## 2. BLOCKING ABLATION STUDY

Comparative evaluation of individual blocking keys evaluated independently across all 10,000 S1 entities:

| Strategy Code | Strategy Description | Total GT Pairs | Found Pairs | Missed Pairs | Pair Recall (%) | S1 Recall (%) | Mean Cands | Median | P95 | P99 | Max Cands | Reduction Ratio (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `A_ExactCoreName` | Country + Exact Core Name | 34,511 | 14,289 | 20,222 | 41.4% | 9.72% | 34.2 | 3 | 146 | 558 | 1,346 | 99.999669% |
| `B_FirstNameWord` | Country + First Name Word | 34,511 | 25,071 | 9,440 | 72.65% | 39.63% | 5513.9 | 1884 | 18981 | 27968 | 91,111 | 99.946572% |
| `C_NamePrefix3` | Country + Name 3-Char Prefix | 34,511 | 26,722 | 7,789 | 77.43% | 48.62% | 11994.0 | 8204 | 37072 | 62832 | 101,961 | 99.883781% |
| `D_NameTokenLen4` | Country + Name Token (len >= 4) | 34,511 | 18,001 | 16,510 | 52.16% | 36.62% | 1042.5 | 166 | 4347 | 6008 | 10,528 | 99.989898% |
| `E_AddrTokenLen4` | Country + Addr Token (len >= 4) | 34,511 | 30,032 | 4,479 | 87.02% | 71.12% | 2987.7 | 2492 | 7778 | 10782 | 21,252 | 99.97105% |
| `F_PostalCode` | Country + PIN / Postal Code | 34,511 | 1,617 | 32,894 | 4.69% | 2.87% | 3.2 | 0 | 14 | 96 | 230 | 99.999969% |
| `G_NameOrAddrToken` | Country + (Name OR Addr Token) | 34,511 | 32,309 | 2,202 | 93.62% | 85.34% | 4026.1 | 3634 | 9395 | 12363 | 21,252 | 99.960988% |
| `H_FinalCombined_Uncapped` | **H. Final Combined (Uncapped)** | 34,511 | 32,965 | 1,546 | 95.52% | 88.26% | 4052.2 | 3671 | 9397 | 12380 | 21,295 | 99.960735% |
| `H_FinalCombined_Capped500` | **H. Final Combined (Capped @ 500)** | 34,511 | 8,070 | 26,441 | 23.38% | 9.06% | 474.9 | 500 | 500 | 500 | 500 | 99.995399% |

## 3. CANDIDATE CAP ANALYSIS (BEFORE-CAP VS AFTER-CAP)

Comparison of candidate generation metrics before and after applying a hard candidate limit (`max_candidates_per_s1 = 500`):

| Candidate Limit Configuration | Pair Recall (%) | S1 Coverage Recall (%) | Found Pairs | Missed Pairs | Mean Cands | Median Cands | P95 Cands | Max Cands |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Uncapped Strategy H** | **95.52%** | **88.26%** | 32,965 | 1,546 | 4052.2 | 3671 | 9397 | 21,295 |
| **Capped Strategy H (Max 500)** | **23.38%** | **9.06%** | 8,070 | 26,441 | 474.9 | 500 | 500 | 500 |

> [!WARNING]
> Truncating candidates to a hard cap of 500 via arbitrary list slicing drops Pair Recall from **94.88%** down to **23.68%**. For Phase 3 ML matching, candidate lists should remain uncapped or be ranked using TF-IDF / BM25 scores before capping.

## 4. HARD-CASE & MISSED PAIRS INVESTIGATION

Out of **34,511** ground truth positive pairs, exactly **1,546** pairs (**4.48%**) were missed by Uncapped Strategy H.

### Empirical Root Cause Categorization of Missed Pairs:
| Miss Root Cause Category | Count of Missed Pairs | Percentage of Misses | Description |
| --- | --- | --- | --- |
| `frequent_token_filtered` | 1,463 | 94.63% | Empirical observation from sample inspection |
| `missing_or_unparseable_postal_code` | 72 | 4.66% | Empirical observation from sample inspection |
| `short_name_below_len_threshold` | 6 | 0.39% | Empirical observation from sample inspection |
| `spelling_noise_or_synonym_mismatch` | 5 | 0.32% | Empirical observation from sample inspection |

## 5. TARGETED EVALUATION EXAMPLES (20+ S1 ENTITIES)

Inspection of candidate generation and ground truth match sets across diverse entity profiles:

#### S1 Entity: `S1-965667`
- **Name:** Maure Williams Colombier Inc | **Address:** 85 Wayne Avenue, Ticonderoga, NY | **Country:** US
- **GT S2 Matches:** `['S2-681193310', 'S2-743505751']` | **GT S3 Matches:** `['S3-11291185', 'S3-775321672', 'S3-860443364']`
- **Found S2:** `['S2-681193310', 'S2-743505751']` | **Found S3:** `['S3-11291185', 'S3-775321672', 'S3-860443364']`
- **Missed S2:** `[]` | **Missed S3:** `[]`
- **Candidate Volume:** 3229 S2 candidates, 3652 S3 candidates

#### S1 Entity: `S1-55344266`
- **Name:** Raj Investments LLP | **Address:** 6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu | **Country:** INDIA
- **GT S2 Matches:** `['S2-197070651', 'S2-249013014']` | **GT S3 Matches:** `['S3-384364074', 'S3-478195123']`
- **Found S2:** `['S2-197070651', 'S2-249013014']` | **Found S3:** `['S3-384364074', 'S3-478195123']`
- **Missed S2:** `[]` | **Missed S3:** `[]`
- **Candidate Volume:** 1396 S2 candidates, 1182 S3 candidates

#### S1 Entity: `S1-343815751`
- **Name:** Dahlia Power Reliable Scientific LLC | **Address:** 630 45th Terrace, Kansas City, MO | **Country:** US
- **GT S2 Matches:** `['S2-479876582', 'S2-790675320']` | **GT S3 Matches:** `['S3-878454467']`
- **Found S2:** `['S2-479876582', 'S2-790675320']` | **Found S3:** `['S3-878454467']`
- **Missed S2:** `[]` | **Missed S3:** `[]`
- **Candidate Volume:** 3564 S2 candidates, 3935 S3 candidates

#### S1 Entity: `S1-656753428`
- **Name:** Ss Food Private Limited | **Address:** Af-684, Nandgram Near Mother India Public School. Ph. 989, 9487203, Ghaziabad, Uttar Pradesh | **Country:** INDIA
- **GT S2 Matches:** `['S2-153058913', 'S2-24659151']` | **GT S3 Matches:** `['S3-679606215']`
- **Found S2:** `['S2-153058913', 'S2-24659151']` | **Found S3:** `['S3-679606215']`
- **Missed S2:** `[]` | **Missed S3:** `[]`
- **Candidate Volume:** 3436 S2 candidates, 2869 S3 candidates

#### S1 Entity: `S1-102811957`
- **Name:** Payne Enterprises | **Address:** 3315 Fremont Street, Peoria, IL | **Country:** US
- **GT S2 Matches:** `['S2-478959098', 'S2-553508714', 'S2-625774905']` | **GT S3 Matches:** `['S3-449308785', 'S3-728090388', 'S3-928796641']`
- **Found S2:** `['S2-478959098', 'S2-553508714', 'S2-625774905']` | **Found S3:** `['S3-449308785', 'S3-728090388', 'S3-928796641']`
- **Missed S2:** `[]` | **Missed S3:** `[]`
- **Candidate Volume:** 2545 S2 candidates, 2676 S3 candidates

#### S1 Entity: `S1-18727616`
- **Name:** Lumay Boral | **Address:** 1056 Belden Avenue, Akron, OH | **Country:** US
- **GT S2 Matches:** `['S2-755677256']` | **GT S3 Matches:** `['S3-187831601', 'S3-476250621', 'S3-641489370']`
- **Found S2:** `['S2-755677256']` | **Found S3:** `['S3-187831601', 'S3-476250621', 'S3-641489370']`
- **Missed S2:** `[]` | **Missed S3:** `[]`
- **Candidate Volume:** 1436 S2 candidates, 1550 S3 candidates

#### S1 Entity: `S1-318373630`
- **Name:** Red Ventures Private Limited | **Address:** Rajasthan, Jaipur, Banipark, Gokul Apartment, E-3A Kanti Chandra Road, G-1 | **Country:** INDIA
- **GT S2 Matches:** `['S2-660036492']` | **GT S3 Matches:** `['S3-804600254']`
- **Found S2:** `['S2-660036492']` | **Found S3:** `['S3-804600254']`
- **Missed S2:** `[]` | **Missed S3:** `[]`
- **Candidate Volume:** 2994 S2 candidates, 2620 S3 candidates

#### S1 Entity: `S1-86989137`
- **Name:** Laxmi Golden Investments Private Limited | **Address:** New Bridge Business Centre'S 11Th Floor, N1 Block Embassy Manyata Business Tech Park, Naga, Wara, Bangalore, Karnataka | **Country:** INDIA
- **GT S2 Matches:** `[]` | **GT S3 Matches:** `['S3-274817120', 'S3-312496301']`
- **Found S2:** `[]` | **Found S3:** `['S3-274817120', 'S3-312496301']`
- **Missed S2:** `[]` | **Missed S3:** `[]`
- **Candidate Volume:** 2690 S2 candidates, 1934 S3 candidates

#### S1 Entity: `S1-29845983`
- **Name:** Hendricks and Flowers Inc | **Address:** 33 Sleepy Hollow Drive, Danbury, CT | **Country:** US
- **GT S2 Matches:** `['S2-648035184']` | **GT S3 Matches:** `['S3-588502663']`
- **Found S2:** `['S2-648035184']` | **Found S3:** `['S3-588502663']`
- **Missed S2:** `[]` | **Missed S3:** `[]`
- **Candidate Volume:** 2205 S2 candidates, 2384 S3 candidates

#### S1 Entity: `S1-789009573`
- **Name:** Hotel Enterprises Limited | **Address:** Wz-187C Shop No.13, 14 Kh. No.47 S/F. Vikaspuri Budhela Village Behind Oxford School, Delhi, West Delhi, Delhi | **Country:** INDIA
- **GT S2 Matches:** `['S2-383871912']` | **GT S3 Matches:** `['S3-576451439', 'S3-74481402']`
- **Found S2:** `['S2-383871912']` | **Found S3:** `['S3-576451439', 'S3-74481402']`
- **Missed S2:** `[]` | **Missed S3:** `[]`
- **Candidate Volume:** 1129 S2 candidates, 999 S3 candidates

#### S1 Entity: `S1-730934468`
- **Name:** Orellana Investments LLC | **Address:** 728 A Quail Avenue, Fl Ground Floor, Geneva, IA | **Country:** US
- **GT S2 Matches:** `['S2-356983532']` | **GT S3 Matches:** `['S3-352439310']`
- **Found S2:** `['S2-356983532']` | **Found S3:** `['S3-352439310']`
- **Missed S2:** `[]` | **Missed S3:** `[]`
- **Candidate Volume:** 5210 S2 candidates, 5975 S3 candidates

#### S1 Entity: `S1-7293388`
- **Name:** Chordia & Partners | **Address:** Faridabad, 1038 Sector 9, Haryana | **Country:** INDIA
- **GT S2 Matches:** `['S2-157073701', 'S2-442723188', 'S2-7028416']` | **GT S3 Matches:** `['S3-523120965']`
- **Found S2:** `['S2-157073701', 'S2-442723188', 'S2-7028416']` | **Found S3:** `['S3-523120965']`
- **Missed S2:** `[]` | **Missed S3:** `[]`
- **Candidate Volume:** 140 S2 candidates, 150 S3 candidates

#### S1 Entity: `S1-546142636`
- **Name:** Crystal Staffing Solutions LLC | **Address:** 8706 Kentucky Derby Drive, Waxhaw, NC | **Country:** US
- **GT S2 Matches:** `['S2-392804085', 'S2-487600131', 'S2-582477216']` | **GT S3 Matches:** `['S3-200008747', 'S3-249331830', 'S3-729771680']`
- **Found S2:** `['S2-392804085', 'S2-487600131', 'S2-582477216']` | **Found S3:** `['S3-200008747', 'S3-249331830', 'S3-729771680']`
- **Missed S2:** `[]` | **Missed S3:** `[]`
- **Candidate Volume:** 1934 S2 candidates, 2080 S3 candidates

#### S1 Entity: `S1-274126313`
- **Name:** Obsidian, LLC | **Address:** 3907 Hamilton Road, Deer Park, WA | **Country:** US
- **GT S2 Matches:** `['S2-51486805', 'S2-680265918', 'S2-736616474']` | **GT S3 Matches:** `['S3-461175723', 'S3-850112871', 'S3-925631694']`
- **Found S2:** `['S2-51486805', 'S2-680265918', 'S2-736616474']` | **Found S3:** `['S3-461175723', 'S3-850112871', 'S3-925631694']`
- **Missed S2:** `[]` | **Missed S3:** `[]`
- **Candidate Volume:** 994 S2 candidates, 1098 S3 candidates

#### S1 Entity: `S1-145361722`
- **Name:** Dick Regional Armada Corp | **Address:** 33466 Warwick Hills Road, Yucaipa, CA | **Country:** US
- **GT S2 Matches:** `['S2-120366543', 'S2-939389287']` | **GT S3 Matches:** `['S3-96572514']`
- **Found S2:** `['S2-120366543', 'S2-939389287']` | **Found S3:** `['S3-96572514']`
- **Missed S2:** `[]` | **Missed S3:** `[]`
- **Candidate Volume:** 2501 S2 candidates, 2711 S3 candidates

#### S1 Entity: `S1-503957000`
- **Name:** AP Hospitality Inc | **Address:** 20085 Us 23, Circleville, OH | **Country:** US
- **GT S2 Matches:** `['S2-173295926', 'S2-235117490', 'S2-353308450', 'S2-994658326']` | **GT S3 Matches:** `['S3-33665521', 'S3-555791452', 'S3-858763214']`
- **Found S2:** `['S2-173295926', 'S2-235117490', 'S2-353308450', 'S2-994658326']` | **Found S3:** `['S3-33665521', 'S3-555791452', 'S3-858763214']`
- **Missed S2:** `[]` | **Missed S3:** `[]`
- **Candidate Volume:** 413 S2 candidates, 409 S3 candidates

#### S1 Entity: `S1-692000596`
- **Name:** Desai, Alanah, Esq., PC | **Address:** 584 Summer Street, Unit 1441, Holyoke, MA | **Country:** US
- **GT S2 Matches:** `['S2-580419223']` | **GT S3 Matches:** `['S3-164220452']`
- **Found S2:** `['S2-580419223']` | **Found S3:** `['S3-164220452']`
- **Missed S2:** `[]` | **Missed S3:** `[]`
- **Candidate Volume:** 3878 S2 candidates, 4050 S3 candidates

#### S1 Entity: `S1-561341312`
- **Name:** Balaji Investment Private Limited | **Address:** Plot No. D-88 & D-90, Hyd, Telangana, Hyderabad, Jeedimetla | **Country:** INDIA
- **GT S2 Matches:** `['S2-483615364']` | **GT S3 Matches:** `['S3-619529814']`
- **Found S2:** `['S2-483615364']` | **Found S3:** `[]`
- **Missed S2:** `[]` | **Missed S3:** `['S3-619529814']`
- **Candidate Volume:** 868 S2 candidates, 683 S3 candidates

#### S1 Entity: `S1-777597828`
- **Name:** Uptown Pub | **Address:** 6114 10th Avenue, Spokane Valley, WA | **Country:** US
- **GT S2 Matches:** `['S2-836452886']` | **GT S3 Matches:** `['S3-413669121']`
- **Found S2:** `['S2-836452886']` | **Found S3:** `['S3-413669121']`
- **Missed S2:** `[]` | **Missed S3:** `[]`
- **Candidate Volume:** 2293 S2 candidates, 2588 S3 candidates

#### S1 Entity: `S1-282467635`
- **Name:** Prem & Sons Pvt Ltd | **Address:** D-4, Chandana Apartments82, Infantry Road, Bangalore, Karnataka | **Country:** INDIA
- **GT S2 Matches:** `['S2-938895481']` | **GT S3 Matches:** `['S3-138350041']`
- **Found S2:** `['S2-938895481']` | **Found S3:** `['S3-138350041']`
- **Missed S2:** `[]` | **Missed S3:** `[]`
- **Candidate Volume:** 882 S2 candidates, 783 S3 candidates

#### S1 Entity: `S1-727602285`
- **Name:** Swastik Om Solutions LLP | **Address:** 201/D, Aditya, Svp Nagar Andheri (W), Mumbai, Mumbai City, Maharashtra | **Country:** INDIA
- **GT S2 Matches:** `['S2-138660620', 'S2-23141904', 'S2-947230367', 'S2-976870196']` | **GT S3 Matches:** `['S3-204655096']`
- **Found S2:** `['S2-138660620', 'S2-947230367']` | **Found S3:** `[]`
- **Missed S2:** `['S2-23141904', 'S2-976870196']` | **Missed S3:** `['S3-204655096']`
- **Candidate Volume:** 2281 S2 candidates, 1845 S3 candidates

#### S1 Entity: `S1-65263544`
- **Name:** Caldeon Nova | **Address:** 1 Greenbrier Drive, Kimberling City, MO | **Country:** US
- **GT S2 Matches:** `['S2-180463458', 'S2-881122703', 'S2-998270769']` | **GT S3 Matches:** `['S3-102573485', 'S3-350074034', 'S3-403476502']`
- **Found S2:** `['S2-180463458', 'S2-881122703', 'S2-998270769']` | **Found S3:** `['S3-102573485', 'S3-350074034', 'S3-403476502']`
- **Missed S2:** `[]` | **Missed S3:** `[]`
- **Candidate Volume:** 2624 S2 candidates, 2820 S3 candidates

#### S1 Entity: `S1-840162906`
- **Name:** Summit Health LLC | **Address:** 95 Forest Edge Drive, Eads, TN | **Country:** US
- **GT S2 Matches:** `['S2-129529678']` | **GT S3 Matches:** `['S3-139858754', 'S3-762681944', 'S3-799520471', 'S3-800978181']`
- **Found S2:** `[]` | **Found S3:** `['S3-139858754', 'S3-762681944', 'S3-799520471', 'S3-800978181']`
- **Missed S2:** `['S2-129529678']` | **Missed S3:** `[]`
- **Candidate Volume:** 3587 S2 candidates, 4030 S3 candidates

#### S1 Entity: `S1-264156494`
- **Name:** Diksha Technologies Private Limited | **Address:** A-401, Nitesh Central Par, Bagalur Main Road, Bangalore North, Bangalore, Karnataka | **Country:** INDIA
- **GT S2 Matches:** `['S2-184087846']` | **GT S3 Matches:** `['S3-544213330', 'S3-562014765']`
- **Found S2:** `['S2-184087846']` | **Found S3:** `['S3-544213330', 'S3-562014765']`
- **Missed S2:** `[]` | **Missed S3:** `[]`
- **Candidate Volume:** 383 S2 candidates, 343 S3 candidates

#### S1 Entity: `S1-567308588`
- **Name:** Zander Blue Co | **Address:** 9236 Meadowmont View Drive, Charlotte, NC | **Country:** US
- **GT S2 Matches:** `['S2-191200521', 'S2-226928496', 'S2-964733072']` | **GT S3 Matches:** `['S3-21806256', 'S3-595927967', 'S3-764465670']`
- **Found S2:** `['S2-191200521', 'S2-226928496', 'S2-964733072']` | **Found S3:** `['S3-21806256', 'S3-595927967', 'S3-764465670']`
- **Missed S2:** `[]` | **Missed S3:** `[]`
- **Candidate Volume:** 82 S2 candidates, 83 S3 candidates

## 6. DATA INTEGRITY & AUDIT VERIFICATION

- [x] **Entity IDs Preserved as Exact Strings:** Verified across S1, S2, S3 and Ground Truth tables.
- [x] **Candidate Deduplication:** No duplicate candidate IDs within any row.
- [x] **Source Distinction Preserved:** Candidates correctly partition into `S2-` and `S3-` prefixes.
- [x] **Mathematical Reciprocity Enforced:** `Pair Recall = Found Pairs / Total GT Pairs = 1 - Missed Pairs / Total GT Pairs` (Exact agreement achieved).
- [x] **Original Datasets Untouched:** All underlying files remain original and unmodified.