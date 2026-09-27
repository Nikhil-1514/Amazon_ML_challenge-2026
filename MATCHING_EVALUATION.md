# Phase 3 Machine Learning Entity Matching & F0.5 Optimization Report

**Generated At:** 2026-09-26  
**Candidate Generator:** `Var5_Layered_Composite_Optimal`  
**Status:** Model Selection, Negative Sampling, Threshold Optimization & Error Analysis Completed  

## 1. DATASET

Summary of dataset entities, candidate pairs generated, and positive/negative training samples:

- **Evaluated S1 Entities (Train Pool):** 10,000
- **Total S2 Records:** 5,034,616
- **Total S3 Records:** 5,285,603
- **Total Candidate Pairs Generated:** 27,451,862
- **Positive Training Pairs:** 22,205
- **Negative Training Pairs:** 315,263
- **Positive / Negative Ratio:** 1:14.2
- **Total Pairwise Features:** 47

## 2. FEATURES IMPLEMENTED

Comprehensive breakdown of all 47 pairwise similarity and blocking evidence features implemented:

### Name Features (15)
- `name_exact` (Exact string equality)
- `core_name_exact` (Legal-suffix-stripped core name equality)
- `name_jw` (Jaro-Winkler distance)
- `name_lev_sim` (Normalized Levenshtein similarity)
- `name_char_3gram_jaccard` (Character 3-gram Jaccard similarity)
- `name_char_4gram_jaccard` (Character 4-gram Jaccard similarity)
- `name_token_jaccard` (Word token Jaccard similarity)
- `name_token_overlap_coef` (Szymkiewicz-Simpson overlap coefficient)
- `name_token_containment` (S1 token containment proportion)
- `name_common_token_cnt` (Count of shared name tokens)
- `name_first_token_equal` (First word exact equality)
- `name_first_word_jw` (First word Jaro-Winkler similarity)
- `name_token_set_sim` (Token set ratio)
- `name_len_diff` (Absolute character length difference)
- `name_rel_len_diff` (Relative length difference)

### Address Features (12)
- `addr_exact` (Exact cleaned address equality)
- `addr_token_jaccard` (Address token Jaccard similarity)
- `addr_overlap_coef` (Address token overlap coefficient)
- `addr_char_3gram_jaccard` (Address 3-gram Jaccard similarity)
- `addr_char_4gram_jaccard` (Address 4-gram Jaccard similarity)
- `addr_jw` (Address Jaro-Winkler similarity)
- `addr_lev_sim` (Address Levenshtein similarity)
- `addr_common_token_cnt` (Common address token count)
- `addr_num_overlap_cnt` (Count of matching numeric tokens in address)
- `addr_num_exact` (Exact equality of extracted numeric sets)
- `addr_locality_overlap_cnt` (Overlap of locality/city tokens)
- `rel_addr_len_diff` (Relative address length difference)

### Postal Features (4)
- `postal_exact` (Exact postal code match)
- `postal_prefix_match` (First 3 digits postal code match)
- `postal_available_both` (Presence indicator in both records)
- `postal_len_equal` (Equality of postal code string lengths)

### Country Features (1)
- `country_exact` (ISO Country code exact match)

### Numeric / Address Features
- `addr_num_overlap_cnt` (Extracted numeric token intersection count)
- `addr_num_exact` (Exact numeric set identity)
- `cross_addr_num_and_name_sim` (Interaction between numeric overlap and name Jaro-Winkler similarity)

### Blocking Evidence Features (7)
- `block_exact_core_name` (Matched via core name index)
- `block_postal` (Matched via postal code index)
- `block_name_token` (Matched via low-frequency name token)
- `block_addr_token` (Matched via address token)
- `block_freq_token_composite` (Matched via frequent token + numeric composite)
- `block_first_name_word` (Matched via first word index)
- `block_reason_cnt` (Total count of satisfied blocking channels)

### Cross-Field Features (6)
- `cross_name_and_addr_exact` (Both name and address match exactly)
- `cross_name_high_and_addr_high` (Name JW >= 0.85 and Address JW >= 0.85)
- `cross_name_sim_x_addr_sim` (Product of Name JW and Address JW)
- `cross_name_sim_plus_addr_sim` (Sum of Name JW and Address JW)
- `cross_postal_exact_and_name_sim` (Product of postal match and Name JW)
- `cross_addr_num_and_name_sim` (Product of numeric overlap indicator and Name JW)

## 3. MODELS TESTED

Evaluation of candidate classifiers trained on identical feature vectors:

| Model Name | Hyperparameters | Precision (%) | Recall (%) | F0.5 Score (%) | TP | FP | FN | Training Time (s) | Inference Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `LogisticRegression` | C=1.0, max_iter=1000 | 96.81% | 87.61% | **94.82%** | 4,730 | 156 | 669 | 1.88s | 0.078s |
| `HistGradientBoosting` | n_estimators=200, learning_rate=0.05, max_depth=8 | 98.35% | 92.61% | **97.14%** | 5,000 | 84 | 399 | 8.35s | 0.886s |
| `XGBoost` | n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8 | 98.43% | 92.98% | **97.29%** | 5,020 | 80 | 379 | 3.19s | 0.235s |
| `RandomForest` | n_estimators=150, max_depth=15, n_jobs=-1 | 98.83% | 86.28% | **96.04%** | 4,658 | 55 | 741 | 13.02s | 0.269s |
| `LightGBM` | n_estimators=300, learning_rate=0.05, max_depth=8, num_leaves=31 | 98.38% | 93.44% | **97.35%** | 5,045 | 83 | 354 | 1.79s | 0.245s |

> [!NOTE]
> **Selected Champion Model:** `LightGBM` due to superior non-linear interaction modeling and optimal F0.5 performance.

## 4. THRESHOLD OPTIMIZATION

Grid search across decision thresholds to maximize macro F0.5 (beta = 0.5, prioritizing precision over recall):

| Threshold | Precision (%) | Recall (%) | F0.5 (%) | TP | FP | FN | Predicted Matches |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `0.050` | 80.53% | 99.69% | **83.75%** | 5,382 | 1,301 | 17 | 6,683 |
| `0.100` | 84.18% | 99.48% | **86.86%** | 5,371 | 1,009 | 28 | 6,380 |
| `0.150` | 86.51% | 99.31% | **88.80%** | 5,362 | 836 | 37 | 6,198 |
| `0.200` | 88.20% | 99.11% | **90.18%** | 5,351 | 716 | 48 | 6,067 |
| `0.300` | 91.90% | 98.56% | **93.16%** | 5,321 | 469 | 78 | 5,790 |
| `0.400` | 93.68% | 98.22% | **94.55%** | 5,303 | 358 | 96 | 5,661 |
| `0.500` | 94.71% | 97.87% | **95.33%** | 5,284 | 295 | 115 | 5,579 |
| `0.600` | 95.64% | 97.46% | **96.00%** | 5,262 | 240 | 137 | 5,502 |
| `0.700` | 96.63% | 96.57% | **96.62%** | 5,214 | 182 | 185 | 5,396 |
| `0.800` | 97.45% | 95.54% | **97.06%** | 5,158 | 135 | 241 | 5,293 |
| `0.850` | 97.82% | 94.80% | **97.20%** | 5,118 | 114 | 281 | 5,232 |
| `0.900` | 98.38% | 93.44% | **97.35%** | 5,045 | 83 | 354 | 5,128 |
| `0.920` **(Selected)** | 98.71% | 92.46% | **97.40%** | 4,992 | 65 | 407 | 5,057 |
| `0.950` | 99.10% | 89.24% | **96.95%** | 4,818 | 44 | 581 | 4,862 |
| `0.970` | 99.53% | 85.65% | **96.40%** | 4,624 | 22 | 775 | 4,646 |
| `0.980` | 99.62% | 83.16% | **95.83%** | 4,490 | 17 | 909 | 4,507 |
| `0.990` | 99.79% | 78.98% | **94.79%** | 4,264 | 9 | 1,135 | 4,273 |
| `0.995` | 99.80% | 73.81% | **93.23%** | 3,985 | 8 | 1,414 | 3,993 |
| `0.999` | 99.90% | 53.66% | **85.21%** | 2,897 | 3 | 2,502 | 2,900 |

**Selected Optimal Threshold:** `0.920`
- **Validation Precision:** **98.71%**
- **Validation Recall:** **92.46%**
- **Validation F0.5 Score:** **97.40%**

## 5. SOURCE BREAKDOWN (S2 VS S3)

Independent evaluation of entity matching performance partitioned by candidate source at the optimal threshold:

| Source Partition | Precision (%) | Recall (%) | F0.5 Score (%) | True Positives (TP) | False Positives (FP) | False Negatives (FN) |
| --- | --- | --- | --- | --- | --- | --- |
| **Source 2 (S2)** | **99.00%** | 92.66% | **97.66%** | 2,374 | 24 | 188 |
| **Source 3 (S3)** | **98.46%** | 92.28% | **97.16%** | 2,618 | 41 | 219 |
| **Combined (S2 + S3)** | **98.71%** | 92.46% | **97.40%** | 4,992 | 65 | 407 |

## 6. NO-MATCH ANALYSIS

Evaluation of matcher performance on true no-match (singleton) S1 entities:

- **True No-Match S1 Entities:** 117
- **Correctly Rejected (No Match Predicted):** 110
- **Incorrectly Matched (False Positives):** 7
- **False-Match Rate:** **5.98%**

## 7. ERROR ANALYSIS

Detailed investigation of false positives and false negatives from the validation set:

### Recurring Failure Patterns
1. **Common Business Names:** Highly generic entity names (e.g., *National Traders*, *Sunrise Enterprises*) without strong address overlap can produce false matches if address token similarities are borderline.
2. **Missing Addresses:** Entities with empty or minimal address strings rely exclusively on name tokens, increasing false positive risk on common brand names.
3. **Spelling Variations & Typos:** Transposed characters or OCR typos in names reduced token Jaccard scores, occasionally causing true matches to fall below threshold (False Negatives).
4. **Short Business Names:** Entities with 2-3 letter names (e.g. *3M*, *HP*, *GE*) suffer low n-gram similarity scores when formatted differently across sources.
5. **Frequent Tokens:** Generic suffixes (*Pvt Ltd*, *Group*, *Services*) that passed through normalization required strong core name matching to avoid noise.
6. **Address Similarity with Different Name:** Co-located businesses in commercial hubs (e.g. *Plot 12 Technopark*) share identical addresses but distinct names.
7. **Postal Code Conflicts:** Differing postal code formats (5-digit vs 6-digit PINs) led to missing postal evidence features.
8. **Duplicate-Looking Businesses:** Multiple legal branches or subsidiaries of the same brand in different districts created ambiguous matches.

### Representative False Positives (Predicted Match = 1, Ground Truth = 0)
| S1 ID | S1 Name | S1 Address | Candidate ID | Candidate Name | Candidate Address | Prob | Blocking Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `S1-404435514` | colline molitor sea llc | 5620 trinity road unit 103 ral | `S3-163265851` | colline molitor sea llc |  | 0.9573 | NAME_TOKEN, EXACT_CORE_NAME |
| `S1-435587816` | iraheta dacie d o | 77 morrison street struthers o | `S3-915899709` | iraheta dsacie do inc | struthers 86 d morrison street | 0.9416 | NAME_TOKEN |
| `S1-545582895` | global techzone foundation lim | plot no 1 gut no 27 bankarwadi | `S3-646524505` | global global techzone foundat | plot no 4 gut no 27 bankarwadi | 0.9589 | NAME_TOKEN |
| `S1-780893579` | godbole city corp | building no 61 245 b2 zeia arc | `S3-362174727` | godbole city industries | builidng no 61 245 b3 zeia arc | 0.9959 | NAME_TOKEN, FREQ_TOKEN_NUMERIC_COMPOSITE |
| `S1-895409233` | elysia s therapy corp | 584 christine drive vacaville  | `S3-478079935` | elysia s therapy co | 597 christine dr vacaville cit | 0.9574 | NAME_TOKEN, FREQ_TOKEN_NUMERIC_COMPOSITE, EXACT_CORE_NAME |
| `S1-92668465` | garciamartinez and schorr hold | 2204 lea crest drive dallas tx | `S2-398949433` | salame and schorr holdings llc | 2204 lea crist dr dallas tx | 0.9740 | NAME_TOKEN |
| `S1-858093716` | secure nutrition services | 1405 hearn avenue bessemer al | `S3-329772346` | secure nutrition services |  | 0.9614 | NAME_TOKEN, FREQ_TOKEN_NUMERIC_COMPOSITE, EXACT_CORE_NAME |
| `S1-921473285` | yamuansh trims ltd | 3rd and 4th floor plot no gen  | `S3-581757185` | yamuinsh yamuinsh trims | h no 4rd and 4th floor plot no | 0.9985 | NAME_TOKEN |
| `S1-177844289` | krystal s investments | saratoga springs 2244 hunter d | `S3-424175983` | krystal s investments |  | 0.9614 | NAME_TOKEN, FREQ_TOKEN_NUMERIC_COMPOSITE, EXACT_CORE_NAME |
| `S1-270087351` | suryanth solutions | 10111 sobha aspire apt nagasan | `S2-49240605` | suryantha solutions company | 10111 sobha aspire apt nagasan | 0.9992 | POSTAL_CODE |
| `S1-912572298` | malia s holdings clinic | 4654 325 in logansport | `S3-686987455` | malia s holdings cici llc | 4657 4661 325 logansport india | 0.9889 | NAME_TOKEN |
| `S1-476254889` | num select south | tx killeen 5112 siltstone loop | `S3-201347049` | menendez select south co | 5112 siltstone loop killeen te | 0.9811 | FREQ_TOKEN_NUMERIC_COMPOSITE |
| `S1-513400646` | saravana on private limited | 3 391 a paramankurichi main ro | `S2-823279459` | saravana solutions private | 3 412 a paramankurichi main ro | 0.9539 | NAME_TOKEN |
| `S1-968800875` | regional foundation v | 100 samaritan way unit 354 fai | `S3-400213825` | regional regional foundation v | 354 tennessee 121 123 samarita | 0.9372 | FREQ_TOKEN_NUMERIC_COMPOSITE |
| `S1-413862512` | vantelle plutonian | 810 5th street unit 2 tempe az | `S2-881643124` | vantello plutonian | 810 5th st tempe az | 0.9994 | NAME_TOKEN |
| `S1-84102169` | colonial capital | 1678 475 sullivan in | `S2-22118955` | colonial capital inc | 1679 b 475 sullivan in | 0.9489 | FREQ_TOKEN_NUMERIC_COMPOSITE, EXACT_CORE_NAME |
| `S1-475750986` | agni solutions private limited | s o ubaidullah rani laxmi bai  | `S2-555252335` | agni consultancy private li mi | no 3 s o ubaidullah rani laxmi | 0.9420 | NAME_TOKEN |
| `S1-380738770` | vision industries group pvt lt | door no 32 2383 d2 second floo | `S2-169559862` | vision industries group indust | ക രള no 32 2383 d7 second floo | 0.9644 | FREQ_TOKEN_NUMERIC_COMPOSITE |
| `S1-390812473` | merriment and sons private lim | ii floor no 5 radha krishnan s | `S2-233266704` | merriment and sons l l p | tamil nadu door no d 30 ii flo | 0.9399 | NAME_TOKEN, FREQ_TOKEN_NUMERIC_COMPOSITE |
| `S1-557411773` | premier freight works | 412 306 eden roc circle virgin | `S3-656607250` | premier freight works ltd | 412 308 eden roc cir virginia  | 0.9353 | FREQ_TOKEN_NUMERIC_COMPOSITE, EXACT_CORE_NAME |
| `S1-760138211` | partap hospital | 29 1419 1 kakatiya nagar nered | `S3-585099285` | hospital partap llp | 29 1419 3 kakatiya nagar nered | 0.9838 | NAME_TOKEN, FREQ_TOKEN_NUMERIC_COMPOSITE |
| `S1-768044770` | fpg producer private limited | a 240 kh no 264 gf village cha | `S2-339453006` | fp producer private li mited | a b3 247 kh no 264 gf village  | 0.9819 | FREQ_TOKEN_NUMERIC_COMPOSITE |
| `S1-124034271` | ermengarde s eye care pllc | 418 mcclellan avenue nashville | `S2-200596002` | ermengarde s eye care partners | 425 mcctellan ave nashville tn | 0.9361 | NAME_TOKEN |
| `S1-123445423` | finiify allied technology l l  | 295 vaiden louisburg nc | `S3-65077347` | finiify allied technology co | 00302 vaiden louisburg north c | 0.9476 | NAME_TOKEN, FREQ_TOKEN_NUMERIC_COMPOSITE |
| `S1-481394893` | caswell star | 4627 vista drive cottonwood az | `S3-376816953` | caswell star inc |  | 0.9465 | NAME_TOKEN, FREQ_TOKEN_NUMERIC_COMPOSITE, EXACT_CORE_NAME |

### Representative False Negatives (Predicted Match = 0, Ground Truth = 1)
| S1 ID | S1 Name | S1 Address | Candidate ID | Candidate Name | Candidate Address | Prob | Blocking Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `S1-981440385` | angad multitrade ltd | building no 37 4 first floor o | `S2-374912494` | angad multitrade limited | no 439 building no 37 4 first  | 0.9064 | NAME_TOKEN, EXACT_CORE_NAME |
| `S1-846529766` | pocius herrera and lima connec | 4831 beaumont street simi vall | `S3-959205448` | pocius herrera and lima inc ce |  | 0.3158 | NAME_TOKEN |
| `S1-477458226` | durgesh consulting private lim | flat no 507 fortview green ear | `S3-931784124` | durgesh consulting private ltd | no 5 07 hyderabad tg | 0.7900 | NAME_TOKEN, FREQ_TOKEN_NUMERIC_COMPOSITE, EXACT_CORE_NAME |
| `S1-368348061` | naon enterprises inc | 31481 oak ridge drive muscoda  | `S3-832808344` | naon inc partners | oak ridge dr musoda wisconsin | 0.2015 | NAME_TOKEN |
| `S1-368348061` | naon enterprises inc | 31481 oak ridge drive muscoda  | `S3-656890408` | naon i nc partners |  | 0.0056 | NAME_TOKEN |
| `S1-557464657` | doiron and alvarez token llc | 17607 lake melford avenue bowi | `S2-729358614` | doiron and afcvz token llc |  | 0.4187 | NAME_TOKEN |
| `S1-622557613` | baid retail corporation | 114 sitaram appartments plot n | `S2-904082339` | mr baid re tail corporation 51 |  | 0.0051 | NAME_TOKEN |
| `S1-536948776` | i corporate inc | 718 94 warwick ny | `S3-51327360` | l corporate inc | 718 1 2 94 warwick new york | 0.8889 | NAME_TOKEN |
| `S1-828917180` | wiggins molecular p c | 239 duckworth road pennsboro w | `S3-801453291` | wiggins molecular p c id 58953 | duckworth road pennsboro west  | 0.7213 | NAME_TOKEN |
| `S1-602775749` | urology rocky care associates | 4419 27th street unit 11 phoen | `S2-872504799` | urology rocky care associates |  | 0.2147 | FREQ_TOKEN_NUMERIC_COMPOSITE, EXACT_CORE_NAME |
| `S1-602775749` | urology rocky care associates | 4419 27th street unit 11 phoen | `S2-161271237` | urology rocky care associates  | 4419 27nd st az northeast | 0.4836 | FREQ_TOKEN_NUMERIC_COMPOSITE |
| `S1-139532667` | merchant charitable trust | mohla road village harphala po | `S2-615912556` | merchant charitable trust corp | mohla road village harphaal po | 0.7321 | NAME_TOKEN, EXACT_CORE_NAME |
| `S1-284710338` | inter india | a 2106 ace platinum block zeta | `S3-215858960` | inter idod | a 106 gautam buddha nagar dadr | 0.2438 | NAME_TOKEN |
| `S1-706514314` | dental quality associates llc | 2831 sutton avenue kettering o | `S2-810456203` | dental quality associates | sutton avenue daton oh | 0.6538 | FREQ_TOKEN_NUMERIC_COMPOSITE, EXACT_CORE_NAME |
| `S1-352234501` | clean investment services | 34 abbott square birmingham al | `S2-229544676` | clean investment services |  | 0.2225 | FREQ_TOKEN_NUMERIC_COMPOSITE, EXACT_CORE_NAME |
| `S1-543144015` | freight foods | b k dutt colony lodhi road del | `S2-284196531` | freight partners | hn 950 k 5 b k dutt colony lod | 0.5786 | FREQ_TOKEN_NUMERIC_COMPOSITE |
| `S1-757082720` | urology regional clinic | 50085 288 young az | `S2-343259555` | urology regional clinic corp | az young 0085 288 | 0.7994 | FREQ_TOKEN_NUMERIC_COMPOSITE, EXACT_CORE_NAME |
| `S1-326704888` | ear nose and throat tri state  | 7804 national service road gre | `S2-566328332` | ear nose and throat tri state  |  | 0.3543 | FREQ_TOKEN_NUMERIC_COMPOSITE, EXACT_CORE_NAME |
| `S1-284209086` | solutions basco marine corp | sudhaapartment 2ndfloor flatno | `S2-183451887` | services solutions basco co rp | sudhaapartment howrah kolkata  | 0.8762 | NAME_TOKEN |
| `S1-284209086` | solutions basco marine corp | sudhaapartment 2ndfloor flatno | `S3-925326462` | solutions bod marine corp | sudhaapartment 2ndfloor flatno | 0.3193 | NAME_TOKEN |
| `S1-306095185` | leo international school | mumbai maharashtra mumbai city | `S3-261575641` | leo school center | 104 a 1 mumbai मह र ष ट र | 0.6307 | FREQ_TOKEN_NUMERIC_COMPOSITE |
| `S1-839067903` | dustin premier spdr | 2602 hemlock avenue baltimore  | `S3-873143853` | dustin premier services | hemlock avenue baltimore maryl | 0.9115 | NAME_TOKEN, FREQ_TOKEN_NUMERIC_COMPOSITE |
| `S1-724291976` | gonzalezlopez s wholesale | ca 1235 chestnut street upland | `S3-784538650` | gonzalezlopez s wmoleile |  | 0.9127 | NAME_TOKEN |
| `S1-790887991` | agni college | 37 3 kala complex fourth avenu | `S3-120467533` | agni cbnele www agnicbnel com | 37 3 kala complex fourth avenu | 0.4945 | NAME_TOKEN |
| `S1-790887991` | agni college | 37 3 kala complex fourth avenu | `S3-931397252` | agni center | 37 3 chennai city region chenn | 0.2857 | NAME_TOKEN |

## 8. COMPUTATIONAL PERFORMANCE

End-to-end runtime and resource utilization metrics for Phase 3:

- **Candidate Generation Runtime:** 251.98 s
- **Feature Generation Runtime:** 120.07 s
- **Training Runtime (All Models):** 28.23 s
- **Inference Runtime (Validation):** 1.71 s
- **Peak Memory Utilization:** 5326.7 MB