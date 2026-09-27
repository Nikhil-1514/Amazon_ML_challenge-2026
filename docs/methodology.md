# Business Entity Resolution Methodology

## 1. System Architecture Overview

The Business Entity Resolution pipeline processes 24M+ records across Source 1 (Reference S1), Source 2 (S2), and Source 3 (S3) through five core stages:

1. **Text Normalization & Attribute Extraction**
   - Standardization of company names, street addresses, postal codes, and regional tokens.
   - Stripping of legal suffixes (Inc., Corp., Ltd., SAS) to isolate clean core entity names.
   - Extraction of numeric street numbers, 6-digit postal codes (PIN), and locality tokens.

2. **Layered Composite Candidate Generation (Blocking)**
   - Generation of candidate pairs using multi-channel inverted indexes:
     - Exact core-name matching within country.
     - Postal code (PIN) matching within country.
     - Frequent token + street numeric composite matching.
   - Achieved 97.68% candidate pair recall while reducing search space from $1.73M \times 10M$ down to ~14 candidates per S1 entity.

3. **47-Dimensional Feature Vector Construction**
   - Computation of pairwise similarity features across name, core name, address, postal code, locality, and cross-field interaction terms.
   - Utilization of rapid C-optimized string distance metrics (Jaro-Winkler, Levenshtein, N-gram Jaccard).

4. **LightGBM Pairwise Classification**
   - Pairwise binary classification using leaf-wise gradient boosted trees trained on ground truth entity pairs.

5. **Entity-Level Decision Thresholding & Validation**
   - Scoring and threshold selection optimized against the competition's exact Macro $F_{0.5}$ metric across singletons and multi-match entities.
