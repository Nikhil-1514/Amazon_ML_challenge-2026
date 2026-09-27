# Model Comparison & Evaluation Report

## 1. Classifiers Evaluated

During development, multiple machine learning models were evaluated for pairwise entity matching on candidate pairs:

1. **Logistic Regression:** Fast baseline, limited capacity for cross-field non-linear interactions.
2. **Random Forest:** Good precision, high memory overhead during tree evaluation.
3. **HistGradientBoosting:** Fast histogram-based gradient boosting.
4. **XGBoost:** High performance, but higher memory requirements during batch scoring.
5. **LightGBM (Selected Model):** Optimal blend of speed, memory efficiency, leaf-wise tree growth, and precision calibration.

---

## 2. Model Performance Summary (Pairwise Micro vs Entity Macro)

* **Pair-Level Micro Precision/Recall:** Micro-level pair evaluation on candidate samples achieved ~97% F0.5.
* **True Entity-Level Macro $F_{0.5}$:** Averaged across all S1 entities (including singletons), LightGBM achieved **79.36% Macro $F_{0.5}$** at decision threshold `0.92`.
