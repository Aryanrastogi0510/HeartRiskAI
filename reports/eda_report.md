# Exploratory Data Analysis & Cardiovascular Dataset Audit

> **Project:** Heart Disease Risk Prediction (Genetic Algorithm + Machine Learning + Expert System)
> **Dataset Source File:** `framingham.csv`
> **Target Clinical Endpoint:** `TenYearCHD` (10-Year Risk of Coronary Heart Disease)

---

## 1. Executive Summary & Record Accounting

- **Total Records Ingested:** 4,240
- **Total Feature Columns:** 16
- **Target Column Present:** Yes
- **Complete Clinical Records (No Missingness):** 3,658 (86.27%)
- **Records with at Least One Missing Field:** 582
- **Duplicate Rows:** 0

---

## 2. Target Class Distribution & Imbalance Risk

The dataset presents significant clinical class imbalance:
- **Class 0 (No CHD in 10 Years):** 3,596 (84.81%)
- **Class 1 (Developed CHD in 10 Years):** 644 (15.19%)
- **Observed Imbalance Ratio:** `5.58:1 (Negative:Positive)`

> [!IMPORTANT]
> **Evaluation Risk Protocol:**
> Because negative instances predominate (~85%), raw accuracy is clinically deceptive.
> In Phase 4, model evaluation must emphasize **PR-AUC (Precision-Recall AUC)**, **ROC-AUC**, and **Sensitivity**.
> **Oversampling or SMOTE MUST NOT be performed prior to train-test splitting.**

---

## 3. Missing Value Prevalence Audit

| Feature Column | Data Type | Missing Count | Missing Percentage | Imputation Strategy Planned |
| :--- | :--- | :--- | :--- | :--- |
| `male` | `int64` | 0 | 0.0% | None (Fully observed) |
| `age` | `int64` | 0 | 0.0% | None (Fully observed) |
| `education` | `float64` | 105 | 2.48% | Mode / Most-frequent (Pipeline fit on train only) |
| `currentSmoker` | `int64` | 0 | 0.0% | None (Fully observed) |
| `cigsPerDay` | `float64` | 29 | 0.68% | Median / Iterative (Pipeline fit on train only) |
| `BPMeds` | `float64` | 53 | 1.25% | Mode / Most-frequent (Pipeline fit on train only) |
| `prevalentStroke` | `int64` | 0 | 0.0% | None (Fully observed) |
| `prevalentHyp` | `int64` | 0 | 0.0% | None (Fully observed) |
| `diabetes` | `int64` | 0 | 0.0% | None (Fully observed) |
| `totChol` | `float64` | 50 | 1.18% | Median / Iterative (Pipeline fit on train only) |
| `sysBP` | `float64` | 0 | 0.0% | None (Fully observed) |
| `diaBP` | `float64` | 0 | 0.0% | None (Fully observed) |
| `BMI` | `float64` | 19 | 0.45% | Median / Iterative (Pipeline fit on train only) |
| `heartRate` | `float64` | 1 | 0.02% | Median / Iterative (Pipeline fit on train only) |
| `glucose` | `float64` | 388 | 9.15% | Median / Iterative (Pipeline fit on train only) |
| `TenYearCHD` | `int64` | 0 | 0.0% | None (Fully observed) |

> [!NOTE]
> **Data Leakage Safeguard:** Missing values must never be imputed across the entire dataset.
> All imputers will be encapsulated inside `sklearn.pipeline.Pipeline` fitted solely on the training fold.

---

## 4. Numerical Feature Distributions

| Feature | Mean | Std Dev | Min | 25% | Median | 75% | Max | Skewness |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `age` | 49.58 | 8.57 | 32.0 | 42.0 | 49.0 | 56.0 | 70.0 | 0.23 |
| `cigsPerDay` | 9.01 | 11.92 | 0.0 | 0.0 | 0.0 | 20.0 | 70.0 | 1.25 |
| `totChol` | 236.7 | 44.59 | 107.0 | 206.0 | 234.0 | 263.0 | 696.0 | 0.87 |
| `sysBP` | 132.35 | 22.03 | 83.5 | 117.0 | 128.0 | 144.0 | 295.0 | 1.15 |
| `diaBP` | 82.9 | 11.91 | 48.0 | 75.0 | 82.0 | 90.0 | 142.5 | 0.71 |
| `BMI` | 25.8 | 4.08 | 15.54 | 23.07 | 25.4 | 28.04 | 56.8 | 0.98 |
| `heartRate` | 75.88 | 12.03 | 44.0 | 68.0 | 75.0 | 83.0 | 143.0 | 0.64 |
| `glucose` | 81.96 | 23.95 | 40.0 | 71.0 | 78.0 | 87.0 | 394.0 | 6.21 |

---

## 5. Categorical Feature Distributions

| Feature | Distinct Levels | Frequencies | Mode |
| :--- | :--- | :--- | :--- |
| `male` | 2 | `0: 2,420, 1: 1,820` | 0.0 |
| `education` | 4 | `1.0: 1,720, 2.0: 1,253, 3.0: 689, 4.0: 473` | 1.0 |
| `currentSmoker` | 2 | `0: 2,145, 1: 2,095` | 0.0 |
| `BPMeds` | 2 | `0.0: 4,063, 1.0: 124` | 0.0 |
| `prevalentStroke` | 2 | `0: 4,215, 1: 25` | 0.0 |
| `prevalentHyp` | 2 | `0: 2,923, 1: 1,317` | 0.0 |
| `diabetes` | 2 | `0: 4,131, 1: 109` | 0.0 |

---

## 6. Physiological Plausibility & Clinical Anomaly Screening

Physiological validation bounds were audited against standard human clinical ranges:

| Biomarker | Plausible Range | Out-of-Bounds Records | Clinical Note |
| :--- | :--- | :--- | :--- |
| `age` | `[18.0, 100.0]` | 0 | All observed values within physiological bounds |
| `cigsPerDay` | `[0.0, 100.0]` | 0 | All observed values within physiological bounds |
| `totChol` | `[80.0, 700.0]` | 0 | All observed values within physiological bounds |
| `sysBP` | `[60.0, 300.0]` | 0 | All observed values within physiological bounds |
| `diaBP` | `[40.0, 200.0]` | 0 | All observed values within physiological bounds |
| `BMI` | `[10.0, 70.0]` | 0 | All observed values within physiological bounds |
| `heartRate` | `[30.0, 220.0]` | 0 | All observed values within physiological bounds |
| `glucose` | `[30.0, 500.0]` | 0 | All observed values within physiological bounds |

**Behavioral Inconsistencies Detected:**
- Non-smoker (`currentSmoker == 0`) with `cigsPerDay > 0`: 0 records
- Smoker (`currentSmoker == 1`) with `cigsPerDay == 0`: 0 records

---

## 7. Correlation Analysis with 10-Year CHD Target

Pearson correlations between numerical clinical predictors and the `TenYearCHD` target:

| Numerical Predictor | Correlation (r) with `TenYearCHD` | Clinical Interpretation |
| :--- | :--- | :--- |
| `age` | `+0.2254` | Moderate positive association |
| `sysBP` | `+0.2164` | Moderate positive association |
| `diaBP` | `+0.1451` | Mild positive association |
| `glucose` | `+0.1256` | Mild positive association |
| `totChol` | `+0.0824` | Mild positive association |
| `BMI` | `+0.0753` | Mild positive association |
| `cigsPerDay` | `+0.0578` | Mild positive association |
| `heartRate` | `+0.0229` | Weak association |

---

## 8. Visual Artifacts Generated

The following academic figures have been rendered to `reports/figures/`:
1. `target_class_distribution.png`: Binary class distribution illustrating class imbalance.
2. `missing_values_summary.png`: Prevalence of missing data across clinical biomarkers.
3. `correlation_heatmap.png`: Full pairwise Pearson correlation matrix.
4. `numerical_feature_distributions.png`: Multi-panel histograms with KDE by CHD outcome.
5. `key_biomarkers_boxplot.png`: Distribution of age, blood pressure, cholesterol, and glucose by CHD outcome.

---
*(Report generated automatically by `src/data_audit.py`)*