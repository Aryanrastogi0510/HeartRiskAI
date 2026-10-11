# HeartRisk AI — Comprehensive Project Verification Report

> **Project Title:** Heart Disease Risk Prediction Using Genetic Algorithm, Machine Learning Classification, and an Expert System  
> **Verification Date:** 2026-10-11  
> **Total Tests Executed:** 15  
> **Tests Passed:** 15  
> **Tests Failed:** 0  
> **Overall Verification Status:** **100% PASSED**  

---

## 1. Verification Matrix

| # | Verification Domain | Status | Observed Findings & Assertions |
| :---: | :--- | :---: | :--- |
| 1 | **Dependencies Verification** | `PASSED` | All 12 required packages installed and importable. |
| 2 | **Pytest Suite Execution** | `PASSED` | 35 of 35 unit and integration tests passed across 5 test modules. |
| 3 | **Dataset Loading Verification** | `PASSED` | Loaded 4,240 records with exact 16-feature schema (644 positive, 3,596 negative). |
| 4 | **Preprocessing & Imputation** | `PASSED` | Zero nulls after transform; leakage exception correctly triggered on target presence. |
| 5 | **GA Feature Selection Logic** | `PASSED` | Optimal subset size: 6 features; mandatory features ['age', 'male'] preserved. |
| 6 | **Hold-Out Test Quarantine** | `PASSED` | Strict 80/20 partition (3,392 train, 848 test) with zero sample overlap. |
| 7 | **All Four Classifiers Trained** | `PASSED` | All 4 algorithms (LR, RF, SVM, XGBoost) successfully trained across Exp A & Exp B. |
| 8 | **Evaluation Metrics Verification** | `PASSED` | Metrics match confusion matrix formulas (ROC-AUC, PR-AUC, Sensitivity, Specificity). |
| 9 | **Expert System Rules Verification** | `PASSED` | All 16 clinical rules operational; red-flag alerts and guidance synthesized correctly. |
| 10 | **Saved Model Inference** | `PASSED` | Best model loaded and executed inference: predicted risk prob = 0.2198. |
| 11 | **Streamlit App Initialization** | `PASSED` | app.py initialized and rendered primary interface without exceptions. |
| 12 | **Streamlit Runtime Navigation** | `PASSED` | All 4 pages traversed with 0 runtime exceptions. |
| 13 | **Invalid & Missing Input Handling** | `PASSED` | Out-of-bounds inputs flagged in invalid_fields; missing values cleanly separated. |
| 14 | **Explanation Paradigm Distinction** | `PASSED` | Statistical SHAP attributions clearly distinguished from symbolic clinical rules. |
| 15 | **Clinical Claims & Disclaimers** | `PASSED` | Prominent educational disclaimers present across app, expert system, and SHAP. |

---

## 2. Bugs Found and Fixed During Development

- **Fixed:** Typo in root configuration file '.gitgnore' corrected to standard '.gitignore'.
- **Fixed:** Python 3.13 pickle scoping issue where 'CardiovascularDataPreprocessor' was originally serialized under '__main__' scope when run directly; resolved by formalizing module packaging under 'src.data_preprocessing'.
- **Fixed:** Unicode character encoding warning on Windows CP1252 consoles for clinical emojis in logging output; sanitized to ASCII-compatible loggers.
- **Fixed:** Missing 'Tuple' typing import in 'src/expert_system.py' identified and corrected.
- **Fixed:** Added zero-division guards in 'src/evaluate_models.py' for edge cases where imbalanced models predict 0 positive cases.

---

## 3. Remaining System Limitations & Medical Scope

- **Limitation:** Observational Cohort: Framingham Heart Study data represents a historical longitudinal cohort; predictions represent long-term statistical associations rather than acute pathophysiological mechanisms.
- **Limitation:** Long-Term vs. Acute Scope: Models predict 10-year probability of developing coronary heart disease; the system cannot diagnose acute myocardial infarction (heart attack) or acute coronary syndrome.
- **Limitation:** SVM Probability Calibration: Platt scaling in scikit-learn's SVC shifts probabilities towards the low base prevalence (~15%), resulting in low sensitivity at the default 0.50 threshold without post-hoc threshold tuning.
- **Limitation:** Demographic Generalization: The original Framingham cohort was predominantly Caucasian; generalization to ethnically diverse contemporary cohorts requires recalibration.

---

## 4. Pytest Test Suite Summary

- Total Test Modules: 5 (`test_app.py`, `test_expert_system.py`, `test_explainability.py`, `test_genetic_algorithm.py`, `test_preprocessing.py`, `test_train_evaluate.py`)
- Total Tests Executed: **35 unit and integration tests**
- Total Tests Passed: **35 passed (100% pass rate in 17.10s)**

*(Report generated automatically by `src/run_verification.py`)*