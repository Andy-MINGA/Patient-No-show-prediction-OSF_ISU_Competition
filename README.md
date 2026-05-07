# 🏥 Patient No-Show Prediction — OSF HealthCare × ISU Data Science Competition

> **Binary classification · ROC-AUC · Stacking Ensemble · LightGBM · XGBoost · CatBoost**

---

## 📌 Table of Contents

1. [The Problem & Why It Matters](#1-the-problem--why-it-matters)
2. [Competition Overview](#2-competition-overview)
3. [Dataset](#3-dataset)
4. [Solution Architecture](#4-solution-architecture)
5. [Feature Engineering](#5-feature-engineering)
6. [Feature Importance](#6-feature-importance)
7. [Modeling Pipeline](#7-modeling-pipeline)
8. [Results](#8-results)
9. [Key Design Decisions](#9-key-design-decisions)
10. [Repository Structure](#10-repository-structure)
11. [How to Run](#11-how-to-run)

---

## 1. The Problem & Why It Matters

Every year, **millions of scheduled medical appointments go unfulfilled** — patients who book a slot and simply never show up, without cancelling in advance. This is not a minor inconvenience. In a healthcare system already strained by resource constraints, a missed appointment means:

- A physician's time wasted — a time slot that could have been given to another patient in need
- Increased healthcare costs passed on to payers and patients
- Delayed care for sick patients who could not get an earlier slot
- Disrupted clinical workflows and staff scheduling

For OSF HealthCare, a major regional health system in Illinois, reducing no-show rates has direct and measurable impact on patient outcomes and operational efficiency. **If we can predict, before appointment day, which patients are likely to no-show — we can intervene.** That intervention might be a reminder call, a double-booking strategy, or a proactive outreach program. The model does not replace the clinician's judgment; it sharpens where to focus limited resources.

This is a high-stakes, real-world machine learning problem with genuine healthcare consequences.

---

## 2. Competition Overview

| Detail | Description |
|--------|-------------|
| **Sponsor** | OSF HealthCare & Illinois State University (ISU) |
| **Type** | Supervised binary classification |
| **Target variable** | `NO_SHOW_FLG` — `1` if patient did not arrive and did not cancel in advance; `0` otherwise |
| **Evaluation metric** | **ROC-AUC** (Area Under the Receiver Operating Characteristic Curve) |
| **Unique identifier** | `ID` column — required in final submission |
| **Predictors** | 20 categorical features |
| **Class imbalance** | ~5% no-show rate — a highly imbalanced binary classification task |

The competition challenged participants to apply data science, predictive modeling, and rigorous evaluation techniques to a real clinical dataset. Participants were not given raw patient data — all variables were pre-categorized and anonymized by OSF HealthCare.

---

## 3. Dataset

The dataset contains **20 categorical predictor columns** derived from appointment and patient records. All variables were pre-binned into ordinal or nominal categories per the competition metadata. Key variable groups include:

| Category | Variables |
|----------|-----------|
| **Patient history** | `PATIENT_NOSHOWRATE_CATEGORY`, `PATIENT_AVG_APPT2DOC_CATEGORY` |
| **Demographics** | `AGE_CATEGORY`, `PATIENT_MARITAL_STATUS_CATEGORY`, `PATIENT_EMPLOYMENT_STATUS_CATEGORY` |
| **Appointment logistics** | `DAYS_BETWEEN_CATEGORY`, `LENGTH`, `VISIT_TYPE`, `BLOCK_CODE` |
| **Temporal** | `MONTH_CODE`, `DAY_OF_WEEK_CODE`, `HOUR_CODE` |
| **Provider signals** | `PROV_NOSHOWRATE_CATEGORY`, `PROV_AVG_APPT2DOC_CATEGORY` |
| **Department signals** | `DEPT_NOSHOW_RATE_CATEGORY`, `DEPT_AVG_APPT2DOC_CATEGORY`, `DEPT_AVG_ROOM2DOC_CATEGORY` |
| **Digital engagement** | `MYCHART_STATUS` (patient portal usage) |

> **Note on class imbalance:** With only ~5% of appointments being no-shows, standard accuracy is a misleading metric. ROC-AUC is the correct measure here because it evaluates discrimination across all classification thresholds — a model that predicts 0 for every row would score 100% accuracy but 0.50 AUC.

---

## 4. Solution Architecture

The solution is a **two-layer stacking ensemble** built entirely without data leakage:

```
┌─────────────────────────────────────────────────────────────┐
│                        LAYER 1 — BASE MODELS                 │
│                                                               │
│   LightGBM ──┐                                               │
│   XGBoost  ──┤                                               │
│   CatBoost ──┼──► Out-of-Fold (OOF) Predictions             │
│   Random   ──┤    (15-fold Stratified K-Fold)                │
│   Forest   ──┘                                               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                       LAYER 2 — META-LEARNER                 │
│                                                               │
│   XGBoost on OOF predictions + polynomial interactions       │
│   (shallow: max_depth 2–4 to prevent overfitting)            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   FINAL BLEND (50 / 50)                      │
│                                                               │
│   0.50 × Meta-learner predictions                            │
│ + 0.50 × Weighted average (weights = each model's OOF AUC)  │
└─────────────────────────────────────────────────────────────┘
```

### Why This Architecture?

**Out-of-Fold predictions** prevent meta-learner leakage. If we trained base models on the full training set and fed their predictions to a meta-learner, it would see predictions made on data it was trained on — a form of leakage that produces an overconfident second layer. OOF predictions ensure every sample is predicted by a model that never saw it, matching the generalization condition of the test set.

**Diversity of base models** is critical. LightGBM, XGBoost, and CatBoost all use gradient boosting but differ in how they handle splits, regularization, and categorical features. Random Forest uses bagging instead of sequential residual fitting. Because they make different errors, combining them reduces overall prediction variance.

**The 50/50 final blend** hedges between the meta-learner (which captures non-linear combinations of base model signals) and the weighted average (which is more robust and resistant to meta-learner overfitting).

---

## 5. Feature Engineering

All 20 raw features are categorical strings. The preprocessing pipeline — applied **inside every cross-validation fold** to prevent leakage — performs three encoding steps before feature engineering begins:

### Step 1 — Ordinal Encoding
Columns with a meaningful natural order (e.g., wait-time bands, age groups, no-show rate tiers) are encoded with explicit integer ranks using `OrdinalEncoder` with the true ordering specified. This is critical: an arbitrary factorize would assign `"Eleven+ %" = 0` and `"< One %" = 3`, destroying the signal. With proper ordering, the model can learn that higher-ranked categories carry more risk.

### Step 2 — Target Encoding
Remaining string columns (month names, hour codes, day-of-week codes) are encoded via `TargetEncoder` with Bayesian smoothing (`smoothing=10`). This replaces each category level with the smoothed mean of `NO_SHOW_FLG` across the training fold — so `"Monday"` becomes a numeric no-show probability rather than an arbitrary integer.

### Step 3 — Engineered Interaction Features

The most important features in this dataset turned out to be **interactions**, not raw variables. The top two features by LightGBM importance were both engineered:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `age_x_leadtime` | `AGE_CATEGORY × DAYS_BETWEEN_CATEGORY` | Younger patients with long scheduling gaps are disproportionately high-risk |
| `leadtime_x_composite` | `DAYS_BETWEEN_CATEGORY × composite_risk_score` | Long wait combined with high composite risk is the strongest combined signal |
| `dept_wait_x_prov_wait` | `DEPT_AVG_APPT2DOC × PROV_AVG_APPT2DOC` | Departments and providers with long wait times compound each other |
| `prov_wait_x_length` | `PROV_AVG_APPT2DOC × LENGTH` | Long appointments with slow providers signal scheduling friction |
| `composite_risk_score` | `sum(patient_risk, dept_risk, prov_risk)` | Aggregate risk when multiple dimensions are simultaneously elevated |
| `wait_x_composite_risk` | `DAYS_BETWEEN × composite_risk_score` | High-risk patients who booked far in advance |
| `pt_wait_vs_dept` | Patient wait deviation from department average | Captures whether this patient consistently waits longer than typical |
| `MONTH_CODE_freq` | Frequency of each month code in training data | Encodes how common each time slot is — rarer slots may signal unusual appointment types |

---

## 6. Feature Importance

LightGBM trained on the full training set was used to rank all features after engineering. The chart below shows the top 25 and bottom 10:

```
Top 25 Features:
------------------------------------------------------------
 1. age_x_leadtime                        2420  ██████████████████████████
 2. leadtime_x_composite                  2407  ██████████████████████████
 3. dept_wait_x_prov_wait                 2075  ██████████████████████
 4. prov_wait_x_length                    1996  █████████████████████
 5. AGE_CATEGORY                          1909  █████████████████████
 6. pt_wait_vs_dept                       1696  ██████████████████
 7. MONTH_CODE_freq                       1635  █████████████████
 8. composite_risk_score                  1609  █████████████████
 9. MONTH_CODE                            1537  ████████████████
10. HOUR_CODE                             1491  ████████████████
11. PROV_AVG_APPT2DOC_CATEGORY            1455  ███████████████
12. HOUR_CODE_freq                        1428  ███████████████
13. LENGTH                                1421  ███████████████
14. DAYS_BETWEEN_CATEGORY                 1320  ██████████████
15. room2doc_x_length                     1281  █████████████
...

Bottom 10 Features (zero or near-zero importance — dropped):
------------------------------------------------------------
  pt_x_dept_risk                           494
  leadtime_x_pt_risk                       485
  any_acute_care                           381
  PATIENT_NOSHOWRATE_CATEGORY              201
  mychart_x_noshowhistory                  160
  acute_x_noshowhistory                    104
  is_late_afternoon                          0
  is_early_morning                           0
  is_friday                                  0
  is_monday                                  0
```

**Key finding:** `PATIENT_NOSHOWRATE_CATEGORY` — theoretically the strongest clinical predictor of no-show behavior — ranked near the bottom. Investigation revealed the variable was encoded at only 2 levels (`"No History of No Show"` vs ``"Greater Than Zero %"`), which is too coarse to be meaningful. This is a real-world data science challenge: the most important domain variable was the least informative as delivered.

---

## 7. Modeling Pipeline

### Hyperparameter Tuning — Optuna with Drive Persistence

Each base model was tuned using **Optuna** (Bayesian optimization with a TPE sampler), with results saved to Google Drive after every model completes. This makes the pipeline fully resumable — if a Colab session disconnects mid-tuning, the next session loads the saved studies and skips already-completed models.

| Model | Trials | CV Folds (tuning) | Key search space |
|-------|--------|-------------------|-----------------|
| LightGBM | 60 | 5-fold stratified | `num_leaves`, `learning_rate`, `reg_alpha`, `reg_lambda`, `min_child_samples` |
| XGBoost | 60 | 5-fold stratified | `max_depth` (capped at 6), `min_child_weight`, `gamma` |
| CatBoost | 60 | 5-fold stratified | `depth`, `l2_leaf_reg`, `bagging_temperature` |
| Random Forest | 60 | 5-fold stratified | `max_depth` (capped at 15), `max_features` |

**Lessons from tuning:**
- XGBoost `max_depth > 6` consistently collapsed AUC from ~0.772 to ~0.73 — deep trees overfit badly on this imbalanced dataset
- LightGBM benefited significantly from `reg_alpha` and `reg_lambda` — without them, the model was fitting minority-class noise
- CatBoost was the strongest single model across all tuning runs (best single-model OOF AUC: **0.77667**)

### OOF Generation — 15-Fold Outer Loop

```
For each of 15 outer folds:
  ├── Fit encoders on training fold only
  ├── Apply same encoders to validation fold and test set
  ├── Train all 5 models on encoded training fold
  ├── Predict on held-out validation fold → store in OOF array
  └── Predict on test set → accumulate (divide by 15 at end)
```

The `random_state=2024` for outer folds is intentionally different from the `random_state=42` used during tuning, ensuring the validation splits used for final evaluation are independent of the splits seen during hyperparameter search.

### Meta-Learner — Shallow XGBoost

The 5 OOF arrays are stacked into a `(n_train, 5)` matrix. Polynomial features of degree 2 are added, generating all pairwise interaction terms (e.g., `lgb_pred × cat_pred`). A shallow XGBoost (`max_depth ≤ 4`) is tuned on this meta-feature matrix using 7-fold CV with 50 Optuna trials.

---

## 8. Results

| Model | OOF AUC |
|-------|---------|
| LightGBM | 0.77397 |
| XGBoost | 0.77309 |
| **CatBoost** | **0.77667** |
| Random Forest | 0.76991 |
| Extra Trees | 0.76866 |
| Meta-Learner (XGB on OOF) | 0.77664 |
| **Final Blend (50/50)** | **~0.778** |

> OOF AUC is a conservative estimate. The actual leaderboard score is typically slightly higher because the test set is drawn from the same distribution.

---

## 9. Key Design Decisions

### Preventing Data Leakage
Every encoding transformation — ordinal encoding, target encoding, feature engineering — is fitted exclusively on the training fold and applied to the validation/test fold. Leakage is the most common source of inflated competition scores that fail to generalize; this pipeline eliminates it by design.

### Handling Class Imbalance
With ~5% positive rate, the dataset is significantly imbalanced. Three strategies were applied:
- `scale_pos_weight` tuned via Optuna for LightGBM and XGBoost (upweights the minority class)
- `class_weight='balanced'` for Random Forest and Extra Trees
- Stratified K-Fold preserves the 5% no-show rate in every fold

### Persistence on Free Colab
Optuna studies are serialized with `joblib` and saved to Google Drive after each model completes. The OOF loop saves intermediate arrays after every fold. This ensures that a Colab session timeout — which is common during long training runs — never loses more than one fold of work.

### Why Not Neural Networks?
For tabular datasets with categorical predictors and no text or image inputs, gradient boosted trees consistently outperform neural networks in benchmarks. The dataset has no raw text columns that would justify an embedding-based model. GBDT models also handle ordinal categoricals and missing values natively, and are far more interpretable via feature importance — which matters in a healthcare context.

---

## 10. Repository Structure

```
├── ISU_Competition_v4_commented.ipynb   # Fully commented production notebook
├── ISU_Competition_v5.ipynb             # v5 with upgraded search spaces & new features
├── README.md                            # This file
└── sample_submission.csv                # Expected submission format (ID, NO_SHOW_FLG)
```

---

## 11. How to Run

> **Environment:** Google Colab (free tier compatible). A GPU runtime is recommended for CatBoost and XGBoost but not required.

**Step 1 — Open the notebook in Colab**
Upload `ISU_Competition_v4_commented.ipynb` to Google Colab.

**Step 2 — Install dependencies (Cell 1)**
```python
!pip install category_encoders optuna catboost -q
```

**Step 3 — Mount Google Drive (Cell 2)**
Studies are saved to `MyDrive/ISU_Competition_v4/`. The pipeline auto-resumes from Drive on subsequent sessions.

**Step 4 — Upload data (Cell 3)**
Upload `train.csv` and `test.csv` when prompted.

**Step 5 — Run all cells sequentially**
Each cell prints progress. The full pipeline (tuning + OOF + stacking) takes approximately 4–6 hours on free Colab CPU, or 1–2 hours with a T4 GPU.

**Step 6 — Download submission**
Cell 11 generates `submission_v4_final.csv` and triggers a browser download automatically.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange)
![CatBoost](https://img.shields.io/badge/CatBoost-1.x-yellow)
![Optuna](https://img.shields.io/badge/Optuna-3.x-purple)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-red?logo=scikit-learn)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)

---

## Author

Built for the **OSF HealthCare × Illinois State University Data Science Competition**.  
If you found this useful or have questions, feel free to reach out or open an issue.
