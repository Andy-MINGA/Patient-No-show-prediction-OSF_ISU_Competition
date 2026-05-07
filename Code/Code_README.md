# Code

The full notebook for this project is hosted on Kaggle.

📓 **[View Notebook on Kaggle](https://www.kaggle.com/code/andyandyminga/patient-no-show-prediction-osf-healthcare-isu)**

---

## File

| File | Description |
|------|-------------|
| `ISU_Competition_v4_commented.ipynb` | Fully commented production notebook — every cell explained in detail |

---

## Pipeline Overview

| Cell | What it does |
|------|-------------|
| 1 | Install dependencies (`category_encoders`, `optuna`, `catboost`) |
| 2 | Imports + Google Drive mount for session persistence |
| 3 | Upload `train.csv` and `test.csv` + define target and ID columns |
| 4 | Preprocessing pipeline — ordinal encoding, target encoding, feature engineering |
| 5 | Persistent Optuna tuning helper — auto-resumes from Drive on session restart |
| 6 | Optuna objective functions for LightGBM, XGBoost, CatBoost, and Random Forest |
| 7 | Run hyperparameter tuning — saves each completed study to Drive |
| 8 | Out-of-fold stacking — 5 models × 15 folds, saves arrays to Drive |
| 9 | OOF verification — prints overall AUC for each base model |
| 10 | Meta-stacker — shallow XGBoost on OOF predictions + polynomial interactions |
| 11 | Final submission file + download |
