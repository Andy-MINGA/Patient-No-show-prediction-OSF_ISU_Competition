# Code

The full notebook for this project is hosted on Kaggle.

📓 **[View Notebook on Kaggle](https://www.kaggle.com/code/andyandyminga/patient-no-show-prediction-osf-healthcare-isu)**

---

## Files

| File | Description |
|------|-------------|
| `ISU_Competition_v4_commented.ipynb` | Fully commented notebook — every cell explained in detail |
| `noshow_merged_v6.ipynb` | Final version used for the 0.7810 leaderboard submission |

---

## Pipeline Overview

| Cell | What it does |
|------|-------------|
| 1 | Install dependencies |
| 2 | Imports + Google Drive mount for persistence |
| 3 | Upload data + holdout split |
| 4 | Column type definitions |
| 5 | Preprocessing + feature engineering functions |
| 6 | Optuna objective functions (LGB, XGB, CatBoost, RF, ET) |
| 7 | Run hyperparameter tuning — saves to Drive after every trial |
| 8 | Out-of-fold stacking — saves to Drive after every fold |
| 9 | Meta-stacker + zero-leakage holdout evaluation |
| 10 | Calibration check |
| 11 | Final AUC summary |
| 12 | Write + download submission file |
| 13 | Feature importance report |
