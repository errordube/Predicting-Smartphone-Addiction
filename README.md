# Smartphone Addiction Prediction — Kaggle Playground Series S6E8

https://www.kaggle.com/competitions/playground-series-s6e8/overview

## Overview

- **Task:** Predict `addicted_label` (0/1) as a probability
- **Metric:** ROC-AUC
- **Train size:** ~691K rows
- **Test size:** ~296K rows
- **Current leaderboard score:** 0.96448 (AUC)

## Dataset

Each row represents an individual with the following features:

| Feature | Description |
|---|---|
| `age` | Age in years |
| `daily_screen_time_hours` | Total daily screen time |
| `social_media_hours` | Daily social media usage |
| `gaming_hours` | Daily gaming usage |
| `work_study_hours` | Daily work/study time |
| `sleep_hours` | Daily sleep duration |
| `notifications_per_day` | Number of notifications received |
| `app_opens_per_day` | Number of times apps are opened |
| `weekend_screen_time` | Screen time on weekends |
| `gender` | Categorical |
| `stress_level` | Categorical (Low/Medium/High) |
| `academic_work_impact` | Categorical (Yes/No) |

All numeric features have meaningful missingness (4–19% missing depending on column); missingness itself was checked and found **not** predictive of the target.

## Approach

- **Model:** LightGBM (`LGBMClassifier`, binary objective, AUC eval metric)
- **Categoricals:** passed natively as pandas `category` dtype (no manual encoding)
- **Missing values:** handled natively by LightGBM's NaN routing — no imputation
- **Validation:** stratified hold-out split with early stopping to select boosting rounds, then refit on full training data
- **Feature engineering tried:** screen-time ratios, weekend/weekday deltas, notification-to-open ratios — none improved AUC over the raw features, so the final model uses the raw feature set

### Feature importance (top drivers)

`daily_screen_time_hours` and `weekend_screen_time` dominate, followed by `social_media_hours`, `work_study_hours`, and `gaming_hours`. `gender`, `stress_level`, and `academic_work_impact` carry very little signal.


## Results

| Version | Local val AUC | Public LB |
|---|---|---|
| LightGBM, raw features, single seed | 0.9624 | 0.96448 |
