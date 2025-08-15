# Lab 5 - Pipeline Report

## 1) Exploratory Data Analysis (EDA) — Jot notes

- Performed histograms, boxplots, and correlation heatmap to inspect distributions and multicollinearity.
- Identified missing values and numeric skew; used median imputation and log1p for skewed features in linear pipeline.
- Visualized target imbalance; used class_weight/SMOTE options for imbalance handling.
- EDA guided feature selection: dropped high-correlation pairs (corr > 0.95).

## 2) Data Preprocessing — Jot notes

- Median imputation for numeric features; drop non-numeric columns for simplicity.
- Log1p applied to skewed numeric features for linear models; standard scaling for LR via pipeline.
- Used stratified train/test split to preserve class proportions and avoid sampling bias.
- Class imbalance handled via class_weight by default; SMOTE option available (use --compare-smote).

## 3) Feature Selection — Jot notes

- Dropped features with absolute correlation > 0.95 to reduce multicollinearity.
- Used RF feature importance to pick top-k features for tree models; L1 selection for LR.
- If L1 yields no features, fallback to initial feature set to avoid empty models.
- Selected features saved in report and used for PSI comparisons.

## 4) Hyperparameter Tuning — Jot notes

- Used RandomizedSearchCV with stratified 5-fold CV; optimized for ROC-AUC.
- Limited n_iter for compute efficiency; used error_score=np.nan to tolerate bad combos.
- Tuned LR (C), RF (n_estimators, max_depth), XGB (if available) with simple grids.
- Saved best params for each model in the report.

## 5) Model Training — Jot notes

- Trained LogisticRegression (benchmark), RandomForest, and optionally XGBoost.
- Used stratified K-fold during tuning and final safe_fit on full train set before evaluation.
- Persisted models to outputs/models/*.pkl for later use.
- Kept pipelines for LR to ensure scaling is applied consistently.

## 6) Model Evaluation and Comparison — Jot notes

- Evaluated ROC-AUC, PR-AUC, Brier, F1, precision, recall on train and test sets.
- Produced ROC and calibration plots (train vs test) to assess discrimination and calibration.
- Compared models in a table and selected best by test ROC-AUC.
- Saved metrics in the report and figures in outputs/figures.

## 7) SHAP Values for Interpretability — Jot notes

- Computed SHAP summary plots for the best model (if shap installed) to explain feature effects.
- Used TreeExplainer for tree models and KernelExplainer fallback for others (small sample).
- Saved SHAP plots to outputs/figures for stakeholder review.
- Interpretable features help align model decisions with business/regulatory needs.

## 8) Population Stability Index (PSI) — Jot notes

- Calculated PSI for selected features (train vs test) to detect drift.
- Reported PSI values and plotted top PSI features (if plotting available).
- Suggested retraining or monitoring if PSI exceeds conservative thresholds (e.g., >0.1/0.25).
- Saved PSI table to outputs/psi_results.csv for auditing.

## 9) Challenges and Reflections — Jot notes

- Main challenges: class imbalance, potential multicollinearity, and computing resources for tuning.
- Addressed via class_weight/SMOTE option, correlation filtering, and limited RandomSearch iterations.
- Ensured robustness with safe_fit/get_probabilities and fallbacks when optional packages missing.
- Recommend monitoring PSI and periodic retraining in production.

### Model summary (test metrics)
| Model | Test ROC-AUC | Test Brier | Test F1 | Best Params | Model Path |
|---|---:|---:|---:|---|---|
| LogisticRegression | 0.911 | 0.134 | 0.239 | `{'clf__penalty': 'l2', 'clf__C': 0.0032903445623126675}` | outputs\models\logistic_regression.pkl |
| RandomForest | 0.954 | 0.022 | 0.327 | `{'clf__n_estimators': 500, 'clf__min_samples_split': 2, 'clf__max_depth': None}` | outputs\models\random_forest.pkl |
| XGBoost | 0.950 | 0.021 | 0.351 | `{'clf__subsample': 0.6, 'clf__n_estimators': 200, 'clf__max_depth': 5, 'clf__learning_rate': 0.01}` | outputs\models\xgboost.pkl |

Top PSI values:
-  Non-industry income and expenditure/revenue: PSI=0.0188
-  Working Capital/Equity: PSI=0.0170
-  Current Liability to Current Assets: PSI=0.0168
-  Current Ratio: PSI=0.0166
-  Inventory/Working Capital: PSI=0.0152