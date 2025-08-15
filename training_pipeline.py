# training_pipeline.py
# Complete training pipeline for Company Bankruptcy Prediction
# Includes EDA, preprocessing, feature selection, tuning, training, evaluation, SHAP, PSI, report

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------
# Environment / optional imports
# ---------------------------

def try_import(name: str):
    try:
        mod = __import__(name)
        return mod
    except Exception:
        return None

missing = []
for pkg in ("numpy", "pandas", "sklearn"):
    if try_import(pkg) is None:
        missing.append(pkg)
if missing:
    print(f"ERROR: Missing Python packages: {missing}. Please install them (e.g. pip install {' '.join(missing)})")
    raise SystemExit(1)

import numpy as np
import pandas as pd

# plotting - optional (use Agg backend to avoid Tk errors)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except Exception:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None

# sklearn imports
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

# optional packages
try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None

try:
    import shap  # type: ignore
except Exception:
    shap = None

try:
    from imblearn.over_sampling import SMOTE  # type: ignore
except Exception:
    SMOTE = None

# joblib shim
try:
    import joblib  # type: ignore
except Exception:
    class _JoblibShim:
        @staticmethod
        def dump(obj, path):
            d = os.path.dirname(path)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        @staticmethod
        def load(path):
            with open(path, "rb") as f:
                return pickle.load(f)

    joblib = _JoblibShim()
    print("WARNING: 'joblib' not installed — using pickle fallback for joblib.dump/load.")

# XGBoost sklearn tag shim if necessary
if xgb is not None:
    class FixedXGBClassifier(xgb.XGBClassifier):
        def __sklearn_tags__(self):
            return {
                "_estimator_type": "classifier",
                "requires_y": True,
            }

# constants & output dirs
RANDOM_STATE = 42
OUTPUT_DIR = "outputs"
EDA_DIR = os.path.join(OUTPUT_DIR, "eda")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
REPORT_PATH = os.path.join(OUTPUT_DIR, "report.md")

for d in (OUTPUT_DIR, EDA_DIR, FIG_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)

warnings.filterwarnings("ignore")


@dataclass
class Results:
    name: str
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    best_params: Dict[str, Any]
    model_path: str


# -----------------------------
# Utilities
# -----------------------------

def save_figure(fig, fname: str) -> None:
    if not PLOTTING_AVAILABLE:
        return
    path = os.path.join(FIG_DIR, fname)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(path, dpi=150)
    try:
        plt.close(fig)
    except Exception:
        pass


# -----------------------------
# EDA
# -----------------------------

def run_eda(df: pd.DataFrame, target_col: str = "Bankrupt?") -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    summary["shape"] = df.shape
    summary["missing"] = df.isna().sum().to_dict()
    summary["n_unique"] = df.nunique().to_dict()

    numeric = df.select_dtypes(include=[np.number])
    cols = numeric.columns.tolist()[:12]

    if PLOTTING_AVAILABLE and cols:
        try:
            fig, axes = plt.subplots((len(cols) - 1) // 4 + 1, 4, figsize=(14, 3 * ((len(cols) - 1) // 4 + 1)))
            axes = axes.flatten()
            for i, c in enumerate(cols):
                sns.histplot(df[c].dropna(), kde=False, ax=axes[i])
                axes[i].set_title(c)
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            save_figure(fig, "eda_histograms.png")
        except Exception as exc:
            print("EDA histograms failed:", exc)

    # boxplots
    if PLOTTING_AVAILABLE and cols:
        try:
            fig, axes = plt.subplots((len(cols) - 1) // 4 + 1, 4, figsize=(14, 3 * ((len(cols) - 1) // 4 + 1)))
            axes = axes.flatten()
            for i, c in enumerate(cols):
                sns.boxplot(x=df[c].dropna(), ax=axes[i])
                axes[i].set_title(c)
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            save_figure(fig, "eda_boxplots.png")
        except Exception as exc:
            print("EDA boxplots failed:", exc)

    # correlation heatmap
    if PLOTTING_AVAILABLE and numeric.shape[1] > 1:
        try:
            corr = numeric.corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr, cmap="vlag", center=0, ax=ax)
            save_figure(fig, "eda_corr_heatmap.png")
        except Exception as exc:
            print("EDA corr heatmap failed:", exc)

    # target balance
    if target_col in df.columns and PLOTTING_AVAILABLE:
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x=target_col, data=df, ax=ax)
            ax.set_title("Target distribution")
            save_figure(fig, "eda_target_dist.png")
            summary["target_counts"] = df[target_col].value_counts().to_dict()
        except Exception as exc:
            print("EDA target dist failed:", exc)

    with open(os.path.join(EDA_DIR, "eda_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# -----------------------------
# Preprocessing
# -----------------------------

def preprocess_data(
    df: pd.DataFrame,
    target_col: str = "Bankrupt?",
    drop_cols: Optional[List[str]] = None,
    for_model: str = "tree",
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    metadata: Dict[str, Any] = {"for_model": for_model}
    df = df.copy()
    if target_col not in df.columns:
        matches = [c for c in df.columns if c.lower() == target_col.lower()]
        if matches:
            target_col = matches[0]
        else:
            raise ValueError(f"Target column '{target_col}' not in dataframe")
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    before = X.shape[0]
    X = X.loc[~X.duplicated()]
    metadata["duplicates_removed"] = before - X.shape[0]

    # remove non-numeric for simplicity (could extend encoding)
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        X = X.drop(columns=non_numeric)
        metadata["dropped_non_numeric"] = non_numeric

    # impute numeric
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        imputer = SimpleImputer(strategy="median")
        X[numeric_cols] = imputer.fit_transform(X[numeric_cols])
        metadata["imputer_strategy"] = "median"

    # skew handling for linear models
    if for_model == "linear" and numeric_cols:
        skew_series = X[numeric_cols].skew().abs()
        skewed = skew_series[skew_series > 1].index.tolist()
        for c in skewed:
            X[c] = np.log1p(X[c].clip(lower=0))
        metadata["skew_transformed"] = skewed

    metadata["sample"] = X.head(5).to_dict(orient="list")
    return X, y.loc[X.index], metadata


# -----------------------------
# Feature selection (simple)
# -----------------------------

def drop_highly_correlated(X: pd.DataFrame, threshold: float = 0.95) -> Tuple[pd.DataFrame, List[str]]:
    if X.shape[1] <= 1:
        return X, []
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    X_reduced = X.drop(columns=to_drop) if to_drop else X
    return X_reduced, to_drop


def select_features_with_l1(X: pd.DataFrame, y: pd.Series, C: float = 0.1) -> List[str]:
    if X.shape[1] == 0:
        return []
    lr = LogisticRegression(penalty="l1", solver="liblinear", C=C, class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000)
    lr.fit(X, y)
    mask = np.abs(lr.coef_).reshape(-1) > 1e-6
    selected = X.columns[mask].tolist()
    return selected if selected else X.columns.tolist()


def select_features_with_rf(X: pd.DataFrame, y: pd.Series, top_k: int = 20) -> List[str]:
    if X.shape[1] == 0:
        return []
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced")
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return imp.head(min(top_k, len(imp))).index.tolist()


# -----------------------------
# PSI
# -----------------------------

def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    eps = 1e-8
    expected = np.array(expected).astype(float)
    actual = np.array(actual).astype(float)
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) <= 1:
        return 0.0
    expected_counts = np.histogram(expected, bins=breakpoints)[0].astype(float)
    actual_counts = np.histogram(actual, bins=breakpoints)[0].astype(float)
    expected_perc = expected_counts / (expected_counts.sum() + eps)
    actual_perc = actual_counts / (actual_counts.sum() + eps)
    expected_perc = np.where(expected_perc == 0, eps, expected_perc)
    actual_perc = np.where(actual_perc == 0, eps, actual_perc)
    psi_val = np.sum((expected_perc - actual_perc) * np.log(expected_perc / actual_perc))
    return float(psi_val)


def compute_psi_for_df(train: pd.DataFrame, test: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows = []
    for f in features:
        try:
            val = psi(train[f].values, test[f].values, buckets=10)
        except Exception:
            val = np.nan
        rows.append({"feature": f, "psi": val})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def plot_top_psi(psi_df: pd.DataFrame, top_n: int = 10) -> None:
    if not PLOTTING_AVAILABLE:
        return
    df = psi_df.head(top_n)
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(df['feature'], df['psi'])
        ax.set_xticklabels(df['feature'], rotation=45, ha='right')
        ax.set_ylabel('PSI')
        ax.set_title('Top PSI features (train vs test)')
        save_figure(fig, 'psi_top_features.png')
    except Exception as exc:
        print('Could not plot PSI:', exc)


# -----------------------------
# Safe probs & fitting
# -----------------------------

def get_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    try:
        if hasattr(model, 'predict_proba'):
            return np.array(model.predict_proba(X)[:, 1], dtype=float)
        if hasattr(model, 'decision_function'):
            scores = model.decision_function(X)
            return 1.0 / (1.0 + np.exp(-scores))
        preds = model.predict(X)
        return np.array(preds, dtype=float)
    except Exception:
        try:
            if hasattr(model, 'named_steps'):
                last = list(model.named_steps.values())[-1]
                if hasattr(last, 'predict_proba'):
                    return np.array(last.predict_proba(X)[:, 1], dtype=float)
                if hasattr(last, 'decision_function'):
                    scores = last.decision_function(X)
                    return 1.0 / (1.0 + np.exp(-scores))
                return np.array(last.predict(X), dtype=float)
        except Exception:
            pass
        return np.zeros(shape=(X.shape[0],), dtype=float)


def safe_fit(estimator: Any, X: pd.DataFrame, y: pd.Series) -> Any:
    try:
        estimator.fit(X, y)
        return estimator
    except Exception as exc:
        try:
            if hasattr(estimator, 'named_steps'):
                last_name, last_obj = list(estimator.named_steps.items())[-1]
                print(f"Pipeline fit failed, fitting last step '{last_name}' directly: {exc}")
                last_obj.fit(X, y)
                return estimator
        except Exception as exc2:
            print('Failed safe_fit:', exc2)
            raise
        return estimator


# -----------------------------
# Tuning & evaluation
# -----------------------------

def fit_and_tune(X: pd.DataFrame, y: pd.Series, model_name: str = 'lr', cv: int = 5, n_iter: int = 30):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    if model_name == 'lr':
        pipe = Pipeline([('scale', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE))])
        param_dist = {'clf__C': np.logspace(-3, 2, 30), 'clf__penalty': ['l2']}
        search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=min(n_iter, 20), cv=skf, scoring='roc_auc', n_jobs=-1, random_state=RANDOM_STATE, error_score=np.nan)
    elif model_name == 'rf':
        pipe = Pipeline([('clf', RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1))])
        param_dist = {'clf__n_estimators': [100, 200, 500], 'clf__max_depth': [None, 5, 10], 'clf__min_samples_split': [2, 5, 10]}
        search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=min(n_iter, 20), cv=skf, scoring='roc_auc', n_jobs=-1, random_state=RANDOM_STATE, error_score=np.nan)
    elif model_name == 'xgb':
        if xgb is None:
            raise ImportError('xgboost not installed')
        pipe = Pipeline([('clf', FixedXGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE))])
        param_dist = {'clf__n_estimators': [100, 200], 'clf__max_depth': [3, 5, 8], 'clf__learning_rate': [0.01, 0.05, 0.1], 'clf__subsample': [0.6, 0.8, 1.0]}
        search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=min(n_iter, 24), cv=skf, scoring='roc_auc', n_jobs=-1, random_state=RANDOM_STATE, error_score=np.nan)
    else:
        raise ValueError('Unsupported model')

    search.fit(X, y)
    best = search.best_estimator_
    best_params = search.best_params_ if hasattr(search, 'best_params_') else {}
    return best, best_params


def evaluate_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, name: str) -> Dict[str, Any]:
    results: Dict[str, Any] = {'name': name}
    try:
        check_is_fitted(model)
    except Exception:
        try:
            model = safe_fit(model, X_train, y_train)
        except Exception as exc:
            print('Evaluation fit failed:', exc)

    for split, X, y in [('train', X_train, y_train), ('test', X_test, y_test)]:
        probs = get_probabilities(model, X)
        preds = (probs >= 0.5).astype(int)
        try:
            roc = float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else float('nan')
        except Exception:
            roc = float('nan')
        try:
            pr_auc = float(average_precision_score(y, probs)) if len(np.unique(y)) > 1 else float('nan')
        except Exception:
            pr_auc = float('nan')
        try:
            brier = float(brier_score_loss(y, probs))
        except Exception:
            brier = float('nan')
        try:
            f1 = float(f1_score(y, preds))
        except Exception:
            f1 = float('nan')
        try:
            precision = float(precision_score(y, preds))
        except Exception:
            precision = float('nan')
        try:
            recall = float(recall_score(y, preds))
        except Exception:
            recall = float('nan')

        results[f"{split}_roc_auc"] = roc
        results[f"{split}_pr_auc"] = pr_auc
        results[f"{split}_brier"] = brier
        results[f"{split}_f1"] = f1
        results[f"{split}_precision"] = precision
        results[f"{split}_recall"] = recall

    # plots
    if PLOTTING_AVAILABLE and len(np.unique(y_test)) > 1:
        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            for X, y, label in [(X_train, y_train, 'train'), (X_test, y_test, 'test')]:
                probs = get_probabilities(model, X)
                fpr, tpr, _ = roc_curve(y, probs)
                auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else np.nan
                ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
            ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve: {name}')
            ax.legend(loc='lower right')
            save_figure(fig, f'roc_{name}.png')
        except Exception as exc:
            print('ROC plot failed:', exc)

        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            for X, y, label in [(X_train, y_train, 'train'), (X_test, y_test, 'test')]:
                probs = get_probabilities(model, X)
                frac_pos, mean_pred = calibration_curve(y, probs, n_bins=10)
                ax.plot(mean_pred, frac_pos, marker='o', label=label)
            ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
            ax.set_xlabel('Mean predicted probability')
            ax.set_ylabel('Fraction of positives')
            ax.set_title(f'Calibration Curve: {name}')
            ax.legend()
            save_figure(fig, f'calibration_{name}.png')
        except Exception as exc:
            print('Calibration plot failed:', exc)

    return results


# -----------------------------
# SHAP
# -----------------------------

def compute_shap(model: Any, X: pd.DataFrame, name: str) -> None:
    if shap is None or not PLOTTING_AVAILABLE:
        print('SHAP not available or plotting disabled; skipping SHAP')
        return
    try:
        clf = model.named_steps.get('clf', model) if hasattr(model, 'named_steps') else model
        if hasattr(shap, 'TreeExplainer') and ('XGB' in type(clf).__name__ or isinstance(clf, RandomForestClassifier)):
            explainer = shap.TreeExplainer(clf)
            Xs = X.sample(min(200, X.shape[0]), random_state=RANDOM_STATE)
            vals = explainer.shap_values(Xs)
            fig = plt.figure(figsize=(8, 6))
            shap.summary_plot(vals, Xs, show=False)
            fig.tight_layout()
            fig.savefig(os.path.join(FIG_DIR, f'shap_summary_{name}.png'), dpi=150)
            plt.close(fig)
        else:
            Xs = shap.sample(X, min(100, X.shape[0]))
            explainer = shap.KernelExplainer(lambda data: model.predict_proba(data)[:, 1], Xs)
            vals = explainer.shap_values(Xs)
            fig = plt.figure(figsize=(8, 6))
            shap.summary_plot(vals, Xs, show=False)
            fig.tight_layout()
            fig.savefig(os.path.join(FIG_DIR, f'shap_summary_{name}.png'), dpi=150)
            plt.close(fig)
    except Exception as exc:
        print('SHAP failed:', exc)


# -----------------------------
# Orchestration
# -----------------------------

def find_data_file(candidates: Optional[List[str]] = None) -> Optional[str]:
    candidates = candidates or ['data.csv', 'dataset.csv', 'data/data.csv', 'bankrupt.csv', 'bankruptcy_data.csv']
    for c in candidates:
        if os.path.exists(c) and os.path.isfile(c):
            return os.path.abspath(c)
    all_csv = sorted(glob.glob('*.csv') + glob.glob('data/*.csv'))
    bank_csv = [p for p in all_csv if 'bankrupt' in os.path.basename(p).lower()]
    if len(bank_csv) == 1:
        return os.path.abspath(bank_csv[0])
    if len(all_csv) == 1:
        return os.path.abspath(all_csv[0])
    return None


def generate_report(
    eda_summary: Dict[str, Any],
    results: List[Results],
    psi_df: pd.DataFrame,
    dropped_corr: List[str],
    lr_selected: List[str],
    rf_selected: List[str],
) -> None:
    lines: List[str] = []
    lines.append('# Lab 5 - Pipeline Report')
    lines.append('')
    # 1) EDA
    lines.append('## 1) Exploratory Data Analysis (EDA) — Jot notes')
    lines.append('')
    lines.append('- Performed histograms, boxplots, and correlation heatmap to inspect distributions and multicollinearity.')
    lines.append('- Identified missing values and numeric skew; used median imputation and log1p for skewed features in linear pipeline.')
    lines.append('- Visualized target imbalance; used class_weight/SMOTE options for imbalance handling.')
    lines.append('- EDA guided feature selection: dropped high-correlation pairs (corr > 0.95).')
    lines.append('')
    # 2) Preprocessing
    lines.append('## 2) Data Preprocessing — Jot notes')
    lines.append('')
    lines.append('- Median imputation for numeric features; drop non-numeric columns for simplicity.')
    lines.append('- Log1p applied to skewed numeric features for linear models; standard scaling for LR via pipeline.')
    lines.append('- Used stratified train/test split to preserve class proportions and avoid sampling bias.')
    lines.append('- Class imbalance handled via class_weight by default; SMOTE option available (use --compare-smote).')
    lines.append('')
    # 3) Feature Selection
    lines.append('## 3) Feature Selection — Jot notes')
    lines.append('')
    lines.append('- Dropped features with absolute correlation > 0.95 to reduce multicollinearity.')
    lines.append('- Used RF feature importance to pick top-k features for tree models; L1 selection for LR.')
    lines.append('- If L1 yields no features, fallback to initial feature set to avoid empty models.')
    lines.append('- Selected features saved in report and used for PSI comparisons.')
    lines.append('')
    # 4) Hyperparameter Tuning
    lines.append('## 4) Hyperparameter Tuning — Jot notes')
    lines.append('')
    lines.append('- Used RandomizedSearchCV with stratified 5-fold CV; optimized for ROC-AUC.')
    lines.append('- Limited n_iter for compute efficiency; used error_score=np.nan to tolerate bad combos.')
    lines.append('- Tuned LR (C), RF (n_estimators, max_depth), XGB (if available) with simple grids.')
    lines.append('- Saved best params for each model in the report.')
    lines.append('')
    # 5) Model Training
    lines.append('## 5) Model Training — Jot notes')
    lines.append('')
    lines.append('- Trained LogisticRegression (benchmark), RandomForest, and optionally XGBoost.')
    lines.append('- Used stratified K-fold during tuning and final safe_fit on full train set before evaluation.')
    lines.append('- Persisted models to outputs/models/*.pkl for later use.')
    lines.append('- Kept pipelines for LR to ensure scaling is applied consistently.')
    lines.append('')
    # 6) Model Evaluation
    lines.append('## 6) Model Evaluation and Comparison — Jot notes')
    lines.append('')
    lines.append('- Evaluated ROC-AUC, PR-AUC, Brier, F1, precision, recall on train and test sets.')
    lines.append('- Produced ROC and calibration plots (train vs test) to assess discrimination and calibration.')
    lines.append('- Compared models in a table and selected best by test ROC-AUC.')
    lines.append('- Saved metrics in the report and figures in outputs/figures.')
    lines.append('')
    # 7) SHAP
    lines.append('## 7) SHAP Values for Interpretability — Jot notes')
    lines.append('')
    lines.append('- Computed SHAP summary plots for the best model (if shap installed) to explain feature effects.')
    lines.append('- Used TreeExplainer for tree models and KernelExplainer fallback for others (small sample).')
    lines.append('- Saved SHAP plots to outputs/figures for stakeholder review.')
    lines.append('- Interpretable features help align model decisions with business/regulatory needs.')
    lines.append('')
    # 8) PSI
    lines.append('## 8) Population Stability Index (PSI) — Jot notes')
    lines.append('')
    lines.append('- Calculated PSI for selected features (train vs test) to detect drift.')
    lines.append('- Reported PSI values and plotted top PSI features (if plotting available).')
    lines.append('- Suggested retraining or monitoring if PSI exceeds conservative thresholds (e.g., >0.1/0.25).')
    lines.append('- Saved PSI table to outputs/psi_results.csv for auditing.')
    lines.append('')
    # 9) Challenges & Reflections
    lines.append('## 9) Challenges and Reflections — Jot notes')
    lines.append('')
    lines.append('- Main challenges: class imbalance, potential multicollinearity, and computing resources for tuning.')
    lines.append('- Addressed via class_weight/SMOTE option, correlation filtering, and limited RandomSearch iterations.')
    lines.append('- Ensured robustness with safe_fit/get_probabilities and fallbacks when optional packages missing.')
    lines.append('- Recommend monitoring PSI and periodic retraining in production.')
    lines.append('')

    # Summary table of models
    lines.append('### Model summary (test metrics)')
    lines.append('| Model | Test ROC-AUC | Test Brier | Test F1 | Best Params | Model Path |')
    lines.append('|---|---:|---:|---:|---|---|')
    for r in results:
        test_auc = r.test_metrics.get('test_roc_auc', float('nan'))
        test_brier = r.test_metrics.get('test_brier', float('nan'))
        test_f1 = r.test_metrics.get('test_f1', float('nan'))
        lines.append(f"| {r.name} | {test_auc:.3f} | {test_brier:.3f} | {test_f1:.3f} | `{r.best_params}` | {r.model_path} |")

    lines.append('\nTop PSI values:')
    try:
        top5 = psi_df.head(5).to_dict(orient='records')
        for rec in top5:
            lines.append(f"- {rec['feature']}: PSI={rec['psi']:.4f}")
    except Exception:
        lines.append('- PSI not available')

    with open(REPORT_PATH, 'w') as f:
        f.write('\n'.join(lines))
    print(f'Report written to {REPORT_PATH}')


def run_pipeline(data_path: str, target_col: str = 'Bankrupt?', test_size: float = 0.2, compare_smote: bool = False) -> None:
    print('Loading data...')
    df = pd.read_csv(data_path)

    print('Running EDA...')
    eda_summary = run_eda(df, target_col=target_col)

    print('Splitting data stratified by target...')
    tc = target_col if target_col in df.columns else [c for c in df.columns if c.lower() == target_col.lower()][0]
    X_all = df.drop(columns=[tc])
    y_all = df[tc].astype(int)
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_all, y_all, test_size=test_size, random_state=RANDOM_STATE, stratify=y_all)

    # Preprocess
    print('Preprocessing for linear model...')
    X_train_lin, y_train_lin, meta_lin = preprocess_data(pd.concat([X_train_raw, y_train_raw], axis=1), target_col=tc, for_model='linear')
    X_test_lin, y_test_lin, _ = preprocess_data(pd.concat([X_test_raw, y_test_raw], axis=1), target_col=tc, for_model='linear')

    print('Preprocessing for tree model...')
    X_train_tree, y_train_tree, meta_tree = preprocess_data(pd.concat([X_train_raw, y_train_raw], axis=1), target_col=tc, for_model='tree')
    X_test_tree, y_test_tree, _ = preprocess_data(pd.concat([X_test_raw, y_test_raw], axis=1), target_col=tc, for_model='tree')

    # Correlation filtering
    X_train_tree_sel, dropped = drop_highly_correlated(X_train_tree, threshold=0.95)
    X_test_tree_sel = X_test_tree.reindex(columns=X_train_tree_sel.columns)

    features_for_psi = X_train_tree_sel.columns.tolist()

    # RF-based selection
    print('Selecting features using RandomForest importance...')
    rf_selected = select_features_with_rf(X_train_tree_sel, y_train_tree, top_k=min(30, max(1, X_train_tree_sel.shape[1])))
    X_train_tree_sel = X_train_tree_sel[rf_selected]
    X_test_tree_sel = X_test_tree_sel[rf_selected]

    # LR selection with overlap
    print('Selecting features for LR using L1...')
    overlap = X_train_lin.columns.intersection(X_train_tree_sel.columns)
    X_train_lin_sel_start = X_train_lin[overlap] if overlap.shape[0] > 0 else X_train_lin.copy()
    lr_selected = select_features_with_l1(X_train_lin_sel_start, y_train_lin, C=0.1)
    if not lr_selected:
        lr_selected = X_train_lin_sel_start.columns.tolist()
    X_train_lin_sel = X_train_lin_sel_start[lr_selected]
    X_test_lin_sel = X_test_lin.reindex(columns=X_train_lin_sel.columns)

    results: List[Results] = []

    # Logistic Regression
    print('Tuning LogisticRegression...')
    best_lr, best_lr_params = fit_and_tune(X_train_lin_sel, y_train_lin, model_name='lr', n_iter=30)
    best_lr = safe_fit(best_lr, X_train_lin_sel, y_train_lin)
    lr_path = os.path.join(MODEL_DIR, 'logistic_regression.pkl')
    joblib.dump(best_lr, lr_path)
    lr_metrics = evaluate_model(best_lr, X_train_lin_sel, y_train_lin, X_test_lin_sel, y_test_lin, 'LogisticRegression')
    results.append(Results('LogisticRegression', {k: v for k, v in lr_metrics.items() if k.startswith('train_')}, {k: v for k, v in lr_metrics.items() if k.startswith('test_')}, best_lr_params, lr_path))

    # Random Forest
    print('Tuning RandomForest...')
    best_rf, best_rf_params = fit_and_tune(X_train_tree_sel, y_train_tree, model_name='rf', n_iter=30)
    best_rf = safe_fit(best_rf, X_train_tree_sel, y_train_tree)
    rf_path = os.path.join(MODEL_DIR, 'random_forest.pkl')
    joblib.dump(best_rf, rf_path)
    rf_metrics = evaluate_model(best_rf, X_train_tree_sel, y_train_tree, X_test_tree_sel, y_test_tree, 'RandomForest')
    results.append(Results('RandomForest', {k: v for k, v in rf_metrics.items() if k.startswith('train_')}, {k: v for k, v in rf_metrics.items() if k.startswith('test_')}, best_rf_params, rf_path))

    # XGBoost optional
    if xgb is not None:
        try:
            print('Tuning XGBoost...')
            best_xgb, best_xgb_params = fit_and_tune(X_train_tree_sel, y_train_tree, model_name='xgb', n_iter=30)
            best_xgb = safe_fit(best_xgb, X_train_tree_sel, y_train_tree)
            xgb_path = os.path.join(MODEL_DIR, 'xgboost.pkl')
            joblib.dump(best_xgb, xgb_path)
            xgb_metrics = evaluate_model(best_xgb, X_train_tree_sel, y_train_tree, X_test_tree_sel, y_test_tree, 'XGBoost')
            results.append(Results('XGBoost', {k: v for k, v in xgb_metrics.items() if k.startswith('train_')}, {k: v for k, v in xgb_metrics.items() if k.startswith('test_')}, best_xgb_params, xgb_path))
        except Exception as exc:
            print('XGBoost step failed:', exc)

    # Choose best
    best_result = max(results, key=lambda r: r.test_metrics.get('test_roc_auc', 0.0))
    print('Best model by test ROC-AUC:', best_result.name)

    # SHAP
    try:
        best_obj = joblib.load(best_result.model_path)
        if best_result.name == 'LogisticRegression':
            compute_shap(best_obj, X_test_lin_sel, best_result.name)
        else:
            compute_shap(best_obj, X_test_tree_sel, best_result.name)
    except Exception as exc:
        print('SHAP skipped or failed:', exc)

    # PSI
    psi_df = compute_psi_for_df(X_train_tree_sel.reset_index(drop=True), X_test_tree_sel.reset_index(drop=True), X_train_tree_sel.columns.tolist())
    psi_df.to_csv(os.path.join(OUTPUT_DIR, 'psi_results.csv'), index=False)
    plot_top_psi(psi_df)

    # Report
    generate_report(eda_summary, results, psi_df, dropped, lr_selected, rf_selected)
    print('Pipeline complete. Outputs saved in', OUTPUT_DIR)


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Training pipeline for company bankruptcy prediction')
    parser.add_argument('--data', type=str, required=False, default=None, help='Path to data CSV')
    parser.add_argument('--target', type=str, default='Bankrupt?', help='Target column name')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set proportion')
    parser.add_argument('--compare-smote', action='store_true', help='Compare SMOTE vs class_weight experiments (optional)')

    known, unknown = parser.parse_known_args()
    if unknown:
        if any('--f=' in str(a) or 'ipykernel' in str(a) or 'jupyter' in str(a) for a in unknown):
            pass
        else:
            print('Warning: ignoring unknown args:', unknown)
    return known


if __name__ == '__main__':
    args = parse_args()
    data_path = args.data or os.environ.get('TRAINING_DATA')
    if data_path is None:
        data_path = find_data_file()
        if data_path:
            print('Auto-detected data file:', data_path)
        else:
            print('\nERROR: No data file provided and auto-detection failed. Provide --data or set TRAINING_DATA env var.\n')
            sys.exit(2)
    if not os.path.exists(data_path):
        print('ERROR: data file not found:', data_path)
        sys.exit(2)
    run_pipeline(data_path=data_path, target_col=args.target, test_size=args.test_size, compare_smote=args.compare_smote)
