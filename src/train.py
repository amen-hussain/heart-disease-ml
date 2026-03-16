"""
train.py
========
Trains, tunes, and persists multiple classification models.

Models
------
- Logistic Regression  (baseline)
- Random Forest
- Gradient Boosting
- Support Vector Machine
- K-Nearest Neighbours

Pipeline
--------
1.  Build sklearn Pipelines (scaler → model)
2.  Stratified 5-fold CV to estimate generalisation
3.  GridSearchCV for hyperparameter tuning
4.  Refit best estimator on full training set
5.  Save model artefacts to models/
"""

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_validate
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, roc_auc_score

warnings.filterwarnings("ignore")

MODELS_DIR = "models"
RANDOM_STATE = 42


# ── Model registry ────────────────────────────────────────────────────────────

def _build_pipelines() -> dict:
    """Return dict of {name: Pipeline}."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, random_state=RANDOM_STATE)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier()),
        ]),
    }


def _param_grids() -> dict:
    """Return hyperparameter search grids per model."""
    return {
        "Logistic Regression": {
            "clf__C": [0.01, 0.1, 1, 10, 100],
            "clf__solver": ["lbfgs", "liblinear"],
        },
        "Random Forest": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 5, 10],
            "clf__min_samples_split": [2, 5],
        },
        "Gradient Boosting": {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.05, 0.1, 0.2],
            "clf__max_depth": [3, 5],
        },
        "SVM": {
            "clf__C": [0.1, 1, 10],
            "clf__kernel": ["rbf", "linear"],
            "clf__gamma": ["scale", "auto"],
        },
        "KNN": {
            "clf__n_neighbors": [3, 5, 7, 11],
            "clf__weights": ["uniform", "distance"],
            "clf__metric": ["euclidean", "manhattan"],
        },
    }


# ── Cross-validation baseline ─────────────────────────────────────────────────

def cross_validate_all(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Run stratified 5-fold CV for every model (default hyperparams).
    Returns a DataFrame with mean ± std for key metrics.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "roc_auc":  "roc_auc",
        "f1":       "f1",
        "precision": "precision",
        "recall":   "recall",
    }

    pipelines = _build_pipelines()
    rows = []
    for name, pipe in pipelines.items():
        print(f"[train] CV → {name} …", end=" ", flush=True)
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        row = {"model": name}
        for metric in scoring:
            vals = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = round(vals.mean(), 4)
            row[f"{metric}_std"]  = round(vals.std(), 4)
        rows.append(row)
        print(f"AUC={row['roc_auc_mean']:.4f} ± {row['roc_auc_std']:.4f}")

    cv_df = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)
    return cv_df


# ── Hyperparameter tuning ─────────────────────────────────────────────────────

def tune_and_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    GridSearchCV for each model, refit on X_train, evaluate on X_test.
    Returns dict of {model_name: fitted_pipeline}.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipelines   = _build_pipelines()
    param_grids = _param_grids()
    best_models = {}

    os.makedirs(MODELS_DIR, exist_ok=True)

    for name, pipe in pipelines.items():
        grid = param_grids[name]
        print(f"[train] Tuning {name} …")
        gs = GridSearchCV(
            pipe, grid,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        best_models[name] = best
        print(f"  Best params : {gs.best_params_}")
        print(f"  CV AUC      : {gs.best_score_:.4f}")

        # Save model artefact
        safe_name = name.lower().replace(" ", "_")
        model_path = os.path.join(MODELS_DIR, f"{safe_name}.joblib")
        joblib.dump(best, model_path)
        print(f"  Saved → {model_path}")

    return best_models


# ── Identify best model ───────────────────────────────────────────────────────

def save_best_model(best_models: dict, eval_results: pd.DataFrame) -> None:
    """Save the top-performing model as best_model.joblib."""
    best_name = eval_results.sort_values("roc_auc", ascending=False).iloc[0]["model"]
    best_pipe  = best_models[best_name]
    path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best_pipe, path)
    # Also record which model it is
    meta = {"best_model": best_name}
    with open(os.path.join(MODELS_DIR, "best_model_meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"\n[train] Best model: '{best_name}' — saved as best_model.joblib")


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from src.data_loader import load_data
    from src.data_cleaning import clean, get_feature_target_split

    df = clean(load_data())
    X, y = get_feature_target_split(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    cv_results = cross_validate_all(X_train, y_train)
    print("\n[train] Cross-validation baseline:\n", cv_results.to_string(index=False))

    best = tune_and_train(X_train, y_train, X_test, y_test)
    print("[train] Training complete.")
