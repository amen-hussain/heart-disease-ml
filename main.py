"""
main.py
=======
End-to-end pipeline runner for the Heart Disease ML project.

Run
---
    python main.py

Steps
-----
1.  Load data
2.  Clean & feature-engineer
3.  Exploratory Data Analysis
4.  Train / test split
5.  Cross-validation baseline
6.  Hyperparameter tuning + training
7.  Evaluation on test set
8.  Demo inference
"""

import sys
import os
import warnings
import time

warnings.filterwarnings("ignore")

# Ensure project root is on path when run as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader   import load_data
from src.data_cleaning import clean, get_feature_target_split
from src.eda           import run_eda
from src.train         import cross_validate_all, tune_and_train, save_best_model
from src.evaluate      import run_evaluation
from src.predict       import predict_patient, load_best_model, EXAMPLE_PATIENTS

RANDOM_STATE = 42


def banner(title: str) -> None:
    width = 60
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def main() -> None:
    start = time.time()

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    banner("STEP 1 — DATA LOADING")
    df_raw = load_data(raw_data_dir="data/raw")

    # ── Step 2: Clean ─────────────────────────────────────────────────────────
    banner("STEP 2 — DATA CLEANING & FEATURE ENGINEERING")
    df = clean(df_raw, processed_dir="data/processed")

    # ── Step 3: EDA ───────────────────────────────────────────────────────────
    banner("STEP 3 — EXPLORATORY DATA ANALYSIS")
    run_eda(df)

    # ── Step 4: Split ─────────────────────────────────────────────────────────
    banner("STEP 4 — TRAIN / TEST SPLIT")
    X, y = get_feature_target_split(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Train set : {X_train.shape[0]} samples")
    print(f"Test set  : {X_test.shape[0]} samples")
    print(f"Features  : {X_train.shape[1]}")

    # ── Step 5: Cross-validation baseline ────────────────────────────────────
    banner("STEP 5 — CROSS-VALIDATION BASELINE (default hyperparams)")
    cv_results = cross_validate_all(X_train, y_train)
    print("\nBaseline CV Results:")
    print(cv_results.to_string(index=False))

    # ── Step 6: Tune + Train ──────────────────────────────────────────────────
    banner("STEP 6 — HYPERPARAMETER TUNING & TRAINING")
    best_models = tune_and_train(X_train, y_train, X_test, y_test)

    # ── Step 7: Evaluate ──────────────────────────────────────────────────────
    banner("STEP 7 — EVALUATION ON TEST SET")
    eval_summary = run_evaluation(best_models, X_test, y_test)

    # Save the overall best model
    save_best_model(best_models, eval_summary)

    # ── Step 8: Inference demo ────────────────────────────────────────────────
    banner("STEP 8 — INFERENCE DEMO")
    model = load_best_model()
    for p in EXAMPLE_PATIENTS:
        result = predict_patient(p["data"], model=model)
        print(f"\n  {p['name']}")
        print(f"    → {'Disease Present' if result['disease_present'] else 'No Disease'} "
              f"| P={result['probability']:.1%} | {result['risk_label']}")

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    banner(f"PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print("\n  Outputs")
    print("  ──────────────────────────────────────────────")
    print("  data/raw/heart_cleveland.csv     — raw dataset")
    print("  data/processed/heart_cleaned.csv — cleaned data")
    print("  reports/figures/                 — all EDA & evaluation plots")
    print("  reports/model_comparison.csv     — metric summary")
    print("  models/                          — saved model artefacts")
    print()


if __name__ == "__main__":
    main()
