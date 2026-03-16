"""
predict.py
==========
Inference module — load the best saved model and predict on new patient data.

Usage
-----
    python src/predict.py

Or import for use in other scripts:

    from src.predict import predict_patient, load_best_model
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

MODELS_DIR  = "models"
META_PATH   = os.path.join(MODELS_DIR, "best_model_meta.json")
MODEL_PATH  = os.path.join(MODELS_DIR, "best_model.joblib")

# Feature columns expected by the model (post feature-engineering)
FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
    "age_decade", "hr_reserve", "bp_chol_interaction", "high_oldpeak", "asymptomatic_cp",
]


def load_best_model():
    """Load the best model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No saved model found at '{MODEL_PATH}'. "
            "Run main.py (or train.py) first."
        )
    model = joblib.load(MODEL_PATH)
    model_name = "Unknown"
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            model_name = json.load(f).get("best_model", "Unknown")
    print(f"[predict] Loaded model: {model_name}")
    return model


def _engineer_features(patient: dict) -> pd.DataFrame:
    """
    Apply the same feature engineering as data_cleaning.py.

    Parameters
    ----------
    patient : raw feature dict (13 original features)

    Returns
    -------
    pd.DataFrame with all engineered columns
    """
    df = pd.DataFrame([patient])
    df["age_decade"]           = (df["age"] // 10).astype(int)
    df["hr_reserve"]           = (220 - df["age"]) - df["thalach"]
    df["bp_chol_interaction"]  = df["trestbps"] * df["chol"] / 1000.0
    df["high_oldpeak"]         = (df["oldpeak"] >= 2.0).astype(int)
    df["asymptomatic_cp"]      = (df["cp"] == 0).astype(int)
    return df[FEATURE_COLS]


def predict_patient(patient: dict, model=None) -> dict:
    """
    Predict heart disease risk for a single patient record.

    Parameters
    ----------
    patient : dict with keys matching the 13 original features
    model   : optional pre-loaded model (loads from disk if None)

    Returns
    -------
    dict with 'prediction' (0/1), 'probability', 'risk_label'
    """
    if model is None:
        model = load_best_model()

    X = _engineer_features(patient)
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]

    risk_label = (
        "HIGH RISK"   if proba >= 0.70 else
        "MODERATE RISK" if proba >= 0.40 else
        "LOW RISK"
    )

    return {
        "prediction":  int(pred),
        "probability": round(float(proba), 4),
        "risk_label":  risk_label,
        "disease_present": bool(pred),
    }


def predict_batch(df_raw: pd.DataFrame, model=None) -> pd.DataFrame:
    """
    Predict on a DataFrame of raw (13-feature) patient records.
    Returns df_raw with added columns: prediction, probability, risk_label.
    """
    if model is None:
        model = load_best_model()

    results = []
    for _, row in df_raw.iterrows():
        r = predict_patient(row.to_dict(), model=model)
        results.append(r)

    result_df = pd.DataFrame(results)
    return pd.concat([df_raw.reset_index(drop=True), result_df], axis=1)


# ── Demo ──────────────────────────────────────────────────────────────────────

EXAMPLE_PATIENTS = [
    {
        "name": "Patient A — 58yo male, high-risk profile",
        "data": {
            "age": 58, "sex": 1, "cp": 0, "trestbps": 150,
            "chol": 270, "fbs": 1, "restecg": 0,
            "thalach": 105, "exang": 1, "oldpeak": 2.8,
            "slope": 1, "ca": 2, "thal": 3,
        },
    },
    {
        "name": "Patient B — 42yo female, low-risk profile",
        "data": {
            "age": 42, "sex": 0, "cp": 2, "trestbps": 120,
            "chol": 210, "fbs": 0, "restecg": 1,
            "thalach": 172, "exang": 0, "oldpeak": 0.2,
            "slope": 2, "ca": 0, "thal": 2,
        },
    },
    {
        "name": "Patient C — 67yo male, very high-risk",
        "data": {
            "age": 67, "sex": 1, "cp": 0, "trestbps": 160,
            "chol": 286, "fbs": 0, "restecg": 0,
            "thalach": 108, "exang": 1, "oldpeak": 1.5,
            "slope": 1, "ca": 3, "thal": 3,
        },
    },
]


if __name__ == "__main__":
    print("=" * 55)
    print("  HEART DISEASE RISK PREDICTION — DEMO")
    print("=" * 55)

    try:
        model = load_best_model()
    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print("Please run:  python main.py  first.\n")
        raise SystemExit(1)

    for p in EXAMPLE_PATIENTS:
        result = predict_patient(p["data"], model=model)
        print(f"\n{'─'*55}")
        print(f"  {p['name']}")
        print(f"{'─'*55}")
        print(f"  Prediction  : {'❤️  Disease Present' if result['disease_present'] else '✅  No Disease'}")
        print(f"  Probability : {result['probability']:.1%}")
        print(f"  Risk Level  : {result['risk_label']}")

    print(f"\n{'='*55}\n")
