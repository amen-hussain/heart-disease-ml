"""
data_cleaning.py
================
Cleans, validates, and engineers features for the Heart Disease dataset.

Steps
-----
1.  Validate schema & expected ranges
2.  Handle missing / sentinel values (UCI uses '?' encoded as NaN)
3.  Binarise multi-class target (0 = no disease, 1 = disease)
4.  Cast categorical columns to the correct dtype
5.  Impute remaining missing values
6.  Feature engineering
7.  Save processed dataset to disk
"""

import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# ── Column metadata ──────────────────────────────────────────────────────────
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC_COLS     = ["age", "trestbps", "chol", "thalach", "oldpeak"]
TARGET_COL       = "target"

# Plausible clinical ranges for numeric features (for outlier flagging)
VALID_RANGES = {
    "age":      (20, 100),
    "trestbps": (80,  220),
    "chol":     (100, 600),
    "thalach":  (60,  220),
    "oldpeak":  (0,    7),
}


# ── Cleaning helpers ──────────────────────────────────────────────────────────

def _report_missing(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("[cleaning] No missing values detected.")
    else:
        print("[cleaning] Missing values:\n", missing.to_string())


def _flag_outliers(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask: True where any numeric feature is out of range."""
    mask = pd.Series(False, index=df.index)
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            out = (df[col] < lo) | (df[col] > hi)
            if out.any():
                print(f"[cleaning] {out.sum()} outlier(s) in '{col}' "
                      f"(range {lo}–{hi}).")
            mask |= out
    return mask


def clean(df: pd.DataFrame, processed_dir: str = "data/processed") -> pd.DataFrame:
    """
    Full cleaning + feature-engineering pipeline.

    Parameters
    ----------
    df : raw DataFrame from data_loader.load_data()

    Returns
    -------
    pd.DataFrame  ready for modelling.
    """
    df = df.copy()

    # ── 1. Binarise target ───────────────────────────────────────────────────
    # Original UCI target: 0 = no disease, 1/2/3/4 = disease present
    df[TARGET_COL] = (df[TARGET_COL] > 0).astype(int)
    print(f"[cleaning] Target distribution:\n{df[TARGET_COL].value_counts().to_string()}")

    # ── 2. Replace sentinel values ───────────────────────────────────────────
    # UCI encodes missing as '?' which pandas reads as NaN automatically via
    # na_values, but float conversion may produce NaN too.
    df.replace("?", np.nan, inplace=True)

    # Force numeric types
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    _report_missing(df)

    # ── 3. Outlier reporting (we log but do NOT drop — small dataset) ────────
    _flag_outliers(df)

    # ── 4. Impute ────────────────────────────────────────────────────────────
    num_imputer = SimpleImputer(strategy="median")
    df[NUMERIC_COLS] = num_imputer.fit_transform(df[NUMERIC_COLS])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[CATEGORICAL_COLS] = cat_imputer.fit_transform(df[CATEGORICAL_COLS])

    # ── 5. Correct dtypes ────────────────────────────────────────────────────
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(int)

    # ── 6. Feature engineering ───────────────────────────────────────────────
    # Age decade bucket
    df["age_decade"] = (df["age"] // 10).astype(int)

    # Max-heart-rate reserve proxy  (predicted max HR = 220 - age)
    df["hr_reserve"] = (220 - df["age"]) - df["thalach"]

    # Blood pressure × cholesterol interaction (cardiovascular risk proxy)
    df["bp_chol_interaction"] = df["trestbps"] * df["chol"] / 1000.0

    # ST depression severity flag
    df["high_oldpeak"] = (df["oldpeak"] >= 2.0).astype(int)

    # Symptomatic chest pain flag (cp == 0 is asymptomatic, strongly predictive)
    df["asymptomatic_cp"] = (df["cp"] == 0).astype(int)

    print(f"[cleaning] Final dataset: {df.shape[0]} rows × {df.shape[1]} columns.")
    print(f"[cleaning] Engineered features: age_decade, hr_reserve, "
          "bp_chol_interaction, high_oldpeak, asymptomatic_cp")

    # ── 7. Save ──────────────────────────────────────────────────────────────
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "heart_cleaned.csv")
    df.to_csv(out_path, index=False)
    print(f"[cleaning] Saved cleaned data → {out_path}")

    return df


def get_feature_target_split(df: pd.DataFrame):
    """Return (X, y) after dropping the target column."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


if __name__ == "__main__":
    from src.data_loader import load_data
    raw = load_data()
    cleaned = clean(raw)
    print(cleaned.describe())
