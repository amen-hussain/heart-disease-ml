"""
tests/test_data_cleaning.py
===========================
Unit tests for the data cleaning and feature engineering pipeline.
Run with:  python -m pytest tests/  OR  python -m unittest discover tests/
"""

import sys
import os
import unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from src.data_loader   import load_data
from src.data_cleaning import clean, get_feature_target_split, NUMERIC_COLS

_raw_df   = None
_clean_df = None

def get_raw():
    global _raw_df
    if _raw_df is None:
        _raw_df = load_data(raw_data_dir="data/raw")
    return _raw_df

def get_clean():
    global _clean_df
    if _clean_df is None:
        _clean_df = clean(get_raw(), processed_dir="/tmp/test_processed")
    return _clean_df


class TestDataLoader(unittest.TestCase):
    def test_shape(self):
        df = get_raw()
        self.assertGreater(df.shape[0], 200)
        self.assertEqual(df.shape[1], 14)

    def test_columns_present(self):
        df = get_raw()
        for col in ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"]:
            self.assertIn(col, df.columns)

    def test_age_range(self):
        df = get_raw()
        self.assertTrue(df["age"].between(20, 100).all())


class TestDataCleaning(unittest.TestCase):
    def test_no_nulls_after_clean(self):
        df = get_clean()
        self.assertEqual(df.isnull().sum().sum(), 0)

    def test_binary_target(self):
        df = get_clean()
        self.assertEqual(set(df["target"].unique()), {0, 1})

    def test_engineered_features_exist(self):
        df = get_clean()
        for col in ["age_decade","hr_reserve","bp_chol_interaction","high_oldpeak","asymptomatic_cp"]:
            self.assertIn(col, df.columns)

    def test_age_decade_range(self):
        df = get_clean()
        self.assertTrue(df["age_decade"].between(2, 9).all())

    def test_binary_flags(self):
        df = get_clean()
        for col in ["high_oldpeak","asymptomatic_cp"]:
            self.assertTrue(set(df[col].unique()).issubset({0, 1}))

    def test_numeric_cols_numeric(self):
        df = get_clean()
        for col in NUMERIC_COLS:
            self.assertTrue(pd.api.types.is_numeric_dtype(df[col]))

    def test_feature_target_split(self):
        df = get_clean()
        X, y = get_feature_target_split(df)
        self.assertNotIn("target", X.columns)
        self.assertEqual(y.name, "target")
        self.assertEqual(len(X), len(y))

    def test_class_balance(self):
        df = get_clean()
        vc = df["target"].value_counts(normalize=True)
        self.assertGreaterEqual(vc.min(), 0.20)


if __name__ == "__main__":
    unittest.main()
