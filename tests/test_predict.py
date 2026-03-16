"""
tests/test_predict.py
=====================
Unit tests for the inference / feature engineering pipeline.
"""

import sys
import os
import unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from src.predict import _engineer_features, FEATURE_COLS, EXAMPLE_PATIENTS

SAMPLE = {
    "age":58,"sex":1,"cp":0,"trestbps":150,"chol":270,"fbs":1,
    "restecg":0,"thalach":105,"exang":1,"oldpeak":2.8,"slope":1,"ca":2,"thal":3
}


class TestFeatureEngineering(unittest.TestCase):
    def test_returns_dataframe(self):
        self.assertIsInstance(_engineer_features(SAMPLE), pd.DataFrame)

    def test_correct_shape(self):
        self.assertEqual(_engineer_features(SAMPLE).shape, (1, len(FEATURE_COLS)))

    def test_all_feature_cols_present(self):
        result = _engineer_features(SAMPLE)
        for col in FEATURE_COLS:
            self.assertIn(col, result.columns)

    def test_age_decade(self):
        result = _engineer_features(SAMPLE)
        self.assertEqual(result["age_decade"].iloc[0], SAMPLE["age"] // 10)

    def test_hr_reserve(self):
        result = _engineer_features(SAMPLE)
        expected = (220 - SAMPLE["age"]) - SAMPLE["thalach"]
        self.assertEqual(result["hr_reserve"].iloc[0], expected)

    def test_high_oldpeak_positive(self):
        p = {**SAMPLE, "oldpeak": 2.5}
        self.assertEqual(_engineer_features(p)["high_oldpeak"].iloc[0], 1)

    def test_high_oldpeak_negative(self):
        p = {**SAMPLE, "oldpeak": 1.0}
        self.assertEqual(_engineer_features(p)["high_oldpeak"].iloc[0], 0)

    def test_asymptomatic_cp_flag(self):
        p = {**SAMPLE, "cp": 0}
        self.assertEqual(_engineer_features(p)["asymptomatic_cp"].iloc[0], 1)

    def test_symptomatic_cp_flag(self):
        p = {**SAMPLE, "cp": 2}
        self.assertEqual(_engineer_features(p)["asymptomatic_cp"].iloc[0], 0)

    def test_no_nan(self):
        self.assertEqual(_engineer_features(SAMPLE).isnull().sum().sum(), 0)


class TestExamplePatients(unittest.TestCase):
    def test_required_keys(self):
        required = {"age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"}
        for p in EXAMPLE_PATIENTS:
            missing = required - set(p["data"].keys())
            self.assertFalse(missing, f"Missing keys: {missing}")

    def test_engineerable(self):
        for p in EXAMPLE_PATIENTS:
            result = _engineer_features(p["data"])
            self.assertEqual(result.shape, (1, len(FEATURE_COLS)))


if __name__ == "__main__":
    unittest.main()
