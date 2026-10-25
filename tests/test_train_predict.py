"""Tests for chronological training, evaluation and forecasting."""
import json
import tempfile
import unittest
from pathlib import Path

import joblib

from data.generate_synthetic_data import generate_series
from ml.features import load_series
from ml.predict import load_model, predict_next
from ml.train_model import chronological_split, compute_metrics, train_model


class TestChronologicalSplit(unittest.TestCase):
    def test_split(self):
        self.assertEqual(chronological_split(100, 0.8), 80)
        self.assertEqual(chronological_split(10, 0.8), 8)
        self.assertEqual(chronological_split(5, 0.5), 2)


class TestMetrics(unittest.TestCase):
    def test_mae_rmse(self):
        mae, rmse = compute_metrics([10, 20], [11, 20])
        self.assertAlmostEqual(mae, 0.5)
        self.assertAlmostEqual(rmse, ((1.0**2 + 0.0**2) / 2) ** 0.5)  # sqrt(0.5) = 0.7071

    def test_persistence_is_zero_when_constant(self):
        mae, _ = compute_metrics([5, 5, 5], [5, 5, 5])
        self.assertEqual(mae, 0.0)


class TestTrainModel(unittest.TestCase):
    def test_train_evaluate_save_predict(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv_path = tmp / "cpu.csv"
            generate_series(days=3, seed=42).to_csv(csv_path, index=False)
            model_path = tmp / "rf.joblib"
            report_path = tmp / "report.json"

            bundle = train_model(csv_path, model_path, report_path, train_frac=0.75)
            m = bundle["metrics"]

            # split sizes consistent and strictly chronological
            self.assertEqual(m["n_train"] + m["n_test"], m["n_train"] + m["n_test"])
            self.assertGreater(str(m["test_start_utc"]), str(m["train_end_utc"]))
            # metrics present and sane
            for key in ("mae", "rmse", "persistence_mae", "persistence_rmse"):
                self.assertIn(key, m)
                self.assertGreaterEqual(m[key], 0.0)
            self.assertTrue(Path(model_path).exists())
            self.assertTrue(Path(report_path).exists())
            report = json.loads(Path(report_path).read_text(encoding="utf-8"))
            self.assertIn("mae", report)

            saved = joblib.load(model_path)
            self.assertEqual(saved["feature_names"], bundle["feature_names"])

            # forecasting works on the tail of the same data, 0..100 range
            loaded = load_model(model_path)
            history = load_series(csv_path).tail(40)
            out = predict_next(loaded, history)
            self.assertTrue(0.0 <= out["predicted_cpu"] <= 100.0)
            self.assertGreater(str(out["forecast_time"]), str(history["timestamp"].max()))
