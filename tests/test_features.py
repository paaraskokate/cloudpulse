"""Unit tests for feature engineering (lags, rolling stats, no leakage)."""
import unittest

import numpy as np

from data.generate_synthetic_data import generate_series
from ml.features import (
    add_features,
    build_latest_features,
    feature_columns,
    infer_interval_minutes,
    load_series,
    make_supervised,
    min_history_rows,
)


class TestFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate_series(days=2, interval_minutes=5, seed=7)

    def test_lag_alignment_no_leakage(self):
        feat = add_features(self.df)
        # lag_1 at row i must equal cpu at row i-1, and never the current value.
        self.assertTrue(np.allclose(feat["lag_1"].to_numpy()[1:], feat["cpu"].to_numpy()[:-1]))
        row3 = feat.iloc[3]
        self.assertAlmostEqual(row3["lag_1"], self.df["cpu"].iloc[2], places=6)

    def test_rolling_features_use_past_window_only(self):
        feat = add_features(self.df)
        w = 3
        i = 20
        expected_mean = np.mean(self.df["cpu"].to_numpy()[i - 3 : i])  # strict past window
        self.assertAlmostEqual(feat[f"roll_mean_{w}"].iloc[i], expected_mean, places=6)

    def test_supervised_has_no_nan_and_is_chronological(self):
        feat = add_features(self.df)
        X, y, times, names = make_supervised(feat)
        self.assertFalse(np.isnan(X).any())
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(times), len(y))
        self.assertTrue(all(times[i] < times[i + 1] for i in range(len(times) - 1)))
        self.assertNotIn("cpu", names)
        self.assertNotIn("timestamp", names)

    def test_latest_features_shape_and_forecast_time(self):
        X, forecast_time, names = build_latest_features(self.df)
        self.assertEqual(X.shape, (1, len(names)))
        expected_t = self.df["timestamp"].iloc[-1] + np.timedelta64(5, "m")
        self.assertEqual(pd_Ts(forecast_time), expected_t)

    def test_infer_interval(self):
        self.assertEqual(infer_interval_minutes(self.df), 5.0)

    def test_min_history_enforced(self):
        small = self.df.head(min_history_rows() - 1)
        with self.assertRaises(ValueError):
            build_latest_features(small)


def pd_Ts(ts):
    import pandas as pd

    return pd.Timestamp(ts)


if __name__ == "__main__":
    unittest.main()
