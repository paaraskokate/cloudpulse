"""Tests for the policy comparison runner (fixed/reactive/predictive)."""
import unittest

import pandas as pd

from controller.allocator import CapacityBounds
from evaluation.compare_policies import build_rows, compare_on_dataframe
from data.generate_synthetic_data import generate_series


class TestPolicyComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate_series(days=2, interval_minutes=5, seed=11)

    def test_build_rows_walk_forward(self):
        rows = build_rows(self.df)  # persistence forecast (no model)
        self.assertGreater(len(rows), 10)
        times = [r["timestamp"] for r in rows]
        self.assertEqual(times, sorted(times))  # chronological
        # persistence prediction equals previous observed value
        self.assertAlmostEqual(rows[3]["predicted"], rows[2]["observed"], places=6)
        # every policy sees identical rows -> fair comparison
        self.assertTrue(all(set(r) >= {"timestamp", "observed", "predicted"} for r in rows))

    def test_compare_returns_aligned_policies(self):
        decisions, summary, options = compare_on_dataframe(self.df, model_bundle=None)
        policies = set(decisions["policy"].unique())
        self.assertEqual(policies, {"fixed", "reactive", "predictive"})
        n_fixed = int((decisions["policy"] == "fixed").sum())
        for p in ("reactive", "predictive"):
            self.assertEqual(int((decisions["policy"] == p).sum()), n_fixed)
        # same timestamps for every policy at the same positions
        fixed_ts = decisions[decisions["policy"] == "fixed"]["timestamp"].tolist()
        for p in ("reactive", "predictive"):
            self.assertEqual(decisions[decisions["policy"] == p]["timestamp"].tolist(), fixed_ts)

    def test_summary_keys_and_estimates_flagged(self):
        _, summary, options = compare_on_dataframe(self.df, model_bundle=None)
        for name in ("fixed", "reactive", "predictive"):
            self.assertIn(name, summary)
            self.assertIn("estimated_cost_usd", summary[name])
            self.assertIn("instance_hours", summary[name])
            self.assertIn("avg_recommended_capacity", summary[name])
        self.assertTrue(options["cost_is_estimate"])

    def test_recommendations_stay_in_bounds_everywhere(self):
        decisions, _, _ = compare_on_dataframe(self.df, bounds=CapacityBounds(1, 2))
        self.assertTrue((decisions["recommended"] >= 1).all())
        self.assertTrue((decisions["recommended"] <= 2).all())

    def test_fixed_policy_uses_one_instance_throughout(self):
        decisions, _, _ = compare_on_dataframe(self.df, model_bundle=None)
        fixed = decisions[decisions["policy"] == "fixed"]
        self.assertEqual(set(fixed["recommended"].unique()), {1})


if __name__ == "__main__":
    unittest.main()
