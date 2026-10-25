"""Tests for fixed / reactive / predictive policies."""
import unittest

from controller.allocator import Allocator, CapacityBounds
from controller.policy import POLICIES, fixed_policy, predictive_policy, reactive_policy

BOUNDS = CapacityBounds(1, 2)


class TestFixedPolicy(unittest.TestCase):
    def test_always_constant(self):
        for observed in (5, 50, 95):
            d = fixed_policy(observed, None, current=2, bounds=BOUNDS)
            self.assertEqual(d["policy"], "fixed")
            self.assertEqual(d["recommended"], 1)
            self.assertEqual(d["action"], "hold")


class TestReactivePolicy(unittest.TestCase):
    def test_scale_up_on_high_observed(self):
        d = reactive_policy(95, 10, current=1, bounds=BOUNDS)
        self.assertEqual(d["recommended"], 2)
        self.assertEqual(d["action"], "scale_up")

    def test_scale_down_needs_sustained_lows(self):
        d1 = reactive_policy(20, None, current=2, bounds=BOUNDS, consecutive_low=1)
        self.assertEqual(d1["action"], "hold")
        d2 = reactive_policy(20, None, current=2, bounds=BOUNDS, consecutive_low=4)
        self.assertEqual(d2["recommended"], 1)
        self.assertEqual(d2["action"], "scale_down")

    def test_scale_down_not_below_min(self):
        d = reactive_policy(10, None, current=1, bounds=BOUNDS, consecutive_low=10)
        self.assertEqual(d["recommended"], 1)
        self.assertEqual(d["action"], "hold")


class TestPredictivePolicy(unittest.TestCase):
    def test_uses_prediction(self):
        alloc = Allocator(bounds=BOUNDS)
        d = predictive_policy(20, 95, current=1, bounds=BOUNDS, allocator=alloc)
        self.assertEqual(d["recommended"], 2)
        self.assertEqual(d["action"], "scale_up")

    def test_includes_fallback_safety(self):
        alloc = Allocator(bounds=BOUNDS)
        d = predictive_policy(90, 10, current=1, bounds=BOUNDS, allocator=alloc)
        self.assertEqual(d["recommended"], 2)
        self.assertTrue(d["fallback_triggered"])

    def test_recommendation_always_within_bounds(self):
        alloc = Allocator(bounds=BOUNDS)
        for observed in (0, 25, 60, 75, 95):
            for predicted in (0, 40, 80, 100):
                d = predictive_policy(observed, predicted, current=1, bounds=BOUNDS, allocator=alloc)
                self.assertTrue(BOUNDS.min_capacity <= d["recommended"] <= BOUNDS.max_capacity)


class TestRegistry(unittest.TestCase):
    def test_all_policies_registered(self):
        self.assertEqual(set(POLICIES), {"fixed", "reactive", "predictive"})


if __name__ == "__main__":
    unittest.main()
