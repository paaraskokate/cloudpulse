"""Tests for the bounded controller: bounds, fallback, cooldown, audit log."""
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from controller.allocator import Allocator, CapacityBounds
from controller.audit import append_row, read_rows


def utcnow():
    return datetime.now(timezone.utc)


class TestBounds(unittest.TestCase):
    def test_clamp(self):
        b = CapacityBounds(1, 2)
        self.assertEqual(b.clamp(0), 1)
        self.assertEqual(b.clamp(1), 1)
        self.assertEqual(b.clamp(2), 2)
        self.assertEqual(b.clamp(99), 2)

    def test_invalid_bounds_rejected(self):
        with self.assertRaises(ValueError):
            CapacityBounds(2, 1)
        with self.assertRaises(ValueError):
            CapacityBounds(0, 2)


class TestAllocatorDecisions(unittest.TestCase):
    def setUp(self):
        self.alloc = Allocator(bounds=CapacityBounds(1, 2))

    def test_prediction_exceeding_max_is_clamped(self):
        """Guide test: 'Force a prediction that would exceed max capacity.'
        Controller clamps recommendation to the configured maximum."""
        d = self.alloc.decide(observed_cpu=60, predicted_cpu=100, current_capacity=1,
                              data_source="test")
        self.assertEqual(d.unbounded_target, 3)
        self.assertEqual(d.recommended_capacity, 2)  # clamped to max
        self.assertTrue(d.bounds_hit)

    def test_high_observed_cpu_fallback_overrides_low_prediction(self):
        """Guide test: fallback fires even if the model predicts low demand."""
        d = self.alloc.decide(observed_cpu=88.0, predicted_cpu=15.0, current_capacity=1)
        self.assertEqual(d.recommended_capacity, 2)
        self.assertTrue(d.fallback_triggered)
        self.assertEqual(d.action, "scale_up")
        self.assertIn("fallback", d.reason.lower())

    def test_scale_down_requires_sustained_low_readings(self):
        d_hold = self.alloc.decide(observed_cpu=10, predicted_cpu=10, current_capacity=2,
                                   recent_observations=[80, 10, 10, 10])
        self.assertEqual(d_hold.action, "hold")
        d_down = self.alloc.decide(observed_cpu=10, predicted_cpu=10, current_capacity=2,
                                   recent_observations=[10, 10, 10, 10])
        self.assertEqual(d_down.action, "scale_down")
        self.assertEqual(d_down.recommended_capacity, 1)

    def test_cooldown_blocks_scale_down(self):
        recent = utcnow() - timedelta(seconds=60)
        d = self.alloc.decide(observed_cpu=10, predicted_cpu=10, current_capacity=2,
                              recent_observations=[10, 10, 10, 10],
                              last_action_at=recent, now=utcnow())
        self.assertEqual(d.action, "hold")
        self.assertTrue(d.cooldown_active)

    def test_fallback_scale_up_not_blocked_by_cooldown(self):
        recent = utcnow() - timedelta(seconds=30)
        d = self.alloc.decide(observed_cpu=90, predicted_cpu=99, current_capacity=1,
                              last_action_at=recent, now=utcnow())
        self.assertEqual(d.recommended_capacity, 2)
        self.assertEqual(d.action, "scale_up")

    def test_prediction_between_bands_holds(self):
        d = self.alloc.decide(observed_cpu=45, predicted_cpu=50, current_capacity=1)
        self.assertEqual(d.action, "hold")
        self.assertEqual(d.recommended_capacity, 1)
        self.assertFalse(d.bounds_hit)

    def test_decision_defaults_to_dry_run_mode(self):
        d = self.alloc.decide(observed_cpu=60, predicted_cpu=80, current_capacity=1)
        self.assertEqual(d.mode, "dry_run")


class TestAuditLog(unittest.TestCase):
    def test_decision_written_and_readable(self):
        alloc = Allocator(bounds=CapacityBounds(1, 2))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit_log.jsonl"
            decision = alloc.decide(observed_cpu=90, predicted_cpu=99, current_capacity=1)
            alloc.log_decision(decision, path=path)
            rows = read_rows(path)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["mode"], "dry_run")
            self.assertEqual(row["final_capacity"], 2)
            self.assertTrue(row["fallback_triggered"])
            self.assertIn("observed_cpu", row)

    def test_append_and_read_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.jsonl"
            append_row({"n": 1}, path)
            append_row({"n": 2}, path)
            rows = read_rows(path, limit=10)
            self.assertEqual([r["n"] for r in rows], [2, 1])  # newest first

    def test_import_jsonl_missing_file_returns_empty(self):
        self.assertEqual(read_rows(Path(tempfile.gettempdir()) / "does_not_exist_xyz.jsonl"), [])


if __name__ == "__main__":
    unittest.main()
