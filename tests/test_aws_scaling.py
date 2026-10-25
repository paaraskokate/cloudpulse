"""Tests for the guarded ASG controller using mocked boto3 (no live calls).

Covers the guide's safety tests: bounds clamping, dry-run (no API call),
target validation before any change, cooldown and the audit trail.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cloud.aws_scaling import (
    ASGNotFoundError,
    AutoScalingController,
    LiveScalingNotEnabledError,
    live_scaling_enabled,
)
from controller.audit import read_rows


def asg_payload(name="cloudpulse-demo-asg", desired=1, mn=1, mx=2, az="ap-south-1a"):
    return {
        "AutoScalingGroups": [
            {
                "AutoScalingGroupName": name,
                "MinSize": mn,
                "MaxSize": mx,
                "DesiredCapacity": desired,
                "AvailabilityZones": [az],
                "Instances": [{"HealthStatus": "Healthy"}],
            }
        ]
    }


class FakeAutoScaling:
    def __init__(self, groups=None):
        self.groups = groups or asg_payload()
        self.set_calls = []

    def describe_auto_scaling_groups(self, **kwargs):
        return self.groups

    def set_desired_capacity(self, **kwargs):
        self.set_calls.append(kwargs)
        # mimic AWS: the group's desired capacity actually changes
        for g in self.groups.get("AutoScalingGroups", []):
            if g["AutoScalingGroupName"] == kwargs["AutoScalingGroupName"]:
                g["DesiredCapacity"] = kwargs["DesiredCapacity"]
        return {}

    def get_caller_identity(self):
        return {"Account": "111122223333"}


class FakeSTS:
    def get_caller_identity(self):
        return {"Account": "111122223333"}


class FakeSession:
    def __init__(self, autoscaling):
        self._autoscaling = autoscaling
        self.region_name = "ap-south-1"

    def client(self, name):
        return self._autoscaling if name == "autoscaling" else FakeSTS()


class TestGuardrails(unittest.TestCase):
    def test_live_disabled_is_default(self):
        with patch.dict(os.environ, {"CLOUDPULSE_ENABLE_LIVE_SCALING": "false"}, clear=False):
            self.assertFalse(live_scaling_enabled())

    def test_live_change_blocked_without_env_flag(self):
        """Dry-run-and-blocked: env flag off -> refuse, no API call."""
        fake = FakeAutoScaling()
        with tempfile.TemporaryDirectory() as tmp:
            audit = Path(tmp) / "audit.jsonl"
            with patch.dict(os.environ, {"CLOUDPULSE_ENABLE_LIVE_SCALING": "false"}):
                with patch("cloud.aws_scaling.build_session", return_value=FakeSession(fake)):
                    ctl = AutoScalingController("cloudpulse-demo-asg", audit_path=audit)
                    with self.assertRaises(LiveScalingNotEnabledError):
                        ctl.apply_capacity(2, reason="test", live=True, confirm_live=True)
                self.assertFalse(live_scaling_enabled())
        self.assertEqual(fake.set_calls, [])

    def test_live_change_blocked_without_confirmation(self):
        fake = FakeAutoScaling()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"CLOUDPULSE_ENABLE_LIVE_SCALING": "true"}):
                with patch("cloud.aws_scaling.build_session", return_value=FakeSession(fake)):
                    ctl = AutoScalingController("cloudpulse-demo-asg", audit_path=Path(tmp))
                    with self.assertRaises(LiveScalingNotEnabledError):
                        ctl.apply_capacity(2, reason="test", live=True, confirm_live=False)
        self.assertEqual(fake.set_calls, [])

    def test_dry_run_makes_no_api_call(self):
        """Guide test: dry-run decision -> audit log says no resource changed."""
        fake = FakeAutoScaling()
        with tempfile.TemporaryDirectory() as tmp:
            audit = Path(tmp) / "audit.jsonl"
            with patch.dict(os.environ, {"CLOUDPULSE_ENABLE_LIVE_SCALING": "true"}):
                with patch("cloud.aws_scaling.build_session", return_value=FakeSession(fake)):
                    ctl = AutoScalingController("cloudpulse-demo-asg", audit_path=audit)
                    entry = ctl.apply_capacity(2, reason="dry run demo", live=False)
                    self.assertEqual(entry["result"], "dry_run_recorded")
                    self.assertEqual(entry["final_capacity"], 2)
        self.assertEqual(fake.set_calls, [])
        self.assertIn("No AWS API call", entry["detail"])

    def test_request_above_max_is_clamped_to_two(self):
        fake = FakeAutoScaling()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"CLOUDPULSE_ENABLE_LIVE_SCALING": "true"}):
                with patch("cloud.aws_scaling.build_session", return_value=FakeSession(fake)):
                    ctl = AutoScalingController("cloudpulse-demo-asg", audit_path=Path(tmp))
                    entry = ctl.apply_capacity(5, reason="too big", live=True, confirm_live=True)
                    self.assertEqual(entry["final_capacity"], 2)
                    self.assertEqual(fake.set_calls[0]["DesiredCapacity"], 2)

    def test_request_below_min_is_clamped_to_one(self):
        fake = FakeAutoScaling(groups=asg_payload(desired=2))  # starts high so 1 is a real change
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"CLOUDPULSE_ENABLE_LIVE_SCALING": "true"}):
                with patch("cloud.aws_scaling.build_session", return_value=FakeSession(fake)):
                    ctl = AutoScalingController("cloudpulse-demo-asg", audit_path=Path(tmp))
                    entry = ctl.apply_capacity(0, reason="too small", live=True, confirm_live=True)
                    self.assertEqual(entry["final_capacity"], 1)
                    self.assertEqual(fake.set_calls[0]["DesiredCapacity"], 1)

    def test_wrong_asg_name_raises_and_changes_nothing(self):
        fake = FakeAutoScaling(groups={"AutoScalingGroups": []})  # nothing exists
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"CLOUDPULSE_ENABLE_LIVE_SCALING": "true"}):
                with patch("cloud.aws_scaling.build_session", return_value=FakeSession(fake)):
                    ctl = AutoScalingController("does-not-exist", audit_path=Path(tmp))
                    with self.assertRaises(ASGNotFoundError):
                        ctl.apply_capacity(2, reason="x", live=True, confirm_live=True)
        self.assertEqual(fake.set_calls, [])


class TestAuditAndCooldown(unittest.TestCase):
    def test_audit_row_written_for_every_attempt(self):
        fake = FakeAutoScaling()
        with tempfile.TemporaryDirectory() as tmp:
            audit = Path(tmp) / "audit.jsonl"
            with patch.dict(os.environ, {"CLOUDPULSE_ENABLE_LIVE_SCALING": "true"}):
                with patch("cloud.aws_scaling.build_session", return_value=FakeSession(fake)):
                    ctl = AutoScalingController("cloudpulse-demo-asg", audit_path=audit)
                    ctl.apply_capacity(2, reason="demo", live=True, confirm_live=True)
                    try:
                        ctl.apply_capacity(2, reason="blocked", live=True, confirm_live=False)
                    except LiveScalingNotEnabledError:
                        pass
            rows = read_rows(audit)
            self.assertEqual(len(rows), 2)
            results = {r["result"] for r in rows}
            self.assertIn("applied", results)
            self.assertIn("blocked_no_confirmation", results)

    def test_cooldown_skips_second_immediate_change(self):
        fake = FakeAutoScaling(groups=asg_payload(desired=1))
        with tempfile.TemporaryDirectory() as tmp:
            audit = Path(tmp) / "audit.jsonl"
            with patch.dict(os.environ, {"CLOUDPULSE_ENABLE_LIVE_SCALING": "true"}):
                with patch("cloud.aws_scaling.build_session", return_value=FakeSession(fake)):
                    ctl = AutoScalingController("cloudpulse-demo-asg", cooldown_seconds=300,
                                                audit_path=audit)
                    first = ctl.apply_capacity(2, reason="up", live=True, confirm_live=True)
                    self.assertEqual(first["result"], "applied")
                    second = ctl.apply_capacity(1, reason="down", live=True, confirm_live=True)
                    self.assertEqual(second["result"], "cooldown_skipped")
                    self.assertEqual(len(fake.set_calls), 1)


if __name__ == "__main__":
    unittest.main()
