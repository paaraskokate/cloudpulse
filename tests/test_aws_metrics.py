"""Tests for the CloudWatch adapter using mocked boto3 (no real AWS calls)."""
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd

from cloud.aws_metrics import (
    ALLOWED_STATISTICS,
    AWSQueryError,
    EmptyMetricDataError,
    fetch_history_frame,
    read_cpu,
)


class FakeCloudWatch:
    def __init__(self, datapoints=None, error=None):
        self.datapoints = datapoints or []
        self.error = error
        self.calls = []

    def get_metric_statistics(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return {"Datapoints": self.datapoints}


class FakeSession:
    def __init__(self, clock):
        self.clock = clock
        self.region_name = "ap-south-1"

    def client(self, name):
        assert name == "cloudwatch"
        return self.clock


def make_points(n=5, statistic="Average"):
    base = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
    return [
        {
            "Timestamp": base + timedelta(minutes=5 * k),
            statistic: float(20 + 10 * k),
            "Unit": "Percent",
        }
        for k in reversed(range(n))
    ]


class TestReadCpu(unittest.TestCase):
    def test_returns_sorted_real_timestamps(self):
        cw = FakeCloudWatch(datapoints=make_points())
        with patch("cloud.aws_metrics.boto3.Session", return_value=FakeSession(cw)):
            points = read_cpu("i-0abc123def4567890", hours=6)
        self.assertEqual(len(points), 5)
        stamps = [p["timestamp"] for p in points]
        self.assertEqual(stamps, sorted(stamps))
        self.assertEqual([p["cpu"] for p in points], [20.0, 30.0, 40.0, 50.0, 60.0])

    def test_query_parameters_are_correct(self):
        cw = FakeCloudWatch(datapoints=make_points())
        with patch("cloud.aws_metrics.boto3.Session", return_value=FakeSession(cw)):
            read_cpu("i-0abc123def4567890", region="ap-south-1", hours=3, period=300)
        kwargs = cw.calls[0]
        self.assertEqual(kwargs["Namespace"], "AWS/EC2")
        self.assertEqual(kwargs["MetricName"], "CPUUtilization")
        self.assertEqual(kwargs["Dimensions"], [{"Name": "InstanceId", "Value": "i-0abc123def4567890"}])
        self.assertEqual(kwargs["Period"], 300)
        self.assertEqual(kwargs["Statistics"], ["Average"])
        span = kwargs["EndTime"] - kwargs["StartTime"]
        self.assertEqual(span, timedelta(hours=3))

    def test_empty_response_raises_actionable_error(self):
        """Guide test: empty/error response -> clear message, no fabricated data."""
        cw = FakeCloudWatch(datapoints=[])  # no points at all
        with patch("cloud.aws_metrics.boto3.Session", return_value=FakeSession(cw)):
            with self.assertRaises(EmptyMetricDataError) as ctx:
                read_cpu("i-0abc123def4567890", hours=6)
        msg = str(ctx.exception)
        for phrase in ("region", "instance", "permission"):
            self.assertIn(phrase, msg.lower())

    def test_denied_permissions_raise_clear_error(self):
        import botocore.exceptions as bex

        err = bex.ClientError({"Error": {"Code": "AccessDenied", "Message": "denied"}},
                              "GetMetricStatistics")
        cw = FakeCloudWatch(error=err)
        with patch("cloud.aws_metrics.boto3.Session", return_value=FakeSession(cw)):
            with self.assertRaises(AWSQueryError) as ctx:
                read_cpu("i-0abc123def4567890")
        self.assertIn("permission", str(ctx.exception).lower())

    def test_invalid_statistic_rejected(self):
        with self.assertRaises(ValueError):
            read_cpu("i-0abc123def4567890", statistic="Median")
        self.assertNotIn("Median", ALLOWED_STATISTICS)

    def test_invalid_instance_id_rejected(self):
        with self.assertRaises(ValueError):
            read_cpu("not-an-instance")

    def test_fetch_history_frame_shape(self):
        cw = FakeCloudWatch(datapoints=make_points())
        with patch("cloud.aws_metrics.boto3.Session", return_value=FakeSession(cw)):
            df = fetch_history_frame("i-0abc123def4567890", hours=1)
        self.assertEqual(list(df.columns), ["timestamp", "cpu"])
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["timestamp"]))
        self.assertEqual(len(df), 5)


if __name__ == "__main__":
    unittest.main()
