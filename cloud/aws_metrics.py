"""Read-only Amazon CloudWatch metrics adapter for CloudPulse.

Fetches real EC2 CPUUtilization datapoints via boto3 (read-only actions).
NEVER fabricates data: an empty or forbidden response raises a clear error
instead of silently returning synthetic rows. Callers must label the mode.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError, ProfileNotFound

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_STATISTICS = ("Average", "Minimum", "Maximum", "Sum")


class AWSQueryError(RuntimeError):
    """AWS call failed (credentials, permissions, region, throttling...)."""


class EmptyMetricDataError(AWSQueryError):
    """AWS responded successfully but there are no datapoints in the window."""


def build_session(profile_name: str | None = None, region_name: str | None = None):
    try:
        session = boto3.Session(profile_name=profile_name, region_name=region_name)
    except ProfileNotFound as exc:
        raise AWSQueryError(
            f"AWS profile '{profile_name}' not found. Run:  aws configure --profile {profile_name}"
        ) from exc
    if not session.region_name:
        raise AWSQueryError(
            "No AWS region configured — pass region=, set CLOUDPULSE_REGION / AWS_DEFAULT_REGION, "
            "or choose a default region during `aws configure`."
        )
    return session


def read_cpu(
    instance_id: str,
    region: str | None = None,
    profile: str = "cloudpulse",
    hours: float = 6,
    period: int = 300,
    statistic: str = "Average",
    end_time: datetime | None = None,
) -> list[dict]:
    """Return real, time-sorted datapoints: [{"timestamp": ..., "cpu": ...}, ...]."""
    if statistic not in ALLOWED_STATISTICS:
        raise ValueError(f"statistic must be one of {ALLOWED_STATISTICS}, got '{statistic}'.")
    instance_id = str(instance_id).strip()
    if not instance_id.startswith("i-"):
        raise ValueError("instance_id must look like 'i-0123456789abcdef0'.")
    end_time = end_time or datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=float(hours))

    session = build_session(profile_name=profile, region_name=region)
    cw = session.client("cloudwatch")
    try:
        result = cw.get_metric_statistics(
            Namespace="AWS/EC2",
            MetricName="CPUUtilization",
            Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
            StartTime=start_time,
            EndTime=end_time,
            Period=int(period),
            Statistics=[statistic],
        )
    except NoCredentialsError as exc:
        raise AWSQueryError(
            "No AWS credentials found. Run `aws configure --profile "
            f"{profile}` or attach an IAM role if running on EC2."
        ) from exc
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "Unknown")
        if "Denied" in code or "Unauthorized" in code:
            hint = "IAM permissions missing — the profile needs cloudwatch:GetMetricStatistics."
        elif "Throttl" in code:
            hint = "AWS is throttling requests — increase the period or retry shortly."
        else:
            hint = "check the region, instance ID and your AWS credentials."
        raise AWSQueryError(f"AWS API error ({code}) while reading CloudWatch: {hint}") from exc
    except BotoCoreError as exc:
        raise AWSQueryError(f"AWS client error: {exc}") from exc

    points = sorted(result.get("Datapoints", []), key=lambda p: p["Timestamp"])
    if not points:
        raise EmptyMetricDataError(
            f"CloudWatch returned no CPU datapoints for {instance_id} in the last {hours}h. Checklist:\n"
            "  1) Has the instance been running for at least ~10 minutes?\n"
            "  2) Is the region correct (metrics live in the instance's region)?\n"
            "  3) Is the instance ID exact?\n"
            "  4) Does the profile have cloudwatch:GetMetricStatistics permission?\n"
            "  5) Try a wider window (increase --hours)."
        )
    return [{"timestamp": p["Timestamp"], "cpu": float(p[statistic])} for p in points]


def fetch_history_frame(
    instance_id: str,
    region: str | None = None,
    profile: str = "cloudpulse",
    hours: float = 6,
    period: int = 300,
    statistic: str = "Average",
) -> pd.DataFrame:
    """Same data as read_cpu, as a (timestamp, cpu) DataFrame with UTC timestamps."""
    points = read_cpu(instance_id, region, profile, hours, period, statistic)
    df = pd.DataFrame(points)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)[["timestamp", "cpu"]]


def export_with_metadata(
    df: pd.DataFrame,
    out_csv: str | Path,
    *,
    instance_id: str,
    region: str | None,
    hours: float,
    period: int,
    statistic: str,
    source: str = "aws-cloudwatch",
    extra: dict | None = None,
) -> tuple[Path, Path]:
    """Write the CSV plus a sidecar metadata JSON describing the export."""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.drop_duplicates("timestamp").sort_values("timestamp").to_csv(out_csv, index=False)
    meta = {
        "source": source,
        "instance_id": instance_id,
        "region": region,
        "window_hours": hours,
        "period_seconds": int(period),
        "statistic": statistic,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(df)),
        "newest_sample_utc": str(df["timestamp"].max()) if len(df) else None,
    }
    if extra:
        meta.update(extra)
    meta_path = out_csv.with_name(out_csv.stem + "_metadata.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_csv, meta_path


def collect(
    instance_id: str,
    region: str | None,
    profile: str,
    hours: float,
    period: int,
    statistic: str,
    out_csv: str | Path,
):
    """Fetch real metrics and export CSV + metadata. Returns (df, csv_path, meta_path)."""
    df = fetch_history_frame(instance_id, region, profile, hours, period, statistic)
    out_csv, meta_path = export_with_metadata(
        df, out_csv,
        instance_id=instance_id, region=region, hours=hours,
        period=period, statistic=statistic,
    )
    return df, out_csv, meta_path


def newest_sample_age_minutes(df: pd.DataFrame, now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    newest = pd.Timestamp(df["timestamp"].max())
    if newest.tzinfo is None:
        newest = newest.tz_localize("UTC")
    return (now - newest.to_pydatetime(warn=False)).total_seconds() / 60.0


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Export real CloudWatch CPU history to CSV.")
    p.add_argument("--instance-id", default=os.getenv("CLOUDPULSE_INSTANCE_ID", ""))
    p.add_argument("--region", default=os.getenv("CLOUDPULSE_REGION", ""))
    p.add_argument("--profile", default=os.getenv("CLOUDPULSE_PROFILE", "cloudpulse"))
    p.add_argument("--hours", type=float, default=6.0)
    p.add_argument("--period", type=int, default=300, help="seconds between datapoints (300 = 5 min)")
    p.add_argument("--statistic", choices=ALLOWED_STATISTICS, default="Average")
    p.add_argument("--out", default=str(ROOT / "data" / "real_cpu.csv"))
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    if not args.instance_id.strip():
        print("ERROR: pass --instance-id i-xxxx or set CLOUDPULSE_INSTANCE_ID.", file=sys.stderr)
        sys.exit(2)
    df, csv_path, meta_path = collect(
        args.instance_id.strip(), args.region.strip() or None, args.profile.strip(),
        args.hours, args.period, args.statistic, args.out,
    )
    age = newest_sample_age_minutes(df)
    print(f"Fetched {len(df)} real CloudWatch datapoints -> {csv_path}")
    print(f"Newest sample : {df['timestamp'].max()} (UTC), {age:.1f} minutes old")
    print(f"Metadata      : {meta_path}")
    print("Next step     : python -m ml.train_model --input data/real_cpu.csv --model-out models/rf_real.joblib")
