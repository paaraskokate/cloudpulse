"""Optional, PRIVATE S3 archive for CloudPulse artifacts.

Uploads exported metric CSVs, metadata JSON, training reports and policy
summaries. The bucket must block all public access. Never upload
credentials, personal data or unredacted account screenshots.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
from cloud.aws_metrics import build_session  # noqa: E402  (needs sys.path = root)


class S3UploadError(RuntimeError):
    """Bucket missing/unreachable or upload denied."""


def object_key(prefix: str, filename: str) -> str:
    """Join a (possibly empty) prefix and filename into an S3 object key."""
    filename = str(filename).strip("/")
    prefix = str(prefix or "").strip("/")
    return f"{prefix}/{filename}" if prefix else filename


def upload_file(
    bucket: str,
    local_path: str | Path,
    profile: str = "cloudpulse",
    region: str | None = None,
    prefix: str = "cloudpulse",
) -> dict:
    """Upload one file as a private object. Returns {bucket, key, size, region}."""
    local = Path(local_path)
    if not local.exists():
        raise FileNotFoundError(f"Nothing to upload: {local}")
    bucket = str(bucket).strip()
    if not bucket:
        raise ValueError("bucket name is required.")

    session = build_session(profile_name=profile, region_name=region)
    s3 = session.client("s3")
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "Unknown")
        raise S3UploadError(
            f"Cannot access bucket '{bucket}' ({code}). Check the bucket name, its region "
            "and that the profile allows s3:ListBucket / s3:GetBucketLocation."
        ) from exc

    key = object_key(prefix, local.name)
    try:
        s3.upload_file(str(local), bucket, key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "Unknown")
        raise S3UploadError(
            f"Upload failed ({code}): check s3:PutObject permission on the bucket. "
            "Keep Block Public Access ENABLED — private objects stay private."
        ) from exc
    return {
        "bucket": bucket,
        "key": key,
        "size_bytes": local.stat().st_size,
        "region": session.region_name,
    }


def archive_with_metadata(
    bucket: str,
    csv_path: str | Path,
    profile: str = "cloudpulse",
    region: str | None = None,
    prefix: str = "cloudpulse",
) -> list[dict]:
    """Upload a CSV and (if present) its *_metadata.json sidecar."""
    csv_path = Path(csv_path)
    results = [upload_file(bucket, csv_path, profile, region, prefix)]
    sidecar = csv_path.with_name(csv_path.stem + "_metadata.json")
    if sidecar.exists():
        results.append(upload_file(bucket, sidecar, profile, region, prefix))
    return results


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Archive CloudPulse exports to S3 (private).")
    p.add_argument("--bucket", required=True)
    p.add_argument("--path", required=True, help="local CSV/JSON file to upload")
    p.add_argument("--profile", default=os.getenv("CLOUDPULSE_PROFILE", "cloudpulse"))
    p.add_argument("--region", default=os.getenv("CLOUDPULSE_REGION", ""))
    p.add_argument("--prefix", default="cloudpulse")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    try:
        for r in archive_with_metadata(
            args.bucket, args.path, args.profile, args.region or None, args.prefix
        ):
            print(f"Uploaded s3://{r['bucket']}/{r['key']}  ({r['size_bytes']} bytes)")
    except (S3UploadError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
