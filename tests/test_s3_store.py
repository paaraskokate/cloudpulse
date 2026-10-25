"""Tests for the S3 archive adapter (mocked boto3 — no real uploads)."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cloud.s3_store import S3UploadError, object_key, upload_file


class FakeS3:
    def __init__(self, head_error=None):
        self.head_error = head_error
        self.uploads = []

    def head_bucket(self, **kwargs):
        if self.head_error:
            raise self.head_error

    def upload_file(self, local, bucket, key):
        self.uploads.append((local, bucket, key))


class FakeSession:
    def __init__(self, client):
        self._client = client
        self.region_name = "ap-south-1"

    def client(self, name):
        assert name == "s3"
        return self._client


class TestObjectKey(unittest.TestCase):
    def test_key_joining(self):
        self.assertEqual(object_key("cloudpulse", "real_cpu.csv"), "cloudpulse/real_cpu.csv")
        self.assertEqual(object_key("", "real_cpu.csv"), "real_cpu.csv")
        self.assertEqual(object_key("a/b", "x.csv"), "a/b/x.csv")
        self.assertEqual(object_key("/cloudpulse/", "x.csv"), "cloudpulse/x.csv")


class TestUpload(unittest.TestCase):
    def test_uploads_under_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write_temp(tmp)
            fake = FakeS3()
            with patch("cloud.s3_store.build_session", return_value=FakeSession(fake)):
                result = upload_file("my-private-bucket", p, prefix="cloudpulse")
            self.assertEqual(result["key"], "cloudpulse/snap.csv")
            self.assertEqual(result["bucket"], "my-private-bucket")
            self.assertEqual(len(fake.uploads), 1)

    def test_unreachable_bucket_raises_clear_error(self):
        import botocore.exceptions as bex

        err = bex.ClientError({"Error": {"Code": "404", "Message": "nope"}}, "HeadBucket")
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write_temp(tmp)
            with patch("cloud.s3_store.build_session", return_value=FakeSession(FakeS3(head_error=err))):
                with self.assertRaises(S3UploadError) as ctx:
                    upload_file("missing-bucket", p)
            self.assertIn("bucket", str(ctx.exception).lower())

    def test_missing_local_file(self):
        with self.assertRaises(FileNotFoundError):
            upload_file("b", "definitely_missing_file.csv")

    def _write_temp(self, tmp):
        p = Path(tmp) / "snap.csv"
        p.write_text("timestamp,cpu\n2026-01-01T00:00:00+00:00,10.0\n", encoding="utf-8")
        return p


if __name__ == "__main__":
    unittest.main()
