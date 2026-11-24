#!/usr/bin/env python3
"""
Neon Object Storage (S3) Diagnostic Tool for FastALPR Pipeline.
Verifies AWS_ENDPOINT_URL_S3, credentials, bucket creation/access, and test upload/read/delete.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env", override=True)
except ImportError:
    pass

endpoint_url = os.environ.get("AWS_ENDPOINT_URL_S3")
access_key = os.environ.get("AWS_ACCESS_KEY_ID")
secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
region = os.environ.get("AWS_REGION", "us-east-2")
bucket_name = os.environ.get("S3_BUCKET", "alpr-snapshots")

if not (endpoint_url and access_key and secret_key):
    print("[ERROR] AWS_ENDPOINT_URL_S3, AWS_ACCESS_KEY_ID, or AWS_SECRET_ACCESS_KEY is missing.")
    print("        Please define them in .env or environment variables.")
    sys.exit(1)

print(f"[CHECK S3] Endpoint: {endpoint_url} | Region: {region} | Target Bucket: {bucket_name}")

try:
    import boto3
    from botocore.config import Config
except ImportError:
    print("[ERROR] boto3 is not installed. Run: pip install boto3")
    sys.exit(1)

try:
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
        config=Config(s3={"addressing_style": "path"}, signature_version="s3v4", connect_timeout=10, read_timeout=20),
    )

    resp = s3.list_buckets()
    buckets = [b["Name"] for b in resp.get("Buckets", []) if b.get("Name")]
    print(f"[CHECK S3] Existing buckets: {buckets}")

    if bucket_name not in buckets:
        try:
            if region and region != "us-east-1":
                s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={"LocationConstraint": region})
            else:
                s3.create_bucket(Bucket=bucket_name)
            print(f"[CHECK S3] Created bucket '{bucket_name}'.")
        except Exception as create_err:
            if buckets:
                print(f"[CHECK S3 WARN] Could not create '{bucket_name}' ({create_err}); testing '{buckets[0]}'.")
                bucket_name = buckets[0]
            else:
                raise create_err

    test_key = f"_healthcheck/test_{int(datetime.now().timestamp())}.txt"
    payload = b"fast-alpr-s3-check"
    s3.put_object(Bucket=bucket_name, Key=test_key, Body=payload, ContentType="text/plain")
    got = s3.get_object(Bucket=bucket_name, Key=test_key)["Body"].read()
    if got != payload:
        raise RuntimeError("S3 read verification mismatch.")
    s3.delete_object(Bucket=bucket_name, Key=test_key)

    print(f"S3 OK (bucket='{bucket_name}')")
    sys.exit(0)
except Exception as e:
    print(f"[ERROR] Neon S3 check failed: {e}")
    sys.exit(1)
