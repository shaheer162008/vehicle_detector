"""
Neon Object Storage (S3-Compatible) Event Sink for FastALPR Pipeline.
Uploads vehicle snapshots, plate snapshots, per-event manifests, run-level
events.json, and annotated output video to Neon Object Storage
using path-style addressing with namespaced keys.

Key layout:
  {camera_id}/{YYYY-MM-DD}/{event_id}/vehicle.png
  {camera_id}/{YYYY-MM-DD}/{event_id}/plate.png
  {camera_id}/{YYYY-MM-DD}/{event_id}/event.json
  runs/{run_id}/events.json
  runs/{run_id}/annotated_output.mp4
"""

import time
from pathlib import Path
from typing import Optional, Dict, Any


def check_boto3_available(announce: bool = True) -> bool:
    """Checks boto3 availability without creating an S3 client or retrying."""
    try:
        import boto3  # noqa: F401
        return True
    except ImportError:
        if announce:
            print('[PREFLIGHT] s3 OFF: boto3 install nahi (pip install "boto3>=1.34.0")')
        return False


def _mask_key(key: str) -> str:
    """Masks access keys for safe logging."""
    if not key or len(key) < 10:
        return "****"
    return f"{key[:4]}...{key[-4:]}"


class S3Sink:
    def __init__(
        self,
        endpoint_url: str,
        access_key_id: str,
        secret_access_key: str,
        region: str = "us-east-2",
        bucket_name: str = "alpr-snapshots",
        snapshots_dir: Optional[Path] = None,
        events_json_path: Optional[Path] = None,
        events_jsonl_path: Optional[Path] = None,
        run_id: str = "",
        camera_id: str = "",
    ):
        self.endpoint_url = endpoint_url
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.region = region or "us-east-2"
        self.bucket_name = bucket_name or "alpr-snapshots"
        self.snapshots_dir = Path(snapshots_dir) if snapshots_dir else Path("output/snapshots")
        self.events_json_path = Path(events_json_path) if events_json_path else Path("output/events.json")
        self.events_jsonl_path = Path(events_jsonl_path) if events_jsonl_path else Path("output/events.jsonl")
        self.run_id = run_id
        self.camera_id = camera_id
        self.running = False
        self._client = None
        self.off_reason: str = ""
        self._uploaded_keys: set[str] = set()
        self.uploaded_count: int = 0
        self.failed_count: int = 0

    def _create_client(self):
        import boto3
        from botocore.config import Config

        s3_cfg = Config(
            s3={"addressing_style": "path"},
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "standard"},
            connect_timeout=10,
            read_timeout=30,
        )
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region,
            config=s3_cfg,
        )

    def _ensure_bucket(self) -> bool:
        """Ensures target S3 bucket exists or resolves an available bucket on Neon Object Storage."""
        if self._client is None:
            return False
        existing: list[str] = []
        try:
            resp = self._client.list_buckets()
            existing = [b["Name"] for b in resp.get("Buckets", []) if b.get("Name")]
            if self.bucket_name in existing:
                print(f"[S3 SINK] Connected to Neon S3 bucket '{self.bucket_name}'.")
                return True
        except Exception as e:
            print(f"[S3 SINK WARNING] list_buckets note: {e}")

        try:
            self._client.head_bucket(Bucket=self.bucket_name)
            print(f"[S3 SINK] Connected to Neon S3 bucket '{self.bucket_name}'.")
            return True
        except Exception as head_err:
            code = str(getattr(head_err, "response", {}).get("Error", {}).get("Code", ""))
            if code not in ("404", "NoSuchBucket", "NotFound") and not existing:
                print(f"[S3 SINK ERROR] Bucket '{self.bucket_name}' unreachable: {head_err}")
                return False

        try:
            create_kwargs: Dict[str, Any] = {"Bucket": self.bucket_name}
            if self.region and self.region != "us-east-1":
                create_kwargs["CreateBucketConfiguration"] = {"LocationConstraint": self.region}
            try:
                self._client.create_bucket(**create_kwargs)
            except Exception:
                self._client.create_bucket(Bucket=self.bucket_name)
            print(f"[S3 SINK] Created and selected Neon S3 bucket '{self.bucket_name}'.")
            return True
        except Exception as e:
            if existing:
                fallback = existing[0]
                print(
                    f"[S3 SINK WARNING] Could not create bucket '{self.bucket_name}' ({e}); "
                    f"falling back to existing bucket '{fallback}'."
                )
                self.bucket_name = fallback
                return True
            print(f"[S3 SINK ERROR] Bucket '{self.bucket_name}' unavailable and could not be created: {e}")
            return False

    def start(self) -> bool:
        """Initializes Neon S3 client and ensures bucket exists. Never retries on ImportError."""
        if not (self.endpoint_url and self.access_key_id and self.secret_access_key):
            self.off_reason = "not configured"
            return False
        if not check_boto3_available(announce=True):
            self.off_reason = "boto3 missing"
            return False

        print(f"[S3 SINK] Connecting: endpoint={self.endpoint_url} key={_mask_key(self.access_key_id)}")
        try:
            self._client = self._create_client()
        except ImportError:
            self.off_reason = "boto3 missing"
            return False
        except Exception as e:
            self.off_reason = "connect failed"
            print(f"[S3 SINK ERROR] Could not initialize boto3 client: {e}")
            return False

        if not self._ensure_bucket():
            self._client = None
            self.off_reason = "connect failed"
            return False

        self.running = True
        return True

    # ------------------------------------------------------------------
    # Namespaced key builders
    # ------------------------------------------------------------------

    def event_vehicle_key(self, event: Dict[str, Any]) -> str:
        """S3 key: {camera_id}/{YYYY-MM-DD}/{event_id}/vehicle.png"""
        cam = event.get("camera_id") or self.camera_id or "unknown"
        date = event.get("date") or "unknown"
        eid = event.get("event_id") or "unknown"
        return f"{cam}/{date}/{eid}/vehicle.png"

    def event_plate_key(self, event: Dict[str, Any]) -> Optional[str]:
        """S3 key: {camera_id}/{YYYY-MM-DD}/{event_id}/plate.png (None if no plate)."""
        if not event.get("snapshot_number_plate"):
            return None
        cam = event.get("camera_id") or self.camera_id or "unknown"
        date = event.get("date") or "unknown"
        eid = event.get("event_id") or "unknown"
        return f"{cam}/{date}/{eid}/plate.png"

    def event_manifest_key(self, event: Dict[str, Any]) -> str:
        """S3 key: {camera_id}/{YYYY-MM-DD}/{event_id}/event.json"""
        cam = event.get("camera_id") or self.camera_id or "unknown"
        date = event.get("date") or "unknown"
        eid = event.get("event_id") or "unknown"
        return f"{cam}/{date}/{eid}/event.json"

    def run_events_key(self) -> str:
        """S3 key: runs/{run_id}/events.json"""
        return f"runs/{self.run_id}/events.json"

    def run_video_key(self) -> str:
        """S3 key: runs/{run_id}/annotated_output.mp4"""
        return f"runs/{self.run_id}/annotated_output.mp4"

    # ------------------------------------------------------------------
    # Synchronous upload & inspection
    # ------------------------------------------------------------------

    def upload_file_sync(self, local_path: Path, s3_key: str, wait_for_file: bool = True) -> bool:
        """Synchronously uploads a file to S3 with retry. Returns True on success."""
        if self._client is None:
            return False

        if wait_for_file:
            for _ in range(50):
                if local_path.exists() and local_path.stat().st_size > 0:
                    break
                time.sleep(0.1)

        if not local_path.exists() or not local_path.is_file():
            return False

        extra_args = {}
        suffix = local_path.suffix.lower()
        if suffix == ".png":
            extra_args["ContentType"] = "image/png"
        elif suffix in (".json", ".jsonl"):
            extra_args["ContentType"] = "application/json"
        elif suffix == ".mp4":
            extra_args["ContentType"] = "video/mp4"

        backoffs = [1, 2, 4]
        last_err = None
        for attempt in range(len(backoffs)):
            try:
                if extra_args:
                    self._client.upload_file(str(local_path), self.bucket_name, s3_key, ExtraArgs=extra_args)
                else:
                    self._client.upload_file(str(local_path), self.bucket_name, s3_key)
                self._uploaded_keys.add(s3_key)
                self.uploaded_count += 1
                return True
            except Exception as e:
                last_err = e
                time.sleep(backoffs[attempt])

        self.failed_count += 1
        print(f"[S3 SINK ERROR] Failed uploading {local_path.name} -> {s3_key}: {last_err}")
        return False

    def upload_bytes_sync(self, data: bytes, s3_key: str, content_type: str = "application/json") -> bool:
        """Uploads raw bytes to S3 synchronously with retry."""
        if self._client is None:
            return False
        backoffs = [1, 2, 4]
        last_err = None
        for attempt in range(len(backoffs)):
            try:
                self._client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=data,
                    ContentType=content_type,
                )
                self._uploaded_keys.add(s3_key)
                self.uploaded_count += 1
                return True
            except Exception as e:
                last_err = e
                time.sleep(backoffs[attempt])

        self.failed_count += 1
        print(f"[S3 SINK ERROR] Failed uploading bytes -> {s3_key}: {last_err}")
        return False

    def head_object(self, s3_key: str) -> bool:
        """Returns True if object exists on S3."""
        if self._client is None or not s3_key:
            return False
        try:
            self._client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except Exception:
            return False

    def sync_final_outputs(self) -> None:
        """Uploads finalized run-level events.json under runs/{run_id}/events.json."""
        if self._client is None or not self.run_id:
            return

        if self.events_json_path.exists():
            key = self.run_events_key()
            ok = self.upload_file_sync(self.events_json_path, key, wait_for_file=False)
            if ok:
                print(f"[S3 SINK] Uploaded {self.events_json_path.name} -> s3://{self.bucket_name}/{key}")

    def stop(self, timeout: float = 30.0) -> None:
        """Performs final run-level events.json sync and logs summary."""
        self.running = False
        if self._client is not None:
            self.sync_final_outputs()
            print(
                f"[S3 SINK] Neon S3 sync complete -> bucket='{self.bucket_name}' | "
                f"uploads={self.uploaded_count} | failed={self.failed_count}"
            )
