"""
Neon Object Storage (S3-Compatible) Event Sink for FastALPR Pipeline.
Uploads vehicle snapshots, plate snapshots, events.json, events.jsonl,
and optional annotated output video to Neon Object Storage using path-style addressing.
"""

import queue
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple


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
        run_prefix: str = "",
        max_queue_size: int = 1000,
    ):
        self.endpoint_url = endpoint_url
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.region = region or "us-east-2"
        self.bucket_name = bucket_name or "alpr-snapshots"
        self.snapshots_dir = Path(snapshots_dir) if snapshots_dir else Path("output/snapshots")
        self.events_json_path = Path(events_json_path) if events_json_path else Path("output/events.json")
        self.events_jsonl_path = Path(events_jsonl_path) if events_jsonl_path else Path("output/events.jsonl")
        self.run_prefix = run_prefix.strip("/")
        self.queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        self._client = None
        self._uploaded_keys: set[str] = set()
        self.uploaded_count: int = 0
        self.failed_count: int = 0
        self._lock = threading.Lock()

    def _s3_key(self, rel_key: str) -> str:
        rel_clean = rel_key.lstrip("/")
        if self.run_prefix:
            return f"{self.run_prefix}/{rel_clean}"
        return rel_clean

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
        try:
            resp = self._client.list_buckets()
            existing = [b["Name"] for b in resp.get("Buckets", []) if b.get("Name")]
            if self.bucket_name in existing:
                print(f"[S3 SINK] Connected to Neon S3 bucket '{self.bucket_name}'.")
                return True
        except Exception as e:
            print(f"[S3 SINK WARNING] list_buckets note: {e}")
            existing = []

        # Try head_bucket first in case list_buckets is restricted
        try:
            self._client.head_bucket(Bucket=self.bucket_name)
            print(f"[S3 SINK] Connected to Neon S3 bucket '{self.bucket_name}'.")
            return True
        except Exception:
            pass

        # Create bucket if missing
        try:
            create_kwargs: Dict[str, Any] = {"Bucket": self.bucket_name}
            if self.region and self.region != "us-east-1":
                create_kwargs["CreateBucketConfiguration"] = {"LocationConstraint": self.region}
            try:
                self._client.create_bucket(**create_kwargs)
            except Exception:
                # Some S3-compatible providers ignore/reject LocationConstraint
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

    def start(self) -> None:
        """Initializes Neon S3 client, ensures bucket exists, and starts background uploader."""
        if not (self.endpoint_url and self.access_key_id and self.secret_access_key):
            return
        try:
            self._client = self._create_client()
        except ImportError:
            print("[S3 SINK ERROR] boto3 is not installed. Run: pip install boto3")
            return
        except Exception as e:
            print(f"[S3 SINK ERROR] Could not initialize boto3 client: {e}")
            return

        if not self._ensure_bucket():
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _enqueue_file(self, local_path: Path, rel_key: str, wait_for_file: bool = True) -> None:
        if not self.running:
            return
        try:
            self.queue.put_nowait((Path(local_path), rel_key, wait_for_file))
        except queue.Full:
            print(f"[S3 SINK WARNING] Upload queue full. Skipping async enqueue for {rel_key}.")

    def publish(self, event: Dict[str, Any]) -> None:
        """Queues snapshots and event manifests associated with an emitted event."""
        if not self.running:
            return

        v_snap = event.get("snapshot_vehicle")
        if v_snap:
            v_name = Path(v_snap).name
            self._enqueue_file(self.snapshots_dir / v_name, f"snapshots/{v_name}", wait_for_file=True)

        p_snap = event.get("snapshot_number_plate")
        if p_snap:
            p_name = Path(p_snap).name
            self._enqueue_file(self.snapshots_dir / p_name, f"snapshots/{p_name}", wait_for_file=True)

        if self.events_json_path.exists():
            self._enqueue_file(self.events_json_path, "events.json", wait_for_file=False)

    def _upload_with_retry(self, local_path: Path, rel_key: str, wait_for_file: bool = True) -> bool:
        if self._client is None:
            return False

        # Wait up to 5s for AsyncSnapshotSaver to flush PNG to disk
        if wait_for_file:
            for _ in range(50):
                if local_path.exists() and local_path.stat().st_size > 0:
                    break
                time.sleep(0.1)

        if not local_path.exists() or not local_path.is_file():
            return False

        s3_key = self._s3_key(rel_key)
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
                with self._lock:
                    self._uploaded_keys.add(s3_key)
                    self.uploaded_count += 1
                return True
            except Exception as e:
                last_err = e
                time.sleep(backoffs[attempt])

        with self._lock:
            self.failed_count += 1
        print(f"[S3 SINK ERROR] Failed uploading {local_path.name} -> {s3_key}: {last_err}")
        return False

    def _worker_loop(self) -> None:
        while self.running or not self.queue.empty():
            try:
                item: Optional[Tuple[Path, str, bool]] = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                self.queue.task_done()
                break

            local_path, rel_key, wait_for_file = item
            try:
                self._upload_with_retry(local_path, rel_key, wait_for_file=wait_for_file)
            finally:
                self.queue.task_done()

    def upload_file_sync(self, local_path: Path, rel_key: str) -> bool:
        """Synchronously uploads a finalized artifact (e.g. events.json or annotated_output.mp4)."""
        if self._client is None:
            return False
        ok = self._upload_with_retry(Path(local_path), rel_key, wait_for_file=False)
        if ok:
            print(f"[S3 SINK] Uploaded {Path(local_path).name} -> s3://{self.bucket_name}/{self._s3_key(rel_key)}")
        return ok

    def sync_final_outputs(self) -> None:
        """Uploads all remaining snapshots and final events.json / events.jsonl after disk flush."""
        if self._client is None:
            return

        if self.snapshots_dir.exists():
            for img_path in sorted(self.snapshots_dir.glob("*.png")):
                rel_key = f"snapshots/{img_path.name}"
                s3_key = self._s3_key(rel_key)
                with self._lock:
                    already = s3_key in self._uploaded_keys
                if not already:
                    self._upload_with_retry(img_path, rel_key, wait_for_file=False)

        if self.events_json_path.exists():
            self._upload_with_retry(self.events_json_path, "events.json", wait_for_file=False)
        if self.events_jsonl_path.exists() and self.events_jsonl_path.stat().st_size > 0:
            self._upload_with_retry(self.events_jsonl_path, "events.jsonl", wait_for_file=False)

    def stop(self, timeout: float = 30.0) -> None:
        """Flushes async queue, performs final snapshot/JSON sync, and closes S3 sink."""
        if self.running:
            self.running = False
            try:
                self.queue.put_nowait(None)
            except Exception:
                pass
            if self.worker_thread and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=timeout)

        if self._client is not None:
            self.sync_final_outputs()
            print(
                f"[S3 SINK] Neon S3 sync complete -> bucket='{self.bucket_name}' | "
                f"uploads={self.uploaded_count} | failed={self.failed_count}"
            )
