"""
Event Bus for FastALPR Pipeline.
Orchestrates event publication across active sinks (JSON, Neon Postgres, Neon S3, WebSocket).
Enforces upload-before-insert ordering: images -> event.json manifest -> DB insert.
Handles failed-event outbox (output/outbox.jsonl), replay, and optional local snapshot cleanup.
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from .postgres_sink import PostgresSink, check_psycopg_available
from .s3_sink import S3Sink, check_boto3_available
from .ws_sink import WebSocketSink


class EventBus:
    def __init__(
        self,
        database_url: Optional[str] = None,
        ws_port: Optional[int] = None,
        enable_ws: bool = True,
        s3_endpoint_url: Optional[str] = None,
        s3_access_key_id: Optional[str] = None,
        s3_secret_access_key: Optional[str] = None,
        s3_region: str = "us-east-2",
        s3_bucket: str = "alpr-snapshots",
        enable_s3: bool = True,
        snapshots_dir: Optional[Path] = None,
        events_json_path: Optional[Path] = None,
        events_jsonl_path: Optional[Path] = None,
        run_id: str = "",
        camera_id: str = "",
        no_local_snapshots: bool = False,
        outbox_path: Optional[Path] = None,
    ):
        self.database_url = database_url
        self.ws_port = ws_port if (enable_ws and ws_port) else None
        self.run_id = run_id
        self.camera_id = camera_id
        self.s3_bucket = s3_bucket
        self.no_local_snapshots = no_local_snapshots
        self.snapshots_dir = Path(snapshots_dir) if snapshots_dir else Path("output/snapshots")
        self.events_json_path = Path(events_json_path) if events_json_path else Path("output/events.json")
        self.outbox_path = Path(outbox_path) if outbox_path else Path("output/outbox.jsonl")
        self._outbox_lock = threading.Lock()

        self.postgres_sink: Optional[PostgresSink] = None
        self.s3_sink: Optional[S3Sink] = None
        self.ws_sink: Optional[WebSocketSink] = None

        self._pg_off_reason: str = "not configured"
        self._s3_off_reason: str = "disabled" if not enable_s3 else "not configured"
        self._ws_off_reason: str = "disabled" if not enable_ws else "not configured"

        # Preflight dependency checks before initializing/starting any sink
        if self.database_url:
            if check_psycopg_available(announce=True):
                self._pg_off_reason = ""
                self.postgres_sink = PostgresSink(
                    database_url=self.database_url,
                    on_insert_success=self._on_db_insert_success,
                    on_insert_failure=self._on_db_insert_failure,
                )
            else:
                self._pg_off_reason = "psycopg missing"

        if enable_s3 and s3_endpoint_url and s3_access_key_id and s3_secret_access_key:
            if check_boto3_available(announce=True):
                self._s3_off_reason = ""
                self.s3_sink = S3Sink(
                    endpoint_url=s3_endpoint_url,
                    access_key_id=s3_access_key_id,
                    secret_access_key=s3_secret_access_key,
                    region=s3_region,
                    bucket_name=s3_bucket,
                    snapshots_dir=self.snapshots_dir,
                    events_json_path=self.events_json_path,
                    events_jsonl_path=events_jsonl_path,
                    run_id=run_id,
                    camera_id=camera_id,
                )
            else:
                self._s3_off_reason = "boto3 missing"

        if self.ws_port:
            try:
                import websockets  # noqa: F401
                self._ws_off_reason = ""
                self.ws_sink = WebSocketSink(port=self.ws_port)
            except ImportError:
                self._ws_off_reason = "websockets missing"
                print('[PREFLIGHT] ws OFF: websockets install nahi (pip install "websockets>=12")')

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(
        self,
        camera_id: str = "",
        camera_name: str = "",
        camera_location: str = "",
        node_id: str = "",
        run_id: str = "",
        source: str = "",
    ) -> None:
        """Starts all active external sinks and registers camera/run rows."""
        if camera_id:
            self.camera_id = camera_id
        if run_id:
            self.run_id = run_id

        if self.postgres_sink:
            if not self.postgres_sink.start():
                self._pg_off_reason = self.postgres_sink.off_reason or "connect failed"
                self.postgres_sink = None
            else:
                if self.camera_id:
                    self.postgres_sink.register_camera(
                        camera_id=self.camera_id,
                        name=camera_name,
                        location=camera_location,
                        node_id=node_id,
                    )
                if self.run_id:
                    self.postgres_sink.start_run(
                        run_id=self.run_id,
                        camera_id=self.camera_id,
                        source=source,
                    )

        if self.s3_sink:
            if self.camera_id:
                self.s3_sink.camera_id = self.camera_id
            if self.run_id:
                self.s3_sink.run_id = self.run_id
            if not self.s3_sink.start():
                self._s3_off_reason = self.s3_sink.off_reason or "connect failed"
                if self._s3_off_reason == "boto3 missing":
                    self.s3_sink = None

        if self.ws_sink:
            try:
                self.ws_sink.start()
            except Exception as e:
                self._ws_off_reason = "start failed"
                print(f"[WS SINK] Failed to start: {e}")
                self.ws_sink = None

    def finish_run(
        self,
        frames: int = 0,
        fps: float = 0.0,
        tracks_seen: int = 0,
        events_emitted: int = 0,
        status: str = "done",
    ) -> None:
        """Updates the runs row on completion or crash."""
        if self.postgres_sink and self.run_id:
            self.postgres_sink.finish_run(
                run_id=self.run_id,
                frames=frames,
                fps=fps,
                tracks_seen=tracks_seen,
                events_emitted=events_emitted,
                status=status,
            )

    # ------------------------------------------------------------------
    # Event publication (upload-before-insert)
    # ------------------------------------------------------------------

    def publish(self, event: Dict[str, Any]) -> None:
        """
        Ordered publication:
        1. Broadcast to WebSocket (non-blocking)
        2. If S3 configured: upload vehicle.png, plate.png (if present), event.json
           - Any S3 failure -> skip DB insert and append to outbox
        3. Only after S3 succeeds (or if S3 is intentionally unconfigured/disabled):
           enqueue DB insert with S3 metadata
        """
        if self.ws_sink:
            try:
                self.ws_sink.publish(event)
            except Exception as e:
                print(f"[EVENT BUS] WebSocket sink error: {e}")

        if self.s3_sink is not None:
            if self.s3_sink._client is None:
                self._write_outbox(event, "s3")
                return

            s3_ok, vehicle_key, plate_key = self._upload_event_to_s3(event, wait_for_file=True)
            if not s3_ok:
                self._write_outbox(event, "s3")
                return

            enriched = dict(event)
            enriched["run_id"] = self.run_id or None
            enriched["s3_bucket"] = self.s3_sink.bucket_name
            enriched["vehicle_key"] = vehicle_key
            enriched["plate_key"] = plate_key

            if self.postgres_sink:
                try:
                    self.postgres_sink.publish(enriched)
                except Exception as e:
                    print(f"[EVENT BUS] Postgres sink error: {e}")
                    self._write_outbox(event, "postgres")
            elif self.no_local_snapshots:
                self._delete_local_snapshots(event)
        else:
            if self.postgres_sink:
                enriched = dict(event)
                enriched["run_id"] = self.run_id or None
                enriched["s3_bucket"] = None
                enriched["vehicle_key"] = None
                enriched["plate_key"] = None
                try:
                    self.postgres_sink.publish(enriched)
                except Exception as e:
                    print(f"[EVENT BUS] Postgres sink error: {e}")
                    self._write_outbox(event, "postgres")

    def _upload_event_to_s3(
        self,
        event: Dict[str, Any],
        wait_for_file: bool = True,
        allow_existing_s3: bool = False,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Uploads vehicle.png, plate.png (if present), and event.json manifest to S3.
        Returns (success, vehicle_key, plate_key).
        """
        vehicle_key = self.s3_sink.event_vehicle_key(event)
        plate_key = self.s3_sink.event_plate_key(event)
        manifest_key = self.s3_sink.event_manifest_key(event)

        # 1. Vehicle snapshot
        v_snap = event.get("snapshot_vehicle")
        if v_snap:
            v_path = self.snapshots_dir / Path(v_snap).name
            uploaded = self.s3_sink.upload_file_sync(v_path, vehicle_key, wait_for_file=wait_for_file)
            if not uploaded and not (allow_existing_s3 and self.s3_sink.head_object(vehicle_key)):
                print(f"[EVENT BUS] S3 vehicle upload failed for {event.get('event_id')}")
                return False, vehicle_key, plate_key

        # 2. Plate snapshot (when plate detected)
        p_snap = event.get("snapshot_number_plate")
        if p_snap and plate_key:
            p_path = self.snapshots_dir / Path(p_snap).name
            uploaded = self.s3_sink.upload_file_sync(p_path, plate_key, wait_for_file=wait_for_file)
            if not uploaded and not (allow_existing_s3 and self.s3_sink.head_object(plate_key)):
                print(f"[EVENT BUS] S3 plate upload failed for {event.get('event_id')}")
                return False, vehicle_key, plate_key

        # 3. Per-event manifest (exact frozen event schema)
        try:
            manifest_bytes = json.dumps(event, indent=2).encode("utf-8")
            if not self.s3_sink.upload_bytes_sync(manifest_bytes, manifest_key):
                print(f"[EVENT BUS] S3 manifest upload failed for {event.get('event_id')}")
                return False, vehicle_key, plate_key
        except Exception as e:
            print(f"[EVENT BUS] S3 manifest error: {e}")
            return False, vehicle_key, plate_key

        return True, vehicle_key, plate_key

    # ------------------------------------------------------------------
    # DB callbacks & outbox handling
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_event_schema(event: Dict[str, Any]) -> Dict[str, Any]:
        """Keeps only the original event schema fields for outbox persistence."""
        allowed = (
            "event_id",
            "camera_id",
            "time",
            "date",
            "snapshot_number_plate",
            "snapshot_vehicle",
            "plate_detection_confidence",
            "number_plate",
            "number_plate_confidence",
            "vehicle_type",
            "remarks",
        )
        return {k: event[k] for k in allowed if k in event}

    def _on_db_insert_success(self, event: Dict[str, Any]) -> None:
        if self.no_local_snapshots:
            self._delete_local_snapshots(event)

    def _on_db_insert_failure(self, event: Dict[str, Any], failed_sink: str) -> None:
        self._write_outbox(self._strip_event_schema(event), failed_sink, run_id=event.get("run_id"))

    def _write_outbox(self, event: Dict[str, Any], failed_sink: str, run_id: Optional[str] = None) -> None:
        """Appends a failed event to output/outbox.jsonl in a thread-safe way."""
        try:
            with self._outbox_lock:
                self.outbox_path.parent.mkdir(parents=True, exist_ok=True)
                entry = {
                    "event": self._strip_event_schema(event),
                    "failed_sink": failed_sink,
                    "run_id": run_id or self.run_id or None,
                    "timestamp": datetime.now().astimezone().isoformat(),
                }
                with open(self.outbox_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            print(f"[OUTBOX] Event {event.get('event_id')} queued to outbox (failed_sink={failed_sink})")
        except Exception as e:
            print(f"[OUTBOX ERROR] Could not write outbox: {e}")

    def replay_outbox(self) -> None:
        """Replays output/outbox.jsonl at run start. Removes entries that succeed on all active sinks."""
        if not self.outbox_path.exists():
            return

        try:
            raw_lines = [ln for ln in self.outbox_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        except Exception as e:
            print(f"[OUTBOX ERROR] Could not read {self.outbox_path}: {e}")
            return

        if not raw_lines:
            return

        print(f"[OUTBOX] Replaying {len(raw_lines)} pending event(s)...")
        remaining: List[str] = []
        replayed = 0

        for line in raw_lines:
            try:
                entry = json.loads(line)
                event = entry.get("event", {})
                orig_run_id = entry.get("run_id") or self.run_id or None
            except Exception:
                remaining.append(line)
                continue

            vehicle_key: Optional[str] = None
            plate_key: Optional[str] = None
            s3_bucket: Optional[str] = None

            if self.s3_sink is not None:
                if self.s3_sink._client is None:
                    remaining.append(line)
                    continue
                s3_ok, vehicle_key, plate_key = self._upload_event_to_s3(
                    event, wait_for_file=False, allow_existing_s3=True
                )
                if not s3_ok:
                    remaining.append(line)
                    continue
                s3_bucket = self.s3_sink.bucket_name

            if self.postgres_sink is not None:
                enriched = dict(event)
                enriched["run_id"] = orig_run_id
                enriched["s3_bucket"] = s3_bucket
                enriched["vehicle_key"] = vehicle_key
                enriched["plate_key"] = plate_key
                if orig_run_id:
                    self.postgres_sink.ensure_run_exists(orig_run_id, event.get("camera_id", self.camera_id))
                if not self.postgres_sink.insert_sync(enriched):
                    remaining.append(line)
                    continue

            if self.no_local_snapshots:
                self._delete_local_snapshots(event)
            replayed += 1

        with self._outbox_lock:
            if remaining:
                self.outbox_path.write_text("\n".join(remaining) + "\n", encoding="utf-8")
            else:
                self.outbox_path.unlink(missing_ok=True)

        print(f"[OUTBOX] Replay done: replayed={replayed}, remaining={len(remaining)}")

    def _delete_local_snapshots(self, event: Dict[str, Any]) -> None:
        """Deletes JSON-referenced local snapshot files after successful S3 + DB persistence."""
        for key in ("snapshot_vehicle", "snapshot_number_plate"):
            fname = event.get(key)
            if fname:
                local_path = self.snapshots_dir / Path(fname).name
                try:
                    local_path.unlink(missing_ok=True)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Artifacts & shutdown
    # ------------------------------------------------------------------

    def upload_artifact(self, local_path: Path, rel_key: str) -> None:
        """Synchronously uploads a finalized output artifact (e.g. annotated video) to S3."""
        if self.s3_sink and self.s3_sink._client is not None:
            s3_key = self.s3_sink.run_video_key() if (rel_key.endswith(".mp4") and self.run_id) else rel_key
            try:
                ok = self.s3_sink.upload_file_sync(local_path, s3_key, wait_for_file=False)
                if ok:
                    print(f"[EVENT BUS] Uploaded artifact -> s3://{self.s3_sink.bucket_name}/{s3_key}")
            except Exception as e:
                print(f"[EVENT BUS] S3 artifact upload error: {e}")

    def print_sync_report(
        self,
        events_json_path: Optional[Path] = None,
        output_video_path: Optional[Path] = None,
        run_id: str = "",
    ) -> str:
        """
        Verifies end-of-run consistency between events.json, Neon Postgres (for run_id),
        and Neon S3 (head_object per key), prints mismatch reasons, and returns summary line.
        """
        mismatches: List[str] = []
        json_path = Path(events_json_path) if events_json_path else self.events_json_path
        rid = run_id or self.run_id

        events: List[Dict[str, Any]] = []
        if json_path.exists():
            try:
                loaded = json.loads(json_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    events = loaded
                else:
                    mismatches.append("events_json_not_a_list")
            except Exception as e:
                mismatches.append(f"events_json_read_error: {e}")
        json_count = len(events)

        db_count = 0
        if self.postgres_sink is not None:
            db_rows_list = self.postgres_sink.get_run_events(rid)
            if db_rows_list is None:
                mismatches.append("postgres_unreachable")
            else:
                db_count = len(db_rows_list)
                db_ids = {r["event_id"] for r in db_rows_list}
                for ev in events:
                    eid = ev.get("event_id")
                    if eid not in db_ids:
                        mismatches.append(f"db_missing_row:{eid}")
                if db_count != json_count:
                    mismatches.append(f"db_count_mismatch:json={json_count},db={db_count}")

        s3_count = 0
        if self.s3_sink is not None:
            if self.s3_sink._client is None:
                mismatches.append("s3_client_unavailable")
            else:
                for ev in events:
                    vk = self.s3_sink.event_vehicle_key(ev)
                    if self.s3_sink.head_object(vk):
                        s3_count += 1
                    else:
                        mismatches.append(f"s3_missing:{vk}")

                    pk = self.s3_sink.event_plate_key(ev)
                    if pk:
                        if self.s3_sink.head_object(pk):
                            s3_count += 1
                        else:
                            mismatches.append(f"s3_missing:{pk}")

                    mk = self.s3_sink.event_manifest_key(ev)
                    if self.s3_sink.head_object(mk):
                        s3_count += 1
                    else:
                        mismatches.append(f"s3_missing:{mk}")

                if json_path.exists() and rid:
                    rk = self.s3_sink.run_events_key()
                    if self.s3_sink.head_object(rk):
                        s3_count += 1
                    else:
                        mismatches.append(f"s3_missing:{rk}")

                if output_video_path and Path(output_video_path).exists() and rid:
                    vid_key = self.s3_sink.run_video_key()
                    if self.s3_sink.head_object(vid_key):
                        s3_count += 1
                    else:
                        mismatches.append(f"s3_missing:{vid_key}")

        report = f"[SYNC] json_events={json_count} db_rows={db_count} s3_objects={s3_count} mismatches={mismatches}"
        print(report)
        for reason in mismatches:
            print(f"[SYNC MISMATCH] {reason}")
        return report

    def get_status_str(self) -> str:
        """Returns console status string for active sinks, including reasons when OFF."""
        pg_status = "ON" if self.postgres_sink is not None else f"OFF({self._pg_off_reason})"
        if self.s3_sink is not None and self.s3_sink._client is not None:
            s3_status = "ON"
        else:
            s3_status = f"OFF({self._s3_off_reason})"
        ws_status = "ON" if self.ws_sink is not None else f"OFF({self._ws_off_reason})"
        return f"[SINKS] json=ON postgres={pg_status} s3={s3_status} ws={ws_status}"

    def stop(self, timeout: float = 30.0) -> None:
        """Flushes and shuts down all sinks (does not force-close PG until close_db)."""
        if self.postgres_sink:
            try:
                self.postgres_sink.stop(timeout=timeout)
            except Exception as e:
                print(f"[EVENT BUS] Postgres shutdown error: {e}")

        if self.s3_sink:
            try:
                self.s3_sink.stop(timeout=max(timeout, 30.0))
            except Exception as e:
                print(f"[EVENT BUS] S3 shutdown error: {e}")

        if self.ws_sink:
            try:
                self.ws_sink.stop(timeout=2.0)
            except Exception as e:
                print(f"[EVENT BUS] WebSocket shutdown error: {e}")

    def close_db(self) -> None:
        """Closes any lingering Postgres connection after final run update."""
        if self.postgres_sink:
            self.postgres_sink.close()
