"""
Neon PostgreSQL Event Sink for FastALPR Pipeline.
Handles background asynchronous database writes, auto-reconnection with backoff,
schema bootstrap, camera/run lifecycle, synchronous replay support, and callbacks.
"""

import re
import time
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple, List


def check_psycopg_available(announce: bool = True) -> bool:
    """Checks psycopg + libpq availability without opening a DB connection or retrying."""
    try:
        import psycopg
        psycopg.pq.version()
        return True
    except Exception:
        if announce:
            print('[PREFLIGHT] postgres OFF: psycopg[binary] install nahi (pip install "psycopg[binary]>=3.1")')
        return False


def mask_database_url(url: str) -> str:
    """Masks credentials in postgresql URL for safe logging."""
    if not url:
        return ""
    try:
        return re.sub(r":([^:@/]+)@", r":****@", url)
    except Exception:
        return "postgresql://****"


_INSERT_EVENT_SQL = """
INSERT INTO events (
    event_id,
    camera_id,
    event_time,
    event_date,
    snapshot_vehicle,
    snapshot_number_plate,
    plate_detection_confidence,
    number_plate,
    number_plate_confidence,
    vehicle_type,
    run_id,
    s3_bucket,
    vehicle_key,
    plate_key
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (event_id) DO NOTHING;
"""


class PostgresSink:
    def __init__(
        self,
        database_url: str,
        schema_path: Optional[Path] = None,
        max_queue_size: int = 1000,
        on_insert_success: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_insert_failure: Optional[Callable[[Dict[str, Any], str], None]] = None,
    ):
        self.database_url = database_url
        self.schema_path = schema_path or Path(__file__).resolve().parent.parent / "sql" / "schema.sql"
        self.queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        self._conn = None
        self._conn_lock = threading.Lock()
        self.inserted_count: int = 0
        self.failed_count: int = 0
        self.init_failed: bool = False
        self.off_reason: str = ""
        self.on_insert_success = on_insert_success
        self.on_insert_failure = on_insert_failure
        self._run_id: Optional[str] = None
        self._camera_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Initializes schema and starts background ingestion worker."""
        if not self.database_url:
            self.off_reason = "not configured"
            return False

        if not check_psycopg_available(announce=True):
            self.init_failed = True
            self.off_reason = "psycopg missing"
            return False

        masked = mask_database_url(self.database_url)
        print(f"[POSTGRES] Connecting to Neon: {masked}")

        if not self._apply_schema():
            self.init_failed = True
            if not self.off_reason:
                self.off_reason = "connect failed"
            return False

        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        return True

    def register_camera(self, camera_id: str, name: str = "", location: str = "", node_id: str = "") -> None:
        """Inserts camera row on startup (ON CONFLICT DO NOTHING)."""
        self._camera_id = camera_id
        if self.init_failed:
            return
        with self._conn_lock:
            if not self._ensure_connection():
                return
            try:
                with self._conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO cameras (camera_id, name, location, node_id) "
                        "VALUES (%s, %s, %s, %s) ON CONFLICT (camera_id) DO NOTHING;",
                        (camera_id, name or None, location or None, node_id or None),
                    )
            except Exception as e:
                print(f"[POSTGRES] Camera register note: {e}")

    def start_run(self, run_id: str, camera_id: str, source: str = "") -> None:
        """Inserts a runs row with status='running'."""
        self._run_id = run_id
        if self.init_failed:
            return
        with self._conn_lock:
            if not self._ensure_connection():
                return
            try:
                with self._conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO runs (run_id, camera_id, source, status) "
                        "VALUES (%s, %s, %s, 'running') ON CONFLICT (run_id) DO NOTHING;",
                        (run_id, camera_id, source or None),
                    )
            except Exception as e:
                print(f"[POSTGRES] Run start note: {e}")

    def ensure_run_exists(self, run_id: str, camera_id: str = "") -> None:
        """Ensures a runs parent row exists before replaying historical events."""
        if not run_id or self.init_failed:
            return
        with self._conn_lock:
            if not self._ensure_connection():
                return
            try:
                with self._conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO runs (run_id, camera_id, status) "
                        "VALUES (%s, %s, 'done') ON CONFLICT (run_id) DO NOTHING;",
                        (run_id, camera_id or None),
                    )
            except Exception as e:
                print(f"[POSTGRES] Ensure run note: {e}")

    def finish_run(
        self,
        run_id: str,
        frames: int = 0,
        fps: float = 0.0,
        tracks_seen: int = 0,
        events_emitted: int = 0,
        status: str = "done",
    ) -> None:
        """Updates runs row on completion or crash."""
        if self.init_failed:
            return
        with self._conn_lock:
            if not self._ensure_connection():
                return
            try:
                with self._conn.cursor() as cur:
                    cur.execute(
                        "UPDATE runs SET finished_at = now(), frames = %s, fps = %s, "
                        "tracks_seen = %s, events_emitted = %s, status = %s "
                        "WHERE run_id = %s;",
                        (frames, fps, tracks_seen, events_emitted, status, run_id),
                    )
            except Exception as e:
                print(f"[POSTGRES] Run finish note: {e}")

    def publish(self, event: Dict[str, Any]) -> None:
        """Enqueues an event for background insertion without blocking the pipeline."""
        if not self.running:
            self.failed_count += 1
            if self.on_insert_failure:
                self.on_insert_failure(event, "postgres")
            return
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            self.failed_count += 1
            print("[POSTGRES WARNING] Queue is full (1000 items). Routing event to outbox.")
            if self.on_insert_failure:
                self.on_insert_failure(event, "postgres")

    def insert_sync(self, event: Dict[str, Any]) -> bool:
        """Synchronously inserts a single event into Postgres. Returns True on success."""
        ok = self._execute_insert(event)
        if ok:
            self.inserted_count += 1
        else:
            self.failed_count += 1
        return ok

    def get_run_events(self, run_id: str) -> Optional[List[Dict[str, Any]]]:
        """Queries events table for all rows belonging to run_id."""
        if not run_id:
            return []
        with self._conn_lock:
            if not self._ensure_connection():
                return None
            try:
                with self._conn.cursor() as cur:
                    cur.execute(
                        "SELECT event_id, vehicle_key, plate_key, s3_bucket FROM events WHERE run_id = %s;",
                        (run_id,),
                    )
                    return [
                        {"event_id": r[0], "vehicle_key": r[1], "plate_key": r[2], "s3_bucket": r[3]}
                        for r in cur.fetchall()
                    ]
            except Exception as e:
                print(f"[POSTGRES] Sync query error: {e}")
                return None

    def stop(self, timeout: float = 10.0) -> None:
        """Flushes remaining queue items within timeout and cleanly shuts down."""
        if not self.running:
            return

        self.running = False
        try:
            self.queue.put_nowait(None)  # Sentinel
        except Exception:
            pass

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=timeout)
        print(f"[POSTGRES] Saved {self.inserted_count} event(s) to Neon PostgreSQL (failed={self.failed_count}).")

    def close(self) -> None:
        """Closes any remaining connection held for run-end reporting."""
        with self._conn_lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    # ------------------------------------------------------------------
    # Schema & connection management
    # ------------------------------------------------------------------

    def _apply_schema(self) -> bool:
        """Executes sql/schema.sql on startup. Never retries on ImportError."""
        if not self.schema_path.exists():
            print(f"[POSTGRES] Schema file not found: {self.schema_path}")
            return False

        if not check_psycopg_available(announce=True):
            self.off_reason = "psycopg missing"
            return False

        import psycopg

        with open(self.schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        backoffs = [1, 2]
        last_err = None
        for delay in backoffs:
            try:
                with psycopg.connect(self.database_url, connect_timeout=10, autocommit=True) as conn:
                    with conn.cursor() as cur:
                        cur.execute(schema_sql)
                print("[POSTGRES] Schema successfully applied (cameras, runs, events).")
                return True
            except ImportError:
                self.off_reason = "psycopg missing"
                return False
            except Exception as e:
                last_err = e
                time.sleep(delay)

        self.off_reason = "connect failed"
        print(f"[POSTGRES ERROR] Failed to connect/apply schema: {last_err}")
        return False

    def _ensure_connection(self) -> bool:
        """Verifies or reconnects to Neon Postgres. Never retries on ImportError."""
        if self.init_failed:
            return False

        if self._conn is not None:
            try:
                with self._conn.cursor() as cur:
                    cur.execute("SELECT 1")
                return True
            except Exception:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

        if not check_psycopg_available(announce=True):
            self.init_failed = True
            self.off_reason = "psycopg missing"
            return False

        import psycopg

        backoffs = [1, 2]
        last_err = None
        for delay in backoffs:
            try:
                self._conn = psycopg.connect(self.database_url, connect_timeout=10, autocommit=True)
                return True
            except ImportError:
                self.init_failed = True
                self.off_reason = "psycopg missing"
                return False
            except Exception as e:
                last_err = e
                time.sleep(delay)

        print(f"[POSTGRES ERROR] Connection failed: {last_err}")
        return False

    # ------------------------------------------------------------------
    # Event insert helper & worker
    # ------------------------------------------------------------------

    @staticmethod
    def _build_insert_params(event: Dict[str, Any]) -> Tuple:
        event_date = event.get("date")
        event_time_str = event.get("time")
        try:
            dt = datetime.fromisoformat(f"{event_date}T{event_time_str}").astimezone()
        except Exception:
            dt = datetime.now().astimezone()

        return (
            event.get("event_id"),
            event.get("camera_id"),
            dt,
            event_date,
            event.get("snapshot_vehicle"),
            event.get("snapshot_number_plate"),
            event.get("plate_detection_confidence"),
            event.get("number_plate"),
            event.get("number_plate_confidence"),
            event.get("vehicle_type", "car"),
            event.get("run_id"),
            event.get("s3_bucket"),
            event.get("vehicle_key"),
            event.get("plate_key"),
        )

    def _execute_insert(self, event: Dict[str, Any]) -> bool:
        params = self._build_insert_params(event)
        try:
            with self._conn_lock:
                if not self._ensure_connection():
                    print(f"[POSTGRES ERROR] Cannot insert event {event.get('event_id')}: DB unreachable.")
                    return False
                try:
                    with self._conn.cursor() as cur:
                        cur.execute(_INSERT_EVENT_SQL, params)
                        if cur.rowcount == 0:
                            print(f"[ID-COLLISION] Event {event.get('event_id')} already exists in DB (ON CONFLICT DO NOTHING)")
                            # Route to outbox so event is not silently dropped
                            return False
                    return True
                except Exception:
                    try:
                        self._conn.close()
                    except Exception:
                        pass
                    self._conn = None
                    if not self._ensure_connection():
                        raise
                    with self._conn.cursor() as cur:
                        cur.execute(_INSERT_EVENT_SQL, params)
                        if cur.rowcount == 0:
                            print(f"[ID-COLLISION] Event {event.get('event_id')} already exists in DB (ON CONFLICT DO NOTHING)")
                            return False
                    return True
        except Exception as e:
            print(f"[POSTGRES ERROR] Failed to insert event {event.get('event_id')}: {e}")
            return False

    def _worker_loop(self) -> None:
        """Background thread pulling events from queue and inserting into Neon."""
        while self.running or not self.queue.empty():
            try:
                event = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if event is None:  # Shutdown sentinel
                self.queue.task_done()
                break

            try:
                if self._execute_insert(event):
                    self.inserted_count += 1
                    if self.on_insert_success:
                        try:
                            self.on_insert_success(event)
                        except Exception as cb_err:
                            print(f"[POSTGRES CALLBACK WARNING] {cb_err}")
                else:
                    self.failed_count += 1
                    if self.on_insert_failure:
                        try:
                            self.on_insert_failure(event, "postgres")
                        except Exception as cb_err:
                            print(f"[POSTGRES OUTBOX WARNING] {cb_err}")
            finally:
                self.queue.task_done()

        self.close()
