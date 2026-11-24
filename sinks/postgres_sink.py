"""
Neon PostgreSQL Event Sink for FastALPR Pipeline.
Handles background asynchronous database writes, auto-reconnection with backoff,
schema bootstrap, and non-blocking queue ingestion.
"""

import os
import re
import time
import queue
import threading
from pathlib import Path
from typing import Optional, Dict, Any


def mask_database_url(url: str) -> str:
    """Masks credentials in postgresql URL for safe logging."""
    if not url:
        return ""
    try:
        return re.sub(r":([^:@/]+)@", r":****@", url)
    except Exception:
        return "postgresql://****"


class PostgresSink:
    def __init__(self, database_url: str, schema_path: Optional[Path] = None, max_queue_size: int = 1000):
        self.database_url = database_url
        self.schema_path = schema_path or Path(__file__).resolve().parent.parent / "sql" / "schema.sql"
        self.queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        self._conn = None
        self.inserted_count: int = 0
        self.failed_count: int = 0

    def start(self) -> None:
        """Initializes schema and starts background ingestion worker."""
        if not self.database_url:
            return

        masked = mask_database_url(self.database_url)
        print(f"[POSTGRES] Connecting to Neon: {masked}")

        # Apply schema idempotently at startup
        self._apply_schema()

        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _apply_schema(self) -> None:
        """Executes sql/schema.sql on startup with retry for Neon serverless cold starts."""
        if not self.schema_path.exists():
            print(f"[POSTGRES] Schema file not found: {self.schema_path}")
            return

        with open(self.schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        backoffs = [1, 2, 4]
        last_err = None
        for attempt in range(len(backoffs)):
            try:
                import psycopg

                with psycopg.connect(self.database_url, connect_timeout=10, autocommit=True) as conn:
                    with conn.cursor() as cur:
                        cur.execute(schema_sql)
                print("[POSTGRES] Schema successfully applied (CREATE TABLE IF NOT EXISTS events).")
                return
            except Exception as e:
                last_err = e
                print(f"[POSTGRES] Schema attempt {attempt + 1}/{len(backoffs)} failed: {e}")
                time.sleep(backoffs[attempt])

        print(f"[POSTGRES ERROR] Failed to apply schema after retries: {last_err}")

    def _ensure_connection(self) -> bool:
        """Verifies or reconnects to Neon Postgres with exponential backoff (1s, 2s, 4s)."""
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

        retries = 3
        backoffs = [1, 2, 4]
        for attempt in range(retries):
            try:
                import psycopg

                self._conn = psycopg.connect(self.database_url, connect_timeout=10, autocommit=True)
                return True
            except Exception as e:
                print(f"[POSTGRES] Connection attempt {attempt + 1}/{retries} failed: {e}")
                time.sleep(backoffs[attempt])
        return False

    def publish(self, event: Dict[str, Any]) -> None:
        """Enqueues an event for background insertion without blocking the pipeline."""
        if not self.running:
            return
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            print("[POSTGRES WARNING] Queue is full (1000 items). Dropping DB push for event.")

    def _worker_loop(self) -> None:
        """Background thread worker pulling events from queue and inserting into Neon."""
        insert_sql = """
        INSERT INTO events (
            event_id,
            camera_id,
            event_time,
            event_date,
            snapshot_vehicle,
            snapshot_number_plate,
            plate_detection_confidence,
            number_plate,
            number_plate_confidence
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (event_id) DO NOTHING;
        """

        while self.running or not self.queue.empty():
            try:
                event = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if event is None:  # Shutdown sentinel
                break

            try:
                if not self._ensure_connection():
                    print(f"[POSTGRES ERROR] Cannot insert event {event.get('event_id')}: DB unreachable.")
                    self.failed_count += 1
                    continue

                # Prepare ISO timestamp from event date and time
                event_date = event.get("date")
                event_time = event.get("time")
                dt_iso = f"{event_date}T{event_time}" if (event_date and event_time) else "now()"
                params = (
                    event.get("event_id"),
                    event.get("camera_id"),
                    dt_iso,
                    event_date,
                    event.get("snapshot_vehicle"),
                    event.get("snapshot_number_plate"),
                    event.get("plate_detection_confidence"),
                    event.get("number_plate"),
                    event.get("number_plate_confidence"),
                )

                try:
                    with self._conn.cursor() as cur:
                        cur.execute(insert_sql, params)
                    self.inserted_count += 1
                except Exception:
                    # Neon idle/SSL disconnect mid-query: force reconnect and retry once
                    try:
                        self._conn.close()
                    except Exception:
                        pass
                    self._conn = None
                    if not self._ensure_connection():
                        raise
                    with self._conn.cursor() as cur:
                        cur.execute(insert_sql, params)
                    self.inserted_count += 1
            except Exception as e:
                self.failed_count += 1
                print(f"[POSTGRES ERROR] Failed to insert event {event.get('event_id')}: {e}")
            finally:
                self.queue.task_done()

        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

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
