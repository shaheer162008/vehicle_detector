#!/usr/bin/env python3
"""
End-to-End Neon Sync Verification Tool for FastALPR Pipeline.
Reads output/events.json and verifies:
  1) DB row exists in Neon Postgres `events` table
  2) `vehicle_key` and `plate_key` (if present) exist on Neon S3 via head_object
  3) Null-plate events have `plate_key IS NULL`
  4) Matching `runs` row has status='done' and events_emitted == len(events.json)

Exit 0 with "NEON VERIFY OK", or exit 1 with mismatch details.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Set

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env", override=True)
    env_local = project_root / ".env.local"
    if env_local.exists():
        load_dotenv(env_local, override=True)
except ImportError:
    pass

from sinks.postgres_sink import mask_database_url


def _mask_key(key: str) -> str:
    if not key or len(key) < 10:
        return "****"
    return f"{key[:4]}...{key[-4:]}"


def _head_object(s3_client, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify output/events.json against Neon Postgres and Neon S3")
    parser.add_argument("--events-json", type=str, default="output/events.json", help="Path to events.json")
    parser.add_argument("--run-id", type=str, default=None, help="Optional explicit run_id to verify in runs table")
    args = parser.parse_args()

    events_path = Path(args.events_json)
    if not events_path.is_absolute():
        # Resolve relative to CWD first, fallback to project_root
        if not events_path.exists():
            alt = project_root / events_path
            if alt.exists():
                events_path = alt

    if not events_path.exists():
        print(f"[VERIFY ERROR] Events file not found: {events_path}")
        return 1

    try:
        events = json.loads(events_path.read_text(encoding="utf-8"))
        if not isinstance(events, list):
            print(f"[VERIFY ERROR] Expected JSON array in {events_path}")
            return 1
    except Exception as e:
        print(f"[VERIFY ERROR] Failed to parse {events_path}: {e}")
        return 1

    database_url = os.environ.get("DATABASE_URL")
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL_S3")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    region = os.environ.get("AWS_REGION", "us-east-2")
    default_bucket = os.environ.get("S3_BUCKET", "alpr-snapshots")

    missing_env = [
        name
        for name, val in (
            ("DATABASE_URL", database_url),
            ("AWS_ENDPOINT_URL_S3", endpoint_url),
            ("AWS_ACCESS_KEY_ID", access_key),
            ("AWS_SECRET_ACCESS_KEY", secret_key),
        )
        if not val
    ]
    if missing_env:
        print(f"[VERIFY ERROR] Missing required environment variable(s): {', '.join(missing_env)}")
        return 1

    try:
        import psycopg
    except ImportError:
        print("[VERIFY ERROR] psycopg is not installed.")
        return 1

    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        print("[VERIFY ERROR] boto3 is not installed.")
        return 1

    print(
        f"[VERIFY] Postgres={mask_database_url(database_url)} | "
        f"S3={endpoint_url} key={_mask_key(access_key)} bucket={default_bucket}"
    )

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=Config(s3={"addressing_style": "path"}, signature_version="s3v4", connect_timeout=10, read_timeout=20),
        )
    except Exception as e:
        print(f"[VERIFY ERROR] Failed to create S3 client: {e}")
        return 1

    mismatches: List[str] = []
    found_run_ids: Set[str] = set()

    try:
        with psycopg.connect(database_url, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                for ev in events:
                    eid = ev.get("event_id")
                    cur.execute(
                        "SELECT event_id, run_id, s3_bucket, vehicle_key, plate_key, number_plate, vehicle_type "
                        "FROM events WHERE event_id = %s;",
                        (eid,),
                    )
                    row = cur.fetchone()
                    if row is None:
                        mismatches.append(f"{eid}: DB row missing")
                        continue

                    _, row_run_id, row_bucket, vehicle_key, plate_key, db_plate, db_vehicle_type = row
                    if row_run_id:
                        found_run_ids.add(row_run_id)
                    bucket = row_bucket or default_bucket

                    # Check vehicle_type NOT NULL
                    if not db_vehicle_type:
                        mismatches.append(f"{eid}: DB vehicle_type is NULL/empty")

                    # Check (2): vehicle_key in DB & S3
                    if not vehicle_key:
                        mismatches.append(f"{eid}: DB vehicle_key is NULL/empty")
                    elif not _head_object(s3, bucket, vehicle_key):
                        mismatches.append(f"{eid}: S3 head_object failed for vehicle_key={vehicle_key} (bucket={bucket})")

                    # Check (2) & (3): plate_key consistency
                    has_plate_text = ev.get("number_plate") is not None
                    has_plate_snap = ev.get("snapshot_number_plate") is not None
                    if not has_plate_text:
                        if plate_key is not None:
                            mismatches.append(
                                f"{eid}: number_plate is null in JSON but DB plate_key={plate_key!r} (expected NULL)"
                            )
                    elif has_plate_snap:
                        if not plate_key:
                            mismatches.append(f"{eid}: snapshot_number_plate={ev.get('snapshot_number_plate')!r} but DB plate_key is NULL")
                        elif not _head_object(s3, bucket, plate_key):
                            mismatches.append(
                                f"{eid}: S3 head_object failed for plate_key={plate_key} (bucket={bucket})"
                            )

                # Check (4): runs table status == 'done' and events_emitted == len(events)
                target_run_id: Optional[str] = args.run_id
                if not target_run_id:
                    if len(found_run_ids) == 1:
                        target_run_id = next(iter(found_run_ids))
                    elif len(found_run_ids) > 1:
                        mismatches.append(f"events.json rows span multiple run_ids: {sorted(found_run_ids)}")
                    elif len(events) == 0:
                        cur.execute("SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1;")
                        latest = cur.fetchone()
                        target_run_id = latest[0] if latest else None
                    else:
                        mismatches.append("no run_id recorded on matched DB events")

                if target_run_id:
                    cur.execute("SELECT status, events_emitted FROM runs WHERE run_id = %s;", (target_run_id,))
                    run_row = cur.fetchone()
                    if run_row is None:
                        mismatches.append(f"runs row missing for run_id={target_run_id}")
                    else:
                        run_status, events_emitted = run_row
                        if run_status != "done":
                            mismatches.append(f"runs.status={run_status!r} for run_id={target_run_id} (expected 'done')")
                        if events_emitted != len(events):
                            mismatches.append(
                                f"runs.events_emitted={events_emitted} != len(events.json)={len(events)} "
                                f"for run_id={target_run_id}"
                            )
    except Exception as e:
        print(f"[VERIFY ERROR] Database verification failed: {e}")
        return 1

    if mismatches:
        print(f"[VERIFY] Found {len(mismatches)} mismatch(es):")
        for m in mismatches:
            print(f"  - {m}")
        return 1

    print(f"NEON VERIFY OK (events={len(events)}, run_id={target_run_id or 'none'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
