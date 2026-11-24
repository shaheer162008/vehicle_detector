#!/usr/bin/env python3
"""
Neon Database Diagnostic Tool for FastALPR Pipeline.
Verifies DATABASE_URL connection, applies schema.sql, performs test insert/read/delete.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Try loading .env if available
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    pass

database_url = os.environ.get("DATABASE_URL")
if not database_url:
    print("[ERROR] DATABASE_URL environment variable is not set.")
    print("        Please define DATABASE_URL in .env or your environment.")
    print("        Example: postgresql://USER:PASS@HOST/DB?sslmode=require")
    sys.exit(1)

from sinks.postgres_sink import mask_database_url

masked = mask_database_url(database_url)
print(f"[CHECK DB] Connecting to database: {masked}")

try:
    import psycopg
except ImportError:
    print("[ERROR] psycopg library is not installed. Please install psycopg[binary]>=3.1")
    sys.exit(1)

# 1. Connect and verify
try:
    conn = psycopg.connect(database_url, connect_timeout=10, autocommit=True)
except Exception as e:
    print(f"[ERROR] Failed to connect to database: {e}")
    sys.exit(1)

# 2. Apply sql/schema.sql
schema_file = project_root / "sql" / "schema.sql"
if not schema_file.exists():
    print(f"[ERROR] Schema file not found: {schema_file}")
    conn.close()
    sys.exit(1)

with open(schema_file, "r", encoding="utf-8") as f:
    schema_sql = f.read()

try:
    with conn.cursor() as cur:
        cur.execute(schema_sql)
    print("[CHECK DB] Schema applied successfully.")
except Exception as e:
    print(f"[ERROR] Failed applying schema.sql: {e}")
    conn.close()
    sys.exit(1)

# 3. Test Row Insert, Read, Delete
test_id = f"TEST_CHECK_{int(datetime.now().timestamp())}"
test_date = datetime.now().strftime("%Y-%m-%d")
test_time = datetime.now().strftime("%H:%M:%S")
test_iso = f"{test_date}T{test_time}"

try:
    with conn.cursor() as cur:
        # Insert
        cur.execute(
            """
            INSERT INTO events (
                event_id, camera_id, event_time, event_date,
                snapshot_vehicle, snapshot_number_plate,
                plate_detection_confidence, number_plate, number_plate_confidence
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (test_id, "CAM_TEST", test_iso, test_date, "test_veh.png", "test_plate.png", 0.99, "TEST999", 0.98),
        )

        # Read
        cur.execute("SELECT event_id, number_plate FROM events WHERE event_id = %s", (test_id,))
        row = cur.fetchone()
        if not row or row[0] != test_id or row[1] != "TEST999":
            raise RuntimeError(f"Read verification failed for row: {row}")

        # Delete
        cur.execute("DELETE FROM events WHERE event_id = %s", (test_id,))

    conn.close()
    print("DB OK")
    sys.exit(0)
except Exception as e:
    print(f"[ERROR] Test row execution failed: {e}")
    try:
        conn.close()
    except Exception:
        pass
    sys.exit(1)
