#!/usr/bin/env python3
"""
Standalone Stage 2 AI Vision Worker for FastALPR.
Polls Neon Postgres `events` for analysis_status='pending', fetches vehicle
snapshots from Neon S3, sends them to Neon AI Gateway (OpenAI-compatible),
and stores structured JSON analysis back into Postgres.

Has no runtime dependency on the video tracking pipeline.
"""

import argparse
import base64
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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

PROMPT_TEXT = "Is gaadi ka make, model, colour, vehicle_type strict JSON mein do; pata na ho to null."

CLAIM_SQL = """
UPDATE events
SET analysis_status = 'processing',
    analysis_attempts = analysis_attempts + 1
WHERE event_id IN (
  SELECT event_id
  FROM events
  WHERE analysis_status = 'pending'
  ORDER BY created_at
  LIMIT 10
  FOR UPDATE SKIP LOCKED
)
RETURNING event_id, vehicle_key, s3_bucket, analysis_attempts;
"""


def _mask_secret(secret: str) -> str:
    if not secret or len(secret) < 10:
        return "****"
    return f"{secret[:4]}...{secret[-4:]}"


def _normalize_gateway_base_url(raw_url: str) -> str:
    base = raw_url.strip().rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def _parse_strict_json(raw_content: str) -> Dict[str, Any]:
    text = (raw_content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]

    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("AI response is not a JSON object")

    colour = parsed.get("colour") if "colour" in parsed else parsed.get("color")
    return {
        "make": parsed.get("make"),
        "model": parsed.get("model"),
        "colour": colour,
        "vehicle_type": parsed.get("vehicle_type"),
    }


def _reset_processing_rows(database_url: str, event_ids: List[str]) -> None:
    """Resets still-processing rows claimed by this worker back to 'pending' on Ctrl+C."""
    if not event_ids:
        return
    try:
        import psycopg

        with psycopg.connect(database_url, connect_timeout=10, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE events SET analysis_status = 'pending' "
                    "WHERE event_id = ANY(%s) AND analysis_status = 'processing';",
                    (event_ids,),
                )
        print(f"[STAGE2] Reset {len(event_ids)} in-flight row(s) back to 'pending'.")
    except Exception as e:
        print(f"[STAGE2 ERROR] Could not reset processing rows on exit: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2 Neon AI Gateway Worker")
    parser.add_argument("--poll-interval", type=float, default=3.0, help="Seconds to wait when idle (default: 3.0)")
    parser.add_argument("--once", action="store_true", default=False, help="Process available pending events once and exit")
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL")
    s3_endpoint = os.environ.get("AWS_ENDPOINT_URL_S3")
    s3_key = os.environ.get("AWS_ACCESS_KEY_ID")
    s3_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    s3_region = os.environ.get("AWS_REGION", "us-east-2")
    default_bucket = os.environ.get("S3_BUCKET", "alpr-snapshots")

    ai_base_raw = os.environ.get("AI_GATEWAY_BASE_URL") or os.environ.get("NEON_AI_GATEWAY_BASE_URL")
    ai_api_key = os.environ.get("AI_GATEWAY_API_KEY") or os.environ.get("NEON_AI_GATEWAY_TOKEN")
    ai_model = os.environ.get("AI_MODEL", "gpt-5-mini")

    missing = [
        name
        for name, val in (
            ("DATABASE_URL", database_url),
            ("AWS_ENDPOINT_URL_S3", s3_endpoint),
            ("AWS_ACCESS_KEY_ID", s3_key),
            ("AWS_SECRET_ACCESS_KEY", s3_secret),
            ("AI_GATEWAY_BASE_URL", ai_base_raw),
            ("AI_GATEWAY_API_KEY", ai_api_key),
        )
        if not val
    ]
    if missing:
        print(f"[STAGE2 ERROR] Missing environment variable(s): {', '.join(missing)}")
        return 1

    try:
        import psycopg
        import boto3
        from botocore.config import Config
        from openai import OpenAI
    except ImportError as e:
        print(f"[STAGE2 ERROR] Missing dependency: {e}")
        return 1

    ai_base_url = _normalize_gateway_base_url(ai_base_raw)
    print(
        f"[STAGE2] Starting worker | DB={mask_database_url(database_url)} | "
        f"AI={ai_base_url} model={ai_model} key={_mask_secret(ai_api_key)}"
    )

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_key,
        aws_secret_access_key=s3_secret,
        region_name=s3_region,
        config=Config(s3={"addressing_style": "path"}, signature_version="s3v4", connect_timeout=10, read_timeout=30),
    )
    ai_client = OpenAI(api_key=ai_api_key, base_url=ai_base_url)

    stop_requested = False

    def _handle_sigterm(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, _handle_sigterm)

    in_flight_ids: List[str] = []
    conn = None

    def _get_conn():
        nonlocal conn
        if conn is not None:
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                return conn
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
        conn = psycopg.connect(database_url, connect_timeout=10, autocommit=True)
        return conn

    try:
        while not stop_requested:
            db = _get_conn()
            with db.cursor() as cur:
                cur.execute(CLAIM_SQL)
                rows = cur.fetchall()

            if not rows:
                if args.once:
                    break
                time.sleep(args.poll_interval)
                continue

            in_flight_ids = [r[0] for r in rows]

            for event_id, vehicle_key, row_bucket, attempts in rows:
                if stop_requested:
                    break
                bucket = row_bucket or default_bucket
                try:
                    if not vehicle_key:
                        raise ValueError("vehicle_key is NULL/empty")

                    obj = s3.get_object(Bucket=bucket, Key=vehicle_key)
                    img_bytes = obj["Body"].read()
                    b64_img = base64.b64encode(img_bytes).decode("ascii")
                    mime = "image/jpeg" if str(vehicle_key).lower().endswith((".jpg", ".jpeg")) else "image/png"

                    resp = ai_client.chat.completions.create(
                        model=ai_model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Return ONLY a strict JSON object with keys: "
                                    "make, model, colour, vehicle_type. Use null if unknown."
                                ),
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": PROMPT_TEXT},
                                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_img}"}},
                                ],
                            },
                        ],
                    )
                    raw_text = resp.choices[0].message.content or ""
                    analysis = _parse_strict_json(raw_text)

                    db = _get_conn()
                    with db.cursor() as cur:
                        cur.execute(
                            "UPDATE events SET analysis = %s::jsonb, analysis_status = 'done', "
                            "analysis_updated_at = now(), analysis_error = NULL "
                            "WHERE event_id = %s;",
                            (json.dumps(analysis), event_id),
                        )
                    print(f"[STAGE2] {event_id} -> done: {analysis}")
                except Exception as err:
                    err_msg = str(err)[:500]
                    next_status = "pending" if (attempts or 0) < 3 else "failed"
                    try:
                        db = _get_conn()
                        with db.cursor() as cur:
                            cur.execute(
                                "UPDATE events SET analysis_status = %s, analysis_error = %s, "
                                "analysis_updated_at = now() WHERE event_id = %s;",
                                (next_status, err_msg, event_id),
                            )
                    except Exception as db_err:
                        print(f"[STAGE2 ERROR] Failed recording error for {event_id}: {db_err}")
                    print(f"[STAGE2] {event_id} -> {next_status} (attempt={attempts}): {err_msg}")
                finally:
                    if event_id in in_flight_ids:
                        in_flight_ids.remove(event_id)

            if args.once:
                break

    except KeyboardInterrupt:
        print("\n[STAGE2] Interrupted (Ctrl+C). Cleaning up...")
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        if in_flight_ids:
            _reset_processing_rows(database_url, in_flight_ids)

    return 0


if __name__ == "__main__":
    sys.exit(main())
