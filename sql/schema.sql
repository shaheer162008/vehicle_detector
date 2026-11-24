-- ============================================================================
-- FastALPR Vehicle Tracking - COMPLETE Neon Postgres setup (idempotent)
-- Run once in Neon SQL Editor (or auto-applied by the pipeline at startup).
-- Safe to re-run; upgrades an older `events` table in place.
-- ============================================================================

-- 1. CAMERAS (optional registry; events do NOT hard-depend on it)
CREATE TABLE IF NOT EXISTS cameras (
  camera_id   TEXT PRIMARY KEY,
  name        TEXT,
  location    TEXT,
  node_id     TEXT,
  is_active   BOOLEAN     NOT NULL DEFAULT TRUE,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 2. RUNS (one row per pipeline run: proof of completeness)
CREATE TABLE IF NOT EXISTS runs (
  run_id          TEXT PRIMARY KEY,
  camera_id       TEXT,
  source          TEXT,
  started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  finished_at     TIMESTAMPTZ,
  frames          INTEGER,
  fps             REAL,
  tracks_seen     INTEGER,
  events_emitted  INTEGER,
  status          TEXT NOT NULL DEFAULT 'running'
);

-- 3. EVENTS (one row per vehicle)
CREATE TABLE IF NOT EXISTS events (
  event_id                    TEXT PRIMARY KEY,
  camera_id                   TEXT        NOT NULL,
  event_time                  TIMESTAMPTZ NOT NULL,
  event_date                  DATE        NOT NULL,
  snapshot_vehicle            TEXT        NOT NULL,
  snapshot_number_plate       TEXT,
  plate_detection_confidence  REAL,
  number_plate                TEXT,
  number_plate_confidence     REAL,
  analysis_status             TEXT        NOT NULL DEFAULT 'pending',
  analysis                    JSONB,
  analysis_updated_at         TIMESTAMPTZ,
  created_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 3b. In-place upgrade for older tables (all idempotent)
ALTER TABLE events ADD COLUMN IF NOT EXISTS run_id            TEXT;
ALTER TABLE events ADD COLUMN IF NOT EXISTS s3_bucket         TEXT;
ALTER TABLE events ADD COLUMN IF NOT EXISTS vehicle_key       TEXT;
ALTER TABLE events ADD COLUMN IF NOT EXISTS plate_key         TEXT;
ALTER TABLE events ADD COLUMN IF NOT EXISTS analysis_error    TEXT;
ALTER TABLE events ADD COLUMN IF NOT EXISTS analysis_attempts INTEGER NOT NULL DEFAULT 0;
ALTER TABLE events ADD COLUMN IF NOT EXISTS has_plate BOOLEAN GENERATED ALWAYS AS (number_plate IS NOT NULL) STORED;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'events_run_id_fk'
  ) THEN
    ALTER TABLE events
      ADD CONSTRAINT events_run_id_fk
        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE SET NULL;
  END IF;
EXCEPTION WHEN others THEN NULL;
END $$;

-- 4. Indexes (all idempotent)
CREATE INDEX IF NOT EXISTS idx_events_cam_time     ON events (camera_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_events_plate        ON events (number_plate) WHERE number_plate IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_events_pending      ON events (analysis_status) WHERE analysis_status = 'pending';
CREATE INDEX IF NOT EXISTS idx_events_run          ON events (run_id)         WHERE run_id IS NOT NULL;

-- 5. Dashboard convenience view
CREATE OR REPLACE VIEW v_latest_events AS
SELECT
  e.event_id,
  e.camera_id,
  c.name        AS camera_name,
  c.location    AS camera_location,
  e.event_time,
  e.event_date,
  e.number_plate,
  e.number_plate_confidence,
  e.plate_detection_confidence,
  e.snapshot_vehicle,
  e.snapshot_number_plate,
  e.vehicle_key,
  e.plate_key,
  e.s3_bucket,
  e.run_id,
  e.analysis_status,
  e.analysis,
  e.created_at
FROM events e
LEFT JOIN cameras c ON c.camera_id = e.camera_id
ORDER BY e.event_time DESC;
