CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY,
  camera_id TEXT NOT NULL,
  event_time TIMESTAMPTZ NOT NULL,
  event_date DATE NOT NULL,
  snapshot_vehicle TEXT NOT NULL,
  snapshot_number_plate TEXT,
  plate_detection_confidence REAL,
  number_plate TEXT,
  number_plate_confidence REAL,
  analysis_status TEXT NOT NULL DEFAULT 'pending',   -- pending | processing | done | failed  (Stage 2 AI model ke liye)
  analysis JSONB,                                    -- Stage 2 output (make/model/colour) yahan likha jayega
  analysis_updated_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_events_cam_time ON events (camera_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_events_plate ON events (number_plate);
CREATE INDEX IF NOT EXISTS idx_events_pending ON events (analysis_status) WHERE analysis_status = 'pending';
