# FastALPR Vehicle Detection & Tracking Pipeline

Single-event-per-vehicle ALPR system built with **YOLO11**, **ByteTrack**, and **FastALPR** (`open-image-models` + `fast-plate-ocr`), with fan-out persistence to **JSON**, **Neon PostgreSQL**, **Neon Object Storage (S3)**, and **WebSocket**.

## Project Layout

```text
├── run.py                  # CLI entry point
├── pipeline.py             # Single-pass YOLO + ByteTrack + FastALPR pipeline & rendering
├── tracker.py              # Track lifecycle, strict stitching vetoes, conf^4 OCR voting & snapshots
├── config.py               # PipelineConfig & .env loader
├── custom_bytetrack.yaml   # ByteTrack config
├── fast_alpr/              # Plate detection & OCR wrapper
├── sinks/                  # EventBus, PostgresSink (Neon DB), S3Sink (Neon S3), WebSocketSink
├── sql/schema.sql          # PostgreSQL events schema
├── tools/                  # check_gpu.py, check_db.py, check_s3.py
├── install.sh / install.bat
├── requirements.txt
├── .env.example
├── stream_2.mp4            # Test clip (Indian plates / GENERAL)
└── stream.mp4              # Test clip (UK plates)
```

## Quick Run (Google Colab T4 GPU)

```python
%cd /content
!find /content -mindepth 1 -maxdepth 1 ! -name 'drive' -exec rm -rf {} +
!git clone -b stable-v2 https://github.com/shaheer162008/vehicle_detector.git
%cd /content/vehicle_detector
!bash install.sh
!python run.py --source stream_2.mp4 --plate-format GENERAL
```

## Environment Configuration (`.env`)

Copy `.env.example` to `.env` and fill in your Neon credentials:

- `DATABASE_URL` — Neon PostgreSQL connection string (`postgresql://USER:PASS@HOST/DB?sslmode=require`)
- `AWS_ENDPOINT_URL_S3`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `S3_BUCKET` — Neon Object Storage (S3)
- `WS_PORT` — optional WebSocket broadcast port (default `8765`)

Verify GPU & Neon connections:

```bash
python tools/check_gpu.py
python tools/check_s3.py
python tools/check_db.py
```

## License

Built on [FastALPR](https://github.com/ankandrew/fast-alpr) (MIT). Vehicle detection uses Ultralytics YOLO (AGPL-3.0).
