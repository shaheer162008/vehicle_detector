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

## Neon setup
1. Run `sql/schema.sql` once in Neon SQL Editor (auto-applied at startup too).
2. Copy `.env.example` to `.env` and fill `DATABASE_URL`, `AWS_*`, `S3_BUCKET` (plus `AI_GATEWAY_*` for Stage 2).
3. Run the pipeline: `python run.py --source stream_2.mp4 --plate-format GENERAL`.
4. Verify DB + S3 sync: `python tools/verify_neon.py`.
5. Optional Stage 2 AI analysis: `python tools/stage2_worker.py`.

## Service (FastAPI + WebSocket Relay)

A long-running control service that Next.js frontends can use to create/start/stop streams, receive real-time events, and preview annotated frames.

```bash
# 1. Install (creates .venv, installs deps, verifies GPU)
./install.sh          # Linux/macOS
.\install.bat         # Windows

# 2. Configure (copy .env.example to .env and fill Neon credentials)
cp .env.example .env
# edit .env with DATABASE_URL, AWS_ENDPOINT_URL_S3, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET

# 3. Start service (loads .env, runs FastAPI on port 8000)
./serve.sh            # Linux/macOS
.\serve.bat           # Windows
```

**REST API** (all endpoints require `X-API-KEY: <SERVICE_TOKEN>` header):
- `GET /health` — GPU, CUDA, ONNX providers, Python version
- `GET /streams` — list streams
- `POST /streams` — create stream `{name, source:{type:"file"|"rtsp", file?, url?}, loop:bool, camera_id, plate_format, vehicle_weights?}` → returns `stream_id`
- `PATCH /streams/{id}` — update loop, name, plate_format
- `DELETE /streams/{id}` — stop & delete
- `POST /streams/{id}/start|stop|restart` — control lifecycle
- `GET /streams/{id}/status` — `{state, fps, frames, events, last_event, uptime_s, restarts, last_error, rtsp_url?}`
- `GET /videos` — list files in `VIDEO_DIR`

**WebSocket** (pass `?token=<SERVICE_TOKEN>`):
- `ws://host:8000/ws/events?stream_id=<id>` — last 10 events + live (includes `vehicle_type`)
- `ws://host:8000/ws/frames/{stream_id}` — binary JPEG annotated frames (latest only, ~10 fps)

**RTSP Testing** (optional, requires `ffmpeg` + `mediamtx`):
```bash
# Start mediamtx (RTSP server)
mediamtx

# Create stream with publish_rtsp=true, then start it
curl -X POST localhost:8000/streams -H "X-API-KEY: $SERVICE_TOKEN" \
  -d '{"name":"test","source":{"type":"file","file":"stream_2.mp4"},"loop":true,"camera_id":"CAM-00001","plate_format":"GENERAL","publish_rtsp":true}'
# Start the stream via POST /streams/{id}/start
```

**Benchmark** (runs pipeline as subprocess, samples GPU):
```bash
python tools/benchmark.py --source stream_2.mp4 --seconds 60
```

## Multi-Camera Support (Up to 40)

The system is designed for up to 40 cameras. Camera configurations are loaded from `cameras.json` (40-camera template) and can be overridden via environment variables:

- `CAMERA_1_RTSP_URL`, `CAMERA_1_NAME`, `CAMERA_1_LOCATION`
- ... up to `CAMERA_40_RTSP_URL`, `CAMERA_40_NAME`, `CAMERA_40_LOCATION`

API endpoint: `GET /cameras` (requires `X-API-KEY`) returns all configured cameras with masked RTSP URLs.

**Note:** VRAM/CPU limits for 40 simultaneous streams depend on hardware. Benchmark with your target resolution/FPS before deploying all 40. Typical GPU (RTX 5060 8GB / Colab T4 16GB) handles 4-8 concurrent streams at 1080p. Scale horizontally (multiple nodes) for 40 cameras.

## Colab / Headless Usage

In Google Colab (or any headless environment), scripts adapt automatically:

- `setup.sh` skips virtual environment creation, uses system Python
- `stream.sh` runs MediaMTX/ffmpeg in background with logs to `logs/`
- `run.sh` defaults to `--no-preview`, sends data to Neon/S3
- No WebSocket tunneling needed - frontend reads from Neon separately

```bash
# Colab one-liner
!git clone -b release-v1 https://github.com/shaheer162008/vehicle_detector.git
%cd vehicle_detector
!bash setup.sh
# Edit .env with your Neon creds
!python run.py --source stream_2.mp4 --plate-format GENERAL --no-preview
```

## License

Built on [FastALPR](https://github.com/ankandrew/fast-alpr) (MIT). Vehicle detection uses Ultralytics YOLO (AGPL-3.0).
