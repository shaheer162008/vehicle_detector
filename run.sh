#!/usr/bin/env bash
# FastALPR Runner (Linux/macOS/Colab)

set -e

cd "$(dirname "$0")"

echo "=== FastALPR Runner ==="

# Check for .venv
if [ ! -f ".venv/bin/python" ] && [ -z "$COLAB_GPU" ] && [ -z "$COLAB_RELEASE_TAG" ] && [ ! -d "/content" ]; then
    echo "[ERROR] .venv not found. Run ./setup.sh first."
    exit 1
fi

if [ -f ".venv/bin/python" ]; then
    VENV_PY=".venv/bin/python"
else
    VENV_PY="python3"
fi

# Run selftest first
echo ""
echo "Running self-test..."
$VENV_PY tools/selftest.py
if [ $? -ne 0 ]; then
    echo ""
    echo "[WARN] Self-test had failures."
    read -p "Continue anyway? [Y/n]: " CONTINUE
    if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

echo ""
echo "Choose mode:"
echo "  [1] Detection -> Neon (CLI mode, python run.py)"
echo "  [2] Service for Next.js frontend (uvicorn service.app:app)"
echo "  [3] Self-test only"
echo ""
read -p "Select [1-3]: " MODE

if [ "$MODE" = "3" ]; then
    echo "Self-test already done. Exiting."
    exit 0
fi

if [ "$MODE" = "2" ]; then
    # Service mode
    echo ""
    echo "Starting FastAPI service for Next.js frontend..."
    echo ""
    echo "Service will run on http://127.0.0.1:8000"
    echo "API docs: http://127.0.0.1:8000/docs"
    echo "WebSocket events: ws://127.0.0.1:8000/ws/events?token=***"
    echo "WebSocket frames: ws://127.0.0.1:8000/ws/frames/{stream_id}?token=***"
    echo ""
    echo "Press Ctrl+C to stop."
    echo ""
    $VENV_PY -m uvicorn service.app:app --host 127.0.0.1 --port 8000
    exit 0
fi

# Mode 1: CLI detection
echo ""
echo "CLI Detection Mode"
echo ""

# Check camera 1 RTSP URL
$VENV_PY -c "
import os
from dotenv import load_dotenv
load_dotenv('.env', override=True)
rtsp = os.environ.get('CAMERA_1_RTSP_URL', '')
if rtsp:
    print('[INFO] CAMERA_1_RTSP_URL:', rtsp)
    if rtsp.startswith('rtsp://127.0.0.1:8554'):
        print('[HINT] This is a local stream. Make sure stream.sh is running!')
else:
    print('[WARN] CAMERA_1_RTSP_URL not set in .env')
    print('[HINT] Run ./stream.sh first to create local RTSP stream')
"

echo ""
read -p "Run for how many minutes? (Enter = unlimited): " MINUTES
MAX_MINUTES=""
if [ -n "$MINUTES" ]; then
    MAX_MINUTES="--max-minutes $MINUTES"
fi

# Check if preview wanted (desktop)
PREVIEW_FLAG="--no-preview"
if [ -z "$COLAB_GPU" ] && [ -z "$COLAB_RELEASE_TAG" ] && [ ! -d "/content" ]; then
    read -p "Show preview window? [y/N]: " PREVIEW
    if [[ "$PREVIEW" =~ ^[Yy]$ ]]; then
        PREVIEW_FLAG=""
    fi
fi

# Determine plate format from .env
PLATE_FMT=$($VENV_PY -c "
import os
from dotenv import load_dotenv
load_dotenv('.env', override=True)
fmt = os.environ.get('PLATE_FORMAT', 'GENERAL')
print(fmt)
")

# Run detection
echo ""
echo "Starting detection..."
echo "Command: $VENV_PY run.py --source \"\$CAMERA_1_RTSP_URL\" --plate-format $PLATE_FMT $MAX_MINUTES $PREVIEW_FLAG"
echo ""

# Export CAMERA_1_RTSP_URL for the command
export CAMERA_1_RTSP_URL=$($VENV_PY -c "
import os
from dotenv import load_dotenv
load_dotenv('.env', override=True)
print(os.environ.get('CAMERA_1_RTSP_URL', ''))
")

$VENV_PY run.py --source "$CAMERA_1_RTSP_URL" --plate-format "$PLATE_FMT" $MAX_MINUTES $PREVIEW_FLAG

echo ""
echo "Detection finished."
read -p "Stop stream processes too? [Y/n]: " STOP_STREAM
if [[ ! "$STOP_STREAM" =~ ^[Nn]$ ]]; then
    ./stop.sh
fi