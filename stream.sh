#!/usr/bin/env bash
# FastALPR Stream Starter (Linux/macOS/Colab)

set -e

cd "$(dirname "$0")"

echo "=== FastALPR Stream Starter ==="

# Check for .venv
if [ ! -f ".venv/bin/python" ]; then
    echo "[ERROR] .venv not found. Run ./setup.sh first."
    exit 1
fi

VENV_PY=".venv/bin/python"
MEDIAMTX="bin/mediamtx"
FFMPEG="ffmpeg"

# Check MediaMTX
if [ ! -f "$MEDIAMTX" ]; then
    echo "[ERROR] MediaMTX not found at $MEDIAMTX. Run ./setup.sh first."
    exit 1
fi

# Check ffmpeg
if ! command -v "$FFMPEG" &>/dev/null; then
    echo "[ERROR] ffmpeg not in PATH. Install ffmpeg or run ./setup.sh"
    exit 1
fi

# 1. Choose source
echo ""
echo "Choose video source:"
echo "  [1] stream_2.mp4 (demo)"
echo "  [2] stream.mp4 (demo)"
echo "  [3] Custom file path"
echo "  [4] Real camera RTSP URL (type manually - hidden input)"
echo "  [Enter] Skip - keep existing CAMERA_1_RTSP_URL in .env"
echo ""
read -p "Select [1-4, Enter]: " CHOICE

SOURCE_FILE=""
IS_RTSP=0
CUSTOM_URL=""

case "$CHOICE" in
    1) SOURCE_FILE="stream_2.mp4" ;;
    2) SOURCE_FILE="stream.mp4" ;;
    3) read -p "Enter file path (relative to repo root): " SOURCE_FILE ;;
    4)
        IS_RTSP=1
        echo "Enter RTSP URL (hidden input - password won't show):"
        read -s CUSTOM_URL
        echo ""
        ;;
    "")
        echo "[INFO] Keeping existing CAMERA_1_RTSP_URL in .env"
        UPDATE_ENV_ONLY=1
        ;;
esac

if [ "$CHOICE" = "" ]; then
    UPDATE_ENV_ONLY=1
fi

if [ "$IS_RTSP" -eq 1 ]; then
    if [ -z "$CUSTOM_URL" ]; then
        echo "[ERROR] No RTSP URL entered"
        exit 1
    fi
    if [[ ! "$CUSTOM_URL" =~ ^rtsp:// ]]; then
        echo "[WARN] URL doesn't start with rtsp://"
    fi
else
    if [ ! -f "$SOURCE_FILE" ]; then
        echo "[ERROR] File not found: $SOURCE_FILE"
        exit 1
    fi
fi

# 2. Start MediaMTX + ffmpeg
if [ "$UPDATE_ENV_ONLY" = "1" ]; then
    echo "[INFO] Keeping existing CAMERA_1_RTSP_URL in .env"
else
    echo ""
    echo "Starting RTSP stream..."

    if [ "$IS_RTSP" -eq 1 ]; then
        # Real camera - just update .env, don't start local stream
        echo "[INFO] Real camera mode - updating .env only"
    else
        # Demo file - start MediaMTX + ffmpeg loop
        if [ ! -f "$SOURCE_FILE" ]; then
            echo "[ERROR] Source file not found: $SOURCE_FILE"
            exit 1
        fi

        # Start MediaMTX in background
        echo "Starting MediaMTX..."
        "$MEDIAMTX" > logs/mediamtx.log 2>&1 &
        MEDIAMTX_PID=$!
        echo $MEDIAMTX_PID > .run/stream_pids.txt

        # Wait for MediaMTX to be ready (port 8554)
        echo "Waiting for MediaMTX on port 8554..."
        for i in {1..30}; do
            if nc -z 127.0.0.1 8554 2>/dev/null; then
                break
            fi
            sleep 1
        done
        if ! nc -z 127.0.0.1 8554 2>/dev/null; then
            echo "[ERROR] MediaMTX failed to start on port 8554"
            kill $MEDIAMTX_PID 2>/dev/null
            exit 1
        fi
        echo "MediaMTX ready."

        # Start ffmpeg in background (loop file -> RTSP)
        echo "Starting ffmpeg..."
        $FFMPEG -hide_banner -loglevel warning -re -stream_loop -1 -i "$SOURCE_FILE" -c copy -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/cam1 > logs/ffmpeg.log 2>&1 &
        FFMPEG_PID=$!
        echo $FFMPEG_PID >> .run/stream_pids.txt

        echo "Stream started!"
        echo "  RTSP URL: rtsp://127.0.0.1:8554/cam1"
        echo "  HLS URL:  http://127.0.0.1:8888/cam1"
        echo "  WebRTC:   http://127.0.0.1:8889/cam1"
    fi

    # 3. Update .env with CAMERA_1_RTSP_URL
    RTSP_URL="rtsp://127.0.0.1:8554/cam1"
    if [ "$IS_RTSP" -eq 1 ]; then
        RTSP_URL="$CUSTOM_URL"
    fi

    $VENV_PY -c "
import os, re
env_path = '.env'
rtsp_url = os.environ.get('RTSP_URL', 'rtsp://127.0.0.1:8554/cam1')
if os.path.exists(env_path):
    with open(env_path, 'r') as f: lines = f.readlines()
else:
    lines = []
new_lines = []
found = False
for line in lines:
    if line.strip().startswith('CAMERA_1_RTSP_URL'):
        new_lines.append('CAMERA_1_RTSP_URL=' + rtsp_url + '\n')
        found = True
    else:
        new_lines.append(line)
if not found:
    new_lines.append('CAMERA_1_RTSP_URL=' + rtsp_url + '\n')
with open(env_path + '.tmp', 'w') as f: f.writelines(new_lines)
os.replace(env_path + '.tmp', env_path)
print('[OK] .env updated: CAMERA_1_RTSP_URL=' + rtsp_url)
" RTSP_URL="$RTSP_URL"

    # Save PIDs for stop script
    if [ "$IS_RTSP" -eq 0 ]; then
        echo $MEDIAMTX_PID > .run/stream_pids.txt
        echo $FFMPEG_PID >> .run/stream_pids.txt
    fi

    echo ""
    echo "Stream started!"
    echo "  RTSP URL: rtsp://127.0.0.1:8554/cam1"
    echo "  HLS URL:  http://127.0.0.1:8888/cam1"
    echo "  WebRTC:   http://127.0.0.1:8889/cam1"
    echo ""
    echo ".env updated with CAMERA_1_RTSP_URL"
    echo ""
    echo "To stop: run ./stop.sh"
fi