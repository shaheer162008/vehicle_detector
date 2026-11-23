#!/usr/bin/env bash
# ==============================================================================
# FastALPR Vehicle Tracking Runner (Turnkey Stream + Detection)
# ==============================================================================

set -e
cd "$(dirname "$0")"

echo "=============================================================================="
echo " === FastALPR Vehicle Tracking Runner (Linux / macOS / Colab) ==="
echo "=============================================================================="

# 0. Load .env file if present
if [ -f ".env" ]; then
    echo "[CONFIG] Loading environment variables from .env..."
    set -a
    source .env
    set +a
fi

# 1. Environment Variable Defaults
STREAM_PORT="${STREAM_PORT:-8554}"
STREAM_VIDEO="${STREAM_VIDEO:-stream/stream.mp4}"
STREAM_URL="${STREAM_URL:-http://127.0.0.1:$STREAM_PORT/live}"
PLATE_FORMAT="${PLATE_FORMAT:-PAKISTAN}"
OUTPUT_VIDEO="${OUTPUT_VIDEO:-output/annotated_output.mp4}"

# Allow override via CLI argument 1
if [ -n "$1" ]; then
    STREAM_VIDEO="$1"
fi

# 2. Virtual Environment Check
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    if command -v python3 &> /dev/null; then
        PYTHON="python3"
    else
        PYTHON="python"
    fi
fi

# 3. Detect Active Compute Device (GPU Preferred)
COMPUTE_DEVICE=$($PYTHON -c "import torch; print(('GPU: ' + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'CPU (No CUDA detected)')" 2>/dev/null || echo "CPU")
echo "[COMPUTE ENGINE] $COMPUTE_DEVICE"
echo "[STREAM SOURCE]  $STREAM_VIDEO"
echo "[STREAM URL]     $STREAM_URL"
echo "[PLATE FORMAT]   $PLATE_FORMAT"

# 4. Check if stream video exists
if [ ! -f "$STREAM_VIDEO" ]; then
    echo "[ERROR] Stream video not found at: $STREAM_VIDEO"
    echo "Please make sure stream/stream.mp4 exists or pass your video path as an argument."
    exit 1
fi

# 5. Launch Live Stream Broadcaster in Background
echo ""
echo "[1/2] Starting Live Camera Stream Broadcaster on port $STREAM_PORT..."
$PYTHON stream/live_stream.py --video "$STREAM_VIDEO" --port "$STREAM_PORT" &
STREAM_PID=$!

cleanup() {
    echo ""
    echo "[CLEANUP] Stopping Live Camera Stream Broadcaster (PID $STREAM_PID)..."
    kill $STREAM_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wait 2 seconds for server to bind port
sleep 2

# 6. Launch ALPR Tracking Pipeline on the Live Stream
echo "[2/2] Launching FastALPR Pipeline on live camera stream..."
echo ""

set +e
$PYTHON run.py --source "$STREAM_URL" --plate-format "$PLATE_FORMAT" --output-video "$OUTPUT_VIDEO" --preview
RUN_EXIT_CODE=$?
set -e

echo ""
echo "=============================================================================="
echo " === Session Finished Successfully ==="
echo " Output Video : $OUTPUT_VIDEO"
echo " Snapshots    : output/snapshots/"
echo " Events JSON  : output/events.json"
echo "=============================================================================="
exit $RUN_EXIT_CODE
