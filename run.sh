#!/usr/bin/env bash
# ==============================================================================
# FastALPR Vehicle Tracking Runner (Linux / macOS / Colab)
# Inspects stream video, updates .env, prompts plate format, and runs pipeline
# ==============================================================================

set -e
cd "$(dirname "$0")"

echo "=============================================================================="
echo " === FastALPR Vehicle Tracking Runner (Linux / macOS / Colab) ==="
echo "=============================================================================="

# 1. Check Virtual Environment
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    if command -v python3 &> /dev/null; then
        PYTHON="python3"
    else
        PYTHON="python"
    fi
fi

# 2. Detect Video File
STREAM_VIDEO=""
if [ -n "$1" ]; then
    STREAM_VIDEO="$1"
else
    if [ -f "stream/stream.mp4" ]; then
        STREAM_VIDEO="stream/stream.mp4"
    elif [ -f "stream/stream_2.mp4" ]; then
        STREAM_VIDEO="stream/stream_2.mp4"
    else
        echo "[ERROR] No video found in stream/ folder!"
        echo "Please place a video as stream/stream.mp4 or pass its path as an argument."
        exit 1
    fi
fi

# 3. Ask User for Number Plate Format Selection (Default: PAKISTAN)
echo ""
echo "=============================================================================="
echo " Select License Plate Format:"
echo "   [1] PAKISTAN  (Default - Punjab, Sindh, Islamabad, KPK, etc.)"
echo "   [2] GENERAL   (Universal Alphanumeric Standard)"
echo "   [3] UK        (British Standard Format)"
echo "=============================================================================="

SELECTED_PLATE_FORMAT="PAKISTAN"
if [ -t 0 ]; then
    read -r -p "Enter selection [1-3, Default=1]: " PLATE_CHOICE || true
    case "$PLATE_CHOICE" in
        2) SELECTED_PLATE_FORMAT="GENERAL" ;;
        3) SELECTED_PLATE_FORMAT="UK" ;;
        *) SELECTED_PLATE_FORMAT="PAKISTAN" ;;
    esac
else
    echo "[INFO] Non-interactive environment detected. Using default: PAKISTAN"
fi

STREAM_PORT="8554"

# 4. Inspect Video & Auto-Configure Environment Variables in .env
echo ""
echo "[INSPECTING] Checking stream video properties..."
$PYTHON stream/inspect_stream.py "$STREAM_VIDEO" "$SELECTED_PLATE_FORMAT" "$STREAM_PORT"

# Load updated .env
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# 5. Detect Active Compute Device (GPU Preferred)
COMPUTE_DEVICE=$($PYTHON -c "import torch; print(('GPU: ' + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'CPU (No CUDA detected)')" 2>/dev/null || echo "CPU")
echo "[COMPUTE ENGINE] $COMPUTE_DEVICE"

# 6. Launch Live Camera Stream Broadcaster in Background
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

# Wait 2 seconds for server to start
sleep 2

# 7. Launch ALPR Tracking Pipeline (Real-Time Window / Colab Live Display + Disk Save)
echo "[2/2] Launching FastALPR Pipeline on live camera stream..."
echo "[INFO] Output video is simultaneously recorded to: $OUTPUT_VIDEO"
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
echo " Plate Crops  : output/plate_crops/"
echo " Events JSON  : output/events.json"
echo "=============================================================================="
exit $RUN_EXIT_CODE
