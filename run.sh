#!/usr/bin/env bash
# ==============================================================================
# FastALPR Vehicle Tracking - Run Script (Linux / Google Colab / Boss PC)
# ==============================================================================

set -e

cd "$(dirname "$0")"

# Load .env if present
if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
fi

OUTPUT_VIDEO="output/annotated_output.mp4"
DRIVE_FILE_ID="1szprqeBN3eQ5vBcRJ8HVIowMu514F94L"
VIDEO_DIR="videos"

echo "================================================="
echo "=== FastALPR Vehicle Tracking Run Script ==="
echo "================================================="

# 1. Python environment detection
PYTHON_CMD="python3"
if [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
elif ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

# 2. Check GPU acceleration (fail early, never run silently on CPU)
echo "[CHECK] Verifying GPU hardware and execution providers..."
if ! $PYTHON_CMD tools/check_gpu.py; then
    echo ""
    echo "[ERROR] GPU check failed. Execution halted to prevent slow CPU execution."
    exit 1
fi
echo ""

# 3. Video Selection Menu or Argument Handling
VIDEO_PATH=""
ARG1="$1"

if [ "$ARG1" = "1" ]; then
    VIDEO_PATH="stream_2.mp4"
elif [ "$ARG1" = "2" ]; then
    VIDEO_PATH="stream.mp4"
elif [ "$ARG1" = "3" ]; then
    mkdir -p "$VIDEO_DIR"
    VIDEO_PATH="$VIDEO_DIR/drive_input.mp4"
    if [ ! -f "$VIDEO_PATH" ]; then
        echo "[INFO] Downloading Google Drive video ($DRIVE_FILE_ID)..."
        $PYTHON_CMD -c "import gdown; gdown.download(id='$DRIVE_FILE_ID', output='$VIDEO_PATH', quiet=False)"
    fi
elif [ "$ARG1" = "4" ] && [ -n "$2" ]; then
    VIDEO_PATH="$2"
elif [ -n "$ARG1" ]; then
    VIDEO_PATH="$ARG1"
else
    echo "================================================="
    echo "            FAST-ALPR VIDEO SOURCE MENU"
    echo "================================================="
    echo "  1) stream_2.mp4 (repo)"
    echo "  2) stream.mp4 (repo)"
    echo "  3) Google Drive video (ID: 1szprqeBN3eQ5vBcRJ8HVIowMu514F94L)"
    echo "  4) Custom video path"
    echo "================================================="
    read -r -p "Choose video source [1-4] (default 1): " V_CHOICE
    V_CHOICE=${V_CHOICE:-1}

    if [ "$V_CHOICE" = "1" ]; then
        VIDEO_PATH="stream_2.mp4"
    elif [ "$V_CHOICE" = "2" ]; then
        VIDEO_PATH="stream.mp4"
    elif [ "$V_CHOICE" = "3" ]; then
        mkdir -p "$VIDEO_DIR"
        VIDEO_PATH="$VIDEO_DIR/drive_input.mp4"
        if [ ! -f "$VIDEO_PATH" ]; then
            echo "[INFO] Downloading Google Drive video ($DRIVE_FILE_ID)..."
            $PYTHON_CMD -c "import gdown; gdown.download(id='$DRIVE_FILE_ID', output='$VIDEO_PATH', quiet=False)"
        fi
    elif [ "$V_CHOICE" = "4" ]; then
        read -r -p "Enter custom video path: " VIDEO_PATH
    else
        VIDEO_PATH="stream_2.mp4"
    fi
fi

if [ ! -f "$VIDEO_PATH" ]; then
    echo "[ERROR] Video file not found: $VIDEO_PATH"
    exit 1
fi

# 4. Plate Format Selection Menu
PLATE_FORMAT="GENERAL"
if [ "$ARG1" = "1" ] && [ -n "$2" ]; then
    PLATE_FORMAT="$2"
elif [ "$ARG1" = "2" ] && [ -n "$2" ]; then
    PLATE_FORMAT="$2"
elif [ "$ARG1" = "3" ] && [ -n "$2" ]; then
    PLATE_FORMAT="$2"
elif [ "$ARG1" = "4" ] && [ -n "$3" ]; then
    PLATE_FORMAT="$3"
elif [ -t 0 ]; then
    echo ""
    echo "================================================="
    echo "             PLATE FORMAT MENU"
    echo "================================================="
    echo "  1) GENERAL (Default)"
    echo "  2) UK"
    echo "  3) PAKISTAN"
    echo "================================================="
    read -r -p "Choose plate format [1-3] (default 1): " F_CHOICE
    F_CHOICE=${F_CHOICE:-1}

    if [ "$F_CHOICE" = "1" ]; then
        PLATE_FORMAT="GENERAL"
    elif [ "$F_CHOICE" = "2" ]; then
        PLATE_FORMAT="UK"
    elif [ "$F_CHOICE" = "3" ]; then
        PLATE_FORMAT="PAKISTAN"
    fi
fi

echo ""
echo "[RUN] Source: $VIDEO_PATH"
echo "[RUN] Format: $PLATE_FORMAT"
echo "[RUN] Output Video: $OUTPUT_VIDEO"
echo ""

$PYTHON_CMD run.py --source "$VIDEO_PATH" --output-video "$OUTPUT_VIDEO" --plate-format "$PLATE_FORMAT" --preview

echo ""
echo "================================================="
echo "=== Run Completed Successfully ==="
echo "Annotated Video: $OUTPUT_VIDEO"
echo "Events JSON    : output/events.json"
echo "Snapshots      : output/snapshots"
echo "================================================="
