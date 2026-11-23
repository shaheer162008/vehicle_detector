#!/usr/bin/env bash
# ==============================================================================
# FastALPR Vehicle Tracking - Run Script (Linux / macOS / Google Colab)
#
# What it does:
#   1. Downloads the input video from Google Drive (skips if already downloaded)
#   2. Runs the vehicle detection + plate tracking pipeline on that video
#   3. Prints the path of the annotated / tracked output video
#
# Usage:
#   ./run.sh                     <- download video from Google Drive and track it
#   ./run.sh /path/to/car.mp4    <- track a local video instead
# ==============================================================================

set -e

cd "$(dirname "$0")"

DRIVE_FILE_ID="1szprqeBN3eQ5vBcRJ8HVIowMu514F94L"
VIDEO_DIR="videos"
VIDEO_PATH="$VIDEO_DIR/drive_input.mp4"
OUTPUT_VIDEO="output/annotated_output.mp4"

echo "=== FastALPR Vehicle Tracking Run Script ==="

# 0. Python check
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi
echo "[INFO] Using Python: $($PYTHON_CMD --version)"

# 0b. Dependency check (install.sh installs these)
if ! $PYTHON_CMD -c "import cv2, ultralytics, fast_alpr" &> /dev/null; then
    echo "[WARNING] Required packages look missing. Run install.sh once, then re-run this script."
fi

# 1. gdown is only needed for the Google Drive download
if ! $PYTHON_CMD -c "import gdown" &> /dev/null; then
    echo "[1/4] Installing gdown (Google Drive downloader)..."
    $PYTHON_CMD -m pip install gdown --no-input --quiet
fi

# 2. Pick the input video: custom local path if given, else Google Drive download
if [ -n "$1" ]; then
    VIDEO_PATH="$1"
    echo "[2/4] Using local video: $VIDEO_PATH"
else
    mkdir -p "$VIDEO_DIR"
    if [ -f "$VIDEO_PATH" ]; then
        echo "[2/4] Video already present: $VIDEO_PATH"
    else
        echo "[2/4] Downloading video from Google Drive..."
        if ! $PYTHON_CMD -c "import gdown; gdown.download(id='$DRIVE_FILE_ID', output='$VIDEO_PATH', quiet=False)"; then
            echo "[ERROR] Google Drive download failed."
            echo "        Make sure the file sharing is set to \"Anyone with the link\"."
            exit 1
        fi
    fi
fi

if [ ! -f "$VIDEO_PATH" ]; then
    echo "[ERROR] Input video not found: $VIDEO_PATH"
    exit 1
fi
if [ ! -s "$VIDEO_PATH" ]; then
    echo "[ERROR] Input video file is empty (0 bytes): $VIDEO_PATH"
    echo "        Delete it and re-run to download again."
    exit 1
fi

# 3. Run the tracking pipeline (annotated video is written by run.py)
echo "[3/4] Running ALPR tracking pipeline on: $VIDEO_PATH"
echo ""
set +e
$PYTHON_CMD run.py --source "$VIDEO_PATH" --output-video "$OUTPUT_VIDEO"
PIPELINE_STATUS=$?
set -e
if [ $PIPELINE_STATUS -ne 0 ]; then
    echo "[ERROR] Pipeline run failed. See messages above."
    exit $PIPELINE_STATUS
fi

# 4. Report the tracked video
if [ -f "$OUTPUT_VIDEO" ]; then
    echo "[4/4] Tracked video ready: $OUTPUT_VIDEO"
else
    echo "[WARNING] Output video not found at: $OUTPUT_VIDEO"
fi

echo "=== Finished ==="
echo "Tracked video : $OUTPUT_VIDEO"
echo "Snapshots     : output/snapshots"
echo "Events JSON   : output/events.json"
