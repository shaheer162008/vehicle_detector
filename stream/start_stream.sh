#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

PYTHON="python3"
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
fi

echo "=== Starting FastALPR Live Camera Stream Broadcaster ==="
$PYTHON stream/live_stream.py --video "stream/stream.mp4" --port 8554
