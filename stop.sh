#!/usr/bin/env bash
# FastALPR Stop (Linux/macOS/Colab)

set -e

cd "$(dirname "$0")"

echo "=== FastALPR Stop ==="

# 1. Stop stream processes (ffmpeg + MediaMTX)
if [ -f ".run/stream_pids.txt" ]; then
    echo "Stopping stream processes..."
    while read pid; do
        if [ -n "$pid" ]; then
            echo "Killing PID $pid..."
            kill -9 "$pid" 2>/dev/null || true
        fi
    done < .run/stream_pids.txt
    rm -f .run/stream_pids.txt
else
    echo "No stream PID file found. Trying to kill by name..."
    pkill -9 ffmpeg 2>/dev/null || true
    pkill -9 mediamtx 2>/dev/null || true
fi

# 2. Stop service (uvicorn)
echo "Stopping service..."
pkill -9 -f "uvicorn.*service.app" 2>/dev/null || true

# 3. Clean up .run directory
if [ -d ".run" ]; then
    rm -rf .run
fi

echo ""
echo "All processes stopped."