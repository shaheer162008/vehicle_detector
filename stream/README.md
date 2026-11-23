# Live Camera Stream Folder

This folder serves as the simulated 24/7 CCTV / IP Camera feed source for testing the ALPR Tracking Pipeline.

## Files
- `stream.mp4`: The primary input video (automatically created from `stream_2.mp4` or user-provided footage).
- `live_stream.py`: High-performance HTTP MJPEG live stream broadcaster that streams `stream.mp4` in a continuous loop at exact camera FPS.
- `start_stream.bat`: Double-click runner on Windows to start the camera stream.
- `start_stream.sh`: Runner for Linux / macOS to start the camera stream.

## Two Ways to Run

### Method 1: Integrated Camera Simulation (Single Process - Recommended)
Simply run:
```bash
# Windows:
run.bat

# Linux / Mac / Colab:
./run.sh
```
The pipeline automatically picks `stream/stream.mp4` and simulates camera timing, frame pacing, and buffer dynamics.

### Method 2: Networked IP Camera Stream (Dual Process)
1. In Terminal 1, start the camera broadcaster:
```bash
python stream/live_stream.py --port 8554
# Or double-click: stream/start_stream.bat
```
Stream is broadcast at: `http://127.0.0.1:8554/live`

2. In Terminal 2, run the tracking pipeline on the live stream:
```bash
python run.py --source http://127.0.0.1:8554/live --plate-format PAKISTAN
```
This tests real network stream latency, dropped frames, and true real-time throughput identically to a physical IP camera!
