"""Colab preview script for FastALPR.

Usage in a Colab notebook (after installing dependencies):

```python
%run colab_preview.py \
    --source "" \
    --simulate-camera \
    --loop-stream \
    --max-frames 300  # optional, limit to avoid infinite loop
```

If `--source` is empty the script will automatically locate or download the demo video
(`stream/stream.mp4`). The video is processed frame‑by‑frame and each annotated frame
is displayed inline using IPython's `display` utilities, which works in headless
environments like Colab. The script respects `--max-frames` (default = 500) to
prevent endless execution; omit it to run indefinitely (use with care).
"""

import argparse, os, sys
from pathlib import Path
from IPython.display import display, clear_output, Image
import cv2
import time

# Import the pipeline class from the repo
sys.path.append(str(Path(__file__).resolve().parent))
from fast_alpr.alpr import VehicleALPRPipeline  # adjust import if needed
from config import PipelineConfig, load_camera_from_json  # placeholder, adapt as repo defines

def download_demo_video():
    """Download the demo video if it does not exist.
    Returns the path to the video file.
    """
    target = Path("stream/stream.mp4")
    if target.exists():
        return str(target)
    url = "https://github.com/ankandrew/fast-alpr/raw/main/stream/stream.mp4"
    try:
        import urllib.request
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[DOWNLOAD] Fetching demo video …")
        urllib.request.urlretrieve(url, str(target))
        print(f"[INFO] Demo video saved to {target}")
    except Exception as e:
        print(f"[ERROR] Could not download demo video: {e}")
        sys.exit(1)
    return str(target)


def parse_args():
    parser = argparse.ArgumentParser(description="FastALPR video preview for Colab")
    parser.add_argument("--source", type=str, default="", help="Path to video file (empty => auto)")
    parser.add_argument("--simulate-camera", action="store_true", help="Treat video as live camera feed")
    parser.add_argument("--loop-stream", action="store_true", help="Loop the video indefinitely")
    parser.add_argument("--max-frames", type=int, default=500, help="Maximum frames to display (0 = unlimited)")
    return parser.parse_args()


def main():
    args = parse_args()
    video_source = args.source or download_demo_video()

    # Build a minimal config – the repository already provides utilities to load config from JSON.
    # Here we use default values; adjust if your repo expects more.
    config = PipelineConfig()
    pipeline = VehicleALPRPipeline(config=config)

    # Open video capture (simulate camera if requested)
    if args.simulate_camera:
        from pipeline import RealtimeCameraCapture
        cap = RealtimeCameraCapture(video_source, loop=args.loop_stream)
    else:
        cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {video_source}")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Run ALPR on the current frame
        results = pipeline.process_frame(frame)  # assuming a helper method exists; otherwise use pipeline directly
        # Draw annotations (the pipeline has a draw method; we call it if present)
        if hasattr(pipeline, "draw_predictions"):
            annotated = pipeline.draw_predictions(frame).image
        else:
            annotated = frame
        # Encode as JPEG for inline display
        _, jpeg = cv2.imencode('.jpg', annotated)
        display(Image(data=jpeg.tobytes()))
        clear_output(wait=True)
        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            break
        # Small sleep to mimic real‑time playback (adjust as needed)
        time.sleep(0.03)
    cap.release()
    print("[INFO] Preview finished (displayed {} frames).".format(frame_idx))

if __name__ == "__main__":
    main()
"