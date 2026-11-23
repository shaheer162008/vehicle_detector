"""Colab preview script for FastALPR.

Usage (run after dependencies are installed):

%run colab_preview.py \
    --source "" \
    --simulate-camera \
    --loop-stream \
    --max-frames 300   # optional; set 0 for unlimited loop
    --save-output       # optional, write annotated video to output/annotated_output.mp4
"""

import argparse, os, sys, time
from pathlib import Path
from IPython.display import display, clear_output, Video, Image
import cv2

# ----------------------------------------------------------------------
# Adjust imports to match the repository layout.
# ----------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent))
# NOTE: If the import path differs, replace the line below with the correct one.
from fast_alpr.alpr import VehicleALPRPipeline   # <-- may be fast_alpr.alpr or fast_alpr.pipeline
from config import PipelineConfig                # <-- adjust if your config class lives elsewhere

def download_demo_video() -> str:
    """Download the demo video if it does not exist and return its path."""
    target = Path("stream/stream.mp4")
    if target.exists():
        return str(target)
    url = "https://github.com/ankandrew/fast-alpr/raw/main/stream/stream.mp4"
    try:
        import urllib.request
        target.parent.mkdir(parents=True, exist_ok=True)
        print("[DOWNLOAD] Fetching demo video …")
        urllib.request.urlretrieve(url, str(target))
        print(f"[INFO] Demo video saved to {target}")
    except Exception as e:
        print(f"[ERROR] Could not download demo video: {e}")
        sys.exit(1)
    return str(target)

def parse_args():
    parser = argparse.ArgumentParser(description="FastALPR video preview for Colab")
    parser.add_argument("--source", type=str, default="", help="Path to video file (empty => auto)")
    parser.add_argument("--simulate-camera", action="store_true",
                        help="Treat the video as a live camera feed (keeps timestamps)")
    parser.add_argument("--loop-stream", action="store_true", help="Loop the video indefinitely")
    parser.add_argument("--max-frames", type=int, default=500,
                        help="Maximum frames to display (0 = unlimited)")
    parser.add_argument("--save-output", action="store_true",
                        help="Write annotated video to output/annotated_output.mp4")
    return parser.parse_args()

def main():
    args = parse_args()
    video_source = args.source or download_demo_video()

    # ------------------------------------------------------------------
    # Build the pipeline – you can tweak any config values here.
    # ------------------------------------------------------------------
    config = PipelineConfig()
    pipeline = VehicleALPRPipeline(config=config)

    # ------------------------------------------------------------------
    # Open the video source
    # ------------------------------------------------------------------
    if args.simulate_camera:
        from pipeline import RealtimeCameraCapture
        cap = RealtimeCameraCapture(video_source, loop=args.loop_stream)
    else:
        cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {video_source}")
        return

    # Prepare writer if we need to save the annotated output
    writer = None
    out_path = None
    if args.save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = Path("output/annotated_output.mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        print(f"[INFO] Saving annotated video to {out_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run ALPR on the current frame (draw_predictions if available)
        if hasattr(pipeline, "draw_predictions"):
            annotated = pipeline.draw_predictions(frame).image
        else:
            annotated = frame

        # Write to file if requested
        if writer:
            writer.write(annotated)

        # Inline display
        _, jpeg = cv2.imencode('.jpg', annotated)
        display(Image(data=jpeg.tobytes()))
        clear_output(wait=True)

        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            break
        time.sleep(0.03)

    cap.release()
    if writer:
        writer.release()
        # Show saved video inline after processing
        print("[INFO] Playback of saved video:")
        display(Video(str(out_path), embed=True, width=640, height=360))

    print(f"[INFO] Preview finished – displayed {frame_idx} frames.")

if __name__ == "__main__":
    main()
