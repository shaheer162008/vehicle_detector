"""
FastALPR Stream Inspector & Environment Configurator.
Extracts stream video metadata (resolution, FPS, total frames)
and updates .env automatically.
"""

import sys
from pathlib import Path

import cv2


def inspect_and_configure():
    video_path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else "stream/stream.mp4"
    plate_format = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else "PAKISTAN"
    port = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else "8554"

    video_file = Path(video_path)
    w, h, fps, frames = 1920, 1080, 25.0, 0

    if video_file.exists():
        cap = cv2.VideoCapture(str(video_file))
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = round(cap.get(cv2.CAP_PROP_FPS) or 25.0, 2)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

    stream_url = f"http://127.0.0.1:{port}/live"
    output_video = "output/annotated_output.mp4"

    env_content = f"""# ==============================================================================
# FastALPR Auto-Configured Environment Variables
# ==============================================================================
STREAM_PORT={port}
STREAM_VIDEO={video_path}
STREAM_URL={stream_url}
STREAM_RESOLUTION={w}x{h}
STREAM_FPS={fps}
STREAM_TOTAL_FRAMES={frames}
PLATE_FORMAT={plate_format}
OUTPUT_VIDEO={output_video}
CAMERA_ID=CAM-00001
"""

    env_file = Path(".env")
    env_file.write_text(env_content, encoding="utf-8")

    print(f"[STREAM METADATA] File: {video_file.name} | Resolution: {w}x{h} | FPS: {fps:.2f} | Frames: {frames}")
    print(f"[ENVIRONMENT] .env updated -> PLATE_FORMAT={plate_format} | STREAM_URL={stream_url}")


if __name__ == "__main__":
    inspect_and_configure()
