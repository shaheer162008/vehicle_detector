"""
Live Camera Stream Simulator (Portrays MP4 video as a live 24/7 CCTV camera stream).
Streams frames over HTTP MJPEG at the video's exact native FPS in a continuous loop.
OpenCV, web browsers, and the ALPR pipeline connect to:
    http://127.0.0.1:8554/live
"""

import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import cv2


class CameraStreamBuffer:
    """Thread-safe frame buffer holding only the latest live camera frame."""

    def __init__(self):
        self._frame_bytes: Optional[bytes] = None
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self.running = True

    def set_frame(self, frame_bytes: bytes) -> None:
        with self._condition:
            self._frame_bytes = frame_bytes
            self._condition.notify_all()

    def get_latest_frame(self, timeout: float = 0.5) -> Optional[bytes]:
        with self._condition:
            if self._frame_bytes is None:
                self._condition.wait(timeout=timeout)
            return self._frame_bytes

    def stop(self) -> None:
        with self._condition:
            self.running = False
            self._condition.notify_all()


class StreamingHandler(BaseHTTPRequestHandler):
    """Serves multipart MJPEG live camera stream."""

    buffer: Optional[CameraStreamBuffer] = None

    def log_message(self, format, *args):
        # Suppress noisy HTTP request logging in console
        return

    def do_GET(self):
        if self.path in ("/live", "/stream", "/"):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()

            try:
                while self.buffer and self.buffer.running:
                    frame_bytes = self.buffer.get_latest_frame(timeout=0.1)
                    if frame_bytes is None:
                        continue
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame_bytes)))
                    self.end_headers()
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b"\r\n")
                    # Give consumer breathing room
                    time.sleep(0.01)
            except (ConnectionResetError, BrokenPipeError):
                pass
        else:
            self.send_error(404)
            self.end_headers()


def camera_capture_loop(video_path: Path, buffer: CameraStreamBuffer, target_fps: Optional[float] = None) -> None:
    """Reads video file and pushes frames to buffer at exact camera wall-clock FPS."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}")
        buffer.stop()
        return

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if native_fps <= 0 or native_fps > 120:
        native_fps = 25.0
    fps = target_fps or native_fps
    frame_interval = 1.0 / fps

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[STREAM ENGINE] Loaded: {video_path.name}")
    print(f"[STREAM ENGINE] Resolution: {width}x{height} | Camera FPS: {fps:.2f} ({frame_interval*1000:.1f}ms/frame)")
    print(f"[STREAM ENGINE] Total Frames: {total_frames} (Continuous 24/7 Loop Active)")

    frame_count = 0
    loop_count = 1

    try:
        while buffer.running:
            t_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret or frame is None:
                # Seamless loop: rewind back to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                loop_count += 1
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.05)
                    continue

            frame_count += 1
            # Encode frame to JPEG
            ret_enc, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret_enc:
                buffer.set_frame(jpeg_buf.tobytes())

            # Real-time wall-clock pacing (ensures camera portrays real-time speed)
            elapsed = time.perf_counter() - t_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        buffer.stop()


def main():
    parser = argparse.ArgumentParser(description="Live Camera Stream Broadcaster (HTTP MJPEG)")
    parser.add_argument(
        "--video",
        type=str,
        default=str(Path(__file__).resolve().parent / "stream.mp4"),
        help="Path to video file to stream (default: stream/stream.mp4)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host IP to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8554,
        help="Port to stream on (default: 8554)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override camera FPS (default: use video native FPS)",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        # Fallback to root stream_2.mp4 or stream.mp4
        root_video = Path(__file__).resolve().parent.parent / "stream_2.mp4"
        if root_video.exists():
            video_path = root_video
        else:
            print(f"[ERROR] Video file not found: {video_path}")
            sys.exit(1)

    stream_buffer = CameraStreamBuffer()
    StreamingHandler.buffer = stream_buffer

    # Start camera capture thread
    cap_thread = threading.Thread(
        target=camera_capture_loop,
        args=(video_path, stream_buffer, args.fps),
        daemon=True,
    )
    cap_thread.start()

    server_address = (args.host, args.port)
    server = ThreadingHTTPServer(server_address, StreamingHandler)
    url_local = f"http://127.0.0.1:{args.port}/live"

    print("=" * 65)
    print(" === FastALPR Live Camera Stream Broadcaster Active ===")
    print("=" * 65)
    print(f" [URL] Local Stream URL   : {url_local}")
    print(f" [SRC] Feeding Video File : {video_path}")
    print(" [INFO] System portrays this as a real-time CCTV IP Camera.")
    print(" [USAGE] Run ALPR pipeline in another terminal:")
    print(f"         python run.py --source {url_local}")
    print(" [EXIT] Press Ctrl+C to stop the camera stream.")
    print("=" * 65)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Stopping Live Camera Stream...")
    finally:
        stream_buffer.stop()
        server.server_close()
        print("[INFO] Stream server terminated cleanly.")


if __name__ == "__main__":
    main()

