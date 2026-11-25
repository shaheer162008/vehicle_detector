"""
WebSocket Relay Client for FastALPR Pipeline Workers.
Connects to the service's internal ingest endpoint and forwards events/frames.
"""

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urlparse

try:
    from websockets.sync.client import connect as ws_connect
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class RelaySink:
    """
    Non-blocking WebSocket relay for pipeline workers.
    - Events: text JSON messages
    - Frames: binary JPEG (latest only, max 1 in queue)
    - Auto-reconnect with exponential backoff
    """

    def __init__(
        self,
        relay_url: str,
        stream_id: str,
        preview_fps: int = 10,
        preview_width: int = 640,
        jpeg_quality: int = 70,
    ):
        if not WEBSOCKETS_AVAILABLE:
            raise RuntimeError("websockets package not installed (pip install websockets)")

        self.relay_url = relay_url
        self.stream_id = stream_id
        self.preview_fps = preview_fps
        self.preview_width = preview_width
        self.jpeg_quality = jpeg_quality

        self._ws = None
        self._connected = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)  # Latest frame only
        self._event_queue: queue.Queue = queue.Queue(maxsize=100)
        self._last_frame_time = 0.0
        self._connect_backoff = 1.0
        self._max_backoff = 30.0

    def start(self) -> bool:
        """Start the relay connection thread."""
        if self._thread and self._thread.is_alive():
            return True

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # Wait briefly for initial connection attempt
        time.sleep(0.5)
        return self._connected

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the relay thread and close connection."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._close_ws()

    def publish_event(self, event: Dict[str, Any]) -> None:
        """Queue an event for sending (non-blocking)."""
        try:
            event_with_stream = dict(event)
            event_with_stream["stream_id"] = self.stream_id
            self._event_queue.put_nowait(event_with_stream)
        except queue.Full:
            # Drop oldest if queue full
            try:
                self._event_queue.get_nowait()
                self._event_queue.put_nowait(event_with_stream)
            except queue.Empty:
                pass

    def publish_frame(self, frame: np.ndarray) -> None:
        """Queue a frame for JPEG encoding and sending (non-blocking, latest only)."""
        now = time.time()
        frame_interval = 1.0 / max(1, self.preview_fps)
        if now - self._last_frame_time < frame_interval:
            return  # Rate limit

        # Resize frame to preview width
        h, w = frame.shape[:2]
        if w > self.preview_width:
            scale = self.preview_width / w
            new_w = self.preview_width
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Encode to JPEG
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ret:
            return

        self._last_frame_time = now
        try:
            self._frame_queue.put_nowait(buf.tobytes())
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
                self._frame_queue.put_nowait(buf.tobytes())
            except queue.Empty:
                pass

    def _run(self) -> None:
        """Main relay thread: connect, send queued events/frames, handle reconnection."""
        while not self._stop_event.is_set():
            try:
                self._connect_and_run()
            except Exception as e:
                logger.warning(f"[RELAY] Connection error: {e}")
            finally:
                self._close_ws()
                self._connected = False

            if self._stop_event.is_set():
                break

            # Exponential backoff
            logger.info(f"[RELAY] Reconnecting in {self._connect_backoff:.1f}s...")
            time.sleep(self._connect_backoff)
            self._connect_backoff = min(self._connect_backoff * 2, self._max_backoff)

    def _connect_and_run(self) -> None:
        """Establish WebSocket connection and run send loop."""
        self._ws = ws_connect(self.relay_url, open_timeout=10)
        self._connected = True
        self._connect_backoff = 1.0  # Reset on successful connection
        logger.info(f"[RELAY] Connected to {self.relay_url}")

        try:
            while not self._stop_event.is_set():
                # Send events (text)
                try:
                    while True:
                        event = self._event_queue.get_nowait()
                        self._ws.send(json.dumps(event))
                except queue.Empty:
                    pass
                except Exception as e:
                    logger.warning(f"[RELAY] Event send error: {e}")
                    raise

                # Send frames (binary) - latest only
                try:
                    frame_data = self._frame_queue.get_nowait()
                    self._ws.send(frame_data)
                except queue.Empty:
                    pass
                except Exception as e:
                    logger.warning(f"[RELAY] Frame send error: {e}")
                    raise

                # Small sleep to prevent busy loop
                time.sleep(0.01)

        except Exception as e:
            logger.warning(f"[RELAY] Send loop error: {e}")
            raise

    def _close_ws(self) -> None:
        """Close WebSocket connection."""
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    @property
    def is_connected(self) -> bool:
        return self._connected