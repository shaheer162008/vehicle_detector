"""
Stream Manager for FastALPR Control Service.
Manages stream configurations, worker subprocesses, and RTSP publishers.
"""

import json
import os
import signal
import socket
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse

from .relay import RelaySink
from .rtsp_publisher import RTSPPublisher


@dataclass
class StreamConfig:
    """Configuration for a single stream."""
    id: str
    name: str
    source: Dict[str, Any]  # {"type": "file"|"rtsp", "file"?: "relative/path.mp4", "url"?: "rtsp://..."}
    loop: bool = False
    camera_id: str = "CAM-00001"
    plate_format: str = "GENERAL"
    vehicle_weights: str = "yolo11m.pt"
    publish_rtsp: bool = False
    autostart: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreamConfig":
        return cls(**data)


@dataclass
class StreamStatus:
    """Runtime status of a stream."""
    state: str = "stopped"  # "stopped", "running", "error", "starting", "stopping"
    fps: float = 0.0
    frames: int = 0
    events: int = 0
    last_event: Optional[Dict[str, Any]] = None
    uptime_s: float = 0.0
    restarts: int = 0
    last_error: Optional[str] = None
    rtsp_url: Optional[str] = None
    started_at: Optional[str] = None
    worker_pid: Optional[int] = None


class StreamManager:
    """Manages stream lifecycle, worker processes, and RTSP publishers."""

    def __init__(
        self,
        streams_file: Path,
        video_dir: Path,
        rtsp_base: str,
        service_token: str,
        service_host: str = "127.0.0.1",
        service_port: int = 8000,
    ):
        self.streams_file = streams_file
        self.video_dir = video_dir.resolve()
        self.rtsp_base = rtsp_base.rstrip("/")
        self.service_token = service_token
        self.service_host = service_host
        self.service_port = service_port

        self._streams: Dict[str, StreamConfig] = {}
        self._status: Dict[str, StreamStatus] = {}
        self._workers: Dict[str, subprocess.Popen] = {}
        self._publishers: Dict[str, RTSPPublisher] = {}
        self._lock = threading.RLock()

        self._load_streams()

    def _load_streams(self) -> None:
        """Load stream configurations from JSON file."""
        if self.streams_file.exists():
            try:
                with open(self.streams_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for stream_data in data:
                    config = StreamConfig.from_dict(stream_data)
                    self._streams[config.id] = config
                    self._status[config.id] = StreamStatus()
            except Exception as e:
                print(f"[MANAGER] Failed to load streams: {e}")

        # Auto-create default live stream if none exist and stream_live.mp4 exists
        if not self._streams:
            live_video = Path("stream_live.mp4")
            if live_video.exists():
                self._create_default_live_stream(live_video)

    def _create_default_live_stream(self, video_path: Path) -> None:
        """Create a default live stream configuration for the live video."""
        stream_id = "live-cam-001"
        config = StreamConfig(
            id=stream_id,
            name="Live Camera Stream",
            source={"type": "file", "file": video_path.name},
            loop=True,
            camera_id="CAM-00001",
            plate_format="GENERAL",
            vehicle_weights="yolo11m.pt",
            publish_rtsp=True,
            autostart=True,
        )
        self._streams[stream_id] = config
        self._status[stream_id] = StreamStatus()
        self._save_streams()
        print(f"[MANAGER] Auto-created default live stream: {stream_id}")

    def _save_streams(self) -> None:
        """Atomically save stream configurations to JSON file."""
        self.streams_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = self.streams_file.with_suffix(".tmp")
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump([s.to_dict() for s in self._streams.values()], f, indent=2)
            os.replace(tmp_file, self.streams_file)
        except Exception as e:
            print(f"[MANAGER] Failed to save streams: {e}")
            tmp_file.unlink(missing_ok=True)

    def _generate_stream_id(self, name: str) -> str:
        """Generate a slug-style stream ID."""
        base = "".join(c.lower() if c.isalnum() else "-" for c in name).strip("-")
        base = base[:36]  # Leave room for suffix
        candidate = base
        suffix = 1
        while candidate in self._streams:
            candidate = f"{base}-{suffix}"
            suffix += 1
        return candidate

    def _validate_file_source(self, file_path: str) -> Path:
        """Validate and resolve file source path within VIDEO_DIR or project root."""
        path = Path(file_path)
        if path.is_absolute():
            raise ValueError("File source must be relative path")

        # First try VIDEO_DIR
        resolved = (self.video_dir / path).resolve()
        if resolved.exists():
            try:
                resolved.relative_to(self.video_dir)
                return resolved
            except ValueError:
                pass

        # Fallback to project root (for stream_live.mp4 etc.)
        project_root = Path(__file__).resolve().parent.parent
        resolved = (project_root / path).resolve()
        if resolved.exists():
            try:
                resolved.relative_to(project_root)
                return resolved
            except ValueError:
                pass

        raise ValueError(f"File not found: {path} (checked VIDEO_DIR and project root)")

    def _build_worker_args(self, config: StreamConfig, stream_id: str) -> List[str]:
        """Build command line arguments for worker subprocess."""
        # Determine video source
        source = config.source
        if source["type"] == "file":
            video_path = self._validate_file_source(source["file"])
            video_source = str(video_path)
        elif source["type"] == "rtsp":
            video_source = source["url"]
        else:
            raise ValueError(f"Unknown source type: {source['type']}")

        # If publishing RTSP, worker consumes the RTSP URL
        if config.publish_rtsp:
            rtsp_url = f"{self.rtsp_base}/{stream_id}"
            video_source = rtsp_url

        # Build relay URL
        relay_url = f"ws://{self.service_host}:{self.service_port}/internal/ingest/{stream_id}?token={self.service_token}"

        args = [
            "python", "-m", "run",
            "--source", video_source,
            "--camera-id", config.camera_id,
            "--plate-format", config.plate_format,
            "--vehicle-weights", config.vehicle_weights,
            "--relay-url", relay_url,
            "--stream-id", stream_id,
            "--no-preview",
        ]

        if config.loop:
            args.append("--loop-file")

        return args

    def create_stream(self, name: str, source: Dict[str, Any], loop: bool, camera_id: str,
                      plate_format: str, vehicle_weights: str = "yolo11m.pt",
                      publish_rtsp: bool = False, autostart: bool = False) -> str:
        """Create a new stream configuration."""
        with self._lock:
            stream_id = self._generate_stream_id(name)
            config = StreamConfig(
                id=stream_id,
                name=name,
                source=source,
                loop=loop,
                camera_id=camera_id,
                plate_format=plate_format,
                vehicle_weights=vehicle_weights,
                publish_rtsp=publish_rtsp,
                autostart=autostart,
            )
            self._streams[stream_id] = config
            self._status[stream_id] = StreamStatus()
            self._save_streams()
            return stream_id

    def get_stream(self, stream_id: str) -> Optional[StreamConfig]:
        with self._lock:
            return self._streams.get(stream_id)

    def list_streams(self) -> List[Dict[str, Any]]:
        with self._lock:
            result = []
            for stream_id, config in self._streams.items():
                status = self._status.get(stream_id, StreamStatus())
                result.append({
                    "id": stream_id,
                    "name": config.name,
                    "source": config.source,
                    "loop": config.loop,
                    "camera_id": config.camera_id,
                    "plate_format": config.plate_format,
                    "vehicle_weights": config.vehicle_weights,
                    "publish_rtsp": config.publish_rtsp,
                    "autostart": config.autostart,
                    "status": self._get_status_dict(stream_id, status),
                })
            return result

    def update_stream(self, stream_id: str, loop: Optional[bool] = None,
                      name: Optional[str] = None, plate_format: Optional[str] = None) -> bool:
        with self._lock:
            config = self._streams.get(stream_id)
            if not config:
                return False
            if loop is not None:
                config.loop = loop
            if name is not None:
                config.name = name
            if plate_format is not None:
                config.plate_format = plate_format
            self._save_streams()
            return True

    def delete_stream(self, stream_id: str) -> bool:
        with self._lock:
            if stream_id not in self._streams:
                return False
            # Stop if running
            self._stop_stream_locked(stream_id)
            del self._streams[stream_id]
            self._status.pop(stream_id, None)
            self._save_streams()
            return True

    def start_stream(self, stream_id: str) -> bool:
        with self._lock:
            return self._start_stream_locked(stream_id)

    def stop_stream(self, stream_id: str) -> bool:
        with self._lock:
            return self._stop_stream_locked(stream_id)

    def restart_stream(self, stream_id: str) -> bool:
        with self._lock:
            self._stop_stream_locked(stream_id)
            time.sleep(0.5)
            return self._start_stream_locked(stream_id)

    def _start_stream_locked(self, stream_id: str) -> bool:
        config = self._streams.get(stream_id)
        if not config:
            return False

        status = self._status.get(stream_id, StreamStatus())
        if status.state == "running":
            return True

        status.state = "starting"
        status.last_error = None

        # Start RTSP publisher if needed
        if config.publish_rtsp:
            source = config.source
            if source["type"] != "file":
                status.state = "error"
                status.last_error = "publish_rtsp only supported for file sources"
                return False

            video_path = self._validate_file_source(source["file"])
            publisher = RTSPPublisher(
                input_file=video_path,
                rtsp_url=f"{self.rtsp_base}/{stream_id}",
                loop=config.loop,
            )
            if not publisher.start():
                status.state = "error"
                status.last_error = "Failed to start RTSP publisher (check ffmpeg/mediamtx)"
                return False
            self._publishers[stream_id] = publisher
            status.rtsp_url = f"{self.rtsp_base}/{stream_id}"

        # Start worker subprocess
        args = self._build_worker_args(config, stream_id)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)

        try:
            worker = subprocess.Popen(
                args,
                cwd=str(Path(__file__).resolve().parent.parent),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            status.state = "error"
            status.last_error = f"Failed to start worker: {e}"
            if stream_id in self._publishers:
                self._publishers[stream_id].stop()
                del self._publishers[stream_id]
            return False

        self._workers[stream_id] = worker
        status.state = "running"
        status.started_at = datetime.now().isoformat()
        status.worker_pid = worker.pid
        status.restarts += 1

        # Start log reader thread
        threading.Thread(target=self._read_worker_logs, args=(stream_id, worker), daemon=True).start()

        # Start status updater thread
        threading.Thread(target=self._update_status_loop, args=(stream_id,), daemon=True).start()

        return True

    def _stop_stream_locked(self, stream_id: str) -> bool:
        # Stop worker
        worker = self._workers.pop(stream_id, None)
        if worker:
            try:
                if os.name == "nt":
                    worker.terminate()
                else:
                    worker.send_signal(signal.SIGINT)
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)
            except Exception:
                pass

        # Stop publisher
        publisher = self._publishers.pop(stream_id, None)
        if publisher:
            publisher.stop()

        status = self._status.get(stream_id)
        if status:
            status.state = "stopped"
            status.worker_pid = None
            status.rtsp_url = None
        return True

    def _read_worker_logs(self, stream_id: str, worker: subprocess.Popen) -> None:
        """Read worker stdout/stderr and update status."""
        try:
            for line in worker.stdout:
                line = line.strip()
                if not line:
                    continue
                # Parse FPS/events from logs
                if "FPS:" in line and "Vehicles:" in line:
                    try:
                        parts = line.split("|")
                        for part in parts:
                            part = part.strip()
                            if part.startswith("FPS:"):
                                self._status[stream_id].fps = float(part.split(":")[1].strip())
                            elif part.startswith("Vehicles:"):
                                veh_part = part.split("(")[0].replace("Vehicles:", "").strip()
                                self._status[stream_id].frames = int(veh_part) if veh_part.isdigit() else 0
                    except Exception:
                        pass
                elif "[EVENT RECORDED]" in line or "-> EVENT" in line:
                    self._status[stream_id].events += 1
        except Exception:
            pass

    def _update_status_loop(self, stream_id: str) -> None:
        """Periodically update stream status (uptime, etc.)."""
        while stream_id in self._workers:
            status = self._status.get(stream_id)
            if status and status.started_at:
                try:
                    started = datetime.fromisoformat(status.started_at)
                    status.uptime_s = (datetime.now() - started).total_seconds()
                except Exception:
                    pass
            time.sleep(5)

        # Worker exited
        status = self._status.get(stream_id)
        if status and status.state == "running":
            status.state = "error"
            worker = self._workers.get(stream_id)
            if worker:
                try:
                    ret = worker.poll()
                    if ret is not None and ret != 0:
                        status.last_error = f"Worker exited with code {ret}"
                    else:
                        status.last_error = "Worker stopped unexpectedly"
                except Exception:
                    status.last_error = "Worker stopped unexpectedly"

    def get_status(self, stream_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            status = self._status.get(stream_id)
            if not status:
                return None
            return self._get_status_dict(stream_id, status)

    def _get_status_dict(self, stream_id: str, status: StreamStatus) -> Dict[str, Any]:
        return {
            "state": status.state,
            "fps": status.fps,
            "frames": status.frames,
            "events": status.events,
            "last_event": status.last_event,
            "uptime_s": status.uptime_s,
            "restarts": status.restarts,
            "last_error": status.last_error,
            "rtsp_url": status.rtsp_url,
        }

    def get_videos(self) -> List[str]:
        """List video files in VIDEO_DIR."""
        videos = []
        if self.video_dir.exists():
            for ext in (".mp4", ".mkv", ".avi", ".mov"):
                videos.extend(sorted(f.name for f in self.video_dir.glob(f"*{ext}")))
        return videos

    def shutdown(self) -> None:
        """Stop all streams gracefully."""
        with self._lock:
            for stream_id in list(self._streams.keys()):
                self._stop_stream_locked(stream_id)