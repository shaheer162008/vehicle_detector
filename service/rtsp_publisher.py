"""
RTSP Publisher for FastALPR Control Service.
Uses ffmpeg + MediaMTX to publish video files as RTSP streams.
Auto-downloads MediaMTX if not present (like the live-stream script).
"""

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Tuple


class RTSPPublisher:
    """Publishes a video file as an RTSP stream using ffmpeg + MediaMTX."""

    def __init__(
        self,
        input_file: Path,
        rtsp_url: str,
        loop: bool = True,
        mediamtx_dir: Optional[Path] = None,
    ):
        self.input_file = input_file
        self.rtsp_url = rtsp_url
        self.loop = loop
        self.mediamtx_dir = mediamtx_dir or (Path(__file__).parent / "mediamtx_bin")
        self._ffmpeg_process: Optional[subprocess.Popen] = None
        self._mediamtx_process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started = False
        self._error: Optional[str] = None
        self._port = 8554
        self._path = "live"

        # Parse port and path from rtsp_url
        try:
            from urllib.parse import urlparse
            parsed = urlparse(rtsp_url)
            self._port = parsed.port or 8554
            self._path = parsed.path.lstrip("/") or "live"
        except Exception:
            pass

    def _find_mediamtx(self) -> Optional[str]:
        """Find MediaMTX executable."""
        exe = "mediamtx.exe" if os.name == "nt" else "mediamtx"
        found = shutil.which("mediamtx")
        if found:
            return found
        local = self.mediamtx_dir / exe
        return str(local) if local.is_file() else None

    def _download_mediamtx(self) -> str:
        """Download MediaMTX from GitHub (like live-stream script)."""
        system = platform.system().lower()
        machine = platform.machine().lower()
        arch_map = {
            "amd64": "amd64", "x86_64": "amd64",
            "arm64": "arm64", "aarch64": "arm64",
            "armv7l": "armv7",
        }
        arch = arch_map.get(machine, "amd64")
        want = f"{system}_{arch}"

        print(f"[RTSP PUBLISHER] Downloading MediaMTX for {want}...")
        api = "https://api.github.com/repos/bluenviron/mediamtx/releases/latest"
        req = urllib.request.Request(api, headers={"User-Agent": "fastalpr-rtsp"})
        with urllib.request.urlopen(req, timeout=30) as r:
            release = json.load(r)

        asset = next((a for a in release["assets"]
                      if want in a["name"] and a["name"].endswith((".zip", ".tar.gz"))), None)
        if not asset:
            raise RuntimeError(f"No MediaMTX build for {want}. Download manually from GitHub.")

        self.mediamtx_dir.mkdir(parents=True, exist_ok=True)
        archive = self.mediamtx_dir / asset["name"]
        urllib.request.urlretrieve(asset["browser_download_url"], archive)

        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as z:
                z.extractall(self.mediamtx_dir)
        else:
            with tarfile.open(archive) as t:
                t.extractall(self.mediamtx_dir)
        archive.unlink(missing_ok=True)

        exe = self.mediamtx_dir / ("mediamtx.exe" if os.name == "nt" else "mediamtx")
        if os.name != "nt":
            os.chmod(exe, 0o755)
        print("[RTSP PUBLISHER] MediaMTX ready.")
        return str(exe)

    def _wait_for_port(self, port: int, timeout: float = 15.0) -> bool:
        """Wait for TCP port to become available."""
        end = time.time() + timeout
        while time.time() < end:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    return True
            except OSError:
                time.sleep(0.3)
        return False

    def _build_ffmpeg_cmd(self) -> list[str]:
        """Build ffmpeg command for streaming."""
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-re",  # Read at native frame rate (simulate live)
        ]
        if self.loop:
            cmd.extend(["-stream_loop", "-1"])
        cmd.extend(["-i", str(self.input_file)])

        # Try stream copy first (if H.264), else encode
        # We'll use copy for efficiency
        cmd += [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-b:v", "4M",
            "-maxrate", "4M",
            "-bufsize", "8M",
            "-g", "50",  # Keyframe every ~2 sec
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            self.rtsp_url,
        ]
        # No audio for now
        cmd += ["-an"]
        return cmd

    def start(self) -> bool:
        """Start MediaMTX server + ffmpeg publisher."""
        if self._ffmpeg_process and self._ffmpeg_process.poll() is None:
            return True

        # 1. Start MediaMTX
        mtx = self._find_mediamtx() or self._download_mediamtx()

        env = os.environ.copy()
        env["MTX_RTSPADDRESS"] = f":{self._port}"
        env["MTX_HLSADDRESS"] = ":8888"
        env["MTX_WEBRTCADDRESS"] = ":8889"

        print(f"[RTSP PUBLISHER] Starting MediaMTX on port {self._port}...")
        self._mediamtx_process = subprocess.Popen(
            [mtx],
            env=env,
            cwd=str(self.mediamtx_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        def cleanup():
            for p in (self._ffmpeg_process, self._mediamtx_process):
                if p and p.poll() is None:
                    p.terminate()

        if not self._wait_for_port(self._port):
            cleanup()
            self._error = f"MediaMTX failed to start on port {self._port}"
            print(f"[RTSP PUBLISHER] {self._error}")
            return False

        # 2. Start ffmpeg publisher
        publish_url = f"rtsp://127.0.0.1:{self._port}/{self._path}"
        cmd = self._build_ffmpeg_cmd()
        # Update the URL in command
        cmd[-1] = publish_url

        print(f"[RTSP PUBLISHER] Publishing {self.input_file.name} -> {publish_url}")

        try:
            self._ffmpeg_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            cleanup()
            self._error = f"Failed to start ffmpeg: {e}"
            print(f"[RTSP PUBLISHER] {self._error}")
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_ffmpeg, daemon=True)
        self._thread.start()

        # Wait briefly for startup
        time.sleep(1.5)
        if self._ffmpeg_process.poll() is not None:
            stdout, stderr = self._ffmpeg_process.communicate()
            cleanup()
            self._error = f"ffmpeg exited: {stderr}"
            print(f"[RTSP PUBLISHER] {self._error}")
            return False

        self._started = True
        print(f"[RTSP PUBLISHER] LIVE! RTSP: {publish_url} | HLS: http://127.0.0.1:8888/{self._path}")
        return True

    def _monitor_ffmpeg(self) -> None:
        """Monitor ffmpeg process - auto-restart if it dies."""
        while not self._stop_event.is_set():
            if not self._ffmpeg_process:
                break
            ret = self._ffmpeg_process.poll()
            if ret is not None:
                if self._stop_event.is_set():
                    break
                print(f"[RTSP PUBLISHER] ffmpeg exited (code {ret}), restarting in 2s...")
                time.sleep(2)
                if self._stop_event.is_set():
                    break
                try:
                    publish_url = f"rtsp://127.0.0.1:{self._port}/{self._path}"
                    cmd = self._build_ffmpeg_cmd()
                    cmd[-1] = publish_url
                    self._ffmpeg_process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        bufsize=1,
                    )
                except Exception as e:
                    print(f"[RTSP PUBLISHER] Restart failed: {e}")
                    break
            else:
                time.sleep(1)

    def stop(self) -> None:
        """Stop both ffmpeg and MediaMTX."""
        self._stop_event.set()
        for p in (self._ffmpeg_process, self._mediamtx_process):
            if p:
                try:
                    p.terminate()
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=2)
                except Exception:
                    pass
        self._ffmpeg_process = None
        self._mediamtx_process = None

        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

        self._started = False

    @property
    def is_running(self) -> bool:
        return (self._ffmpeg_process is not None and self._ffmpeg_process.poll() is None and
                self._mediamtx_process is not None and self._mediamtx_process.poll() is None)

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def rtsp_url(self) -> str:
        return f"rtsp://127.0.0.1:{self._port}/{self._path}"

    @property
    def hls_url(self) -> str:
        return f"http://127.0.0.1:8888/{self._path}"

    @property
    def webrtc_url(self) -> str:
        return f"http://127.0.0.1:8889/{self._path}"