"""
Vehicle ALPR Pipeline:
Single-pass YOLO11 + ByteTrack tracking, FastALPR crop OCR, live Colab/Desktop preview,
clean annotated H.264 video encoding, and fan-out event persistence (JSON, Neon Postgres, Neon S3).
"""

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def register_nvidia_dll_paths() -> None:
    """Registers NVIDIA CUDA/cuDNN library paths before ONNX Runtime loads."""
    try:
        import site

        candidates = list(site.getsitepackages())
        user_site = site.getusersitepackages()
        if user_site:
            candidates.append(user_site)

        for candidate in candidates:
            nvidia_root = Path(candidate) / "nvidia"
            if not nvidia_root.is_dir():
                continue
            for sub in ("bin", "lib"):
                for lib_dir in nvidia_root.rglob(sub):
                    path = str(lib_dir)
                    if os.name == "nt" and hasattr(os, "add_dll_directory"):
                        try:
                            os.add_dll_directory(path)
                        except Exception:
                            pass
                    if path not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
                    if os.name != "nt" and path not in os.environ.get("LD_LIBRARY_PATH", ""):
                        os.environ["LD_LIBRARY_PATH"] = path + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    except Exception:
        pass


def _in_ipython_notebook() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ != "TerminalInteractiveShell"
    except Exception:
        return False


def _has_gui_display() -> bool:
    return os.name == "nt" or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


register_nvidia_dll_paths()

from config import PipelineConfig
from fast_alpr import ALPR
from tracker import VehicleTrackerManager, filter_nested_boxes


class ColabLivePreviewWorker:
    """Non-blocking JPEG display stream inside Google Colab / Jupyter cells."""

    def __init__(self, max_w: int = 960, max_h: int = 540, jpeg_quality: int = 70):
        self.max_w = max_w
        self.max_h = max_h
        self.jpeg_quality = jpeg_quality
        self.queue: queue.Queue = queue.Queue(maxsize=1)
        self.running = False
        self.user_closed = False
        self.thread: Optional[threading.Thread] = None
        self.display_handle = None
        try:
            from IPython.display import HTML, display

            self.display_handle = display(HTML("<b>[FastALPR] Live Preview Active...</b>"), display_id=True)
            self._has_ipython = True
        except Exception:
            self._has_ipython = False

    def start(self) -> None:
        if not self._has_ipython or self.display_handle is None:
            return
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def update_frame(self, frame: np.ndarray) -> None:
        if not self.running:
            return
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            pass

    def _worker_loop(self) -> None:
        from IPython.display import Image

        while self.running:
            try:
                frame = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                h, w = frame.shape[:2]
                scale = min(self.max_w / w, self.max_h / h, 1.0)
                preview = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
                ret, buf = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                if ret and self.display_handle is not None:
                    self.display_handle.update(Image(data=buf.tobytes(), format="jpg"))
            except Exception:
                pass

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)


class DesktopPreviewWorker:
    """OpenCV GUI preview window for local desktop execution."""

    def __init__(self, max_w: int = 960, max_h: int = 540):
        self.max_w = max_w
        self.max_h = max_h
        self.running = False
        self.user_closed = False
        self.window_name = "FastALPR Realtime Preview"

    def start(self) -> None:
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.max_w, self.max_h)
            self.running = True
        except Exception as e:
            print(f"[PREVIEW] Display unavailable: {e}")
            self.running = False

    def update_frame(self, frame: np.ndarray) -> None:
        if not self.running:
            return
        h, w = frame.shape[:2]
        scale = min(self.max_w / w, self.max_h / h, 1.0)
        preview = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
        try:
            cv2.imshow(self.window_name, preview)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q"), ord("Q")):
                self.running = False
                self.user_closed = True
        except Exception:
            self.running = False

    def stop(self) -> None:
        self.running = False
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass


class VehicleALPRPipeline:
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        run_id: Optional[str] = None,
        video_source: Optional[str] = None,
    ):
        self.config = config or PipelineConfig()
        self.run_id = run_id or getattr(self.config, "run_id", "") or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.video_source = video_source
        self.vehicle_model = None
        self.alpr = None
        self.byte_tracker = None
        self.has_cuda = False

        from sinks import EventBus
        self.event_bus = EventBus(
            database_url=getattr(self.config, "database_url", None),
            ws_port=getattr(self.config, "ws_port", 8765),
            enable_ws=getattr(self.config, "enable_ws", True),
            s3_endpoint_url=getattr(self.config, "s3_endpoint_url", None),
            s3_access_key_id=getattr(self.config, "s3_access_key_id", None),
            s3_secret_access_key=getattr(self.config, "s3_secret_access_key", None),
            s3_region=getattr(self.config, "s3_region", "us-east-2"),
            s3_bucket=getattr(self.config, "s3_bucket", "alpr-snapshots"),
            enable_s3=getattr(self.config, "enable_s3", True),
            snapshots_dir=self.config.snapshots_dir,
            events_json_path=self.config.events_json_path,
            events_jsonl_path=self.config.events_jsonl_path,
            run_id=self.run_id,
            camera_id=self.config.camera_id,
            no_local_snapshots=getattr(self.config, "no_local_snapshots", False),
        )
        self.tracker_manager = VehicleTrackerManager(self.config, event_bus=self.event_bus)
        self.preview_worker: Optional[ColabLivePreviewWorker | DesktopPreviewWorker] = None

    def _extract_ocr_confidences(self, ocr_item, plate_crop: np.ndarray, frame_idx: int) -> Tuple[str, float, List[float]]:
        """Extracts raw text, mean OCR confidence, and per-character confidences."""
        raw_text = str(ocr_item.text) if (ocr_item and ocr_item.text) else ""
        per_char_confs: Optional[List[float]] = None

        if ocr_item is not None and isinstance(ocr_item.confidence, (list, tuple, np.ndarray)):
            per_char_confs = [float(c) for c in ocr_item.confidence]
            ocr_conf = float(np.mean(per_char_confs)) if per_char_confs else 0.0
        elif ocr_item is not None and isinstance(ocr_item.confidence, (float, int)):
            recognizer = getattr(getattr(self.alpr, "ocr", None), "ocr_model", None)
            if recognizer is not None and plate_crop is not None and plate_crop.size > 0:
                rec_crop = plate_crop
                mode = getattr(getattr(recognizer, "config", None), "image_color_mode", None)
                if mode == "grayscale":
                    rec_crop = cv2.cvtColor(rec_crop, cv2.COLOR_BGR2GRAY)
                elif mode == "rgb":
                    rec_crop = cv2.cvtColor(rec_crop, cv2.COLOR_BGR2RGB)
                try:
                    pred = recognizer.run_one(rec_crop, return_confidence=True)
                    if getattr(pred, "char_probs", None) is not None:
                        per_char_confs = [float(x) for x in pred.char_probs.tolist()]
                        if getattr(pred, "plate", None):
                            raw_text = str(pred.plate)
                except Exception:
                    pass
            ocr_conf = float(np.mean(per_char_confs)) if per_char_confs else float(ocr_item.confidence)
            if not per_char_confs:
                per_char_confs = [ocr_conf] * len(raw_text)
        else:
            ocr_conf, per_char_confs = 0.0, []

        if getattr(self.config, "debug_reads", False) and raw_text:
            print(f"[DEBUG READ] Frame {frame_idx:04d} | Text: '{raw_text:<10}' | Conf: {ocr_conf:.3f}")

        return raw_text, ocr_conf, per_char_confs

    def _init_byte_tracker(self) -> None:
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace, YAML

        tracker_cfg_path = Path(self.config.tracker_config)
        if not tracker_cfg_path.is_absolute():
            tracker_cfg_path = REPO_ROOT / tracker_cfg_path
        cfg = IterableSimpleNamespace(**YAML.load(str(tracker_cfg_path)))
        self.byte_tracker = BYTETracker(args=cfg)

    def initialize_models(self) -> None:
        """Verifies CUDA/ONNX providers and loads YOLO11 + FastALPR models."""
        try:
            register_nvidia_dll_paths()
            import onnxruntime as ort

            if hasattr(ort, "preload_dlls"):
                try:
                    ort.preload_dlls()
                except Exception:
                    pass
            providers = ort.get_available_providers()
            self.has_cuda = "CUDAExecutionProvider" in providers
            print(f"[ONNX RUNTIME] Providers: {providers} | CUDA={self.has_cuda}")
            detector_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.has_cuda else ["CPUExecutionProvider"]
        except Exception as e:
            print(f"[ONNX RUNTIME] Provider check warning: {e}")
            detector_providers = None

        if self.vehicle_model is None:
            from ultralytics import YOLO

            print(f"[LOAD] Vehicle YOLO: {self.config.vehicle_model_name}")
            self.vehicle_model = YOLO(self.config.vehicle_model_name)

        if self.alpr is None:
            print(f"[LOAD] FastALPR ({self.config.alpr_detector_model} + {self.config.alpr_ocr_model})")
            try:
                self.alpr = ALPR(
                    detector_model=self.config.alpr_detector_model,
                    detector_conf_thresh=self.config.alpr_detector_conf_thresh,
                    detector_providers=detector_providers,
                    ocr_model=self.config.alpr_ocr_model,
                    ocr_device="cuda" if self.has_cuda else "auto",
                )
            except Exception as e:
                print(f"[LOAD FALLBACK] Using default ALPR models ({e})")
                self.config.alpr_detector_model = "yolo-v9-t-384-license-plate-end2end"
                self.config.alpr_ocr_model = "cct-xs-v2-global-model"
                self.alpr = ALPR(
                    detector_model=self.config.alpr_detector_model,
                    detector_conf_thresh=self.config.alpr_detector_conf_thresh,
                    detector_providers=detector_providers,
                    ocr_model=self.config.alpr_ocr_model,
                    ocr_device="cuda" if self.has_cuda else "auto",
                )

    def _open_capture(self, video_source: str | int | Path, start_frame: int) -> cv2.VideoCapture:
        src_str = str(video_source)
        if isinstance(video_source, int) or src_str.isdigit():
            cap = cv2.VideoCapture(int(video_source))
        elif src_str.startswith(("rtsp://", "http://", "https://", "rtsps://")):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(src_str, cv2.CAP_FFMPEG)
        else:
            video_path = Path(video_source)
            if not video_path.exists():
                raise FileNotFoundError(f"Video not found: {video_path.resolve()}")
            cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {video_source}")
        if start_frame > 0 and not cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
            print(f"[INPUT WARNING] Could not seek to frame {start_frame}.")
        return cap

    def _start_preview_worker(self) -> None:
        if not self.config.show_preview:
            return
        if _in_ipython_notebook():
            worker = ColabLivePreviewWorker(
                max_w=self.config.preview_max_width,
                max_h=self.config.preview_max_height,
                jpeg_quality=self.config.preview_jpeg_quality,
            )
            worker.start()
            if worker.running:
                self.preview_worker = worker
        elif _has_gui_display():
            worker = DesktopPreviewWorker(
                max_w=self.config.preview_max_width,
                max_h=self.config.preview_max_height,
            )
            worker.start()
            if worker.running:
                self.preview_worker = worker
        else:
            print("[PREVIEW] Headless mode: skipping GUI preview window.")

    def _cleanup_unreferenced_snapshots(self) -> None:
        """Removes any file in output/snapshots not referenced in emitted events."""
        if getattr(self.config, "debug_outputs", False) or not self.config.snapshots_dir.exists():
            return
        referenced: set[str] = set()
        events = list(self.tracker_manager.emitted_events)
        if self.config.events_json_path.exists():
            try:
                loaded = json.loads(self.config.events_json_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    events.extend(loaded)
            except Exception:
                pass
        for ev in events:
            if ev.get("snapshot_vehicle"):
                referenced.add(Path(ev["snapshot_vehicle"]).name)
            if ev.get("snapshot_number_plate"):
                referenced.add(Path(ev["snapshot_number_plate"]).name)
        for item in self.config.snapshots_dir.iterdir():
            if item.is_file() and item.name not in referenced:
                item.unlink(missing_ok=True)

    def _finalize_video(self, raw_path: Path, final_path: Path) -> None:
        ffmpeg_bin = shutil.which("ffmpeg")
        if self.config.convert_h264 and ffmpeg_bin:
            cmd = [
                ffmpeg_bin, "-y", "-i", str(raw_path),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "23",
                "-movflags", "+faststart", str(final_path),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                raw_path.unlink(missing_ok=True)
                print(f"[OUTPUT] Saved H.264 video: {final_path.resolve()}")
            else:
                print(f"[FFMPEG WARNING] {res.stderr.strip()[:200]}")
                raw_path.rename(final_path)
        else:
            raw_path.rename(final_path)

        if final_path.exists():
            self.event_bus.upload_artifact(final_path, final_path.name)

    def _run_alpr_for_tracks(self, frame: np.ndarray, active_ids: set[int], frame_idx: int) -> None:
        unresolved = [
            t for tid in active_ids
            if (t := self.tracker_manager.tracks.get(tid)) is not None
            and not t.plate_confident_reached
            and (not self.config.enable_smart_gating
                 or (t.entered_readable_zone and t.best_vehicle_area >= self.config.min_readable_vehicle_area))
        ]
        if not unresolved:
            return

        if not self.config.alpr_on_crops:
            preds = self.alpr.predict(frame)
            if preds:
                self.tracker_manager.match_and_update_plates(preds, frame, frame_idx, self._extract_ocr_confidences)
            return

        hf, wf = frame.shape[:2]
        for t in unresolved:
            box = t.raw_box if t.raw_box is not None else t.smoothed_box
            if box is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            px, py = int((x2 - x1) * 0.08), int((y2 - y1) * 0.08)
            cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
            cx2, cy2 = min(wf, x2 + px), min(hf, y2 + py)
            crop_w, crop_h = cx2 - cx1, cy2 - cy1
            if crop_w < 50 or crop_h < 40:
                continue

            v_crop = frame[cy1:cy2, cx1:cx2]
            scale = 1.0
            if crop_w < self.config.alpr_crop_min_width:
                scale = float(self.config.alpr_crop_min_width) / float(crop_w)
                v_crop_scaled = cv2.resize(
                    v_crop, (self.config.alpr_crop_min_width, max(64, int(crop_h * scale))), interpolation=cv2.INTER_CUBIC
                )
            else:
                v_crop_scaled = v_crop

            for item in self.alpr.predict(v_crop_scaled):
                if not (item.ocr and item.ocr.text):
                    continue
                pb = item.detection.bounding_box
                px1 = max(0, int(pb.x1 / scale))
                py1 = max(0, int(pb.y1 / scale))
                px2 = min(v_crop.shape[1], int(pb.x2 / scale))
                py2 = min(v_crop.shape[0], int(pb.y2 / scale))
                plate_crop = v_crop[py1:py2, px1:px2].copy()

                pfx1, pfy1, pfx2, pfy2 = cx1 + px1, cy1 + py1, cx1 + px2, cy1 + py2
                if t.raw_box is not None and t.current_frame_candidate is not None:
                    rx1, ry1, rx2, ry2 = t.raw_box
                    if (pfx1 >= rx1 - 5) and (pfy1 >= ry1 - 5) and (pfx2 <= rx2 + 5) and (pfy2 <= ry2 + 5):
                        t.current_frame_candidate["plate_inside"] = True

                if t.raw_box is not None and not t.plate_confident_reached:
                    rx1, ry1, rx2, ry2 = t.raw_box
                    rw, rh = max(1.0, rx2 - rx1), max(1.0, ry2 - ry1)
                    cur_unclipped = (ry1 > 3.0) and (ry2 < float(hf) - 3.0)
                    prev_unclipped = bool(t.plate_anchor and t.plate_anchor.get("unclipped"))
                    if (not prev_unclipped) or cur_unclipped:
                        t.plate_rel_box = [(pfx1 - rx1) / rw, (pfy1 - ry1) / rh, (pfx2 - rx1) / rw, (pfy2 - ry1) / rh]
                        t.plate_anchor = {
                            "plate_box": [float(pfx1), float(pfy1), float(pfx2), float(pfy2)],
                            "veh_box": [float(rx1), float(ry1), float(rx2), float(ry2)],
                            "clipped_top": bool(ry1 <= 3.0),
                            "clipped_bottom": bool(ry2 >= float(hf) - 3.0),
                            "unclipped": cur_unclipped,
                        }

                raw_text, ocr_conf, char_confs = self._extract_ocr_confidences(item.ocr, plate_crop, frame_idx)
                self.tracker_manager.update_track_plate(
                    track_id=t.track_id,
                    raw_text=raw_text,
                    ocr_conf=ocr_conf,
                    det_conf=float(item.detection.confidence),
                    plate_crop=plate_crop,
                    frame=frame,
                    per_char_confs=char_confs,
                    plate_w=max(1, px2 - px1),
                    plate_h=max(1, py2 - py1),
                    frame_idx=frame_idx,
                    plate_bbox_frame=[float(pfx1), float(pfy1), float(pfx2), float(pfy2)],
                )

    def process_video(
        self,
        video_source: str | int | Path,
        output_video_path: Optional[str | Path] = None,
        max_frames: Optional[int] = None,
        start_frame: int = 0,
    ) -> list[dict]:
        """Processes video end-to-end with single-pass YOLO + ByteTrack and fan-out sinks."""
        self.initialize_models()
        self._init_byte_tracker()
        self.event_bus.start(
            camera_id=self.config.camera_id,
            camera_name=getattr(self.config, "camera_name", ""),
            camera_location=getattr(self.config, "camera_location", ""),
            node_id=getattr(self.config, "node_id", "A"),
            run_id=self.run_id,
            source=str(video_source or self.video_source or ""),
        )
        if getattr(self.config, "replay_outbox", False):
            self.event_bus.replay_outbox()

        cap = None
        writer, raw_output_path, frames_written = None, None, 0
        frame_idx = 0
        prof = {"detect": 0.0, "track": 0.0, "alpr": 0.0, "render": 0.0}
        wall_t0 = time.perf_counter()
        run_status = "done"
        completed = False

        try:
            cap = self._open_capture(video_source, start_frame)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            expected_duration = total_frames / fps if total_frames > 0 else 0.0

            if self.config.save_video and output_video_path is not None:
                out_p = Path(output_video_path)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                raw_output_path = out_p.with_name("raw_" + out_p.name)
                writer = cv2.VideoWriter(str(raw_output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

            self._start_preview_worker()
            print(f"[PROCESS] {width}x{height} @ {fps:.1f} FPS | {total_frames} frames ({expected_duration:.2f}s)")

            rolling_fps, fps_t0, fps_cnt = fps, time.time(), 0
            while cap.isOpened():
                if max_frames and frame_idx >= max_frames:
                    break
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frame_idx += 1
                fps_cnt += 1

                now_t = time.time()
                if (now_t - fps_t0) >= 1.0:
                    rolling_fps = fps_cnt / (now_t - fps_t0)
                    fps_cnt, fps_t0 = 0, now_t

                # 1. Single-pass Vehicle Detection
                t0 = time.perf_counter()
                det_res = self.vehicle_model.predict(
                    source=frame,
                    classes=self.config.vehicle_classes,
                    conf=self.config.vehicle_conf_thresh,
                    imgsz=self.config.vehicle_imgsz,
                    verbose=False,
                )
                prof["detect"] += time.perf_counter() - t0

                # 2. ByteTrack Association (1:1 raw box via det index)
                t0 = time.perf_counter()
                active_ids: set[int] = set()
                if det_res and det_res[0].boxes is not None:
                    boxes_obj = det_res[0].boxes
                    raw_xyxys = boxes_obj.xyxy.cpu().tolist()
                    tracks = self.byte_tracker.update(boxes_obj.cpu().numpy(), frame)

                    if len(tracks) > 0:
                        k_boxes = [row[:4].tolist() for row in tracks]
                        scores = [float(row[5]) for row in tracks]
                        clean_boxes, _ = filter_nested_boxes(k_boxes, scores, self.config.box_containment_thresh)
                        clean_arr = np.array(clean_boxes) if clean_boxes else np.empty((0, 4))

                        for row in tracks:
                            k_box = [float(v) for v in row[:4]]
                            if len(clean_arr) > 0 and any(np.allclose(k_box, cb, atol=2.0) for cb in clean_arr):
                                t_id = int(row[4])
                                det_idx = int(row[7]) if len(row) > 7 else -1
                                r_box = raw_xyxys[det_idx] if 0 <= det_idx < len(raw_xyxys) else k_box
                                active_ids.add(t_id)
                                self.tracker_manager.update_track(t_id, k_box, frame, frame_idx, raw_det_box=r_box)

                self.tracker_manager.mark_disappeared_tracks(active_ids)
                prof["track"] += time.perf_counter() - t0

                # 3. FastALPR OCR (every K frames on unresolved tracks)
                t0 = time.perf_counter()
                if frame_idx % self.config.alpr_frame_interval == 0:
                    self._run_alpr_for_tracks(frame, active_ids, frame_idx)
                prof["alpr"] += time.perf_counter() - t0

                # 4. Event emission
                self.tracker_manager.check_and_emit_events(frame_idx)

                # 5. Render & write video / preview
                t0 = time.perf_counter()
                if self.config.save_video or self.config.show_preview:
                    annotated = self._render_frame(frame, live_fps=rolling_fps)
                    if writer is not None:
                        writer.write(annotated)
                        frames_written += 1
                    if self.preview_worker is not None and (frame_idx % self.config.preview_interval == 0):
                        self.preview_worker.update_frame(annotated)
                        if getattr(self.preview_worker, "user_closed", False):
                            print("[INFO] Preview closed by user.")
                            break
                prof["render"] += time.perf_counter() - t0

                if frame_idx % 60 == 0 or frame_idx == 1:
                    u, p, np_c, act, sm = self.tracker_manager.get_counts()
                    print(f"[FRAME {frame_idx:05d}/{total_frames}] FPS:{rolling_fps:4.1f} | Vehicles:{u} (Plates:{p}, NoPlate:{np_c}) | Active:{act} | Small:{sm}")
            completed = True
        except KeyboardInterrupt:
            run_status = "done"
            raise
        except Exception:
            run_status = "failed"
            raise
        finally:
            if cap is not None:
                cap.release()
            if self.preview_worker is not None:
                self.preview_worker.stop()
            self.tracker_manager.check_and_emit_events(frame_idx, force_all=True)
            if writer is not None:
                writer.release()
            self.tracker_manager.snapshot_saver.stop()
            self.event_bus.stop(timeout=30.0)
            self._cleanup_unreferenced_snapshots()
            if not completed:
                wall_s = time.perf_counter() - wall_t0
                self.event_bus.finish_run(
                    frames=frame_idx,
                    fps=round(frame_idx / max(1e-5, wall_s), 2),
                    tracks_seen=self.tracker_manager.total_tracks_seen,
                    events_emitted=len(self.tracker_manager.emitted_events),
                    status=run_status,
                )
                print(self.event_bus.get_status_str())
                self.event_bus.close_db()

        if raw_output_path is not None and raw_output_path.exists():
            self._finalize_video(raw_output_path, Path(output_video_path))

        wall_s = time.perf_counter() - wall_t0
        self.event_bus.finish_run(
            frames=frame_idx,
            fps=round(frame_idx / max(1e-5, wall_s), 2),
            tracks_seen=self.tracker_manager.total_tracks_seen,
            events_emitted=len(self.tracker_manager.emitted_events),
            status=run_status,
        )
        print(self.event_bus.get_status_str())
        self.event_bus.close_db()
        print(
            f"\n[SUMMARY] frames={frame_idx} time={wall_s:.2f}s ({frame_idx/max(1e-5,wall_s):.1f} FPS) | "
            f"tracks_seen={self.tracker_manager.total_tracks_seen} "
            f"events_emitted={len(self.tracker_manager.emitted_events)} "
            f"stitched={self.tracker_manager.total_tracks_stitched} "
            f"dropped={self.tracker_manager.total_tracks_dropped}"
        )
        print(
            f"[PROFILE] detect={prof['detect']:.1f}s | track={prof['track']:.1f}s | "
            f"alpr={prof['alpr']:.1f}s | render={prof['render']:.1f}s"
        )
        for m in self.tracker_manager.emitted_events_meta:
            plate = m["plate_text"] or "NO PLATE"
            conf = f"({m['plate_conf']*100:.0f}%)" if m.get("plate_conf") else ""
            print(f"  - {m['event_id']} | tracks={m['merged_track_ids']} | {plate} {conf}")

        return self.tracker_manager.emitted_events

    def _render_frame(self, frame: np.ndarray, live_fps: float = 0.0) -> np.ndarray:
        """Draws vehicle boxes, persistent anchored green plate boxes, and HUD counters."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        if self.config.roi_polygon and len(self.config.roi_polygon) >= 3:
            cv2.polylines(annotated, [np.array(self.config.roi_polygon, np.int32).reshape((-1, 1, 2))], True, (255, 255, 0), 2)

        for track in self.tracker_manager.tracks.values():
            if track.stitched_into is not None or track.disappeared_count > 0 or track.smoothed_box is None:
                continue
            if not track.entered_readable_zone and track.best_plate_text is None:
                continue

            box_draw = track.hud_smoothed_box or track.smoothed_box
            x1, y1, x2, y2 = [max(0, min(lim, int(v))) for v, lim in zip(box_draw, (w, h, w, h))]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 140, 255), 2)

            label = (
                f"{track.global_id} | {track.best_plate_text} ({(track.best_plate_conf or 0.0) * 100:.0f}%)"
                if track.best_plate_text
                else f"{track.global_id} | NO PLATE"
            )
            (tw, th), base = cv2.getTextSize(label, font, 0.55, 2)
            ly = max(th + 5, y1 - 4)
            cv2.rectangle(annotated, (x1, ly - th - 5), (x1 + tw + 6, ly + base + 2), (0, 140, 255), -1)
            cv2.putText(annotated, label, (x1 + 3, ly), font, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

            if self.config.draw_plate_boxes and (track.plate_rel_box is not None or track.plate_anchor is not None):
                ref_b = track.raw_box if (track.disappeared_count == 0 and track.raw_box is not None) else box_draw
                bx1, by1 = max(0.0, float(ref_b[0])), max(0.0, float(ref_b[1]))
                bx2, by2 = min(float(w), float(ref_b[2])), min(float(h), float(ref_b[3]))
                vw, vh = max(1.0, bx2 - bx1), max(1.0, by2 - by1)

                if track.plate_anchor is not None:
                    ax1, ay1, ax2, ay2 = track.plate_anchor["veh_box"]
                    apx1, apy1, apx2, apy2 = track.plate_anchor["plate_box"]
                    aw, ah = max(1.0, ax2 - ax1), max(1.0, ay2 - ay1)
                    sw = vw / aw
                    px1, px2 = bx1 + ((apx1 - ax1) / aw) * vw, bx1 + ((apx2 - ax1) / aw) * vw
                    if by1 <= 3.0 and by2 < float(h) - 3.0 and not track.plate_anchor.get("clipped_bottom", False):
                        py2 = by2 - (ay2 - apy2) * sw
                        py1 = py2 - (apy2 - apy1) * sw
                    elif by2 >= float(h) - 3.0 and by1 > 3.0 and not track.plate_anchor.get("clipped_top", False):
                        py1 = by1 + (apy1 - ay1) * sw
                        py2 = py1 + (apy2 - apy1) * sw
                    else:
                        py1, py2 = by1 + ((apy1 - ay1) / ah) * vh, by1 + ((apy2 - ay1) / ah) * vh
                else:
                    rb = track.plate_rel_box
                    px1, py1, px2, py2 = bx1 + rb[0] * vw, by1 + rb[1] * vh, bx1 + rb[2] * vw, by1 + rb[3] * vh

                ipx1, ipy1, ipx2, ipy2 = max(0, int(px1)), max(0, int(py1)), min(w, int(px2)), min(h, int(py2))
                if ipx2 > ipx1 and ipy2 > ipy1:
                    cv2.rectangle(annotated, (ipx1, ipy1), (ipx2, ipy2), (0, 255, 0), 3)
                    p_lbl = (
                        f"{track.best_plate_text} {(track.best_plate_conf or 0.0) * 100:.0f}%"
                        if track.best_plate_text
                        else "reading..."
                    )
                    (ptw, pth), _ = cv2.getTextSize(p_lbl, font, 0.65, 2)
                    p_top = ipy1 - pth - 8 >= 0
                    ply = (ipy1 - 4) if p_top else (ipy2 + pth + 4)
                    py_bg1, py_bg2 = ((ipy1 - pth - 8, ipy1) if p_top else (ipy2, ipy2 + pth + 8))
                    cv2.rectangle(annotated, (ipx1, py_bg1), (ipx1 + ptw + 6, py_bg2), (0, 255, 0), -1)
                    cv2.putText(annotated, p_lbl, (ipx1 + 3, ply), font, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

        u_cnt, p_cnt, np_cnt, act_cnt, sm_cnt = self.tracker_manager.get_counts()
        hud = [
            f"CAMERA: {self.config.camera_id} | FPS: {live_fps:.1f}",
            f"Unique Vehicles: {u_cnt} (Plates: {p_cnt}, NoPlate: {np_cnt})",
            f"Active: {act_cnt} | Ignored Small: {sm_cnt}",
        ]
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (410, 24 * len(hud) + 18), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
        cv2.rectangle(annotated, (10, 10), (410, 24 * len(hud) + 18), (255, 255, 255), 1)
        for i, line in enumerate(hud):
            cv2.putText(annotated, line, (20, 32 + i * 24), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated
