"""
Track state management, box deduplication, character-level OCR voting,
strict non-chimeric track stitching, and single-event emission.
"""

import json
import math
import os
import queue
import re
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from config import PipelineConfig

BASE36_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class AsyncSnapshotSaver:
    """Non-blocking background worker for saving PNG snapshots with fast lossless compression."""

    def __init__(self):
        self.queue: queue.Queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def submit(self, path: Path, image: np.ndarray) -> None:
        if self.running and image is not None and image.size > 0:
            self.queue.put((str(path), image))

    def _worker(self) -> None:
        png_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
        while self.running or not self.queue.empty():
            try:
                path_str, img = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                cv2.imwrite(path_str, img, png_params)
            except Exception as e:
                print(f"[SNAPSHOT SAVE ERROR] Could not save {path_str}: {e}")
            finally:
                self.queue.task_done()

    def stop(self) -> None:
        self.running = False
        self.queue.join()


def levenshtein_distance(s1: str, s2: str, wildcard: Optional[str] = None) -> int:
    """Computes Levenshtein edit distance between two strings (optional wildcard has zero sub cost)."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            same = (c1 == c2) or (wildcard is not None and (c1 == wildcard or c2 == wildcard))
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (0 if same else 1)))
        prev = curr
    return prev[-1]


def generate_infinite_event_id(node_id: str, camera_id_str: str, now: datetime, seq_idx: int = 0) -> str:
    """Generates unique event ID: {NODE}{CAM}-{YYMMDD}{HHMMSS}-{SEQ}."""
    match = re.search(r"\d+", str(camera_id_str))
    cam_int = int(match.group(0)) if match else 1
    cam_str = f"{cam_int:03d}" if cam_int > 99 else f"{cam_int:02d}"
    n = seq_idx % 1296
    seq_str = BASE36_ALPHABET[n // 36] + BASE36_ALPHABET[n % 36]
    return f"{node_id}{cam_str}-{now.strftime('%y%m%d%H%M%S')}-{seq_str}"


def get_image_sharpness(img: Optional[np.ndarray]) -> float:
    """Calculates image sharpness score using Laplacian variance."""
    if img is None or img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def filter_nested_boxes(
    boxes: List[List[float]],
    scores: List[float],
    containment_thresh: float = 0.70,
) -> Tuple[List[List[float]], List[float]]:
    """Suppresses nested duplicate boxes where one box is substantially contained inside another."""
    if len(boxes) <= 1:
        return boxes, scores

    keep: Set[int] = set(range(len(boxes)))
    for i in range(len(boxes)):
        if i not in keep:
            continue
        ax1, ay1, ax2, ay2 = boxes[i]
        area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
        if area_a <= 0:
            keep.discard(i)
            continue

        for j in range(len(boxes)):
            if i == j or j not in keep:
                continue
            bx1, by1, bx2, by2 = boxes[j]
            area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
            if area_b <= 0:
                continue

            iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
            ih = max(0.0, min(ay2, by2) - max(ay1, by1))
            inter = iw * ih
            if (inter / area_a) >= containment_thresh or (inter / area_b) >= containment_thresh:
                if scores[i] < scores[j]:
                    keep.discard(i)
                    break
                keep.discard(j)

    idx_sorted = sorted(keep)
    return [boxes[k] for k in idx_sorted], [scores[k] for k in idx_sorted]


def _normalize_plate_chars(clean: str, fmt: str, syntax_correction: bool, uk_syntax_correction: bool) -> str:
    """Applies country-specific optical character disambiguation when enabled."""
    d2l = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G"}
    l2d = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8"}

    if fmt == "UK" and uk_syntax_correction and len(clean) == 7:
        chars = list(clean)
        for i in (0, 1, 4, 5, 6):
            chars[i] = d2l.get(chars[i], chars[i])
        for i in (2, 3):
            chars[i] = l2d.get(chars[i], chars[i])
        return "".join(chars)

    if fmt == "PAKISTAN" and syntax_correction and len(clean) >= 3:
        chars = list(clean)
        if any(c.isalpha() for c in chars[:3]):
            for i in range(min(2, len(chars))):
                chars[i] = d2l.get(chars[i], chars[i])
        if any(c.isdigit() for c in chars[-3:]):
            for i in range(max(2, len(chars) - 4), len(chars)):
                chars[i] = l2d.get(chars[i], chars[i])
        return "".join(chars)

    return clean


def is_fuzzy_duplicate_plate(p1: str, p2: str, max_distance: int = 2) -> bool:
    """Compares two plates ignoring '?' positions; requires >= 5 known characters on both."""
    if sum(1 for c in p1 if c != "?") < 5 or sum(1 for c in p2 if c != "?") < 5:
        return False
    if abs(len(p1) - len(p2)) > max_distance:
        return False
    return levenshtein_distance(p1, p2, wildcard="?") <= max_distance


def character_level_voting(
    readings: List[Tuple],
    min_length: int = 4,
    regex_pattern: Optional[str] = None,
    uk_syntax_correction: bool = False,
    plate_format: str = "PAKISTAN",
    syntax_correction: bool = False,
) -> Optional[Tuple[str, float]]:
    """
    Confidence^4 weighted character voting across OCR reads of a single track.
    Replaces positions with '?' if winning share < 0.5 or avg char conf < 0.5.
    """
    if not readings:
        return None

    valid_readings: List[Tuple[str, float, List[float]]] = []
    for item in readings:
        text = str(item[0]).upper()
        conf = float(item[1])
        char_confs_in = [float(c) for c in item[2]] if (len(item) >= 3 and item[2]) else [conf] * len(text)

        clean_chars, clean_confs = [], []
        for ch, c_conf in zip(text, char_confs_in):
            if ch.isalnum():
                clean_chars.append(ch)
                clean_confs.append(float(c_conf))

        clean_text = _normalize_plate_chars("".join(clean_chars), plate_format, syntax_correction, uk_syntax_correction)
        if len(clean_text) >= 3:
            valid_readings.append((clean_text, conf, clean_confs))

    if not valid_readings:
        return None

    length_votes: Dict[int, float] = defaultdict(float)
    for text, conf, _ in valid_readings:
        length_votes[len(text)] += (conf ** 4)

    best_len = max(length_votes, key=lambda l: length_votes[l])
    same_len = [r for r in valid_readings if len(r[0]) == best_len]

    consensus_chars, char_confs = [], []
    for pos in range(best_len):
        pos_votes: Dict[str, float] = defaultdict(float)
        mult = 1.35 if pos in (0, 1) else 1.0
        for text, conf, _ in same_len:
            pos_votes[text[pos]] += (conf ** 4) * mult

        total_w = sum(pos_votes.values())
        best_char = max(pos_votes, key=lambda c: pos_votes[c])
        share = pos_votes[best_char] / max(1e-5, total_w)

        matching_confs = [
            (p_confs[pos] if pos < len(p_confs) else conf)
            for text, conf, p_confs in same_len
            if text[pos] == best_char
        ]
        avg_char_conf = float(np.mean(matching_confs)) if matching_confs else 0.0
        char_confs.append(avg_char_conf)
        consensus_chars.append("?" if (share < 0.5 or avg_char_conf < 0.5) else best_char)

    raw_consensus = "".join(consensus_chars)
    if "?" not in raw_consensus:
        raw_consensus = _normalize_plate_chars(raw_consensus, plate_format, syntax_correction, uk_syntax_correction)

    known_confs = [c for ch, c in zip(consensus_chars, char_confs) if ch != "?"]
    avg_confidence = (
        float(np.mean(known_confs))
        if known_confs
        else (float(np.mean(char_confs)) if char_confs else float(same_len[0][1]))
    )
    return raw_consensus, avg_confidence


@dataclass
class TrackState:
    track_id: int
    global_id: str
    first_frame_idx: int
    last_frame_idx: int
    first_frame: int = 0
    last_frame: int = 0
    frames_tracked: int = 1
    disappeared_count: int = 0
    smoothed_box: Optional[List[float]] = None
    hud_smoothed_box: Optional[List[float]] = None
    raw_box: Optional[List[float]] = None
    first_raw_box: Optional[List[float]] = None
    last_raw_box: Optional[List[float]] = None
    frame_boxes: Dict[int, List[float]] = field(default_factory=dict)
    last_center: Optional[Tuple[float, float]] = None

    # Snapshot selection
    best_raw_box: Optional[List[float]] = None
    best_vehicle_box: Optional[List[int]] = None
    best_full_frame: Optional[np.ndarray] = None
    best_vehicle_crop: Optional[np.ndarray] = None
    best_vehicle_area: float = 0.0
    max_sharpness: float = 0.0
    best_unnorm_score: float = -1.0
    best_snapshot_score: float = -1.0
    best_snapshot_frame_idx: int = -1
    best_snapshot_complete: bool = False
    best_snapshot_sharp: float = 0.0
    has_complete_frame: bool = False
    max_vehicle_area: float = 0.0
    current_frame_candidate: Optional[dict] = None

    # Stitching & gating
    merged_track_ids: List[int] = field(default_factory=list)
    stitched_into: Optional[int] = None
    entered_readable_zone: bool = False

    # Plate consensus & crops
    plate_readings: List[Tuple] = field(default_factory=list)
    all_plate_reads: List[dict] = field(default_factory=list)
    emitted_event_id: Optional[str] = None
    plate_reads_flushed: bool = False
    best_plate_crop: Optional[np.ndarray] = None
    best_plate_text: Optional[str] = None
    best_plate_conf: Optional[float] = None
    best_plate_det_conf: Optional[float] = None

    # Vehicle type classification (conf-weighted voting)
    class_votes: Dict[str, float] = field(default_factory=dict)
    vehicle_type: str = ""

    # Lifecycle & visualization
    has_event_emitted: bool = False
    plate_confident_reached: bool = False
    has_unclipped_high_conf_plate: bool = False
    last_plate_box_frame: Optional[List[float]] = None
    last_plate_box_frame_idx: int = -1
    plate_rel_box: Optional[List[float]] = None
    plate_anchor: Optional[dict] = None
    kalman_box: Optional[List[float]] = None

    def __post_init__(self) -> None:
        if self.first_frame == 0 and self.first_frame_idx != 0:
            self.first_frame = self.first_frame_idx
        elif self.first_frame_idx == 0 and self.first_frame != 0:
            self.first_frame_idx = self.first_frame
        if self.last_frame == 0 and self.last_frame_idx != 0:
            self.last_frame = self.last_frame_idx
        elif self.last_frame_idx == 0 and self.last_frame != 0:
            self.last_frame_idx = self.last_frame
        if self.first_raw_box is None and self.raw_box is not None:
            self.first_raw_box = [float(v) for v in self.raw_box]
        if self.last_raw_box is None and self.raw_box is not None:
            self.last_raw_box = [float(v) for v in self.raw_box]


class VehicleTrackerManager:
    def __init__(self, config: PipelineConfig, event_bus: Optional[Any] = None):
        self.config = config
        self.event_bus = event_bus
        self.tracks: Dict[int, TrackState] = {}
        self.emitted_events: List[dict] = []
        self.emitted_events_meta: List[dict] = []
        self.emitted_track_ids: Set[int] = set()
        self.recent_emitted_plates: Dict[str, Tuple[float, str]] = {}
        self.event_counter: int = 0
        self._last_sec_key: Optional[str] = None
        self._sec_seq_counter: int = 0

        self.snapshot_saver = AsyncSnapshotSaver()

        self.total_unique_vehicles: int = 0
        self.total_with_plate: int = 0
        self.total_without_plate: int = 0
        self.total_filtered_too_small: int = 0
        self.total_tracks_seen: int = 0
        self.total_tracks_stitched: int = 0
        self.total_tracks_dropped: int = 0
        self.total_tracks_emitted: int = 0

        if self.config.events_jsonl_path.exists():
            self.config.events_jsonl_path.write_text("", encoding="utf-8")
        if (
            getattr(self.config, "debug_outputs", False)
            and self.config.collect_plate_data
            and self.config.plate_reads_jsonl_path.exists()
        ):
            self.config.plate_reads_jsonl_path.write_text("", encoding="utf-8")

    def is_in_roi(self, box: List[float], frame_shape: Tuple[int, int]) -> bool:
        """Checks whether the vehicle box lies inside the configured ROI or lower frame."""
        h, _ = frame_shape[:2]
        x1, y1, x2, y2 = box
        bottom_cx = (x1 + x2) / 2.0
        bottom_cy = y2

        if self.config.roi_polygon and len(self.config.roi_polygon) >= 3:
            pts = np.array(self.config.roi_polygon, dtype=np.int32)
            return cv2.pointPolygonTest(pts, (float(bottom_cx), float(bottom_cy)), measureDist=False) >= 0

        horizon_y = self.config.roi_top_cutoff_ratio * h
        return (bottom_cy >= horizon_y) and (y1 >= horizon_y * 0.7)

    def update_track(
        self,
        track_id: int,
        box: List[float],
        frame: np.ndarray,
        frame_idx: int,
        raw_det_box: Optional[List[float]] = None,
        vehicle_type: Optional[str] = None,
        det_conf: float = 0.0,
    ) -> TrackState:
        """Creates or updates track state, snapshot candidates, and border completeness."""
        kalman_box = [float(v) for v in box]
        raw_box = [float(v) for v in (raw_det_box if raw_det_box is not None else box)]

        rx1, ry1, rx2, ry2 = raw_box
        h, w = frame.shape[:2]
        cx1, cy1 = max(0.0, min(float(w), rx1)), max(0.0, min(float(h), ry1))
        cx2, cy2 = max(0.0, min(float(w), rx2)), max(0.0, min(float(h), ry2))

        raw_bw = max(0.0, rx2 - rx1)
        raw_bh = max(0.0, ry2 - ry1)
        raw_area = float(raw_bw * raw_bh)
        inside_area = float(max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1))

        clipped_sides = int(rx1 < 0.0) + int(ry1 < 0.0) + int(rx2 > float(w)) + int(ry2 > float(h))
        clipped_part = max(0.0, raw_area - inside_area) / max(1e-5, raw_area)
        complete = (clipped_sides <= 1) and (clipped_part < 0.15)

        pad_x, pad_y = int(raw_bw * 0.06), int(raw_bh * 0.06)
        vx1, vy1 = max(0, int(rx1 - pad_x)), max(0, int(ry1 - pad_y))
        vx2, vy2 = min(w, int(rx2 + pad_x)), min(h, int(ry2 + pad_y))
        vehicle_crop = frame[vy1:vy2, vx1:vx2].copy() if (vx2 > vx1 and vy2 > vy1) else None

        raw_sharp = get_image_sharpness(vehicle_crop) if vehicle_crop is not None else 0.0
        base_score = inside_area * raw_sharp * (1.0 if complete else 0.35)
        cx, cy = (rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0

        in_roi = self.is_in_roi([cx1, cy1, cx2, cy2], (h, w))
        is_readable = (
            raw_bw >= self.config.min_vehicle_width
            and raw_bh >= self.config.min_vehicle_height
            and raw_area >= self.config.min_readable_vehicle_area
        )

        if track_id not in self.tracks:
            self.total_tracks_seen += 1
            track = TrackState(
                track_id=track_id,
                global_id=f"{self.config.camera_id}:{track_id}",
                first_frame_idx=frame_idx,
                last_frame_idx=frame_idx,
                first_frame=frame_idx,
                last_frame=frame_idx,
                frames_tracked=1,
                smoothed_box=[rx1, ry1, rx2, ry2],
                hud_smoothed_box=list(kalman_box),
                kalman_box=kalman_box,
                raw_box=raw_box,
                first_raw_box=list(raw_box),
                last_raw_box=list(raw_box),
                frame_boxes={frame_idx: raw_box},
                last_center=(cx, cy),
                best_raw_box=raw_box if raw_det_box is not None else None,
                best_vehicle_box=[int(rx1), int(ry1), int(rx2), int(ry2)] if raw_det_box is not None else None,
                best_full_frame=frame.copy() if (raw_area > 0 and raw_det_box is not None) else None,
                best_vehicle_crop=vehicle_crop if raw_det_box is not None else None,
                best_vehicle_area=raw_area if raw_det_box is not None else 0.0,
                max_vehicle_area=raw_area,
                has_complete_frame=complete,
                max_sharpness=raw_sharp,
                merged_track_ids=[track_id],
                entered_readable_zone=(in_roi and is_readable),
            )
            self.tracks[track_id] = track
        else:
            track = self.tracks[track_id]
            track.kalman_box = kalman_box

        # Vehicle type voting (conf-weighted)
        if vehicle_type:
            track.class_votes[vehicle_type] = track.class_votes.get(vehicle_type, 0.0) + det_conf
            # Update vehicle_type to current winner
            if track.class_votes:
                track.vehicle_type = max(track.class_votes, key=track.class_votes.get)

            # Spatial jump guard: never wipe a locked plate or border-clipping box shifts
            if track.last_center is not None and track.frames_tracked > 2 and not track.plate_confident_reached:
                jump_dist = math.hypot(cx - track.last_center[0], cy - track.last_center[1])
                border_clipped = (rx1 <= 8.0) or (ry1 <= 8.0) or (rx2 >= float(w) - 8.0) or (ry2 >= float(h) - 8.0)
                if not border_clipped and jump_dist > self.config.max_spatial_jump_px:
                    track.plate_readings.clear()
                    track.best_plate_text = None
                    track.best_plate_conf = None
                    track.plate_confident_reached = False

            track.last_center = (cx, cy)
            track.last_frame_idx = frame_idx
            track.last_frame = frame_idx
            track.frames_tracked += 1
            track.disappeared_count = 0
            track.raw_box = raw_box
            track.last_raw_box = list(raw_box)
            if track.first_raw_box is None:
                track.first_raw_box = list(raw_box)
            track.frame_boxes[frame_idx] = raw_box
            track.has_complete_frame = track.has_complete_frame or complete
            track.max_vehicle_area = max(track.max_vehicle_area, raw_area)
            track.max_sharpness = max(track.max_sharpness, raw_sharp)
            if in_roi and is_readable:
                track.entered_readable_zone = True

            alpha = self.config.box_smoothing_factor
            if alpha == 0.0 or track.smoothed_box is None:
                track.smoothed_box = [rx1, ry1, rx2, ry2]
            else:
                track.smoothed_box = [alpha * c + (1.0 - alpha) * s for c, s in zip(raw_box, track.smoothed_box)]

            hud_alpha = getattr(self.config, "hud_box_smoothing_factor", 0.65)
            if track.hud_smoothed_box is None:
                track.hud_smoothed_box = list(kalman_box)
            else:
                track.hud_smoothed_box = [
                    hud_alpha * c + (1.0 - hud_alpha) * s for c, s in zip(kalman_box, track.hud_smoothed_box)
                ]

        if raw_det_box is not None:
            track.current_frame_candidate = {
                "frame_idx": frame_idx,
                "raw_box": raw_box,
                "full_frame": frame,
                "vehicle_crop": vehicle_crop,
                "raw_sharp": raw_sharp,
                "base_score": base_score,
                "complete": complete,
                "area_px": inside_area,
                "plate_inside": False,
            }
        else:
            track.current_frame_candidate = None

        return track

    def commit_frame_snapshots(self) -> None:
        """Scores and selects the best clean vehicle snapshot frame (1.5x bonus when plate inside)."""
        for track in self.tracks.values():
            cand = track.current_frame_candidate
            if cand is None:
                continue

            unnorm_score = cand["base_score"] * (1.5 if cand.get("plate_inside", False) else 1.0)
            track.max_sharpness = max(track.max_sharpness, cand["raw_sharp"])

            if unnorm_score > track.best_unnorm_score or track.best_full_frame is None:
                track.best_unnorm_score = unnorm_score
                track.best_snapshot_score = unnorm_score / max(1e-5, track.max_sharpness)
                track.best_full_frame = cand["full_frame"].copy()
                track.best_raw_box = cand["raw_box"]
                track.best_vehicle_box = [int(v) for v in cand["raw_box"]]
                track.best_vehicle_crop = cand["vehicle_crop"]
                track.best_snapshot_frame_idx = cand["frame_idx"]
                track.best_snapshot_complete = cand["complete"]
                track.best_snapshot_sharp = cand["raw_sharp"]

            track.current_frame_candidate = None

    def mark_disappeared_tracks(self, active_track_ids: set[int]) -> None:
        """Increments disappeared frame count for tracks missing from current frame."""
        for t_id, track in self.tracks.items():
            if t_id not in active_track_ids:
                track.disappeared_count += 1

    def update_track_plate(
        self,
        track_id: int,
        raw_text: str,
        ocr_conf: float,
        det_conf: float,
        plate_crop: np.ndarray,
        frame: np.ndarray,
        per_char_confs: Optional[List[float]] = None,
        plate_w: int = 0,
        plate_h: int = 0,
        frame_idx: int = 0,
        plate_bbox_frame: Optional[List[float]] = None,
    ) -> None:
        """Associates a plate OCR observation to a track with edge-cut filtering, voting, and locking."""
        track = self.tracks.get(track_id)
        if track is None or track.has_event_emitted:
            return

        h_f, w_f = frame.shape[:2]

        # Reject plates touching the 4px frame border to prevent partial edge reads
        if plate_bbox_frame is not None:
            pfx1, pfy1, pfx2, pfy2 = plate_bbox_frame
            if (pfx1 <= 4.0) or (pfy1 <= 4.0) or (pfx2 >= float(w_f) - 4.0) or (pfy2 >= float(h_f) - 4.0):
                return

        if plate_bbox_frame is not None and not track.plate_confident_reached:
            track.last_plate_box_frame = [float(v) for v in plate_bbox_frame]
            track.last_plate_box_frame_idx = frame_idx
            vbox = track.frame_boxes.get(frame_idx) or track.raw_box or track.smoothed_box
            if vbox is not None:
                vx1, vy1, vx2, vy2 = vbox
                vw, vh = max(1.0, vx2 - vx1), max(1.0, vy2 - vy1)
                cur_unclipped = (vy1 > 3.0) and (vy2 < float(h_f) - 3.0)
                prev_unclipped = bool(track.plate_anchor and track.plate_anchor.get("unclipped"))
                if (not prev_unclipped) or cur_unclipped:
                    px1, py1, px2, py2 = plate_bbox_frame
                    track.plate_rel_box = [(px1 - vx1) / vw, (py1 - vy1) / vh, (px2 - vx1) / vw, (py2 - vy1) / vh]
                    track.plate_anchor = {
                        "plate_box": [float(px1), float(py1), float(px2), float(py2)],
                        "veh_box": [float(vx1), float(vy1), float(vx2), float(vy2)],
                        "clipped_top": bool(vy1 <= 3.0),
                        "clipped_bottom": bool(vy2 >= float(h_f) - 3.0),
                        "unclipped": cur_unclipped,
                    }

        raw_str = str(raw_text).strip().upper()
        if per_char_confs is None or len(per_char_confs) != len(raw_str):
            per_char_confs = [ocr_conf] * len(raw_str)

        clean_chars, clean_confs = [], []
        for ch, c_conf in zip(raw_str, per_char_confs):
            if ch.isalnum():
                clean_chars.append(ch)
                clean_confs.append(float(c_conf))

        clean_text = _normalize_plate_chars(
            "".join(clean_chars),
            self.config.plate_format,
            self.config.plate_syntax_correction,
            self.config.uk_syntax_correction,
        )
        if len(clean_text) < min(self.config.min_plate_length, self.config.min_known_chars):
            return

        if track.plate_confident_reached and track.best_plate_text:
            return

        # Ignore degraded border-clipped reads once an unclipped crisp read exists
        vbox_chk = track.frame_boxes.get(frame_idx) or track.raw_box
        veh_clipped = bool(vbox_chk is not None and (vbox_chk[1] <= 5.0 or vbox_chk[3] >= float(h_f) - 5.0))
        if track.has_unclipped_high_conf_plate and veh_clipped and ocr_conf < 0.85:
            return
        if (not veh_clipped) and ocr_conf >= 0.85 and len(clean_text) >= self.config.min_plate_length:
            track.has_unclipped_high_conf_plate = True

        if get_image_sharpness(plate_crop) < self.config.min_plate_sharpness and track.best_plate_crop is not None:
            return

        actual_plate_w = plate_w if plate_w > 0 else (plate_crop.shape[1] if plate_crop is not None else 0)
        actual_plate_h = plate_h if plate_h > 0 else (plate_crop.shape[0] if plate_crop is not None else 0)
        track.plate_readings.append((clean_text, ocr_conf, clean_confs, actual_plate_w, actual_plate_h, frame_idx))

        if getattr(self.config, "debug_outputs", False) and self.config.collect_plate_data:
            track.all_plate_reads.append({
                "frame_idx": frame_idx,
                "raw_text": clean_text,
                "per_character_confidences": [round(float(c), 4) for c in clean_confs],
                "plate_w": int(actual_plate_w),
                "plate_h": int(actual_plate_h),
                "crop": plate_crop.copy() if plate_crop is not None else None,
                "score": float(ocr_conf) * float(actual_plate_w),
                "crop_file_name": None,
            })

        if track.best_plate_det_conf is None or float(det_conf) > float(track.best_plate_det_conf):
            track.best_plate_det_conf = float(det_conf)

        if track.best_plate_conf is None or ocr_conf > track.best_plate_conf:
            track.best_plate_crop = plate_crop.copy() if plate_crop is not None else None

        consensus = character_level_voting(
            readings=track.plate_readings,
            min_length=self.config.min_plate_length,
            regex_pattern=self.config.plate_regex_pattern if self.config.enforce_regex else None,
            uk_syntax_correction=self.config.uk_syntax_correction,
            plate_format=self.config.plate_format,
            syntax_correction=self.config.plate_syntax_correction,
        )

        if consensus is not None and not track.plate_confident_reached:
            voted_text, voted_conf = consensus
            track.best_plate_text = voted_text
            track.best_plate_conf = voted_conf

            is_ultra_high_conf = (
                "?" not in voted_text
                and len(voted_text) >= self.config.min_plate_length
                and voted_conf >= 0.88
                and actual_plate_w >= 60
            )
            is_multi_read_conf = (
                "?" not in voted_text
                and len(voted_text) >= self.config.min_plate_length
                and voted_conf >= self.config.min_confident_plate_conf
                and len(track.plate_readings) >= self.config.min_plate_reads_for_event
            )
            if is_ultra_high_conf or is_multi_read_conf:
                track.plate_confident_reached = True

    def match_and_update_plates(
        self,
        alpr_results: list,
        frame: np.ndarray,
        frame_idx: int,
        ocr_extractor: Optional[Callable] = None,
    ) -> None:
        """Fallback matcher when running ALPR on full frames instead of vehicle crops."""
        h, w = frame.shape[:2]
        for alpr_res in alpr_results:
            det, ocr = alpr_res.detection, alpr_res.ocr
            if ocr is None or not ocr.text:
                continue

            bbox = det.bounding_box
            px1, py1, px2, py2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
            pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0

            best_id, smallest_area = None, float("inf")
            for t_id, track in self.tracks.items():
                if track.disappeared_count > 0 or track.smoothed_box is None:
                    continue
                vx1, vy1, vx2, vy2 = track.smoothed_box
                if vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2:
                    area = (vx2 - vx1) * (vy2 - vy1)
                    if area < smallest_area:
                        smallest_area, best_id = area, t_id

            if best_id is None:
                continue

            c_px1, c_py1 = max(0, int(px1)), max(0, int(py1))
            c_px2, c_py2 = min(w, int(px2)), min(h, int(py2))
            plate_crop = frame[c_py1:c_py2, c_px1:c_px2].copy()
            if ocr_extractor is not None:
                raw_text, ocr_conf, per_char_confs = ocr_extractor(ocr, plate_crop, frame_idx)
            else:
                raw_text = str(ocr.text)
                if isinstance(ocr.confidence, (list, tuple, np.ndarray)):
                    per_char_confs = [float(c) for c in ocr.confidence]
                    ocr_conf = float(np.mean(per_char_confs)) if per_char_confs else 0.0
                else:
                    ocr_conf = float(ocr.confidence)
                    per_char_confs = [ocr_conf] * len(raw_text)

            self.update_track_plate(
                track_id=best_id,
                raw_text=raw_text,
                ocr_conf=ocr_conf,
                det_conf=float(det.confidence),
                plate_crop=plate_crop,
                frame=frame,
                per_char_confs=per_char_confs,
                plate_w=max(1, c_px2 - c_px1),
                plate_h=max(1, c_py2 - c_py1),
                frame_idx=frame_idx,
                plate_bbox_frame=[float(px1), float(py1), float(px2), float(py2)],
            )

            matched_track = self.tracks.get(best_id)
            if matched_track and matched_track.current_frame_candidate and matched_track.raw_box:
                rx1, ry1, rx2, ry2 = matched_track.raw_box
                if (c_px1 >= rx1 - 5) and (c_py1 >= ry1 - 5) and (c_px2 <= rx2 + 5) and (c_py2 <= ry2 + 5):
                    matched_track.current_frame_candidate["plate_inside"] = True

    def evaluate_stitching(self, t1: TrackState, t2: TrackState) -> Tuple[bool, float, int]:
        """
        Strict non-chimeric track stitching rules:
        Hard vetoes: overlapping frames, different plate texts (Levenshtein >= 2), gap < 0 or > max_gap.
        Spatial match uses t1.last_raw_box vs t2.first_raw_box ONLY (never best_raw_box).
        """
        t1_first = t1.first_frame if t1.first_frame is not None else t1.first_frame_idx
        t1_last = t1.last_frame if t1.last_frame is not None else t1.last_frame_idx
        t2_first = t2.first_frame if t2.first_frame is not None else t2.first_frame_idx
        t2_last = t2.last_frame if t2.last_frame is not None else t2.last_frame_idx

        if t1_first > t2_first:
            t1, t2 = t2, t1
            t1_first, t2_first = t2_first, t1_first
            t1_last, t2_last = t2_last, t1_last

        if max(t1_first, t2_first) <= min(t1_last, t2_last):
            return False, 0.0, 0

        if t1.best_plate_text and t2.best_plate_text:
            if levenshtein_distance(t1.best_plate_text, t2.best_plate_text) >= 2:
                return False, 0.0, 0

        max_gap = getattr(self.config, "stitch_max_gap_frames", 15)
        gap = t2_first - t1_last
        if gap < 0 or gap > max_gap:
            return False, 0.0, gap

        b1, b2 = t1.last_raw_box, t2.first_raw_box
        if b1 is None or b2 is None:
            return False, 0.0, gap

        iw = max(0.0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
        ih = max(0.0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
        inter = iw * ih
        area1 = max(0.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
        area2 = max(0.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
        union = area1 + area2 - inter
        iou = (inter / union) if union > 0.0 else 0.0

        if iou >= 0.30:
            return True, iou, gap

        c1_x, c1_y = (b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0
        c2_x, c2_y = (b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0

        if b1[1] <= 40.0 and (c2_y - c1_y) > 80.0:
            return False, iou, gap
        if area1 <= 0.0 or area2 <= 0.0 or (max(area1, area2) / min(area1, area2)) > 2.2:
            return False, iou, gap
        if bool(t1.best_plate_text) != bool(t2.best_plate_text) and iou < 0.10:
            return False, iou, gap

        box_diag = min(math.hypot(max(0.0, b1[2] - b1[0]), max(0.0, b1[3] - b1[1])),
                       math.hypot(max(0.0, b2[2] - b2[0]), max(0.0, b2[3] - b2[1])))
        if box_diag > 0 and math.hypot(c1_x - c2_x, c1_y - c2_y) <= 0.75 * box_diag:
            return True, iou, gap

        return False, iou, gap

    def merge_tracks(self, survivor: TrackState, absorbed: TrackState) -> None:
        """Merges absorbed track metadata & best snapshot into survivor without cross-track voting."""
        for tid in absorbed.merged_track_ids:
            if tid not in survivor.merged_track_ids:
                survivor.merged_track_ids.append(tid)

        if absorbed.first_frame_idx < survivor.first_frame_idx and absorbed.first_raw_box is not None:
            survivor.first_raw_box = list(absorbed.first_raw_box)
        if absorbed.last_frame_idx > survivor.last_frame_idx and absorbed.last_raw_box is not None:
            survivor.last_raw_box = list(absorbed.last_raw_box)

        survivor.first_frame_idx = min(survivor.first_frame_idx, absorbed.first_frame_idx)
        survivor.last_frame_idx = max(survivor.last_frame_idx, absorbed.last_frame_idx)
        survivor.first_frame, survivor.last_frame = survivor.first_frame_idx, survivor.last_frame_idx
        survivor.frames_tracked += absorbed.frames_tracked
        survivor.has_complete_frame = survivor.has_complete_frame or absorbed.has_complete_frame
        survivor.max_vehicle_area = max(survivor.max_vehicle_area, absorbed.max_vehicle_area)

        for f, b in absorbed.frame_boxes.items():
            survivor.frame_boxes.setdefault(f, b)

        if absorbed.best_plate_crop is not None and (
            survivor.best_plate_crop is None or (absorbed.best_plate_conf or 0.0) > (survivor.best_plate_conf or 0.0)
        ):
            survivor.best_plate_crop = absorbed.best_plate_crop

        if absorbed.best_plate_det_conf is not None and (
            survivor.best_plate_det_conf is None or float(absorbed.best_plate_det_conf) > float(survivor.best_plate_det_conf)
        ):
            survivor.best_plate_det_conf = float(absorbed.best_plate_det_conf)

        if absorbed.last_plate_box_frame is not None and (
            survivor.last_plate_box_frame is None or absorbed.last_plate_box_frame_idx > survivor.last_plate_box_frame_idx
        ):
            survivor.last_plate_box_frame = absorbed.last_plate_box_frame
            survivor.last_plate_box_frame_idx = absorbed.last_plate_box_frame_idx
            if survivor.plate_rel_box is None and absorbed.plate_rel_box is not None:
                survivor.plate_rel_box = absorbed.plate_rel_box
            if absorbed.plate_anchor is not None:
                surv_unclipped = bool(survivor.plate_anchor and survivor.plate_anchor.get("unclipped"))
                if survivor.plate_anchor is None or (absorbed.plate_anchor.get("unclipped") and not surv_unclipped):
                    survivor.plate_anchor = absorbed.plate_anchor

        # Respect plate lock without cross-track re-voting
        if not (survivor.plate_confident_reached and survivor.best_plate_text):
            if absorbed.plate_confident_reached and absorbed.best_plate_text:
                survivor.best_plate_text = absorbed.best_plate_text
                survivor.best_plate_conf = absorbed.best_plate_conf
                survivor.best_plate_crop = absorbed.best_plate_crop
                survivor.best_plate_det_conf = absorbed.best_plate_det_conf
                survivor.plate_confident_reached = True
            elif absorbed.best_plate_text and (
                not survivor.best_plate_text or (absorbed.best_plate_conf or 0.0) > (survivor.best_plate_conf or 0.0)
            ):
                survivor.best_plate_text = absorbed.best_plate_text
                survivor.best_plate_conf = absorbed.best_plate_conf
                survivor.best_plate_crop = absorbed.best_plate_crop
                survivor.best_plate_det_conf = absorbed.best_plate_det_conf

        if absorbed.plate_confident_reached:
            survivor.plate_confident_reached = True

        # Merge vehicle type votes (conf-weighted)
        for vtype, vote in absorbed.class_votes.items():
            survivor.class_votes[vtype] = survivor.class_votes.get(vtype, 0.0) + vote
        if survivor.class_votes:
            survivor.vehicle_type = max(survivor.class_votes, key=survivor.class_votes.get)

        comb_sharp = max(survivor.max_sharpness, absorbed.max_sharpness)
        survivor.max_sharpness = comb_sharp
        score_surv = survivor.best_unnorm_score / max(1e-5, comb_sharp)
        score_abs = absorbed.best_unnorm_score / max(1e-5, comb_sharp)
        if score_abs > score_surv and absorbed.best_full_frame is not None:
            survivor.best_unnorm_score = absorbed.best_unnorm_score
            survivor.best_snapshot_score = score_abs
            survivor.best_full_frame = absorbed.best_full_frame
            survivor.best_raw_box = absorbed.best_raw_box
            survivor.best_vehicle_box = absorbed.best_vehicle_box
            survivor.best_vehicle_crop = absorbed.best_vehicle_crop
            survivor.best_snapshot_frame_idx = absorbed.best_snapshot_frame_idx
            survivor.best_snapshot_complete = absorbed.best_snapshot_complete
            survivor.best_snapshot_sharp = absorbed.best_snapshot_sharp
        else:
            survivor.best_snapshot_score = score_surv

        absorbed.stitched_into = survivor.track_id
        absorbed.has_event_emitted = True

    def try_stitch_track(self, track: TrackState) -> Tuple[TrackState, Optional[Tuple[float, int]]]:
        """Attempts to stitch track with a matching open track."""
        for other_id, other in list(self.tracks.items()):
            if other_id == track.track_id or other.stitched_into is not None or other.has_event_emitted:
                continue
            if other_id in track.merged_track_ids:
                continue

            can_stitch, iou, gap = self.evaluate_stitching(track, other)
            if can_stitch:
                if other.disappeared_count < self.config.max_disappeared_frames:
                    self.merge_tracks(survivor=other, absorbed=track)
                    return other, (iou, gap)
                self.merge_tracks(survivor=track, absorbed=other)
                return track, (iou, gap)
        return track, None

    def finalize_track_plate_reads(self, track: TrackState) -> None:
        """Flushes debug plate reads and top crops only when --debug-outputs is enabled."""
        if not getattr(self.config, "debug_outputs", False) or not self.config.collect_plate_data or track.plate_reads_flushed:
            track.plate_reads_flushed = True
            return
        if not track.all_plate_reads:
            track.plate_reads_flushed = True
            return

        saved = 0
        for r in sorted(track.all_plate_reads, key=lambda x: x.get("score", 0.0), reverse=True):
            crop = r.get("crop")
            if saved < 5 and crop is not None and crop.size > 0:
                prefix = track.emitted_event_id or f"T{track.track_id}"
                name = f"crop_{prefix}_f{r['frame_idx']}_{saved + 1}.png"
                self.snapshot_saver.submit(self.config.plate_crops_dir / name, crop)
                r["crop_file_name"] = name
                saved += 1
            r["crop"] = None

        self.config.plate_reads_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.plate_reads_jsonl_path, "a", encoding="utf-8") as f:
            for r in track.all_plate_reads:
                f.write(json.dumps({
                    "event_id": track.emitted_event_id,
                    "track_id": track.track_id,
                    "frame_idx": r["frame_idx"],
                    "raw_text": r["raw_text"],
                    "per_character_confidences": r["per_character_confidences"],
                    "plate_w": r["plate_w"],
                    "plate_h": r["plate_h"],
                    "crop_file_name": r.get("crop_file_name"),
                }) + "\n")
        track.plate_reads_flushed = True

    def check_and_emit_events(self, current_frame_idx: int, force_all: bool = False) -> List[dict]:
        """Evaluates ended tracks (or all tracks on force_all), enforces completeness, and emits events."""
        emitted_now: List[dict] = []
        now_ts = datetime.now().timestamp()
        self.commit_frame_snapshots()

        for t_id in list(self.tracks.keys()):
            track = self.tracks.get(t_id)
            if track is None or track.has_event_emitted or (t_id in self.emitted_track_ids) or track.stitched_into is not None:
                continue

            if not ((track.disappeared_count >= self.config.max_disappeared_frames) or force_all):
                continue

            plate_str = track.best_plate_text or "NO_PLATE"
            if track.frames_tracked < self.config.min_track_frames:
                self.total_filtered_too_small += 1
                self.total_tracks_dropped += 1
                print(
                    f"[TRACK] id={t_id} frames={track.first_frame}-{track.last_frame} "
                    f"plate={plate_str} -> DROPPED (frames_tracked={track.frames_tracked} < min_track_frames={self.config.min_track_frames})"
                )
                self.finalize_track_plate_reads(track)
                del self.tracks[t_id]
                continue

            if track.max_vehicle_area < self.config.min_readable_vehicle_area:
                self.total_filtered_too_small += 1
                self.total_tracks_dropped += 1
                print(
                    f"[TRACK] id={t_id} frames={track.first_frame}-{track.last_frame} "
                    f"plate={plate_str} -> DROPPED (max_area={track.max_vehicle_area:.0f} < min_readable_vehicle_area={self.config.min_readable_vehicle_area})"
                )
                self.finalize_track_plate_reads(track)
                del self.tracks[t_id]
                continue

            survivor, match_info = self.try_stitch_track(track)
            if survivor.track_id != track.track_id:
                iou, gap = match_info if match_info is not None else (0.0, 0)
                print(
                    f"[TRACK] id={t_id} frames={track.first_frame}-{track.last_frame} "
                    f"plate={plate_str} -> STITCHED into track {survivor.track_id} (iou={iou:.2f}, gap={gap})"
                )
                self.total_tracks_stitched += 1
                continue

            track.has_event_emitted = True
            self.emitted_track_ids.update(track.merged_track_ids)

            event = self._emit_event(track, now_ts)
            if event is not None:
                emitted_now.append(event)
                self.total_tracks_emitted += 1
                ev_plate = event.get("number_plate") or "NO_PLATE"
                print(
                    f"[TRACK] id={t_id} frames={track.first_frame}-{track.last_frame} "
                    f"plate={ev_plate} -> EVENT {event['event_id']}"
                )
            else:
                self.total_tracks_dropped += 1
                print(
                    f"[TRACK] id={t_id} frames={track.first_frame}-{track.last_frame} "
                    f"plate={plate_str} -> DROPPED (duplicate plate candidate suppressed)"
                )

        if force_all:
            for track in list(self.tracks.values()):
                self.finalize_track_plate_reads(track)

        return emitted_now

    def _validate_and_finalize_plate(self, track: TrackState) -> bool:
        """Applies partial/regex plate policy and returns whether a plate string is kept."""
        if track.best_plate_text is None or track.best_plate_conf is None:
            return False

        candidate = track.best_plate_text
        if sum(1 for c in candidate if c != "?") < self.config.min_known_chars:
            track.best_plate_text = None
            track.best_plate_conf = None
            track.best_plate_det_conf = None
            track.best_plate_crop = None
            return False

        if self.config.plate_format == "UK":
            pat = self.config.plate_regex_pattern or r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$"
            is_valid = bool(re.match(pat, candidate)) and len(candidate) >= self.config.min_plate_length
        elif self.config.plate_format == "PAKISTAN":
            is_valid = (
                self.config.min_plate_length <= len(candidate) <= 10
                and (any(c.isalpha() for c in candidate) or any(c.isdigit() for c in candidate))
            )
            if self.config.enforce_regex and self.config.plate_regex_pattern:
                is_valid = is_valid and bool(re.match(self.config.plate_regex_pattern, candidate))
        else:
            is_valid = len(candidate) >= self.config.min_plate_length
            if self.config.enforce_regex and self.config.plate_regex_pattern:
                is_valid = is_valid and bool(re.match(self.config.plate_regex_pattern, candidate))

        if ("?" in candidate) or not is_valid:
            if self.config.partial_plate_policy == "drop":
                track.best_plate_text = None
                track.best_plate_conf = None
                track.best_plate_det_conf = None
                track.best_plate_crop = None
                return False
            track.best_plate_conf = min(0.60, float(track.best_plate_conf or 0.60))

        return True

    def _emit_event(self, track: TrackState, current_timestamp: float) -> Optional[dict]:
        """Formats event JSON, queues snapshot writes, writes events.json atomically, and publishes to EventBus."""
        now = datetime.now()
        has_plate = self._validate_and_finalize_plate(track)

        if has_plate and track.best_plate_text:
            for existing_plate, (last_ts, _) in list(self.recent_emitted_plates.items()):
                if (current_timestamp - last_ts) <= self.config.duplicate_plate_window_sec:
                    if is_fuzzy_duplicate_plate(
                        track.best_plate_text, existing_plate, max_distance=self.config.fuzzy_plate_max_distance
                    ):
                        self.finalize_track_plate_reads(track)
                        return None

        sec_key = now.strftime("%y%m%d%H%M%S")
        self._sec_seq_counter = (self._sec_seq_counter + 1) if sec_key == self._last_sec_key else 0
        self._last_sec_key = sec_key

        self.event_counter += 1
        event_id = generate_infinite_event_id(self.config.node_id, self.config.camera_id, now, self._sec_seq_counter)
        track.emitted_event_id = event_id
        if has_plate and track.best_plate_text:
            self.recent_emitted_plates[track.best_plate_text] = (current_timestamp, event_id)

        self.finalize_track_plate_reads(track)

        # Save tight clean vehicle snapshot (6% padding)
        vehicle_filename = f"vehicle_{event_id}.png"
        vehicle_save_path = self.config.snapshots_dir / vehicle_filename
        crop_w, crop_h, raw_box_display = 0, 0, None

        if track.best_full_frame is not None and track.best_raw_box is not None:
            hf, wf = track.best_full_frame.shape[:2]
            rx1, ry1, rx2, ry2 = track.best_raw_box
            px, py = int(max(1.0, rx2 - rx1) * 0.06), int(max(1.0, ry2 - ry1) * 0.06)
            crop = track.best_full_frame[max(0, int(ry1 - py)):min(hf, int(ry2 + py)),
                                         max(0, int(rx1 - px)):min(wf, int(rx2 + px))].copy()
            crop_h, crop_w = crop.shape[:2]
            raw_box_display = [int(v) for v in track.best_raw_box]
            self.snapshot_saver.submit(vehicle_save_path, crop)
        elif track.best_vehicle_crop is not None and track.best_vehicle_crop.size > 0:
            crop_h, crop_w = track.best_vehicle_crop.shape[:2]
            raw_box_display = [int(v) for v in track.best_raw_box] if track.best_raw_box else None
            self.snapshot_saver.submit(vehicle_save_path, track.best_vehicle_crop)
        else:
            crop_w = crop_h = 100
            self.snapshot_saver.submit(vehicle_save_path, np.zeros((100, 100, 3), dtype=np.uint8))

        if getattr(self.config, "debug_outputs", False) and track.best_full_frame is not None and track.best_raw_box:
            ctx = track.best_full_frame.copy()
            vx1, vy1, vx2, vy2 = [int(v) for v in track.best_raw_box]
            cv2.rectangle(ctx, (vx1, vy1), (vx2, vy2), (0, 255, 0) if has_plate else (0, 140, 255), 3)
            self.snapshot_saver.submit(self.config.snapshots_dir / f"context_{event_id}.png", ctx)

        plate_filename: Optional[str] = None
        if has_plate and track.best_plate_crop is not None and track.best_plate_crop.size > 0:
            plate_filename = f"plate_{event_id}.png"
            self.snapshot_saver.submit(self.config.snapshots_dir / plate_filename, track.best_plate_crop)

        event_data = {
            "event_id": event_id,
            "camera_id": self.config.camera_id,
            "time": now.strftime("%H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
            "snapshot_number_plate": plate_filename if has_plate else None,
            "snapshot_vehicle": vehicle_filename,
            "plate_detection_confidence": (
                round(float(track.best_plate_det_conf), 2)
                if has_plate and track.best_plate_det_conf is not None
                else None
            ),
            "number_plate": track.best_plate_text if has_plate else None,
            "number_plate_confidence": (
                round(float(track.best_plate_conf), 2)
                if has_plate and track.best_plate_conf is not None
                else None
            ),
            "vehicle_type": track.vehicle_type or "car",  # Always present, default to "car"
        }

        if self.config.include_remarks:
            if has_plate:
                event_data["remarks"] = (
                    "Number plate successfully detected and recognized"
                    if (track.best_plate_conf or 0.0) >= self.config.min_confident_plate_conf
                    else "Low confidence / faint characters detected on plate"
                )
            else:
                event_data["remarks"] = "No number plate detected on vehicle"

        self.emitted_events_meta.append({
            "event_id": event_id,
            "merged_track_ids": sorted(set(track.merged_track_ids)),
            "snapshot_frame_idx": track.best_snapshot_frame_idx,
            "is_complete": track.best_snapshot_complete,
            "plate_text": track.best_plate_text if has_plate else None,
            "plate_conf": round(float(track.best_plate_conf), 2) if (has_plate and track.best_plate_conf) else None,
            "snapshot_size": f"{crop_w}x{crop_h}",
            "raw_box": raw_box_display,
        })

        self.total_unique_vehicles += 1
        if has_plate:
            self.total_with_plate += 1
        else:
            self.total_without_plate += 1

        self.emitted_events.append(event_data)

        with open(self.config.events_jsonl_path, "a", encoding="utf-8") as f_jsonl:
            f_jsonl.write(json.dumps(event_data) + "\n")

        temp_json = self.config.events_json_path.with_suffix(".tmp")
        with open(temp_json, "w", encoding="utf-8") as f_json:
            json.dump(self.emitted_events, f_json, indent=2)
        os.replace(temp_json, self.config.events_json_path)

        if self.event_bus is not None:
            try:
                self.event_bus.publish(event_data)
            except Exception as e:
                print(f"[EVENT BUS] Publish error: {e}")

        if self.config.print_events_to_terminal:
            print(f"\n[EVENT RECORDED] {json.dumps(event_data, indent=2)}")

        return event_data

    def get_counts(self) -> Tuple[int, int, int, int, int]:
        """Returns (total_unique, with_plate, without_plate, active, filtered_too_small)."""
        confirmed_active = [
            t for t in self.tracks.values()
            if t.stitched_into is None
            and not t.has_event_emitted
            and t.entered_readable_zone
            and t.frames_tracked >= self.config.min_track_frames
            and (t.has_complete_frame or t.max_vehicle_area >= self.config.min_readable_vehicle_area * 2.0)
        ]
        total_u = len(self.emitted_events) + len(confirmed_active)
        total_p = sum(1 for e in self.emitted_events if e.get("number_plate")) + sum(
            1 for t in confirmed_active if t.best_plate_text
        )
        active = sum(1 for t in self.tracks.values() if t.disappeared_count == 0 and t.stitched_into is None)
        return total_u, total_p, max(0, total_u - total_p), active, self.total_filtered_too_small
