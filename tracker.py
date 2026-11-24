"""
Track state management, box deduplication, plate voting, and single-event emission.
Phase 1 Enterprise Enhancements:
1. Stage 2 (Analyzer) Event Interface & Queue.
2. Confidence-squared ($conf^2$) weighted character voting with UK syntax correction.
3. Fast Async Snapshot Saver (Level 1 lossless PNG compression for 4x disk speed).
4. Spatial Jump & ID Switch Guard (prevents identity bleed).
5. Clean console output (events written to JSON without terminal spam).
"""

import json
import math
import os
import queue
import random
import re
import string
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from config import PipelineConfig


class Stage2EventInterface(ABC):
    """
    Abstract interface for downstream Stage 2 Analyzers
    (Vehicle Make/Model/Color, Speed estimation, Driver face, Blacklist lookup).
    """

    @abstractmethod
    def on_event(
        self,
        event_id: str,
        camera_id: str,
        number_plate: Optional[str],
        vehicle_snapshot_path: Path,
        plate_snapshot_path: Optional[Path],
        event_data: dict,
    ) -> None:
        """Called automatically whenever a new verified vehicle event is finalized."""
        pass


class AsyncSnapshotSaver:
    """Non-blocking background worker for saving snapshots with fast lossless compression."""

    def __init__(self):
        self.queue: queue.Queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def submit(self, path: Path, image: np.ndarray) -> None:
        if self.running and image is not None and image.size > 0:
            self.queue.put((str(path), image))

    def _worker(self) -> None:
        # Fast PNG compression parameters (Level 1: 4x faster, 100% lossless)
        png_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]

        while self.running or not self.queue.empty():
            try:
                item = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            path_str, img = item
            try:
                cv2.imwrite(path_str, img, png_params)
            except Exception as e:
                print(f"[SNAPSHOT SAVE ERROR] Could not save {path_str}: {e}")
            finally:
                self.queue.task_done()

    def stop(self) -> None:
        self.running = False
        self.queue.join()


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes Levenshtein edit distance between two strings in pure Python."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


BASE36_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def int_to_base36_2char(n: int) -> str:
    """Encodes integer to 2-character base36 string ('00', '01' ... '0Z', '10' ... 'ZZ')."""
    n = n % 1296
    d1 = n // 36
    d2 = n % 36
    return BASE36_ALPHABET[d1] + BASE36_ALPHABET[d2]


def format_camera_number_code(camera_id_str: str) -> str:
    """
    Extracts camera number from camera_id string:
    - 2-digit zero-padded for 0-99 (e.g. 'CAM-00007' -> '07').
    - 3-digit for > 99 (e.g. 'CAM-00105' -> '105').
    """
    match = re.search(r"\d+", str(camera_id_str))
    if match:
        cam_int = int(match.group(0))
        return f"{cam_int:03d}" if cam_int > 99 else f"{cam_int:02d}"
    return "01"


def generate_infinite_event_id(
    node_id: str,
    camera_id_str: str,
    now: datetime,
    seq_idx: int = 0,
) -> str:
    """
    Generates structured enterprise event ID:
      {NODE}{CAM}-{YYMMDD}{HHMMSS}-{SEQ}
    Example: A07-261122054312-00
    """
    cam_str = format_camera_number_code(camera_id_str)
    timestamp_str = now.strftime("%y%m%d%H%M%S")
    seq_str = int_to_base36_2char(seq_idx)
    return f"{node_id}{cam_str}-{timestamp_str}-{seq_str}"


def test_event_id_uniqueness() -> bool:
    """Verifies that 10,000 generated event IDs have zero collisions."""
    seen_ids = set()
    from datetime import timedelta
    base_time = datetime(2026, 11, 22, 5, 43, 12)
    for i in range(10000):
        sec_offset = i // 1296
        seq_idx = i % 1296
        t = base_time + timedelta(seconds=sec_offset)
        eid = generate_infinite_event_id(node_id="A", camera_id_str="CAM-00007", now=t, seq_idx=seq_idx)
        assert eid not in seen_ids, f"Collision detected for event ID: {eid}"
        seen_ids.add(eid)
    assert len(seen_ids) == 10000
    return True


# Run verification check immediately
assert test_event_id_uniqueness() is True


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
    """
    Suppresses duplicate / nested bounding boxes where one box is substantially
    contained within another (e.g. car detected twice or sub-part detected as separate box).
    """
    if len(boxes) <= 1:
        return boxes, scores

    keep_indices: Set[int] = set(range(len(boxes)))

    for i in range(len(boxes)):
        if i not in keep_indices:
            continue
        ax1, ay1, ax2, ay2 = boxes[i]
        area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
        if area_a <= 0:
            keep_indices.discard(i)
            continue

        for j in range(len(boxes)):
            if i == j or j not in keep_indices:
                continue
            bx1, by1, bx2, by2 = boxes[j]
            area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
            if area_b <= 0:
                continue

            # Compute intersection area
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            intersection = iw * ih

            containment_a = intersection / area_a
            containment_b = intersection / area_b

            if containment_a >= containment_thresh or containment_b >= containment_thresh:
                if scores[i] < scores[j]:
                    keep_indices.discard(i)
                    break
                else:
                    keep_indices.discard(j)

    kept_boxes = [boxes[k] for k in sorted(keep_indices)]
    kept_scores = [scores[k] for k in sorted(keep_indices)]
    return kept_boxes, kept_scores


def normalize_pakistan_plate_syntax(plate: str) -> str:
    """
    Normalizes standard Pakistani license plate formats (e.g. LEA-18-5001, ICT-VS-221, MNP-123, FD-1234).
    Resolves optical character confusion between digits and letters based on Pakistani prefix/suffix layout:
    - Leading 2 to 4 characters are alphabetic jurisdiction codes (e.g. LEA, ICT, LHR, KHI, FD, MNP, ISB).
    - Trailing characters are predominantly digits.
    """
    clean = re.sub(r"[^A-Z0-9]", "", plate.upper())
    if len(clean) < 3:
        return clean

    digit_to_letter = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G"}
    letter_to_digit = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8"}

    chars = list(clean)

    # Pakistani prefixes: check if first 2-3 characters look like digits mistaken for letters
    for i in range(min(2, len(chars))):
        if chars[i] in digit_to_letter and any(c.isalpha() for c in chars[:3]):
            chars[i] = digit_to_letter[chars[i]]

    # Trailing part: if the tail (last 2-4 characters) is numeric, fix letter confusions
    for i in range(max(2, len(chars) - 4), len(chars)):
        if chars[i] in letter_to_digit and any(c.isdigit() for c in chars[-3:]):
            chars[i] = letter_to_digit[chars[i]]

    return "".join(chars)


def normalize_uk_plate_syntax(plate: str) -> str:
    """
    Normalizes standard UK plate format (2 letters, 2 digits, 3 letters e.g. AB12CDE).
    Resolves optical character confusion (0 vs O, 1 vs I, 5 vs S, 8 vs B).
    """
    if len(plate) != 7:
        return plate

    digit_to_letter = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G"}
    letter_to_digit = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8"}

    chars = list(plate)
    # Positions 0 and 1: Must be letters
    for i in (0, 1):
        if chars[i] in digit_to_letter:
            chars[i] = digit_to_letter[chars[i]]

    # Positions 2 and 3: Must be digits
    for i in (2, 3):
        if chars[i] in letter_to_digit:
            chars[i] = letter_to_digit[chars[i]]

    # Positions 4, 5, 6: Must be letters
    for i in (4, 5, 6):
        if chars[i] in digit_to_letter:
            chars[i] = digit_to_letter[chars[i]]

    return "".join(chars)


def is_fuzzy_duplicate_plate(p1: str, p2: str, max_distance: int = 2) -> bool:
    """
    Compares two plates ignoring '?' positions.
    Only suppresses if BOTH plates have >= 5 known (non-'?') characters.
    """
    known1 = [c for c in p1 if c != "?"]
    known2 = [c for c in p2 if c != "?"]
    if len(known1) < 5 or len(known2) < 5:
        return False

    # If same length, compare position-by-position ignoring '?' positions
    if len(p1) == len(p2):
        mismatches = 0
        for c1, c2 in zip(p1, p2):
            if c1 != "?" and c2 != "?" and c1 != c2:
                mismatches += 1
                if mismatches > max_distance:
                    return False
        return True

    # If length difference exceeds max_distance, cannot match
    if abs(len(p1) - len(p2)) > max_distance:
        return False

    # Levenshtein distance where '?' has zero substitution/match cost
    dp = [[0] * (len(p2) + 1) for _ in range(len(p1) + 1)]
    for i in range(len(p1) + 1):
        dp[i][0] = i
    for j in range(len(p2) + 1):
        dp[0][j] = j

    for i in range(1, len(p1) + 1):
        for j in range(1, len(p2) + 1):
            c1, c2 = p1[i - 1], p2[j - 1]
            cost = 0 if (c1 == "?" or c2 == "?" or c1 == c2) else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # match/sub
            )

    return dp[len(p1)][len(p2)] <= max_distance


def plates_match_differ_by_one(p1: Optional[str], p2: Optional[str]) -> bool:
    """
    Returns True if plates are identical or differ by <= 1 character,
    ignoring '?' positions. Both must have >= 3 known characters. (Requirement A3)
    """
    if not p1 or not p2:
        return False
    k1 = [c for c in p1 if c != "?"]
    k2 = [c for c in p2 if c != "?"]
    if len(k1) < 3 or len(k2) < 3:
        return False

    if len(p1) == len(p2):
        mismatches = sum(1 for c1, c2 in zip(p1, p2) if c1 != "?" and c2 != "?" and c1 != c2)
        return mismatches <= 1

    if abs(len(p1) - len(p2)) > 1:
        return False

    dp = [[0] * (len(p2) + 1) for _ in range(len(p1) + 1)]
    for i in range(len(p1) + 1):
        dp[i][0] = i
    for j in range(len(p2) + 1):
        dp[0][j] = j
    for i in range(1, len(p1) + 1):
        for j in range(1, len(p2) + 1):
            c1, c2 = p1[i - 1], p2[j - 1]
            cost = 0 if (c1 == "?" or c2 == "?" or c1 == c2) else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(p1)][len(p2)] <= 1


def character_level_voting(
    readings: List[Tuple],
    min_length: int = 4,
    regex_pattern: Optional[str] = None,
    uk_syntax_correction: bool = False,
    plate_format: str = "PAKISTAN",
    syntax_correction: bool = False,
) -> Optional[Tuple[str, float]]:
    """
    Performs confidence-squared ($conf^2$) weighted character voting across multiple plate readings.
    High confidence reads dominate, eliminating flickering run-to-run discrepancies.
    Positions where the winning character's share of position weight < 0.5 or average
    per-character confidence < 0.5 are replaced with '?'.
    Final text may contain '?'. Length is decided by majority length vote.
    """
    if not readings:
        return None

    valid_readings: List[Tuple[str, float, List[float]]] = []
    regex = re.compile(regex_pattern) if regex_pattern else None

    for item in readings:
        if len(item) >= 3:
            text = str(item[0])
            conf = float(item[1])
            char_confs_in = [float(c) for c in item[2]] if item[2] else [conf] * len(text)
        else:
            text = str(item[0])
            conf = float(item[1])
            char_confs_in = [conf] * len(text)

        clean_chars = []
        clean_confs = []
        for ch, c_conf in zip(text.upper(), char_confs_in):
            if ch.isalnum():
                clean_chars.append(ch)
                clean_confs.append(float(c_conf))

        clean_text = "".join(clean_chars)
        if plate_format == "UK" and uk_syntax_correction and len(clean_text) == 7:
            clean_text = normalize_uk_plate_syntax(clean_text)
        elif plate_format == "PAKISTAN":
            if syntax_correction:
                clean_text = normalize_pakistan_plate_syntax(clean_text)

        # Allow partial reads as long as they have at least 3 characters
        if len(clean_text) < 3:
            continue

        valid_readings.append((clean_text, conf, clean_confs))

    if not valid_readings:
        return None

    # Group by length to find majority plate length
    length_votes = defaultdict(float)
    for text, conf, _ in valid_readings:
        length_votes[len(text)] += (conf ** 4)

    best_len = max(length_votes.keys(), key=lambda l: length_votes[l])
    same_len_readings = [r for r in valid_readings if len(r[0]) == best_len]

    consensus_chars = []
    char_confs = []

    for pos in range(best_len):
        pos_votes = defaultdict(float)
        # Extra priority for leading area code letters (pos 0 and 1)
        pos_multiplier = 1.35 if pos in (0, 1) else 1.0

        for text, conf, p_confs in same_len_readings:
            char = text[pos]
            weight = (conf ** 4) * pos_multiplier
            pos_votes[char] += weight

        total_weight = sum(pos_votes.values())
        best_char = max(pos_votes.keys(), key=lambda c: pos_votes[c])
        winning_weight = pos_votes[best_char]

        # 1. Winning character's share of position weight
        share = winning_weight / max(1e-5, total_weight)

        # 2. Average per-character confidence of winning character at this position
        matching_char_confs = [
            (p_confs[pos] if pos < len(p_confs) else conf)
            for text, conf, p_confs in same_len_readings
            if text[pos] == best_char
        ]
        avg_char_conf = float(np.mean(matching_char_confs)) if matching_char_confs else 0.0

        # Replace position with '?' when share < 0.5 or average per-character confidence < 0.5
        if share < 0.5 or avg_char_conf < 0.5:
            final_char = "?"
            char_confs.append(avg_char_conf)
        else:
            final_char = best_char
            char_confs.append(avg_char_conf)

        consensus_chars.append(final_char)

    raw_consensus = "".join(consensus_chars)
    if plate_format == "UK" and uk_syntax_correction and len(raw_consensus) == 7 and "?" not in raw_consensus:
        raw_consensus = normalize_uk_plate_syntax(raw_consensus)
    elif plate_format == "PAKISTAN":
        if syntax_correction and "?" not in raw_consensus:
            raw_consensus = normalize_pakistan_plate_syntax(raw_consensus)

    known_confs = [c for ch, c in zip(consensus_chars, char_confs) if ch != "?"]
    avg_confidence = float(np.mean(known_confs)) if known_confs else (float(np.mean(char_confs)) if char_confs else float(same_len_readings[0][1]))
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
    # Smoothed bounding box [x1, y1, x2, y2]
    smoothed_box: Optional[List[float]] = None
    hud_smoothed_box: Optional[List[float]] = None
    raw_box: Optional[List[float]] = None
    first_raw_box: Optional[List[float]] = None
    last_raw_box: Optional[List[float]] = None
    frame_boxes: Dict[int, List[float]] = field(default_factory=dict)

    def __post_init__(self):
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
    # Previous box center for spatial jump tracking
    last_center: Optional[Tuple[float, float]] = None
    # Best vehicle snapshot data
    best_raw_box: Optional[List[float]] = None
    best_vehicle_box: Optional[List[int]] = None
    best_full_frame: Optional[np.ndarray] = None
    best_vehicle_crop: Optional[np.ndarray] = None
    best_vehicle_area: float = 0.0
    # Snapshot scoring metrics (Requirement C)
    max_sharpness: float = 0.0
    best_unnorm_score: float = -1.0
    best_snapshot_score: float = -1.0
    best_snapshot_frame_idx: int = -1
    best_snapshot_complete: bool = False
    best_snapshot_sharp: float = 0.0
    has_complete_frame: bool = False
    max_vehicle_area: float = 0.0
    current_frame_candidate: Optional[dict] = None
    # Track stitching (Requirement A)
    merged_track_ids: List[int] = field(default_factory=list)
    stitched_into: Optional[int] = None
    # Readable Zone Gate
    entered_readable_zone: bool = False
    # Plate readings history: list of (text, mean_conf, per_char_confs, plate_w, plate_h, frame_idx)
    plate_readings: List[Tuple] = field(default_factory=list)
    # OCR Data collection for fine-tuning
    all_plate_reads: List[dict] = field(default_factory=list)
    emitted_event_id: Optional[str] = None
    plate_reads_flushed: bool = False
    # Best plate snapshot & OCR data
    best_plate_crop: Optional[np.ndarray] = None
    best_plate_text: Optional[str] = None
    best_plate_conf: Optional[float] = None
    best_plate_det_conf: Optional[float] = None
    # Lifecycle flags
    has_event_emitted: bool = False
    plate_confident_reached: bool = False
    has_unclipped_high_conf_plate: bool = False
    # Plate box visualization in video (Part B)
    last_plate_box_frame: Optional[List[float]] = None
    last_plate_box_frame_idx: int = -1
    plate_rel_box: Optional[List[float]] = None
    plate_anchor: Optional[dict] = None
    kalman_box: Optional[List[float]] = None


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

        # Async background snapshot saver
        self.snapshot_saver = AsyncSnapshotSaver()

        # Stage 2 (Analyzer) downstream interface queue & handlers
        self.stage2_queue: queue.Queue = queue.Queue(maxsize=self.config.stage2_queue_maxsize)
        self.stage2_handlers: List[Stage2EventInterface] = []

        # Stats counters
        self.total_unique_vehicles: int = 0
        self.total_with_plate: int = 0
        self.total_without_plate: int = 0
        self.total_filtered_too_small: int = 0
        self.total_tracks_seen: int = 0
        self.total_tracks_stitched: int = 0
        self.total_tracks_dropped: int = 0
        self.total_tracks_emitted: int = 0

        # Clean output jsonl files for this run
        if self.config.events_jsonl_path.exists():
            with open(self.config.events_jsonl_path, "w", encoding="utf-8") as _:
                pass
        if getattr(self.config, "debug_outputs", False) and self.config.collect_plate_data and self.config.plate_reads_jsonl_path.exists():
            with open(self.config.plate_reads_jsonl_path, "w", encoding="utf-8") as _:
                pass

    def register_stage2_handler(self, handler: Stage2EventInterface) -> None:
        """Registers a downstream Stage 2 analyzer callback."""
        self.stage2_handlers.append(handler)

    def is_in_roi(self, box: List[float], frame_shape: Tuple[int, int]) -> bool:
        """
        Determines if the vehicle box is inside the readable zone.
        Excludes the upper horizon (top 32% of frame) where vehicles are too far away.
        """
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = box
        bottom_cx = (x1 + x2) / 2.0
        bottom_cy = y2

        if self.config.roi_polygon is not None and len(self.config.roi_polygon) >= 3:
            pts = np.array(self.config.roi_polygon, dtype=np.int32)
            dist = cv2.pointPolygonTest(pts, (float(bottom_cx), float(bottom_cy)), measureDist=False)
            return dist >= 0

        horizon_y = self.config.roi_top_cutoff_ratio * h
        return (bottom_cy >= horizon_y) and (y1 >= horizon_y * 0.7)

    def update_track(
        self,
        track_id: int,
        box: List[float],
        frame: np.ndarray,
        frame_idx: int,
        raw_det_box: Optional[List[float]] = None,
    ) -> TrackState:
        """
        Updates an existing track or initializes a new track state.
        Detects spatial jumps (ID switches) and enforces the readable zone gate.
        Uses the RAW YOLO detection box for the vehicle snapshot crop and best-frame score (Requirement A1).
        Evaluates completeness: clipped on <= 1 side and clipped part < 15% (Requirement A4).
        """
        kalman_box = [float(v) for v in box]
        if raw_det_box is not None:
            raw_box = [float(v) for v in raw_det_box]
        else:
            raw_box = [float(v) for v in box]

        rx1, ry1, rx2, ry2 = raw_box
        h, w = frame.shape[:2]

        cx1 = max(0.0, min(float(w), rx1))
        cy1 = max(0.0, min(float(h), ry1))
        cx2 = max(0.0, min(float(w), rx2))
        cy2 = max(0.0, min(float(h), ry2))

        raw_bw = max(0.0, rx2 - rx1)
        raw_bh = max(0.0, ry2 - ry1)
        raw_area = float(raw_bw * raw_bh)

        inside_bw = max(0.0, cx2 - cx1)
        inside_bh = max(0.0, cy2 - cy1)
        inside_area = float(inside_bw * inside_bh)

        inside = inside_area / max(1e-5, raw_area)
        # Requirement A4: "complete" means: vehicle box is clipped by the frame on at most one side and the clipped part is < 15% of the box area.
        # Fixed 12 px margin rule removed.
        clipped_sides = int(rx1 < 0.0) + int(ry1 < 0.0) + int(rx2 > float(w)) + int(ry2 > float(h))
        clipped_part = max(0.0, raw_area - inside_area) / max(1e-5, raw_area)
        complete = (clipped_sides <= 1) and (clipped_part < 0.15)

        # Vehicle crop with 6% padding clamped to frame (clean, no boxes/labels)
        pad_x = int(raw_bw * 0.06)
        pad_y = int(raw_bh * 0.06)
        vx1 = max(0, int(rx1 - pad_x))
        vy1 = max(0, int(ry1 - pad_y))
        vx2 = min(w, int(rx2 + pad_x))
        vy2 = min(h, int(ry2 + pad_y))
        vehicle_crop = frame[vy1:vy2, vx1:vx2].copy() if (vx2 > vx1 and vy2 > vy1) else None

        raw_sharp = get_image_sharpness(vehicle_crop) if vehicle_crop is not None else 0.0
        area_px = inside_area
        score_multiplier = 1.0 if complete else 0.35
        base_score = area_px * raw_sharp * score_multiplier

        cx, cy = (rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0
        in_roi = self.is_in_roi([float(cx1), float(cy1), float(cx2), float(cy2)], (h, w))
        is_readable_size = (
            raw_bw >= self.config.min_vehicle_width
            and raw_bh >= self.config.min_vehicle_height
            and raw_area >= self.config.min_readable_vehicle_area
        )

        global_id = f"{self.config.camera_id}:{track_id}"

        if track_id not in self.tracks:
            self.total_tracks_seen += 1
            track = TrackState(
                track_id=track_id,
                global_id=global_id,
                first_frame_idx=frame_idx,
                last_frame_idx=frame_idx,
                first_frame=frame_idx,
                last_frame=frame_idx,
                frames_tracked=1,
                disappeared_count=0,
                smoothed_box=[rx1, ry1, rx2, ry2],
                hud_smoothed_box=[kalman_box[0], kalman_box[1], kalman_box[2], kalman_box[3]],
                kalman_box=kalman_box,
                raw_box=raw_box,
                first_raw_box=[float(v) for v in raw_box],
                last_raw_box=[float(v) for v in raw_box],
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
                entered_readable_zone=(in_roi and is_readable_size),
            )
            self.tracks[track_id] = track
        else:
            track = self.tracks[track_id]
            track.kalman_box = kalman_box

            # Spatial Jump Check: Never clear a locked confident plate, and ignore border-clipping center shifts
            if track.last_center is not None and track.frames_tracked > 2 and not track.plate_confident_reached:
                prev_cx, prev_cy = track.last_center
                jump_dist = math.hypot(cx - prev_cx, cy - prev_cy)
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
            track.last_raw_box = [float(v) for v in raw_box]
            if track.first_raw_box is None:
                track.first_raw_box = [float(v) for v in raw_box]
            if track.first_frame == 0 and track.first_frame_idx != 0:
                track.first_frame = track.first_frame_idx
            track.frame_boxes[frame_idx] = raw_box
            if complete:
                track.has_complete_frame = True
            track.max_vehicle_area = max(track.max_vehicle_area, raw_area)
            track.max_sharpness = max(track.max_sharpness, raw_sharp)

            if in_roi and is_readable_size:
                track.entered_readable_zone = True

            # Smoothed box with box_smoothing_factor (0.0 for raw box / zero lag)
            alpha = self.config.box_smoothing_factor
            if alpha == 0.0 or track.smoothed_box is None:
                track.smoothed_box = [rx1, ry1, rx2, ry2]
            else:
                track.smoothed_box = [
                    alpha * float(c) + (1.0 - alpha) * float(s)
                    for c, s in zip(raw_box, track.smoothed_box)
                ]

            # HUD smoothing for visualization display uses Kalman box for stability
            hud_alpha = getattr(self.config, "hud_box_smoothing_factor", 0.65)
            if track.hud_smoothed_box is None:
                track.hud_smoothed_box = [kalman_box[0], kalman_box[1], kalman_box[2], kalman_box[3]]
            else:
                track.hud_smoothed_box = [
                    hud_alpha * float(c) + (1.0 - hud_alpha) * float(s)
                    for c, s in zip(kalman_box, track.hud_smoothed_box)
                ]

        # Stage candidate for commit after ALPR evaluation in this frame
        # Only stage snapshot candidate if a real raw YOLO detection matched this frame (Requirement A1)
        if raw_det_box is not None:
            track.current_frame_candidate = {
                "frame_idx": frame_idx,
                "raw_box": raw_box,
                "full_frame": frame,
                "vehicle_crop": vehicle_crop,
                "raw_sharp": raw_sharp,
                "base_score": base_score,
                "complete": complete,
                "area_px": area_px,
                "plate_inside": False,
            }
        else:
            track.current_frame_candidate = None

        return track

    def commit_frame_snapshots(self) -> None:
        """
        Commits per-frame snapshot scores across all active tracks.
        Applies 1.5x plate bonus if a plate was detected inside the vehicle box (Requirement C).
        """
        for track in self.tracks.values():
            cand = track.current_frame_candidate
            if cand is None:
                continue

            bonus = 1.5 if cand.get("plate_inside", False) else 1.0
            unnorm_score = cand["base_score"] * bonus
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
        """Increments disappeared count for tracks absent from current frame."""
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
        """
        Directly associates a plate prediction to a track.
        Enforces plate mutability lock, consensus voting, and edge-cut rejection (Requirement A2).
        Records plate box in full-frame coordinates for HUD visualization (Part B).
        """
        track = self.tracks.get(track_id)
        if track is None or track.has_event_emitted:
            return

        # Requirement A2: Ignore edge-cut plate reads within 4 px of frame border on ANY side
        if plate_bbox_frame is not None:
            pfx1, pfy1, pfx2, pfy2 = plate_bbox_frame
            h_f, w_f = frame.shape[:2]
            if (pfx1 <= 4.0) or (pfy1 <= 4.0) or (pfx2 >= float(w_f) - 4.0) or (pfy2 >= float(h_f) - 4.0):
                if getattr(self.config, "debug_reads", False):
                    print(
                        f"[DEBUG TRACKER] Plate read rejected (reject_reason='plate_touches_frame_edge'): "
                        f"text='{raw_text}' box={[round(v, 1) for v in plate_bbox_frame]} frame={w_f}x{h_f}"
                    )
                return

        # Keep last plate box visible for visualization (Part B)
        if plate_bbox_frame is not None and not track.plate_confident_reached:
            h_f_vis, w_f_vis = frame.shape[:2]
            track.last_plate_box_frame = [float(v) for v in plate_bbox_frame]
            track.last_plate_box_frame_idx = frame_idx
            vbox = track.frame_boxes.get(frame_idx) or track.raw_box or track.smoothed_box
            if vbox is not None:
                vx1, vy1, vx2, vy2 = vbox
                vw = max(1.0, vx2 - vx1)
                vh = max(1.0, vy2 - vy1)
                cur_unclipped = (vy1 > 3.0) and (vy2 < float(h_f_vis) - 3.0)
                prev_anchor = getattr(track, "plate_anchor", None)
                prev_unclipped = bool(prev_anchor and prev_anchor.get("unclipped"))
                if (not prev_unclipped) or cur_unclipped:
                    px1, py1, px2, py2 = plate_bbox_frame
                    track.plate_rel_box = [
                        (px1 - vx1) / vw,
                        (py1 - vy1) / vh,
                        (px2 - vx1) / vw,
                        (py2 - vy1) / vh,
                    ]
                    track.plate_anchor = {
                        "plate_box": [float(px1), float(py1), float(px2), float(py2)],
                        "veh_box": [float(vx1), float(vy1), float(vx2), float(vy2)],
                        "clipped_top": bool(vy1 <= 3.0),
                        "clipped_bottom": bool(vy2 >= float(h_f_vis) - 3.0),
                        "unclipped": cur_unclipped,
                    }

        raw_str = str(raw_text).strip().upper()
        if per_char_confs is None or len(per_char_confs) != len(raw_str):
            per_char_confs = [ocr_conf] * len(raw_str)

        clean_chars = []
        clean_confs = []
        for ch, c_conf in zip(raw_str, per_char_confs):
            if ch.isalnum():
                clean_chars.append(ch)
                clean_confs.append(float(c_conf))

        clean_text = "".join(clean_chars)
        if self.config.plate_format == "UK" and self.config.uk_syntax_correction and len(clean_text) == 7:
            clean_text = normalize_uk_plate_syntax(clean_text)
        elif self.config.plate_format == "PAKISTAN":
            if self.config.plate_syntax_correction:
                clean_text = normalize_pakistan_plate_syntax(clean_text)

        min_len_req = min(self.config.min_plate_length, self.config.min_known_chars)
        if len(clean_text) < min_len_req:
            return

        # Plate Mutability Lock: Do not let a conflicting plate overwrite established text
        if track.plate_confident_reached and track.best_plate_text:
            return

        # Ignore degraded border-clipped vehicle reads when a crisp unclipped read already exists
        h_f_chk, _ = frame.shape[:2]
        vbox_chk = track.frame_boxes.get(frame_idx) or track.raw_box
        veh_clipped = bool(vbox_chk is not None and (vbox_chk[1] <= 5.0 or vbox_chk[3] >= float(h_f_chk) - 5.0))
        if track.has_unclipped_high_conf_plate and veh_clipped and ocr_conf < 0.85:
            return
        if (not veh_clipped) and ocr_conf >= 0.85 and len(clean_text) >= self.config.min_plate_length:
            track.has_unclipped_high_conf_plate = True

        sharpness = get_image_sharpness(plate_crop)
        if sharpness < self.config.min_plate_sharpness and track.best_plate_crop is not None:
            return

        actual_plate_w = plate_w if plate_w > 0 else (plate_crop.shape[1] if plate_crop is not None else 0)
        actual_plate_h = plate_h if plate_h > 0 else (plate_crop.shape[0] if plate_crop is not None else 0)

        # Store each read as (text, mean_conf, per_char_confs, plate_w, plate_h, frame_idx)
        read_entry = (clean_text, ocr_conf, clean_confs, actual_plate_w, actual_plate_h, frame_idx)
        track.plate_readings.append(read_entry)

        if self.config.collect_plate_data:
            score = float(ocr_conf) * float(actual_plate_w)
            track.all_plate_reads.append({
                "frame_idx": frame_idx,
                "raw_text": clean_text,
                "per_character_confidences": [round(float(c), 4) for c in clean_confs],
                "plate_w": int(actual_plate_w),
                "plate_h": int(actual_plate_h),
                "crop": plate_crop.copy() if plate_crop is not None else None,
                "score": score,
                "crop_file_name": None,
            })

        # Real maximum detector confidence across all detections of this vehicle
        if track.best_plate_det_conf is None or float(det_conf) > float(track.best_plate_det_conf):
            track.best_plate_det_conf = float(det_conf)

        is_better = (track.best_plate_conf is None) or (ocr_conf > track.best_plate_conf)
        if is_better:
            track.best_plate_crop = plate_crop.copy() if plate_crop is not None else None

        consensus_result = character_level_voting(
            readings=track.plate_readings,
            min_length=self.config.min_plate_length,
            regex_pattern=self.config.plate_regex_pattern if self.config.enforce_regex else None,
            uk_syntax_correction=self.config.uk_syntax_correction,
            plate_format=self.config.plate_format,
            syntax_correction=self.config.plate_syntax_correction,
        )

        if consensus_result is not None:
            voted_text, voted_conf = consensus_result
            # Plate lock: jab track.plate_confident_reached ho jaye, plate text aur confidence dobara overwrite na ho
            if not track.plate_confident_reached:
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
        """Matches full-frame ALPR plate detections to vehicle tracks."""
        h, w = frame.shape[:2]

        for alpr_res in alpr_results:
            det = alpr_res.detection
            ocr = alpr_res.ocr
            if ocr is None or not ocr.text:
                continue

            bbox = det.bounding_box
            px1, py1, px2, py2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
            pcx = (px1 + px2) / 2.0
            pcy = (py1 + py2) / 2.0

            best_match_id: Optional[int] = None
            smallest_enclosing_area = float("inf")

            for t_id, track in self.tracks.items():
                if track.disappeared_count > 0 or track.smoothed_box is None:
                    continue
                vx1, vy1, vx2, vy2 = track.smoothed_box
                if vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2:
                    v_area = (vx2 - vx1) * (vy2 - vy1)
                    if v_area < smallest_enclosing_area:
                        smallest_enclosing_area = v_area
                        best_match_id = t_id

            if best_match_id is not None:
                c_px1, c_py1 = max(0, int(px1)), max(0, int(py1))
                c_px2, c_py2 = min(w, int(px2)), min(h, int(py2))
                plate_crop = frame[c_py1:c_py2, c_px1:c_px2].copy()
                plate_w = max(1, c_px2 - c_px1)
                plate_h = max(1, c_py2 - c_py1)

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
                    track_id=best_match_id,
                    raw_text=raw_text,
                    ocr_conf=ocr_conf,
                    det_conf=float(det.confidence),
                    plate_crop=plate_crop,
                    frame=frame,
                    per_char_confs=per_char_confs,
                    plate_w=plate_w,
                    plate_h=plate_h,
                    frame_idx=frame_idx,
                    plate_bbox_frame=[float(px1), float(py1), float(px2), float(py2)],
                )

                # Check if plate is fully inside raw vehicle box for 1.5x score bonus
                matched_track = self.tracks.get(best_match_id)
                if matched_track and matched_track.current_frame_candidate:
                    rb = matched_track.raw_box
                    if rb is not None:
                        rx1, ry1, rx2, ry2 = rb
                        if (c_px1 >= rx1 - 5) and (c_py1 >= ry1 - 5) and (c_px2 <= rx2 + 5) and (c_py2 <= ry2 + 5):
                            matched_track.current_frame_candidate["plate_inside"] = True

    def _boxes_overlap_match(self, b1: Optional[List[float]], b2: Optional[List[float]]) -> bool:
        """Computes IoU and containment between two boxes."""
        if b1 is None or b2 is None:
            return False
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        iw = max(0.0, x2 - x1)
        ih = max(0.0, y2 - y1)
        inter = iw * ih
        area1 = max(0.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
        area2 = max(0.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
        if area1 <= 0 or area2 <= 0:
            return False
        union = area1 + area2 - inter
        iou = inter / union if union > 0 else 0.0
        containment = max(inter / area1, inter / area2)
        return (iou > 0.25) or (containment > 0.60)

    def evaluate_stitching(self, t1: TrackState, t2: TrackState) -> Tuple[bool, float, int]:
        """
        Evaluates whether two tracks represent the same vehicle.
        HARD VETO rules (if any is True, return False):
        1. Frame ranges overlap (dono aik hi waqt par zinda thay) -> False.
        2. Both have confident plate and plate texts differ (Levenshtein >= 2) -> False.
        3. Gap (t2.first_frame - t1.last_frame) > stitch_max_gap_frames (default 15) or gap < 0 -> False.

        After passing all vetoes:
        Match: Compare t1.last_raw_box with t2.first_raw_box ONLY (IoU >= 0.30 or centre distance <= 0.75 * box diagonal).
        NEVER compare with best_raw_box.
        """
        t1_first = t1.first_frame if t1.first_frame is not None else t1.first_frame_idx
        t1_last = t1.last_frame if t1.last_frame is not None else t1.last_frame_idx
        t2_first = t2.first_frame if t2.first_frame is not None else t2.first_frame_idx
        t2_last = t2.last_frame if t2.last_frame is not None else t2.last_frame_idx

        # Ensure chronological order (t1 earlier, t2 later)
        if t1_first > t2_first:
            t1, t2 = t2, t1
            t1_first, t2_first = t2_first, t1_first
            t1_last, t2_last = t2_last, t1_last

        # 1. Time overlap veto: t1 and t2 frame ranges overlap (both active at the same time) -> False
        if max(t1_first, t2_first) <= min(t1_last, t2_last):
            return False, 0.0, 0

        # 2. Confident plate mismatch veto: both have plates and text differs (Levenshtein >= 2) -> False
        if t1.best_plate_text and t2.best_plate_text:
            if levenshtein_distance(t1.best_plate_text, t2.best_plate_text) >= 2:
                return False, 0.0, 0

        # 3. Gap veto: (t2.first_frame - t1.last_frame) > stitch_max_gap_frames or gap < 0 -> False
        max_gap = getattr(self.config, "stitch_max_gap_frames", 15)
        gap = t2_first - t1_last
        if gap < 0 or gap > max_gap:
            return False, 0.0, gap

        # Match comparison: t1.last_raw_box with t2.first_raw_box ONLY (NEVER best_raw_box)
        b1 = t1.last_raw_box
        b2 = t2.first_raw_box
        if b1 is None or b2 is None:
            return False, 0.0, gap

        # IoU check (>= 0.30)
        ix1 = max(b1[0], b2[0])
        iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2])
        iy2 = min(b1[3], b2[3])
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih

        area1 = max(0.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
        area2 = max(0.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
        union = area1 + area2 - inter
        iou = (inter / union) if union > 0.0 else 0.0

        if iou >= 0.30:
            return True, iou, gap

        # Below iou >= 0.30, enforce strict spatial/size/plate sanity guards so distinct cars never merge
        c1_x = (b1[0] + b1[2]) / 2.0
        c1_y = (b1[1] + b1[3]) / 2.0
        c2_x = (b2[0] + b2[2]) / 2.0
        c2_y = (b2[1] + b2[3]) / 2.0

        # 4. Exit-to-entry guard: car exiting top edge (y1 <= 40) cannot reappear much lower (c2_y > c1_y + 80)
        if b1[1] <= 40.0 and (c2_y - c1_y) > 80.0:
            return False, iou, gap

        # 5. Box area ratio guard: sliver/far box cannot stitch with full-size vehicle when iou < 0.30
        if area1 <= 0.0 or area2 <= 0.0:
            return False, iou, gap
        if (max(area1, area2) / min(area1, area2)) > 2.2:
            return False, iou, gap

        # 6. Plated vs unplated guard: do not absorb an unplated edge/partial track into a plated car without overlap
        if bool(t1.best_plate_text) != bool(t2.best_plate_text) and iou < 0.10:
            return False, iou, gap

        center_dist = math.hypot(c1_x - c2_x, c1_y - c2_y)

        w1 = max(0.0, b1[2] - b1[0])
        h1 = max(0.0, b1[3] - b1[1])
        diag1 = math.hypot(w1, h1)

        w2 = max(0.0, b2[2] - b2[0])
        h2 = max(0.0, b2[3] - b2[1])
        diag2 = math.hypot(w2, h2)

        box_diag = min(diag1, diag2)
        if box_diag > 0 and center_dist <= 0.75 * box_diag:
            return True, iou, gap

        return False, iou, gap

    def can_stitch_tracks(self, t1: TrackState, t2: TrackState) -> bool:
        can_stitch, _, _ = self.evaluate_stitching(t1, t2)
        return can_stitch

    def merge_tracks(self, survivor: TrackState, absorbed: TrackState) -> None:
        """
        Merges 'absorbed' into 'survivor' track:
        Combines merged track IDs, frame boxes, and selects the highest scoring snapshot.
        Maintains first_raw_box, last_raw_box, first_frame, and last_frame.
        Respects plate lock on survivor and absorbed without cross-track voting.
        """
        for tid in absorbed.merged_track_ids:
            if tid not in survivor.merged_track_ids:
                survivor.merged_track_ids.append(tid)

        if absorbed.first_frame_idx < survivor.first_frame_idx:
            if absorbed.first_raw_box is not None:
                survivor.first_raw_box = [float(v) for v in absorbed.first_raw_box]
        if absorbed.last_frame_idx > survivor.last_frame_idx:
            if absorbed.last_raw_box is not None:
                survivor.last_raw_box = [float(v) for v in absorbed.last_raw_box]

        survivor.first_frame_idx = min(survivor.first_frame_idx, absorbed.first_frame_idx)
        survivor.last_frame_idx = max(survivor.last_frame_idx, absorbed.last_frame_idx)
        survivor.first_frame = survivor.first_frame_idx
        survivor.last_frame = survivor.last_frame_idx
        survivor.frames_tracked += absorbed.frames_tracked
        survivor.has_complete_frame = survivor.has_complete_frame or absorbed.has_complete_frame
        survivor.max_vehicle_area = max(survivor.max_vehicle_area, absorbed.max_vehicle_area)

        for f, b in absorbed.frame_boxes.items():
            if f not in survivor.frame_boxes:
                survivor.frame_boxes[f] = b

        if absorbed.best_plate_crop is not None:
            if (
                survivor.best_plate_crop is None
                or (absorbed.best_plate_conf or 0.0) > (survivor.best_plate_conf or 0.0)
            ):
                survivor.best_plate_crop = absorbed.best_plate_crop

        if absorbed.best_plate_det_conf is not None:
            if (
                survivor.best_plate_det_conf is None
                or float(absorbed.best_plate_det_conf) > float(survivor.best_plate_det_conf)
            ):
                survivor.best_plate_det_conf = float(absorbed.best_plate_det_conf)

        if absorbed.last_plate_box_frame is not None:
            if (
                survivor.last_plate_box_frame is None
                or absorbed.last_plate_box_frame_idx > survivor.last_plate_box_frame_idx
            ):
                survivor.last_plate_box_frame = absorbed.last_plate_box_frame
                survivor.last_plate_box_frame_idx = absorbed.last_plate_box_frame_idx
                if getattr(absorbed, "plate_rel_box", None) is not None and getattr(survivor, "plate_rel_box", None) is None:
                    survivor.plate_rel_box = absorbed.plate_rel_box
                if getattr(absorbed, "plate_anchor", None) is not None:
                    surv_unclipped = bool(getattr(survivor, "plate_anchor", None) and survivor.plate_anchor.get("unclipped"))
                    abs_unclipped = bool(absorbed.plate_anchor.get("unclipped"))
                    if getattr(survivor, "plate_anchor", None) is None or (abs_unclipped and not surv_unclipped):
                        survivor.plate_anchor = absorbed.plate_anchor

        # Plate lock: preserve locked plate on survivor or absorbed (NO cross-track voting)
        if survivor.plate_confident_reached and survivor.best_plate_text:
            pass
        elif absorbed.plate_confident_reached and absorbed.best_plate_text:
            survivor.best_plate_text = absorbed.best_plate_text
            survivor.best_plate_conf = absorbed.best_plate_conf
            survivor.best_plate_crop = absorbed.best_plate_crop
            survivor.best_plate_det_conf = absorbed.best_plate_det_conf
            survivor.plate_confident_reached = True
        elif not survivor.best_plate_text and absorbed.best_plate_text:
            survivor.best_plate_text = absorbed.best_plate_text
            survivor.best_plate_conf = absorbed.best_plate_conf
            survivor.best_plate_crop = absorbed.best_plate_crop
            survivor.best_plate_det_conf = absorbed.best_plate_det_conf
        elif survivor.best_plate_text and absorbed.best_plate_text:
            if (absorbed.best_plate_conf or 0.0) > (survivor.best_plate_conf or 0.0):
                survivor.best_plate_text = absorbed.best_plate_text
                survivor.best_plate_conf = absorbed.best_plate_conf
                survivor.best_plate_crop = absorbed.best_plate_crop
                survivor.best_plate_det_conf = absorbed.best_plate_det_conf

        if absorbed.plate_confident_reached:
            survivor.plate_confident_reached = True

        # Best snapshot selection across both (Requirement C)
        comb_max_sharp = max(survivor.max_sharpness, absorbed.max_sharpness)
        survivor.max_sharpness = comb_max_sharp

        score_surv = survivor.best_unnorm_score / max(1e-5, comb_max_sharp)
        score_abs = absorbed.best_unnorm_score / max(1e-5, comb_max_sharp)

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
        """
        Attempts to stitch 'track' with any matching track.
        Returns (surviving_track, match_info) where match_info is (iou, gap) if stitched, else None.
        """
        current_track = track
        for other_id, other in list(self.tracks.items()):
            if other_id == current_track.track_id:
                continue
            if other.stitched_into is not None or other.has_event_emitted:
                continue
            if other_id in current_track.merged_track_ids:
                continue

            can_stitch, iou, gap = self.evaluate_stitching(current_track, other)
            if can_stitch:
                if other.disappeared_count < self.config.max_disappeared_frames:
                    self.merge_tracks(survivor=other, absorbed=current_track)
                    return other, (iou, gap)
                else:
                    self.merge_tracks(survivor=current_track, absorbed=other)
                    return current_track, (iou, gap)
        return current_track, None

    def finalize_track_plate_reads(self, track: TrackState) -> None:
        """
        Flushes collected OCR reads for this track to output/plate_reads.jsonl.
        Saves up to 5 best crops (by score = confidence * plate_w) to output/plate_crops/.
        """
        if not getattr(self.config, "debug_outputs", False) or not self.config.collect_plate_data or track.plate_reads_flushed:
            track.plate_reads_flushed = True
            return

        if not track.all_plate_reads:
            track.plate_reads_flushed = True
            return

        # Sort reads by score = confidence * plate_w descending
        sorted_reads = sorted(track.all_plate_reads, key=lambda r: r.get("score", 0.0), reverse=True)

        # Top 5 crops saved
        saved_crops_count = 0
        for r in sorted_reads:
            crop = r.get("crop")
            if saved_crops_count < 5 and crop is not None and crop.size > 0:
                prefix = track.emitted_event_id if track.emitted_event_id else f"T{track.track_id}"
                crop_filename = f"crop_{prefix}_f{r['frame_idx']}_{saved_crops_count + 1}.png"
                crop_path = self.config.plate_crops_dir / crop_filename
                self.snapshot_saver.submit(crop_path, crop)
                r["crop_file_name"] = crop_filename
                saved_crops_count += 1
            else:
                r["crop_file_name"] = None
            r["crop"] = None

        # Write lines to plate_reads.jsonl
        self.config.plate_reads_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.plate_reads_jsonl_path, "a", encoding="utf-8") as f:
            for r in track.all_plate_reads:
                entry = {
                    "event_id": track.emitted_event_id,
                    "track_id": track.track_id,
                    "frame_idx": r["frame_idx"],
                    "raw_text": r["raw_text"],
                    "per_character_confidences": r["per_character_confidences"],
                    "plate_w": r["plate_w"],
                    "plate_h": r["plate_h"],
                    "crop_file_name": r.get("crop_file_name"),
                }
                f.write(json.dumps(entry) + "\n")

        track.plate_reads_flushed = True

    def check_and_emit_events(
        self,
        current_frame_idx: int,
        force_all: bool = False,
    ) -> List[dict]:
        """Evaluates active and disappeared tracks. Enforces strict stitching, completeness, and emission criteria."""
        emitted_now: List[dict] = []
        now_ts = datetime.now().timestamp()

        # Step 1: Commit frame snapshots for all active tracks in current frame
        self.commit_frame_snapshots()

        for t_id in list(self.tracks.keys()):
            track = self.tracks.get(t_id)
            if track is None or track.has_event_emitted or (t_id in self.emitted_track_ids) or track.stitched_into is not None:
                continue

            # A track emits when it ends (disappeared or force_all at end of video)
            is_ended = (track.disappeared_count >= self.config.max_disappeared_frames) or force_all
            if not is_ended:
                continue

            plate_str = track.best_plate_text or "NO_PLATE"

            # Minimum Track Frames Filter
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

            # Area Filter: horizon vehicles below minimum readable area
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

            # -------------------------------------------------------------
            # A. TRACK STITCHING: Check if this track can stitch with another track
            # -------------------------------------------------------------
            survivor, match_info = self.try_stitch_track(track)
            if survivor.track_id != track.track_id:
                # 'track' was merged into an active or earlier track ('survivor')!
                iou, gap = match_info if match_info is not None else (0.0, 0)
                print(
                    f"[TRACK] id={t_id} frames={track.first_frame}-{track.last_frame} "
                    f"plate={plate_str} -> STITCHED into track {survivor.track_id} (iou={iou:.2f}, gap={gap})"
                )
                self.total_tracks_stitched += 1
                continue

            track.has_event_emitted = True
            for mid in track.merged_track_ids:
                self.emitted_track_ids.add(mid)

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
            for t_id, track in list(self.tracks.items()):
                self.finalize_track_plate_reads(track)

        return emitted_now

    def _emit_event(self, track: TrackState, current_timestamp: float) -> Optional[dict]:
        """Formats single event JSON, submits async snapshots, and dispatches to Stage 2."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        has_plate = track.best_plate_text is not None and track.best_plate_conf is not None

        # Partial Plate Policy: Reject incomplete or invalid plate reads
        if has_plate:
            plate_candidate = track.best_plate_text
            known_chars = [c for c in plate_candidate if c != "?"]
            num_known = len(known_chars)

            if num_known < self.config.min_known_chars:
                has_plate = False
                track.best_plate_text = None
                track.best_plate_conf = None
                track.best_plate_det_conf = None
                track.best_plate_crop = None
            else:
                is_valid = False
                if self.config.plate_format == "UK":
                    uk_regex = re.compile(self.config.plate_regex_pattern or r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$")
                    is_valid = bool(uk_regex.match(plate_candidate)) and len(plate_candidate) >= self.config.min_plate_length
                elif self.config.plate_format == "PAKISTAN":
                    has_letter = any(c.isalpha() for c in plate_candidate)
                    has_digit = any(c.isdigit() for c in plate_candidate)
                    is_valid = (
                        len(plate_candidate) >= self.config.min_plate_length
                        and len(plate_candidate) <= 10
                        and (has_letter or has_digit)
                    )
                    if self.config.enforce_regex and self.config.plate_regex_pattern:
                        pk_regex = re.compile(self.config.plate_regex_pattern)
                        is_valid = is_valid and bool(pk_regex.match(plate_candidate))
                else:  # GENERAL format
                    is_valid = len(plate_candidate) >= self.config.min_plate_length
                    if self.config.enforce_regex and self.config.plate_regex_pattern:
                        gen_regex = re.compile(self.config.plate_regex_pattern)
                        is_valid = is_valid and bool(gen_regex.match(plate_candidate))

                is_partial_or_invalid = ("?" in plate_candidate) or (not is_valid)

                if is_partial_or_invalid:
                    if self.config.partial_plate_policy == "drop":
                        has_plate = False
                        track.best_plate_text = None
                        track.best_plate_conf = None
                        track.best_plate_det_conf = None
                        track.best_plate_crop = None
                    elif self.config.partial_plate_policy == "keep_low_conf":
                        mean_known = float(track.best_plate_conf or 0.60)
                        track.best_plate_conf = min(0.60, mean_known)

        # Fuzzy Duplicate Suppression
        if has_plate:
            plate_candidate = track.best_plate_text
            for existing_plate, (last_seen_ts, prev_event_id) in list(self.recent_emitted_plates.items()):
                if (current_timestamp - last_seen_ts) <= self.config.duplicate_plate_window_sec:
                    if is_fuzzy_duplicate_plate(plate_candidate, existing_plate, max_distance=self.config.fuzzy_plate_max_distance):
                        self.finalize_track_plate_reads(track)
                        return None

            self.recent_emitted_plates[plate_candidate] = (current_timestamp, f"EVT_{self.event_counter + 1}")

        sec_key = now.strftime("%y%m%d%H%M%S")
        if sec_key != self._last_sec_key:
            self._last_sec_key = sec_key
            self._sec_seq_counter = 0
        else:
            self._sec_seq_counter += 1

        self.event_counter += 1
        event_id = generate_infinite_event_id(
            node_id=self.config.node_id,
            camera_id_str=self.config.camera_id,
            now=now,
            seq_idx=self._sec_seq_counter,
        )
        track.emitted_event_id = event_id
        if has_plate:
            self.recent_emitted_plates[track.best_plate_text] = (current_timestamp, event_id)

        # Flush plate reads now that event_id is known
        self.finalize_track_plate_reads(track)

        # -------------------------------------------------------------
        # D. TIGHT SNAPSHOT:
        # Save snapshot_vehicle as a CLEAN crop (no drawn box, no label) of
        # the raw box with 6% padding clamped to the frame.
        # Save the annotated full frame separately as context_{event_id}.png (unreferenced in events.json).
        # -------------------------------------------------------------
        vehicle_filename = f"vehicle_{event_id}.png"
        vehicle_save_path = self.config.snapshots_dir / vehicle_filename

        crop_w_display = 0
        crop_h_display = 0
        raw_box_display = None

        if track.best_full_frame is not None and track.best_raw_box is not None:
            h_f, w_f = track.best_full_frame.shape[:2]
            rx1, ry1, rx2, ry2 = track.best_raw_box
            bw = max(1.0, rx2 - rx1)
            bh = max(1.0, ry2 - ry1)
            pad_x = int(bw * 0.06)
            pad_y = int(bh * 0.06)
            cx1 = max(0, int(rx1 - pad_x))
            cy1 = max(0, int(ry1 - pad_y))
            cx2 = min(w_f, int(rx2 + pad_x))
            cy2 = min(h_f, int(ry2 + pad_y))
            clean_vehicle_crop = track.best_full_frame[cy1:cy2, cx1:cx2].copy()
            crop_h_display, crop_w_display = clean_vehicle_crop.shape[:2]
            raw_box_display = [int(v) for v in track.best_raw_box]
            self.snapshot_saver.submit(vehicle_save_path, clean_vehicle_crop)
        elif track.best_vehicle_crop is not None and track.best_vehicle_crop.size > 0:
            crop_h_display, crop_w_display = track.best_vehicle_crop.shape[:2]
            raw_box_display = [int(v) for v in track.best_raw_box] if track.best_raw_box is not None else None
            self.snapshot_saver.submit(vehicle_save_path, track.best_vehicle_crop)
        else:
            placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
            crop_h_display, crop_w_display = 100, 100
            raw_box_display = None
            self.snapshot_saver.submit(vehicle_save_path, placeholder)

        print(
            f"[EVENT SNAPSHOT] Event {event_id}: Saved Snapshot Size={crop_w_display}x{crop_h_display}, "
            f"Raw Box={raw_box_display}"
        )

        # Context Full Frame with Annotation (only in debug-outputs mode)
        if getattr(self.config, "debug_outputs", False):
            context_filename = f"context_{event_id}.png"
            context_save_path = self.config.snapshots_dir / context_filename
            if track.best_full_frame is not None and track.best_raw_box is not None:
                annotated_context = track.best_full_frame.copy()
                h_f, w_f = annotated_context.shape[:2]
                vx1, vy1, vx2, vy2 = [int(v) for v in track.best_raw_box]
                vx1, vy1 = max(0, vx1), max(0, vy1)
                vx2, vy2 = min(w_f, vx2), min(h_f, vy2)

                box_color = (0, 255, 0) if has_plate else (0, 140, 255)
                cv2.rectangle(annotated_context, (vx1, vy1), (vx2, vy2), box_color, 3)

                label = (
                    f"{track.global_id} | {track.best_plate_text}"
                    if has_plate
                    else f"{track.global_id} | NO PLATE"
                )

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.65
                thickness = 2
                (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                label_y = max(th + 6, vy1 - 8)
                cv2.rectangle(
                    annotated_context,
                    (vx1, label_y - th - 6),
                    (vx1 + tw + 6, label_y + baseline + 2),
                    box_color,
                    -1,
                )
                cv2.putText(
                    annotated_context,
                    label,
                    (vx1 + 3, label_y),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness,
                    cv2.LINE_AA,
                )
                self.snapshot_saver.submit(context_save_path, annotated_context)

        # Cropped Plate snapshot
        plate_filename: Optional[str] = None
        plate_save_path: Optional[Path] = None
        if has_plate and track.best_plate_crop is not None and track.best_plate_crop.size > 0:
            plate_filename = f"plate_{event_id}.png"
            plate_save_path = self.config.snapshots_dir / plate_filename
            self.snapshot_saver.submit(plate_save_path, track.best_plate_crop)

        # JSON Event Output (Schema 100% Preserved)
        event_data = {
            "event_id": event_id,
            "camera_id": self.config.camera_id,
            "time": time_str,
            "date": date_str,
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
        }

        # Store metadata for final report (Requirement F)
        meta = {
            "event_id": event_id,
            "merged_track_ids": sorted(list(set(track.merged_track_ids))),
            "snapshot_frame_idx": track.best_snapshot_frame_idx,
            "is_complete": track.best_snapshot_complete,
            "plate_text": track.best_plate_text if has_plate else None,
            "plate_conf": round(float(track.best_plate_conf), 2) if (has_plate and track.best_plate_conf) else None,
            "snapshot_size": f"{crop_w_display}x{crop_h_display}",
            "raw_box": raw_box_display,
        }
        self.emitted_events_meta.append(meta)

        if self.config.include_remarks:
            if has_plate:
                if track.best_plate_conf >= self.config.min_confident_plate_conf:
                    event_data["remarks"] = "Number plate successfully detected and recognized"
                else:
                    event_data["remarks"] = "Low confidence / faint characters detected on plate"
            else:
                event_data["remarks"] = "No number plate detected on vehicle"

        # Update statistics
        self.total_unique_vehicles += 1
        if has_plate:
            self.total_with_plate += 1
        else:
            self.total_without_plate += 1

        self.emitted_events.append(event_data)

        # Write to JSONL
        with open(self.config.events_jsonl_path, "a", encoding="utf-8") as f_jsonl:
            f_jsonl.write(json.dumps(event_data) + "\n")

        # Rewrite full JSON atomically (temp file + os.replace)
        temp_json = self.config.events_json_path.with_suffix(".tmp")
        with open(temp_json, "w", encoding="utf-8") as f_json:
            json.dump(self.emitted_events, f_json, indent=2)
        os.replace(temp_json, self.config.events_json_path)

        # Publish to external sinks (Neon PostgreSQL + WebSocket) via EventBus
        if self.event_bus is not None:
            try:
                self.event_bus.publish(event_data)
            except Exception as e:
                print(f"[EVENT BUS] Publish error: {e}")

        # Console print only if explicitly requested
        if self.config.print_events_to_terminal:
            print(f"\n[EVENT RECORDED] {json.dumps(event_data, indent=2)}")

        # -------------------------------------------------------------
        # Stage 2 (Analyzer) Dispatch: Submit package to queue & handlers
        # -------------------------------------------------------------
        if self.config.enable_stage2_interface:
            stage2_pkg = {
                "event_id": event_id,
                "camera_id": self.config.camera_id,
                "number_plate": track.best_plate_text if has_plate else None,
                "vehicle_snapshot_path": vehicle_save_path,
                "plate_snapshot_path": plate_save_path,
                "event_data": event_data,
            }
            if not self.stage2_queue.full():
                try:
                    self.stage2_queue.put_nowait(stage2_pkg)
                except queue.Full:
                    pass

            for handler in self.stage2_handlers:
                try:
                    handler.on_event(
                        event_id=event_id,
                        camera_id=self.config.camera_id,
                        number_plate=track.best_plate_text if has_plate else None,
                        vehicle_snapshot_path=vehicle_save_path,
                        plate_snapshot_path=plate_save_path,
                        event_data=event_data,
                    )
                except Exception as e:
                    print(f"[STAGE 2 HANDLER ERROR] {e}")

        return event_data

    def get_counts(self) -> Tuple[int, int, int, int, int]:
        """
        Returns (total_unique, with_plate, without_plate, active, filtered_too_small).
        Rule B6: Count a vehicle in 'Total Unique Vehicles' as soon as its track is confirmed
        (not when it closes), and in 'Vehicles With Plate' as soon as a plate has been read.
        At the end of the run the HUD totals must match the final summary table.
        Merged tracks count once.
        """
        confirmed_active = [
            t for t in self.tracks.values()
            if t.stitched_into is None
            and not t.has_event_emitted
            and t.entered_readable_zone
            and (t.frames_tracked >= self.config.min_track_frames)
            and (t.has_complete_frame or t.max_vehicle_area >= self.config.min_readable_vehicle_area * 2.0)
        ]
        total_u = len(self.emitted_events) + len(confirmed_active)
        total_p = sum(1 for e in self.emitted_events if e.get("number_plate")) + sum(1 for t in confirmed_active if t.best_plate_text)
        total_np = max(0, total_u - total_p)
        active_count = sum(1 for t in self.tracks.values() if t.disappeared_count == 0 and t.stitched_into is None)
        return (
            total_u,
            total_p,
            total_np,
            active_count,
            self.total_filtered_too_small,
        )
