"""
Configuration settings for Vehicle Tracking and FastALPR Pipeline (Phase 1 & Phase 2 Ready).
All thresholds, paths, model names, ROI polygons, and performance settings are defined here.
Supports Pakistani, General, and UK plate formats, multi-camera JSON setups, and enterprise deployments.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class PipelineConfig:
    # -------------------------------------------------------------
    # Camera & System Identification
    # -------------------------------------------------------------
    camera_id: str = "CAM-00001"
    camera_name: str = "Main Boulevard - North Entry Lane 1"
    camera_location: str = "North Gate Toll Plaza / Entry Checkpost 1"
    rtsp_url: str = ""
    node_id: str = "A"

    # -------------------------------------------------------------
    # Vehicle Detector & Tracker Configuration
    # NOTE ON LICENSE:
    # Ultralytics YOLO models (YOLOv8, YOLO11, etc.) are licensed under AGPL-3.0.
    # For commercial closed-source deployments, consider:
    # 1. YOLOv10 (Apache-2.0)
    # 2. RT-DETR (Apache-2.0)
    # 3. YOLOX (Apache-2.0)
    # 4. Enterprise Commercial License from Ultralytics
    # -------------------------------------------------------------
    vehicle_model_name: str = "yolo11m.pt"  # Medium YOLO11 model
    vehicle_conf_thresh: float = 0.30       # Confidence threshold for vehicles
    vehicle_imgsz: int = 960                # Inference image size for YOLO
    vehicle_half_precision: bool = True     # Use FP16 half precision on CUDA
    # Target COCO classes: 2=car, 3=motorcycle, 5=bus, 7=truck
    vehicle_classes: List[int] = field(default_factory=lambda: [2, 3, 5, 7])
    tracker_config: str = "custom_bytetrack.yaml"  # Tuned ByteTrack config

    # Spatial Jump Threshold (in pixels) to detect ID switches between consecutive frames
    max_spatial_jump_px: float = 140.0

    # Nested & Duplicate Box Elimination
    box_containment_thresh: float = 0.70  # Merge if one box is >= 70% inside another
    class_agnostic_nms_iou: float = 0.60  # Class-agnostic NMS IoU threshold

    # -------------------------------------------------------------
    # Readable Zone (ROI) & Distance Filtering
    # Horizon / far-away small vehicles outside readable zone are COMPLETELY ignored
    # -------------------------------------------------------------
    enable_smart_gating: bool = True
    min_vehicle_width: int = 40
    min_vehicle_height: int = 35
    min_readable_vehicle_area: float = 2500.0  # Minimum box area to be considered readable
    # Horizon exclusion: 0.0 disables cutoff to capture far-away carriageway vehicles
    roi_top_cutoff_ratio: float = 0.0
    # Optional ROI polygon [[x, y], ...]. If None, whole frame lower zone is used.
    roi_polygon: Optional[List[List[int]]] = None

    # -------------------------------------------------------------
    # FastALPR License Plate Detection & OCR Configuration
    # -------------------------------------------------------------
    alpr_detector_model: str = "yolo-v9-s-608-license-plate-end2end"
    alpr_detector_conf_thresh: float = 0.35
    alpr_ocr_model: str = "cct-s-v2-global-model"
    # ALPR frame interval: Run ALPR every 3 frames on unresolved readable vehicles
    alpr_frame_interval: int = 3

    # ALPR Mode: Run on vehicle crops (much faster) vs full frame
    alpr_on_crops: bool = True
    alpr_crop_min_width: int = 608

    # Plate Format Standard: "UK", "PAKISTAN", or "GENERAL"
    plate_format: str = "UK"
    plate_country_code: str = "UK"
    plate_syntax_correction: bool = False
    min_plate_length: int = 7              # UK standard format is 7 characters (e.g. AB12CDE)
    min_plate_sharpness: float = 25.0
    min_plate_reads_for_event: int = 2     # Minimum plate reads needed before declaring confident
    plate_regex_pattern: Optional[str] = None
    enforce_regex: bool = False
    uk_syntax_correction: bool = True
    # Partial plate policy: "keep_low_conf" (keep text with '?', confidence capped at 0.60) or "drop"
    partial_plate_policy: str = "keep_low_conf"
    min_known_chars: int = 3               # Minimum non-'?' characters required to keep partial plate
    collect_plate_data: bool = True        # Collect OCR reads and plate crops for fine-tuning

    # -------------------------------------------------------------
    # Single Detection & Fuzzy Duplicate Suppression
    # -------------------------------------------------------------
    min_track_frames: int = 6             # Minimum frames tracked before emitting event
    max_disappeared_frames: int = 45      # Frames lost before closing track (1.5s at 30 fps)
    min_confident_plate_conf: float = 0.70
    duplicate_plate_window_sec: float = 60.0  # Suppress duplicate events within 60s
    fuzzy_plate_max_distance: int = 2         # Max Levenshtein distance for duplicate matching
    box_smoothing_factor: float = 0.0         # 0.0 for raw detector boxes (zero lag on ALPR & vehicle snapshots)
    hud_box_smoothing_factor: float = 0.65    # Smoothing factor for HUD video visualization only
    show_hud_stats: bool = False              # Show/hide HUD top-left statistics dashboard overlay
    include_remarks: bool = False             # Set True only if remarks field is requested
    print_events_to_terminal: bool = False    # Set True to print event JSON on console
    debug_reads: bool = False                 # Set True to print verbose per-read OCR confidences
    draw_plate_boxes: bool = True             # Set True to draw green plate bounding boxes in annotated video

    # -------------------------------------------------------------
    # Stage 2 (Downstream Analyzer) Interface Setup
    # -------------------------------------------------------------
    enable_stage2_interface: bool = True
    stage2_queue_maxsize: int = 1000

    # -------------------------------------------------------------
    # Real-Time Colab Preview & Performance
    # -------------------------------------------------------------
    show_preview: bool = True
    preview_interval: int = 2             # Update preview every N-th frame
    preview_max_width: int = 960
    preview_max_height: int = 540
    preview_jpeg_quality: int = 70
    sync_realtime: bool = False           # Set False for max speed processing
    save_video: bool = True
    convert_h264: bool = True             # Convert to H.264 / yuv420p with ffmpeg

    # -------------------------------------------------------------
    # Paths & Storage
    # -------------------------------------------------------------
    base_output_dir: Path = Path("output")
    snapshots_dir: Path = Path("output/snapshots")
    events_json_path: Path = Path("output/events.json")
    events_jsonl_path: Path = Path("output/events.jsonl")
    plate_reads_jsonl_path: Path = Path("output/plate_reads.jsonl")
    plate_crops_dir: Path = Path("output/plate_crops")
    drive_output_dir: Path = Path("/content/drive/MyDrive/ALPR_Output")

    def __post_init__(self) -> None:
        self.base_output_dir = Path(self.base_output_dir)
        self.snapshots_dir = Path(self.snapshots_dir)
        self.events_json_path = Path(self.events_json_path)
        self.events_jsonl_path = Path(self.events_jsonl_path)
        self.plate_reads_jsonl_path = Path(self.plate_reads_jsonl_path)
        self.plate_crops_dir = Path(self.plate_crops_dir)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        if self.collect_plate_data:
            self.plate_crops_dir.mkdir(parents=True, exist_ok=True)

        fmt = self.plate_format.upper()
        if fmt == "PAKISTAN":
            self.plate_country_code = "PK"
            self.uk_syntax_correction = False
            if self.plate_regex_pattern is None:
                # Flexible Pakistani pattern: Letters (province/city) + optional series/year + numbers
                self.plate_regex_pattern = r"^[A-Z]{1,4}[0-9]{1,5}$|^[A-Z]{1,4}[0-9]{1,2}[A-Z]{0,2}[0-9]{1,4}$"
            if self.min_plate_length == 7:  # If default was UK's 7, relax to 4
                self.min_plate_length = 4
        elif fmt == "UK":
            self.plate_country_code = "UK"
            self.uk_syntax_correction = True
            if self.plate_regex_pattern is None:
                self.plate_regex_pattern = r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$"
            self.min_plate_length = 7
        else:  # GENERAL
            self.plate_country_code = "GEN"
            self.uk_syntax_correction = False
            if self.plate_regex_pattern is None:
                self.plate_regex_pattern = r"^[A-Z0-9]{3,12}$"
            if self.min_plate_length == 7:
                self.min_plate_length = 4


def load_camera_from_json(
    json_path: Union[str, Path],
    camera_id: str,
    base_config: Optional[PipelineConfig] = None,
) -> PipelineConfig:
    """
    Loads camera configuration for a specific camera_id from cameras.json.
    Falls back gracefully if file or camera is not found.
    """
    cfg = base_config or PipelineConfig()
    path = Path(json_path)

    if not path.exists():
        print(f"[CONFIG WARNING] Camera config file not found: {path}. Using default configuration.")
        return cfg

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cameras_list = data.get("cameras", [])
        matched = next((c for c in cameras_list if c.get("camera_id") == camera_id), None)

        if not matched:
            print(f"[CONFIG WARNING] Camera ID '{camera_id}' not found in {path}. Using default configuration.")
            return cfg

        cfg.camera_id = matched.get("camera_id", camera_id)
        cfg.camera_name = matched.get("name", cfg.camera_name)
        cfg.camera_location = matched.get("location", cfg.camera_location)
        cfg.rtsp_url = matched.get("rtsp_url", cfg.rtsp_url)
        if "plate_format" in matched:
            cfg.plate_format = matched["plate_format"]
        if "min_readable_vehicle_area" in matched:
            cfg.min_readable_vehicle_area = float(matched["min_readable_vehicle_area"])
        if "roi_polygon" in matched and matched["roi_polygon"]:
            cfg.roi_polygon = matched["roi_polygon"]

        cfg.__post_init__()
        print(f"[CONFIG LOADED] Successfully loaded camera '{camera_id}': {cfg.camera_name} ({cfg.camera_location})")
        return cfg
    except Exception as e:
        print(f"[CONFIG ERROR] Failed to parse {path}: {e}. Using defaults.")
        return cfg
