"""
Configuration settings for the Vehicle Detection, Tracking & FastALPR Pipeline.
Loads environment variables from .env (with override=True) for Neon Postgres & Neon S3.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union, Dict

try:
    from dotenv import load_dotenv
    _root = Path(__file__).resolve().parent
    load_dotenv(_root / ".env", override=True)
    if (_root / ".env.local").exists():
        load_dotenv(_root / ".env.local", override=True)
except ImportError:
    pass


@dataclass
class PipelineConfig:
    # Camera & System Identification
    camera_id: str = "CAM-00001"
    camera_name: str = "Camera 1"
    camera_location: str = ""
    rtsp_url: str = ""
    node_id: str = "A"

    # Vehicle Detector & ByteTrack Tracker (Thresholds preserved)
    vehicle_model_name: str = "yolo11m.pt"
    vehicle_conf_thresh: float = 0.30
    vehicle_imgsz: int = 960
    vehicle_half_precision: bool = True
    vehicle_classes: List[int] = field(default_factory=lambda: [2, 3, 5, 7])  # car, motorcycle, bus, truck

    # Vehicle Type Classification
    vehicle_weights: str = "yolo11m.pt"  # custom model with class names
    ignore_classes: List[str] = field(default_factory=lambda: ["other"])  # class names to drop
    conf_per_class: Dict[str, float] = field(default_factory=dict)  # optional per-class conf override

    # File input options
    loop_file: bool = False
    simulate_realtime: bool = False

    tracker_config: str = "custom_bytetrack.yaml"

    max_spatial_jump_px: float = 140.0
    box_containment_thresh: float = 0.70

    # Readable Zone & Distance Filtering
    enable_smart_gating: bool = True
    min_vehicle_width: int = 40
    min_vehicle_height: int = 35
    min_readable_vehicle_area: float = 2500.0
    roi_top_cutoff_ratio: float = 0.0
    roi_polygon: Optional[List[List[int]]] = None

    # FastALPR License Plate Detection & OCR
    alpr_detector_model: str = "yolo-v9-s-608-license-plate-end2end"
    alpr_detector_conf_thresh: float = 0.35
    alpr_ocr_model: str = "cct-s-v2-global-model"
    alpr_frame_interval: int = 3
    alpr_on_crops: bool = True
    alpr_crop_min_width: int = 608

    # Plate Formatting & Consensus
    plate_format: str = "GENERAL"  # "GENERAL", "UK", or "PAKISTAN"
    plate_country_code: str = "GEN"
    plate_syntax_correction: bool = False
    min_plate_length: int = 4
    min_plate_sharpness: float = 25.0
    min_plate_reads_for_event: int = 2
    plate_regex_pattern: Optional[str] = None
    enforce_regex: bool = False
    uk_syntax_correction: bool = True
    partial_plate_policy: str = "keep_low_conf"  # "keep_low_conf" or "drop"
    min_known_chars: int = 3
    collect_plate_data: bool = True

    # Single Detection, Stitching & Emission
    min_track_frames: int = 6
    max_disappeared_frames: int = 45
    min_confident_plate_conf: float = 0.70
    duplicate_plate_window_sec: float = 60.0
    fuzzy_plate_max_distance: int = 2
    stitch_max_gap_frames: int = 15
    box_smoothing_factor: float = 0.0
    hud_box_smoothing_factor: float = 0.65
    include_remarks: bool = False
    print_events_to_terminal: bool = False
    debug_reads: bool = False
    debug_outputs: bool = False
    draw_plate_boxes: bool = True

    # Event Sinks (Neon Postgres, Neon S3 Object Storage, WebSocket)
    database_url: Optional[str] = field(default_factory=lambda: os.environ.get("DATABASE_URL"))
    ws_port: Optional[int] = field(
        default_factory=lambda: int(os.environ.get("WS_PORT", "8765")) if os.environ.get("WS_PORT") else 8765
    )
    enable_ws: bool = True
    s3_endpoint_url: Optional[str] = field(default_factory=lambda: os.environ.get("AWS_ENDPOINT_URL_S3"))
    s3_access_key_id: Optional[str] = field(default_factory=lambda: os.environ.get("AWS_ACCESS_KEY_ID"))
    s3_secret_access_key: Optional[str] = field(default_factory=lambda: os.environ.get("AWS_SECRET_ACCESS_KEY"))
    s3_region: str = field(default_factory=lambda: os.environ.get("AWS_REGION", "us-east-2"))
    s3_bucket: str = field(default_factory=lambda: os.environ.get("S3_BUCKET", "alpr-snapshots"))
    enable_s3: bool = True
    run_id: str = ""
    replay_outbox: bool = False
    no_local_snapshots: bool = False

    # Preview & Video Output
    show_preview: bool = True
    preview_interval: int = 2
    preview_max_width: int = 960
    preview_max_height: int = 540
    preview_jpeg_quality: int = 70
    sync_realtime: bool = False
    save_video: bool = True
    convert_h264: bool = True

    # Paths
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
        if self.debug_outputs and self.collect_plate_data:
            self.plate_crops_dir.mkdir(parents=True, exist_ok=True)

        fmt = self.plate_format.upper()
        if fmt == "PAKISTAN":
            self.plate_country_code = "PK"
            self.uk_syntax_correction = False
            if self.plate_regex_pattern is None:
                self.plate_regex_pattern = r"^[A-Z]{1,4}[0-9]{1,5}$|^[A-Z]{1,4}[0-9]{1,2}[A-Z]{0,2}[0-9]{1,4}$"
            self.min_plate_length = min(self.min_plate_length, 4)
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
            self.min_plate_length = min(self.min_plate_length, 4)


def load_camera_from_json(
    json_path: Union[str, Path],
    camera_id: str,
    base_config: Optional[PipelineConfig] = None,
) -> PipelineConfig:
    """Loads optional per-camera settings from a user-provided JSON file."""
    cfg = base_config or PipelineConfig()
    path = Path(json_path)
    if not path.exists():
        print(f"[CONFIG WARNING] Camera config not found: {path}. Using defaults.")
        return cfg

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        matched = next((c for c in data.get("cameras", []) if c.get("camera_id") == camera_id), None)
        if not matched:
            print(f"[CONFIG WARNING] Camera ID '{camera_id}' not found in {path}. Using defaults.")
            return cfg

        cfg.camera_id = matched.get("camera_id", camera_id)
        cfg.camera_name = matched.get("name", cfg.camera_name)
        cfg.camera_location = matched.get("location", cfg.camera_location)
        cfg.rtsp_url = matched.get("rtsp_url", cfg.rtsp_url)
        if "plate_format" in matched:
            cfg.plate_format = matched["plate_format"]
        if "min_readable_vehicle_area" in matched:
            cfg.min_readable_vehicle_area = float(matched["min_readable_vehicle_area"])
        if matched.get("roi_polygon"):
            cfg.roi_polygon = matched["roi_polygon"]
        cfg.__post_init__()
        return cfg
    except Exception as e:
        print(f"[CONFIG ERROR] Failed to parse {path}: {e}. Using defaults.")
        return cfg
