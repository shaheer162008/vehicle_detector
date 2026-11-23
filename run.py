"""
CLI Entry point to run Vehicle ALPR Tracking Pipeline (Enterprise Multi-Camera & Single Video Ready).
Examples:
    # Single offline video (default fallback):
    python run.py --source stream.mp4

    # Run with 40-camera configuration from cameras.json:
    python run.py --cameras-config cameras.json --camera-id CAM-00001

    # Run specific camera config tested on an offline video clip:
    python run.py --cameras-config cameras.json --camera-id CAM-00001 --source stream.mp4
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from config import PipelineConfig, load_camera_from_json
from pipeline import VehicleALPRPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Vehicle ALPR Single-Detection Pipeline (Enterprise Safe City Ready)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to input video file (e.g. stream.mp4), RTSP stream, or camera index. If omitted, uses camera RTSP from cameras.json.",
    )
    parser.add_argument(
        "--cameras-config",
        type=str,
        default=None,
        help="Optional path to cameras.json containing 40-camera definitions.",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default="CAM-00001",
        help="Camera ID string for event logging (e.g. CAM-00001). Default: CAM-00001.",
    )
    parser.add_argument(
        "--plate-format",
        type=str,
        choices=["PAKISTAN", "GENERAL", "UK"],
        default="UK",
        help="License plate format standard (default: UK).",
    )
    parser.add_argument(
        "--node-id",
        type=str,
        default="A",
        help="Node ID string for distributed deployment (default: 'A').",
    )
    parser.add_argument(
        "--plate-syntax-correction",
        action="store_true",
        default=False,
        help="Enable plate syntax normalization (default: False).",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default="output/annotated_output.mp4",
        help="Path where annotated output video will be saved.",
    )
    parser.add_argument(
        "--vehicle-conf",
        type=float,
        default=0.30,
        help="Confidence threshold for vehicle detection (default: 0.30).",
    )
    parser.add_argument(
        "--alpr-interval",
        type=int,
        default=3,
        help="Run ALPR every K frames to optimize compute (default: 3).",
    )
    parser.add_argument(
        "--alpr-on-crops",
        action="store_true",
        default=True,
        help="Run ALPR on vehicle crops for 3x speedup and higher accuracy (default: enabled).",
    )
    parser.add_argument(
        "--full-frame-alpr",
        dest="alpr_on_crops",
        action="store_false",
        help="Run ALPR on full frames instead of vehicle crops.",
    )
    parser.add_argument(
        "--min-track-frames",
        type=int,
        default=6,
        help="Minimum frames a vehicle must be tracked before emitting event (default: 6).",
    )
    parser.add_argument(
        "--disappeared-frames",
        type=int,
        default=45,
        help="Frames before an untracked vehicle is considered disappeared (default: 45).",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=2500.0,
        help="Minimum vehicle box area to be considered readable (default: 2500).",
    )
    parser.add_argument(
        "--partial-plate-policy",
        type=str,
        choices=["drop", "keep_low_conf"],
        default="keep_low_conf",
        help="Policy for incomplete plate reads ('keep_low_conf' keeps with '?' and capped conf, 'drop' marks as without-plate).",
    )
    parser.add_argument(
        "--no-plate-data",
        dest="collect_plate_data",
        action="store_false",
        default=True,
        help="Turn off saving plate reads to plate_reads.jsonl and crops to plate_crops/.",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        default=False,
        help="Do not archive existing output/ folder before starting.",
    )
    parser.add_argument(
        "--debug-reads",
        action="store_true",
        default=False,
        help="Print verbose OCR reads and per-character confidences during execution.",
    )
    parser.add_argument(
        "--debug-outputs",
        action="store_true",
        default=False,
        help="Save extra debug outputs (context images, plate crops, plate reads). Default False.",
    )
    parser.add_argument(
        "--no-plate-boxes",
        dest="draw_plate_boxes",
        action="store_false",
        default=True,
        help="Disable drawing green license plate bounding boxes in the annotated video.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        default=True,
        help="Enable Colab real-time visual preview via IPython.display (default: enabled).",
    )
    parser.add_argument(
        "--no-preview",
        dest="preview",
        action="store_false",
        help="Disable live visual preview.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        default=True,
        help="Save annotated output video file (default: enabled).",
    )
    parser.add_argument(
        "--no-save-video",
        dest="save_video",
        action="store_false",
        help="Do not save output video file to disk.",
    )
    parser.add_argument(
        "--sync-realtime",
        action="store_true",
        default=False,
        help="Synchronize playback to real wall-clock time (default: False for max FPS).",
    )
    parser.add_argument(
        "--enforce-regex",
        action="store_true",
        default=False,
        help="Strictly enforce country regex validation on plate reads.",
    )
    parser.add_argument(
        "--uk-syntax",
        action="store_true",
        default=False,
        help="Enable UK plate syntax normalization (e.g. resolve 0/O, 1/I, 8/B).",
    )
    parser.add_argument(
        "--print-events",
        action="store_true",
        default=False,
        help="Print event JSON to terminal console (default: False to reduce console noise).",
    )
    parser.add_argument(
        "--include-remarks",
        action="store_true",
        default=False,
        help="Include optional 'remarks' field in JSON event output.",
    )
    parser.add_argument(
        "--box-smoothing",
        type=float,
        default=0.0,
        help="Box smoothing factor for tracker (default: 0.0 for zero lag on ALPR / snapshots).",
    )
    parser.add_argument(
        "--hud-smoothing",
        type=float,
        default=0.65,
        help="Box smoothing factor for HUD visualization display only (default: 0.65).",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        help="Optional JSON string of ROI polygon coordinates, e.g. '[[100,200],[1100,200],[1200,700],[50,700]]'",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Optional frame number to start processing from.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 0. Output directory management
    output_dir = Path("output")
    if not args.keep_output and output_dir.exists():
        try:
            has_files = any(output_dir.iterdir())
        except Exception:
            has_files = False
        if has_files:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_dir = Path(f"output_prev_{timestamp_str}")
            try:
                shutil.move(str(output_dir), str(archive_dir))
                print(f"[OUTPUT] Existing 'output/' directory archived to '{archive_dir}'")
            except Exception as e:
                print(f"[OUTPUT WARNING] Could not archive output/ to {archive_dir}: {e}")

    roi_poly = None
    if args.roi:
        try:
            roi_poly = json.loads(args.roi)
        except Exception as e:
            print(f"[CONFIG WARNING] Could not parse --roi JSON: {e}")

    # 1. Base Configuration
    config = PipelineConfig(
        camera_id=args.camera_id,
        node_id=args.node_id,
        plate_format=args.plate_format,
        plate_syntax_correction=args.plate_syntax_correction,
        vehicle_conf_thresh=args.vehicle_conf,
        alpr_frame_interval=args.alpr_interval,
        alpr_on_crops=args.alpr_on_crops,
        min_track_frames=args.min_track_frames,
        max_disappeared_frames=args.disappeared_frames,
        min_readable_vehicle_area=args.min_area,
        partial_plate_policy=args.partial_plate_policy,
        collect_plate_data=args.collect_plate_data,
        show_preview=args.preview,
        save_video=args.save_video,
        sync_realtime=args.sync_realtime,
        enforce_regex=args.enforce_regex,
        uk_syntax_correction=args.uk_syntax,
        print_events_to_terminal=args.print_events,
        include_remarks=args.include_remarks,
        box_smoothing_factor=args.box_smoothing,
        hud_box_smoothing_factor=args.hud_smoothing,
        debug_reads=args.debug_reads,
        debug_outputs=args.debug_outputs,
        draw_plate_boxes=args.draw_plate_boxes,
        roi_polygon=roi_poly,
    )

    # 2. Check for cameras.json override if provided
    video_source = args.source
    if args.cameras_config:
        config = load_camera_from_json(
            json_path=args.cameras_config,
            camera_id=args.camera_id,
            base_config=config,
        )
        if not video_source:
            video_source = config.rtsp_url
            print(f"[INPUT] Using RTSP URL from cameras.json for {config.camera_id}: {video_source}")

    # 3. Fallback logic if video_source is still None
    if not video_source:
        # Check if default stream.mp4 exists in current working directory
        if Path("stream.mp4").exists():
            video_source = "stream.mp4"
            print(f"[INPUT] No source specified. Automatically defaulted to local video: {video_source}")
        else:
            print("[ERROR] No input video source provided!")
            print("Usage Examples:")
            print("  1. Custom offline video:     python run.py --source stream.mp4")
            print("  2. Multi-camera RTSP stream: python run.py --cameras-config cameras.json --camera-id CAM-00001")
            sys.exit(1)

    print(f"[SYSTEM] Pipeline ready for Camera: {config.camera_id} | Format: {config.plate_format}")
    if config.camera_location:
        print(f"[LOCATION] {config.camera_name} - {config.camera_location}")

    pipeline = VehicleALPRPipeline(config=config)
    try:
        pipeline.process_video(
            video_source=video_source,
            output_video_path=args.output_video,
            max_frames=args.max_frames,
            start_frame=args.start_frame,
        )
    except KeyboardInterrupt:
        print("\n[INFO] User interrupted pipeline. Finalizing and saving outputs...")
    except Exception as e:
        print(f"[FATAL ERROR] Pipeline crashed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
