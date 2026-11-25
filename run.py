"""
CLI entry point for the Vehicle Detection, Tracking & FastALPR Pipeline.
Example:
    python run.py --source stream_2.mp4 --plate-format GENERAL
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime as _dt
from pathlib import Path


def _purge_stale_modules() -> None:
    """Clears cached repo modules when invoked via %run inside Jupyter/Colab so re-clones always load fresh code."""
    prefixes = ("config", "pipeline", "tracker", "sinks", "fast_alpr")
    for mod_name in list(sys.modules.keys()):
        if mod_name == prefixes or mod_name.startswith(tuple(f"{p}." for p in prefixes)) or mod_name in prefixes:
            sys.modules.pop(mod_name, None)


def _check_onnx_cuda_warning() -> None:
    """Warns once if PyTorch CUDA is available while ONNX Runtime lacks CUDAExecutionProvider."""
    try:
        import torch

        if not torch.cuda.is_available():
            return
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:
                pass
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            print("[WARN] ONNX CPU par chal raha hai (onnxruntime-gpu install nahi) - OCR/plate detector slow")
    except Exception:
        pass


def _clean_output_dir(output_dir: Path, preserve_outbox: bool) -> None:
    """Cleans previous run artifacts while preserving outbox.jsonl and its referenced snapshots on --replay-outbox."""
    if not output_dir.exists():
        return

    outbox_file = output_dir / "outbox.jsonl"
    if not preserve_outbox or not outbox_file.exists():
        try:
            shutil.rmtree(str(output_dir), ignore_errors=True)
        except Exception as e:
            print(f"[OUTPUT WARNING] Could not clean output/: {e}")
        return

    # Collect snapshot filenames referenced by pending outbox entries
    keep_snapshots: set[str] = set()
    try:
        for line in outbox_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ev = json.loads(line).get("event", {})
            for key in ("snapshot_vehicle", "snapshot_number_plate"):
                if ev.get(key):
                    keep_snapshots.add(Path(ev[key]).name)
    except Exception:
        pass

    for item in output_dir.iterdir():
        if item.name == "outbox.jsonl":
            continue
        if item.name == "snapshots" and item.is_dir():
            for snap in item.iterdir():
                if snap.is_file() and snap.name not in keep_snapshots:
                    snap.unlink(missing_ok=True)
            continue
        if item.is_dir():
            shutil.rmtree(str(item), ignore_errors=True)
        else:
            item.unlink(missing_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="FastALPR Vehicle Tracking Pipeline")
    parser.add_argument("--source", type=str, default=None, help="Input video file, RTSP URL, or camera index.")
    parser.add_argument("--cameras-config", type=str, default=None, help="Optional path to custom cameras.json.")
    parser.add_argument("--camera-id", type=str, default="CAM-00001", help="Camera ID (default: CAM-00001).")
    parser.add_argument(
        "--plate-format",
        type=str,
        choices=["PAKISTAN", "GENERAL", "UK"],
        default="GENERAL",
        help="Plate format standard (default: GENERAL).",
    )
    parser.add_argument("--node-id", type=str, default="A", help="Node ID prefix for event IDs (default: 'A').")
    parser.add_argument("--plate-syntax-correction", action="store_true", default=False)
    parser.add_argument("--output-video", type=str, default="output/annotated_output.mp4")
    parser.add_argument("--vehicle-conf", type=float, default=0.30)
    parser.add_argument("--vehicle-weights", type=str, default="yolo11m.pt", help="Vehicle detection model (custom .pt or .engine)")
    parser.add_argument("--ignore-classes", type=str, default="other", help="Comma-separated class names to ignore (default: 'other')")
    parser.add_argument("--conf-per-class", type=str, default="", help="Optional per-class conf overrides, e.g. 'car:0.25,truck:0.35'")
    parser.add_argument("--alpr-interval", type=int, default=3)
    parser.add_argument("--alpr-on-crops", action="store_true", default=True)
    parser.add_argument("--full-frame-alpr", dest="alpr_on_crops", action="store_false")
    parser.add_argument("--min-track-frames", type=int, default=6)
    parser.add_argument("--disappeared-frames", type=int, default=45)
    parser.add_argument("--min-area", type=float, default=2500.0)
    parser.add_argument("--partial-plate-policy", type=str, choices=["drop", "keep_low_conf"], default="keep_low_conf")
    parser.add_argument("--no-plate-data", dest="collect_plate_data", action="store_false", default=True)
    parser.add_argument("--keep-output", action="store_true", default=False, help="Keep existing output/ directory.")
    parser.add_argument("--debug-reads", action="store_true", default=False)
    parser.add_argument("--debug-outputs", action="store_true", default=False)
    parser.add_argument("--database-url", default=None, help="Neon PostgreSQL URL (defaults to DATABASE_URL env).")
    parser.add_argument("--ws-port", type=int, default=None, help="WebSocket port (defaults to WS_PORT env or 8765).")
    parser.add_argument("--no-ws", dest="enable_ws", action="store_false", default=True)
    parser.add_argument("--s3-bucket", type=str, default=None, help="Neon S3 bucket name (defaults to S3_BUCKET or 'alpr-snapshots').")
    parser.add_argument("--no-s3", dest="enable_s3", action="store_false", default=True)
    parser.add_argument("--replay-outbox", action="store_true", default=False, help="Replay failed events from output/outbox.jsonl at run start.")
    parser.add_argument("--no-local-snapshots", action="store_true", default=False, help="Delete local snapshots after successful S3 upload + DB insert.")
    parser.add_argument("--no-plate-boxes", dest="draw_plate_boxes", action="store_false", default=True)
    parser.add_argument("--preview", action="store_true", default=True)
    parser.add_argument("--no-preview", dest="preview", action="store_false")
    parser.add_argument("--save-video", action="store_true", default=True)
    parser.add_argument("--no-save-video", dest="save_video", action="store_false")
    parser.add_argument("--sync-realtime", action="store_true", default=False)
    parser.add_argument("--enforce-regex", action="store_true", default=False)
    parser.add_argument("--uk-syntax", action="store_true", default=False)
    parser.add_argument("--print-events", action="store_true", default=False)
    parser.add_argument("--include-remarks", action="store_true", default=False)
    parser.add_argument("--box-smoothing", type=float, default=0.0)
    parser.add_argument("--hud-smoothing", type=float, default=0.65)
    parser.add_argument("--roi", type=str, default=None, help="JSON polygon ROI coordinates.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    # File loop & realtime simulation
    parser.add_argument("--loop-file", action="store_true", default=False, help="Loop video file on EOF")
    parser.add_argument("--simulate-realtime", action="store_true", default=False, help="Pace file playback at native FPS")
    # Timer & performance
    parser.add_argument("--max-minutes", type=float, default=0.0, help="Stop after N minutes (0 = unlimited)")
    parser.add_argument("--perf-profile", type=str, choices=["full", "light", "auto"], default="auto", help="Performance profile: full (GPU), light (CPU), auto (detect)")
    # Service relay args
    parser.add_argument("--relay-url", type=str, default=None, help="Internal WebSocket relay URL (ws://host:port/internal/ingest/stream_id?token=...)")
    parser.add_argument("--stream-id", type=str, default=None, help="Stream ID for service relay")
    parser.add_argument("--preview-fps", type=int, default=10, help="Preview frame rate for relay (default: 10)")
    parser.add_argument("--preview-width", type=int, default=640, help="Preview frame width for relay (default: 640)")
    return parser.parse_args()


def main() -> None:
    _purge_stale_modules()

    repo_root = str(Path(__file__).resolve().parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        from dotenv import load_dotenv
        load_dotenv(Path(repo_root) / ".env", override=True)
        env_local = Path(repo_root) / ".env.local"
        if env_local.exists():
            load_dotenv(env_local, override=True)
    except ImportError:
        pass

    from config import PipelineConfig, load_camera_from_json
    from pipeline import VehicleALPRPipeline

    args = parse_args()

    if not args.keep_output:
        _clean_output_dir(Path("output"), preserve_outbox=args.replay_outbox)

    roi_poly = None
    if args.roi:
        try:
            roi_poly = json.loads(args.roi)
        except Exception as e:
            print(f"[CONFIG WARNING] Could not parse --roi JSON: {e}")

    ws_port = args.ws_port
    if ws_port is None:
        ws_env = os.environ.get("WS_PORT", "8765")
        ws_port = int(ws_env) if ws_env else 8765

    run_id = _dt.now().strftime("%Y%m%d-%H%M%S")

    # Parse ignore_classes
    ignore_classes = [c.strip() for c in args.ignore_classes.split(",") if c.strip()] if args.ignore_classes else ["other"]

    # Parse conf_per_class
    conf_per_class = {}
    if args.conf_per_class:
        for pair in args.conf_per_class.split(","):
            if ":" in pair:
                cls, conf = pair.split(":", 1)
                conf_per_class[cls.strip()] = float(conf.strip())

    # Env defaults for source, weights, plate_format, perf_profile
    video_source = args.source
    vehicle_weights = args.vehicle_weights or os.environ.get("VEHICLE_WEIGHTS", "yolo11m.pt")
    plate_format = args.plate_format or os.environ.get("PLATE_FORMAT", "GENERAL")
    perf_profile = args.perf_profile or os.environ.get("PERF_PROFILE", "auto")

    # Auto-detect GPU for perf_profile
    if perf_profile == "auto":
        try:
            import torch
            perf_profile = "full" if torch.cuda.is_available() else "light"
        except Exception:
            perf_profile = "light"
    print(f"[PERF] Profile: {perf_profile}")

    # Apply light profile settings (CPU-friendly)
    alpr_interval = args.alpr_interval
    vehicle_imgsz = 960
    if perf_profile == "light":
        alpr_interval = max(args.alpr_interval, 5)  # Less frequent ALPR
        vehicle_imgsz = 640  # Smaller inference size
        print(f"[PERF] Light profile: alpr_interval={alpr_interval}, vehicle_imgsz={vehicle_imgsz}")

    # Parse ignore_classes
    ignore_classes = [c.strip() for c in args.ignore_classes.split(",") if c.strip()] if args.ignore_classes else ["other"]

    # Parse conf_per_class
    conf_per_class = {}
    if args.conf_per_class:
        for pair in args.conf_per_class.split(","):
            if ":" in pair:
                cls, conf = pair.split(":", 1)
                conf_per_class[cls.strip()] = float(conf.strip())

    config = PipelineConfig(
        camera_id=args.camera_id,
        node_id=args.node_id,
        plate_format=plate_format,
        plate_syntax_correction=args.plate_syntax_correction,
        vehicle_conf_thresh=args.vehicle_conf,
        vehicle_weights=vehicle_weights,
        ignore_classes=ignore_classes,
        conf_per_class=conf_per_class,
        loop_file=args.loop_file,
        simulate_realtime=args.simulate_realtime,
        alpr_frame_interval=alpr_interval,
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
        database_url=args.database_url or os.environ.get("DATABASE_URL"),
        ws_port=ws_port,
        enable_ws=args.enable_ws,
        s3_endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3"),
        s3_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        s3_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        s3_region=os.environ.get("AWS_REGION", "us-east-2"),
        s3_bucket=args.s3_bucket or os.environ.get("S3_BUCKET", "alpr-snapshots"),
        enable_s3=args.enable_s3,
        run_id=run_id,
        replay_outbox=args.replay_outbox,
        no_local_snapshots=args.no_local_snapshots,
        draw_plate_boxes=args.draw_plate_boxes,
        roi_polygon=roi_poly,
    )

    # Determine video source
    if not video_source and args.cameras_config:
        config = load_camera_from_json(args.cameras_config, args.camera_id, base_config=config)
        if not video_source:
            video_source = config.rtsp_url

    # Env default for camera 1 RTSP URL
    if not video_source:
        cam1_rtsp = os.environ.get("CAMERA_1_RTSP_URL")
        if cam1_rtsp and cam1_rtsp.strip():
            video_source = cam1_rtsp.strip()
            print(f"[INPUT] Using CAMERA_1_RTSP_URL from env: {video_source}")

    if not video_source:
        for fallback in ("stream_2.mp4", "stream.mp4"):
            if Path(fallback).exists():
                video_source = fallback
                print(f"[INPUT] No --source specified; using {video_source}")
                break
        if not video_source:
            print("[ERROR] No input video source provided. Pass --source <video_or_rtsp>.")
            sys.exit(1)

    print(f"[SYSTEM] Camera: {config.camera_id} | Format: {config.plate_format} | Run: {run_id}")
    if args.max_minutes > 0:
        print(f"[TIMER] Will stop after {args.max_minutes} minutes")
    _check_onnx_cuda_warning()
    pipeline = VehicleALPRPipeline(
        config=config,
        run_id=run_id,
        video_source=video_source,
        relay_url=args.relay_url,
        stream_id=args.stream_id,
        preview_fps=args.preview_fps,
        preview_width=args.preview_width,
    )

    # Timer for max-minutes
    import threading
    stop_event = threading.Event()
    if args.max_minutes > 0:
        def timer_stop():
            time.sleep(args.max_minutes * 60)
            print(f"\n[TIMER] {args.max_minutes} minutes elapsed - stopping gracefully...")
            stop_event.set()
        timer_thread = threading.Thread(target=timer_stop, daemon=True)
        timer_thread.start()

    try:
        pipeline.process_video(
            video_source=video_source,
            output_video_path=args.output_video,
            max_frames=args.max_frames,
            start_frame=args.start_frame,
            stop_event=stop_event,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Outputs finalized.")
    except Exception as e:
        print(f"[FATAL ERROR] Pipeline crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Print final performance summary from pipeline
    print(f"\n[PERF] Run completed: {run_id}")
    print(f"[PERF] Camera: {config.camera_id} | Profile: {perf_profile} | Weights: {vehicle_weights}")
    print(f"[PERF] Plate format: {plate_format} | Device: {'GPU' if perf_profile == 'full' else 'CPU'}")


if __name__ == "__main__":
    main()
