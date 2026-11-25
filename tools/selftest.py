#!/usr/bin/env python3
"""
FastALPR Self-Test: Verifies environment, dependencies, and configuration.
Exit code 0 = all PASS, 1 = any FAIL.
"""

import importlib
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env():
    """Load .env file."""
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env", override=True)
    except ImportError:
        pass


def check_python_version() -> tuple[bool, str]:
    """Check Python version is 3.10+."""
    v = sys.version_info
    if v.major == 3 and v.minor >= 10:
        return True, f"Python {v.major}.{v.minor}.{v.micro}"
    return False, f"Python {v.major}.{v.minor}.{v.micro} (need 3.10+)"


def check_venv() -> tuple[bool, str]:
    """Check if running inside virtual environment."""
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    return in_venv, "Inside venv" if in_venv else "NOT in venv (OK for Colab)"


def check_imports() -> list[tuple[bool, str]]:
    """Check critical imports."""
    modules = [
        ("cv2", "opencv-python"),
        ("ultralytics", "ultralytics"),
        ("onnxruntime", "onnxruntime"),
        ("fast_alpr", "fast-plate-ocr"),
        ("psycopg", "psycopg[binary]"),
        ("boto3", "boto3"),
        ("websockets", "websockets"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("pydantic", "pydantic"),
    ]
    results = []
    for mod, pkg in modules:
        try:
            importlib.import_module(mod)
            results.append((True, f"{mod} (from {pkg})"))
        except ImportError as e:
            results.append((False, f"{mod} (from {pkg}) - {e}"))
    return results


def check_torch_cuda() -> tuple[bool, str]:
    """Check PyTorch CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            return True, f"CUDA available: {torch.cuda.get_device_name(0)} ({torch.version.cuda})"
        return False, "CUDA NOT available (CPU only)"
    except Exception as e:
        return False, f"torch check failed: {e}"


def check_ort_providers() -> tuple[bool, str]:
    """Check ONNX Runtime providers."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        has_cuda = "CUDAExecutionProvider" in providers
        return has_cuda, f"Providers: {providers}"
    except Exception as e:
        return False, f"ORT check failed: {e}"


def check_ffmpeg() -> tuple[bool, str]:
    """Check ffmpeg availability."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            result = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True, timeout=5)
            version_line = result.stdout.split("\n")[0]
            return True, f"ffmpeg: {version_line}"
        except Exception:
            return True, f"ffmpeg found at {ffmpeg}"
    return False, "ffmpeg NOT found in PATH"


def check_mediamtx() -> tuple[bool, str]:
    """Check MediaMTX binary."""
    # Check in bin/ first
    bin_path = REPO_ROOT / "bin" / ("mediamtx.exe" if os.name == "nt" else "mediamtx")
    if bin_path.exists():
        return True, f"MediaMTX at {bin_path}"
    # Check PATH
    mtx = shutil.which("mediamtx")
    if mtx:
        return True, f"MediaMTX in PATH: {mtx}"
    return False, "MediaMTX NOT found (bin/ or PATH)"


def check_env_keys() -> tuple[bool, str]:
    """Check required .env keys exist (values not printed)."""
    required = [
        "DATABASE_URL",
        "AWS_ENDPOINT_URL_S3",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
        "S3_BUCKET",
        "SERVICE_TOKEN",
        "SERVICE_PORT",
        "ALLOWED_ORIGINS",
        "VIDEO_DIR",
        "RTSP_BASE",
        "VEHICLE_WEIGHTS",
        "PLATE_FORMAT",
        "PERF_PROFILE",
    ]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        return False, f"Missing keys: {', '.join(missing)}"
    return True, f"All {len(required)} keys present (values masked)"


def check_neon_db() -> tuple[bool, str]:
    """Test Neon DB connection and schema."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return False, "DATABASE_URL not set"
    try:
        import psycopg
        with psycopg.connect(db_url, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'events' AND column_name = 'vehicle_type'
                """)
                if cur.fetchone() is None:
                    return False, "events table missing vehicle_type column (run schema.sql)"
        return True, "Neon DB connected, schema OK"
    except Exception as e:
        return False, f"Neon DB connection failed: {e}"


def check_s3() -> tuple[bool, str]:
    """Test S3 connection (head_bucket / create-if-missing)."""
    endpoint = os.environ.get("AWS_ENDPOINT_URL_S3")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    region = os.environ.get("AWS_REGION", "us-east-2")
    bucket = os.environ.get("S3_BUCKET", "alpr-snapshots")

    if not (endpoint and access_key and secret_key):
        return False, "S3 credentials not fully configured"

    try:
        import boto3
        from botocore.config import Config
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=Config(s3={"addressing_style": "path"}, signature_version="s3v4", connect_timeout=10, read_timeout=20),
        )
        # Try head_bucket
        try:
            s3.head_bucket(Bucket=bucket)
        except Exception:
            # Try create
            try:
                if region != "us-east-1":
                    s3.create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": region})
                else:
                    s3.create_bucket(Bucket=bucket)
            except Exception as e:
                return False, f"S3 bucket create failed: {e}"
        return True, f"S3 OK: bucket '{bucket}' at {endpoint}"
    except Exception as e:
        return False, f"S3 connection failed: {e}"


def check_camera_rtsp() -> tuple[bool, str]:
    """Check camera 1 RTSP URL TCP reachable."""
    rtsp_url = os.environ.get("CAMERA_1_RTSP_URL", "")
    if not rtsp_url:
        return False, "CAMERA_1_RTSP_URL not set"
    if rtsp_url.startswith("rtsp://"):
        # Parse host:port
        try:
            host_port = rtsp_url.split("://")[1].split("/")[0]
            if "@" in host_port:
                host_port = host_port.split("@")[1]
            host, port = host_port.split(":")
            port = int(port)
        except Exception:
            return False, f"Invalid RTSP URL format: {rtsp_url}"

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                return True, f"RTSP port {host}:{port} reachable"
            return False, f"RTSP port {host}:{port} NOT reachable"
        except Exception as e:
            return False, f"RTSP check failed: {e}"
    elif rtsp_url.startswith("http://") or rtsp_url.startswith("https://"):
        return True, f"HTTP stream URL set (TCP check not applicable): {rtsp_url}"
    return False, f"Unknown URL scheme: {rtsp_url}"


def check_yolo_weights() -> tuple[bool, str]:
    """Check YOLO weights file exists or can be downloaded."""
    weights = os.environ.get("VEHICLE_WEIGHTS", "yolo11m.pt")
    if Path(weights).exists():
        return True, f"Weights file found: {weights}"
    # Check in repo root
    repo_weights = REPO_ROOT / weights
    if repo_weights.exists():
        return True, f"Weights file found: {repo_weights}"
    return True, f"Weights '{weights}' not found locally (will auto-download via ultralytics)"


def main() -> int:
    print("=== FastALPR Self-Test ===")
    load_env()

    all_pass = True
    results = []

    # 1. Python version
    ok, msg = check_python_version()
    results.append(("Python Version", ok, msg))
    all_pass &= ok

    # 2. Virtual env
    ok, msg = check_venv()
    results.append(("Virtual Env", ok, msg))
    # venv not required for Colab, so don't fail

    # 3. Imports
    imports = check_imports()
    for ok, msg in imports:
        results.append((f"Import: {msg.split()[0]}", ok, msg))
        all_pass &= ok

    # 4. Torch CUDA
    ok, msg = check_torch_cuda()
    results.append(("PyTorch CUDA", ok, msg))
    # GPU not required, WARN only
    if not ok:
        results[-1] = ("PyTorch CUDA", "WARN", msg)

    # 5. ONNX Runtime providers
    ok, msg = check_ort_providers()
    results.append(("ONNX Runtime", ok, msg))
    if not ok:
        results[-1] = ("ONNX Runtime", "WARN", msg)

    # 6. ffmpeg
    ok, msg = check_ffmpeg()
    results.append(("ffmpeg", ok, msg))
    all_pass &= ok

    # 7. MediaMTX
    ok, msg = check_mediamtx()
    results.append(("MediaMTX", ok, msg))
    all_pass &= ok

    # 8. .env keys
    ok, msg = check_env_keys()
    results.append((".env Keys", ok, msg))
    all_pass &= ok

    # 9. Neon DB
    ok, msg = check_neon_db()
    results.append(("Neon DB", ok, msg))
    all_pass &= ok

    # 10. S3
    ok, msg = check_s3()
    results.append(("Neon S3", ok, msg))
    all_pass &= ok

    # 11. Camera RTSP
    ok, msg = check_camera_rtsp()
    results.append(("Camera RTSP", ok, msg))
    if not ok:
        results[-1] = ("Camera RTSP", "WARN", msg + " (run stream.bat/sh first)")

    # 12. YOLO weights
    ok, msg = check_yolo_weights()
    results.append(("YOLO Weights", ok, msg))

    # Print summary table
    print("\n=== SELF-TEST RESULTS ===")
    for name, status, msg in results:
        if status == "WARN":
            icon = "⚠"
        elif status:
            icon = "✓"
        else:
            icon = "✗"
        print(f"  {icon} {name}: {msg}")

    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())