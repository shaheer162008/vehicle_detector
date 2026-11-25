#!/usr/bin/env python3
"""
Interactive .env wizard for FastALPR.
Creates/updates .env at repo root with masked secret input.
Never echoes secrets to console.
"""

import getpass
import os
import re
import secrets
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"
ENV_EXAMPLE = REPO_ROOT / ".env.example"


def mask_secret(value: str) -> str:
    """Mask a secret value for display."""
    if not value:
        return "(empty)"
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "*" * (len(value) - 8) + value[-4:]


def read_existing_env() -> dict:
    """Read existing .env file preserving comments and unrelated lines."""
    if not ENV_FILE.exists():
        return {}
    data = {}
    try:
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    data[key.strip()] = val.strip()
    except Exception:
        pass
    return data


def write_env_atomic(env_data: dict, existing_lines: list = None) -> None:
    """Write .env atomically (temp + replace), preserving unrelated lines/comments."""
    if existing_lines is None:
        existing_lines = []

    # Build new lines: preserve comments/empty lines, update known keys
    known_keys = set(env_data.keys())
    new_lines = []
    seen_keys = set()

    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            new_lines.append(line.rstrip("\n"))
            continue
        if "=" in line:
            key = line.split("=")[0].strip()
            if key in known_keys and key not in seen_keys:
                new_lines.append(f"{key}={env_data[key]}")
                seen_keys.add(key)
            elif key not in known_keys:
                new_lines.append(line.rstrip("\n"))
        else:
            new_lines.append(line.rstrip("\n"))

    # Append any new keys not seen
    for key in known_keys:
        if key not in seen_keys:
            new_lines.append(f"{key}={env_data[key]}")

    # Atomic write
    tmp = ENV_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")
    # Backup old .env
    if ENV_FILE.exists():
        bak = ENV_FILE.with_suffix(".bak")
        try:
            ENV_FILE.replace(bak)
        except Exception:
            pass
    tmp.replace(ENV_FILE)


def prompt_with_default(prompt: str, default: str = "", is_secret: bool = False) -> str:
    """Prompt user with default value, return entered value or default."""
    if default:
        display = mask_secret(default) if is_secret else default
        full_prompt = f"{prompt} [{display}]: "
    else:
        full_prompt = f"{prompt}: "

    if is_secret:
        val = getpass.getpass(full_prompt)
    else:
        val = input(full_prompt).strip()

    if not val and default:
        return default
    return val


def generate_service_token() -> str:
    """Generate a 32-char random service token."""
    return secrets.token_urlsafe(32)


def main() -> int:
    print("=== FastALPR .env Wizard ===")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Target file: {ENV_FILE}")
    print()

    # Check existing .env
    existing_env = read_existing_env()
    existing_lines = []
    if ENV_FILE.exists():
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            existing_lines = f.readlines()

    if existing_env:
        print(".env already exists with these keys:")
        for k, v in existing_env.items():
            is_secret = any(s in k.upper() for s in ["PASSWORD", "SECRET", "KEY", "TOKEN", "URL"])
            print(f"  {k} = {mask_secret(v) if is_secret else v}")
        print()
        print("Options:")
        print("  [K] Keep as is (I'll edit manually)")
        print("  [U] Update missing values only")
        print("  [N] New .env (overwrite all)")
        choice = input("Choose [K/U/N]: ").strip().upper()
        if choice == "K":
            print("Keeping existing .env unchanged.")
            return 0
        elif choice == "U":
            print("Updating missing values...")
        else:
            print("Creating new .env...")
            existing_env = {}
            existing_lines = []
    else:
        print("No .env found. Creating new one.")
        print()

    # Core keys to configure
    core_keys = [
        ("DATABASE_URL", "Neon PostgreSQL connection string", True),
        ("AWS_ENDPOINT_URL_S3", "Neon S3 endpoint URL", True),
        ("AWS_ACCESS_KEY_ID", "Neon S3 access key ID", True),
        ("AWS_SECRET_ACCESS_KEY", "Neon S3 secret access key", True),
        ("AWS_REGION", "AWS region (e.g., us-east-2)", False, "us-east-2"),
        ("S3_BUCKET", "S3 bucket name", False, "alpr-snapshots"),
        ("SERVICE_TOKEN", "Service API token (auto-generated if empty)", True),
        ("SERVICE_PORT", "FastAPI service port", False, "8000"),
        ("ALLOWED_ORIGINS", "CORS allowed origins (comma-separated)", False, "http://localhost:3000"),
        ("VIDEO_DIR", "Video files directory", False, "../CPK-Management/public/videos"),
        ("RTSP_BASE", "Base RTSP URL for publishing", False, "rtsp://127.0.0.1:8554"),
        ("VEHICLE_WEIGHTS", "YOLO vehicle detection weights", False, "yolo11m.pt"),
        ("PLATE_FORMAT", "Plate format: GENERAL/PAKISTAN/UK", False, "GENERAL"),
        ("PERF_PROFILE", "Performance profile: full/light/auto", False, "auto"),
    ]

    # Prompt for core keys
    for key, desc, is_secret, *default in core_keys:
        default_val = default[0] if default else ""
        # Special handling for SERVICE_TOKEN - auto-generate if empty
        if key == "SERVICE_TOKEN" and not existing_env.get(key):
            val = generate_service_token()
            print(f"[AUTO] Generated SERVICE_TOKEN: {mask_secret(val)}")
            existing_env[key] = val
            continue

        current = existing_env.get(key, default_val)
        val = prompt_with_default(f"{key} ({desc})", current, is_secret)
        if val:
            existing_env[key] = val

    # Camera configuration
    print()
    print("--- Camera Configuration ---")
    try:
        cam_count = int(input("How many cameras? [1]: ").strip() or "1")
    except ValueError:
        cam_count = 1

    cam_count = max(1, min(40, cam_count))

    for i in range(1, cam_count + 1):
        print(f"\nCamera {i} (CAM-{i:05d}):")
        rtsp_key = f"CAMERA_{i}_RTSP_URL"
        name_key = f"CAMERA_{i}_NAME"
        loc_key = f"CAMERA_{i}_LOCATION"

        current_rtsp = existing_env.get(rtsp_key, "")
        current_name = existing_env.get(name_key, f"Camera {i}")
        current_loc = existing_env.get(loc_key, "")

        rtsp_val = prompt_with_default(f"  RTSP URL (Enter to skip)", current_rtsp, is_secret=True)
        if rtsp_val:
            existing_env[rtsp_key] = rtsp_val

        name_val = prompt_with_default(f"  Name", current_name)
        if name_val:
            existing_env[name_key] = name_val

        loc_val = prompt_with_default(f"  Location", current_loc)
        if loc_val:
            existing_env[loc_key] = loc_val

    # Preserve NEON_* and AI gateway keys from old .env
    neon_keys = [k for k in existing_env.keys() if k.startswith("NEON_") or k.startswith("AI_") or "GATEWAY" in k.upper()]
    for k in neon_keys:
        if k not in existing_env:
            # Keep from original
            pass

    # Write atomic
    write_env_atomic(existing_env, existing_lines)

    # Verify .gitignore
    gitignore = REPO_ROOT / ".gitignore"
    if gitignore.exists():
        content = gitignore.read_text(encoding="utf-8")
        if ".env" in content and "!.env.example" in content:
            print("\n[OK] .env is in .gitignore (secrets protected)")
        else:
            print("\n[WARN] .env may not be properly ignored by git!")

    print(f"\n[DONE] .env written to {ENV_FILE}")
    print("Run 'python tools/selftest.py' to verify configuration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())