#!/usr/bin/env python3
"""
GPU Diagnostic Tool for FastALPR Pipeline.
Verifies PyTorch CUDA acceleration and ONNX Runtime CUDA Execution Provider.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path so we can import register_nvidia_dll_paths
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from pipeline import register_nvidia_dll_paths
    register_nvidia_dll_paths()
except Exception as e:
    print(f"[WARN] register_nvidia_dll_paths warning: {e}")

# 1. PyTorch CUDA check
try:
    import torch
except ImportError:
    print("[ERROR] PyTorch (torch) is not installed.")
    sys.exit(1)

if not torch.cuda.is_available():
    print("[ERROR] torch.cuda.is_available() is False. No CUDA-capable GPU detected by PyTorch.")
    sys.exit(1)

device_count = torch.cuda.device_count()
device_name = torch.cuda.get_device_name(0)
cap_major, cap_minor = torch.cuda.get_device_capability(0)

print(f"[TORCH CUDA] Available: True | Devices: {device_count}")
print(f"[TORCH CUDA] Device 0: {device_name} (Compute Capability: {cap_major}.{cap_minor})")

# 2. ONNX Runtime Provider check
try:
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception:
            pass
except ImportError:
    print("[ERROR] onnxruntime is not installed.")
    sys.exit(1)

providers = ort.get_available_providers()
print(f"[ONNX RUNTIME] Available Providers: {providers}")

if "CUDAExecutionProvider" not in providers:
    print("\n[ERROR] CUDAExecutionProvider is MISSING from onnxruntime providers!")
    print(f"        Current providers: {', '.join(providers)}")
    print("        Fix: Ensure 'onnxruntime-gpu' is installed, 'onnxruntime' (CPU) is uninstalled,")
    print("             and NVIDIA CUDA/cuDNN DLLs are registered/present in PATH.")
    sys.exit(1)

print("\nGPU OK")
sys.exit(0)
