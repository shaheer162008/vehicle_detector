#!/usr/bin/env bash
# ==============================================================================
# FastALPR + Vehicle Tracking Automated Fast Installer (Linux / macOS / Colab)
# Creates isolated virtual environment (.venv) on laptops or uses Colab environment
# ==============================================================================

set -e
cd "$(dirname "$0")"

echo "=============================================================================="
echo " === FastALPR + Vehicle Tracking Automated Fast Installer (Linux / macOS) ==="
echo "=============================================================================="

# 1. Detect System Python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "[ERROR] Python 3 was not found in PATH! Please install Python 3.10+."
    exit 1
fi

echo "[INFO] Found System Python: $($PYTHON_CMD --version)"

# 2. Detect Google Colab / Container Environment vs Desktop Laptop
IS_COLAB=false
if [ -d "/content" ] || $PYTHON_CMD -c "import google.colab" &>/dev/null; then
    IS_COLAB=true
    echo "[INFO] Google Colab container environment detected."
    # Remove broken .venv if created earlier
    rm -rf .venv 2>/dev/null || true
    VENV_PYTHON="$PYTHON_CMD"
else
    # On desktop Linux/Mac: setup isolated virtual environment
    if [ ! -f ".venv/bin/pip" ]; then
        echo "[1/4] Setting up isolated virtual environment (.venv)..."
        rm -rf .venv 2>/dev/null || true
        if $PYTHON_CMD -m venv .venv 2>/dev/null && [ -f ".venv/bin/pip" ]; then
            echo "[OK] Created virtual environment (.venv)."
        else
            echo "[WARNING] python3-venv / ensurepip not installed on host. Operating directly with $PYTHON_CMD."
            rm -rf .venv 2>/dev/null || true
        fi
    fi

    if [ -f ".venv/bin/pip" ]; then
        VENV_PYTHON=".venv/bin/python"
        echo "[OK] Using virtual environment: $VENV_PYTHON"
    else
        VENV_PYTHON="$PYTHON_CMD"
        echo "[INFO] Using python: $VENV_PYTHON"
    fi
fi

# 3. Upgrade pip
echo "[2/4] Ensuring pip is up to date..."
$VENV_PYTHON -m pip install --upgrade pip --no-input --quiet 2>/dev/null || true

# 4. Detect NVIDIA GPU / CUDA Capability (GPU Preferred)
HAS_NVIDIA=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        HAS_NVIDIA=true
    fi
fi

if [ "$HAS_NVIDIA" = true ]; then
    echo "[3/4] NVIDIA GPU detected! Ensuring CUDA PyTorch and ONNXRuntime GPU..."
    # Check if torch with CUDA is already installed (common in Colab)
    if ! $VENV_PYTHON -c "import torch; assert torch.cuda.is_available()" &>/dev/null; then
        $VENV_PYTHON -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --no-input --prefer-binary
    else
        echo "[OK] CUDA PyTorch already available in environment."
    fi
    $VENV_PYTHON -m pip install "onnxruntime-gpu>=1.19.0" --no-input --prefer-binary || $VENV_PYTHON -m pip install "onnxruntime>=1.19.0" --no-input --prefer-binary
else
    echo "[3/4] No NVIDIA GPU detected. Installing CPU-optimized PyTorch and ONNXRuntime..."
    $VENV_PYTHON -m pip install torch torchvision --no-input --prefer-binary
    $VENV_PYTHON -m pip install "onnxruntime>=1.19.0" --no-input --prefer-binary
fi

# 5. Install Project Requirements
echo "[4/4] Installing project dependencies from requirements.txt..."
$VENV_PYTHON -m pip install -r requirements.txt --no-input --prefer-binary

echo ""
echo "=============================================================================="
echo " === Verifying Installation ==="
echo "=============================================================================="
$VENV_PYTHON -c "import cv2, torch, ultralytics, onnxruntime; dev='GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'; print(f'[STATUS] PyTorch: {torch.__version__} | Ultralytics: {ultralytics.__version__} | ONNXRuntime: {onnxruntime.__version__} | Active Device: {dev}')"

echo ""
echo "[SUCCESS] Turnkey installation completed successfully!"
echo "[INFO] You can now run: ./run.sh"
echo "=============================================================================="
