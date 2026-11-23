#!/usr/bin/env bash
# ==============================================================================
# FastALPR + Vehicle Tracking Automated Fast Installer (Linux / macOS / Colab)
# Creates isolated virtual environment (.venv) and prefers GPU/CUDA
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

# 2. Create Isolated Virtual Environment (.venv)
if [ ! -f ".venv/bin/python" ]; then
    echo "[1/4] Creating isolated virtual environment (.venv)..."
    $PYTHON_CMD -m venv .venv || {
        echo "[WARNING] python3-venv might be missing. Trying without venv on Colab/container..."
    }
fi

if [ -f ".venv/bin/python" ]; then
    VENV_PYTHON=".venv/bin/python"
    echo "[OK] Using virtual environment: $VENV_PYTHON"
else
    VENV_PYTHON="$PYTHON_CMD"
    echo "[INFO] Operating directly with system python: $VENV_PYTHON"
fi

echo "[2/4] Upgrading pip..."
$VENV_PYTHON -m pip install --upgrade pip --no-input --quiet

# 3. Detect NVIDIA GPU / CUDA Capability (GPU Preferred)
HAS_NVIDIA=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        HAS_NVIDIA=true
    fi
fi

if [ "$HAS_NVIDIA" = true ]; then
    echo "[3/4] NVIDIA GPU detected! Installing CUDA 12 PyTorch and ONNXRuntime GPU..."
    $VENV_PYTHON -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --no-input --prefer-binary
    $VENV_PYTHON -m pip install "onnxruntime-gpu>=1.19.0" --no-input --prefer-binary
else
    echo "[3/4] No NVIDIA GPU detected. Installing CPU-optimized PyTorch and ONNXRuntime..."
    $VENV_PYTHON -m pip install torch torchvision --no-input --prefer-binary
    $VENV_PYTHON -m pip install "onnxruntime>=1.19.0" --no-input --prefer-binary
fi

# 4. Install Project Requirements
echo "[4/4] Installing project dependencies from requirements.txt..."
$VENV_PYTHON -m pip install -r requirements.txt --no-input --prefer-binary

echo ""
echo "=============================================================================="
echo " === Verifying Installation ==="
echo "=============================================================================="
$VENV_PYTHON -c "import cv2, torch, ultralytics; dev='GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'; print(f'[STATUS] PyTorch: {torch.__version__} | Ultralytics: {ultralytics.__version__} | Active Device: {dev}')"

echo ""
echo "[SUCCESS] Turnkey installation completed successfully!"
echo "[INFO] You can now run: ./run.sh"
echo "=============================================================================="
