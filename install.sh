#!/usr/bin/env bash
# ==============================================================================
# Automated Fast Installer for Linux / Google Colab / Boss PC
# ==============================================================================

set -e

cd "$(dirname "$0")"

echo "=== FastALPR + Vehicle Tracking Fast Installer ==="

# 1. Select Python version (Colab/Kaggle uses active python3; local Linux prefers 3.12/3.11)
PY_CMD=""
if [ -n "$COLAB_RELEASE_TAG" ] || [ -n "$COLAB_GPU" ] || [ -d "/content" ] || [ -n "$KAGGLE_CONTAINER_NAME" ]; then
    for cmd in python3 python; do
        if command -v "$cmd" &> /dev/null; then
            PY_CMD="$cmd"
            break
        fi
    done
else
    for cmd in python3.12 python3.11 python3.13 python3 python; do
        if command -v "$cmd" &> /dev/null; then
            PY_CMD="$cmd"
            break
        fi
    done
fi

if [ -z "$PY_CMD" ]; then
    echo "[ERROR] No Python found. Please install Python 3.11 or 3.12."
    exit 1
fi

echo "[INFO] Using Python: $($PY_CMD --version) ($(command -v "$PY_CMD"))"

# Create virtual environment only on local machines (not inside Colab/Kaggle containers)
VENV_PY="$PY_CMD"
if [ -z "$COLAB_RELEASE_TAG" ] && [ -z "$COLAB_GPU" ] && [ ! -d "/content" ] && [ -z "$KAGGLE_CONTAINER_NAME" ]; then
    if [ -d ".venv" ]; then
        rm -rf .venv
    fi
    if $PY_CMD -m venv .venv 2>/dev/null; then
        VENV_PY=".venv/bin/python"
        echo "[INFO] Created virtual environment (.venv)"
    fi
fi

echo "[1/4] Upgrading pip..."
$VENV_PY -m pip install --upgrade pip --no-input

echo "[2/4] Installing requirements.txt with pre-built binary wheels..."
$VENV_PY -m pip install -r requirements.txt --no-input --prefer-binary

echo "[3/4] Ensuring clean GPU builds (uninstalling CPU onnxruntime, ensuring CUDA torch + onnxruntime-gpu)..."
$VENV_PY -m pip uninstall -y onnxruntime >/dev/null 2>&1 || true
if $VENV_PY -c "import torch, torchvision; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "[INFO] CUDA PyTorch already active, skipping 2.5GB re-download."
else
    $VENV_PY -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128 --no-input
fi
$VENV_PY -m pip install "onnxruntime-gpu[cuda,cudnn]==1.22.0" --no-input

echo "[4/4] Verifying GPU acceleration with check_gpu.py..."
if ! $VENV_PY tools/check_gpu.py; then
    echo ""
    echo "================================================="
    echo "INSTALL FAILED: GPU not usable"
    echo "================================================="
    exit 1
fi

echo ""
echo "================================================="
echo "=== Installation Completed Successfully! ==="
echo "================================================="
