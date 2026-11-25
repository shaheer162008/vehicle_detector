#!/usr/bin/env bash
# FastALPR Setup (Linux / Colab / macOS)

set -e

cd "$(dirname "$0")"

echo "=== FastALPR Setup (Linux) ==="

# 1. Find Python 3.10-3.12
PY_CMD=""
for v in 3.12 3.11 3.10; do
    if command -v "python$v" &>/dev/null; then
        PY_CMD="python$v"
        break
    fi
done

if [ -z "$PY_CMD" ]; then
    if command -v python3 &>/dev/null; then
        PY_CMD="python3"
    elif command -v python &>/dev/null; then
        PY_CMD="python"
    fi
fi

if [ -z "$PY_CMD" ]; then
    echo "[ERROR] Python 3.10-3.12 not found. Install python3.12 or python3.11"
    exit 1
fi

echo "[INFO] Using Python: $($PY_CMD --version)"

# 2. Check if Colab (skip venv)
IS_COLAB=0
if [ -n "$COLAB_GPU" ] || [ -n "$COLAB_RELEASE_TAG" ] || [ -d "/content" ]; then
    IS_COLAB=1
fi

VENV_PY="$PY_CMD"
if [ "$IS_COLAB" -eq 0 ]; then
    if [ -d ".venv" ]; then
        echo "[INFO] Reusing existing .venv"
    else
        echo "[INFO] Creating .venv..."
        $PY_CMD -m venv .venv
    fi
    VENV_PY=".venv/bin/python"
else
    echo "[INFO] Colab detected - using system Python"
fi

# 3. Upgrade pip & install requirements
echo "[1/5] Upgrading pip..."
$VENV_PY -m pip install --upgrade pip --no-input

echo "[2/5] Installing requirements.txt..."
$VENV_PY -m pip install -r requirements.txt --no-input --prefer-binary

# 4. GPU detection & torch/onnxruntime
echo "[3/5] Detecting GPU..."
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    echo "[INFO] NVIDIA GPU detected - installing CUDA torch + onnxruntime-gpu"
    $VENV_PY -m pip uninstall -y onnxruntime >/dev/null 2>&1 || true
    $VENV_PY -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128 --no-input
    $VENV_PY -m pip install "onnxruntime-gpu[cuda,cudnn]==1.22.0" --no-input
else
    echo "[INFO] No NVIDIA GPU - installing CPU torch + onnxruntime"
    $VENV_PY -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-input
    $VENV_PY -m pip install onnxruntime --no-input
fi

# 5. Install ffmpeg (imageio-ffmpeg) & download MediaMTX
echo "[4/5] Installing ffmpeg (imageio-ffmpeg) & MediaMTX..."
$VENV_PY -m pip install imageio-ffmpeg --no-input

echo "[INFO] Downloading MediaMTX..."
mkdir -p bin
ARCH=$(uname -m)
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
case "$OS-$ARCH" in
    linux-x86_64) ASSET="linux_amd64.tar.gz" ;;
    linux-aarch64|linux-arm64) ASSET="linux_arm64v8.tar.gz" ;;
    darwin-x86_64) ASSET="darwin_amd64.tar.gz" ;;
    darwin-arm64) ASSET="darwin_arm64.tar.gz" ;;
    *) ASSET="linux_amd64.tar.gz" ;;
esac

RELEASE=$(curl -s -H "User-Agent: fastalpr-setup" https://api.github.com/repos/bluenviron/mediamtx/releases/latest)
URL=$(echo "$RELEASE" | grep -o "\"browser_download_url\": \"[^\"]*$ASSET\"" | head -1 | cut -d'"' -f4)
if [ -n "$URL" ]; then
    curl -L -o "mediamtx.tar.gz" "$URL"
    tar -xzf "mediamtx.tar.gz" -C bin
    rm "mediamtx.tar.gz"
    chmod +x bin/mediamtx
    echo "[OK] MediaMTX extracted to bin/"
else
    echo "[WARN] MediaMTX $ASSET not found in release. Manually place mediamtx in bin/"
fi

# 6. Env wizard + selftest
echo "[5/5] Running env wizard..."
$VENV_PY tools/env_wizard.py

echo ""
echo "================================================="
echo "=== Setup Complete ==="
echo "================================================="
echo ""
echo "Next steps:"
echo "  1. Edit .env if needed (Neon creds, camera URLs)"
echo "  2. Run: ./stream.sh    (to start RTSP stream)"
echo "  3. Run: ./run.sh       (to start detection)"
echo "  4. Or: ./run.sh mode 2 (for Next.js service)"
echo ""