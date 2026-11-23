@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ==============================================================================
echo  === FastALPR + Vehicle Tracking Automated Fast Installer (Windows) ===
echo ==============================================================================

rem 1. Detect System Python
set "SYS_PYTHON="
where python >nul 2>nul
if %errorlevel% equ 0 (
    set "SYS_PYTHON=python"
) else (
    where py >nul 2>nul
    if %errorlevel% equ 0 (
        set "SYS_PYTHON=py"
    )
)

if "%SYS_PYTHON%"=="" (
    echo [ERROR] Python was not found in PATH!
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo [INFO] Found System Python: %SYS_PYTHON%

rem 2. Create Isolated Virtual Environment (.venv)
if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating isolated virtual environment ^(.venv^)...
    %SYS_PYTHON% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [1/4] Virtual environment ^(.venv^) already exists.
)

set "VENV_PYTHON=.venv\Scripts\python.exe"

echo [2/4] Upgrading pip in .venv...
"%VENV_PYTHON%" -m pip install --upgrade pip --no-input --quiet

rem 3. Detect NVIDIA GPU / CUDA Capability (GPU Preferred)
set "HAS_NVIDIA=0"
where nvidia-smi >nul 2>nul
if %errorlevel% equ 0 (
    nvidia-smi >nul 2>nul
    if !errorlevel! equ 0 (
        set "HAS_NVIDIA=1"
    )
)

if "!HAS_NVIDIA!"=="1" (
    echo [3/4] NVIDIA GPU detected! Installing CUDA 12 PyTorch and ONNXRuntime GPU...
    "%VENV_PYTHON%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --no-input --prefer-binary
    "%VENV_PYTHON%" -m pip install "onnxruntime-gpu>=1.19.0" --no-input --prefer-binary
) else (
    echo [3/4] No NVIDIA GPU detected. Installing CPU-optimized PyTorch and ONNXRuntime...
    "%VENV_PYTHON%" -m pip install torch torchvision --no-input --prefer-binary
    "%VENV_PYTHON%" -m pip install "onnxruntime>=1.19.0" --no-input --prefer-binary
)

rem 4. Install Project Requirements
echo [4/4] Installing project dependencies into .venv...
"%VENV_PYTHON%" -m pip install -r requirements.txt --no-input --prefer-binary
if errorlevel 1 (
    echo [WARNING] Some dependencies had warnings during install. Verifying environment...
)

echo.
echo ==============================================================================
echo  === Verifying Installation ===
echo ==============================================================================
"%VENV_PYTHON%" -c "import cv2, torch, ultralytics, onnxruntime; dev='GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'; print(f'[STATUS] PyTorch: {torch.__version__} | Ultralytics: {ultralytics.__version__} | ONNXRuntime: {onnxruntime.__version__} | Active Device: {dev}')"
if errorlevel 1 (
    echo [ERROR] Environment verification failed. Check errors above.
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Turnkey installation completed successfully!
echo [INFO] You can now double-click or run: run.bat
echo ==============================================================================
endlocal
