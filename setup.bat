@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo === FastALPR Setup (Windows) ===

rem 1. Find Python 3.10-3.12
set "PY_CMD="
for %%v in (3.12 3.11 3.10) do (
    py -%%v --version >nul 2>nul
    if !errorlevel! equ 0 (
        set "PY_CMD=py -%%v"
        goto found_python
    )
)

rem Fallback: try python3 / python
python3 --version >nul 2>nul
if %errorlevel% equ 0 (
    set "PY_CMD=python3"
    goto found_python
)

python --version >nul 2>nul
if %errorlevel% equ 0 (
    set "PY_CMD=python"
    goto found_python
)

echo [ERROR] Python 3.10-3.12 not found. Install from python.org
exit /b 1

:found_python
echo [INFO] Using Python: %PY_CMD%

rem 2. Check if Colab (skip venv)
set "IS_COLAB=0"
if defined COLAB_GPU set "IS_COLAB=1"
if defined COLAB_RELEASE_TAG set "IS_COLAB=1"
if exist "/content" set "IS_COLAB=1"

set "VENV_PY=%PY_CMD%"
if "%IS_COLAB%"=="0" (
    if exist ".venv" (
        echo [INFO] Reusing existing .venv
    ) else (
        echo [INFO] Creating .venv...
        %PY_CMD% -m venv .venv
        if %errorlevel% neq 0 (
            echo [ERROR] Failed to create .venv
            exit /b 1
        )
    )
    set "VENV_PY=.venv\Scripts\python.exe"
) else (
    echo [INFO] Colab detected - using system Python
)

rem 3. Upgrade pip & install requirements
echo [1/5] Upgrading pip...
%VENV_PY% -m pip install --upgrade pip --no-input

echo [2/5] Installing requirements.txt...
%VENV_PY% -m pip install -r requirements.txt --no-input --prefer-binary

rem 4. GPU detection & torch/onnxruntime
echo [3/5] Detecting GPU...
%VENV_PY% -c "import subprocess; r=subprocess.run(['nvidia-smi'], capture_output=True); exit(0 if r.returncode==0 else 1)" >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] NVIDIA GPU detected - installing CUDA torch + onnxruntime-gpu
    %VENV_PY% -m pip uninstall -y onnxruntime >nul 2>&1
    %VENV_PY% -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128 --no-input
    %VENV_PY% -m pip install "onnxruntime-gpu[cuda,cudnn]==1.22.0" --no-input
) else (
    echo [INFO] No NVIDIA GPU - installing CPU torch + onnxruntime
    %VENV_PY% -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-input
    %VENV_PY% -m pip install onnxruntime --no-input
)

rem 5. Install ffmpeg (imageio-ffmpeg) & download MediaMTX
echo [4/5] Installing ffmpeg (imageio-ffmpeg) & MediaMTX...
%VENV_PY% -m pip install imageio-ffmpeg --no-input

echo [INFO] Downloading MediaMTX...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$url = 'https://api.github.com/repos/bluenviron/mediamtx/releases/latest'; ^
     $release = Invoke-RestMethod -Uri $url -Headers @{'User-Agent'='fastalpr-setup'}; ^
     $asset = $release.assets | Where-Object { $_.name -like 'windows_amd64.zip' } | Select-Object -First 1; ^
     if ($asset) { ^
         Invoke-WebRequest -Uri $asset.browser_download_url -OutFile 'mediamtx.zip'; ^
         Expand-Archive -Force -Path 'mediamtx.zip' -DestinationPath 'bin'; ^
         Remove-Item 'mediamtx.zip'; ^
         Write-Host '[OK] MediaMTX extracted to bin/'; ^
     } else { ^
         Write-Warning '[WARN] MediaMTX windows_amd64.zip not found in release. Manually place mediamtx.exe in bin/'; ^
     }"

if not exist "bin\mediamtx.exe" (
    echo [WARN] MediaMTX not found in bin/. Please manually download from https://github.com/bluenviron/mediamtx/releases and place mediamtx.exe in bin/
) else (
    echo [OK] MediaMTX ready at bin\mediamtx.exe
)

rem 6. Env wizard + selftest
echo [5/5] Running env wizard...
%VENV_PY% tools/env_wizard.py

echo.
echo =================================================
echo === Setup Complete ===
echo =================================================
echo.
echo Next steps:
echo   1. Edit .env if needed (Neon creds, camera URLs)
echo   2. Run: .\stream.bat   (to start RTSP stream)
echo   3. Run: .\run.bat      (to start detection)
echo   4. Or: .\run.bat mode 2 (for Next.js service)
echo.
pause