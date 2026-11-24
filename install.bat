@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo === FastALPR + Vehicle Tracking Fast Installer ===

rem 1. Select Python version (Prefer 3.11 or 3.12, fallback to 3.13 or system python)
set "PY_CMD="

py -3.12 --version >nul 2>nul
if %errorlevel% equ 0 (
    set "PY_CMD=py -3.12"
    goto found_python
)

py -3.11 --version >nul 2>nul
if %errorlevel% equ 0 (
    set "PY_CMD=py -3.11"
    goto found_python
)

py -3.13 --version >nul 2>nul
if %errorlevel% equ 0 (
    set "PY_CMD=py -3.13"
    goto found_python
)

python --version >nul 2>nul
if %errorlevel% equ 0 (
    set "PY_CMD=python"
    goto found_python
)

echo [ERROR] No suitable Python installation found. Please install Python 3.11 or 3.12 (or 3.13).
exit /b 1

:found_python
echo [INFO] Creating virtual environment using: %PY_CMD%
if exist ".venv" (
    echo [INFO] Re-creating .venv...
    rmdir /s /q ".venv" >nul 2>&1
)

%PY_CMD% -m venv .venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment (.venv).
    exit /b 1
)

set "VENV_PY=.venv\Scripts\python.exe"

echo [1/4] Upgrading pip...
"%VENV_PY%" -m pip install --upgrade pip --no-input

echo [2/4] Installing requirements.txt with pre-built binary wheels...
"%VENV_PY%" -m pip install -r requirements.txt --no-input --prefer-binary

echo [3/4] Ensuring clean GPU builds (uninstalling CPU onnxruntime, installing cu128 torch + onnxruntime-gpu)...
"%VENV_PY%" -m pip uninstall -y onnxruntime >nul 2>&1
"%VENV_PY%" -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128 --no-input
"%VENV_PY%" -m pip install "onnxruntime-gpu[cuda,cudnn]==1.22.0" --no-input

echo [4/4] Verifying GPU acceleration with check_gpu.py...
"%VENV_PY%" tools\check_gpu.py
if %errorlevel% neq 0 (
    echo.
    echo =================================================
    echo INSTALL FAILED: GPU not usable
    echo =================================================
    exit /b 1
)

echo.
echo =================================================
echo === Installation Completed Successfully! ===
echo =================================================
endlocal
