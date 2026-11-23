@echo off
setlocal
cd /d "%~dp0"

set "DRIVE_FILE_ID=1szprqeBN3eQ5vBcRJ8HVIowMu514F94L"
set "VIDEO_DIR=videos"
set "VIDEO_PATH=videos\drive_input.mp4"
set "OUTPUT_VIDEO=output\annotated_output.mp4"

echo === FastALPR Vehicle Tracking Run Script ===

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found in PATH. Install Python 3.10+ or run install.bat first.
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] CUDA environment not found. Run install.bat first.
    exit /b 1
)

set "PYTHON=.venv\Scripts\python.exe"

"%PYTHON%" -c "import cv2, torch, ultralytics, fast_alpr, onnxruntime" >nul 2>nul
if errorlevel 1 (
    echo [WARNING] Required CUDA packages look missing. Run install.bat once, then re-run this script.
)

"%PYTHON%" -c "import gdown" >nul 2>nul
if errorlevel 1 (
    echo [1/4] Installing gdown ^(Google Drive downloader^)...
    "%PYTHON%" -m pip install gdown --no-input --quiet
)

if not "%~1"=="" goto custom_input

if not exist "%VIDEO_DIR%" mkdir "%VIDEO_DIR%"
if exist "%VIDEO_PATH%" goto input_ready

echo [2/4] Downloading video from Google Drive...
"%PYTHON%" -c "import gdown; gdown.download(id='%DRIVE_FILE_ID%', output=r'%VIDEO_PATH%', quiet=False)"
if errorlevel 1 (
    echo [ERROR] Google Drive download failed.
    echo         Make sure the file sharing is set to "Anyone with the link".
    exit /b 1
)
goto input_ready

:custom_input
set "VIDEO_PATH=%~1"
echo [2/4] Using local video: %VIDEO_PATH%
goto input_ready

:input_ready
if not exist "%VIDEO_PATH%" (
    echo [ERROR] Input video not found: %VIDEO_PATH%
    exit /b 1
)
for %%A in ("%VIDEO_PATH%") do set "VIDEO_SIZE=%%~zA"
if "%VIDEO_SIZE%"=="0" (
    echo [ERROR] Input video file is empty ^(0 bytes^): %VIDEO_PATH%
    echo         Delete it and re-run to download again.
    exit /b 1
)

echo [3/4] Running ALPR tracking pipeline on: %VIDEO_PATH%
echo.
"%PYTHON%" -c "import cv2; c=cv2.VideoCapture(r'%VIDEO_PATH%'); print(int(c.get(cv2.CAP_PROP_FRAME_COUNT)/2)); c.release()" > .start_frame.tmp
set /p START_FRAME=<.start_frame.tmp
del /q .start_frame.tmp
if "%START_FRAME%"=="" set "START_FRAME=0"
echo [3/4] Starting preview from frame %START_FRAME%
"%PYTHON%" run.py --source "%VIDEO_PATH%" --output-video "%OUTPUT_VIDEO%" --plate-format PAKISTAN --preview --no-save-video --start-frame %START_FRAME%
if errorlevel 1 (
    echo.
    echo [ERROR] Pipeline run failed. See messages above.
    exit /b 1
)

echo.
echo === Preview session finished ===
echo Snapshots     : output\snapshots
echo Events JSON   : output\events.json
endlocal
