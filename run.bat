@echo off
rem ==============================================================================
rem FastALPR Vehicle Tracking - Run Script (Windows)
rem
rem What it does:
rem   1. Downloads the input video from Google Drive (skips if already downloaded)
rem   2. Runs the vehicle detection + plate tracking pipeline on that video
rem   3. Opens the annotated / tracked output video for playback
rem
rem Usage:
rem   run.bat                          <- download video from Google Drive and track it
rem   run.bat "D:\my_videos\car.mp4"   <- track a local video instead
rem ==============================================================================

setlocal
cd /d "%~dp0"

set "DRIVE_FILE_ID=1szprqeBN3eQ5vBcRJ8HVIowMu514F94L"
set "VIDEO_DIR=videos"
set "VIDEO_PATH=%VIDEO_DIR%\drive_input.mp4"
set "OUTPUT_VIDEO=output\annotated_output.mp4"

echo === FastALPR Vehicle Tracking Run Script ===

rem 0. Python check
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found in PATH. Install Python 3.10+ or run install.bat first.
    exit /b 1
)

rem 0b. Dependency check (install.bat installs these)
python -c "import cv2, ultralytics, fast_alpr" >nul 2>nul
if errorlevel 1 (
    echo [WARNING] Required packages look missing. Run install.bat once, then re-run this script.
)

rem 1. gdown is only needed for the Google Drive download
python -c "import gdown" >nul 2>nul
if errorlevel 1 (
    echo [1/4] Installing gdown ^(Google Drive downloader^)...
    python -m pip install gdown --no-input --quiet
)

rem 2. Pick the input video: custom local path if given, else Google Drive download
if not "%~1"=="" (
    set "VIDEO_PATH=%~1"
    echo [2/4] Using local video: %VIDEO_PATH%
    goto :check_input
)

if not exist "%VIDEO_DIR%" mkdir "%VIDEO_DIR%"
if exist "%VIDEO_PATH%" (
    echo [2/4] Video already present: %VIDEO_PATH%
) else (
    echo [2/4] Downloading video from Google Drive...
    python -c "import gdown; gdown.download(id='%DRIVE_FILE_ID%', output=r'%VIDEO_PATH%', quiet=False)"
    if errorlevel 1 (
        echo [ERROR] Google Drive download failed.
        echo         Make sure the file sharing is set to "Anyone with the link".
        exit /b 1
    )
)

:check_input
if not exist "%VIDEO_PATH%" (
    echo [ERROR] Input video not found: %VIDEO_PATH%
    exit /b 1
)
for %%A in ("%VIDEO_PATH%") do set "VIDEO_SIZE=%%~zA"
if "%VIDEO_SIZE%"=="0" (
    echo [ERROR] Input video file is empty (0 bytes): %VIDEO_PATH%
    echo         Delete it and re-run to download again.
    exit /b 1
)

rem 3. Run the tracking pipeline (annotated video is written by run.py)
echo [3/4] Running ALPR tracking pipeline on: %VIDEO_PATH%
echo.
python run.py --source "%VIDEO_PATH%" --output-video "%OUTPUT_VIDEO%"
if errorlevel 1 (
    echo.
    echo [ERROR] Pipeline run failed. See messages above.
    exit /b 1
)

rem 4. Play the tracked result
if exist "%OUTPUT_VIDEO%" (
    echo.
    echo [4/4] Opening tracked video: %OUTPUT_VIDEO%
    start "" "%OUTPUT_VIDEO%"
) else (
    echo [WARNING] Output video not found at: %OUTPUT_VIDEO%
)

echo.
echo === Finished ===
echo Tracked video : %OUTPUT_VIDEO%
echo Snapshots     : output\snapshots
echo Events JSON   : output\events.json
endlocal
