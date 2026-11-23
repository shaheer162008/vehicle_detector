@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ==============================================================================
echo  === FastALPR Vehicle Tracking Runner (Turnkey Stream + Detection) ===
echo ==============================================================================

rem 1. Check & Setup Virtual Environment
if not exist ".venv\Scripts\python.exe" (
    echo [WARNING] Virtual environment ^(.venv^) not found!
    echo Running install.bat to setup your environment first...
    call install.bat
    if errorlevel 1 (
        echo [ERROR] Installation failed. Cannot continue.
        pause
        exit /b 1
    )
)

set "PYTHON=.venv\Scripts\python.exe"

rem 2. Detect Video File
set "STREAM_VIDEO="
if not "%~1"=="" (
    set "STREAM_VIDEO=%~1"
) else (
    if exist "stream\stream.mp4" (
        set "STREAM_VIDEO=stream\stream.mp4"
    ) else if exist "stream\stream_2.mp4" (
        set "STREAM_VIDEO=stream\stream_2.mp4"
    ) else (
        echo [ERROR] No video found in stream\ folder!
        echo Please place a video as stream\stream.mp4 or pass its path as an argument.
        pause
        exit /b 1
    )
)

rem 3. Ask User for Number Plate Format Selection (Default: PAKISTAN)
echo.
echo ==============================================================================
echo  Select License Plate Format:
echo   [1] PAKISTAN  (Default - Punjab, Sindh, Islamabad, KPK, etc.)
echo   [2] GENERAL   (Universal Alphanumeric Standard)
echo   [3] UK        (British Standard Format)
echo ==============================================================================
set "PLATE_CHOICE="
set /p "PLATE_CHOICE=Enter selection [1-3, Default=1]: "

if "!PLATE_CHOICE!"=="2" (
    set "SELECTED_PLATE_FORMAT=GENERAL"
) else if "!PLATE_CHOICE!"=="3" (
    set "SELECTED_PLATE_FORMAT=UK"
) else (
    set "SELECTED_PLATE_FORMAT=PAKISTAN"
)

set "STREAM_PORT=8554"

rem 4. Inspect Video & Auto-Configure Environment Variables in .env
echo.
echo [INSPECTING] Checking stream video properties...
"%PYTHON%" stream\inspect_stream.py "!STREAM_VIDEO!" "!SELECTED_PLATE_FORMAT!" "!STREAM_PORT!"

rem Load updated .env
if exist ".env" (
    for /f "usebackq tokens=1* delims==" %%A in (".env") do (
        set "k=%%A"
        set "v=%%B"
        if not "!k:~0,1!"=="#" (
            set "!k!=!v!"
        )
    )
)

rem 5. Detect Active Compute Device (GPU Preferred)
for /f "delims=" %%D in ('"%PYTHON%" -c "import torch; print(('GPU: ' + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'CPU (No CUDA detected)')"') do (
    set "COMPUTE_DEVICE=%%D"
)
echo [COMPUTE ENGINE] %COMPUTE_DEVICE%

rem 6. Launch Live Camera Stream Broadcaster in Background
echo.
echo [1/2] Starting Live Camera Stream Broadcaster on port %STREAM_PORT%...
start "FastALPR_StreamBroadcaster" /min "%PYTHON%" stream\live_stream.py --video "!STREAM_VIDEO!" --port !STREAM_PORT!

rem Wait 2 seconds for server to start
timeout /t 2 /nobreak >nul

rem 7. Launch ALPR Tracking Pipeline (Real-Time Window + Disk Save)
echo [2/2] Launching FastALPR Pipeline on live camera stream...
echo [INFO] Live preview window will open. Press 'q' or ESC in window to exit.
echo [INFO] Output video is simultaneously recorded to: %OUTPUT_VIDEO%
echo.

"%PYTHON%" run.py --source "!STREAM_URL!" --plate-format "!PLATE_FORMAT!" --output-video "!OUTPUT_VIDEO!" --no-preview

set "RUN_EXIT_CODE=%errorlevel%"

rem 8. Terminate Background Stream Broadcaster Cleanly
echo.
echo [CLEANUP] Stopping Live Camera Stream Broadcaster...
taskkill /F /FI "WINDOWTITLE eq FastALPR_StreamBroadcaster*" >nul 2>&1

echo.
echo ==============================================================================
echo  === Session Finished Successfully ===
echo  Output Video : %OUTPUT_VIDEO%
echo  Snapshots    : output\snapshots\
echo  Plate Crops  : output\plate_crops\
echo  Events JSON  : output\events.json
echo ==============================================================================
exit /b %RUN_EXIT_CODE%
