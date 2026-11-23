@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ==============================================================================
echo  === FastALPR Vehicle Tracking Runner (Turnkey Stream + Detection) ===
echo ==============================================================================

rem 0. Load .env file if present
if exist ".env" (
    echo [CONFIG] Loading environment variables from .env...
    for /f "usebackq tokens=1* delims==" %%A in (".env") do (
        set "key=%%A"
        set "val=%%B"
        if not "!key:~0,1!"=="#" (
            set "!key!=!val!"
        )
    )
)

rem 1. Set Environment Variable Defaults
if "%STREAM_PORT%"=="" set "STREAM_PORT=8554"
if "%STREAM_VIDEO%"=="" set "STREAM_VIDEO=stream\stream.mp4"
if "%STREAM_URL%"=="" set "STREAM_URL=http://127.0.0.1:%STREAM_PORT%/live"
if "%PLATE_FORMAT%"=="" set "PLATE_FORMAT=PAKISTAN"
if "%OUTPUT_VIDEO%"=="" set "OUTPUT_VIDEO=output\annotated_output.mp4"

rem Allow override via CLI argument 1
if not "%~1"=="" set "STREAM_VIDEO=%~1"

rem 2. Verify Virtual Environment
if not exist ".venv\Scripts\python.exe" (
    echo [WARNING] Virtual environment ^(.venv^) not found!
    echo Running install.bat to setup your GPU environment first...
    call install.bat
    if errorlevel 1 (
        echo [ERROR] Installation failed. Cannot continue.
        pause
        exit /b 1
    )
)

set "PYTHON=.venv\Scripts\python.exe"

rem 3. Detect Active Compute Device (GPU Preferred)
for /f "delims=" %%D in ('"%PYTHON%" -c "import torch; print(('GPU: ' + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'CPU (No CUDA detected)')"') do (
    set "COMPUTE_DEVICE=%%D"
)
echo [COMPUTE ENGINE] %COMPUTE_DEVICE%
echo [STREAM SOURCE]  %STREAM_VIDEO%
echo [STREAM URL]     %STREAM_URL%
echo [PLATE FORMAT]   %PLATE_FORMAT%

rem 4. Check if stream video exists
if not exist "%STREAM_VIDEO%" (
    echo [ERROR] Stream video not found at: %STREAM_VIDEO%
    echo Please make sure stream\stream.mp4 exists or pass your video path as an argument.
    pause
    exit /b 1
)

rem 5. Launch Live Stream Broadcaster in Background
echo.
echo [1/2] Starting Live Camera Stream Broadcaster on port %STREAM_PORT%...
start "FastALPR_StreamBroadcaster" /min "%PYTHON%" stream\live_stream.py --video "%STREAM_VIDEO%" --port %STREAM_PORT%

rem Wait 2 seconds for server to bind port
timeout /t 2 /nobreak >nul

rem 6. Launch ALPR Tracking Pipeline on the Live Stream
echo [2/2] Launching FastALPR Pipeline on live camera stream...
echo [INFO] Press 'q' or ESC in preview window to exit.
echo.

"%PYTHON%" run.py --source "%STREAM_URL%" --plate-format "%PLATE_FORMAT%" --output-video "%OUTPUT_VIDEO%" --preview

set "RUN_EXIT_CODE=%errorlevel%"

rem 7. Terminate the Background Stream Broadcaster Cleanly
echo.
echo [CLEANUP] Stopping Live Camera Stream Broadcaster...
taskkill /F /FI "WINDOWTITLE eq FastALPR_StreamBroadcaster*" >nul 2>&1

echo.
echo ==============================================================================
echo  === Session Finished Successfully ===
echo  Output Video : %OUTPUT_VIDEO%
echo  Snapshots    : output\snapshots\
echo  Events JSON  : output\events.json
echo ==============================================================================
exit /b %RUN_EXIT_CODE%
