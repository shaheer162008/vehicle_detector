@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo === FastALPR Runner ===

rem Check for .venv
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Run setup.bat first.
    exit /b 1
)

set "VENV_PY=.venv\Scripts\python.exe"

rem Run selftest first
echo.
echo Running self-test...
%VENV_PY% tools/selftest.py
if %errorlevel% neq 0 (
    echo.
    echo [WARN] Self-test had failures.
    set /p "CONTINUE=Continue anyway? [Y/n]: "
    if /i "%CONTINUE%" neq "Y" if /i "%CONTINUE%" neq "YES" (
        echo Aborted.
        exit /b 1
    )
)

echo.
echo Choose mode:
echo   [1] Detection -> Neon (CLI mode, python run.py)
echo   [2] Service for Next.js frontend (uvicorn service.app:app)
echo   [3] Self-test only
echo.
set /p "MODE=Select [1-3]: "

if "%MODE%"=="3" (
    echo Self-test already done. Exiting.
    exit /b 0
)

if "%MODE%"=="2" (
    rem Service mode
    echo.
    echo Starting FastAPI service for Next.js frontend...
    echo.
    echo Service will run on http://127.0.0.1:8000
    echo API docs: http://127.0.0.1:8000/docs
    echo WebSocket events: ws://127.0.0.1:8000/ws/events?token=***
    echo WebSocket frames: ws://127.0.0.1:8000/ws/frames/{stream_id}?token=***
    echo.
    echo Press Ctrl+C to stop.
    echo.
    %VENV_PY% -m uvicorn service.app:app --host 127.0.0.1 --port 8000
    exit /b 0
)

rem Mode 1: CLI detection
echo.
echo CLI Detection Mode
echo.

rem Check camera 1 RTSP URL
%VENV_PY% -c "
import os
from dotenv import load_dotenv
load_dotenv('.env', override=True)
rtsp = os.environ.get('CAMERA_1_RTSP_URL', '')
if rtsp:
    print('[INFO] CAMERA_1_RTSP_URL:', rtsp)
    if rtsp.startswith('rtsp://127.0.0.1:8554'):
        print('[HINT] This is a local stream. Make sure stream.bat is running!')
else:
    print('[WARN] CAMERA_1_RTSP_URL not set in .env')
    print('[HINT] Run stream.bat first to create local RTSP stream')
"

echo.
set /p "MINUTES=Run for how many minutes? (Enter = unlimited): "
set "MAX_MINUTES="
if "%MINUTES%" neq "" set "MAX_MINUTES=--max-minutes %MINUTES%"

rem Check if preview wanted (Windows desktop)
set "PREVIEW_FLAG=--no-preview"
if not defined COLAB_GPU if not defined COLAB_RELEASE_TAG (
    set /p "PREVIEW=Show preview window? [y/N]: "
    if /i "%PREVIEW%"=="Y" if /i "%PREVIEW%"=="YES" (
        set "PREVIEW_FLAG="
    )
)

rem Determine plate format from .env
%VENV_PY% -c "
import os
from dotenv import load_dotenv
load_dotenv('.env', override=True)
fmt = os.environ.get('PLATE_FORMAT', 'GENERAL')
print('PLATE_FORMAT=' + fmt)
" > .tmp_fmt.txt
set /p "PLATE_FMT=" < .tmp_fmt.txt
del .tmp_fmt.txt

rem Run detection
echo.
echo Starting detection...
echo Command: %VENV_PY% run.py --source "%CAMERA_1_RTSP_URL%" --plate-format %PLATE_FMT:PLATE_FORMAT=% %MAX_MINUTES% %PREVIEW_FLAG%
echo.

%VENV_PY% run.py --source "%CAMERA_1_RTSP_URL%" --plate-format %PLATE_FMT:PLATE_FORMAT=% %MAX_MINUTES% %PREVIEW_FLAG%

echo.
echo Detection finished.
set /p "STOP_STREAM=Stop stream processes too? [Y/n]: "
if /i "%STOP_STREAM%" neq "N" if /i "%STOP_STREAM%" neq "NO" (
    call stop.bat
)