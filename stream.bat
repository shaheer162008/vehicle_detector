@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo === FastALPR Stream Starter ===

rem Check for .venv
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Run setup.bat first.
    exit /b 1
)

set "VENV_PY=.venv\Scripts\python.exe"
set "MEDIAMTX=bin\mediamtx.exe"
set "FFMPEG=ffmpeg"

rem Check MediaMTX
if not exist "%MEDIAMTX%" (
    echo [ERROR] MediaMTX not found at %MEDIAMTX%. Run setup.bat first.
    exit /b 1
)

rem Check ffmpeg
where %FFMPEG% >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] ffmpeg not in PATH. Install from https://ffmpeg.org or run setup.bat
    exit /b 1
)

rem 1. Choose source
echo.
echo Choose video source:
echo   [1] stream_2.mp4 (demo)
echo   [2] stream.mp4 (demo)
echo   [3] Custom file path
echo   [4] Real camera RTSP URL (type manually)
echo   [Enter] Skip - keep existing CAMERA_1_RTSP_URL in .env
echo.
set /p "CHOICE=Select [1-4, Enter]: "

set "SOURCE_FILE="
set "IS_RTSP=0"
set "CUSTOM_URL="

if "%CHOICE%"=="1" set "SOURCE_FILE=stream_2.mp4"
if "%CHOICE%"=="2" set "SOURCE_FILE=stream.mp4"
if "%CHOICE%"=="3" (
    set /p "SOURCE_FILE=Enter file path (relative to repo root): "
)
if "%CHOICE%"=="4" (
    set "IS_RTSP=1"
    echo Enter RTSP URL (hidden input - password won't show):
    powershell -Command "$url = Read-Host -AsSecureString | ConvertFrom-SecureString | ConvertTo-SecureString -AsPlainText -Force; Write-Host $url" 2>nul
    if errorlevel 1 (
        set /p "CUSTOM_URL=RTSP URL: "
    ) else (
        for /f "delims=" %%a in ('powershell -Command "$url = Read-Host -AsSecureString | ConvertFrom-SecureString | ConvertTo-SecureString -AsPlainText -Force; Write-Host $url"') do set "CUSTOM_URL=%%a"
    )
)

if "%CHOICE%"=="" (
    echo [INFO] Keeping existing CAMERA_1_RTSP_URL in .env
    goto :update_env_only
)

if "%IS_RTSP%"=="1" (
    if "%CUSTOM_URL%"=="" (
        echo [ERROR] No RTSP URL entered
        exit /b 1
    )
    if not "%CUSTOM_URL%"=="rtsp://*" (
        echo [WARN] URL doesn't start with rtsp://
    )
) else (
    if not exist "%SOURCE_FILE%" (
        echo [ERROR] File not found: %SOURCE_FILE%
        exit /b 1
    )
)

rem 2. Start MediaMTX + ffmpeg in NEW WINDOW
echo.
echo Starting RTSP stream...

if "%IS_RTSP%"=="1" (
    rem Real camera - just update .env, don't start local stream
    goto :update_env_only
)

rem Demo file - start MediaMTX + ffmpeg loop
if not exist "%SOURCE_FILE%" (
    echo [ERROR] Source file not found: %SOURCE_FILE%
    exit /b 1
)

rem Start MediaMTX in new window
start "FastALPR MediaMTX" cmd /k "%MEDIAMTX%"

rem Wait for MediaMTX to be ready (port 8554)
echo Waiting for MediaMTX on port 8554...
for /l %%i in (1,1,30) do (
    powershell -Command "try { $tcp = New-Object System.Net.Sockets.TcpClient; $tcp.Connect('127.0.0.1', 8554); $tcp.Close(); exit 0 } catch { exit 1 }" >nul 2>&1
    if !errorlevel! equ 0 (
        goto :mtx_ready
    )
    timeout /t 1 /nobreak >nul
)
echo [ERROR] MediaMTX failed to start on port 8554
exit /b 1

:mtx_ready
echo MediaMTX ready.

rem Start ffmpeg in new window (loop file -> RTSP)
start "FastALPR ffmpeg" cmd /k "%FFMPEG% -hide_banner -loglevel warning -re -stream_loop -1 -i %SOURCE_FILE% -c copy -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/cam1"

rem 3. Update .env with CAMERA_1_RTSP_URL
:update_env_only
%VENV_PY% -c "
import os, re
env_path = '.env'
rtsp_url = 'rtsp://127.0.0.1:8554/cam1'
if os.path.exists(env_path):
    with open(env_path, 'r') as f: lines = f.readlines()
else:
    lines = []
new_lines = []
found = False
for line in lines:
    if line.strip().startswith('CAMERA_1_RTSP_URL'):
        new_lines.append('CAMERA_1_RTSP_URL=' + rtsp_url + '\n')
        found = True
    else:
        new_lines.append(line)
if not found:
    new_lines.append('CAMERA_1_RTSP_URL=' + rtsp_url + '\n')
with open(env_path + '.tmp', 'w') as f: f.writelines(new_lines)
os.replace(env_path + '.tmp', env_path)
print('[OK] .env updated: CAMERA_1_RTSP_URL=' + rtsp_url)
"

rem Save PIDs for stop script
echo %MEDIAMTX_PID% > .run\stream_pids.txt 2>nul
echo %FFMPEG_PID% >> .run\stream_pids.txt 2>nul

echo.
echo Stream started!
echo   RTSP URL: rtsp://127.0.0.1:8554/cam1
echo   HLS URL:  http://127.0.0.1:8888/cam1
echo   WebRTC:   http://127.0.0.1:8889/cam1
echo.
echo .env updated with CAMERA_1_RTSP_URL
echo.
echo To stop: run stop.bat
pause