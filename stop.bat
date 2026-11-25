@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo === FastALPR Stop ===

rem 1. Stop stream processes (ffmpeg + MediaMTX)
if exist ".run\stream_pids.txt" (
    echo Stopping stream processes...
    for /f "delims=" %%p in (.run\stream_pids.txt) do (
        echo Killing PID %%p...
        taskkill /F /PID %%p >nul 2>&1
    )
    del ".run\stream_pids.txt"
) else (
    echo No stream PID file found. Trying to kill by name...
    taskkill /F /IM ffmpeg.exe >nul 2>&1
    taskkill /F /IM mediamtx.exe >nul 2>&1
)

rem 2. Stop service (uvicorn)
echo Stopping service...
taskkill /F /FI "WINDOWTITLE eq FastALPR*" >nul 2>&1
wmic process where "commandline like '%uvicorn%service.app%'" delete >nul 2>&1

rem 3. Clean up .run directory
if exist ".run" (
    rmdir /s /q ".run" 2>nul
)

echo.
echo All processes stopped.