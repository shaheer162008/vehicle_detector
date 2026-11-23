@echo off
setlocal
cd /d "%~dp0\.."

set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"

echo === Starting FastALPR Live Camera Stream Broadcaster ===
"%PYTHON%" stream\live_stream.py --video "stream\stream.mp4" --port 8554
pause
