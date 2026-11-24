@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem Load .env if present
if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        set "line=%%A"
        if not "!line:~0,1!"=="#" (
            if not "%%B"=="" set "%%A=%%B"
        )
    )
)

set "OUTPUT_VIDEO=output\annotated_output.mp4"
set "DRIVE_FILE_ID=1szprqeBN3eQ5vBcRJ8HVIowMu514F94L"
set "VIDEO_DIR=videos"

echo =================================================
echo === FastALPR Vehicle Tracking Run Script ===
echo =================================================

rem 1. Python environment detection
set "PYTHON="
if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        set "PYTHON=python"
    ) else (
        echo [ERROR] Python not found. Run install.bat first.
        exit /b 1
    )
)

rem 2. Check GPU acceleration (fail early, never run silently on CPU)
echo [CHECK] Verifying GPU hardware and execution providers...
"%PYTHON%" tools\check_gpu.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] GPU check failed. Execution halted to prevent slow CPU execution.
    exit /b 1
)
echo.

rem 3. Video Selection Menu or Argument Handling
set "VIDEO_PATH="
set "ARG1=%~1"

if "%ARG1%"=="1" (
    set "VIDEO_PATH=stream_2.mp4"
    goto video_selected
)
if "%ARG1%"=="2" (
    set "VIDEO_PATH=stream.mp4"
    goto video_selected
)
if "%ARG1%"=="3" (
    goto download_drive_video
)
if "%ARG1%"=="4" (
    if not "%~2"=="" (
        set "VIDEO_PATH=%~2"
        goto video_selected
    )
)
if not "%ARG1%"=="" (
    set "VIDEO_PATH=%ARG1%"
    goto video_selected
)

:menu_video
echo =================================================
echo             FAST-ALPR VIDEO SOURCE MENU
echo =================================================
echo   1) stream_2.mp4 (repo)
echo   2) stream.mp4 (repo)
echo   3) Google Drive video (ID: 1szprqeBN3eQ5vBcRJ8HVIowMu514F94L)
echo   4) Custom video path
echo =================================================
set /p "V_CHOICE=Choose video source [1-4] (default 1): "
if "%V_CHOICE%"=="" set "V_CHOICE=1"

if "%V_CHOICE%"=="1" (
    set "VIDEO_PATH=stream_2.mp4"
    goto video_selected
)
if "%V_CHOICE%"=="2" (
    set "VIDEO_PATH=stream.mp4"
    goto video_selected
)
if "%V_CHOICE%"=="3" (
    goto download_drive_video
)
if "%V_CHOICE%"=="4" (
    set /p "VIDEO_PATH=Enter custom video path: "
    goto video_selected
)

set "VIDEO_PATH=stream_2.mp4"
goto video_selected

:download_drive_video
if not exist "%VIDEO_DIR%" mkdir "%VIDEO_DIR%"
set "VIDEO_PATH=%VIDEO_DIR%\drive_input.mp4"
if not exist "%VIDEO_PATH%" (
    echo [INFO] Downloading Google Drive video (%DRIVE_FILE_ID%)...
    "%PYTHON%" -c "import gdown; gdown.download(id='%DRIVE_FILE_ID%', output=r'%VIDEO_PATH%', quiet=False)"
    if %errorlevel% neq 0 (
        echo [ERROR] Google Drive download failed.
        exit /b 1
    )
)
goto video_selected

:video_selected
if not exist "%VIDEO_PATH%" (
    echo [ERROR] Video file not found: %VIDEO_PATH%
    exit /b 1
)

rem 4. Plate Format Selection Menu
set "PLATE_FORMAT=GENERAL"
if "%ARG1%"=="1" if not "%~2"=="" set "PLATE_FORMAT=%~2" & goto run_pipeline
if "%ARG1%"=="2" if not "%~2"=="" set "PLATE_FORMAT=%~2" & goto run_pipeline
if "%ARG1%"=="3" if not "%~2"=="" set "PLATE_FORMAT=%~2" & goto run_pipeline
if "%ARG1%"=="4" if not "%~3"=="" set "PLATE_FORMAT=%~3" & goto run_pipeline

echo.
echo =================================================
echo              PLATE FORMAT MENU
echo =================================================
echo   1) GENERAL (Default)
echo   2) UK
echo   3) PAKISTAN
echo =================================================
set /p "F_CHOICE=Choose plate format [1-3] (default 1): "
if "%F_CHOICE%"=="" set "F_CHOICE=1"

if "%F_CHOICE%"=="1" set "PLATE_FORMAT=GENERAL"
if "%F_CHOICE%"=="2" set "PLATE_FORMAT=UK"
if "%F_CHOICE%"=="3" set "PLATE_FORMAT=PAKISTAN"

:run_pipeline
echo.
echo [RUN] Source: %VIDEO_PATH%
echo [RUN] Format: %PLATE_FORMAT%
echo [RUN] Output Video: %OUTPUT_VIDEO%
echo.

"%PYTHON%" run.py --source "%VIDEO_PATH%" --output-video "%OUTPUT_VIDEO%" --plate-format "%PLATE_FORMAT%" --preview
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Pipeline execution failed.
    exit /b 1
)

echo.
echo =================================================
echo === Run Completed Successfully ===
echo Annotated Video: %OUTPUT_VIDEO%
echo Events JSON    : output\events.json
echo Snapshots      : output\snapshots
echo =================================================
endlocal
