@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================
echo  Ideogram 4 Captioner - Portable Installer
echo ============================================
echo.

set "PYTHON_VERSION=3.12.7"
set "PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-embed-amd64.zip"
set "PYTHON_DIR=python"

if exist "%PYTHON_DIR%\python.exe" (
    echo [OK] Python already installed in '%PYTHON_DIR%\'
    goto :ensure_pip
)

echo [1/5] Downloading Python %PYTHON_VERSION% embedded...
powershell -NoProfile -Command "& {try { Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile 'python.zip' -UseBasicParsing } catch { exit 1 }}"

if not exist "python.zip" (
    echo [ERROR] Download failed. Check your internet connection.
    pause
    exit /b 1
)

echo [2/5] Extracting Python to '%PYTHON_DIR%\'...
powershell -NoProfile -Command "& {Expand-Archive -Path 'python.zip' -DestinationPath '%PYTHON_DIR%' -Force}"
del python.zip

if not exist "%PYTHON_DIR%\python.exe" (
    echo [ERROR] Extraction failed.
    pause
    exit /b 1
)

echo [3/5] Configuring embedded Python (enabling site-packages)...
rem CRITICAL: Embedded Python ships with #import site (commented out).
rem We rewrite the _pth file to enable site-packages properly.
for %%f in ("%PYTHON_DIR%\python*._pth") do (
    echo Rewriting: %%f
    > "%%f" (
        echo python312.zip
        echo .
        echo Lib\site-packages
        echo.
        echo import site
    )
)

:ensure_pip
echo [4/5] Ensuring pip is installed...
"%PYTHON_DIR%\python.exe" -m pip --version >nul 2>&1
if errorlevel 1 (
    echo Pip not found, installing via get-pip.py...
    powershell -NoProfile -Command "& {Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'get-pip.py' -UseBasicParsing}"
    if not exist "get-pip.py" (
        echo [ERROR] Could not download get-pip.py
        pause
        exit /b 1
    )
    "%PYTHON_DIR%\python.exe" get-pip.py --no-warn-script-location
    del get-pip.py

    "%PYTHON_DIR%\python.exe" -m pip --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Pip installation failed even after get-pip.py.
        echo Likely cause: _pth file not configured correctly.
        echo Try deleting the 'python' folder and re-running install.bat.
        pause
        exit /b 1
    )
) else (
    echo [OK] Pip already installed.
)

echo.
echo [5/5] Installing Python dependencies...
"%PYTHON_DIR%\python.exe" -m pip install --upgrade pip --no-warn-script-location
"%PYTHON_DIR%\python.exe" -m pip install -r requirements.txt --no-warn-script-location

if errorlevel 1 (
    echo.
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Installation complete!
echo ============================================
echo.
echo Next steps:
echo   1. Start LM Studio and load a vision model
echo      (Qwen 3.6 27B Dense or Gemma 4 31B recommended)
echo   2. Start LM Studio's local server (default: localhost:1234)
echo   3. Double-click 'run.bat' to launch the captioner
echo.
pause
