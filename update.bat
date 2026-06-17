@echo off
cd /d "%~dp0"

if not exist "python\python.exe" (
    echo [ERROR] Python not installed. Run 'install.bat' first.
    pause
    exit /b 1
)

echo ============================================
echo  Updating dependencies
echo ============================================
echo.

python\python.exe -m pip install --upgrade pip --no-warn-script-location
python\python.exe -m pip install --upgrade -r requirements.txt --no-warn-script-location

if errorlevel 1 (
    echo.
    echo [ERROR] Update failed.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Update complete!
echo ============================================
pause
