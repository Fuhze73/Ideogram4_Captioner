@echo off
cd /d "%~dp0"

if not exist "python\python.exe" (
    echo [ERROR] Python not installed. Run 'install.bat' first.
    pause
    exit /b 1
)

echo Starting Ideogram 4 Captioner...
echo Open http://127.0.0.1:7860 in your browser if it doesn't auto-open.
echo.

python\python.exe Run_gui_gradio.py

pause
