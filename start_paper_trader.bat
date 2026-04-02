@echo off
echo ============================================================
echo   SAVAGE22 PAPER TRADER
echo   Starting at %date% %time%
echo ============================================================
echo.

cd /d "C:\Users\C\Documents\Savage22 Server"

:: Check Python
python --version 2>nul
if errorlevel 1 (
    echo ERROR: Python not found in PATH!
    pause
    exit /b 1
)

:: Check if paper_trades.db exists (first run setup)
if not exist "paper_trades.db" (
    echo First run detected. Database will be created automatically.
    echo.
)

:: Run the paper trader
echo Starting paper trader...
echo Press Ctrl+C to stop.
echo.
python paper_trader.py

if errorlevel 1 (
    echo.
    echo Paper trader exited with error code %errorlevel%
    pause
)
