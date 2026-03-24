@echo off
echo ============================================================
echo   SAVAGE22 V3.1 - PAPER TRADER
echo   Starting streamers + live trader
echo ============================================================
echo.

cd /d "%~dp0"

echo [1/2] Starting streamer supervisor...
start "Streamers" cmd /k "python streamer_supervisor.py"
timeout /t 10 >nul

echo [2/2] Starting paper trader...
start "Paper Trader" cmd /k "python -u live_trader.py --mode paper"

echo.
echo Both running in separate windows.
echo Kill switch: create file KILL_SWITCH in this folder to halt trading.
echo.
pause
