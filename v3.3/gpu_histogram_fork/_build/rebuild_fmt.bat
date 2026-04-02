@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d "C:\Users\C\Documents\Savage22 Server\v3.3\gpu_histogram_fork\_build\LightGBM\build_utf8"
echo === REBUILDING ===
ninja 2>&1
echo === BUILD EXIT: %ERRORLEVEL% ===
if %ERRORLEVEL% EQU 0 (
    echo === BUILD SUCCESS ===
    dir /b *.dll *.exe *.lib 2>/dev/null
)
