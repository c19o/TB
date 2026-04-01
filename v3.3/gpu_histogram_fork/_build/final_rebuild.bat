@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d "C:\Users\C\Documents\Savage22 Server\v3.3\gpu_histogram_fork\_build\LightGBM\build_link"
echo === FINAL REBUILD (all fixes) ===
ninja 2>&1
echo === BUILD EXIT: %ERRORLEVEL% ===
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========= BUILD SUCCESS =========
    copy /Y lib_lightgbm.dll ..\python-package\lib_lightgbm.dll >/dev/null
    copy /Y lib_lightgbm.dll ..\python-package\lightgbm\lib\lib_lightgbm.dll >/dev/null 2>/dev/null
    echo DLL copied
)
