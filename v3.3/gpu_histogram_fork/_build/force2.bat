@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d "C:\Users\C\Documents\Savage22 Server\v3.3\gpu_histogram_fork\_build\LightGBM\build_link"
ninja 2>&1
echo EXIT: %ERRORLEVEL%
if %ERRORLEVEL% EQU 0 (
    copy /Y lib_lightgbm.dll "C:\Users\C\AppData\Local\Programs\Python\Python312\Lib\site-packages\lightgbm\bin\lib_lightgbm.dll" >/dev/null
    copy /Y lib_lightgbm.dll "C:\Users\C\AppData\Local\Programs\Python\Python312\Lib\site-packages\lightgbm\lib\lib_lightgbm.dll" >/dev/null
    echo DLL UPDATED
)
