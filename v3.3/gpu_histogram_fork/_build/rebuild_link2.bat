@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d "C:\Users\C\Documents\Savage22 Server\v3.3\gpu_histogram_fork\_build\LightGBM"
if exist build_link rmdir /s /q build_link
mkdir build_link
cd build_link
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA_SPARSE=ON -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe" -DCMAKE_CUDA_HOST_COMPILER=cl.exe -DCMAKE_CUDA_FLAGS="--allow-unsupported-compiler -Xcompiler=/utf-8" 2>&1 | tail -5
echo === CMAKE EXIT: %ERRORLEVEL% ===
if %ERRORLEVEL% NEQ 0 goto :eof
cmake --build . --config Release 2>&1 | tail -20
echo === BUILD EXIT: %ERRORLEVEL% ===
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo   BUILD SUCCESS
    echo ========================================
    dir /b *.dll *.exe 2>/dev/null
)
