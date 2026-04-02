@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d "C:\Users\C\Documents\Savage22 Server\v3.3\gpu_histogram_fork\_build\LightGBM"
if exist build_utf8 rmdir /s /q build_utf8
mkdir build_utf8
cd build_utf8
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA_SPARSE=ON -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe" -DCMAKE_CUDA_HOST_COMPILER=cl.exe -DCMAKE_CUDA_FLAGS="--allow-unsupported-compiler -Xcompiler=/utf-8" 2>&1
if %ERRORLEVEL% NEQ 0 goto :eof
echo === BUILDING ===
cmake --build . --config Release 2>&1
echo === BUILD EXIT: %ERRORLEVEL% ===
if %ERRORLEVEL% EQU 0 (
    echo === BUILD SUCCESS ===
    dir /b *.dll *.exe *.lib 2>/dev/null
)
