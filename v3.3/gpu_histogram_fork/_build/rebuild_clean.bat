@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d "C:\Users\C\Documents\Savage22 Server\v3.3\gpu_histogram_fork\_build\LightGBM"
if exist build_final rmdir /s /q build_final
mkdir build_final
cd build_final
echo === CLEAN CONFIGURE ===
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA_SPARSE=ON -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe" -DCMAKE_CUDA_HOST_COMPILER=cl.exe -DCMAKE_CUDA_FLAGS="--allow-unsupported-compiler" 2>&1
echo === CMAKE EXIT: %ERRORLEVEL% ===
if %ERRORLEVEL% NEQ 0 goto :eof
echo === BUILDING ===
cmake --build . --config Release 2>&1
echo === BUILD EXIT: %ERRORLEVEL% ===
if %ERRORLEVEL% EQU 0 (
    echo === SUCCESS ===
    dir /b *.dll *.exe *.lib 2>/dev/null
)
