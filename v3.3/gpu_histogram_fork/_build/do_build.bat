@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d "C:\Users\C\Documents\Savage22 Server\v3.3\gpu_histogram_fork\_build\LightGBM"
if exist build5 rmdir /s /q build5
mkdir build5
cd build5
echo === CONFIGURING ===
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA_SPARSE=ON -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe" -DCMAKE_CUDA_HOST_COMPILER=cl.exe -DCMAKE_CUDA_FLAGS="--allow-unsupported-compiler" -DCMAKE_CXX_FLAGS="/W0 /wd4293 /wd4244" -DCMAKE_C_FLAGS="/W0" 2>&1
echo === CMAKE EXIT: %ERRORLEVEL% ===
if %ERRORLEVEL% NEQ 0 goto :eof
echo === BUILDING ===
cmake --build . --config Release 2>&1
echo === BUILD EXIT: %ERRORLEVEL% ===
echo === DONE ===
