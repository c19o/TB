# Building the GPU Histogram Fork on Windows 11

Complete build instructions for the LightGBM CUDA sparse histogram co-processor
on Windows 11 with RTX 3090 (sm_86).

Target: `gpu_histogram.dll` loaded by Python via ctypes.

---

## Prerequisites

### 1. CUDA Toolkit 12.6

Download: https://developer.nvidia.com/cuda-12-6-0-download-archive

- Select: Windows > x86_64 > 11 > exe (local)
- Run the installer, choose **Custom** install
- Required components:
  - CUDA > Development > Compiler > nvcc
  - CUDA > Development > Libraries > cudart
  - CUDA > Development > Headers
  - CUDA > Runtime
  - CUDA > Visual Studio Integration (critical for CMake detection)
- Install path: default `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6`

Why 12.6: supports sm_80 through sm_90 (all GPUs we use). sm_100 (B200) needs
12.8+ but we build with PTX fallback so it JIT-compiles on newer GPUs.

After install, verify in a **new** terminal:

```bash
nvcc --version
# Should show: release 12.6
```

If `nvcc` is not found, add to PATH manually:

```bash
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin:$PATH"
```

### 2. Visual Studio Build Tools 2022

Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/

Run the installer, select **Desktop development with C++**. Required components:

- MSVC v143 - VS 2022 C++ x64/x86 build tools (Latest)
- Windows 11 SDK (10.0.22621.0 or later)
- C++ CMake tools for Windows

Total install: ~6-8 GB. Only the build tools are needed, not the full IDE.

### 3. CMake 3.18+

CMake ships with Visual Studio Build Tools (if you selected "C++ CMake tools").
Verify:

```bash
cmake --version
# Should show 3.18 or higher (VS 2022 ships 3.26+)
```

If not found, install standalone from https://cmake.org/download/ and add to PATH.

### 4. Git

Already installed if you are using Git Bash. Verify:

```bash
git --version
```

### 5. Python 3.12 with Dependencies

```bash
pip install numpy scipy cupy-cuda12x pytest
```

CuPy is needed for the GPU tests (not for the C library build itself).

---

## Build Steps

All commands assume Git Bash. Run from the `gpu_histogram_fork` directory:

```
C:\Users\C\Documents\Savage22 Server\v3.3\gpu_histogram_fork
```

### Step 1: Verify CUDA and GPU

```bash
nvidia-smi
# Should show RTX 3090, Driver 5xx.xx, CUDA 12.x

nvcc --version
# Should show release 12.6
```

### Step 2: Create Build Directory

```bash
mkdir -p build
cd build
```

### Step 3: Configure with CMake

This must run from a terminal that has the Visual Studio environment loaded.
In Git Bash, CMake's `-G` flag handles this automatically:

```bash
cmake .. -G "Visual Studio 17 2022" -A x64 \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_FP64=ON \
    -DUSE_SHARED_MEM_TILING=ON
```

Expected output (verify these lines appear):

```
-- CUDA Toolkit version: 12.6.x
-- CUDA architectures: 80-real;86-real;89-real;90-real;90-virtual
-- Histogram accumulator type: double (FP64)
-- Shared memory tiling: ENABLED
```

**Troubleshooting CMake:**

| Error | Fix |
|-------|-----|
| `No CUDA toolset found` | Reinstall CUDA Toolkit with VS Integration checked |
| `No CMAKE_CUDA_COMPILER could be found` | Ensure `nvcc` is on PATH, restart terminal |
| `Generator "Visual Studio 17 2022" not found` | Install VS Build Tools 2022, not 2019 |
| `CUDAToolkit not found` | Set `CMAKE_CUDA_COMPILER` manually: `-DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe"` |

### Step 4: Build

```bash
cmake --build . --config Release
```

This compiles the fat binary CUDA kernel for sm_80/86/89/90 + PTX.
Build time: 2-5 minutes on a fast CPU (nvcc compiles each arch sequentially).

Expected output ends with:

```
Build succeeded.
    0 Warning(s)
    0 Error(s)
```

The output DLL is at: `build/Release/gpu_histogram.dll`

**Troubleshooting Build:**

| Error | Fix |
|-------|-----|
| `nvcc fatal: Unsupported gpu architecture 'compute_100'` | CUDA < 12.8 does not support sm_100. The CMakeLists.txt handles this automatically (only adds sm_100 if CUDA >= 12.8). If you see this, your CMake configure step failed silently. |
| `LINK : fatal error LNK1181: cannot open input file 'cudart.lib'` | CUDA lib path not found. Set: `-DCMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include"` |
| `cl.exe not found` | Open "x64 Native Tools Command Prompt for VS 2022" instead of plain cmd/bash, then run cmake from there |

### Step 5: Install the DLL

The CMakeLists.txt post-build step copies the DLL to `lib/`:

```bash
# Verify the DLL exists
ls -la ../lib/gpu_histogram.dll
# Should show ~2-10 MB file
```

If the post-build copy failed, do it manually:

```bash
mkdir -p ../lib
cp Release/gpu_histogram.dll ../lib/
```

### Step 6: Install Python Package

From the `gpu_histogram_fork` root (not the build dir):

```bash
cd ..
pip install -e .
```

This installs the `savage22-gpu-histogram` package in editable mode so the
Python wrapper (`src/gpu_histogram_wrapper.py`) can find the DLL.

### Step 7: Verify Python Import

```bash
python -c "
from src.gpu_histogram_wrapper import GPUHistogramBuilder, is_gpu_available
print('Import OK')
print(f'GPU available: {is_gpu_available()}')
"
```

Expected output:

```
Import OK
GPU available: True
```

**If `GPU available: False`:**

The wrapper searches for the library in this order:
1. `src/` directory (same dir as the .py file)
2. `lib/` directory (one level up from src/)
3. `sys.prefix/lib`
4. Directories in `LD_LIBRARY_PATH` (or `PATH` on Windows for DLL search)

On Windows, `ctypes.CDLL` also searches `PATH`. Fix:

```bash
# Option A: Copy DLL next to the wrapper
cp lib/gpu_histogram.dll src/

# Option B: Add lib/ to PATH
export PATH="$(pwd)/lib:$PATH"
```

**Note:** The wrapper currently looks for `libgpu_histogram.so` (Linux name).
On Windows, it must find `gpu_histogram.dll`. If the wrapper fails to load,
you may need to update `_LIB_NAME` in `src/gpu_histogram_wrapper.py`:

```python
# Change this line:
_LIB_NAME = "libgpu_histogram.so"
# To:
import platform
_LIB_NAME = "gpu_histogram.dll" if platform.system() == "Windows" else "libgpu_histogram.so"
```

### Step 8: Run Synthetic Test

```bash
pytest tests/ -v --tb=short
```

This runs the CPU vs GPU histogram equivalence tests with synthetic sparse
binary data. All tests should pass. The GPU tests are skipped if CUDA is
not available.

### Step 9: Run Real 1w Test

Use the actual 1w training data to verify the GPU histogram builder works
with production-scale sparse matrices:

```bash
python -c "
import numpy as np
import scipy.sparse as sp
from src.gpu_histogram_wrapper import GPUHistogramBuilder

# Load the 1w cross features (from a prior training run)
print('Loading 1w sparse matrix...')
npz = sp.load_npz('../v2_cross_names_BTC_1w.npz')  # adjust path
print(f'Shape: {npz.shape}, NNZ: {npz.nnz:,}')

# Initialize GPU builder
with GPUHistogramBuilder(npz, device_id=0) as gpu:
    used, total = gpu.get_vram_usage()
    print(f'VRAM: {used/1e9:.2f} GB / {total/1e9:.2f} GB')

    # Simulate one histogram build
    n_rows = npz.shape[0]
    grad = np.random.randn(n_rows).astype(np.float64)
    hess = np.abs(np.random.randn(n_rows)).astype(np.float64)
    gpu.update_gradients(grad, hess, num_class=1)

    # Build histogram for all rows (root node)
    row_idx = np.arange(n_rows, dtype=np.int32)
    hist = gpu.build_histogram(row_idx, class_id=0)
    print(f'Histogram shape: {hist.shape}')
    print(f'Non-zero grad bins: {np.count_nonzero(hist[:, 1, 0]):,}')
    print('SUCCESS: GPU histogram build completed')
"
```

---

## Windows-Specific Gotchas

### DLL Loading Order

Windows searches for DLLs in this order:
1. Directory of the calling executable (python.exe)
2. System directory (System32)
3. Windows directory
4. Current working directory
5. Directories in `PATH`

The `ctypes.CDLL` call may also need the CUDA runtime DLL (`cudart64_12.dll`).
If you get `OSError: [WinError 126] The specified module could not be found`,
the CUDA bin directory is not on PATH:

```bash
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin:$PATH"
```

### Visual Studio CUDA Integration

The CUDA Toolkit installer has a checkbox for "Visual Studio Integration" that
installs `.props` and `.targets` files into the VS directory. If you installed
CUDA **before** VS Build Tools, this integration is missing. Fix:

1. Uninstall CUDA Toolkit (Control Panel > Programs)
2. Install Visual Studio Build Tools 2022 first
3. Reinstall CUDA Toolkit with VS Integration checked

Alternatively, point CMake to nvcc explicitly:

```bash
cmake .. -G "Visual Studio 17 2022" -A x64 \
    -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe"
```

### Python DLL Dependency Chain

`gpu_histogram.dll` depends on `cudart64_12.dll` (CUDA runtime). Python must
be able to find both. If the CUDA bin dir is on PATH (it should be after CUDA
install), this works automatically. If not:

```python
# Add at the top of your script, before importing gpu_histogram_wrapper:
import os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")
```

### Long Path Issues

The project lives under `C:\Users\C\Documents\Savage22 Server\v3.3\gpu_histogram_fork`.
The space in "Savage22 Server" and the deep nesting can cause issues with some tools.
If CMake or nvcc fails with path errors:

```bash
# Option A: Use short path
cd /c/Users/C/Documents/"Savage22 Server"/v3.3/gpu_histogram_fork

# Option B: Create a junction (no spaces)
cmd //c "mklink /J C:\gpu_hist \"C:\Users\C\Documents\Savage22 Server\v3.3\gpu_histogram_fork\""
cd /c/gpu_hist
```

### Anti-Virus False Positives

Windows Defender may quarantine `gpu_histogram.dll` because it is a newly built
unsigned DLL that accesses GPU hardware. If the DLL disappears after build:

1. Open Windows Security > Virus & threat protection > Protection history
2. Find the quarantined file
3. Click "Allow on device"
4. Alternatively, add the `lib/` directory to the exclusion list

---

## Automated Build Script

The `build_windows.ps1` script in this directory automates steps 2-6.
Run from PowerShell (not Git Bash):

```powershell
.\build_windows.ps1
```

It will:
1. Verify prerequisites (CUDA, CMake, VS Build Tools)
2. Create build directory and run CMake configure
3. Build the Release DLL
4. Copy DLL to lib/
5. Install the Python package
6. Run the import verification test

---

## Build Matrix

| Component | Version | Notes |
|-----------|---------|-------|
| Windows | 11 Enterprise LTSC 2024 | 10.0.26100 |
| CUDA Toolkit | 12.6 | sm_80/86/89/90 + PTX |
| VS Build Tools | 2022 (v143) | MSVC + Windows SDK |
| CMake | 3.18+ (ships with VS) | Needs FindCUDAToolkit |
| Python | 3.12 | numpy, scipy, cupy-cuda12x |
| GPU | RTX 3090 24GB | sm_86, CC 8.6 |
| Driver | 535+ | CUDA 12.x compat |

## Output Artifacts

After a successful build:

```
gpu_histogram_fork/
  build/
    Release/
      gpu_histogram.dll      # Main build output
      gpu_histogram.lib      # Import library (for C++ linking)
  lib/
    gpu_histogram.dll        # Copied here by post-build step
```

The Python wrapper loads from `lib/gpu_histogram.dll`.
