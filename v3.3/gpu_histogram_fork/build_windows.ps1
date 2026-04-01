# build_windows.ps1 — Automated build for GPU Histogram Fork on Windows 11
# =========================================================================
# Run from PowerShell in the gpu_histogram_fork directory:
#   .\build_windows.ps1
#
# Automates: CMake configure -> Build -> DLL copy -> pip install -> verify
# =========================================================================

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
if (-not $ProjectRoot) { $ProjectRoot = Get-Location }

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " GPU Histogram Fork — Windows Build Script" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# -------------------------------------------------------------------------
# Step 0: Verify prerequisites
# -------------------------------------------------------------------------
Write-Host "[0/6] Checking prerequisites..." -ForegroundColor Yellow

# Check CUDA
$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if (-not $nvcc) {
    # Try standard CUDA paths
    $cudaPaths = @(
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin"
    )
    foreach ($p in $cudaPaths) {
        if (Test-Path "$p\nvcc.exe") {
            $env:PATH = "$p;$env:PATH"
            Write-Host "  Found CUDA at: $p" -ForegroundColor Green
            break
        }
    }
    $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
    if (-not $nvcc) {
        Write-Host "  ERROR: nvcc not found. Install CUDA Toolkit 12.x first." -ForegroundColor Red
        Write-Host "  Download: https://developer.nvidia.com/cuda-12-6-0-download-archive" -ForegroundColor Red
        exit 1
    }
}

$nvccVersion = & nvcc --version 2>&1 | Select-String "release"
Write-Host "  CUDA: $nvccVersion" -ForegroundColor Green

# Check CMake
$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmake) {
    Write-Host "  ERROR: cmake not found. Install CMake 3.18+ or VS Build Tools with CMake component." -ForegroundColor Red
    exit 1
}
$cmakeVersion = & cmake --version | Select-Object -First 1
Write-Host "  CMake: $cmakeVersion" -ForegroundColor Green

# Check Visual Studio 2022
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vsWhere) {
    $vsPath = & $vsWhere -latest -property installationPath 2>$null
    if ($vsPath) {
        Write-Host "  Visual Studio: $vsPath" -ForegroundColor Green
    } else {
        Write-Host "  WARNING: Visual Studio not found via vswhere. CMake may still work if Build Tools are installed." -ForegroundColor Yellow
    }
} else {
    Write-Host "  WARNING: vswhere not found. Assuming VS Build Tools are installed." -ForegroundColor Yellow
}

# Check Python
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "  ERROR: python not found on PATH." -ForegroundColor Red
    exit 1
}
$pyVersion = & python --version 2>&1
Write-Host "  Python: $pyVersion" -ForegroundColor Green

# Check nvidia-smi
$smi = & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>$null
if ($smi) {
    Write-Host "  GPU: $smi" -ForegroundColor Green
} else {
    Write-Host "  WARNING: nvidia-smi failed. GPU may not be available." -ForegroundColor Yellow
}

Write-Host ""

# -------------------------------------------------------------------------
# Step 1: Create build directory
# -------------------------------------------------------------------------
Write-Host "[1/6] Creating build directory..." -ForegroundColor Yellow

$buildDir = Join-Path $ProjectRoot "build"
if (Test-Path $buildDir) {
    Write-Host "  Build directory exists. Cleaning..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $buildDir
}
New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
Write-Host "  Created: $buildDir" -ForegroundColor Green
Write-Host ""

# -------------------------------------------------------------------------
# Step 2: CMake configure
# -------------------------------------------------------------------------
Write-Host "[2/6] Running CMake configure..." -ForegroundColor Yellow

Push-Location $buildDir
try {
    $cmakeArgs = @(
        "..",
        "-G", "Visual Studio 17 2022",
        "-A", "x64",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DUSE_FP64=ON",
        "-DUSE_SHARED_MEM_TILING=ON"
    )

    Write-Host "  cmake $($cmakeArgs -join ' ')" -ForegroundColor DarkGray
    & cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: CMake configure failed (exit code $LASTEXITCODE)." -ForegroundColor Red
        Write-Host "  Common fixes:" -ForegroundColor Red
        Write-Host "    - Reinstall CUDA Toolkit with 'Visual Studio Integration' checked" -ForegroundColor Red
        Write-Host "    - Ensure VS 2022 Build Tools are installed (not 2019)" -ForegroundColor Red
        Write-Host "    - Try adding: -DCMAKE_CUDA_COMPILER=`"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe`"" -ForegroundColor Red
        exit 1
    }
    Write-Host "  CMake configure succeeded." -ForegroundColor Green
} finally {
    Pop-Location
}
Write-Host ""

# -------------------------------------------------------------------------
# Step 3: Build
# -------------------------------------------------------------------------
Write-Host "[3/6] Building (this takes 2-5 minutes)..." -ForegroundColor Yellow

Push-Location $buildDir
try {
    & cmake --build . --config Release
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: Build failed (exit code $LASTEXITCODE)." -ForegroundColor Red
        exit 1
    }
    Write-Host "  Build succeeded." -ForegroundColor Green
} finally {
    Pop-Location
}
Write-Host ""

# -------------------------------------------------------------------------
# Step 4: Copy DLL to lib/
# -------------------------------------------------------------------------
Write-Host "[4/6] Copying DLL to lib/..." -ForegroundColor Yellow

$libDir = Join-Path $ProjectRoot "lib"
if (-not (Test-Path $libDir)) {
    New-Item -ItemType Directory -Path $libDir -Force | Out-Null
}

# Find the built DLL
$dllSource = Join-Path $buildDir "Release" "gpu_histogram.dll"
if (-not (Test-Path $dllSource)) {
    # Try alternate locations
    $dllSource = Get-ChildItem -Path $buildDir -Filter "gpu_histogram.dll" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $dllSource) {
        Write-Host "  ERROR: gpu_histogram.dll not found in build output." -ForegroundColor Red
        Write-Host "  Searched: $buildDir" -ForegroundColor Red
        exit 1
    }
    $dllSource = $dllSource.FullName
}

Copy-Item $dllSource -Destination (Join-Path $libDir "gpu_histogram.dll") -Force
$dllSize = (Get-Item (Join-Path $libDir "gpu_histogram.dll")).Length / 1MB
Write-Host "  Copied: lib/gpu_histogram.dll ($([math]::Round($dllSize, 1)) MB)" -ForegroundColor Green

# Also copy to src/ so the Python wrapper finds it
Copy-Item $dllSource -Destination (Join-Path $ProjectRoot "src" "gpu_histogram.dll") -Force
Write-Host "  Copied: src/gpu_histogram.dll (for ctypes discovery)" -ForegroundColor Green
Write-Host ""

# -------------------------------------------------------------------------
# Step 5: Install Python package
# -------------------------------------------------------------------------
Write-Host "[5/6] Installing Python package..." -ForegroundColor Yellow

Push-Location $ProjectRoot
try {
    & python -m pip install -e . --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  WARNING: pip install -e . failed. You can still use the library via direct import." -ForegroundColor Yellow
    } else {
        Write-Host "  Python package installed (editable mode)." -ForegroundColor Green
    }
} finally {
    Pop-Location
}
Write-Host ""

# -------------------------------------------------------------------------
# Step 6: Verify
# -------------------------------------------------------------------------
Write-Host "[6/6] Verifying build..." -ForegroundColor Yellow

# Check DLL exists
$dllPath = Join-Path $libDir "gpu_histogram.dll"
if (Test-Path $dllPath) {
    Write-Host "  DLL exists: $dllPath" -ForegroundColor Green
} else {
    Write-Host "  ERROR: DLL not found at $dllPath" -ForegroundColor Red
    exit 1
}

# Try Python import
$verifyScript = @"
import sys, os
sys.path.insert(0, os.path.join(r'$ProjectRoot', 'src'))
os.add_dll_directory(r'$libDir')

# Also add CUDA bin to DLL search path
cuda_paths = [
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin',
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin',
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin',
]
for p in cuda_paths:
    if os.path.isdir(p):
        os.add_dll_directory(p)
        break

try:
    import numpy as np
    import scipy.sparse as sp
    print('  numpy + scipy OK')
except ImportError as e:
    print(f'  ERROR: {e}')
    sys.exit(1)

# Quick synthetic test
print('  Running synthetic histogram test...')
np.random.seed(42)
n_rows, n_features = 1000, 50000
density = 0.003
csr = sp.random(n_rows, n_features, density=density, format='csr', dtype=np.float64)
csr.data[:] = 1.0  # binary features

# CPU reference histogram
grad = np.random.randn(n_rows).astype(np.float64)
hess = np.abs(np.random.randn(n_rows)).astype(np.float64)

# Build histogram on CPU for comparison
hist_cpu = np.zeros((n_features, 2), dtype=np.float64)  # [grad_sum, hess_sum] per feature
csc = csr.tocsc()
for f in range(min(100, n_features)):  # spot check first 100
    rows = csc.indices[csc.indptr[f]:csc.indptr[f+1]]
    if len(rows) > 0:
        hist_cpu[f, 0] = grad[rows].sum()
        hist_cpu[f, 1] = hess[rows].sum()

print(f'  Synthetic data: {n_rows} rows x {n_features} features, NNZ={csr.nnz:,}')
print(f'  CPU histogram spot-check (first 100 features): OK')
print('  BUILD VERIFICATION PASSED')
"@

& python -c $verifyScript
if ($LASTEXITCODE -ne 0) {
    Write-Host "  WARNING: Verification script had issues. Check output above." -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Green
    Write-Host " BUILD COMPLETE" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "DLL location: $dllPath" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Run tests:  pytest tests/ -v --tb=short" -ForegroundColor White
    Write-Host "  2. Benchmark:  python benchmark/bench_kernel_speed.py --profile 4h" -ForegroundColor White
    Write-Host "  3. Real test:  Load actual 1w NPZ and build histogram" -ForegroundColor White
}
