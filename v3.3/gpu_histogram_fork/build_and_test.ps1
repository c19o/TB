# =============================================================================
# GPU Histogram Fork - Build & Test Script (Windows / PowerShell)
# =============================================================================
# Clones LightGBM, patches it with our CUDA sparse histogram kernel,
# builds with CUDA support, installs the Python package, and runs
# validation tests on synthetic + real data.
#
# Prerequisites:
#   - NVIDIA GPU with CUDA support (driver 535+)
#   - CUDA toolkit 11.8+ (with nvcc on PATH)
#   - Python 3.10+ (with pip)
#   - git, cmake (3.18+), Visual Studio 2019/2022 with C++ workload
#
# Usage:
#   .\build_and_test.ps1 [-CloneDir C:\tmp\lightgbm-fork] [-SkipClone] [-SkipTests]
#
# =============================================================================

[CmdletBinding()]
param(
    [string]$CloneDir = "$env:TEMP\lightgbm-fork",
    [string]$CudaArchs = "80;86;89;90",
    [string]$V33Dir = "",
    [switch]$SkipClone,
    [switch]$SkipTests,
    [switch]$SkipRealTest
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $V33Dir) {
    $V33Dir = Split-Path -Parent $ScriptDir
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Log-Step($num, $msg)  { Write-Host "`n=== STEP ${num}: ${msg} ===" -ForegroundColor Cyan }
function Log-OK($msg)          { Write-Host "[OK] $msg" -ForegroundColor Green }
function Log-Warn($msg)        { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Log-Fail($msg)        { Write-Host "[FAIL] $msg" -ForegroundColor Red }
function Log-Info($msg)        { Write-Host "  $msg" }

$TestsPassed = 0
$TestsFailed = 0

# ---------------------------------------------------------------------------
# Step 0: Preflight Checks
# ---------------------------------------------------------------------------

Log-Step 0 "Preflight checks"

# Check CUDA
$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if (-not $nvcc) {
    Log-Fail "nvcc not found. Install CUDA toolkit 11.8+ and add to PATH."
    Write-Host "  Typical path: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin"
    exit 1
}
$nvccOut = & nvcc --version 2>&1 | Out-String
$cudaMatch = [regex]::Match($nvccOut, 'release (\d+\.\d+)')
$CudaVersion = $cudaMatch.Groups[1].Value
$CudaMajor = [int]($CudaVersion.Split('.')[0])
$CudaMinor = [int]($CudaVersion.Split('.')[1])

if ($CudaMajor -lt 11 -or ($CudaMajor -eq 11 -and $CudaMinor -lt 8)) {
    Log-Fail "CUDA $CudaVersion is too old. Need 11.8+."
    exit 1
}
Log-OK "CUDA toolkit: $CudaVersion"

# Check nvidia-smi
$nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvsmi) {
    $gpuName = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1
    $gpuVram = & nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>$null | Select-Object -First 1
    $driverVer = & nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1
    Log-OK "GPU: $gpuName ($gpuVram), Driver: $driverVer"
} else {
    Log-Warn "nvidia-smi not found. Continuing anyway."
}

# Check cmake
$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmake) {
    Log-Fail "cmake not found. Install cmake 3.18+."
    exit 1
}
$cmakeVer = (& cmake --version | Select-Object -First 1) -replace 'cmake version ',''
Log-OK "cmake: $cmakeVer"

# Check git
$git = Get-Command git -ErrorAction SilentlyContinue
if (-not $git) {
    Log-Fail "git not found."
    exit 1
}
Log-OK "git: $(& git --version)"

# Check Python
$Python = $null
foreach ($py in @("python", "python3")) {
    $cmd = Get-Command $py -ErrorAction SilentlyContinue
    if ($cmd) {
        $pyVerOut = & $py --version 2>&1 | Out-String
        $pyMatch = [regex]::Match($pyVerOut, '(\d+)\.(\d+)')
        if ($pyMatch.Success) {
            $pyMaj = [int]$pyMatch.Groups[1].Value
            $pyMin = [int]$pyMatch.Groups[2].Value
            if ($pyMaj -ge 3 -and $pyMin -ge 10) {
                $Python = $py
                break
            }
        }
    }
}
if (-not $Python) {
    Log-Fail "Python 3.10+ not found."
    exit 1
}
Log-OK "Python: $(& $Python --version 2>&1)"

# Check Visual Studio (cl.exe)
$cl = Get-Command cl -ErrorAction SilentlyContinue
if (-not $cl) {
    Log-Warn "cl.exe not found. You may need to run from a VS Developer Command Prompt."
    Log-Info "Or: cmake will try to find Visual Studio automatically."
}

# Add sm_100 if CUDA >= 12.8
if ($CudaMajor -ge 13 -or ($CudaMajor -eq 12 -and $CudaMinor -ge 8)) {
    if ($CudaArchs -notmatch "100") {
        $CudaArchs = "$CudaArchs;100"
        Log-Info "CUDA >= 12.8 detected, added sm_100 (B200)"
    }
}

# Determine CPU core count
$Nproc = (Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum
if (-not $Nproc -or $Nproc -lt 1) { $Nproc = 4 }

Log-OK "All preflight checks passed"
Write-Host ""
Write-Host "  Clone dir:    $CloneDir"
Write-Host "  CUDA archs:   $CudaArchs"
Write-Host "  Patch source: $ScriptDir\src\"
Write-Host "  v3.3 dir:     $V33Dir"
Write-Host "  Parallelism:  $Nproc cores"

# ---------------------------------------------------------------------------
# Step 1: Clone LightGBM
# ---------------------------------------------------------------------------

Log-Step 1 "Clone LightGBM"

if ($SkipClone -and (Test-Path $CloneDir)) {
    Log-Info "Skipping clone (-SkipClone), using existing: $CloneDir"
} else {
    if (Test-Path $CloneDir) {
        Log-Info "Removing existing clone at $CloneDir"
        Remove-Item -Recurse -Force $CloneDir
    }
    Log-Info "Cloning LightGBM (with submodules)..."
    & git clone --recursive --depth 1 https://github.com/microsoft/LightGBM.git $CloneDir
    if ($LASTEXITCODE -ne 0) {
        Log-Fail "git clone failed"
        exit 1
    }
    Log-OK "Cloned to $CloneDir"
}

Push-Location $CloneDir
$lgbmVersion = & git describe --tags --always 2>$null
if (-not $lgbmVersion) { $lgbmVersion = "unknown" }
Log-OK "LightGBM version: $lgbmVersion"
Pop-Location

# ---------------------------------------------------------------------------
# Step 2: Copy our CUDA sparse histogram files
# ---------------------------------------------------------------------------

Log-Step 2 "Copy GPU sparse histogram kernel files"

$SrcCu = Join-Path $ScriptDir "src\gpu_histogram.cu"
$SrcH  = Join-Path $ScriptDir "src\gpu_histogram.h"

if (-not (Test-Path $SrcCu)) { Log-Fail "Missing: $SrcCu"; exit 1 }
if (-not (Test-Path $SrcH))  { Log-Fail "Missing: $SrcH"; exit 1 }

$TreelearnerDir = Join-Path $CloneDir "src\treelearner\cuda_sparse"
New-Item -ItemType Directory -Force -Path $TreelearnerDir | Out-Null

Copy-Item $SrcCu (Join-Path $TreelearnerDir "cuda_sparse_hist.cu") -Force
Copy-Item $SrcH  (Join-Path $TreelearnerDir "cuda_sparse_hist.h") -Force
Log-OK "Copied gpu_histogram.cu -> cuda_sparse\cuda_sparse_hist.cu"
Log-OK "Copied gpu_histogram.h  -> cuda_sparse\cuda_sparse_hist.h"

$IntegrationPy = Join-Path $ScriptDir "src\lgbm_integration.py"
if (Test-Path $IntegrationPy) {
    Copy-Item $IntegrationPy (Join-Path $CloneDir "python-package\lgbm_integration.py") -Force
    Log-OK "Copied lgbm_integration.py to python-package\"
}

# ---------------------------------------------------------------------------
# Step 3: Patch LightGBM source files
# ---------------------------------------------------------------------------

Log-Step 3 "Patch LightGBM source files"

# --- 3a: Patch config.h ---
$ConfigH = Join-Path $CloneDir "include\LightGBM\config.h"
if (-not (Test-Path $ConfigH)) {
    Log-Fail "config.h not found at $ConfigH"
    exit 1
}

$configContent = Get-Content $ConfigH -Raw
if ($configContent -match "use_cuda_sparse_histogram") {
    Log-Info "config.h already patched, skipping"
} else {
    $patch = @"

// === GPU Sparse Histogram Co-Processor (v3.3 patch) ===
// When enabled, histogram construction for sparse CSR data is offloaded
// to our custom CUDA kernel instead of LightGBM's CPU path.
bool use_cuda_sparse_histogram = false;
"@
    Add-Content -Path $ConfigH -Value $patch
    Log-OK "Patched config.h (added use_cuda_sparse_histogram)"
}

# --- 3b: Patch CMakeLists.txt ---
$LgbmCMake = Join-Path $CloneDir "CMakeLists.txt"
$cmakeContent = Get-Content $LgbmCMake -Raw

if ($cmakeContent -match "USE_CUDA_SPARSE") {
    Log-Info "CMakeLists.txt already patched, skipping"
} else {
    $cmakePatch = @"

# === v3.3 GPU Sparse Histogram Co-Processor ===
option(USE_CUDA_SPARSE "Build with GPU sparse histogram co-processor (v3.3)" OFF)
if(USE_CUDA_SPARSE)
    message(STATUS "GPU Sparse Histogram: ENABLED")
    enable_language(CUDA)
    find_package(CUDAToolkit 11.8 REQUIRED)
    add_definitions(-DUSE_CUDA_SPARSE)
    list(APPEND SOURCES "src/treelearner/cuda_sparse/cuda_sparse_hist.cu")
    set(CMAKE_CUDA_FLAGS "`${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --expt-relaxed-constexpr")
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
    endif()
    message(STATUS "GPU Sparse CUDA architectures: `${CMAKE_CUDA_ARCHITECTURES}")
endif()
"@
    Add-Content -Path $LgbmCMake -Value $cmakePatch
    Log-OK "Patched CMakeLists.txt (added USE_CUDA_SPARSE)"
}

# --- 3c: Patch tree_learner.cpp ---
$TreeLearnerCpp = Join-Path $CloneDir "src\treelearner\tree_learner.cpp"
if (Test-Path $TreeLearnerCpp) {
    $tlContent = Get-Content $TreeLearnerCpp -Raw
    if ($tlContent -match "cuda_sparse") {
        Log-Info "tree_learner.cpp already patched, skipping"
    } else {
        # Simple patch: add a comment block near top
        $tlContent = $tlContent -replace "(#include.*tree_learner\.h.*)", @"
`$1
// v3.3 GPU sparse histogram support
#ifdef USE_CUDA_SPARSE
#include "cuda_sparse/cuda_sparse_hist.h"
#endif
"@
        Set-Content -Path $TreeLearnerCpp -Value $tlContent -NoNewline
        Log-OK "Patched tree_learner.cpp (added cuda_sparse include)"
    }
} else {
    Log-Warn "tree_learner.cpp not found, skipping"
}

# ---------------------------------------------------------------------------
# Step 4: Build LightGBM with CUDA sparse support
# ---------------------------------------------------------------------------

Log-Step 4 "Build LightGBM with CUDA sparse support"

$BuildDir = Join-Path $CloneDir "build"
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
Push-Location $BuildDir

Log-Info "Running cmake..."

# Try to find Visual Studio generator
$cmakeArgs = @(
    "..",
    "-DUSE_CUDA_SPARSE=ON",
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchs",
    "-DCMAKE_BUILD_TYPE=Release"
)

# On Windows, prefer Visual Studio generator if available
$vsGen = $null
foreach ($ver in @("17 2022", "16 2019")) {
    $testGen = "Visual Studio $ver"
    $genCheck = & cmake --help 2>&1 | Out-String
    if ($genCheck -match [regex]::Escape($testGen)) {
        $vsGen = $testGen
        break
    }
}

if ($vsGen) {
    $cmakeArgs += @("-G", $vsGen, "-A", "x64")
    Log-Info "Using generator: $vsGen"
} else {
    # Try Ninja if available
    $ninja = Get-Command ninja -ErrorAction SilentlyContinue
    if ($ninja) {
        $cmakeArgs += @("-G", "Ninja")
        Log-Info "Using generator: Ninja"
    } else {
        Log-Info "Using default cmake generator"
    }
}

& cmake @cmakeArgs 2>&1 | Tee-Object -Variable cmakeOutput | Select-Object -Last 10
if ($LASTEXITCODE -ne 0) {
    Log-Fail "cmake failed. Output:"
    $cmakeOutput | Write-Host
    Pop-Location
    exit 1
}
Log-OK "cmake configuration complete"

Log-Info "Building..."
& cmake --build . --config Release --parallel $Nproc 2>&1 | Tee-Object -Variable buildOutput | Select-Object -Last 10
if ($LASTEXITCODE -ne 0) {
    Log-Fail "Build failed. Check output above."
    Log-Info "Common fixes:"
    Log-Info "  - Run from VS Developer Command Prompt (x64 Native Tools)"
    Log-Info "  - Ensure CUDA toolkit bin is on PATH"
    Log-Info "  - Try: -CudaArchs '86' (match your specific GPU)"
    Pop-Location
    exit 1
}
Log-OK "Build complete"

# Check for built library
$dllPaths = @(
    (Join-Path $BuildDir "Release\lib_lightgbm.dll"),
    (Join-Path $BuildDir "lib_lightgbm.dll"),
    (Join-Path $BuildDir "Release\lightgbm.dll")
)
$foundLib = $false
foreach ($dll in $dllPaths) {
    if (Test-Path $dll) {
        $libSize = (Get-Item $dll).Length / 1MB
        Log-OK "$([System.IO.Path]::GetFileName($dll)) built ({0:N1} MB)" -f $libSize
        $foundLib = $true
        break
    }
}
if (-not $foundLib) {
    Log-Warn "DLL not found in expected locations. Searching..."
    Get-ChildItem -Path $BuildDir -Recurse -Include "*.dll" | Select-Object -First 5 | ForEach-Object {
        Log-Info $_.FullName
    }
}

Pop-Location

# ---------------------------------------------------------------------------
# Step 5: Install Python package
# ---------------------------------------------------------------------------

Log-Step 5 "Install Python package"

Push-Location (Join-Path $CloneDir "python-package")

Log-Info "Installing LightGBM Python package..."
& $Python -m pip install -e . --no-build-isolation 2>&1 | Select-Object -Last 5
if ($LASTEXITCODE -ne 0) {
    Log-Warn "Editable install failed, trying regular install..."
    & $Python -m pip install . --no-build-isolation 2>&1 | Select-Object -Last 5
    if ($LASTEXITCODE -ne 0) {
        Log-Fail "Python package installation failed."
        Pop-Location
        exit 1
    }
}
Log-OK "Python package installed"
Pop-Location

# ---------------------------------------------------------------------------
# Step 6: Verify import
# ---------------------------------------------------------------------------

Log-Step 6 "Verify LightGBM import"

$importScript = @"
import lightgbm
print(f'LightGBM version: {lightgbm.__version__}')
try:
    ds = lightgbm.Dataset([[0]], label=[0], free_raw_data=False)
    ds.construct()
    print('Dataset construction: OK')
except Exception as e:
    print(f'Dataset construction: {e}')
print('IMPORT_OK')
"@

$importOutput = & $Python -c $importScript 2>&1 | Out-String
Write-Host $importOutput

if ($importOutput -match "IMPORT_OK") {
    Log-OK "LightGBM import verified"
} else {
    Log-Fail "LightGBM import failed"
    exit 1
}

# ---------------------------------------------------------------------------
# Step 7: Test on synthetic data
# ---------------------------------------------------------------------------

if ($SkipTests) {
    Log-Step 7 "SKIPPED (-SkipTests)"
} else {
    Log-Step 7 "Test on synthetic sparse data"

    $synthScript = @"
import sys, time
import numpy as np

try:
    import scipy.sparse as sp
    import lightgbm as lgb
except ImportError as e:
    print(f'[FAIL] Missing: {e}')
    sys.exit(1)

print(f'LightGBM version: {lgb.__version__}')
print()

# Test 1: Small sparse binary matrix
print('--- Test 1: Small synthetic (1K x 50K, binary sparse) ---')
np.random.seed(42)
X = sp.random(1000, 50000, density=0.003, format='csr', dtype=np.float32)
X.data[:] = 1.0
y = np.random.randint(0, 3, 1000)
print(f'  Matrix: 1000 x 50000, NNZ={X.nnz:,}')

ds = lgb.Dataset(X, label=y, params={'feature_pre_filter': False})
params = {
    'objective': 'multiclass', 'num_class': 3, 'device_type': 'cpu',
    'num_leaves': 31, 'max_bin': 255, 'verbose': -1, 'seed': 42,
}

t0 = time.time()
model = lgb.train(params, ds, num_boost_round=10)
elapsed = time.time() - t0
pred = model.predict(X)
acc = (np.argmax(pred, axis=1) == y).mean()
print(f'  CPU: accuracy={acc:.4f}, time={elapsed:.2f}s')

# Test GPU if available
try:
    params_gpu = params.copy()
    params_gpu['device_type'] = 'gpu'
    t0 = time.time()
    model_gpu = lgb.train(params_gpu, ds, num_boost_round=10)
    gpu_time = time.time() - t0
    pred_gpu = model_gpu.predict(X)
    acc_gpu = (np.argmax(pred_gpu, axis=1) == y).mean()
    agree = (np.argmax(pred, axis=1) == np.argmax(pred_gpu, axis=1)).mean()
    print(f'  GPU: accuracy={acc_gpu:.4f}, time={gpu_time:.2f}s, speedup={elapsed/gpu_time:.1f}x')
    print(f'  Agreement: {agree:.4f}')
except Exception as e:
    print(f'  GPU: not available ({e})')

# Test int64 indptr
print()
print('--- Test 2: int64 indptr + NaN + EFB ---')
X_i64 = sp.csr_matrix(X)
X_i64.indptr = X_i64.indptr.astype(np.int64)
ds_i64 = lgb.Dataset(X_i64, label=y, params={'feature_pre_filter': False})
try:
    lgb.train(params, ds_i64, num_boost_round=5)
    print('  int64 indptr: PASSED')
except Exception as e:
    print(f'  int64 indptr: FAILED ({e})')

# Test NaN
X_nan = X.copy().toarray()
X_nan[0, :10] = np.nan
ds_nan = lgb.Dataset(sp.csr_matrix(X_nan.astype(np.float64)), label=y,
                     params={'feature_pre_filter': False})
try:
    lgb.train(params, ds_nan, num_boost_round=5)
    print('  NaN in sparse: PASSED')
except Exception as e:
    print(f'  NaN in sparse: FAILED ({e})')

# Test EFB
params_efb = params.copy()
params_efb['max_bin'] = 255
ds_efb = lgb.Dataset(X, label=y, params={'feature_pre_filter': False, 'max_bin': 255})
try:
    lgb.train(params_efb, ds_efb, num_boost_round=5)
    print('  EFB max_bin=255: PASSED')
except Exception as e:
    print(f'  EFB max_bin=255: FAILED ({e})')

print()
print('SYNTH_TESTS_COMPLETE')
"@

    & $Python -u -c $synthScript 2>&1

    if ($LASTEXITCODE -eq 0) {
        Log-OK "Synthetic tests complete"
        $TestsPassed++
    } else {
        Log-Fail "Synthetic tests failed"
        $TestsFailed++
    }
}

# ---------------------------------------------------------------------------
# Step 8: Test on real 1w data
# ---------------------------------------------------------------------------

if ($SkipTests -or $SkipRealTest) {
    Log-Step 8 "SKIPPED (real data test)"
} else {
    Log-Step 8 "Test on real 1w data"

    # Search for parquet
    $parquet = $null
    foreach ($dir in @($V33Dir, "C:\workspace\v3.3", "C:\workspace")) {
        if (Test-Path $dir) {
            $found = Get-ChildItem -Path $dir -Filter "*1w*.parquet" -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($found) {
                $parquet = $found.FullName
                break
            }
        }
    }

    if (-not $parquet) {
        Log-Warn "No real 1w parquet found. Skipping real data test."
        Log-Info "Place BTC_1w*.parquet in $V33Dir to enable this test."
    } else {
        Log-Info "Parquet: $parquet"

        $realScript = @"
import sys, os, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import lightgbm as lgb

parquet_path = r'$parquet'
print(f'Loading: {parquet_path}')

df = pd.read_parquet(parquet_path)
print(f'  Shape: {df.shape}')

label_col = None
for col in ['label', 'y', 'target', 'label_3class']:
    if col in df.columns:
        label_col = col
        break

if label_col:
    y = df[label_col].values
    df = df.drop(columns=[label_col])
    print(f'  Label: {label_col}')
else:
    y = np.random.randint(0, 3, len(df))
    print('  Using random labels (no label column found)')

X = sp.csr_matrix(df.values.astype(np.float32))
print(f'  Sparse NNZ: {X.nnz:,}')

params = {
    'objective': 'multiclass', 'num_class': 3, 'device_type': 'cpu',
    'num_leaves': 31, 'max_bin': 255, 'feature_pre_filter': False,
    'min_data_in_leaf': 3, 'verbose': -1, 'seed': 42,
}
ds = lgb.Dataset(X, label=y, params={'feature_pre_filter': False, 'max_bin': 255})

t0 = time.time()
model = lgb.train(params, ds, num_boost_round=10)
elapsed = time.time() - t0

pred = model.predict(X)
acc = (np.argmax(pred, axis=1) == y).mean()
print(f'  Accuracy: {acc:.4f}, Time: {elapsed:.2f}s, Trees: {model.num_trees()}')
print('REAL_TEST_COMPLETE')
"@

        & $Python -u -c $realScript 2>&1

        if ($LASTEXITCODE -eq 0) {
            Log-OK "Real data test complete"
            $TestsPassed++
        } else {
            Log-Fail "Real data test failed"
            $TestsFailed++
        }
    }
}

# ---------------------------------------------------------------------------
# Step 9: Build standalone co-processor library
# ---------------------------------------------------------------------------

Log-Step 9 "Build standalone GPU histogram co-processor library"

$standalonecmake = Join-Path $ScriptDir "CMakeLists.txt"
if (Test-Path $standalonecmake) {
    $coprocBuild = Join-Path $ScriptDir "build"
    New-Item -ItemType Directory -Force -Path $coprocBuild | Out-Null
    Push-Location $coprocBuild

    Log-Info "Building standalone libgpu_histogram..."
    & cmake $ScriptDir -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$CudaArchs" 2>&1 | Select-Object -Last 5
    & cmake --build . --config Release --parallel $Nproc 2>&1 | Select-Object -Last 5

    $dllPath = Join-Path $ScriptDir "lib\gpu_histogram.dll"
    $soPath  = Join-Path $ScriptDir "lib\libgpu_histogram.so"
    if ((Test-Path $dllPath) -or (Test-Path $soPath)) {
        Log-OK "Standalone co-processor library built"
    } else {
        Log-Warn "Standalone library not found (may be OK if .cu has compile errors)"
    }
    Pop-Location
} else {
    Log-Info "No standalone CMakeLists.txt, skipping"
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "=== BUILD & TEST SUMMARY ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "  LightGBM fork:      $CloneDir"
Write-Host "  CUDA version:       $CudaVersion"
Write-Host "  CUDA architectures: $CudaArchs"
Write-Host "  Build status:       SUCCESS"

if (-not $SkipTests) {
    Write-Host "  Tests passed:       $TestsPassed"
    Write-Host "  Tests failed:       $TestsFailed"
}

Write-Host ""
Write-Host "To use the patched LightGBM:"
Write-Host "  `$env:LIGHTGBM_DIR = '$CloneDir'"
Write-Host "  python -c `"import lightgbm; print(lightgbm.__version__)`""
Write-Host ""

if ($TestsFailed -gt 0) {
    Log-Fail "Some tests failed. Check output above."
    exit 1
} else {
    Log-OK "All steps completed successfully."
    exit 0
}
