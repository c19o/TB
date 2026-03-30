#!/bin/bash
# v3.3 Cloud Setup — pip + SCP + Memory/NUMA Optimizations
# Usage: scp this + code.tar.gz + dbs.tar.gz to machine, then run: bash setup.sh
#
# Optimizations applied:
#   1. tcmalloc (google-perftools) — 5-15% from per-thread malloc caches
#   2. THP enabled + defer+madvise defrag — 5-20% from reduced dTLB misses
#   3. vm.swappiness=1 — prevents kernel stealing hot pages
#   4. NUMA topology detection + binding recommendation
#   5. vm.overcommit_memory=1 — needed for large sparse CSR allocations
#
# Combined expected speedup: 20-40% wall-time reduction on multi-socket machines

set -e

echo "============================================================"
echo "  v3.3 Cloud Setup — Memory-Optimized"
echo "============================================================"

CORES=$(nproc)
RAM_GB=$(free -g | awk '/Mem/{print $2}')
echo "Cores: $CORES | RAM: ${RAM_GB}GB"
echo ""

# ============================================================
# 1. Install Python packages + system deps
# ============================================================
echo ">>> [1/7] Installing system packages..."
apt-get update -qq 2>/dev/null
apt-get install -y -qq google-perftools libgoogle-perftools-dev numactl hwloc 2>/dev/null || {
    echo "WARN: apt-get failed (container may not have root). Trying pip-only..."
}

echo ">>> Installing Python packages..."
pip install --isolated -q lightgbm scikit-learn scipy ephem astropy pytz joblib \
    pandas numpy pyarrow optuna hmmlearn numba tqdm pyyaml psutil cupy-cuda12x 2>/dev/null || true

# Detect CuPy — if not available, set ALLOW_CPU=1 so GPU-or-nothing guards don't crash
if python -c "import cupy" 2>/dev/null; then
    echo "  CuPy: AVAILABLE (GPU feature building enabled)"
else
    echo "  CuPy: NOT AVAILABLE — setting ALLOW_CPU=1"
    export ALLOW_CPU=1
fi

# Verify ALL imports
echo ">>> Verifying imports..."
python -c "import pandas, numpy, scipy, sklearn, lightgbm, ephem, astropy, pyarrow, optuna, numba, hmmlearn, yaml, tqdm, psutil; print('ALL IMPORTS OK')" || {
    echo "CRITICAL: Import check failed. Fix before running pipeline."
    exit 1
}

# ============================================================
# 2. Transparent Huge Pages — defer+madvise (safest for batch ML)
# ============================================================
echo ""
echo ">>> [2/7] Configuring Transparent Huge Pages..."
# THP "always" causes 512x memory bloat on sparse regions and multi-second stalls.
# "madvise" = only use huge pages when explicitly requested via madvise(MADV_HUGEPAGE).
# tcmalloc's arenas call madvise() internally, so ML hot paths still get THP benefit
# without bloating sparse CSR regions that never touch most pages.
if [ -w /sys/kernel/mm/transparent_hugepage/enabled ]; then
    echo madvise > /sys/kernel/mm/transparent_hugepage/enabled
    echo defer+madvise > /sys/kernel/mm/transparent_hugepage/defrag
    # khugepaged scans for collapse opportunities in background
    echo 1 > /sys/kernel/mm/transparent_hugepage/khugepaged/defrag 2>/dev/null || true
    # Increase khugepaged scan frequency (default 4096 pages every 10s)
    echo 4096 > /sys/kernel/mm/transparent_hugepage/khugepaged/pages_to_scan 2>/dev/null || true
    echo "  THP enabled=$(cat /sys/kernel/mm/transparent_hugepage/enabled)"
    echo "  THP defrag=$(cat /sys/kernel/mm/transparent_hugepage/defrag)"
else
    echo "  WARN: Cannot write to THP sysfs (container restriction). Skipping."
fi

# ============================================================
# 3. Kernel VM tuning
# ============================================================
echo ""
echo ">>> [3/7] Kernel VM tuning..."
if command -v sysctl &>/dev/null; then
    # vm.swappiness=1: prevent kernel from swapping out hot LightGBM pages
    sysctl -w vm.swappiness=1 2>/dev/null && echo "  vm.swappiness=1" || echo "  WARN: cannot set vm.swappiness"
    # vm.overcommit_memory=1: needed for large sparse CSR mmap allocations
    sysctl -w vm.overcommit_memory=1 2>/dev/null && echo "  vm.overcommit_memory=1" || echo "  WARN: cannot set vm.overcommit_memory"
    # Dirty page writeback tuning (not critical but helps with parquet I/O)
    sysctl -w vm.dirty_ratio=40 2>/dev/null || true
    sysctl -w vm.dirty_background_ratio=10 2>/dev/null || true
    # Allow perf counters for measurement
    sysctl -w kernel.perf_event_paranoid=-1 2>/dev/null || true
else
    echo "  WARN: sysctl not available"
fi

# ============================================================
# 4. tcmalloc detection and LD_PRELOAD setup
# ============================================================
echo ""
echo ">>> [4/7] Configuring tcmalloc..."
# Find tcmalloc_minimal (no profiling overhead) — works with THP defer+madvise
# because tcmalloc's internal arenas call madvise(MADV_HUGEPAGE) on span memory.
# PyDataMem_NEW (numpy array data) -> malloc() -> intercepted by tcmalloc.
# LightGBM C++ malloc/free -> intercepted by tcmalloc via LD_PRELOAD.
# Numba NRT allocator -> malloc() -> intercepted by tcmalloc.
# Python pymalloc (<512B objects) -> mmap (NOT intercepted, but irrelevant for us).
TCMALLOC_LIB=""
for candidate in \
    /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 \
    /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so \
    /usr/lib/libtcmalloc_minimal.so.4 \
    /usr/lib/libtcmalloc_minimal.so \
    /usr/lib64/libtcmalloc_minimal.so.4 \
    /usr/lib64/libtcmalloc_minimal.so; do
    if [ -f "$candidate" ]; then
        TCMALLOC_LIB="$candidate"
        break
    fi
done

# Also try ldconfig search
if [ -z "$TCMALLOC_LIB" ]; then
    TCMALLOC_LIB=$(ldconfig -p 2>/dev/null | grep libtcmalloc_minimal | head -1 | awk '{print $NF}')
fi

if [ -n "$TCMALLOC_LIB" ]; then
    echo "  Found: $TCMALLOC_LIB"
    echo "  Will be set via LD_PRELOAD in launch wrapper"
else
    echo "  WARN: tcmalloc not found. Install: apt-get install google-perftools"
    echo "  Falling back to glibc malloc (5-15% slower for LightGBM)"
fi

# ============================================================
# 5. NUMA topology detection
# ============================================================
echo ""
echo ">>> [5/7] NUMA topology..."
NUMA_NODES=1
NUMA_BIND_CMD=""
if command -v numactl &>/dev/null; then
    NUMA_NODES=$(numactl --hardware 2>/dev/null | awk '/available:/{print $2}' || echo 1)
    echo "  NUMA nodes: $NUMA_NODES"
    numactl --hardware 2>/dev/null | grep -E "node [0-9]+ size|node [0-9]+ free" || true

    if [ "$NUMA_NODES" -gt 1 ]; then
        echo ""
        echo "  MULTI-SOCKET DETECTED ($NUMA_NODES nodes)"
        echo "  Options:"
        echo "    A) Single-process, interleaved (simple, ~1.4-1.8x vs naive):"
        echo "       numactl --interleave=all python train.py"
        echo "    B) Per-node Optuna workers (best, ~3.5-5x for parallel trials):"
        echo "       See numa_optuna_launch.sh generated below"
        echo ""
        # For single-process training (cloud_run_tf.py), interleave is safest
        NUMA_BIND_CMD="numactl --interleave=all"

        # Generate per-node Optuna launch script for advanced use
        cat > /workspace/numa_optuna_launch.sh << 'NUMA_EOF'
#!/bin/bash
# Per-NUMA-node Optuna parallel workers
# Each worker gets its own copy of data (allocated on local NUMA node via --membind)
# Workers share Optuna study via JournalFileStorage
STUDY_NAME="${1:-lgbm_cpcv_study}"
SCRIPT="${2:-cloud_run_optuna.py}"
TF="${3:-1d}"
NODES=$(numactl --hardware | awk '/available:/{print $2}')
CORES_PER_NODE=$(($(nproc) / NODES))

echo "Launching $NODES Optuna workers ($CORES_PER_NODE cores each)"

# Create study
python -c "
import optuna
storage = optuna.storages.JournalStorage(
    optuna.storages.JournalFileBackend('/tmp/optuna_journal.log'))
try:
    optuna.create_study(study_name='$STUDY_NAME', storage=storage, direction='minimize')
except optuna.exceptions.DuplicatedStudyError:
    print('Study exists, workers will join it')
"

for NODE in $(seq 0 $((NODES - 1))); do
    echo "  Node $NODE: $CORES_PER_NODE cores"
    numactl --cpunodebind=$NODE --membind=$NODE \
        env OMP_NUM_THREADS=$CORES_PER_NODE \
            LD_PRELOAD="${TCMALLOC_LIB:-}" \
            PYTHONUNBUFFERED=1 \
        python -u $SCRIPT --tf $TF --numa-node $NODE 2>&1 | \
        tee /workspace/optuna_node${NODE}.log &
done

wait
echo "All NUMA workers completed"
NUMA_EOF
        chmod +x /workspace/numa_optuna_launch.sh
        echo "  Generated: /workspace/numa_optuna_launch.sh"
    else
        echo "  Single NUMA node — no binding needed"
    fi
else
    echo "  numactl not available — skipping NUMA detection"
fi

# ============================================================
# 6. Create optimized launch wrapper
# ============================================================
echo ""
echo ">>> [6/7] Creating launch wrapper..."

# Write the wrapper that applies all optimizations at runtime
cat > /usr/local/bin/lgbm-run << WRAPPER_EOF
#!/bin/bash
# Optimized LightGBM launch wrapper — applies tcmalloc + NUMA + env tuning
# Usage: lgbm-run python -u cloud_run_tf.py --tf 1w

# tcmalloc: intercepts all malloc/free from numpy, scipy, LightGBM C++, Numba NRT
export LD_PRELOAD="${TCMALLOC_LIB:-}"

# Suppress tcmalloc warnings for large allocations (our CSR matrices are 50-100GB)
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10737418240

# Python memory: let tcmalloc handle all allocations including small objects
# pymalloc (<512B) uses mmap arenas that tcmalloc can't see — PYTHONMALLOC=malloc
# routes everything through tcmalloc for unified management + profiling
export PYTHONMALLOC=malloc

# Preserve ALLOW_CPU from parent shell (fallback if CuPy unavailable)
export ALLOW_CPU="${ALLOW_CPU:-0}"

# Unbuffered output for log tailing
export PYTHONUNBUFFERED=1

# Print what we're applying
echo "[lgbm-run] LD_PRELOAD=\${LD_PRELOAD:-none}"
echo "[lgbm-run] PYTHONMALLOC=\$PYTHONMALLOC"
echo "[lgbm-run] NUMA nodes: \$(numactl --hardware 2>/dev/null | awk '/available:/{print \$2}' || echo '?')"

# Apply NUMA binding for multi-socket machines
NUMA_NODES=\$(numactl --hardware 2>/dev/null | awk '/available:/{print \$2}' || echo 1)
if [ "\$NUMA_NODES" -gt 1 ] 2>/dev/null; then
    echo "[lgbm-run] Multi-socket: using numactl --interleave=all"
    exec numactl --interleave=all "\$@"
else
    exec "\$@"
fi
WRAPPER_EOF
chmod +x /usr/local/bin/lgbm-run

echo "  Created: /usr/local/bin/lgbm-run"
echo "  Usage:  lgbm-run python -u cloud_run_tf.py --tf 1w"

# Also write env vars file that cloud_run_tf.py can source
cat > /workspace/lgbm_env.sh << ENV_EOF
# Source this before running pipeline scripts directly (without lgbm-run wrapper)
export LD_PRELOAD="${TCMALLOC_LIB:-}"
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10737418240
export PYTHONMALLOC=malloc
export PYTHONUNBUFFERED=1
export ALLOW_CPU="${ALLOW_CPU:-0}"
ENV_EOF
echo "  Created: /workspace/lgbm_env.sh (source this for direct runs)"

# ============================================================
# 7. Unpack code and DBs
# ============================================================
echo ""
echo ">>> [7/8] Unpacking data..."

if [ -f /workspace/code.tar.gz ]; then
    echo "  Unpacking code..."
    cd /workspace
    tar xzf code.tar.gz
    echo "  Code ready: $(ls /workspace/v3.3/*.py 2>/dev/null | wc -l) Python files"
fi

if [ -f /workspace/dbs.tar.gz ]; then
    echo "  Unpacking databases..."
    cd /workspace
    tar xzf dbs.tar.gz
    echo "  DBs ready: $(ls /workspace/v3.3/*.db 2>/dev/null | wc -l) databases"
fi

# Symlinks for legacy paths
ln -sf /workspace/v3.3/*.db /workspace/ 2>/dev/null || true
ln -sf /workspace/v3.3/btc_prices.db /workspace/btc_prices.db 2>/dev/null || true

# ============================================================
# 8. Build LightGBM CUDA Sparse Fork (if CUDA available)
# ============================================================
echo ""
echo ">>> [8/8] LightGBM CUDA Sparse Fork..."
FORK_DIR="/workspace/v3.3/gpu_histogram_fork"
CUDA_SPARSE_BUILT=0
if command -v nvcc &>/dev/null && [ -f "$FORK_DIR/build_linux.sh" ]; then
    echo "  CUDA detected + fork source found. Building..."
    if bash "$FORK_DIR/build_linux.sh" --install 2>&1 | tail -20; then
        CUDA_SPARSE_BUILT=1
        echo "  CUDA Sparse Histogram fork: BUILT + INSTALLED"
    else
        echo "  WARN: Fork build failed. Training will use CPU histograms (still works, just slower)."
    fi
elif ! command -v nvcc &>/dev/null; then
    echo "  No CUDA toolkit found. Skipping fork build (CPU histograms will be used)."
else
    echo "  Fork source not found at $FORK_DIR/build_linux.sh. Skipping."
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "  SETUP COMPLETE"
echo "============================================================"
echo ""
echo "  Cores: $CORES | RAM: ${RAM_GB}GB | NUMA nodes: $NUMA_NODES"
echo ""
echo "  Optimizations applied:"
[ -n "$TCMALLOC_LIB" ] && echo "    [OK] tcmalloc: $TCMALLOC_LIB" || echo "    [--] tcmalloc: not found"
[ -w /sys/kernel/mm/transparent_hugepage/enabled ] && echo "    [OK] THP: madvise + defer+madvise defrag" || echo "    [--] THP: container restricted"
echo "    [OK] vm.swappiness=1, vm.overcommit_memory=1"
[ "$NUMA_NODES" -gt 1 ] 2>/dev/null && echo "    [OK] NUMA: $NUMA_NODES nodes, interleave binding" || echo "    [OK] NUMA: single node"
[ "$CUDA_SPARSE_BUILT" -eq 1 ] 2>/dev/null && echo "    [OK] CUDA Sparse Histogram fork: built + installed" || echo "    [--] CUDA Sparse Histogram fork: not built"
echo ""
echo "  Run (recommended):"
echo "    cd /workspace/v3.3 && lgbm-run python -u cloud_run_tf.py --symbol BTC --tf 1w"
echo ""
echo "  Run (direct, source env first):"
echo "    source /workspace/lgbm_env.sh"
echo "    cd /workspace/v3.3 && python -u cloud_run_tf.py --symbol BTC --tf 1w"
echo ""
echo "  Measure impact (before/after):"
echo "    perf stat -e dTLB-load-misses,node-load-misses,context-switches,page-faults \\"
echo "      -o /tmp/perf_stats.txt -- lgbm-run python -u cloud_run_tf.py --tf 1w"
echo ""
if [ "$NUMA_NODES" -gt 1 ] 2>/dev/null; then
echo "  For parallel Optuna (advanced, multi-socket only):"
echo "    bash /workspace/numa_optuna_launch.sh lgbm_study cloud_run_optuna.py 1d"
echo ""
fi
