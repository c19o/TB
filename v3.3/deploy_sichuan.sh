#!/bin/bash
# ============================================================================
# deploy_sichuan.sh — One-shot deploy to vast.ai Sichuan 8x RTX 3090 machine
# ============================================================================
#
# Handles everything from SSH wait -> SCP -> setup -> verify -> launch.
# Run locally from git bash on Windows.
#
# Machine: Instance 33876301, 8x RTX 3090, 128 cores, 774GB RAM, $1.12/hr
#
# Usage:
#   ./v3.3/deploy_sichuan.sh [--dry-run]
#
# ============================================================================

set -euo pipefail

# ── Constants ───────────────────────────────────────────────
INSTANCE_ID=33876301
TF="1d"
REMOTE_DIR="/workspace"
V33_REMOTE="/workspace/v3.3"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
V33_DIR="$PROJECT_DIR/v3.3"

# ── Color codes ─────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Parse args ──────────────────────────────────────────────
DRY_RUN=false
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=true
done

# ── Helpers ─────────────────────────────────────────────────
log_step()   { echo -e "${GREEN}[STEP]${NC} $1"; }
log_info()   { echo -e "${CYAN}[INFO]${NC} $1"; }
log_warn()   { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()  { echo -e "${RED}[ERROR]${NC} $1"; }
REMOTE_ENV_BASE="PYTHONUNBUFFERED=1 V30_DATA_DIR=/workspace/v3.3 SAVAGE22_DB_DIR=/workspace SAVAGE22_V1_DIR=/workspace"
REMOTE_ALLOW_CPU_DETECT="if python -c 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec(\"cudf\") else 1)' >/dev/null 2>&1; then unset ALLOW_CPU; else export ALLOW_CPU=1; fi"
log_header() {
    echo ""
    echo -e "${BOLD}${CYAN}============================================================${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BOLD}${CYAN}============================================================${NC}"
    echo ""
}

remote() {
    # Execute command on remote machine
    ssh -p "$SSH_PORT" -o ConnectTimeout=15 -o StrictHostKeyChecking=no "root@$SSH_HOST" "$@"
}

upload() {
    # SCP a file or glob to remote path
    local src="$1"
    local dst="$2"
    scp -P "$SSH_PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=15 "$src" "root@$SSH_HOST:$dst"
}

# ============================================================
# PHASE 0: Wait for SSH to be available
# ============================================================
log_header "PHASE 0: Wait for SSH (Instance $INSTANCE_ID)"

MAX_WAIT=300
WAIT_INTERVAL=10
ELAPSED=0

while true; do
    # Query vastai for instance SSH info
    INSTANCE_JSON=$(vastai show instances --raw 2>/dev/null || echo "[]")
    SSH_HOST=$(echo "$INSTANCE_JSON" | python -c "
import json, sys
data = json.load(sys.stdin)
for inst in data:
    if inst['id'] == $INSTANCE_ID:
        print(inst.get('ssh_host', ''))
        break
" 2>/dev/null || echo "")
    SSH_PORT=$(echo "$INSTANCE_JSON" | python -c "
import json, sys
data = json.load(sys.stdin)
for inst in data:
    if inst['id'] == $INSTANCE_ID:
        print(inst.get('ssh_port', ''))
        break
" 2>/dev/null || echo "")
    ACTUAL_STATUS=$(echo "$INSTANCE_JSON" | python -c "
import json, sys
data = json.load(sys.stdin)
for inst in data:
    if inst['id'] == $INSTANCE_ID:
        print(inst.get('actual_status', ''))
        break
" 2>/dev/null || echo "")

    if [[ -n "$SSH_HOST" && -n "$SSH_PORT" && "$SSH_HOST" != "" && "$SSH_PORT" != "" ]]; then
        # Try actual SSH connection
        if ssh -p "$SSH_PORT" -o ConnectTimeout=5 -o StrictHostKeyChecking=no "root@$SSH_HOST" 'echo SSH_OK' 2>/dev/null | grep -q "SSH_OK"; then
            log_step "SSH available at root@$SSH_HOST:$SSH_PORT (status: $ACTUAL_STATUS)"
            break
        fi
    fi

    ELAPSED=$((ELAPSED + WAIT_INTERVAL))
    if [[ $ELAPSED -ge $MAX_WAIT ]]; then
        log_error "SSH not available after ${MAX_WAIT}s. Instance status: $ACTUAL_STATUS"
        log_error "Host: $SSH_HOST, Port: $SSH_PORT"
        exit 1
    fi

    log_info "Waiting for SSH... (${ELAPSED}s / ${MAX_WAIT}s, status: $ACTUAL_STATUS)"
    sleep $WAIT_INTERVAL
done

# ── Banner ──────────────────────────────────────────────────
log_header "SICHUAN 8x RTX 3090 DEPLOYMENT"
echo -e "  Instance: ${BOLD}$INSTANCE_ID${NC}"
echo -e "  SSH:      ${BOLD}root@$SSH_HOST:$SSH_PORT${NC}"
echo -e "  Cost:     ${BOLD}\$1.12/hr${NC}"
echo -e "  Target:   ${BOLD}$TF${NC}"
echo -e "  Project:  ${BOLD}$PROJECT_DIR${NC}"
echo -e "  Dry run:  ${BOLD}$DRY_RUN${NC}"
echo ""

if $DRY_RUN; then
    log_warn "DRY RUN — no files will be transferred or commands executed"
fi

# ============================================================
# PHASE 1: Create remote directories
# ============================================================
log_header "PHASE 1: Create Remote Directories"

if ! $DRY_RUN; then
    remote "mkdir -p $V33_REMOTE"
    log_step "Created $V33_REMOTE"
else
    log_info "[DRY RUN] Would create $V33_REMOTE"
fi

# ============================================================
# PHASE 2: SCP v3.3/*.py files
# ============================================================
log_header "PHASE 2: Upload v3.3 Python Scripts"

PY_COUNT=0
if ! $DRY_RUN; then
    for f in "$V33_DIR"/*.py; do
        [[ -f "$f" ]] || continue
        upload "$f" "$V33_REMOTE/"
        PY_COUNT=$((PY_COUNT + 1))
    done
    log_step "Uploaded $PY_COUNT .py files to $V33_REMOTE/"
else
    PY_COUNT=$(ls "$V33_DIR"/*.py 2>/dev/null | wc -l)
    log_info "[DRY RUN] Would upload $PY_COUNT .py files"
fi

# ============================================================
# PHASE 3: SCP deploy_manifest.json + other JSON/txt data
# ============================================================
log_header "PHASE 3: Upload Manifest + Data Files"

DATA_FILES=(
    "$V33_DIR/deploy_manifest.json"
    "$V33_DIR/kp_history_gfz.txt"
    "$V33_DIR/ml_multi_tf_configs.json"
    "$V33_DIR/optuna_configs_all.json"
)

DATA_COUNT=0
for f in "${DATA_FILES[@]}"; do
    if [[ -f "$f" ]]; then
        if ! $DRY_RUN; then
            upload "$f" "$V33_REMOTE/"
        fi
        DATA_COUNT=$((DATA_COUNT + 1))
        log_info "  $(basename "$f")"
    else
        log_warn "  Missing: $(basename "$f")"
    fi
done
log_step "Uploaded $DATA_COUNT data files"

# ============================================================
# PHASE 4: SCP all *.db files from project root
# ============================================================
log_header "PHASE 4: Upload Database Files"

DB_COUNT=0
DB_LIST=""
for f in "$PROJECT_DIR"/*.db; do
    [[ -f "$f" ]] || continue
    fname=$(basename "$f")
    # Skip empty DBs
    fsize=$(stat -c%s "$f" 2>/dev/null || wc -c < "$f")
    if [[ "$fsize" -le 0 ]]; then
        log_warn "  Skipping empty: $fname"
        continue
    fi
    fsize_mb=$(echo "scale=1; $fsize / 1048576" | bc 2>/dev/null || echo "?")
    log_info "  $fname (${fsize_mb} MB)"
    if ! $DRY_RUN; then
        upload "$f" "$REMOTE_DIR/"
    fi
    DB_COUNT=$((DB_COUNT + 1))
    DB_LIST="$DB_LIST $fname"
done

# Also upload v3.3-specific DBs that aren't symlinks
for f in "$V33_DIR"/multi_asset_prices.db "$V33_DIR"/v2_signals.db "$V33_DIR"/llm_cache.db; do
    if [[ -f "$f" ]]; then
        fname=$(basename "$f")
        fsize=$(stat -c%s "$f" 2>/dev/null || wc -c < "$f")
        fsize_mb=$(echo "scale=1; $fsize / 1048576" | bc 2>/dev/null || echo "?")
        log_info "  v3.3/$fname (${fsize_mb} MB)"
        if ! $DRY_RUN; then
            upload "$f" "$REMOTE_DIR/"
        fi
        DB_COUNT=$((DB_COUNT + 1))
    fi
done

log_step "Uploaded $DB_COUNT database files"

# Verify DB count on remote
if ! $DRY_RUN; then
    REMOTE_DB_COUNT=$(remote "ls $REMOTE_DIR/*.db 2>/dev/null | wc -l")
    log_info "Remote DB count: $REMOTE_DB_COUNT (need >= 16)"
    if [[ "$REMOTE_DB_COUNT" -lt 16 ]]; then
        log_warn "Fewer than 16 DBs on remote. Some data may be missing."
    fi
fi

# ============================================================
# PHASE 5: Setup symlinks (DBs -> v3.3/)
# ============================================================
log_header "PHASE 5: Symlink DBs into v3.3/"

if ! $DRY_RUN; then
    remote "ln -sf $REMOTE_DIR/*.db $V33_REMOTE/ 2>/dev/null; echo SYMLINKS_OK"
    log_step "Symlinked all *.db from /workspace/ into /workspace/v3.3/"
else
    log_info "[DRY RUN] Would symlink $REMOTE_DIR/*.db -> $V33_REMOTE/"
fi

# ============================================================
# PHASE 6: Install OpenCL ICD
# ============================================================
log_header "PHASE 6: Install OpenCL ICD"

if ! $DRY_RUN; then
    remote "mkdir -p /etc/OpenCL/vendors && echo 'libnvidia-opencl.so.1' > /etc/OpenCL/vendors/nvidia.icd && echo OPENCL_ICD_OK"
    log_step "OpenCL ICD registered"
else
    log_info "[DRY RUN] Would register OpenCL ICD"
fi

# ============================================================
# PHASE 7: Install jemalloc
# ============================================================
log_header "PHASE 7: Install jemalloc"

if ! $DRY_RUN; then
    remote "apt-get update -qq && apt-get install -y -qq libjemalloc2 2>&1 | tail -3"
    log_step "jemalloc installed"
else
    log_info "[DRY RUN] Would install libjemalloc2"
fi

# ============================================================
# PHASE 8: Rebuild LightGBM with GPU support
# ============================================================
log_header "PHASE 8: Rebuild LightGBM with GPU"

if ! $DRY_RUN; then
    log_step "Installing libboost-dev + cmake (build deps)..."
    remote "apt-get install -y -qq libboost-dev libboost-system-dev libboost-filesystem-dev cmake ocl-icd-libopencl1 2>&1 | tail -5"

    log_step "Building LightGBM with GPU support (this takes 2-5 min)..."
    remote "pip install --force-reinstall --no-binary lightgbm 'lightgbm' --config-settings=cmake.define.USE_GPU=ON 2>&1 | tail -10"

    # Verify GPU device is available
    GPU_CHECK=$(remote "python -c \"
import lightgbm as lgb
print('LightGBM version:', lgb.__version__)
# Quick GPU test
import numpy as np
X = np.random.rand(100, 10).astype(np.float32)
y = np.random.randint(0, 2, 100)
ds = lgb.Dataset(X, y)
params = {'device': 'gpu', 'verbose': -1, 'num_leaves': 4, 'n_estimators': 2}
try:
    m = lgb.train(params, ds, num_boost_round=2)
    print('GPU device: OK')
except Exception as e:
    print(f'GPU device: FAIL - {e}')
\" 2>&1")
    echo "$GPU_CHECK"
    if echo "$GPU_CHECK" | grep -q "GPU device: OK"; then
        log_step "LightGBM GPU verified"
    else
        log_error "LightGBM GPU test FAILED"
        log_error "Check output above. Continuing anyway (deploy_verify will catch this)."
    fi
else
    log_info "[DRY RUN] Would rebuild LightGBM with GPU"
fi

# ============================================================
# PHASE 9: Install remaining pip packages + clear __pycache__
# ============================================================
log_header "PHASE 9: Pip Packages + Cleanup"

if ! $DRY_RUN; then
    log_step "Installing required pip packages..."
    remote "pip install --quiet scikit-learn scipy ephem astropy pytz joblib pandas numpy pyarrow optuna hmmlearn numba tqdm pyyaml 2>&1 | tail -5"

    log_step "Testing ALL imports..."
    IMPORT_CHECK=$(remote "python -c \"
import pandas, numpy, scipy, sklearn, lightgbm, ephem, astropy, pyarrow, optuna, numba, hmmlearn, yaml, tqdm
print('ALL IMPORTS OK')
\" 2>&1")
    echo "  $IMPORT_CHECK"

    log_step "Clearing __pycache__..."
    remote "find $V33_REMOTE -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; echo CACHE_CLEARED"
    remote "find $REMOTE_DIR -maxdepth 1 -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; echo CACHE_CLEARED"
    log_step "Cleanup complete"
else
    log_info "[DRY RUN] Would install pip packages and clear __pycache__"
fi

# ============================================================
# PHASE 10: Set environment variables
# ============================================================
log_header "PHASE 10: Environment Setup"

if ! $DRY_RUN; then
    remote "cat >> ~/.bashrc << 'ENVEOF'
export PYTHONUNBUFFERED=1
export V30_DATA_DIR=/workspace/v3.3
export SAVAGE22_DB_DIR=/workspace
export SAVAGE22_V1_DIR=/workspace
export OMP_NUM_THREADS=4
export NUMBA_NUM_THREADS=4
export MKL_DYNAMIC=FALSE
ENVEOF
echo ENV_OK"
    log_step "Environment variables set in ~/.bashrc"
else
    log_info "[DRY RUN] Would set env vars"
fi

# ============================================================
# PHASE 11: GPU + System Verification
# ============================================================
log_header "PHASE 11: GPU + System Verification"

if ! $DRY_RUN; then
    log_step "Checking GPUs..."
    remote "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader"
    GPU_COUNT=$(remote "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l")
    log_info "GPUs detected: $GPU_COUNT"

    log_step "Checking CPU + RAM..."
    remote "echo \"CPU cores: \$(nproc)\" && echo \"RAM: \$(free -g | awk '/Mem:/{print \$2}') GB\""
else
    log_info "[DRY RUN] Would verify GPUs and system"
fi

# ============================================================
# PHASE 12: Deploy Verification
# ============================================================
log_header "PHASE 12: deploy_verify.py --tf $TF"

if ! $DRY_RUN; then
    log_step "Running deploy_verify.py..."
    VERIFY_EXIT=0
    remote "cd $V33_REMOTE && $REMOTE_ALLOW_CPU_DETECT && $REMOTE_ENV_BASE python -u deploy_verify.py --tf $TF 2>&1" || VERIFY_EXIT=$?

    if [[ "$VERIFY_EXIT" -ne 0 ]]; then
        log_error "deploy_verify.py FAILED (exit $VERIFY_EXIT)"
        log_error "Fix issues before proceeding. SSH in with:"
        echo -e "  ssh -p $SSH_PORT root@$SSH_HOST"
        exit 1
    fi
    log_step "deploy_verify.py PASSED"
else
    log_info "[DRY RUN] Would run deploy_verify.py --tf $TF"
fi

# ============================================================
# PHASE 13: Pipeline Plumbing Test
# ============================================================
log_header "PHASE 13: test_pipeline_plumbing.py --tf $TF"

if ! $DRY_RUN; then
    log_step "Running pipeline plumbing test..."
    PLUMB_EXIT=0
    remote "cd $V33_REMOTE && $REMOTE_ALLOW_CPU_DETECT && $REMOTE_ENV_BASE python -u test_pipeline_plumbing.py --tf $TF 2>&1" || PLUMB_EXIT=$?

    if [[ "$PLUMB_EXIT" -ne 0 ]]; then
        log_error "test_pipeline_plumbing.py FAILED (exit $PLUMB_EXIT)"
        log_error "Fix issues before proceeding. SSH in with:"
        echo -e "  ssh -p $SSH_PORT root@$SSH_HOST"
        exit 1
    fi
    log_step "Pipeline plumbing test PASSED"
else
    log_info "[DRY RUN] Would run test_pipeline_plumbing.py --tf $TF"
fi

# ============================================================
# PHASE 14: Launch Training
# ============================================================
log_header "PHASE 14: Launch cloud_run_tf.py --symbol BTC --tf $TF"

LAUNCH_CMD="cd $V33_REMOTE && $REMOTE_ALLOW_CPU_DETECT && $REMOTE_ENV_BASE python -u cloud_run_tf.py --symbol BTC --tf $TF 2>&1 | tee /workspace/pipeline_${TF}.log"

if ! $DRY_RUN; then
    log_step "Launching training in tmux session 'train'..."
    remote "tmux kill-session -t train 2>/dev/null || true"
    remote "tmux new-session -d -s train \"cd $V33_REMOTE && $REMOTE_ALLOW_CPU_DETECT && $REMOTE_ENV_BASE python -u cloud_run_tf.py --symbol BTC --tf $TF 2>&1 | tee /workspace/pipeline_${TF}.log\""
    log_step "Training launched in tmux session 'train'"
else
    log_info "[DRY RUN] Would launch: $LAUNCH_CMD"
fi

# ============================================================
# SUMMARY
# ============================================================
log_header "DEPLOYMENT COMPLETE"

echo -e "  ${BOLD}Instance:${NC}  $INSTANCE_ID (Sichuan 8x RTX 3090)"
echo -e "  ${BOLD}SSH:${NC}       root@$SSH_HOST:$SSH_PORT"
echo -e "  ${BOLD}Cost:${NC}      \$1.12/hr"
echo -e "  ${BOLD}Pipeline:${NC}  cloud_run_tf.py --symbol BTC --tf $TF"
echo ""
echo -e "${BOLD}${CYAN}  MONITORING COMMANDS:${NC}"
echo ""
echo -e "  ${GREEN}# Attach to training session:${NC}"
echo -e "  ssh -p $SSH_PORT root@$SSH_HOST -t 'tmux attach -t train'"
echo ""
echo -e "  ${GREEN}# Tail the pipeline log:${NC}"
echo -e "  ssh -p $SSH_PORT root@$SSH_HOST 'tail -f /workspace/pipeline_${TF}.log'"
echo ""
echo -e "  ${GREEN}# Check GPU utilization:${NC}"
echo -e "  ssh -p $SSH_PORT root@$SSH_HOST 'nvidia-smi'"
echo ""
echo -e "  ${GREEN}# Download results when done:${NC}"
echo -e "  scp -P $SSH_PORT root@$SSH_HOST:/workspace/v3.3/model_${TF}.json ."
echo -e "  scp -P $SSH_PORT root@$SSH_HOST:/workspace/v3.3/optuna_configs_${TF}.json ."
echo -e "  scp -P $SSH_PORT 'root@$SSH_HOST:/workspace/v3.3/*${TF}*' ./v3.3/"
echo ""
echo -e "${YELLOW}  Cost: \$1.12/hr. Stop when done: vastai stop instance $INSTANCE_ID${NC}"
echo ""
