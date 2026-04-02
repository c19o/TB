#!/bin/bash
# ============================================================================
# deploy_to_cloud.sh — Automated vast.ai Upload & Launch for Savage22 Training
# ============================================================================
#
# Uploads all code, data, and checkpoints to a vast.ai instance, installs
# dependencies, verifies GPU, and launches v2_cloud_runner.py in tmux.
#
# Usage:
#   ./deploy_to_cloud.sh SSH_HOST SSH_PORT DPH [--dry-run]
#
# Examples:
#   ./deploy_to_cloud.sh root@ssh7.vast.ai 13562 2.88
#   ./deploy_to_cloud.sh root@ssh4.vast.ai 22345 1.15 --dry-run
#
# ============================================================================

set -euo pipefail

# ── Color codes ──────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ── Parse arguments ──────────────────────────────────────────
DRY_RUN=false

if [[ $# -lt 3 ]]; then
    echo -e "${RED}ERROR: Missing required arguments${NC}"
    echo ""
    echo "Usage: $0 SSH_HOST SSH_PORT DPH [--dry-run]"
    echo ""
    echo "  SSH_HOST  e.g. root@ssh7.vast.ai"
    echo "  SSH_PORT  e.g. 13562"
    echo "  DPH       cost per hour, e.g. 2.88"
    echo ""
    echo "Options:"
    echo "  --dry-run   Show what would be uploaded without doing it"
    exit 1
fi

SSH_HOST="$1"
SSH_PORT="$2"
DPH="$3"

# Check for --dry-run flag
for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then
        DRY_RUN=true
    fi
done

# ── Project paths ────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
V2_DIR="$PROJECT_DIR/v2"
REMOTE_DIR="/workspace"
SSH_CMD="ssh -p $SSH_PORT -o ConnectTimeout=15 -o StrictHostKeyChecking=no"
RSYNC_SSH="ssh -p $SSH_PORT -o StrictHostKeyChecking=no"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# ── Helper functions ─────────────────────────────────────────

log_header() {
    echo ""
    echo -e "${BOLD}${CYAN}============================================================${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BOLD}${CYAN}============================================================${NC}"
    echo ""
}

log_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

# Calculate file sizes for a list of files
calc_size() {
    local total=0
    for f in "$@"; do
        if [[ -f "$f" ]]; then
            local s
            s=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
            total=$((total + s))
        fi
    done
    echo $total
}

human_size() {
    local bytes=$1
    if (( bytes >= 1073741824 )); then
        echo "$(echo "scale=2; $bytes / 1073741824" | bc) GB"
    elif (( bytes >= 1048576 )); then
        echo "$(echo "scale=1; $bytes / 1048576" | bc) MB"
    elif (( bytes >= 1024 )); then
        echo "$(echo "scale=1; $bytes / 1024" | bc) KB"
    else
        echo "$bytes B"
    fi
}

# ── Banner ───────────────────────────────────────────────────
log_header "SAVAGE22 CLOUD DEPLOYMENT"
echo -e "  Host:     ${BOLD}$SSH_HOST${NC}"
echo -e "  Port:     ${BOLD}$SSH_PORT${NC}"
echo -e "  Cost:     ${BOLD}\$$DPH/hr${NC}"
echo -e "  Project:  ${BOLD}$PROJECT_DIR${NC}"
echo -e "  Remote:   ${BOLD}$REMOTE_DIR${NC}"
echo -e "  Time:     ${BOLD}$TIMESTAMP${NC}"
echo -e "  Dry run:  ${BOLD}$DRY_RUN${NC}"
echo ""

# ── Estimate total pipeline cost ─────────────────────────────
EST_HOURS="1.5"
EST_COST=$(echo "$DPH * $EST_HOURS" | bc)
log_info "Estimated pipeline duration: ~${EST_HOURS} hours"
log_info "Estimated cost: ~\$${EST_COST} at \$$DPH/hr"
echo ""

# ============================================================
# PHASE 0: SSH Connectivity Check
# ============================================================
log_header "PHASE 0: SSH Connectivity Check"

log_step "Testing SSH connection to $SSH_HOST:$SSH_PORT..."
if $DRY_RUN; then
    log_info "[DRY RUN] Would test: $SSH_CMD $SSH_HOST 'echo ok'"
else
    if $SSH_CMD "$SSH_HOST" 'echo "SSH_OK"' 2>/dev/null | grep -q "SSH_OK"; then
        log_step "SSH connection successful"
    else
        log_error "Cannot connect to $SSH_HOST:$SSH_PORT"
        log_error "Check that the instance is running and SSH key is configured"
        exit 1
    fi

    # Check remote disk space
    log_step "Checking remote disk space..."
    REMOTE_DISK=$($SSH_CMD "$SSH_HOST" 'df -h /workspace 2>/dev/null | tail -1' 2>/dev/null || echo "unknown")
    log_info "Remote disk: $REMOTE_DISK"
fi

# ============================================================
# PHASE 1: Upload V2 Scripts
# ============================================================
log_header "PHASE 1: Upload V2 Scripts Directory"

# V2 Python scripts (the main pipeline)
V2_SCRIPTS=(
    "$V2_DIR/v2_cloud_runner.py"
    "$V2_DIR/cloud_runner.py"
    "$V2_DIR/build_features_v2.py"
    "$V2_DIR/build_features_complete.py"
    "$V2_DIR/build_1w_features.py"
    "$V2_DIR/build_1d_features.py"
    "$V2_DIR/build_4h_features.py"
    "$V2_DIR/build_1h_features.py"
    "$V2_DIR/build_15m_features.py"
    "$V2_DIR/build_5m_features.py"
    "$V2_DIR/build_sports_features.py"
    "$V2_DIR/config.py"
    "$V2_DIR/data_access.py"
    "$V2_DIR/data_access_v2.py"
    "$V2_DIR/feature_library.py"
    "$V2_DIR/feature_classifier.py"
    "$V2_DIR/v2_feature_layers.py"
    "$V2_DIR/v2_cross_generator.py"
    "$V2_DIR/gpu_cross_builder.py"
    "$V2_DIR/knn_feature_engine.py"
    "$V2_DIR/ml_multi_tf.py"
    "$V2_DIR/exhaustive_optimizer.py"
    "$V2_DIR/lstm_sequence_model.py"
    "$V2_DIR/v2_lstm_trainer.py"
    "$V2_DIR/v2_multi_asset_trainer.py"
    "$V2_DIR/backtest_validation.py"
    "$V2_DIR/backtesting_audit.py"
    "$V2_DIR/meta_labeling.py"
    "$V2_DIR/smoke_test_pipeline.py"
    "$V2_DIR/verify_gpu_features.py"
    "$V2_DIR/universal_gematria.py"
    "$V2_DIR/universal_numerology.py"
    "$V2_DIR/universal_astro.py"
    "$V2_DIR/universal_sentiment.py"
    "$V2_DIR/live_trader.py"
    "$V2_DIR/bitget_api.py"
    "$V2_DIR/portfolio_aggregator.py"
    "$V2_DIR/hardware_detect.py"
    "$V2_DIR/gcp_feature_builder.py"
    "$V2_DIR/llm_features.py"
    "$V2_DIR/lstm_precursor_model.py"
    "$V2_DIR/run_full_pipeline.sh"
)

log_step "Uploading V2 scripts via rsync..."
SCRIPT_COUNT=0
for f in "${V2_SCRIPTS[@]}"; do
    if [[ -f "$f" ]]; then
        SCRIPT_COUNT=$((SCRIPT_COUNT + 1))
    else
        log_warn "Missing: $f"
    fi
done
log_info "Found $SCRIPT_COUNT V2 scripts to upload"

if $DRY_RUN; then
    log_info "[DRY RUN] Would rsync $SCRIPT_COUNT V2 scripts to $REMOTE_DIR/"
else
    # Use rsync with include/exclude for the V2 directory
    rsync -avz --progress \
        -e "$RSYNC_SSH" \
        --include='*.py' \
        --include='*.sh' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='*.log' \
        --exclude='*.db' \
        --exclude='*.parquet' \
        --exclude='*.npz' \
        --exclude='*.html' \
        --exclude='*.html_Files/' \
        --exclude='node_modules/' \
        --exclude='dashboard/' \
        --exclude='.git/' \
        "$V2_DIR/" "$SSH_HOST:$REMOTE_DIR/" \
        || { log_error "V2 scripts rsync failed"; exit 1; }
    log_step "V2 scripts uploaded successfully"
fi

# ============================================================
# PHASE 2: Upload V1 Scripts (imported by V2)
# ============================================================
log_header "PHASE 2: Upload V1 Scripts (dependencies)"

# V1 scripts that V2 build scripts import
V1_SCRIPTS=(
    "$PROJECT_DIR/feature_library.py"
    "$PROJECT_DIR/data_access.py"
    "$PROJECT_DIR/universal_gematria.py"
    "$PROJECT_DIR/universal_numerology.py"
    "$PROJECT_DIR/universal_astro.py"
    "$PROJECT_DIR/universal_sentiment.py"
    "$PROJECT_DIR/gpu_cross_builder.py"
    "$PROJECT_DIR/knn_feature_engine.py"
    "$PROJECT_DIR/ml_multi_tf.py"
    "$PROJECT_DIR/exhaustive_optimizer.py"
    "$PROJECT_DIR/lstm_sequence_model.py"
    "$PROJECT_DIR/backtest_validation.py"
    "$PROJECT_DIR/backtesting_audit.py"
    "$PROJECT_DIR/meta_labeling.py"
    "$PROJECT_DIR/smoke_test_pipeline.py"
    "$PROJECT_DIR/cloud_runner.py"
    "$PROJECT_DIR/config.py"
    "$PROJECT_DIR/bitget_api.py"
    "$PROJECT_DIR/gcp_feature_builder.py"
    "$PROJECT_DIR/llm_features.py"
    "$PROJECT_DIR/hardware_detect.py"
    "$PROJECT_DIR/portfolio_aggregator.py"
    "$PROJECT_DIR/verify_gpu_features.py"
    "$PROJECT_DIR/trade_journal.py"
    "$PROJECT_DIR/self_learner.py"
    "$PROJECT_DIR/retrain_scheduler.py"
    "$PROJECT_DIR/hypothetical_tracker.py"
)

log_step "Uploading V1 scripts (feature_library.py, data_access.py, universal_*, etc.)..."
V1_COUNT=0
V1_UPLOAD_LIST=()
for f in "${V1_SCRIPTS[@]}"; do
    if [[ -f "$f" ]]; then
        V1_COUNT=$((V1_COUNT + 1))
        V1_UPLOAD_LIST+=("$f")
    else
        log_warn "Missing V1 script: $(basename "$f")"
    fi
done
log_info "Found $V1_COUNT V1 scripts to upload"

if $DRY_RUN; then
    log_info "[DRY RUN] Would upload $V1_COUNT V1 scripts to $REMOTE_DIR/"
    for f in "${V1_UPLOAD_LIST[@]}"; do
        log_info "  $(basename "$f")"
    done
else
    # Upload V1 scripts individually (they go to /workspace root alongside V2)
    for f in "${V1_UPLOAD_LIST[@]}"; do
        rsync -avz -e "$RSYNC_SSH" "$f" "$SSH_HOST:$REMOTE_DIR/" \
            || log_warn "Failed to upload $(basename "$f")"
    done
    log_step "V1 scripts uploaded successfully"
fi

# ============================================================
# PHASE 3: Upload Database Files
# ============================================================
log_header "PHASE 3: Upload Database Files"

# All databases needed for training
DATABASES=(
    "$PROJECT_DIR/btc_prices.db"
    "$PROJECT_DIR/astrology_full.db"
    "$PROJECT_DIR/ephemeris_cache.db"
    "$PROJECT_DIR/fear_greed.db"
    "$PROJECT_DIR/funding_rates.db"
    "$PROJECT_DIR/google_trends.db"
    "$PROJECT_DIR/news_articles.db"
    "$PROJECT_DIR/tweets.db"
    "$PROJECT_DIR/macro_data.db"
    "$PROJECT_DIR/onchain_data.db"
    "$PROJECT_DIR/open_interest.db"
    "$PROJECT_DIR/space_weather.db"
    "$PROJECT_DIR/sports_results.db"
    "$PROJECT_DIR/llm_cache.db"
    "$PROJECT_DIR/trade_journal.db"
    "$PROJECT_DIR/features_complete.db"
    "$PROJECT_DIR/features_1d.db"
    "$PROJECT_DIR/features_1w.db"
    "$PROJECT_DIR/features_4h.db"
    "$PROJECT_DIR/features_1h.db"
    "$PROJECT_DIR/features_15m.db"
    "$PROJECT_DIR/features_5m.db"
)

# Also V2 databases
V2_DATABASES=(
    "$V2_DIR/v2_signals.db"
    "$V2_DIR/multi_asset_prices.db"
)

DB_TOTAL_SIZE=0
DB_COUNT=0
DB_UPLOAD_LIST=()

for f in "${DATABASES[@]}" "${V2_DATABASES[@]}"; do
    if [[ -f "$f" ]]; then
        DB_COUNT=$((DB_COUNT + 1))
        DB_UPLOAD_LIST+=("$f")
        fsize=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
        DB_TOTAL_SIZE=$((DB_TOTAL_SIZE + fsize))
    else
        log_warn "Missing DB: $(basename "$f")"
    fi
done

log_info "Found $DB_COUNT databases ($(human_size $DB_TOTAL_SIZE) total)"

if $DRY_RUN; then
    log_info "[DRY RUN] Would upload $DB_COUNT databases"
    for f in "${DB_UPLOAD_LIST[@]}"; do
        fsize=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
        log_info "  $(basename "$f") ($(human_size $fsize))"
    done
else
    log_step "Uploading databases (this may take a while for large DBs)..."
    for f in "${DB_UPLOAD_LIST[@]}"; do
        fname=$(basename "$f")
        fsize=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
        log_info "Uploading $fname ($(human_size $fsize))..."

        # V2 databases go to /workspace (same dir as V2 scripts)
        rsync -avz --progress -e "$RSYNC_SSH" "$f" "$SSH_HOST:$REMOTE_DIR/" \
            || { log_error "Failed to upload $fname"; exit 1; }
    done
    log_step "All databases uploaded"
fi

# ============================================================
# PHASE 4: Upload Data Files (configs, CSVs, JSON)
# ============================================================
log_header "PHASE 4: Upload Data & Config Files"

DATA_FILES=(
    # Core data files
    "$PROJECT_DIR/kp_history.txt"
    "$PROJECT_DIR/dst_index.json"
    "$PROJECT_DIR/dynamic_config.json"
    "$PROJECT_DIR/systematic_cross_results_1h.csv"
    "$PROJECT_DIR/systematic_cross_results_4h.csv"
    "$PROJECT_DIR/systematic_cross_results_all.csv"
    "$PROJECT_DIR/ml_multi_tf_configs.json"
    "$PROJECT_DIR/hypotheses.json"
    "$PROJECT_DIR/geomag_storms.json"
    "$PROJECT_DIR/hebrew_calendar.json"
    "$PROJECT_DIR/signal_calendar.json"
    "$PROJECT_DIR/heartmath_power_data.json"
    "$PROJECT_DIR/solar_flares_recent.json"
    "$PROJECT_DIR/solar_flux_recent.json"
    "$PROJECT_DIR/solar_wind.json"
    "$PROJECT_DIR/kp_7day.json"
    "$PROJECT_DIR/kp_noaa_daily.json"
    "$PROJECT_DIR/schumann_wp_api.json"
    # Feature list JSONs
    "$PROJECT_DIR/features_1w_all.json"
    "$PROJECT_DIR/features_1d_all.json"
    "$PROJECT_DIR/features_4h_all.json"
    "$PROJECT_DIR/features_1h_all.json"
    "$PROJECT_DIR/features_15m_all.json"
    "$PROJECT_DIR/features_5m_all.json"
    "$PROJECT_DIR/features_1d_pruned.json"
    "$PROJECT_DIR/features_4h_pruned.json"
    "$PROJECT_DIR/features_1h_pruned.json"
    "$PROJECT_DIR/features_15m_pruned.json"
    "$PROJECT_DIR/features_5m_pruned.json"
    # Model configs
    "$PROJECT_DIR/model_1w.json"
    "$PROJECT_DIR/model_1d.json"
    "$PROJECT_DIR/model_4h.json"
    "$PROJECT_DIR/model_1h.json"
    "$PROJECT_DIR/model_15m.json"
    "$PROJECT_DIR/model_5m.json"
    # V2 data files
    "$V2_DIR/kp_history.txt"
    "$V2_DIR/dynamic_config.json"
    "$V2_DIR/ml_multi_tf_configs.json"
    "$V2_DIR/hypotheses.json"
)

DATA_COUNT=0
DATA_UPLOAD=()
for f in "${DATA_FILES[@]}"; do
    if [[ -f "$f" ]]; then
        DATA_COUNT=$((DATA_COUNT + 1))
        DATA_UPLOAD+=("$f")
    fi
done
log_info "Found $DATA_COUNT data/config files"

if $DRY_RUN; then
    log_info "[DRY RUN] Would upload $DATA_COUNT data files"
else
    for f in "${DATA_UPLOAD[@]}"; do
        rsync -avz -e "$RSYNC_SSH" "$f" "$SSH_HOST:$REMOTE_DIR/" \
            || log_warn "Failed: $(basename "$f")"
    done
    log_step "Data files uploaded"
fi

# ============================================================
# PHASE 5: Upload Parquets & NPZ (checkpoints/resume)
# ============================================================
log_header "PHASE 5: Upload Parquets & NPZ (checkpoint/resume)"

PARQUET_COUNT=0
PARQUET_SIZE=0
NPZ_COUNT=0
NPZ_SIZE=0

# Count V1 parquets
while IFS= read -r -d '' f; do
    PARQUET_COUNT=$((PARQUET_COUNT + 1))
    fsize=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
    PARQUET_SIZE=$((PARQUET_SIZE + fsize))
done < <(find "$PROJECT_DIR" -maxdepth 1 -name '*.parquet' -print0 2>/dev/null)

# Count V2 parquets
while IFS= read -r -d '' f; do
    PARQUET_COUNT=$((PARQUET_COUNT + 1))
    fsize=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
    PARQUET_SIZE=$((PARQUET_SIZE + fsize))
done < <(find "$V2_DIR" -maxdepth 1 -name '*.parquet' -print0 2>/dev/null)

# Count V2 NPZ files
while IFS= read -r -d '' f; do
    NPZ_COUNT=$((NPZ_COUNT + 1))
    fsize=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
    NPZ_SIZE=$((NPZ_SIZE + fsize))
done < <(find "$V2_DIR" -maxdepth 1 -name '*.npz' -print0 2>/dev/null)

# Count V2 cross name JSONs
CROSS_JSON_COUNT=0
while IFS= read -r -d '' f; do
    CROSS_JSON_COUNT=$((CROSS_JSON_COUNT + 1))
done < <(find "$V2_DIR" -maxdepth 1 -name 'v2_cross_names_*.json' -print0 2>/dev/null)

log_info "Found $PARQUET_COUNT parquets ($(human_size $PARQUET_SIZE))"
log_info "Found $NPZ_COUNT .npz files ($(human_size $NPZ_SIZE))"
log_info "Found $CROSS_JSON_COUNT cross-name JSON files"

if $DRY_RUN; then
    log_info "[DRY RUN] Would upload parquets, NPZ, and cross-name JSONs"
else
    # Upload V1 parquets
    if compgen -G "$PROJECT_DIR/*.parquet" > /dev/null 2>&1; then
        log_step "Uploading V1 parquets..."
        rsync -avz --progress -e "$RSYNC_SSH" \
            "$PROJECT_DIR/"*.parquet "$SSH_HOST:$REMOTE_DIR/" \
            || log_warn "Some V1 parquets failed"
    fi

    # Upload V2 parquets
    if compgen -G "$V2_DIR/*.parquet" > /dev/null 2>&1; then
        log_step "Uploading V2 parquets..."
        rsync -avz --progress -e "$RSYNC_SSH" \
            "$V2_DIR/"*.parquet "$SSH_HOST:$REMOTE_DIR/" \
            || log_warn "Some V2 parquets failed"
    fi

    # Upload V2 NPZ files (sparse cross matrices)
    if compgen -G "$V2_DIR/*.npz" > /dev/null 2>&1; then
        log_step "Uploading V2 .npz sparse matrices..."
        rsync -avz --progress -e "$RSYNC_SSH" \
            "$V2_DIR/"*.npz "$SSH_HOST:$REMOTE_DIR/" \
            || log_warn "Some NPZ files failed"
    fi

    # SKIP v2_cross_names_*.json — 1.8GB, not needed for training
    # (cross names are regenerated at inference time by v2_cross_generator.py)
    log_step "Skipping v2_cross_names_*.json (1.8GB, not needed for training)"

    log_step "Parquets & NPZ uploaded"
fi

# ============================================================
# PHASE 6: Upload Pipeline Manifest (if exists, for resume)
# ============================================================
log_header "PHASE 6: Upload Pipeline Manifest (resume support)"

MANIFEST="$V2_DIR/pipeline_manifest.json"
if [[ -f "$MANIFEST" ]]; then
    log_info "Found existing pipeline_manifest.json — uploading for resume"
    if ! $DRY_RUN; then
        rsync -avz -e "$RSYNC_SSH" "$MANIFEST" "$SSH_HOST:$REMOTE_DIR/" \
            || log_warn "Manifest upload failed"
    fi
else
    log_info "No pipeline_manifest.json found (fresh run)"
fi

# Also upload any existing model files (.xgb, .pkl, .pt)
MODEL_COUNT=0
for ext in xgb pkl pt pth json; do
    while IFS= read -r -d '' f; do
        MODEL_COUNT=$((MODEL_COUNT + 1))
        if ! $DRY_RUN; then
            rsync -avz -e "$RSYNC_SSH" "$f" "$SSH_HOST:$REMOTE_DIR/" 2>/dev/null || true
        fi
    done < <(find "$V2_DIR" -maxdepth 1 -name "*model*.$ext" -print0 2>/dev/null)
    while IFS= read -r -d '' f; do
        MODEL_COUNT=$((MODEL_COUNT + 1))
        if ! $DRY_RUN; then
            rsync -avz -e "$RSYNC_SSH" "$f" "$SSH_HOST:$REMOTE_DIR/" 2>/dev/null || true
        fi
    done < <(find "$V2_DIR" -maxdepth 1 -name "*blend_config*.$ext" -print0 2>/dev/null)
    while IFS= read -r -d '' f; do
        MODEL_COUNT=$((MODEL_COUNT + 1))
        if ! $DRY_RUN; then
            rsync -avz -e "$RSYNC_SSH" "$f" "$SSH_HOST:$REMOTE_DIR/" 2>/dev/null || true
        fi
    done < <(find "$V2_DIR" -maxdepth 1 -name "*exhaustive_config*.$ext" -print0 2>/dev/null)
done

if [[ $MODEL_COUNT -gt 0 ]]; then
    log_info "Uploaded $MODEL_COUNT existing model/config files"
else
    log_info "No existing model files found (fresh training)"
fi

# ============================================================
# PHASE 7: Remote Setup (pip install, GPU check)
# ============================================================
log_header "PHASE 7: Remote Setup"

if $DRY_RUN; then
    log_info "[DRY RUN] Would install packages and verify GPU"
    log_info "  pip install: lightgbm scikit-learn scipy hmmlearn psutil numba optuna torch"
    log_info "  nvidia-smi check"
    log_info "  cudf/cupy import check"
else
    log_step "Installing missing pip packages..."
    $SSH_CMD "$SSH_HOST" bash -c "'
        set -e
        echo \"=== Installing pip packages ===\"
        pip install --quiet lightgbm scikit-learn scipy hmmlearn psutil numba optuna torch 2>&1 | tail -5
        echo \"=== pip install complete ===\"
    '" 2>&1 || { log_error "pip install failed"; exit 1; }
    log_step "Packages installed"

    log_step "Setting env vars (PYTHONUNBUFFERED, OMP_NUM_THREADS=4)..."
    $SSH_CMD "$SSH_HOST" bash -c "'
        echo \"export PYTHONUNBUFFERED=1\" >> ~/.bashrc
        echo \"export PYTHONUNBUFFERED=1\" >> ~/.profile
        echo \"export OMP_NUM_THREADS=4\" >> ~/.bashrc
        echo \"export OMP_NUM_THREADS=4\" >> ~/.profile
        echo \"export NUMBA_NUM_THREADS=4\" >> ~/.bashrc
        echo \"export NUMBA_NUM_THREADS=4\" >> ~/.profile
        echo \"export MKL_DYNAMIC=FALSE\" >> ~/.bashrc
        echo \"export MKL_DYNAMIC=FALSE\" >> ~/.profile
    '" 2>&1 || log_warn "Could not set env vars in profile"

    log_step "Verifying GPU visibility..."
    GPU_INFO=$($SSH_CMD "$SSH_HOST" 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1' 2>&1) || true
    if [[ -z "$GPU_INFO" || "$GPU_INFO" == *"not found"* ]]; then
        log_error "nvidia-smi not available — no GPU detected!"
        log_error "Aborting. Check that the instance has GPU support."
        exit 1
    fi
    echo ""
    echo -e "${BOLD}  GPU(s) detected:${NC}"
    GPU_NUM=0
    while IFS= read -r line; do
        echo -e "    GPU $GPU_NUM: $line"
        GPU_NUM=$((GPU_NUM + 1))
    done <<< "$GPU_INFO"
    echo ""
    log_info "Total GPUs: $GPU_NUM"

    log_step "Verifying RAPIDS (cudf/cupy) imports..."
    RAPIDS_OK=$($SSH_CMD "$SSH_HOST" bash -c "'
        python -c \"
import sys
try:
    import cudf
    print(f'cudf {cudf.__version__}', end=' ')
except ImportError:
    print('cudf MISSING', end=' ')
try:
    import cupy
    print(f'cupy {cupy.__version__}', end=' ')
except ImportError:
    print('cupy MISSING', end=' ')
try:
    import xgboost
    print(f'xgb {xgboost.__version__}')
except ImportError:
    print('xgboost MISSING')
\" 2>&1
    '" 2>&1) || true
    log_info "RAPIDS check: $RAPIDS_OK"

    if [[ "$RAPIDS_OK" == *"MISSING"* ]]; then
        log_warn "Some RAPIDS packages missing — pipeline may use CPU fallback for some ops"
        log_warn "Consider using rapidsai/base docker image for full GPU acceleration"
    fi
fi

# ============================================================
# PHASE 8: Verify Upload
# ============================================================
log_header "PHASE 8: Verify Upload"

if $DRY_RUN; then
    log_info "[DRY RUN] Would verify file counts on remote"
else
    log_step "Counting files on remote..."
    REMOTE_COUNTS=$($SSH_CMD "$SSH_HOST" bash -c "'
        cd /workspace 2>/dev/null || true
        echo \"Python scripts: \$(find . -maxdepth 1 -name \"*.py\" | wc -l)\"
        echo \"Database files: \$(find . -maxdepth 1 -name \"*.db\" | wc -l)\"
        echo \"Parquet files:  \$(find . -maxdepth 1 -name \"*.parquet\" | wc -l)\"
        echo \"NPZ files:     \$(find . -maxdepth 1 -name \"*.npz\" | wc -l)\"
        echo \"JSON files:    \$(find . -maxdepth 1 -name \"*.json\" | wc -l)\"
        echo \"Total size:    \$(du -sh . 2>/dev/null | cut -f1)\"
    '" 2>&1) || true
    echo ""
    echo -e "${BOLD}  Remote file counts:${NC}"
    echo "$REMOTE_COUNTS" | while IFS= read -r line; do
        echo "    $line"
    done
    echo ""
fi

# ============================================================
# PHASE 9: Launch Training Pipeline
# ============================================================
log_header "PHASE 9: Launch Training Pipeline"

LAUNCH_CMD="cd $REMOTE_DIR && PYTHONUNBUFFERED=1 python v2_cloud_runner.py --dph $DPH 2>&1 | tee pipeline.log"

if $DRY_RUN; then
    log_info "[DRY RUN] Would launch in tmux:"
    log_info "  tmux new-session -d -s train '$LAUNCH_CMD'"
    echo ""
    log_info "[DRY RUN] Complete. No files were transferred."
else
    log_step "Creating tmux session 'train' and launching pipeline..."
    $SSH_CMD "$SSH_HOST" "tmux kill-session -t train 2>/dev/null; tmux new-session -d -s train '$LAUNCH_CMD'" 2>&1 \
        || { log_error "Failed to launch tmux session"; exit 1; }
    log_step "Pipeline launched in tmux session 'train'"
fi

# ============================================================
# SUMMARY
# ============================================================
log_header "DEPLOYMENT COMPLETE"

echo -e "  ${BOLD}Instance:${NC}  $SSH_HOST:$SSH_PORT"
echo -e "  ${BOLD}Cost:${NC}      \$$DPH/hr (estimated ~\$$EST_COST total)"
echo -e "  ${BOLD}Pipeline:${NC}  v2_cloud_runner.py --dph $DPH"
echo ""
echo -e "${BOLD}${CYAN}  MONITORING COMMANDS:${NC}"
echo ""
echo -e "  ${GREEN}# Attach to training session:${NC}"
echo -e "  ssh -p $SSH_PORT $SSH_HOST -t 'tmux attach -t train'"
echo ""
echo -e "  ${GREEN}# Tail the pipeline log:${NC}"
echo -e "  ssh -p $SSH_PORT $SSH_HOST 'tail -f /workspace/pipeline.log'"
echo ""
echo -e "  ${GREEN}# Check GPU utilization:${NC}"
echo -e "  ssh -p $SSH_PORT $SSH_HOST 'nvidia-smi'"
echo ""
echo -e "  ${GREEN}# Watch GPU continuously:${NC}"
echo -e "  ssh -p $SSH_PORT $SSH_HOST 'watch -n 2 nvidia-smi'"
echo ""
echo -e "  ${GREEN}# Check pipeline manifest:${NC}"
echo -e "  ssh -p $SSH_PORT $SSH_HOST 'cat /workspace/pipeline_manifest.json | python -m json.tool'"
echo ""
echo -e "  ${GREEN}# Monitor with companion script:${NC}"
echo -e "  ./monitor_cloud.sh $SSH_HOST $SSH_PORT $DPH"
echo ""
echo -e "  ${GREEN}# Download results when done:${NC}"
echo -e "  rsync -avz -e 'ssh -p $SSH_PORT' $SSH_HOST:/workspace/*.json $SSH_HOST:/workspace/*.pkl $SSH_HOST:/workspace/*.xgb ."
echo ""
echo -e "${YELLOW}  Remember: Pipeline cost is \$$DPH/hr. Stop the instance when done!${NC}"
echo -e "${YELLOW}  vastai stop instance <INSTANCE_ID>${NC}"
echo ""
