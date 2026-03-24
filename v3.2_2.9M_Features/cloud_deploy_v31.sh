#!/bin/bash
# cloud_deploy_v31.sh — Deploy + run full 7-step pipeline on vast.ai
# Usage: bash cloud_deploy_v31.sh <SSH_HOST> <SSH_PORT>
# Example: bash cloud_deploy_v31.sh root@ssh5.vast.ai 12345

set -e
HOST=$1
PORT=$2

if [ -z "$HOST" ] || [ -z "$PORT" ]; then
    echo "Usage: bash cloud_deploy_v31.sh <SSH_HOST> <SSH_PORT>"
    exit 1
fi

SSH="ssh -o StrictHostKeyChecking=no -p $PORT $HOST"
SCP="scp -o StrictHostKeyChecking=no -P $PORT"

echo "========================================================"
echo "  SAVAGE22 v3.1 — CLOUD DEPLOYMENT"
echo "  Target: $HOST:$PORT"
echo "  $(date)"
echo "========================================================"

# ── Step 1: Upload compressed package ──
echo ""
echo "=== STEP 1: Upload (compressed tar.gz) ==="
UPLOAD_START=$(date +%s)
$SCP ../v31_cloud_upload.tar.gz $HOST:/tmp/v31_cloud_upload.tar.gz
UPLOAD_END=$(date +%s)
echo "  Upload done: $(( UPLOAD_END - UPLOAD_START ))s"

# ── Step 2: Extract + setup on cloud ──
echo ""
echo "=== STEP 2: Extract + install deps ==="
$SSH << 'SETUP_EOF'
set -e
mkdir -p /workspace/savage22
cd /workspace/savage22

# Extract
tar xzf /tmp/v31_cloud_upload.tar.gz --strip-components=0
rm /tmp/v31_cloud_upload.tar.gz

# Move v3.0 data to expected location
mkdir -p "v3.0 (LGBM)"
mv -f "v3.0 (LGBM)"/* "v3.0 (LGBM)/" 2>/dev/null || true

# Move v3.1 code to workspace root
cp -f v3.1/*.py .
cp -f v3.1/*.json . 2>/dev/null || true
cp -f v3.1/*.sh . 2>/dev/null || true

# Move V1 DBs to parent (config.py V1_DIR = parent of PROJECT_DIR)
for db in tweets.db news_articles.db astrology_full.db ephemeris_cache.db \
          fear_greed.db sports_results.db space_weather.db macro_data.db \
          onchain_data.db funding_rates.db open_interest.db google_trends.db; do
    [ -f "$db" ] && mv -f "$db" ../
done

# Move multi_asset_prices.db to workspace
[ -f "v3.0 (LGBM)/multi_asset_prices.db" ] && cp "v3.0 (LGBM)/multi_asset_prices.db" .

# Symlink parquets from v3.0 to local dir
for f in "v3.0 (LGBM)"/features_*.parquet; do
    base=$(basename "$f")
    [ ! -f "$base" ] && ln -sf "$f" "$base"
done

# Install deps
pip install -q lightgbm optuna scipy numba hmmlearn scikit-learn torch 2>&1 | tail -3
pip install -q psutil 2>&1 | tail -1

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}')" 2>/dev/null || echo "No GPU detected"
python -c "import lightgbm; print(f'LightGBM {lightgbm.__version__}')"
python -c "import optuna; print(f'Optuna {optuna.__version__}')"

# Set env vars
export PYTHONUNBUFFERED=1
export V30_DATA_DIR="/workspace/savage22/v3.0 (LGBM)"
export SAVAGE22_V1_DIR="/workspace"

echo "Setup complete. Files:"
ls -la *.parquet 2>/dev/null | wc -l
echo " parquets"
ls -la *.py | wc -l
echo " python files"
SETUP_EOF

echo "  Setup done"

# ── Step 3: Run full pipeline ──
echo ""
echo "=== STEP 3: Full 7-step pipeline ==="
TRAIN_START=$(date +%s)

$SSH << 'PIPELINE_EOF'
cd /workspace/savage22
export PYTHONUNBUFFERED=1
export V30_DATA_DIR="/workspace/savage22/v3.0 (LGBM)"
export SAVAGE22_V1_DIR="/workspace"

# Run pipeline: 1w, 1d, 4h, 1h first (features exist), then 15m (needs building)
echo "Starting pipeline at $(date)"

# Phase by phase for visibility
for TF in 1w 1d 4h 1h; do
    echo ""
    echo "===== $TF: TRAIN + OPTUNA + META + LSTM + PBO + AUDIT ====="
    python -u pipeline_orchestrator.py --tf $TF 2>&1
done

# 15m last (needs feature building)
echo ""
echo "===== 15m: FULL 7-STEP (build + train + optuna + meta + lstm + pbo + audit) ====="
python -u pipeline_orchestrator.py --tf 15m 2>&1

echo ""
echo "Pipeline complete at $(date)"
python -u pipeline_orchestrator.py --status
PIPELINE_EOF

TRAIN_END=$(date +%s)
echo "  Pipeline done: $(( TRAIN_END - TRAIN_START ))s ($(( (TRAIN_END - TRAIN_START) / 60 )) min)"

# ── Step 4: Download results ──
echo ""
echo "=== STEP 4: Download results ==="
mkdir -p ../v31_cloud_results
$SCP -r "$HOST:/workspace/savage22/model_*.json" ../v31_cloud_results/
$SCP -r "$HOST:/workspace/savage22/features_*_all.json" ../v31_cloud_results/
$SCP -r "$HOST:/workspace/savage22/optuna_configs_*.json" ../v31_cloud_results/
$SCP -r "$HOST:/workspace/savage22/meta_model_*.pkl" ../v31_cloud_results/
$SCP -r "$HOST:/workspace/savage22/platt_*.pkl" ../v31_cloud_results/
$SCP -r "$HOST:/workspace/savage22/lstm_*.pt" ../v31_cloud_results/ 2>/dev/null || true
$SCP -r "$HOST:/workspace/savage22/pbo_results_*.json" ../v31_cloud_results/ 2>/dev/null || true
$SCP -r "$HOST:/workspace/savage22/audit_*.json" ../v31_cloud_results/ 2>/dev/null || true
$SCP -r "$HOST:/workspace/savage22/pipeline_manifest.json" ../v31_cloud_results/
$SCP -r "$HOST:/workspace/savage22/cpcv_oos_predictions_*.pkl" ../v31_cloud_results/ 2>/dev/null || true

echo ""
echo "=== RESULTS ==="
ls -lh ../v31_cloud_results/

TOTAL_END=$(date +%s)
echo ""
echo "========================================================"
echo "  TOTAL TIME: $(( TOTAL_END - UPLOAD_START ))s ($(( (TOTAL_END - UPLOAD_START) / 60 )) min)"
echo "  Upload: $(( UPLOAD_END - UPLOAD_START ))s"
echo "  Pipeline: $(( TRAIN_END - TRAIN_START ))s"
echo "========================================================"
