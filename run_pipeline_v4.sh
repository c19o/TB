#!/bin/bash
# run_pipeline_v4.sh — Dynamic GPU pipeline for 8x H200
# All builds parallel on CPU. As each finishes, trains with ALL GPUs via Dask.
# Finished GPUs always help remaining work.
set -e
cd /workspace/savage22
export PYTHONUNBUFFERED=1

echo "========================================================"
echo "  SAVAGE22 PIPELINE v4 — $(date)"
echo "  8x H200 | Dynamic GPU pooling"
echo "  All builds parallel → train each with ALL GPUs as ready"
echo "========================================================"
START=$(date +%s)

# ═══ PHASE 1: ALL BUILDS PARALLEL (CPU + CuPy for crosses) ═══
echo ""
echo "=== ALL 6 BUILDS LAUNCHING PARALLEL ==="

# Stagger 5M to run solo after others (RAM safety)
# Small TFs + 1H + 15M parallel, then 5M
python -u build_1w_features.py > build_1w.log 2>&1 &
PID_1W=$!
python -u build_1d_features.py > build_1d.log 2>&1 &
PID_1D=$!
python -u build_4h_features.py > build_4h.log 2>&1 &
PID_4H=$!
python -u build_1h_features.py > build_1h.log 2>&1 &
PID_1H=$!
python -u build_15m_features.py > build_15m.log 2>&1 &
PID_15M=$!

echo "  Launched: 1W 1D 4H 1H 15M (5M waits for RAM)"

# Track which TFs are ready to train
TRAINED=""

train_if_ready() {
    local tf=$1
    local parquet="features_${tf}.parquet"
    if [ -f "$parquet" ] && [[ ! "$TRAINED" == *"$tf"* ]]; then
        SIZE=$(du -h "$parquet" | cut -f1)
        echo ""
        echo "  *** ${tf} BUILD DONE (${SIZE}) — TRAINING WITH ALL GPUs ***"
        T0=$(date +%s)
        python -u ml_multi_tf.py --tf "$tf" 2>&1 | tee "train_${tf}.log"
        T1=$(date +%s)
        echo "  *** ${tf} TRAINED in $(( T1 - T0 ))s ***"
        TRAINED="$TRAINED $tf"

        # Run optimizer for this TF immediately
        echo "  *** ${tf} OPTIMIZING ***"
        python -u exhaustive_optimizer.py --tf "$tf" 2>&1 | tee "optimizer_${tf}.log"
        echo "  *** ${tf} COMPLETE ***"
    fi
}

# Monitor builds and train as each finishes
while true; do
    # Check which builds are still running
    RUNNING=0
    for pid in $PID_1W $PID_1D $PID_4H $PID_1H $PID_15M; do
        if kill -0 $pid 2>/dev/null; then
            RUNNING=$((RUNNING + 1))
        fi
    done

    ELAPSED=$(( $(date +%s) - START ))
    MEM=$(free -h | grep Mem | awk '{print $3}')
    echo "  [${ELAPSED}s] ${RUNNING}/5 builds running | RAM: ${MEM} | Trained:${TRAINED:-none}"

    # Check each TF — train immediately when ready
    train_if_ready "1w"
    train_if_ready "1d"
    train_if_ready "4h"
    train_if_ready "1h"
    train_if_ready "15m"

    # All first 5 done?
    [ "$RUNNING" -eq 0 ] && break
    sleep 10
done

# ═══ 5M BUILD (solo for RAM safety) ═══
echo ""
echo "=== 5M BUILD (solo) ==="
sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
free -h | grep Mem

python -u build_5m_features.py > build_5m.log 2>&1 &
PID_5M=$!

while kill -0 $PID_5M 2>/dev/null; do
    ELAPSED=$(( $(date +%s) - START ))
    MEM=$(free -h | grep Mem | awk '{print $3}')
    LAST=$(tail -1 build_5m.log 2>/dev/null || echo 'building...')
    echo "  [${ELAPSED}s] 5M: ${LAST:0:70} | RAM: ${MEM}"

    # Train it as soon as parquet appears
    train_if_ready "5m"
    sleep 10
done
train_if_ready "5m"

# ═══ LSTM (uses 1 GPU, fast) ═══
echo ""
echo "=== LSTM TRAINING ==="
T0=$(date +%s)
python -u lstm_sequence_model.py 2>&1 | tee lstm.log
T1=$(date +%s)
echo "  LSTM complete: $(( T1 - T0 ))s"

# ═══ SUMMARY ═══
TOTAL=$(( $(date +%s) - START ))
echo ""
echo "========================================================"
echo "  PIPELINE COMPLETE — ${TOTAL}s ($(( TOTAL / 60 )) min)"
echo "========================================================"
echo ""
echo "Models:"
ls -lh model_*.json 2>/dev/null
echo ""
echo "Feature matrices:"
ls -lh features_*.parquet 2>/dev/null
echo ""
echo "Optimizer configs:"
ls -lh exhaustive_configs*.json 2>/dev/null
echo ""
echo "LSTM:"
ls -lh lstm_*.pt 2>/dev/null
