#!/bin/bash
# run_pipeline_v3.sh — Staggered pipeline for 8x H100 SXM (2TB RAM)
# Builds staggered to prevent OOM. Dask multi-GPU XGBoost.
set -e
cd /workspace/savage22
export PYTHONUNBUFFERED=1

echo "========================================================"
echo "  SAVAGE22 PIPELINE v3 — $(date)"
echo "  8x H100 SXM | Dask Multi-GPU XGBoost"
echo "  2TB RAM | Staggered builds (5M solo)"
echo "========================================================"

# ═══ PHASE 1a: SMALL TF BUILDS (1W/1D/4H/1H parallel) ═══
echo ""
echo "=== PHASE 1a: SMALL TF BUILDS (4 parallel) ==="
START_P1=$(date +%s)

python -u build_1w_features.py > build_1w.log 2>&1 &
PID_1W=$!
python -u build_1d_features.py > build_1d.log 2>&1 &
PID_1D=$!
python -u build_4h_features.py > build_4h.log 2>&1 &
PID_4H=$!
python -u build_1h_features.py > build_1h.log 2>&1 &
PID_1H=$!

echo "  Launched: 1W($PID_1W) 1D($PID_1D) 4H($PID_4H) 1H($PID_1H)"

while true; do
    RUNNING=$(ps -p $PID_1W,$PID_1D,$PID_4H,$PID_1H --no-headers 2>/dev/null | wc -l)
    ELAPSED=$(( $(date +%s) - START_P1 ))
    echo "  [${ELAPSED}s] $RUNNING/4 builds running..."
    for tf in 1w 1d 4h 1h; do
        if [ -f "features_${tf}.parquet" ]; then
            SIZE=$(du -h "features_${tf}.parquet" | cut -f1)
            echo "    ${tf}: DONE (${SIZE})"
        else
            LAST=$(tail -1 "build_${tf}.log" 2>/dev/null || echo 'starting...')
            echo "    ${tf}: ${LAST:0:80}"
        fi
    done
    [ "$RUNNING" -eq 0 ] && break
    sleep 15
done

# Check for failures
for tf in 1w 1d 4h 1h; do
    if [ ! -f "features_${tf}.parquet" ]; then
        echo "  WARNING: ${tf} build failed — see build_${tf}.log"
        tail -10 "build_${tf}.log" 2>/dev/null
    fi
done

END_P1A=$(date +%s)
echo "  Phase 1a complete: $(( END_P1A - START_P1 ))s"
echo "  RAM after small builds:"
free -h | grep Mem

# ═══ PHASE 1b: 15M BUILD ═══
echo ""
echo "=== PHASE 1b: 15M BUILD ==="
START_P1B=$(date +%s)

python -u build_15m_features.py > build_15m.log 2>&1 &
PID_15M=$!
echo "  Launched: 15M($PID_15M)"

while true; do
    RUNNING=$(ps -p $PID_15M --no-headers 2>/dev/null | wc -l)
    ELAPSED=$(( $(date +%s) - START_P1B ))
    if [ -f "features_15m.parquet" ]; then
        SIZE=$(du -h "features_15m.parquet" | cut -f1)
        echo "  [${ELAPSED}s] 15M: DONE (${SIZE})"
        break
    else
        LAST=$(tail -1 "build_15m.log" 2>/dev/null || echo 'starting...')
        MEM=$(free -h | grep Mem | awk '{print $3}')
        echo "  [${ELAPSED}s] 15M: ${LAST:0:60} | RAM: ${MEM}"
    fi
    [ "$RUNNING" -eq 0 ] && break
    sleep 15
done

if [ ! -f "features_15m.parquet" ]; then
    echo "  WARNING: 15M build failed"
    tail -10 "build_15m.log" 2>/dev/null
fi

END_P1B=$(date +%s)
echo "  Phase 1b complete: $(( END_P1B - START_P1B ))s"

# ═══ PHASE 1c: 5M BUILD (SOLO — needs ~250GB RAM) ═══
echo ""
echo "=== PHASE 1c: 5M BUILD (SOLO) ==="
echo "  Clearing memory before 5M..."
sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
free -h | grep Mem
START_P1C=$(date +%s)

python -u build_5m_features.py > build_5m.log 2>&1 &
PID_5M=$!
echo "  Launched: 5M($PID_5M)"

while true; do
    RUNNING=$(ps -p $PID_5M --no-headers 2>/dev/null | wc -l)
    ELAPSED=$(( $(date +%s) - START_P1C ))
    if [ -f "features_5m.parquet" ]; then
        SIZE=$(du -h "features_5m.parquet" | cut -f1)
        echo "  [${ELAPSED}s] 5M: DONE (${SIZE})"
        break
    else
        LAST=$(tail -1 "build_5m.log" 2>/dev/null || echo 'starting...')
        MEM=$(free -h | grep Mem | awk '{print $3}')
        echo "  [${ELAPSED}s] 5M: ${LAST:0:60} | RAM: ${MEM}"
    fi
    [ "$RUNNING" -eq 0 ] && break
    sleep 15
done

if [ ! -f "features_5m.parquet" ]; then
    echo "  FAILED: 5M build failed"
    tail -20 "build_5m.log" 2>/dev/null
fi

END_P1C=$(date +%s)
echo "  Phase 1c complete: $(( END_P1C - START_P1C ))s"

echo ""
echo "=== ALL FEATURE MATRICES ==="
ls -lh features_*.parquet 2>/dev/null
END_P1=$(date +%s)
echo "  Phase 1 total: $(( END_P1 - START_P1 ))s"

# ═══ PHASE 2: GPU CROSS GENERATION ═══
echo ""
echo "=== PHASE 2: GPU CROSS GENERATION ==="
START_P2=$(date +%s)

CUDA_VISIBLE_DEVICES=0 python -u gpu_cross_builder.py --tf 1w --gpu 0 > crosses_1w.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u gpu_cross_builder.py --tf 1d --gpu 0 > crosses_1d.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -u gpu_cross_builder.py --tf 4h --gpu 0 > crosses_4h.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -u gpu_cross_builder.py --tf 1h --gpu 0 > crosses_1h.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python -u gpu_cross_builder.py --tf 15m --gpu 0 > crosses_15m.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python -u gpu_cross_builder.py --tf 5m --gpu 0 > crosses_5m.log 2>&1 &

echo "  Launched 6 GPU cross builders on GPUs 0-5"

while true; do
    RUNNING=$(jobs -r | wc -l)
    ELAPSED=$(( $(date +%s) - START_P2 ))
    echo "  [${ELAPSED}s] $RUNNING cross builders running..."
    [ "$RUNNING" -eq 0 ] && break
    sleep 15
done

wait
END_P2=$(date +%s)
echo "  Phase 2 complete: $(( END_P2 - START_P2 ))s"
echo "  Updated parquets:"
ls -lh features_*.parquet 2>/dev/null

# ═══ PHASE 3: DASK MULTI-GPU XGBOOST ═══
echo ""
echo "=== PHASE 3: DASK MULTI-GPU XGBOOST (8x H100) ==="
START_P3=$(date +%s)
python -u ml_multi_tf.py 2>&1 | tee train.log
END_P3=$(date +%s)
echo "  Phase 3 complete: $(( END_P3 - START_P3 ))s"

# ═══ PHASE 4: EXHAUSTIVE OPTIMIZER ═══
echo ""
echo "=== PHASE 4: EXHAUSTIVE OPTIMIZER ==="
START_P4=$(date +%s)
python -u exhaustive_optimizer.py 2>&1 | tee optimizer.log
END_P4=$(date +%s)
echo "  Phase 4 complete: $(( END_P4 - START_P4 ))s"

# ═══ PHASE 5: LSTM ═══
echo ""
echo "=== PHASE 5: LSTM TRAINING ==="
START_P5=$(date +%s)
python -u lstm_sequence_model.py 2>&1 | tee lstm.log
END_P5=$(date +%s)
echo "  Phase 5 complete: $(( END_P5 - START_P5 ))s"

# ═══ SUMMARY ═══
TOTAL=$(( END_P5 - START_P1 ))
echo ""
echo "========================================================"
echo "  PIPELINE COMPLETE — ${TOTAL}s ($(( TOTAL / 60 )) min)"
echo "  Phase 1a (small builds): $(( END_P1A - START_P1 ))s"
echo "  Phase 1b (15M):          $(( END_P1B - START_P1B ))s"
echo "  Phase 1c (5M solo):      $(( END_P1C - START_P1C ))s"
echo "  Phase 2 (GPU crosses):   $(( END_P2 - START_P2 ))s"
echo "  Phase 3 (Dask XGBoost):  $(( END_P3 - START_P3 ))s"
echo "  Phase 4 (optimizer):     $(( END_P4 - START_P4 ))s"
echo "  Phase 5 (LSTM):          $(( END_P5 - START_P5 ))s"
echo "========================================================"
echo ""
echo "Output files:"
ls -lh model_*.json features_*_all.json platt_*.pkl exhaustive_configs.json lstm_*.pt 2>/dev/null
echo ""
echo "Feature matrices:"
ls -lh features_*.parquet 2>/dev/null
