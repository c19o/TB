#!/bin/bash
# run_full_pipeline.sh — Full training pipeline on vast.ai
# Runs all 6 TFs: base build (cudf.pandas) → GPU crosses → XGBoost → optimizer → LSTM
# Usage: bash run_full_pipeline.sh

set -e
cd /workspace/savage22

echo "========================================================"
echo "  SAVAGE22 FULL PIPELINE — $(date)"
echo "  8x RTX 3090 | cudf.pandas + CuPy GPU"
echo "========================================================"

# ═══ PHASE 1: BASE FEATURE BUILDS (cudf.pandas accelerated) ═══
echo ""
echo "=== PHASE 1: BASE FEATURE BUILDS (6 TFs parallel) ==="
START_P1=$(date +%s)

# Small TFs in parallel (1W, 1D, 4H)
CUDA_VISIBLE_DEVICES=0 python -u -m cudf.pandas build_1w_features.py > build_1w.log 2>&1 &
PID_1W=$!
CUDA_VISIBLE_DEVICES=1 python -u -m cudf.pandas build_1d_features.py > build_1d.log 2>&1 &
PID_1D=$!
CUDA_VISIBLE_DEVICES=2 python -u -m cudf.pandas build_4h_features.py > build_4h.log 2>&1 &
PID_4H=$!

# Medium TFs
CUDA_VISIBLE_DEVICES=3 python -u -m cudf.pandas build_1h_features.py > build_1h.log 2>&1 &
PID_1H=$!

# Large TFs
CUDA_VISIBLE_DEVICES=4 python -u -m cudf.pandas build_15m_features.py > build_15m.log 2>&1 &
PID_15M=$!
CUDA_VISIBLE_DEVICES=5 python -u -m cudf.pandas build_5m_features.py > build_5m.log 2>&1 &
PID_5M=$!

echo "  Launched: 1W($PID_1W) 1D($PID_1D) 4H($PID_4H) 1H($PID_1H) 15M($PID_15M) 5M($PID_5M)"

# Wait for all with progress
while true; do
    RUNNING=$(ps -p $PID_1W,$PID_1D,$PID_4H,$PID_1H,$PID_15M,$PID_5M --no-headers 2>/dev/null | wc -l)
    ELAPSED=$(( $(date +%s) - START_P1 ))
    echo "  [${ELAPSED}s] $RUNNING builds still running..."
    for tf in 1w 1d 4h 1h 15m 5m; do
        if [ -f "features_${tf}.parquet" ]; then
            SIZE=$(stat -c%s "features_${tf}.parquet" 2>/dev/null)
            echo "    ${tf}: DONE ($(echo "scale=1; $SIZE/1073741824" | bc) GB)"
        else
            LOGSIZE=$(stat -c%s "build_${tf}.log" 2>/dev/null || echo 0)
            echo "    ${tf}: building (log: ${LOGSIZE}B)"
        fi
    done
    [ "$RUNNING" -eq 0 ] && break
    sleep 30
done

END_P1=$(date +%s)
echo "  Phase 1 complete: $(( END_P1 - START_P1 ))s"

# ═══ PHASE 2: GPU CROSS GENERATION (CuPy batch multiply) ═══
echo ""
echo "=== PHASE 2: GPU CROSS GENERATION (6 TFs parallel on 6 GPUs) ==="
START_P2=$(date +%s)

# Each TF on its own GPU
CUDA_VISIBLE_DEVICES=0 python -u gpu_cross_builder.py --tf 1w --gpu 0 > crosses_1w.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u gpu_cross_builder.py --tf 1d --gpu 0 > crosses_1d.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -u gpu_cross_builder.py --tf 4h --gpu 0 > crosses_4h.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -u gpu_cross_builder.py --tf 1h --gpu 0 > crosses_1h.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python -u gpu_cross_builder.py --tf 15m --gpu 0 > crosses_15m.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python -u gpu_cross_builder.py --tf 5m --gpu 0 > crosses_5m.log 2>&1 &

wait
END_P2=$(date +%s)
echo "  Phase 2 complete: $(( END_P2 - START_P2 ))s"

echo ""
echo "=== FEATURE MATRIX SIZES ==="
ls -lh features_*.parquet

# ═══ PHASE 3: XGBOOST TRAINING ═══
echo ""
echo "=== PHASE 3: XGBOOST TRAINING (all TFs) ==="
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
echo "  PIPELINE COMPLETE — Total: ${TOTAL}s ($(echo "scale=1; $TOTAL/60" | bc) min)"
echo "  Phase 1 (builds):    $(( END_P1 - START_P1 ))s"
echo "  Phase 2 (GPU crosses): $(( END_P2 - START_P2 ))s"
echo "  Phase 3 (XGBoost):   $(( END_P3 - START_P3 ))s"
echo "  Phase 4 (optimizer): $(( END_P4 - START_P4 ))s"
echo "  Phase 5 (LSTM):      $(( END_P5 - START_P5 ))s"
echo "========================================================"
echo ""
echo "Output files:"
ls -lh model_*.json features_*_all.json platt_*.pkl exhaustive_configs.json lstm_*.pt 2>/dev/null
