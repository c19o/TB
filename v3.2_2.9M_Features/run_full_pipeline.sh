#!/bin/bash
# run_full_pipeline.sh — Local Training Pipeline (RTX 3090)
# ============================================================
# LOCAL ONLY — no cloud, no cudf.pandas, no multi-GPU
# Single RTX 3090 + i9-13900K
# TFs: 1w, 1d, 4h, 1h, 15m (no 5m)
# Model: LightGBM (CPU, force_col_wise) + LSTM (GPU)
# ============================================================
# Usage: bash run_full_pipeline.sh 2>&1 | tee pipeline.log

set -e
cd "$(dirname "$0")"

echo "========================================================"
echo "  SAVAGE22 v3.1 LOCAL PIPELINE — $(date)"
echo "  RTX 3090 | i9-13900K | LightGBM + LSTM"
echo "  TFs: 1w, 1d, 4h, 1h, 15m"
echo "========================================================"

# ═══ PHASE 1: BASE FEATURE BUILDS ═══
# Stagger: small TFs parallel → 1h → 15m
echo ""
echo "=== PHASE 1: BASE FEATURE BUILDS (staggered) ==="
START_P1=$(date +%s)

# --- 1W, 1D, 4H in parallel (small, ~2-5 min each) ---
echo "  Starting 1W + 1D + 4H in parallel..."
python -u build_1w_features.py > build_1w.log 2>&1 &
PID_1W=$!
python -u build_1d_features.py > build_1d.log 2>&1 &
PID_1D=$!
python -u build_4h_features.py > build_4h.log 2>&1 &
PID_4H=$!

echo "  Launched: 1W($PID_1W) 1D($PID_1D) 4H($PID_4H)"

# Wait for parallel group with progress
while true; do
    RUNNING=$(ps -p $PID_1W,$PID_1D,$PID_4H --no-headers 2>/dev/null | wc -l)
    ELAPSED=$(( $(date +%s) - START_P1 ))
    echo "  [${ELAPSED}s] $RUNNING builds still running..."
    for tf in 1w 1d 4h; do
        if [ -f "features_${tf}.parquet" ]; then
            SIZE=$(stat -c%s "features_${tf}.parquet" 2>/dev/null || stat -f%z "features_${tf}.parquet" 2>/dev/null)
            echo "    ${tf}: DONE ($(echo "scale=1; $SIZE/1073741824" | bc) GB)"
        else
            LOGSIZE=$(stat -c%s "build_${tf}.log" 2>/dev/null || stat -f%z "build_${tf}.log" 2>/dev/null || echo 0)
            echo "    ${tf}: building (log: ${LOGSIZE}B)"
        fi
    done
    [ "$RUNNING" -eq 0 ] && break
    sleep 30
done

# Check exit codes
wait $PID_1W || { echo "FATAL: 1W build failed"; exit 1; }
wait $PID_1D || { echo "FATAL: 1D build failed"; exit 1; }
wait $PID_4H || { echo "FATAL: 4H build failed"; exit 1; }
echo "  1W + 1D + 4H complete."

# --- 1H sequential (medium, needs more RAM) ---
echo ""
echo "  Starting 1H build (sequential)..."
python -u build_1h_features.py 2>&1 | tee build_1h.log
echo "  1H complete."

# --- 15M sequential (large, needs full RAM) ---
echo ""
echo "  Starting 15M build (sequential)..."
python -u build_15m_features.py 2>&1 | tee build_15m.log
echo "  15M complete."

END_P1=$(date +%s)
echo ""
echo "  Phase 1 complete: $(( END_P1 - START_P1 ))s"
echo ""
echo "=== FEATURE MATRIX SIZES ==="
ls -lh features_*.parquet 2>/dev/null

# ═══ PHASE 2: LIGHTGBM TRAINING (CPU, force_col_wise) ═══
echo ""
echo "=== PHASE 2: LIGHTGBM TRAINING (all TFs) ==="
START_P2=$(date +%s)
python -u ml_multi_tf.py 2>&1 | tee train.log
END_P2=$(date +%s)
echo "  Phase 2 complete: $(( END_P2 - START_P2 ))s"

# ═══ PHASE 3: OPTUNA OPTIMIZER ═══
echo ""
echo "=== PHASE 3: OPTUNA OPTIMIZER ==="
START_P3=$(date +%s)
python -u run_optuna_local.py 2>&1 | tee optuna.log
END_P3=$(date +%s)
echo "  Phase 3 complete: $(( END_P3 - START_P3 ))s"

# ═══ PHASE 4: META-LABELING ═══
echo ""
echo "=== PHASE 4: META-LABELING ==="
START_P4=$(date +%s)
python -u meta_labeling.py 2>&1 | tee meta.log
END_P4=$(date +%s)
echo "  Phase 4 complete: $(( END_P4 - START_P4 ))s"

# ═══ PHASE 5: LSTM TRAINING (GPU, RTX 3090) ═══
echo ""
echo "=== PHASE 5: LSTM TRAINING (RTX 3090) ==="
START_P5=$(date +%s)
python -u lstm_sequence_model.py --train --all 2>&1 | tee lstm.log
END_P5=$(date +%s)
echo "  Phase 5 complete: $(( END_P5 - START_P5 ))s"

# ═══ PHASE 6: PBO + DEFLATED SHARPE VALIDATION ═══
echo ""
echo "=== PHASE 6: PBO + DEFLATED SHARPE VALIDATION ==="
START_P6=$(date +%s)
python -u backtest_validation.py 2>&1 | tee pbo.log
END_P6=$(date +%s)
echo "  Phase 6 complete: $(( END_P6 - START_P6 ))s"

# ═══ PHASE 7: BACKTESTING AUDIT ═══
echo ""
echo "=== PHASE 7: BACKTESTING AUDIT ==="
START_P7=$(date +%s)
for tf in 1w 1d 4h 1h 15m; do
    echo "  Auditing $tf..."
    python -u backtesting_audit.py --tf $tf 2>&1 | tee audit_${tf}.log
done
END_P7=$(date +%s)
echo "  Phase 7 complete: $(( END_P7 - START_P7 ))s"

# ═══ SUMMARY ═══
TOTAL=$(( END_P7 - START_P1 ))
echo ""
echo "========================================================"
echo "  PIPELINE COMPLETE — Total: ${TOTAL}s ($(echo "scale=1; $TOTAL/60" | bc) min)"
echo "  Phase 1 (feature builds):  $(( END_P1 - START_P1 ))s"
echo "  Phase 2 (LightGBM):        $(( END_P2 - START_P2 ))s"
echo "  Phase 3 (Optuna):           $(( END_P3 - START_P3 ))s"
echo "  Phase 4 (meta-labeling):    $(( END_P4 - START_P4 ))s"
echo "  Phase 5 (LSTM):             $(( END_P5 - START_P5 ))s"
echo "  Phase 6 (PBO/DSR):          $(( END_P6 - START_P6 ))s"
echo "  Phase 7 (audit):            $(( END_P7 - START_P7 ))s"
echo "========================================================"
echo ""
echo "Output files:"
ls -lh model_*.json features_*_all.json platt_*.pkl optuna_configs*.json lstm_*.pt meta_model_*.pkl validation_report_*.json 2>/dev/null
