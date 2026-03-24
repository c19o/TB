#!/bin/bash
# deploy_5_machines.sh — Deploy full 7-step pipeline across 5 cloud machines
# Each machine gets ONE timeframe, builds crosses from scratch, runs full pipeline
# Usage: bash deploy_5_machines.sh

set -e
BASE="C:/Users/C/Documents/Savage22 Server"
RESULTS="$BASE/v32_cloud_results"
mkdir -p "$RESULTS"

# Machine assignments (update SSH details from vastai show instances)
# Format: TF:HOST:PORT:INSTANCE_ID
MACHINES=(
    "1w:ssh9.vast.ai:20174:33420174"
    "1d:ssh1.vast.ai:20174:33420175"
    "4h:ssh6.vast.ai:20176:33420176"
    "1h:ssh8.vast.ai:20066:33420067"
    "15m:ssh7.vast.ai:20110:33420111"
)

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"

deploy_one() {
    local TF=$1 HOST=$2 PORT=$3 ID=$4
    local LOG="$RESULTS/deploy_${TF}.log"

    echo "[$TF] Starting deployment to $HOST:$PORT (instance $ID)" | tee "$LOG"

    # Wait for machine to be ready
    for i in $(seq 1 30); do
        ssh $SSH_OPTS -p $PORT root@$HOST 'echo ready' 2>/dev/null && break
        echo "[$TF] Waiting... ($i)" >> "$LOG"
        sleep 10
    done

    # Find parquet
    if [ "$TF" = "15m" ]; then
        PQ="$BASE/v3.1/features_BTC_15m.parquet"
    else
        PQ="$BASE/v3.0 (LGBM)/features_BTC_${TF}.parquet"
    fi

    # Create package
    echo "[$TF] Creating package..." | tee -a "$LOG"
    tar czf "/tmp/v32_${TF}.tar.gz" \
        -C "$BASE" v3.2/*.py v3.2/*.json v3.2/*.pkl \
        -C "$BASE" "$(realpath --relative-to="$BASE" "$PQ")" \
        -C "$BASE" "v3.0 (LGBM)/multi_asset_prices.db" \
        tweets.db news_articles.db astrology_full.db ephemeris_cache.db \
        fear_greed.db sports_results.db space_weather.db macro_data.db \
        onchain_data.db funding_rates.db open_interest.db google_trends.db \
        2>/dev/null

    # Upload
    echo "[$TF] Uploading $(du -h /tmp/v32_${TF}.tar.gz | awk '{print $1}')..." | tee -a "$LOG"
    scp $SSH_OPTS -P $PORT "/tmp/v32_${TF}.tar.gz" root@$HOST:/workspace/

    # Setup + run full pipeline
    echo "[$TF] Setting up + running full 7-step pipeline..." | tee -a "$LOG"
    ssh $SSH_OPTS -p $PORT root@$HOST "
        cd /workspace
        tar xzf v32_${TF}.tar.gz && rm v32_${TF}.tar.gz
        cp v3.2/*.py v3.2/*.json v3.2/*.pkl . 2>/dev/null
        # Find and place parquet
        find . -name 'features_BTC_${TF}.parquet' -exec cp {} . \;
        # Place DBs
        find . -name 'multi_asset_prices.db' -exec cp {} . \;
        cp multi_asset_prices.db btc_prices.db 2>/dev/null
        for db in tweets.db news_articles.db astrology_full.db ephemeris_cache.db fear_greed.db sports_results.db space_weather.db macro_data.db onchain_data.db funding_rates.db open_interest.db google_trends.db; do
            [ -f \"\$db\" ] && mv \"\$db\" ../
        done
        # Symlink V1 DBs
        for db in /workspace/*.db; do
            [ -f \"\$db\" ] && ln -sf \"\$db\" \$(basename \"\$db\") 2>/dev/null
        done
        # Install deps
        pip install -q lightgbm optuna scipy numba scikit-learn pandas pyarrow psutil hmmlearn 2>&1 | tail -1

        export PYTHONUNBUFFERED=1

        echo '=== STEP 1: BUILD CROSSES ==='
        python -X utf8 -u v2_cross_generator.py --tf ${TF} --asset BTC 2>&1 | tail -5

        echo '=== STEP 2: TRAIN ==='
        python -X utf8 -u ml_multi_tf.py --tf ${TF} 2>&1 | tail -10

        echo '=== STEP 3: OPTUNA ==='
        python -X utf8 -u run_optuna_local.py --tf ${TF} 2>&1 | tail -5

        echo '=== STEP 4: META ==='
        python -X utf8 -u meta_labeling.py --tf ${TF} 2>&1 | tail -3

        echo '=== STEP 5: LSTM ==='
        pip install -q torch 2>&1 | tail -1
        python -X utf8 -u lstm_sequence_model.py --tf ${TF} --train 2>&1 | tail -5

        echo '=== STEP 6: PBO ==='
        python -X utf8 -u backtest_validation.py --tf ${TF} 2>&1 | tail -3

        echo '=== STEP 7: AUDIT ==='
        python -X utf8 -u backtesting_audit.py --tf ${TF} 2>&1 | tail -3

        echo '=== COMPLETE ==='
        ls -la model_${TF}.json optuna_configs_${TF}.json meta_model_${TF}.pkl platt_${TF}.pkl 2>/dev/null
    " 2>&1 | tee -a "$LOG"

    # Download results
    echo "[$TF] Downloading results..." | tee -a "$LOG"
    for f in model_${TF}.json features_${TF}_all.json optuna_configs_${TF}.json \
             meta_model_${TF}.pkl platt_${TF}.pkl cpcv_oos_predictions_${TF}.pkl \
             lstm_${TF}.pt pbo_results_${TF}.json audit_${TF}.json \
             feature_importance_stability_${TF}.json ml_multi_tf_results.txt; do
        scp $SSH_OPTS -P $PORT "root@$HOST:/workspace/$f" "$RESULTS/" 2>/dev/null
    done

    echo "[$TF] DONE!" | tee -a "$LOG"
}

# Deploy all 5 in parallel
for m in "${MACHINES[@]}"; do
    IFS=: read -r TF HOST PORT ID <<< "$m"
    deploy_one "$TF" "$HOST" "$PORT" "$ID" &
done

echo "All 5 deployments launched in parallel. Waiting for all to finish..."
wait
echo "=========================================="
echo "ALL 5 TIMEFRAMES COMPLETE"
echo "=========================================="
ls -lh "$RESULTS/"
