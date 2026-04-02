#!/bin/bash
set -euo pipefail
cd /workspace/v3.3
export PYTHONUNBUFFERED=1
export V30_DATA_DIR=/workspace/v3.3
export SAVAGE22_DB_DIR=/workspace
export SAVAGE22_V1_DIR=/workspace
export ALLOW_CPU=1
HB=/workspace/cloud_run_1w_heartbeat.json
phase(){
  local phase_name="$1"
  local detail_text="$2"
  PHASE_NAME="$phase_name" DETAIL_TEXT="$detail_text" HB_PATH="$HB" python - <<'PY'
import json, os, pathlib, time
path = pathlib.Path(os.environ['HB_PATH'])
payload = {
    'updated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'tf': '1w',
    'phase': os.environ['PHASE_NAME'],
    'detail': os.environ['DETAIL_TEXT'],
}
path.write_text(json.dumps(payload, sort_keys=True))
PY
}
trap 'phase failed "command failed"' ERR
phase start 'clean weekly sequence launched'
rm -f v2_crosses_BTC_1w.npz v2_cross_names_BTC_1w.json pipeline_manifest.json lgbm_dataset_1w.bin
phase step1 'build_features_v2'
python -u build_features_v2.py --symbol BTC --tf 1w | tee /workspace/step1_build_features_1w.log
phase step2 'baseline ml_multi_tf'
python -u ml_multi_tf.py --tf 1w --boost-rounds 800 | tee /workspace/step2_train_baseline_1w.log
phase step3 'run_optuna_local'
python -u run_optuna_local.py --tf 1w | tee /workspace/step3_optuna_1w.log
phase step4 'retrain ml_multi_tf'
python -u ml_multi_tf.py --tf 1w --boost-rounds 800 | tee /workspace/step4_retrain_1w.log
phase step5 'exhaustive_optimizer'
python -u exhaustive_optimizer.py --tf 1w --n-trials 200 | tee /workspace/step5_optimizer_1w.log
phase step6 'meta_labeling'
python -u meta_labeling.py --tf 1w | tee /workspace/step6_meta_1w.log
phase complete '1w pipeline complete'