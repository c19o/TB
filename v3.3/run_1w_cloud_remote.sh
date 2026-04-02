#!/usr/bin/env bash
set -euo pipefail

write_hb() {
  PHASE="$1" DETAIL="${2:-}" python - <<'PY'
import json
import os
import time

payload = {
    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "tf": "1w",
    "phase": os.environ["PHASE"],
    "detail": os.environ.get("DETAIL", ""),
}
with open("/workspace/cloud_run_1w_heartbeat.json", "w", encoding="utf-8") as fh:
    json.dump(payload, fh, sort_keys=True)
PY
}

trap 'write_hb failed "command failed: ${BASH_COMMAND}"' ERR

export PYTHONUNBUFFERED=1
export V30_DATA_DIR=/workspace/v3.3
export SAVAGE22_DB_DIR=/workspace
export SAVAGE22_V1_DIR=/workspace
export ALLOW_CPU=1

cd /workspace/v3.3

write_hb cleanup "Clearing stale weekly artifacts and cross files"
rm -f v2_crosses_BTC_1w.npz v2_cross_names_BTC_1w.json pipeline_manifest.json
rm -f features_BTC_1w.parquet lgbm_dataset_1w.bin
rm -f model_1w.json model_1w_cpcv_backup.json model_1w_fold*.txt
rm -f optuna_configs_1w.json meta_model_1w.pkl cpcv_oos_predictions_1w.pkl cpcv_oos_1w.pkl
rm -f shap_analysis_1w.json feature_importance_stability_1w.json platt_1w.pkl calibrator_1w.pkl

rm -f /workspace/step0_manifest_1w.log
rm -f /workspace/step0_deploy_verify_1w.log
rm -f /workspace/step0_pipeline_plumbing_1w.log
rm -f /workspace/step0_validate_1w.log
rm -f /workspace/step1_build_features_1w.log
rm -f /workspace/step3_train_baseline_1w.log
rm -f /workspace/step4_optuna_1w.log
rm -f /workspace/step5_retrain_optuna_1w.log
rm -f /workspace/step6_optimizer_1w.log
rm -f /workspace/step7_meta_1w.log

write_hb verify "Running deploy and pipeline preflight"
python -u deploy_manifest.py > /workspace/step0_manifest_1w.log 2>&1
python -u deploy_verify.py --tf 1w > /workspace/step0_deploy_verify_1w.log 2>&1
python -u test_pipeline_plumbing.py --tf 1w > /workspace/step0_pipeline_plumbing_1w.log 2>&1 || true
python -u validate.py --tf 1w --cloud > /workspace/step0_validate_1w.log 2>&1

write_hb step1_features "Building weekly base features"
python -u build_features_v2.py --symbol BTC --tf 1w | tee /workspace/step1_build_features_1w.log

write_hb step3_baseline "Training weekly baseline model"
python -u ml_multi_tf.py --tf 1w --boost-rounds 800 | tee /workspace/step3_train_baseline_1w.log

write_hb step4_optuna "Running weekly Optuna search"
python -u run_optuna_local.py --tf 1w | tee /workspace/step4_optuna_1w.log

write_hb step5_retrain "Retraining weekly model with tuned params"
python -u ml_multi_tf.py --tf 1w --boost-rounds 800 | tee /workspace/step5_retrain_optuna_1w.log

write_hb step6_optimizer "Running weekly execution optimizer"
python -u exhaustive_optimizer.py --tf 1w --n-trials 200 | tee /workspace/step6_optimizer_1w.log

if [[ -f meta_labeling.py ]]; then
  write_hb step7_meta "Running weekly meta labeling"
  python -u meta_labeling.py --tf 1w | tee /workspace/step7_meta_1w.log || true
fi

write_hb complete "Weekly training pipeline finished"
