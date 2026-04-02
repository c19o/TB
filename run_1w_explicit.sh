set -euo pipefail
cd /workspace/v3.3
export ALLOW_CPU=1
export PYTHONUNBUFFERED=1
export V30_DATA_DIR=/workspace/v3.3
export SAVAGE22_DB_DIR=/workspace
export SAVAGE22_V1_DIR=/workspace
heartbeat() {
  PHASE="$1" DETAIL="${2:-}" python - <<'PY'
import json,os,time,pathlib
p=pathlib.Path('/workspace/cloud_run_1w_heartbeat.json')
p.write_text(json.dumps({
  'updated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
  'tf':'1w',
  'phase': os.environ['PHASE'],
  'detail': os.environ.get('DETAIL','')
}, sort_keys=True))
PY
}
trap 'heartbeat failed "command failed"' ERR
rm -f /workspace/lgbm_dataset_1w.bin /workspace/cpcv_checkpoint_1w.pkl /workspace/model_1w.json /workspace/model_1w_prev.json /workspace/model_1w_cpcv_backup.json /workspace/optuna_configs_1w.json /workspace/meta_model_1w.pkl /workspace/cpcv_oos_1w.pkl /workspace/cpcv_oos_predictions_1w.pkl /workspace/shap_analysis_1w.json /workspace/feature_importance_1w.json /workspace/feature_importance_stability_1w.json /workspace/validation_report_1w.json /workspace/fi_pipeline_1w_summary.json /workspace/cloud_run_1w_heartbeat.json
rm -f /workspace/v3.3/features_BTC_1w.parquet /workspace/v3.3/lgbm_dataset_1w.bin /workspace/v3.3/v2_crosses_BTC_1w.npz /workspace/v3.3/v2_cross_names_BTC_1w.json /workspace/v3.3/model_1w.json /workspace/v3.3/model_1w_prev.json /workspace/v3.3/model_1w_cpcv_backup.json /workspace/v3.3/optuna_configs_1w.json /workspace/v3.3/meta_model_1w.pkl /workspace/v3.3/cpcv_oos_1w.pkl /workspace/v3.3/cpcv_oos_predictions_1w.pkl /workspace/v3.3/shap_analysis_1w.json /workspace/v3.3/feature_importance_1w.json /workspace/v3.3/feature_importance_stability_1w.json /workspace/v3.3/validation_report_1w.json
heartbeat start 'fresh 1w run started'
python -u build_features_v2.py --symbol BTC --tf 1w | tee /workspace/step1_build_features_1w.log
heartbeat step1 'features built'
python -u v2_cross_generator.py --tf 1w --symbol BTC --save-sparse | tee /workspace/step2_crosses_1w.log
heartbeat step2 'crosses built'
python -u ml_multi_tf.py --tf 1w --boost-rounds 800 | tee /workspace/step3_train_baseline_1w.log
heartbeat step3 'baseline trained'
python -u run_optuna_local.py --tf 1w | tee /workspace/step4_optuna_1w.log
heartbeat step4 'optuna complete'
python -u ml_multi_tf.py --tf 1w --boost-rounds 800 | tee /workspace/step5_retrain_optuna_1w.log
heartbeat step5 'retrain complete'
python -u exhaustive_optimizer.py --tf 1w --n-trials 200 | tee /workspace/step6_optimizer_1w.log
heartbeat step6 'optimizer complete'
heartbeat complete '1w complete'
