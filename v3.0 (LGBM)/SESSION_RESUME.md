# Session Resume — V3.0 (LGBM) — 2026-03-23

## STATUS: CODE COMPLETE — WAITING ON CLOUD
V3 code 100% done, smoke test 27/27 passed. vast.ai had infrastructure issues (RAPIDS image pull failures, SSH key permission bugs, port forwarding errors across multiple regions). Need to retry deployment.

## WHAT TO DO NEXT SESSION
1. **Rent a machine** — try RAPIDS image first. If SSH fails, check `vastai logs <ID>` for "bad ownership" error
2. If RAPIDS won't SSH, try `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel` + `pip install --extra-index-url https://pypi.nvidia.com cudf-cu12` (DON'T install cuml — breaks torch)
3. Upload: v3 scripts (tar), 14 DBs + multi_asset_prices.db (1.3GB), config files
4. Set `SAVAGE22_V1_DIR=/workspace/v3` so config.py finds DBs
5. Launch: `PYTHONUNBUFFERED=1 python3 -u v2_cloud_runner.py --engine lightgbm --dph <COST>`
6. Monitor: `tail -f full_pipeline.log`
7. Download artifacts when done

## MACHINE REQUIREMENTS (NO OOM on 5m)
- **RAM: 700GB+** (5m peak ~450GB, need headroom)
- **GPU: 2+ with 24GB+ VRAM each** (cuDF feature builds + LSTM)
- **CPU: 64+ cores** (LightGBM force_col_wise)
- **Budget: $1-3/hr** ideally

### Search command:
```bash
vastai search offers 'cpu_ram>=700 verified=true rentable=true gpu_ram>=24 num_gpus>=2' -o 'dph_total' --limit 15
```

### Known-broken machines (DO NOT USE):
- Belgium m49722, m48486, m30443, m49822 — SSH authorized_keys permission bug
- Australia m15350 — port forwarding failed
- RAPIDS image `25.12-cuda12-py3.11` — broken SSH on ALL hosts tested (bad authorized_keys ownership)
- RAPIDS image `25.02-cuda12-py3.12` — pulled 30+ min on Nebraska, never started

### Fallback: PyTorch image + pip cuDF
```bash
vastai create instance <ID> --image pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel --disk 200
# After SSH:
pip install --extra-index-url https://pypi.nvidia.com cudf-cu12
pip install lightgbm optuna scipy scikit-learn hmmlearn psutil numba cupy-cuda12x
# DO NOT pip install cuml-cu12 — breaks torch shared libs
```

## GIT
- Branch: `v3-lgbm` (11 commits, all pushed)
- Remote: `https://github.com/c19o/TB.git`

## V3 CHANGES FROM V2 (Summary)
1. XGBoost → LightGBM (CPU-only, force_col_wise, max_bin=15, EFB for sparse)
2. 30M Exhaustive → Optuna 200 TPE trials (Sortino, n_jobs=4)
3. min_data_in_leaf=3 + min_gain_to_split=2.0 (rare signals usable)
4. Co-occurrence filter MIN_CO_OCCURRENCE=8 (CPCV math constraint)
5. Direct COO extraction from GPU tensor (prevents 1.36TB OOM on 5m)
6. Incremental CSR per cross type (no giant COO→CSR bottleneck)
7. Eliminated tolil()/tocsr() in CPCV fold loop (150-450s savings)
8. CPCV workers capped at 15, num_threads per worker = cores/workers
9. LSTM DataLoader workers raised to 32, Optuna n_jobs=4
10. All XGBoost references removed from 18+ files

## SMOKE TEST RESULTS (local, 2026-03-22)
```
27 passed, 0 failed, 10.8s elapsed
- 10/10 modules imported (no XGBoost dependency)
- Co-occurrence filter: 60,181 → 38,539 columns (36% reduction)
- LightGBM: 4.5s train, 92.8% accuracy, 3-class (1146, 3) output
- Optuna: 5 trials with simulate_batch
- Meta-labeling: AUC=0.933, 11/50 trades approved
```

## PHILOSOPHY (NON-NEGOTIABLE)
- NO FILTERING. LightGBM decides via tree splits, not us
- NO FALLBACKS. One pipeline for all. If it breaks, fix it
- Esoteric signals ARE the edge. Never regularize them away
- NaN = missing. NEVER convert to 0
- Sparse = the edge, not noise
- NEVER deviate from the plan under any circumstance

## ESTIMATED PIPELINE TIME (once deployed)
| Phase | Est |
|-------|-----|
| Feature builds (all TFs, all assets) | ~6-8 hrs |
| LightGBM training (6 TFs) | ~30 min |
| Optuna (200 trials × 6 TFs) | ~10 min |
| Validation + Meta | ~5 min |
| LSTM (6 TFs) | ~20 min |
| Audit | ~2 min |
| **Total** | **~8-10 hrs** |
