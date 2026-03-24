# Microstep Plan — Finishing v3.2

## TRACK 1: Belgium 15m (210K rows dense, all 2.3M features)
1. [ ] Wait for Belgium to boot
2. [ ] Build 15m tar with 210K row subsample logic in ml_multi_tf.py
3. [ ] Upload tar to Belgium (~70s)
4. [ ] Extract and launch pipeline
5. [ ] Monitor: verify "Converting sparse to dense" in log
6. [ ] Monitor: verify load average > 100 (multi-core training)
7. [ ] Monitor: SPARSE verification
8. [ ] Download ALL artifacts including NPZ before killing
9. [ ] Verify directional accuracy by confidence tier
10. [ ] Kill Belgium

## TRACK 2: Optuna for 1d and 1h (parallel machines)
1. [ ] Rent 2 machines (need parquet + NPZ + code)
2. [ ] Build Optuna-only tars (include NPZ from downloaded results)
3. [ ] Upload to both machines
4. [ ] Launch run_optuna_local.py --tf 1d / --tf 1h
5. [ ] Fix 1d class imbalance: add is_unbalance=True to Optuna params
6. [ ] Monitor Optuna progress
7. [ ] Download results
8. [ ] Kill machines

## TRACK 3: Texas 15m (keep running as validation)
- Let it finish sparse single-threaded run
- Compare results with Belgium dense run
- Download everything before killing

## DEFERRED
- 1w Optuna (too few rows, Optuna won't help much per Perplexity)
- 4h Optuna (run after 1d/1h validated)
- Temporal cascade ensemble (v3.3 feature)
