# V3.3 Session Resume — 2026-03-25

## STATUS: Local 1w test in progress, no cloud machines running

## MICROSTEP PLAN

### Phase 1: Prove plumbing locally on 1w (IN PROGRESS)
1. [x] Build v3.3 features locally (13900K CPU, ~5-10 min)
2. [ ] Build v3.3 crosses locally (3090 GPU, ~5 min)
3. [ ] Train 1w with sparse+parallel CPCV (4 folds, ~15-20 min)
4. [ ] Verify: v3.3 fingerprint columns present, feature count > v3.2
5. [ ] Verify: PARALLEL CPCV in log, fold completions, model saved
6. [ ] Record: exact feature count, cross count, fold times, total time

### Phase 2: Deploy remaining TFs to vast.ai
7. [ ] Search vast.ai for machines (desktop CPU priority: Ryzen 9 / i9)
8. [ ] Present machine picks to user with real CPU models + pricing
9. [ ] Rent machines (user approval per memory rules)
10. [ ] Upload tar with ALL v3.3 code + write launch.sh to disk
11. [ ] Smoke test each machine
12. [ ] Launch with launch.sh (not inline SSH — prevents process death)
13. [ ] Post-launch validation: PARALLEL CPCV, fold times, no errors
14. [ ] Download artifacts as each TF completes
15. [ ] Kill machines

### Phase 3: Validate production readiness
16. [ ] All 5 models present + correct feature counts
17. [ ] SHAP shows cross features contributing
18. [ ] PBO audit passes
19. [ ] Paper trade test
20. [ ] Commit final artifacts + update SESSION_RESUME

## BUGS FIXED THIS SESSION (8 total)

| # | Fix | File | Status |
|---|-----|------|--------|
| A | Sparse flag tracking | ml_multi_tf.py | Reverted to `True` (sparse+parallel is fastest) |
| B | Step 5 --search-mode removed | cloud_run_tf.py | DONE |
| C | SHAP .toarray() OOM | cloud_run_tf.py | DONE — split importance only |
| D | Audit .db vs .parquet | backtesting_audit.py | DONE |
| E | Artifact name mismatch | cloud_run_tf.py | DONE |
| F | NNZ guard (15m int32 overflow) | ml_multi_tf.py | DONE |
| G | GPU skip >100K rows | v2_cross_generator.py | DONE |
| H | Cross gen dedup ordering | v2_cross_generator.py | DONE — NPZ saved after dedup |
| + | Feature fingerprint check | cloud_run_tf.py | DONE |
| + | Tiered worker concurrency | ml_multi_tf.py | DONE — OOM-safe for 1h/15m |
| + | cross_cols=[] init | ml_multi_tf.py + backtesting_audit.py | DONE |
| + | psutil /proc/meminfo fallback | ml_multi_tf.py | DONE |

## RESEARCH FINDINGS (5 Perplexity queries)

- **LightGBM CPU is THE fastest** for our ultra-sparse matrix (confirmed by Perplexity with full context)
- **XGBoost GPU fits in H100** but likely SLOWER for 96% sparse binary features
- **GOSS** worth testing for 1.5-2x per-fold speedup (training step only)
- **Tiered concurrency** prevents OOM: 3 workers for 15m, all folds for small TFs
- **ScalaGBM** = research GPU GBDT (39x claim), NOT production-ready
- **Desktop CPUs** (Ryzen 9 5.7 GHz, i9 5.5 GHz) are 1.6x faster per fold than server EPYCs (3.5-3.7 GHz)
- **vast.ai cpu_ghz is unreliable** — must identify actual CPU model

## MACHINE STRATEGY FOR VAST.AI

Priority: Desktop CPU (Ryzen 9 / i9) > EPYC server. GHz matters more than core count.

| TF | Min RAM | Workers | CSR×Workers | Est. fold time (desktop) | Est. total |
|----|---------|---------|-------------|--------------------------|------------|
| 1w | 64 GB | 4 | 1.1 GB | ~10 min | ~25 min |
| 1d | 64 GB | 4 | 11.5 GB | ~15 min | ~1 hr |
| 4h | 128 GB | 8 | 33 GB | ~15 min | ~1 hr |
| 1h | 256 GB | 8 | 108 GB | ~20 min | ~2 hrs |
| 15m | 256 GB | 3 | 48 GB | ~30 min | ~4-5 hrs |

## GIT STATUS
- Commit d1e8d55: Cross gen dedup fix
- Branch: v3.3 (3 commits ahead of main today)
- Uncommitted: tiered concurrency, sparse revert, local test
