# CPCV Fold Strategy — Research & Decision Log

## Decision: (4,1)=4 folds for ALL timeframes (2026-03-27)

### Why (4,1) is sufficient

**Perplexity confirmed (3 separate consultations):**

1. **Production model accuracy is IDENTICAL regardless of fold count.** The final model trains on ALL rows with ALL features. It never sees fold results. CPCV fold count only affects evaluation quality, not the trading model.

2. **Platt calibration with 4 folds is sufficient** for a 3-class model with 5K-294K rows. Slightly noisier than 15 folds, mitigated by regularization (LogisticRegression max_iter=500). Total OOS sample count matters more than fold count.

3. **Meta-labeling on 4 OOS sets vs 15** — negligible difference when row counts are 17K+ (4h and above). Simple logistic meta-model doesn't need 15 distinct OOS sets.

4. **PBO with 4 paths is borderline** for statistical significance, but PBO is informational only — not used for trading decisions or strategy selection. Acceptable for a single-operator system.

5. **60-70% compute savings** — from (5,2)=10 or (6,2)=15 folds down to (4,1)=4. Directly translates to 60-70% less cloud cost and wall time.

### What each fold count actually affects

| Component | 4 folds | 10-15 folds | Impact on live trading |
|-----------|---------|-------------|----------------------|
| Final model | Identical | Identical | ZERO — trains on all data |
| Platt calibration | Slightly noisier | Smoother | Negligible — regularized LR |
| Meta-labeling | Slightly less training data | More training data | Minor — simple logistic model |
| PBO reliability | Borderline | Statistically robust | Informational only |
| Feature importance stability | Less stable ranking | More stable ranking | Diagnostic only |
| Wall time | 60-70% faster | Baseline | Significant cost savings |

### The matrix thesis perspective

The matrix thesis says: every feature matters, every row matters, the model decides. CPCV fold count doesn't touch any of these:
- ALL features present in every fold (no filtering)
- ALL rows in the final model (no subsampling)
- Same LightGBM params, same EFB, same sparse CSR
- The model's tree splits are identical whether we evaluated it with 4 or 15 folds

### Per-TF config

```python
TF_CPCV_GROUPS = {
    '1w': (4, 1),   # 818 rows, 4 folds → ~205 rows/test set
    '1d': (4, 1),   # 5,733 rows, 4 folds → ~1,433 rows/test set
    '4h': (4, 1),   # 17,520 rows, 4 folds → ~4,380 rows/test set
    '1h': (4, 1),   # 75,405 rows, 4 folds → ~18,851 rows/test set
    '15m': (4, 1),  # 293,980 rows, 4 folds → ~73,495 rows/test set
}
```

All test sets are large enough for reliable OOS evaluation. Even 1w with 205 test rows covers ~4 years of weekly bars.

---

## Multi-Machine Distribution (RESEARCHED, NOT IMPLEMENTED)

### The plan that was designed but scrapped

We designed and Perplexity-audited a full N-machine CPCV distribution system:
- Primary machine does feature build + cross gen
- N worker machines each run a subset of CPCV folds
- Pre-computed splits.pkl shipped to all workers (eliminates determinism risk)
- Merge checkpoints on primary, then final retrain + downstream

### Why it was scrapped
- Cross gen is 60-80% of wall time — can't be parallelized across machines
- Adding fold workers only saves 20-40% of total time
- SCP overhead + monitoring complexity + merge step
- Cost savings minimal (workers are cheap but still add up)
- (4,1)=4 folds makes the fold phase fast enough on one machine

### If you want to revive it later

The code changes would be:
1. `ml_multi_tf.py`: Add FOLD_START/FOLD_END/MACHINE_ID env vars, per-machine checkpoint naming, partial run early exit, --merge-and-retrain flag, --load-splits flag
2. `cloud_run_tf.py`: Add --folds-only flag (skip steps 0-3), --merge-and-retrain flag
3. New `merge_cpcv_checkpoints.py`: Glob checkpoints, validate unique fold IDs, merge OOS predictions
4. New `distribute_folds.py`: Compute fold assignments for N machines

### Matrix audit results (Perplexity deep audit)
- SCP NPZ fidelity: SAFE (binary-safe, scipy preserves dtypes)
- EFB non-determinism: SAFE (each fold is a different model anyway)
- HMM state permutation: ALREADY FIXED (fit_hmm_on_window sorts by mean return)
- int64 indptr: SAFE (scipy load_npz preserves int64)
- Final model quality: IDENTICAL (trains on all data, never reads fold results)
- Required safeguards: sha256sum on NPZ after SCP, assert indptr.dtype==int64, pin scipy/numpy versions

### When multi-machine IS worth it
- If you increase folds back to (6,2)=15 and want speed
- If cross-gen optimizations (Numba prange) cut cross gen to <1hr, making folds the bottleneck
- For 1h/15m where each fold takes 100-150 min (4 folds = 6-10 hrs just for CPCV)
- Max useful machines: ~5-7 for 15 folds, capped by SCP overhead
