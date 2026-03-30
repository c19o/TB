# EXPERT: LightGBM Dataset Construction for 10M+ Binary Sparse Features

## Research Sources
- Perplexity searches (2026-03-30): LightGBM Dataset optimization, forcedbins, parallel construction, max_bin=2
- GitHub issues: #5205 (slow sparse loading), #6311 (Dataset construction bottleneck), #876 (binning optional), #5478 (sparse save_binary)
- LightGBM 4.6.0 docs: Parameters, Features, Parameters-Tuning, C API, Python API

---

## 1. The Problem

Dataset.construct() is the slowest step for our matrix:
- **10M+ pure binary (0/1) sparse features** in CSR format
- **227K rows** (typical for 1h/15m timeframes)
- All features are co-occurrence crosses: mutually exclusive binary indicators
- LightGBM must bin, index, and optionally EFB-bundle every feature

The bottleneck is **PushDataToMultiValBin** -- an inner loop that enumerates all features for each row in sparse data. With 10M+ columns, this dominates runtime even though each feature has trivial 0/1 values.

---

## 2. Current V3.3 Implementation

### Config (config.py)
```python
V3_LGBM_PARAMS = {
    "max_bin": 255,           # max EFB bundle size (254/bundle)
    "force_col_wise": True,   # recommended for many columns
    "is_enable_sparse": True,
    "feature_pre_filter": False,  # CRITICAL: True kills rare features
    "min_data_in_bin": 1,
    "num_threads": 0,         # auto-detect
}
```

### Parallel Construction (run_optuna_local.py)
Already implemented `_parallel_dataset_construct()`:
- Splits CSR column-wise into chunks
- Builds each chunk in parallel via ProcessPoolExecutor (spawn context)
- Merges via `add_features_from()` (metadata only, no re-binning)
- Uses `save_binary()` / reload for inter-process transfer
- 10-50x faster than single-threaded for millions of features (GitHub #5205 endorsed)

---

## 3. Key Findings from Research

### 3a. max_bin=255 is Correct (Do NOT change to 2)

**Why max_bin=255 stays:**
- `max_bin` controls **EFB bundle capacity**, not per-feature bin count
- Binary features automatically get 2 bins regardless of max_bin setting
- With max_bin=255, EFB packs up to 254 mutually exclusive binary features per bundle
- 10M features / 254 per bundle = ~39K effective bundles
- Changing to max_bin=2 would **disable EFB bundling** (only 1 feature/bundle = 10M bundles)
- This is already validated: `validate.py` line 81 enforces `max_bin == 255`

**Per-feature bin count:**
LightGBM assigns bins based on actual distinct values, not max_bin. A binary 0/1 feature always gets exactly 2 bins. max_bin=255 only means "allow up to 255 bins if a feature needs them."

### 3b. bin_construct_sample_cnt Can Be Reduced

- Default: 200,000 rows sampled to discover bin boundaries
- For binary features, there are only 2 distinct values (0 and 1)
- Sampling 200K rows to find {0, 1} is overkill
- **Reducing to ~5,000-10,000 saves scanning time** without quality loss
- Perplexity confirms: "binary-only case is one of the rare setups where marginal value of large bin-sampling is especially low"
- **Caveat:** Do NOT set too low (<1000) -- LightGBM warns this can cause issues

### 3c. forcedbins_filename -- Skip Bin Discovery Entirely

LightGBM supports pre-specifying bin upper bounds via a JSON file:
```json
[
    {"feature": 0, "bin_upper_bound": [0.5]},
    {"feature": 1, "bin_upper_bound": [0.5]},
    ...
]
```

For pure binary features, threshold 0.5 perfectly splits 0 from 1.

**Status:** Documented feature, works via `params["forcedbins_filename"]`. However:
- Generating a 10M-entry JSON file has its own cost
- LightGBM still runs ingestion/mapping even with forced bins
- The real bottleneck is sparse traversal, not quantile computation
- **Verdict: marginal benefit for our case since bin discovery is already trivial for binary data**

### 3d. save_binary Caching -- Major Win for Repeated Construction

`Dataset.save_binary()` serializes the fully constructed internal representation. Reloading skips ALL binning/EFB/indexing.

**Current status:** Already used in `_parallel_dataset_construct()` for inter-chunk transfer, but NOT used for cross-fold caching.

**Opportunity:** In CPCV with 5 folds, the same features are re-binned up to 5 times with slightly different row subsets. Caching the parent Dataset and creating fold-specific subsets could eliminate redundant construction.

### 3e. num_threads and Parallelism

- `num_threads` affects Dataset.construct() but **does not scale linearly** for sparse data
- Bottleneck is memory bandwidth and sparse index traversal, not compute
- `force_col_wise=True` is the most important threading knob (already set)
- Set num_threads to physical cores, not SMT threads
- **Our parallel chunk approach bypasses this limitation entirely** by using multi-process parallelism

### 3f. Known LightGBM Bug: O(features * rows_per_feature) Inner Loop

GitHub #5205 documents that sparse dataset loading has an inner loop that enumerates all features for each row. For 10M features, even if most are zero, the metadata traversal is O(nnz) with high constant factors.

**This is the fundamental bottleneck.** No parameter tuning eliminates it -- only the parallel chunk approach (which we already have) or a LightGBM source patch can fix it.

---

## 4. Optimization Recommendations

### 4a. ALREADY DONE (keep these)

| Setting | Value | Why |
|---------|-------|-----|
| `max_bin=255` | EFB bundles 254 features each | 10M -> ~39K effective features |
| `force_col_wise=True` | Column-oriented histogram | Recommended for wide datasets |
| `is_enable_sparse=True` | Sparse optimization | CSR native path |
| `feature_pre_filter=False` | Protect rare features | Non-negotiable for matrix thesis |
| `min_data_in_bin=1` | Allow singleton bins | Rare esoteric signals |
| Parallel chunk construction | Multi-process column splits | 10-50x faster (GitHub #5205) |

### 4b. IMPLEMENT: bin_construct_sample_cnt Reduction

**Impact: moderate (10-30% faster Dataset.construct())**

Add to Dataset params:
```python
'bin_construct_sample_cnt': 5000  # Binary features: only need to see {0,1}
```

Why 5000: guarantees discovering both 0 and 1 for any feature with >=0.1% prevalence (which covers all our cross features). Default 200K is 40x more than needed.

Add to both:
1. `_parallel_dataset_construct()` params dict (line 1041)
2. Single-threaded fallback params (line 1028)
3. `_ds_params` in feature_classifier.py (line 163)

### 4c. IMPLEMENT: Dataset Binary Caching for CPCV Folds

**Impact: high (eliminates repeated construction across folds)**

Strategy:
1. Build the full Dataset ONCE for all features
2. Save to binary: `full_ds.save_binary('full_dataset.bin')`
3. For each CPCV fold, reload from binary (skips binning)
4. Use `subset()` or row-slicing for fold-specific train/val splits

```python
# Build once
full_ds = _parallel_dataset_construct(X_csr, y_dummy, weights)
full_ds.save_binary(cache_path)

# Per fold (fast reload, no re-binning)
for fold_idx, (train_idx, val_idx) in enumerate(cpcv_splits):
    dtrain = lgb.Dataset(cache_path).construct().subset(train_idx)
    dval = lgb.Dataset(cache_path).construct().subset(val_idx)
```

**Caveat:** `subset()` API availability depends on LightGBM version. Alternative: build per-fold from cached binary chunks.

### 4d. CONSIDER: use_missing=False, zero_as_missing=False

For pure binary sparse features where 0 means "feature OFF" (not missing):
```python
'use_missing': False,
'zero_as_missing': False,
```

**Current behavior:** CSR structural zeros = 0.0 (correct per CLAUDE.md rule). But LightGBM's default `use_missing=True` adds an extra bin for NaN handling on every feature. With 10M+ features, this adds overhead.

**Risk:** If ANY features legitimately use NaN (non-cross features), this breaks them. Only safe if the Dataset is purely binary crosses.

### 4e. CONSIDER: enable_bundle Benchmark

EFB (Exclusive Feature Bundling) adds upfront construction cost to save training time.

- For 10M binary features -> ~39K bundles: EFB is **critical** for training speed
- But EFB discovery itself is O(features^2) conflict graph construction
- If construction is the bottleneck and training is fast, consider building without EFB for HPO trials and with EFB for final model

**Verdict:** Keep EFB always on. The 39K-bundle training speedup far outweighs construction cost.

---

## 5. Parameter Quick Reference

### Optimal Dataset params for V3.3 matrix:
```python
_ds_params = {
    'feature_pre_filter': False,     # protect rare features
    'max_bin': 255,                   # EFB: 254 features/bundle
    'min_data_in_bin': 1,             # rare signal bins
    'is_enable_sparse': True,         # CSR native
    'force_col_wise': True,           # wide dataset optimization
    'bin_construct_sample_cnt': 5000, # binary features need minimal sampling
}
```

### Training params (unchanged):
```python
V3_LGBM_PARAMS = {
    'enable_bundle': True,   # EFB mandatory
    'force_col_wise': True,
    'num_threads': 0,        # auto = physical cores
    ...
}
```

---

## 6. Architecture: Why Parallel Chunks Work

```
                    10M features (CSR)
                          |
              +-----------+-----------+
              |           |           |
         Chunk 0     Chunk 1    ...  Chunk N
        (625K cols)  (625K cols)    (625K cols)
              |           |           |
         [Process 0] [Process 1] [Process N]
         construct()  construct()  construct()
         save_binary  save_binary  save_binary
              |           |           |
              +-----------+-----------+
                          |
                    add_features_from()
                    (metadata merge, no re-binning)
                          |
                    Full Dataset (10M features)
```

- Each process handles ~625K columns (for 16 workers)
- PushDataToMultiValBin runs in parallel across processes
- Binary save/reload is the IPC mechanism
- Merge is O(chunks) metadata concatenation
- **Bypasses the O(features) inner loop bottleneck** by dividing it across processes

### Worker count formula:
```python
n_workers = min(64, max(4, cpu_count // 4))
```
- 13900K (24 cores): 6 workers -> ~1.7M cols/worker
- 128-core cloud: 32 workers -> ~312K cols/worker

---

## 7. Timing Expectations

| Feature Count | Method | 227K Rows | Notes |
|--------------|--------|-----------|-------|
| 100K | Single-threaded | ~5s | Below parallel threshold |
| 1M | Single-threaded | ~120s | Dominated by PushDataToMultiValBin |
| 1M | Parallel (8 workers) | ~20s | 6x speedup |
| 6M | Single-threaded | ~600s+ | Impractical |
| 6M | Parallel (16 workers) | ~60-90s | Current V3.3 approach |
| 10M | Parallel (16 workers) | ~100-150s | Expected for 15m TF |
| 10M | Parallel + sample_cnt=5K | ~80-120s | With bin sampling reduction |

---

## 8. Known Limitations

1. **LightGBM does not support skipping binning entirely** -- histogram-based design requires it (GitHub #876: "Make binning optional" -- closed, won't fix)

2. **forcedbins_filename** exists but does not eliminate sparse traversal overhead -- only skips quantile computation (which is already trivial for binary)

3. **EFB conflict graph** is O(features^2) in theory but LightGBM uses a greedy approximation that is near-linear in practice for sparse data

4. **save_binary reload** still requires deserializing the full internal representation -- faster than construction but not instant for 10M features

5. **subset() API** may not be available in all LightGBM versions -- test before relying on it for fold caching

---

## 9. Action Items

| Priority | Item | Impact | Effort |
|----------|------|--------|--------|
| HIGH | Add `bin_construct_sample_cnt=5000` to all Dataset params | 10-30% construction speedup | 5 min |
| HIGH | Cache constructed Dataset binary across CPCV folds | Eliminates N-1 redundant constructions | 30 min |
| LOW | Test `use_missing=False` for pure binary Datasets | Minor speedup, risk if mixed features | 15 min |
| NONE | Change max_bin | DO NOT CHANGE -- 255 is correct for EFB | - |
| NONE | Disable EFB | DO NOT DISABLE -- 10M->39K bundles is critical | - |
