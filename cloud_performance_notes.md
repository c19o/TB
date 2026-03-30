# Cloud Performance Notes — March 2026 (Updated after 3 failed runs)

## CRITICAL BUGS FOUND AND FIXED

### Bug 1: DOY crosses not saved to parquet
- `_add_trend_cross_features()` modified `df` in-place but used `pd.concat` for batch assignment
- `pd.concat` creates a NEW DataFrame — the caller's reference still pointed to OLD df without dx_ columns
- Fix: function now returns `df`, caller does `result = _add_trend_cross_features(result, tf_name)`

### Bug 2: cudf.pandas proxy breaks CuPy arrays
- `.values` on cudf.pandas proxy returns a cudf array, not numpy
- `np.column_stack()` on cudf arrays fails silently or produces wrong results
- CuPy can't transfer cudf proxy objects to GPU
- Fix: explicit `_np.array(data, dtype=_np.float32)` conversion before any CuPy operation

### Bug 3: parquet validation uses unsupported parameter
- `pd.read_parquet(path, nrows=5)` — `nrows` not supported by all pyarrow versions
- Fix: `pd.read_parquet(path).head(5)`

### Bug 4: build_1w and build_1d missing parquet save
- Only saved to SQLite which crashes at >2000 columns
- Fix: parquet-first save, SQLite as non-fatal backup

### Bug 5: DOY cross column assignment bottleneck
- 135K individual `df[col] = data` calls take 50+ minutes on 2.2GHz CPU
- Fix: collect all dx_ columns in dict, then `pd.concat([df, pd.DataFrame(dict)])` once
- Expected improvement: ~60% faster

## Bottleneck Discovery: DOY Cross Column Assignment

The DOY × ALL CONTEXTS cross generation has TWO phases:
1. **GPU computation** (CuPy outer product) — FAST, minutes even on A40
2. **Column assignment** (writing ~135K columns back to pandas DataFrame) — SLOW, CPU-bound

The GPU part computes `doy_batch[:, :, None] * ctx_gpu[:, None, :]` in seconds.
But then it loops through each result and does `df[f'dx_{d}_x_{ctx_short}'] = col_data`
one column at a time. With 135K columns, this is 135K individual pandas column assignments.
Each one triggers pandas internal reindexing/memory allocation.

**This is the real bottleneck — not GPU, not RAM, not VRAM. It's pandas column assignment speed.**

### Observed Timings

| Machine | TF | Rows | DOY × Contexts | GPU Time | Column Write Time | Total |
|---------|-----|------|----------------|----------|-------------------|-------|
| 4× H100 ($6.53/hr) | 1h | 57K | 365×1011 | ~5 min | ~15+ min | ~20 min |
| 4× A40 ($1.15/hr) | 4h | 14K | 365×995 | ~3 min | ~50+ min (est) | ~53+ min |
| 4× A40 ($1.15/hr) | 1w | 339 | 287×854 | ~1 min | ~25+ min | ~26+ min |

### Why A40 is Much Slower Than Expected
- GPU computation is only ~30% of DOY cross time
- The other 70% is CPU-bound pandas column assignment
- A40 and H100 machines both have 2.2GHz EPYC CPUs — SAME speed
- But the A40 machine is doing 3 TFs SIMULTANEOUSLY (Phase 1 parallel)
- 3 processes × 135K column assignments competing for CPU cache = ~3x slower per process
- H100 was also slow on this but we killed it before completion

### Potential V2 Fixes for Column Assignment Speed

1. **Batch column assignment** — instead of `df[col] = data` one at a time,
   build a numpy 2D array of all crosses then `pd.DataFrame(array, columns=names)` once.
   Eliminates 135K individual pandas operations → 1 operation.

2. **Pre-allocate DataFrame** — create the DataFrame with all 135K NaN columns first,
   then fill them. Avoids pandas growing the DataFrame 135K times.

3. **Write directly to parquet** — skip pandas DataFrame entirely for dx_ columns.
   Write CuPy results directly to Arrow/Parquet column chunks.

4. **Use numpy array, not DataFrame** — keep dx_ crosses as a separate numpy array
   alongside the pandas DataFrame. Only combine at final parquet save.

### Machine Selection Lessons

| Priority | What Matters | Why |
|----------|-------------|-----|
| 1 | **CPU clock speed** | Column assignment is single-threaded Python |
| 2 | **CPU cores** | Parallel TF builds compete for CPU |
| 3 | **RAM** | 5m needs ~450GB, must fit |
| 4 | **VRAM** | DOY GPU computation needs ~80-180GB across GPUs |
| 5 | **GPU speed** | Only matters for the actual CuPy multiply (~30% of time) |

**Best machine for this workload: fast CPU (3.7GHz+) + moderate GPU + lots of RAM.**
The 4× A100 SXM4 in Massachusetts ($4.37/hr, 3.7GHz) would have been faster
than both the H100 India ($6.53/hr, 3.7GHz but we killed it) and A40 Belgium ($1.15/hr, 2.2GHz).

### Phase 1 Parallel Was a Mistake for A40
Running 3 TFs simultaneously on a 2.2GHz machine means:
- 3 processes doing 135K column assignments each
- CPU cache thrashing between 3 large DataFrames
- Each process ~3x slower than running solo

Sequential would have been better on this slow CPU.
Parallel only makes sense on fast CPU (3.7GHz+) with many cores.

### Cost Comparison for Full 6-TF Pipeline

| Machine | $/hr | Est. Time | Total Cost | Bottleneck |
|---------|------|-----------|------------|------------|
| 4× H100 SXM India | $6.53 | ~90 min | ~$9.80 | Killed early, didn't finish |
| 4× A40 Belgium | $1.15 | ~3-4 hrs | ~$3.50-4.60 | CPU-bound column writes |
| 4× A100 SXM4 MA | $4.37 | ~60-75 min | ~$4.40-5.50 | Would have been best balance |
| Ideal: batch column fix | any | -60% time | -60% cost | V2 optimization |

## ALL Bottlenecks Observed Across Cloud Runs

### 1. DOY Cross Column Assignment (BIGGEST — described above)
- 135K individual `df[col] = data` calls
- Each one triggers pandas internal copy/realloc
- ~70% of total DOY cross time is this, not GPU compute
- Fix: batch into numpy array, assign once

### 2. Systematic Cross Generation (241 seconds on H100)
- `_add_cross_features()` and systematic crosses from CSV
- Created 29,940 systematic crosses + 3,511 skipped
- Loops through features in Python, checking conditions
- Fix: vectorize with numpy boolean ops instead of Python loops

### 3. HMM Fitting Per CPCV Fold
- 3 seeds × 100 iterations × GaussianHMM on daily data
- Runs for EACH of 15 CPCV folds = 45 HMM fits total
- Each fit ~30-60 seconds = 22-45 minutes just on HMM
- Fix: fit HMM ONCE on full training data, cache states, reuse across folds
- Or: use pre-computed regime features from parquet, skip HMM entirely

### 4. Esoteric Feature Bucketing (tweet/news groupby)
- `tw.groupby('bucket').agg(...)` with 20+ aggregation functions
- Runs separately for each bucket size (per-bar, daily)
- Then maps back via `.map(dict)` which cudf.pandas can't accelerate
- ~25s for tweets, ~15s for news on 1H
- Fix: pre-aggregate at data loading time, store aggregated results

### 5. cudf.pandas Fallback Overhead
- ~40 operations fall back to CPU (dict .map(), .rolling().quantile(), etc.)
- Each fallback requires GPU→CPU data transfer + CPU compute + CPU→GPU transfer
- Invisible but adds ~10-20% overhead on the GPU-accelerated ops
- Fix: replace .map(dict) with .merge() joins, replace .rolling().quantile() with approximation

### 6. Space Weather Reindex
- 34,412 days of space weather reindexed to 547K bars (5m)
- Forward-fill across sub-daily bars
- Timezone mismatch crashes (now fixed) but reindex itself is slow
- Fix: pre-compute space weather at bar frequency during data loading

### 7. Parquet Save for 150K+ Columns
- PyArrow Parquet writer allocates per-column buffers
- 150K columns = 150K encoder instances
- Known OOM issue for very wide DataFrames
- Current fix: SQLite fails gracefully, parquet works
- Better fix: save as chunked parquet (30 files × 5K cols) or numpy .npy

### 8. XGBoost DMatrix Construction
- Converting 547K × 60K float32 array to DMatrix
- Quantile sketch scales linearly with feature count
- With CPCV (15 folds), this happens 15 times
- Fix: sparse CSR input (dx_ crosses are 99.7% zeros), or cache DMatrix

### 9. MI Pre-screening Inside CV Folds
- `mutual_info_classif` on 500K rows × 60K features is expensive
- Runs inside each CPCV fold = 15 times
- sklearn MI uses k-nearest-neighbors internally, O(n * features * k)
- Fix: subsample rows for MI computation (10K rows is enough for MI estimates)

### 10. Output Buffering (not performance but wastes monitoring time)
- cudf.pandas wraps stdout, kills unbuffered mode even with PYTHONUNBUFFERED=1
- Long-running steps appear frozen because print output is buffered
- Fix: flush=True on every print, or write to separate log file with os.write()

### 11. Multi-Process GPU Contention
- When 3 TFs build simultaneously, all use GPU0 for cudf.pandas base features
- CuPy multi-GPU distributes DOY crosses but cudf.pandas doesn't
- GPU0 maxes out while GPU1-3 idle during base feature compute
- Fix: stagger builds so only one is in cudf.pandas phase at a time,
  or use CUDA MPS (Multi-Process Service) to share GPU more efficiently

### 12. KNN Feature Computation
- knn_feature_engine.py fits KNN on each bar's history
- O(n² × features) computation
- Often the slowest single compute_* call
- Fix: subsample historical bars for KNN, or pre-compute KNN features offline

## Priority Fix Order for V2

| Fix | Impact | Effort | Priority |
|-----|--------|--------|----------|
| Batch DOY column assignment | -60% build time | 2 hrs | 1 |
| Cache HMM across CPCV folds | -30 min per TF | 1 hr | 2 |
| Subsample MI pre-screening | -5 min per fold | 30 min | 3 |
| Vectorize systematic crosses | -4 min per TF | 2 hrs | 4 |
| Pre-aggregate esoteric buckets | -40s per TF | 3 hrs | 5 |
| Sparse CSR for XGBoost | -2 min per fold | 1 hr | 6 |
| Replace .map(dict) with .merge() | -10% overhead | 4 hrs | 7 |
| Pre-compute space weather bars | -5s per TF | 1 hr | 8 |
| Chunked parquet writes | prevents OOM | 1 hr | 9 |
| Subsample KNN | -30s per TF | 30 min | 10 |

**Combined impact: ~60-70% faster pipeline.**
Current 3-4 hour run on A40 → ~1-1.5 hours.
Current ~90 min on H100 → ~30-40 min.
