# EXPERT: Python Performance for Sparse ML Pipelines

**Context**: LightGBM pipeline with 2-10M sparse binary features, CPCV parallel folds, Optuna parallel trials, pandas feature engineering from OHLCV data.

---

## 1. SharedMemory IPC for Sparse Matrix CPCV Parallelism

### Core Pattern: Share CSR Buffers, Not Objects

SciPy CSR is just three arrays (`data`, `indices`, `indptr`) + shape. Share those buffers via `multiprocessing.shared_memory.SharedMemory`, then reconstruct lightweight CSR views in each worker. **Never pickle/send the sparse matrix through queues.**

For binary sparse features, `data` is all ones -- the largest buffers are `indices` and `indptr`. These are the serialization bottleneck.

### Recommended Architecture

```
Parent process:
  1. Load/build one canonical CSR matrix
  2. Export data, indices, indptr to shared memory (or memmap)

Worker initializer:
  3. Attach to shared buffers by name
  4. Create one global CSR view (cached in process globals)

Fold task receives ONLY:
  5. Fold row indices, purge/embargo boundaries, labels, hyperparams
  6. Worker builds fold-local row subsets, trains LightGBM (low num_threads)
  7. Returns ONLY metrics/small artifacts
```

### SharedMemory vs Memmap

| Method | Best When | Advantage |
|--------|-----------|-----------|
| `SharedMemory` | Matrix fits in RAM, single host, lowest latency | Zero-copy, deterministic lifecycle |
| `numpy.memmap` | Near-RAM-limit, many CPCV folds, crash resilience | OS page cache sharing, survives worker crashes |
| Joblib memmap | Already using joblib, want minimal changes | Auto-memmaps arrays > `max_nbytes`, uses `/dev/shm` |

**Python 3.13+ note**: `SharedMemory` has a `track` parameter. Independent processes with separate resource trackers can accidentally delete segments when the first tracker-owned process exits. If workers come from a common multiprocessing ancestor, normal tracking is fine; otherwise manage lifecycle carefully or set `track=False`.

### Minimal Implementation Shape

```python
# Parent: publish CSR buffers
shm_data = SharedMemory(create=True, size=X.data.nbytes, name='csr_data')
shm_idx  = SharedMemory(create=True, size=X.indices.nbytes, name='csr_indices')
shm_ptr  = SharedMemory(create=True, size=X.indptr.nbytes, name='csr_indptr')
# Copy once
np.ndarray(X.data.shape, dtype=X.data.dtype, buffer=shm_data.buf)[:] = X.data
# ... same for indices, indptr

# Worker initializer: attach + reconstruct
def init_worker(shape, dtypes):
    global _X
    shm_d = SharedMemory(name='csr_data', create=False)
    data = np.ndarray(shape_d, dtype=dtypes['data'], buffer=shm_d.buf)
    # ... same for indices, indptr
    _X = csr_array((data, indices, indptr), shape=shape)

# Fold task: slice rows, train, return score only
def train_fold(fold_rows, labels, params):
    X_train = _X[fold_rows]
    ds = lgb.Dataset(X_train, label=labels)
    model = lgb.train(params, ds)
    return score
```

### Critical Anti-Pattern: Oversubscription

The #1 reason multiprocessing is slower than expected with sparse workloads: **nested parallelism**. You parallelize across folds while LightGBM also uses OpenMP threads.

**Rules**:
- Parallelize at ONE level only (usually across folds)
- Set LightGBM `num_threads` low per worker (1-4 depending on core count)
- Pin `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`
- Formula: `num_workers x num_threads <= total_cores`

### Process Model

On Linux, **never use `fork` with LightGBM** -- documented OpenMP deadlock risk. Use `spawn` or `forkserver`. Accept the memory cost of fresh processes.

---

## 2. Python Free-Threading (3.13/3.14) Impact on LightGBM

### What Changed

- Python 3.13: Optional free-threaded build (GIL disabled)
- Python 3.14: Maturing, ~1% overhead on macOS aarch64, ~8% on x86-64 Linux
- Still optional, not default. Many C extensions not ready (may re-enable GIL on import)

### What It Helps

Free-threading helps **Python orchestration layers**, not LightGBM's native core:
- CPCV fold scheduling and bookkeeping
- Optuna trial coordination, sampler logic, callbacks
- Sparse matrix slicing, custom label logic, metric aggregation
- Result aggregation and artifact writing

### What It Does NOT Help

- LightGBM tree-building (already native OpenMP, outside Python)
- OpenMP oversubscription risk (unchanged)
- Memory bandwidth contention on large sparse matrices
- The documented OpenMP + fork deadlock on Linux

### Practical Impact by Component

| Component | GIL-Free Benefit | Notes |
|-----------|-----------------|-------|
| CPCV fold scheduler | **Medium** | Helps if substantial Python work before/after LightGBM fit |
| Optuna trial orchestration | **Medium-High** | Samplers, pruners, callbacks are Python-heavy |
| LightGBM training core | **None** | Already native code |
| Sparse matrix ops | **Low-Medium** | SciPy/NumPy mostly release GIL already |
| Feature assembly | **High** | Pure Python UDFs benefit most |

### Recommended Architecture (2025-2026)

1. **Single process, shared memory** when possible -- avoids dataset duplication entirely
2. **Thread-based outer parallelism** for CPCV/Optuna IF all libraries confirmed free-thread-safe
3. **Low `num_threads` per LightGBM** when outer concurrency > 1
4. **Avoid Linux `fork`** with multithreaded LightGBM workers -- use `spawn`
5. **Benchmark three layouts** on real hardware:
   - `(many workers x 1 thread)`
   - `(few workers x medium threads)`
   - `(threads outside + LightGBM low-thread inside)`

### When to Adopt

- Python 3.14 free-threading: **promising but validate your exact stack**
- Test that SciPy, LightGBM, Optuna, NumPy all work without re-enabling GIL
- For production 2026: treat as opt-in optimization, not default
- **Do not rely on it** -- design for multiprocessing first, thread-based as upgrade path

---

## 3. Pickle Alternatives for Large Sparse Matrix IPC

### Ranked Options (Fastest to Slowest)

| Rank | Method | Speed | When to Use |
|------|--------|-------|-------------|
| 1 | **Threading** (no IPC needed) | Fastest | If model code releases GIL well |
| 2 | **SharedMemory CSR buffers** | Near-zero copy | Single host, repeated reuse of same matrix |
| 3 | **Joblib memmap of CSR internals** | Good | Want simpler integration |
| 4 | **Protocol 5 / stdlib pickle** | OK | Drop-in improvement, no redesign |
| 5 | **cloudpickle (loky default)** | Slowest | Only when needed for dynamic functions |

### Why Pickle Is the Bottleneck

Joblib's `loky` backend uses `cloudpickle` by default. Even when the CSR payload is mostly NumPy buffers, wrapping and shipping the sparse object repeatedly pays serialization overhead. **cloudpickle can be much slower than stdlib pickle** on large object graphs.

### Quick Wins Without Redesign

1. Set `LOKY_PICKLER=pickle` environment variable (switches loky from cloudpickle to stdlib pickle)
2. Use Protocol 5 out-of-band buffers for NumPy arrays
3. Set joblib `max_nbytes` to trigger automatic memmapping for large arrays
4. Prefer `/dev/shm` (tmpfs) for joblib temp folder on Linux

### Zero-Copy Design

The real fix: **stop serializing the matrix entirely**.

```python
# Instead of this (slow -- pickles entire sparse matrix per fold):
joblib.Parallel(n_jobs=8)(
    delayed(train_fold)(X_train, y_train, params) for X_train, y_train in folds
)

# Do this (fast -- passes only row indices):
publish_csr_to_shared_memory(X)  # once
joblib.Parallel(n_jobs=8, prefer='processes')(
    delayed(train_fold)(fold_row_ids, labels, params) for fold_row_ids, labels in folds
)
# Worker reconstructs CSR view from shared memory, slices rows locally
```

### Index Dtype Matters

Keep `indices`/`indptr` as **int32 where valid** (< 2^31 non-zeros). Halves buffer size and copying cost. For matrices exceeding int32 NNZ limit, use int64 `indptr` (already implemented in v3.3 for the large TFs).

### Ray Note

Ray's object store provides zero-copy reads for NumPy arrays but sparse matrix support is unclear. Explicit shared CSR buffers are more reliable for SciPy sparse IPC than Ray's generic serialization.

---

## 4. Vectorized Feature Engineering (Replacing pandas .apply())

### The Problem

`pandas.DataFrame.apply()` with Python UDFs is the **build bottleneck**. It iterates row-by-row in Python, paying interpreter overhead per element. For 2-10M features from OHLCV data, this is catastrophically slow.

### Performance Hierarchy

| Method | Relative Speed | Notes |
|--------|---------------|-------|
| Pure pandas vectorization | **1x** (baseline, fastest) | `df['a'] * df['b']`, `np.where()` |
| NumPy array operations | **~1x** | `.to_numpy()` then operate on arrays |
| `np.vectorize()` | **~2x faster than apply** | Syntactic sugar, NOT true vectorization |
| `df.apply()` | **~5-10x slower** | Python loop per row |
| `iterrows()` | **~100-700x slower** | Never use |

### Replacement Strategies

**1. Direct Vectorization (Best)**
```python
# BAD: apply with UDF
df['signal'] = df.apply(lambda r: r['close'] / r['open'] if r['volume'] > 0 else 0, axis=1)

# GOOD: vectorized
df['signal'] = np.where(df['volume'] > 0, df['close'] / df['open'], 0)
```

**2. np.select for Multi-Condition Logic**
```python
# BAD: complex conditional apply
df['tier'] = df.apply(classify_row, axis=1)

# GOOD: np.select
conditions = [df['val'] > 100, df['val'] > 50, df['val'] > 0]
choices = ['high', 'medium', 'low']
df['tier'] = np.select(conditions, choices, default='none')
```

**3. Batch Column Assignment (Critical for 2M+ Features)**
```python
# BAD: column-at-a-time assignment (each triggers pandas internal checks)
for feat_name, feat_values in features.items():
    df[feat_name] = feat_values  # O(n) overhead per assignment

# GOOD: batch with pd.concat
new_cols = pd.DataFrame(features)  # dict of {name: array}
df = pd.concat([df, new_cols], axis=1)

# ALSO GOOD: df.assign() for smaller batches
df = df.assign(**{name: values for name, values in batch.items()})
```

**This is the single biggest speedup for feature builds** -- batch assignment with `pd.concat` or `df.assign()` gives ~60% speedup over column-at-a-time for large feature counts.

**4. Numba JIT for Unavoidable Custom Logic**
```python
from numba import njit

@njit
def custom_feature(close, high, low, n):
    result = np.empty(len(close))
    for i in range(n, len(close)):
        result[i] = (close[i] - low[i-n:i].min()) / (high[i-n:i].max() - low[i-n:i].min() + 1e-10)
    result[:n] = np.nan
    return result

# Pass numpy arrays, not pandas Series
df['feat'] = custom_feature(df['close'].values, df['high'].values, df['low'].values, 14)
```

**5. Rolling/Expanding Window Operations**
```python
# BAD: apply with rolling
df['feat'] = df['close'].rolling(14).apply(custom_func)

# GOOD: use built-in aggregations
df['feat'] = df['close'].rolling(14).mean()
df['feat2'] = df['close'].rolling(14).std()

# For custom rolling: use stride_tricks or numba
```

### Modern Alternatives (2025-2026)

| Tool | Best For | Speedup vs pandas apply |
|------|----------|------------------------|
| **Polars** | Full DataFrame replacement | 5-50x, native Rust, lazy evaluation |
| **DuckDB** | SQL-style feature engineering | 10-100x, keeps logic in engine |
| **Numba @njit** | Custom numeric UDFs | 50-200x, compiles to machine code |
| **NumExpr** | Simple arithmetic on large arrays | 2-10x, multi-threaded evaluation |
| **CuPy + cuDF** | GPU-accelerated features | 10-1000x (if GPU available) |

### Specific Recommendations for This Pipeline

1. **Batch all column assignments** using `pd.concat` -- 60% speedup, no code restructuring needed
2. **Replace `.apply()` UDFs** with `np.where()`, `np.select()`, vectorized pandas ops
3. **Use `.to_numpy()` + Numba** for any remaining custom rolling/window features
4. **Pre-compute intermediate arrays** instead of repeated column access in loops
5. **Profile with `line_profiler`** to find the actual top-10 slowest UDFs and target those first

---

## 5. Concurrency Budget Calculator

For a given machine, the total thread budget is:

```
total_cores = physical_cores (not hyperthreads for compute-bound work)

Option A (fold-parallel, sequential LightGBM):
  num_workers = total_cores / lgbm_threads
  lgbm_threads = 2-4 (compromise)

Option B (sequential folds, parallel LightGBM):
  num_workers = 1
  lgbm_threads = total_cores

Option C (Optuna parallel trials):
  concurrent_trials x lgbm_threads <= total_cores
  e.g., 16 cores: 4 trials x 4 threads, or 8 trials x 2 threads
```

### Hardware-Specific Recommendations

| Machine | Cores | CPCV Folds | Optuna Trials | LightGBM Threads |
|---------|-------|------------|---------------|-----------------|
| Local 13900K | 24P | 6 workers | 6 trials | 4 threads each |
| 64-core cloud | 64 | 8 workers | 16 trials | 4-8 threads each |
| 128-core cloud | 128 | 10 workers | 32 trials | 4 threads each |
| H200 (weak CPU) | 16 | 4 workers | 4 trials | 4 threads each |

**Always benchmark**: the optimal split depends on memory bandwidth, cache topology, and matrix density. The formula is a starting point, not gospel.

---

## Summary: Priority Actions

1. **Immediate (no code restructuring)**: Batch column assignment with `pd.concat`, set `OMP_NUM_THREADS`, set `LOKY_PICKLER=pickle`
2. **Short-term**: Implement SharedMemory CSR buffer sharing for CPCV folds, replace top-10 `.apply()` UDFs with vectorized equivalents
3. **Medium-term**: Profile and Numba-compile remaining custom UDFs, implement memmap fallback for large TFs
4. **Future**: Evaluate Python 3.14 free-threading for thread-based CPCV/Optuna orchestration, consider Polars for feature builds

---

## Sources

### SharedMemory + Sparse IPC
- [Python multiprocessing.shared_memory docs](https://docs.python.org/3/library/multiprocessing.shared_memory.html)
- [SciPy CSR array documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html)
- [Troubleshooting Memory Errors in Python Parallel Processing](https://layer6.ai/troubleshooting-memory-errors-in-python-parallel-processing/)
- [How to Use Python's multiprocessing for CPU-Intensive Tasks](https://oneuptime.com/blog/post/2025-01-06-python-multiprocessing-cpu-tasks/view)
- [LightGBM Dataset intent](https://stackoverflow.com/questions/65924856/lightgbm-intent-of-lightgbm-dataset)
- [LightGBM Distributed Learning Guide](https://lightgbm.readthedocs.io/en/latest/Parallel-Learning-Guide.html)

### Free-Threading Python
- [Python 3.14 free-threading docs](https://docs.python.org/3/howto/free-threading-python.html)
- [LightGBM FAQ (OpenMP + fork warning)](https://lightgbm.readthedocs.io/en/latest/FAQ.html)
- [PEP 779: Criteria for supported status for free-threaded Python](https://peps.python.org/pep-0779/)
- [Free-Threading Python vs Multiprocessing (Reddit)](https://www.reddit.com/r/Python/comments/1pw2hve/freethreading_python_vs_multiprocessing_overhead/)
- [LightGBM hangs with OpenMP + forking](https://github.com/microsoft/LightGBM/issues/4751)

### Pickle Alternatives
- [Joblib parallel documentation](https://joblib.readthedocs.io/en/latest/parallel.html)
- [Joblib SharedMemory + Pickle 5 IPC proposal](https://github.com/joblib/joblib/issues/1094)
- [Using large arrays with multiprocessing](https://e-dorigatti.github.io/python/2020/06/19/multiprocessing-large-objects.html)
- [SciPy int64 CSR indices RFC](https://github.com/scipy/scipy/issues/16774)

### Vectorized Feature Engineering
- [pandas enhancing performance docs](https://pandas.pydata.org/docs/user_guide/enhancingperf.html)
- [Efficient Pandas: Apply vs Vectorized Operations](https://towardsdatascience.com/efficient-pandas-apply-vs-vectorized-operations-91ca17669e84/)
- [Pandas vectorization: faster code, slower code, bloated memory](https://pythonspeed.com/articles/pandas-vectorization/)
- [5 DuckDB UDF Tricks That Outrun Pandas apply](https://medium.com/@sparknp1/5-duckdb-udf-tricks-that-outrun-pandas-apply-68a43e6d3b93)
- [Top 5 tips to make your pandas code absurdly fast](https://tryolabs.com/blog/2023/02/08/top-5-tips-to-make-your-pandas-code-absurdly-fast)
