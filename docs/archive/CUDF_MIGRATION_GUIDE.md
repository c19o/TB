# Pandas -> cuDF Migration Guide (Practical)

Research compiled 2026-03-20 from Perplexity searches against current RAPIDS docs.

---

## 1. Drop-In Replacement: `import cudf as pd` vs `cudf.pandas`

**DO NOT use `import cudf as pd`.** It gives you the cuDF API directly with no fallback -- anything pandas-specific will just crash.

**USE `cudf.pandas` instead.** This is NVIDIA's official drop-in accelerator:
```python
# At the TOP of your script, before any other imports:
import cudf.pandas
cudf.pandas.install()

import pandas as pd  # now GPU-accelerated with CPU fallback
```

### What works
- `cudf.pandas` passes **93% of pandas' 187,000+ unit tests**
- Tested with NumPy, scikit-learn, XGBoost, PyTorch, Matplotlib
- Unsupported ops automatically fall back to CPU pandas (transparent)

### What breaks
| Issue | Details |
|-------|---------|
| **Row ordering** | Joins/merges can produce non-deterministic row order |
| **Null handling** | cuDF uses `NA` not `NaN`, preserves int dtype instead of promoting to float |
| **Unsupported kwargs** | Many methods have partial parameter support: `axis`, `level`, `numeric_only`, `kind`, `inplace`, `freq`, `regex` may not work depending on method |
| **groupby.apply** | Index layout can differ from pandas |
| **Proxy objects** | `cudf.pandas` uses proxy DataFrames -- C-extensions or NumPy internals may trigger eager device-to-host copies |
| **Joblib** | Currently marked unsupported |
| **pandas version** | Requires pandas 2.x. cuDF 24.02 was the last release supporting pandas 1.5.x |
| **query()** | Only supports numeric, datetime, timedelta, or bool dtypes |
| **diff()** | Numeric columns only |

### Migration strategy
1. Start with `cudf.pandas` (zero code changes)
2. Profile with `cudf.pandas` profiler to find CPU fallback hotspots
3. Rewrite only the hotspot functions to native cuDF API
4. Test row ordering on all joins/merges

---

## 2. Rolling Window Custom Functions

cuDF **does** support `Rolling.apply(func)` but with strict limitations.

### What works
```python
import cudf
import math

def my_udf(window):
    acc = 0.0
    for x in window:
        acc = max(acc, math.sqrt(x))
    return acc

s = cudf.Series([16, 25, 36, 49, 64], dtype="float64")
out = s.rolling(window=3, min_periods=3).apply(my_udf)
```

### Constraints
- Function takes a **1D numeric window array** only
- **No null values** in the window
- **No `args` or `kwargs`** parameter support
- **No external state** -- global variables become compile-time constants (Numba JIT)
- Only `math` module functions + basic Python ops (loops, conditionals, arithmetic)
- No sorting, no complex library calls, no dynamic parameters

### When `.rolling().apply()` won't work -- alternatives

**Option A: Rewrite as built-in rolling ops**
```python
# Instead of .rolling(w).apply(custom_std), use:
df['col'].rolling(w).std()
df['col'].rolling(w).mean()
df['col'].rolling(w).min()
df['col'].rolling(w).max()
df['col'].rolling(w).sum()
```

**Option B: Custom CUDA kernel (escape hatch)**
```python
from numba import cuda
import cupy as cp

@cuda.jit
def rolling_custom_kernel(input_arr, output_arr, window_size):
    i = cuda.grid(1)
    if i >= window_size - 1 and i < len(input_arr):
        # Custom logic over input_arr[i-window_size+1:i+1]
        val = 0.0
        for j in range(window_size):
            val += input_arr[i - window_size + 1 + j]
        output_arr[i] = val

# Convert cuDF Series -> CuPy array -> run kernel -> back to cuDF
arr_in = cp.asarray(series.values)
arr_out = cp.zeros_like(arr_in)
threads = 256
blocks = (len(arr_in) + threads - 1) // threads
rolling_custom_kernel[blocks, threads](arr_in, arr_out, window_size)
result = cudf.Series(arr_out)
```

**Option C: Keep on CPU for complex rolling** -- Use `cudf.pandas` and let complex rolling ops fall back to CPU automatically.

---

## 3. Timezone / Datetime Handling

### Core rules
| Operation | Use when | Example |
|-----------|----------|---------|
| `tz_localize(tz)` | Naive datetime -> add timezone | `s.dt.tz_localize("UTC")` |
| `tz_convert(tz)` | Aware datetime -> change timezone | `s.dt.tz_convert("US/Eastern")` |
| `to_datetime(..., utc=True)` | Parse strings directly as UTC | `cudf.to_datetime(s, utc=True)` |
| `tz_convert(None)` | Remove timezone (converts to UTC first) | `s.dt.tz_convert(None)` |

### Standard pattern
```python
import cudf

# Parse naive strings
s = cudf.to_datetime(cudf.Series(["2024-01-01 09:00:00"]))

# Interpret as New York local time
s_ny = s.dt.tz_localize("America/New_York")

# Convert to UTC
s_utc = s_ny.dt.tz_convert("UTC")

# Strip timezone (result is naive UTC)
s_naive = s_utc.dt.tz_convert(None)
```

### Gotchas
- **DST handling is limited**: `ambiguous` and `nonexistent` params only support `'NaT'` -- ambiguous/nonexistent timestamps become NaT
- **`tz_convert(None)` converts to UTC first**, then strips. You get naive UTC, NOT naive local time
- **Best practice for trading data**: Store everything as naive UTC (no timezone). Localize only for display. This avoids all DST edge cases.

---

## 4. Dask XGBoost Walk-Forward Cross Validation

There is **no built-in walk-forward CV** in Dask XGBoost. You must orchestrate folds manually.

### Complete pattern
```python
import xgboost.dask as dxgb
from dask.distributed import Client

client = Client()  # or connect to existing cluster

# Define walk-forward folds (rolling or expanding window)
def make_walk_forward_folds(df, n_folds=5, train_size=None):
    """Generate chronological train/valid index pairs."""
    total = len(df)
    fold_size = total // (n_folds + 1)
    folds = []
    for i in range(n_folds):
        if train_size:  # Rolling window (fixed train size)
            train_start = max(0, (i + 1) * fold_size - train_size)
        else:  # Expanding window
            train_start = 0
        train_end = (i + 1) * fold_size
        valid_start = train_end
        valid_end = valid_start + fold_size
        folds.append((
            (train_start, train_end),
            (valid_start, valid_end)
        ))
    return folds

# Run walk-forward CV
params = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "device": "cuda",
    "eval_metric": "rmse",
    "max_depth": 6,
    "learning_rate": 0.05,
}

fold_metrics = []
for (tr_start, tr_end), (va_start, va_end) in folds:
    X_train = X.iloc[tr_start:tr_end]
    y_train = y.iloc[tr_start:tr_end]
    X_valid = X.iloc[va_start:va_end]
    y_valid = y.iloc[va_start:va_end]

    # Build FRESH DaskQuantileDMatrix per fold
    dtrain = dxgb.DaskQuantileDMatrix(client, X_train, y_train)
    dvalid = dxgb.DaskQuantileDMatrix(client, X_valid, y_valid)

    result = dxgb.train(
        client,
        params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=50,
    )

    booster = result["booster"]
    best_iter = booster.best_iteration
    best_score = result["history"]["valid"]["rmse"][best_iter]
    fold_metrics.append({
        "fold_train": (tr_start, tr_end),
        "fold_valid": (va_start, va_end),
        "best_iteration": best_iter,
        "best_rmse": best_score,
    })

avg_rmse = sum(f["best_rmse"] for f in fold_metrics) / len(fold_metrics)
```

### Key rules
- **Never shuffle** -- breaks temporal structure, leaks future data
- **Build fresh `DaskQuantileDMatrix` per fold** -- do NOT try to index into a global matrix
- `DaskQuantileDMatrix` saves memory vs `DaskDMatrix` (use histogram method)
- DMatrix construction triggers lazy Dask computation -- precompute upstream transforms first
- Early stopping works natively with `evals` parameter

---

## 5. Bulk Column Multiplication (150K+ operations)

### Fastest approaches (ranked)

**1. `df.eval()` with multi-line expressions (BEST for bulk)**
```python
# Build expression string programmatically
lines = []
for i in range(num_pairs):
    lines.append(f"product_{i} = col_a_{i} * col_b_{i}")

# Execute in chunks (eval has expression length limits)
CHUNK = 50  # expressions per eval call
for start in range(0, len(lines), CHUNK):
    chunk_expr = "\n".join(lines[start:start+CHUNK])
    df.eval(chunk_expr, inplace=True)
```

**2. Direct vectorized `*` operator**
```python
for a_col, b_col, out_col in column_pairs:
    df[out_col] = df[a_col] * df[b_col]
```
Simple but has Python-side orchestration overhead per column.

**3. CuPy matrix multiply (best for truly massive bulk)**
```python
import cupy as cp

# Extract all source columns as CuPy 2D array
a_cols = cp.asarray(df[a_column_list].values)  # shape: (rows, 150000)
b_cols = cp.asarray(df[b_column_list].values)  # shape: (rows, 150000)

# Element-wise multiply all at once
products = a_cols * b_cols  # single GPU kernel launch

# Write back
for i, name in enumerate(output_names):
    df[name] = cudf.Series(products[:, i])
```

**4. `pylibcudf.binaryop` (lowest level)**
```python
import pylibcudf
# For library/pipeline builders, not typical user code
```

### Performance rules
- **Pre-cast dtypes** to float32/float64 before operations -- eval() won't auto-cast
- **Minimize kernel launches** -- batch operations rather than 150K individual calls
- **Memory bandwidth is the bottleneck** for element-wise ops, not compute
- **Avoid `apply_rows`** and row-oriented UDFs for simple arithmetic

---

## Quick Decision Matrix

| Situation | Recommended Approach |
|-----------|---------------------|
| First migration attempt | `cudf.pandas.install()` -- zero code changes |
| Simple rolling (mean/std/sum) | Native cuDF rolling methods |
| Complex rolling UDF | Numba CUDA kernel or let cudf.pandas fall back |
| Timezone handling | Store as naive UTC, localize only for display |
| Walk-forward CV with XGBoost | Manual fold loop + per-fold DaskQuantileDMatrix |
| Bulk column math (150K ops) | CuPy array extraction -> element-wise -> write back |
| Debugging GPU vs CPU | Use `cudf.pandas` profiler to find fallback hotspots |
