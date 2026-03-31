"""Benchmark 1 trial: CPU vs GPU cuda_sparse on real data."""
import time, os, sys, numpy as np, scipy.sparse as sp
sys.path.insert(0, "/workspace/v3.3")
import lightgbm as lgb

TF = os.environ.get("BENCH_TF", "4h")
USE_GPU = os.environ.get("BENCH_GPU", "0") == "1"
GPU_ID = int(os.environ.get("BENCH_GPU_ID", "0"))

mode = "GPU cuda_sparse (GPU %d)" % GPU_ID if USE_GPU else "CPU"
print("=== BENCHMARK: %s | %s ===" % (TF, mode))

# Step 1: Load data
t0 = time.time()
import pandas as pd
df = pd.read_parquet("/workspace/v3.3/features_BTC_%s.parquet" % TF)
X_base = df.select_dtypes(include=[np.number]).values.astype(np.float32)
crosses = sp.load_npz("/workspace/v3.3/v2_crosses_BTC_%s.npz" % TF)

from feature_library import compute_triple_barrier_labels
y = compute_triple_barrier_labels(df, TF)

X_base_sp = sp.csr_matrix(X_base)
X_all = sp.hstack([X_base_sp, crosses], format="csr").astype(np.float32)
valid_mask = ~np.isnan(y)
X_valid = X_all[valid_mask]
y_valid = y[valid_mask].astype(int)

t1 = time.time()
print("[1] Data load: %.1fs | Shape: %s, NNZ: %s" % (t1-t0, X_valid.shape, "{:,}".format(X_valid.nnz)))

# Step 2: Build Dataset (parallel if available)
t2 = time.time()
try:
    from run_optuna_local import _parallel_dataset_construct
    ds = _parallel_dataset_construct(X_valid, y_valid)
    method = "PARALLEL"
except Exception as e:
    print("[2] Parallel build failed (%s), falling back to single-thread" % e)
    ds = lgb.Dataset(X_valid, label=y_valid, params={"feature_pre_filter": False, "max_bin": 255})
    ds.construct()
    method = "SINGLE-THREAD"
t3 = time.time()
print("[2] Dataset build (%s): %.1fs" % (method, t3-t2))

# Step 3: Create fold (80/20 split)
n = len(y_valid)
train_idx = list(range(int(n*0.8)))
val_idx = list(range(int(n*0.8), n))
dtrain = ds.subset(train_idx)
dval = ds.subset(val_idx)
t4 = time.time()
print("[3] Fold subset: %.1fs" % (t4-t3))

# Step 4+5: Train
params = {
    "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
    "num_leaves": 31, "min_data_in_leaf": 10, "feature_fraction": 0.9,
    "feature_fraction_bynode": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l1": 2.0, "lambda_l2": 10.0, "min_gain_to_split": 3.0,
    "max_depth": 6, "learning_rate": 0.15, "max_bin": 255,
    "is_enable_sparse": True, "feature_pre_filter": False,
    "verbosity": 1,
}

if USE_GPU:
    gpu_params = params.copy()
    gpu_params["device_type"] = "cuda_sparse"
    gpu_params["gpu_device_id"] = GPU_ID
    gpu_params["histogram_pool_size"] = 1024
    for k in ["force_col_wise", "force_row_wise", "device"]:
        gpu_params.pop(k, None)

    _gpu_data = X_valid[train_idx]
    dtrain.construct()
    dval.construct()
    booster = lgb.Booster(gpu_params, dtrain)
    booster.add_valid(dval, "val")
    booster.set_external_csr(_gpu_data)

    t5 = time.time()
    print("[4] GPU init + CSR upload: %.1fs" % (t5-t4))

    best_score = float("inf")
    best_iter = 0
    for rnd in range(60):
        booster.update()
        val_result = booster.eval_valid()[0]
        val_score = val_result[2]
        if val_score < best_score:
            best_score = val_score
            best_iter = rnd
    t6 = time.time()
    print("[5] Training 60 rounds: %.1fs (%.2fs/round)" % (t6-t5, (t6-t5)/60))
    print("    Best val mlogloss: %.6f at round %d" % (best_score, best_iter))
else:
    params["num_threads"] = 0
    params["force_col_wise"] = True

    t5 = time.time()
    print("[4] CPU setup: %.1fs" % (t5-t4))
    booster = lgb.train(params, dtrain, num_boost_round=60,
                        valid_sets=[dval], valid_names=["val"],
                        callbacks=[lgb.early_stopping(15), lgb.log_evaluation(10)])
    t6 = time.time()
    print("[5] Training (CPU): %.1fs (%.2fs/round)" % (t6-t5, (t6-t5)/60))
    print("    Best val mlogloss: %.6f" % booster.best_score["val"]["multi_logloss"])

print("\n=== TOTAL: %.1fs ===" % (t6-t0))
