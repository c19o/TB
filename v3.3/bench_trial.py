#!/usr/bin/env python
import os
import sys
import time

import lightgbm as lgb
import pandas as pd
import scipy.sparse as sp

from path_contract import ARTIFACT_ROOT, CODE_ROOT, artifact_path

sys.path.insert(0, CODE_ROOT)

TF = os.environ.get("TF", "1w")
df = pd.read_parquet(artifact_path(f"features_BTC_{TF}.parquet"))
crosses = sp.load_npz(artifact_path(f"v2_crosses_BTC_{TF}.npz"))

X = crosses
y = df["target"].values if "target" in df.columns else df.iloc[:, -1].values

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "device_type": "cpu",
    "max_bin": 7,
    "feature_pre_filter": False,
}

dtrain = lgb.Dataset(X, label=y, free_raw_data=False)

t0 = time.time()
bst = lgb.train(params, dtrain, num_boost_round=50)
print(f"train_seconds={time.time() - t0:.2f}")
print(f"artifact_root={ARTIFACT_ROOT}")
print(bst.current_iteration())
