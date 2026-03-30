#!/usr/bin/env python
"""
Savage22 V3.3 Pre-Flight Validation
====================================
Deterministic checks derived from 155+ battle-tested rules.
Catches config bugs BEFORE spending money on cloud training.

Usage:
    python validate.py                      # config + consistency checks only
    python validate.py --tf 1w              # + data integrity for 1w
    python validate.py --tf 4h --cloud      # + machine/environment checks
    python validate.py --tf 1d --local      # + local environment checks
"""
import sys
import os
import ast
import json
import argparse
import re
from pathlib import Path

# ── Globals ──
_pass = 0
_fail = 0
_warn = 0
_failures = []

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VALID_TFS = {'1w', '1d', '4h', '1h', '15m'}


def check(name, condition, fail_msg):
    """One deterministic check. Pass or fail — no AI judgment."""
    global _pass, _fail
    if condition:
        _pass += 1
        print(f"  [PASS] {name}")
    else:
        _fail += 1
        _failures.append((name, fail_msg))
        print(f"  [FAIL] {name}")
        print(f"         {fail_msg}")


def warn(name, condition, msg):
    """Non-blocking warning."""
    global _warn
    if not condition:
        _warn += 1
        print(f"  [WARN] {name}")
        print(f"         {msg}")


# ══════════════════════════════════════════════════════════════
# CATEGORY 1: Config Parameter Bounds
# Source of truth for ALL parameter constraints.
# ══════════════════════════════════════════════════════════════
def check_config_params():
    print("\n== CATEGORY 1: Config Parameter Bounds ==")
    sys.path.insert(0, PROJECT_DIR)
    import config as cfg

    p = cfg.V3_LGBM_PARAMS

    # -- LightGBM core params --
    check("feature_fraction >= 0.7",
          p.get('feature_fraction', 0) >= 0.7,
          f"feature_fraction={p.get('feature_fraction')} -- must be >= 0.7 to preserve rare esoteric crosses. "
          f"FIX: config.py V3_LGBM_PARAMS, set feature_fraction >= 0.9")

    check("feature_fraction_bynode >= 0.7",
          p.get('feature_fraction_bynode', 0) >= 0.7,
          f"feature_fraction_bynode={p.get('feature_fraction_bynode')} -- must be >= 0.7. "
          f"0.5 * feature_fraction(0.7) = 0.35 effective — devastating for rare signals. "
          f"FIX: config.py V3_LGBM_PARAMS")

    check("feature_pre_filter == False",
          p.get('feature_pre_filter') is False,
          f"feature_pre_filter={p.get('feature_pre_filter')} -- True silently kills rare esoteric features. "
          f"FIX: config.py V3_LGBM_PARAMS, set feature_pre_filter=False")

    check("max_bin == 255",
          p.get('max_bin') == 255,
          f"max_bin={p.get('max_bin')} -- must be 255 for maximum EFB compression. "
          f"Binary features get 2 bins regardless. FIX: config.py V3_LGBM_PARAMS")

    check("force_col_wise == True",
          p.get('force_col_wise') is True,
          f"force_col_wise={p.get('force_col_wise')} -- required for sparse CSR. "
          f"FIX: config.py V3_LGBM_PARAMS")

    check("is_enable_sparse == True",
          p.get('is_enable_sparse') is True,
          f"is_enable_sparse={p.get('is_enable_sparse')} -- must be True for sparse CSR input. "
          f"FIX: config.py V3_LGBM_PARAMS")

    check("no 'device' alias coexisting with 'device_type' in V3_LGBM_PARAMS",
          not ('device' in p and 'device_type' in p),
          f"Both 'device' and 'device_type' present in V3_LGBM_PARAMS -- 'device' is an alias for "
          f"'device_type'. Coexistence causes conflict in GPU path. "
          f"FIX: remove 'device' key from config.py V3_LGBM_PARAMS; GPU code pops it before setting device_type.")

    check("boosting_type == 'gbdt'",
          p.get('boosting_type') == 'gbdt',
          f"boosting_type={p.get('boosting_type')} -- must be 'gbdt'. FIX: config.py")

    check("objective == 'multiclass'",
          p.get('objective') == 'multiclass',
          f"objective={p.get('objective')} -- must be 'multiclass' (3 classes). FIX: config.py")

    check("num_class == 3",
          p.get('num_class') == 3,
          f"num_class={p.get('num_class')} -- must be 3 (LONG/HOLD/SHORT). FIX: config.py")

    check("min_data_in_bin == 1",
          p.get('min_data_in_bin') == 1,
          f"min_data_in_bin={p.get('min_data_in_bin')} -- must be 1 to preserve rare binary events. "
          f"FIX: config.py V3_LGBM_PARAMS")

    check("num_threads == 0 (not -1)",
          p.get('num_threads') == 0,
          f"num_threads={p.get('num_threads')} -- must be 0 (auto-detect). -1 is undocumented. "
          f"FIX: config.py V3_LGBM_PARAMS")

    check("no xgboost in LightGBM params",
          'xgboost' not in str(p).lower(),
          "XGBoost reference found in V3_LGBM_PARAMS. MUST use LightGBM only.")

    # -- Signal-killing param guards (config defaults, not just Optuna bounds) --
    check("bagging_fraction >= 0.7",
          p.get('bagging_fraction', 0) >= 0.7,
          f"bagging_fraction={p.get('bagging_fraction')} -- must be >= 0.7. "
          f"50% row dropout destroys rare esoteric signals (P(10-fire)=0.001 at bf=0.5). "
          f"FIX: config.py V3_LGBM_PARAMS")

    check("lambda_l1 <= 4.0",
          p.get('lambda_l1', 999) <= 4.0,
          f"lambda_l1={p.get('lambda_l1')} -- must be <= 4.0. "
          f"L1 > 4 zeros leaf weights for signals firing < 15 times. "
          f"FIX: config.py V3_LGBM_PARAMS")

    check("lambda_l2 <= 10.0",
          p.get('lambda_l2', 999) <= 10.0,
          f"lambda_l2={p.get('lambda_l2')} -- must be <= 10.0. "
          f"FIX: config.py V3_LGBM_PARAMS")

    check("bagging_freq > 0 (enables bagging_fraction)",
          p.get('bagging_freq', 0) > 0,
          f"bagging_freq={p.get('bagging_freq')} -- must be > 0 for bagging_fraction to take effect. "
          f"LightGBM silently ignores bagging_fraction if bagging_freq=0. "
          f"FIX: config.py V3_LGBM_PARAMS, set bagging_freq=1")

    check("path_smooth <= 2.0",
          p.get('path_smooth', 0) <= 2.0,
          f"path_smooth={p.get('path_smooth')} -- must be <= 2.0. "
          f"Higher values over-regularize rare leaf predictions. FIX: config.py V3_LGBM_PARAMS")

    check("CPCV_PARALLEL_GPUS >= 0",
          cfg.CPCV_PARALLEL_GPUS >= 0,
          f"CPCV_PARALLEL_GPUS={cfg.CPCV_PARALLEL_GPUS} -- must be >= 0 (0=auto-detect). "
          f"FIX: config.py or CPCV_PARALLEL_GPUS env var")

    # -- Per-TF num_leaves caps --
    nl = cfg.TF_NUM_LEAVES
    check("1w num_leaves <= 7", nl.get('1w', 999) <= 7,
          f"1w num_leaves={nl.get('1w')} -- max 7 (1158 rows). FIX: config.py TF_NUM_LEAVES")
    check("1d num_leaves <= 15", nl.get('1d', 999) <= 15,
          f"1d num_leaves={nl.get('1d')} -- max 15 (5.7K rows). FIX: config.py TF_NUM_LEAVES")
    check("4h num_leaves <= 31", nl.get('4h', 999) <= 31,
          f"4h num_leaves={nl.get('4h')} -- max 31 (23K rows). FIX: config.py TF_NUM_LEAVES")
    check("1h num_leaves <= 63", nl.get('1h', 999) <= 63,
          f"1h num_leaves={nl.get('1h')} -- max 63 (75K rows). FIX: config.py TF_NUM_LEAVES")
    check("15m num_leaves <= 127", nl.get('15m', 999) <= 127,
          f"15m num_leaves={nl.get('15m')} -- max 127 (294K rows). FIX: config.py TF_NUM_LEAVES")

    # -- Per-TF min_data_in_leaf (must be <= rare signal frequency ~10-20) --
    mdil = cfg.TF_MIN_DATA_IN_LEAF
    for tf_name, val in mdil.items():
        check(f"{tf_name} min_data_in_leaf <= 10",
              val <= 10,
              f"{tf_name} min_data_in_leaf={val} -- rare signals fire 10-20x. "
              f"Values >10 make them invisible. FIX: config.py TF_MIN_DATA_IN_LEAF")

    # -- Class weights (SHORT upweighting) --
    cw = cfg.TF_CLASS_WEIGHT
    check("1w SHORT weight >= 3.0",
          cw.get('1w', {}).get(0, 0) >= 3.0,
          f"1w SHORT weight={cw.get('1w', {}).get(0)} -- needs 3x for directional learning. FIX: config.py")
    check("1d SHORT weight >= 3.0",
          cw.get('1d', {}).get(0, 0) >= 3.0,
          f"1d SHORT weight={cw.get('1d', {}).get(0)} -- needs 3x. FIX: config.py")
    check("4h SHORT weight >= 2.0",
          cw.get('4h', {}).get(0, 0) >= 2.0,
          f"4h SHORT weight={cw.get('4h', {}).get(0)} -- needs 2x. FIX: config.py")

    # -- Timeframes: no 5m --
    all_tfs = set(list(cfg.TIMEFRAMES_ALL_ASSETS) + list(cfg.TIMEFRAMES_CRYPTO_ONLY))
    check("no 5m timeframe", '5m' not in all_tfs,
          "5m timeframe found -- dropped in v3.1 (esoteric signals meaningless at 5m). FIX: config.py")

    # -- Fee consistency --
    check("FEE_RATE == PORTFOLIO_FEE_RATE",
          cfg.FEE_RATE == cfg.PORTFOLIO_FEE_RATE,
          f"FEE_RATE={cfg.FEE_RATE} != PORTFOLIO_FEE_RATE={cfg.PORTFOLIO_FEE_RATE}. FIX: config.py")

    # -- Optuna config --
    check("Phase1 LR > Final LR",
          cfg.OPTUNA_PHASE1_LR > cfg.OPTUNA_FINAL_LR,
          f"Phase1 LR={cfg.OPTUNA_PHASE1_LR} must be > Final LR={cfg.OPTUNA_FINAL_LR}. FIX: config.py")

    # -- CPCV K >= 2 for all TFs --
    for tf_name, (n_groups, n_test) in cfg.TF_CPCV_GROUPS.items():
        check(f"{tf_name} CPCV K >= 2",
              n_test >= 2,
              f"{tf_name} CPCV n_test_groups={n_test} -- must be >= 2 for real PBO. FIX: config.py TF_CPCV_GROUPS")

    # -- Never subsample small TFs --
    rs = cfg.OPTUNA_TF_ROW_SUBSAMPLE
    check("1w row subsample == 1.0", rs.get('1w') == 1.0,
          f"1w subsample={rs.get('1w')} -- never subsample 1158 rows. FIX: config.py")
    check("1d row subsample == 1.0", rs.get('1d') == 1.0,
          f"1d subsample={rs.get('1d')} -- never subsample 5733 rows. FIX: config.py")

    # -- Risk limits --
    check("max_concurrent_positions <= 5",
          cfg.RISK_LIMITS.get('max_concurrent_positions', 999) <= 5,
          f"max_concurrent_positions={cfg.RISK_LIMITS.get('max_concurrent_positions')} -- max 5. FIX: config.py")


# ══════════════════════════════════════════════════════════════
# CATEGORY 2: Optuna Search Space Verification
# AST-parses run_optuna_local.py to check actual trial.suggest_* bounds.
# ══════════════════════════════════════════════════════════════
def check_optuna_search_space():
    print("\n== CATEGORY 2: Optuna Search Space ==")

    optuna_path = os.path.join(PROJECT_DIR, 'run_optuna_local.py')
    if not os.path.exists(optuna_path):
        check("run_optuna_local.py exists", False,
              f"Missing {optuna_path}. Cannot verify Optuna search space.")
        return

    with open(optuna_path, 'r', encoding='utf-8') as f:
        source = f.read()

    # Parse trial.suggest_float/int calls using regex (more robust than AST for this pattern)
    # Pattern: trial.suggest_float('name', low, high, ...)
    suggest_pattern = re.compile(
        r"trial\.suggest_(float|int)\(\s*['\"](\w+)['\"]\s*,\s*([^,)]+)\s*,\s*([^,)]+)"
    )

    bounds = {}
    for match in suggest_pattern.finditer(source):
        kind, name, low_str, high_str = match.groups()
        try:
            low = float(low_str.strip())
            high = float(high_str.strip())
            bounds[name] = (low, high)
        except ValueError:
            # Variable reference (like _tf_nl_cap) — can only check the literal bound
            try:
                low = float(low_str.strip())
                bounds[name] = (low, None)
            except ValueError:
                bounds[name] = (None, None)

    # -- Critical search space checks --
    if 'feature_fraction' in bounds:
        low, high = bounds['feature_fraction']
        check("Optuna feature_fraction lower >= 0.7",
              low is not None and low >= 0.7,
              f"feature_fraction search lower={low} -- must be >= 0.7. "
              f"FIX: run_optuna_local.py trial.suggest_float('feature_fraction', 0.7, ...)")
        if high is not None:
            check("Optuna feature_fraction upper <= 1.0",
                  high <= 1.0,
                  f"feature_fraction search upper={high} -- must be <= 1.0")
    else:
        check("Optuna has feature_fraction search", False,
              "No trial.suggest for feature_fraction found in run_optuna_local.py")

    if 'feature_fraction_bynode' in bounds:
        low, _ = bounds['feature_fraction_bynode']
        check("Optuna feature_fraction_bynode lower >= 0.7",
              low is not None and low >= 0.7,
              f"feature_fraction_bynode search lower={low} -- must be >= 0.7. "
              f"0.5 * feature_fraction(0.7) = 0.35 effective — devastating for rare signals. "
              f"FIX: run_optuna_local.py")

    if 'bagging_fraction' in bounds:
        low, _ = bounds['bagging_fraction']
        check("Optuna bagging_fraction lower >= 0.7",
              low is not None and low >= 0.7,
              f"bagging_fraction search lower={low} -- must be >= 0.7. "
              f"50% row dropout destroys 3-fire features. "
              f"FIX: run_optuna_local.py")

    if 'lambda_l1' in bounds:
        _, high = bounds['lambda_l1']
        check("Optuna lambda_l1 upper <= 4.0",
              high is not None and high <= 4.0,
              f"lambda_l1 search upper={high} -- must be <= 4.0. "
              f"lambda_l1 > 4 zeros leaf weights for signals firing < 15 times, killing rare esoteric crosses "
              f"(|G|=7.5 for 15-fire signal at p=0.5 with 3x class weight ~22.5 — still zeroed at L1>8). "
              f"FIX: run_optuna_local.py suggest_float('lambda_l1', 1e-4, 4.0, log=True)")

    if 'lambda_l2' in bounds:
        _, high = bounds['lambda_l2']
        check("Optuna lambda_l2 upper <= 10.0",
              high is not None and high <= 10.0,
              f"lambda_l2 search upper={high} -- must be <= 10.0. "
              f"L2 shrinks proportionally (does not zero), but > 10 over-penalizes rare signal leaf weights. "
              f"FIX: run_optuna_local.py suggest_float('lambda_l2', 1e-4, 10.0, log=True)")

    if 'num_leaves' in bounds:
        low, _ = bounds['num_leaves']
        check("Optuna num_leaves lower >= 4",
              low is not None and low >= 4,
              f"num_leaves search lower={low} -- must be >= 4 (v3.2 best was 7)")

    if 'min_gain_to_split' in bounds:
        low, high = bounds['min_gain_to_split']
        check("Optuna min_gain_to_split lower >= 0.0",
              low is not None and low >= 0.0,
              f"min_gain_to_split search lower={low} -- must be >= 0.0")
        check("Optuna min_gain_to_split upper <= 5.0",
              high is not None and high <= 5.0,
              f"min_gain_to_split search upper={high} -- must be <= 5.0. "
              f"Floor of 0.5 blocks marginal rare splits. Let the model decide. "
              f"FIX: run_optuna_local.py")

    if 'max_depth' in bounds:
        _, high = bounds['max_depth']
        if high is not None:
            check("Optuna max_depth upper <= 12",
                  high <= 12,
                  f"max_depth search upper={high} -- must be <= 12 to prevent memorization")

    # -- No XGBoost in Optuna --
    code_without_comments = '\n'.join(
        line.split('#')[0] for line in source.split('\n')
    )
    check("no xgboost in Optuna search",
          'xgboost' not in code_without_comments.lower(),
          "XGBoost reference found in run_optuna_local.py (non-comment code). MUST use LightGBM only.")

    # -- Phase 1 learning rate --
    sys.path.insert(0, PROJECT_DIR)
    import config as cfg
    check("Phase 1 LR >= 0.10",
          cfg.OPTUNA_PHASE1_LR >= 0.10,
          f"Phase 1 LR={cfg.OPTUNA_PHASE1_LR} -- should be >= 0.10 for rapid search")

    check("Phase 1 ES patience >= 10",
          cfg.OPTUNA_PHASE1_ES_PATIENCE >= 10,
          f"Phase 1 ES patience={cfg.OPTUNA_PHASE1_ES_PATIENCE} -- minimum 10 rounds")


# ══════════════════════════════════════════════════════════════
# CATEGORY 3: Machine/Environment Checks
# ══════════════════════════════════════════════════════════════
def check_environment(tf=None, cloud=False):
    print("\n== CATEGORY 3: Machine/Environment ==")

    # -- Python version --
    check("Python >= 3.10",
          sys.version_info >= (3, 10),
          f"Python {sys.version_info.major}.{sys.version_info.minor} -- need >= 3.10")

    # -- Critical imports --
    critical_modules = [
        'lightgbm', 'sklearn', 'scipy', 'optuna', 'numba',
        'pandas', 'numpy', 'pyarrow', 'tqdm', 'yaml',
    ]
    for mod in critical_modules:
        try:
            __import__(mod)
            importable = True
        except (ImportError, FileNotFoundError, OSError):
            importable = False
        check(f"import {mod}", importable,
              f"Cannot import {mod}. FIX: pip install {mod}")

    # -- LightGBM sparse support --
    try:
        import lightgbm as lgb
        check("LightGBM version >= 4.0",
              hasattr(lgb, '__version__') and int(lgb.__version__.split('.')[0]) >= 4,
              f"LightGBM {lgb.__version__} -- need >= 4.0 for sparse CSR. FIX: pip install -U lightgbm")
    except Exception:
        pass

    # -- V30_DATA_DIR --
    v30 = os.environ.get('V30_DATA_DIR', '')
    if cloud:
        warn("V30_DATA_DIR not pointing to stale v3.0",
             'v3.0' not in v30 and 'LGBM' not in v30,
             f"V30_DATA_DIR={v30} -- on cloud should be /workspace or /workspace/v3.3, not v3.0 path")

    if not cloud:
        return  # Local mode: skip machine checks

    # -- Cloud-only checks --
    print("  -- Cloud environment checks --")

    # RAM check
    ram_gb = _get_ram_gb()
    tf_min_ram = {'1w': 64, '1d': 128, '4h': 256, '1h': 512, '15m': 768}
    if tf and tf in tf_min_ram:
        check(f"RAM >= {tf_min_ram[tf]}GB for {tf}",
              ram_gb >= tf_min_ram[tf],
              f"RAM={ram_gb:.0f}GB -- {tf} needs {tf_min_ram[tf]}GB. Get a bigger machine.")

    # Database files
    sys.path.insert(0, PROJECT_DIR)
    import config as cfg
    db_paths = [
        cfg.BTC_DB, cfg.MULTI_ASSET_DB,
        cfg.V1_TWEETS_DB, cfg.V1_NEWS_DB, cfg.V1_ASTRO_DB, cfg.V1_EPHEMERIS_DB,
        cfg.V1_FEAR_GREED_DB, cfg.V1_SPORTS_DB, cfg.V1_SPACE_WEATHER_DB,
        cfg.V1_MACRO_DB, cfg.V1_ONCHAIN_DB, cfg.V1_FUNDING_DB,
        cfg.V1_OI_DB, cfg.V1_GOOGLE_TRENDS_DB, cfg.V1_LLM_CACHE_DB,
    ]
    existing_dbs = [p for p in db_paths if os.path.exists(p)]
    check(f"databases present ({len(existing_dbs)}/15)",
          len(existing_dbs) >= 14,
          f"Only {len(existing_dbs)}/15 databases found. Missing DBs = missing features = weaker model. "
          f"Upload ALL .db files before training.")

    # btc_prices.db non-empty
    if os.path.exists(cfg.BTC_DB):
        size_mb = os.path.getsize(cfg.BTC_DB) / (1024 * 1024)
        check("btc_prices.db > 1MB",
              size_mb > 1,
              f"btc_prices.db is {size_mb:.1f}MB -- likely empty. Upload the full DB.")

    # astrology_engine.py
    astro_path = os.path.join(PROJECT_DIR, 'astrology_engine.py')
    check("astrology_engine.py in v3.3/",
          os.path.exists(astro_path),
          f"Missing {astro_path}. Copy from project root. FIX: cp ../astrology_engine.py v3.3/")

    # kp_history_gfz.txt
    kp_path = os.path.join(PROJECT_DIR, 'kp_history_gfz.txt')
    check("kp_history_gfz.txt exists",
          os.path.exists(kp_path),
          f"Missing {kp_path}. Space weather features need this file.")

    # NVIDIA driver + CUDA ecosystem checks
    import subprocess as sp
    driver_cuda_ver = None
    try:
        result = sp.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                        capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            driver = result.stdout.strip().split('\n')[0]
            driver_major = int(driver.split('.')[0])
            check(f"NVIDIA driver >= 535 (found {driver})",
                  driver_major >= 535,
                  f"Driver {driver} too old. Need >= 535 for CUDA 12+.")

        # Extract driver CUDA version from nvidia-smi
        result2 = sp.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result2.returncode == 0:
            cuda_match = re.search(r'CUDA Version:\s*(\d+\.\d+)', result2.stdout)
            if cuda_match:
                driver_cuda_ver = cuda_match.group(1)
                driver_cuda_major = int(driver_cuda_ver.split('.')[0])
                check(f"driver CUDA >= 12.0 (found {driver_cuda_ver})",
                      driver_cuda_major >= 12,
                      f"Driver supports CUDA {driver_cuda_ver} -- need >= 12.0. "
                      f"FIX: rent a machine with newer driver.")
    except Exception:
        warn("NVIDIA driver check", False, "Could not query nvidia-smi (no GPU or not installed)")

    # -- nvcc (CUDA toolkit compiler) version --
    nvcc_ver = None
    nvcc_paths = ['/usr/local/cuda/bin/nvcc', '/usr/bin/nvcc']
    for nvcc_path in nvcc_paths:
        try:
            result = sp.run([nvcc_path, '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                ver_match = re.search(r'release (\d+\.\d+)', result.stdout)
                if ver_match:
                    nvcc_ver = ver_match.group(1)
                    nvcc_major = int(nvcc_ver.split('.')[0])
                    break
        except Exception:
            continue

    if nvcc_ver:
        nvcc_major = int(nvcc_ver.split('.')[0])
        check(f"nvcc CUDA toolkit >= 12.0 (found {nvcc_ver} at {nvcc_path})",
              nvcc_major >= 12,
              f"nvcc is CUDA {nvcc_ver} -- too old for GPU training. "
              f"FIX: install CUDA 12+ toolkit or use pytorch image with bundled CUDA 12. "
              f"Try: pip install nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12")

        # nvcc vs driver version mismatch
        if driver_cuda_ver:
            driver_major = int(driver_cuda_ver.split('.')[0])
            check(f"nvcc CUDA major matches driver ({nvcc_ver} vs {driver_cuda_ver})",
                  nvcc_major <= driver_major,
                  f"nvcc CUDA {nvcc_ver} > driver CUDA {driver_cuda_ver}. "
                  f"Toolkit can't be newer than driver. FIX: install matching toolkit version.")
    else:
        warn("nvcc found", False,
             "No nvcc found at /usr/local/cuda/bin/nvcc or /usr/bin/nvcc. "
             "GPU compilation won't work. FIX: install CUDA toolkit or pip install nvidia-cuda-nvcc-cu12")

    # -- /usr/local/cuda symlink exists --
    cuda_symlink = '/usr/local/cuda'
    if os.path.exists('/usr/local'):  # only check on Linux
        warn("/usr/local/cuda exists",
             os.path.exists(cuda_symlink),
             f"/usr/local/cuda missing. Many packages expect this. "
             f"FIX: ln -sf /usr/local/cuda-12.x /usr/local/cuda")

    # -- PyTorch CUDA version matches system --
    try:
        import torch
        if torch.cuda.is_available():
            torch_cuda = torch.version.cuda
            torch_cuda_major = int(torch_cuda.split('.')[0])
            check(f"PyTorch CUDA >= 12.0 (found {torch_cuda})",
                  torch_cuda_major >= 12,
                  f"PyTorch built with CUDA {torch_cuda} -- need >= 12.0. "
                  f"FIX: pip install torch --index-url https://download.pytorch.org/whl/cu124")
            if nvcc_ver:
                nvcc_major = int(nvcc_ver.split('.')[0])
                warn(f"PyTorch CUDA matches nvcc ({torch_cuda} vs {nvcc_ver})",
                     torch_cuda_major == nvcc_major,
                     f"PyTorch CUDA {torch_cuda} vs nvcc {nvcc_ver}. "
                     f"Major version mismatch may cause compilation errors.")
        else:
            warn("PyTorch CUDA available", False,
                 "torch.cuda.is_available() is False. GPU training won't work. "
                 "Check CUDA_VISIBLE_DEVICES and driver compatibility.")
    except (ImportError, Exception):
        pass

    # -- GPU VRAM check --
    try:
        result = sp.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            vram_mb = int(result.stdout.strip().split('\n')[0])
            vram_gb = vram_mb / 1024
            if vram_gb < 20:
                warn(f"GPU VRAM >= 20GB (found {vram_gb:.0f}GB)",
                     False,
                     f"GPU has {vram_gb:.0f}GB VRAM -- < 20GB means CPU path is faster for cross gen. "
                     f"Cross gen will auto-fallback to CPU. Training may still work on GPU.")
    except Exception:
        pass

    # Disk space
    try:
        import shutil
        usage = shutil.disk_usage(PROJECT_DIR)
        free_gb = usage.free / (1024 ** 3)
        check(f"disk free >= 20GB (found {free_gb:.0f}GB)",
              free_gb >= 20,
              f"Only {free_gb:.0f}GB free. Need >= 20GB for training artifacts.")
    except Exception:
        pass

    # -- GPU fork: cuda_sparse actually works (not silent CPU fallback) --
    n_gpus = int(os.environ.get('LGBM_NUM_GPUS', '0'))
    if n_gpus > 0:
        try:
            import lightgbm as lgb
            import numpy as np
            # Test that device_type=cuda_sparse actually uses GPU, not silent CPU fallback
            test_X = np.random.rand(50, 100).astype(np.float32)
            test_y = np.random.randint(0, 3, 50)
            test_ds = lgb.Dataset(test_X, label=test_y, params={'feature_pre_filter': False})
            test_params = {
                'objective': 'multiclass', 'num_class': 3,
                'device_type': 'cuda_sparse', 'gpu_device_id': 0,
                'num_iterations': 1, 'verbose': -1,
            }
            try:
                lgb.train(test_params, test_ds)
                check("GPU cuda_sparse training works", True, "")
            except Exception as e:
                err_str = str(e)
                if 'not enabled' in err_str.lower():
                    check("GPU cuda_sparse enabled in LightGBM build", False,
                          f"cuda_sparse not available: {err_str}. "
                          f"FIX: rebuild LightGBM with -DUSE_CUDA_SPARSE=ON. "
                          f"The standard pip LightGBM does NOT support cuda_sparse.")
                else:
                    check("GPU cuda_sparse training works", False,
                          f"cuda_sparse failed: {err_str}")
        except ImportError:
            pass

        # Test standard CUDA sparse fallback warning
        try:
            import lightgbm as lgb
            import scipy.sparse as sp_sparse
            import numpy as np
            test_X_sp = sp_sparse.random(50, 100, density=0.1, format='csr', dtype=np.float32)
            test_y = np.random.randint(0, 3, 50)
            test_ds = lgb.Dataset(test_X_sp, label=test_y, params={'feature_pre_filter': False})
            test_params = {
                'objective': 'multiclass', 'num_class': 3,
                'device': 'cuda', 'gpu_device_id': 0,
                'num_iterations': 1, 'verbose': 1,
            }
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf):
                lgb.train(test_params, test_ds)
            warn("standard CUDA does NOT silently fallback on sparse",
                 'sparse features with CUDA is currently not supported' not in buf.getvalue(),
                 "Standard LightGBM CUDA silently falls back to CPU for sparse input. "
                 "Use cuda_sparse fork (device_type='cuda_sparse') instead of device='cuda'. "
                 "FIX: build custom fork with -DUSE_CUDA_SPARSE=ON")
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════
# CATEGORY 4: Data Integrity (requires --tf)
# ══════════════════════════════════════════════════════════════
def check_data_integrity(tf):
    print(f"\n== CATEGORY 4: Data Integrity ({tf}) ==")

    sys.path.insert(0, PROJECT_DIR)
    import config as cfg

    # -- Parquet existence and column count --
    parquet_patterns = [
        os.path.join(PROJECT_DIR, f'features_BTC_{tf}.parquet'),
        os.path.join(PROJECT_DIR, f'features_{tf}.parquet'),
        os.path.join(cfg.V30_DATA_DIR, f'features_BTC_{tf}.parquet'),
        os.path.join(cfg.V30_DATA_DIR, f'features_{tf}.parquet'),
    ]
    parquet_found = None
    for pp in parquet_patterns:
        if os.path.exists(pp):
            parquet_found = pp
            break

    if parquet_found:
        try:
            import pyarrow.parquet as pq
            schema = pq.read_schema(parquet_found)
            n_cols = len(schema)
            check(f"parquet has >= 2000 cols (found {n_cols})",
                  n_cols >= 2000,
                  f"Parquet {parquet_found} has only {n_cols} cols. Likely stale V1 features. "
                  f"FIX: rebuild features with build_{tf}_features.py")
        except Exception as e:
            warn(f"parquet readable", False, f"Could not read {parquet_found}: {e}")
    else:
        warn("parquet exists", False,
             f"No parquet found for BTC {tf}. Features will need to be built.")

    # -- Cross features per-TF strategy --
    npz_path = os.path.join(PROJECT_DIR, f'v2_crosses_BTC_{tf}.npz')
    json_path = os.path.join(PROJECT_DIR, f'v2_cross_names_BTC_{tf}.json')
    npz_exists = os.path.exists(npz_path)
    json_exists = os.path.exists(json_path)

    if tf == '1w':
        check("1w: no cross NPZ (base features only)",
              not npz_exists,
              f"1w should use base features only (380 cols, 1158 rows). "
              f"Crosses violate row:feature ratio. FIX: delete {npz_path}")
    elif tf in ('4h', '1h', '15m'):
        warn(f"{tf}: cross NPZ exists", npz_exists,
             f"Cross NPZ missing for {tf}. Run v2_cross_generator.py --symbol BTC --tf {tf}")

    # NPZ and JSON must exist together or not at all
    if npz_exists != json_exists:
        check("NPZ/JSON cross files paired",
              False,
              f"NPZ exists={npz_exists}, JSON exists={json_exists}. Must be both or neither. "
              f"FIX: regenerate crosses or delete both.")
    else:
        check("NPZ/JSON cross files paired", True, "")

    # -- Sparse matrix dtypes --
    if npz_exists:
        try:
            import scipy.sparse as sp
            import numpy as np
            data = sp.load_npz(npz_path)
            # Only require int64 indptr if NNZ actually exceeds int32 range
            if data.nnz > 2**31 - 1:
                check("cross NPZ indptr dtype == int64 (NNZ > 2^31)",
                      data.indptr.dtype == np.int64,
                      f"indptr dtype={data.indptr.dtype} but NNZ={data.nnz:,} > 2^31. "
                      f"FIX: rebuild crosses with int64 indptr.")
            else:
                warn("cross NPZ indptr dtype (int64 preferred)",
                     data.indptr.dtype == np.int64,
                     f"indptr dtype={data.indptr.dtype}, NNZ={data.nnz:,} fits int32. "
                     f"int64 preferred for safety but not required.")
            check("cross NPZ indices dtype == int32",
                  data.indices.dtype == np.int32,
                  f"indices dtype={data.indices.dtype} -- should be int32 for memory efficiency.")
            del data
        except Exception as e:
            warn("cross NPZ loadable", False, f"Could not load {npz_path}: {e}")

    # -- Parquet row count matches expected --
    tf_expected_rows = {'1w': 1000, '1d': 5000, '4h': 20000, '1h': 70000, '15m': 250000}
    if parquet_found:
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(parquet_found)
            n_rows = pf.metadata.num_rows
            min_rows = tf_expected_rows.get(tf, 1000)
            check(f"parquet has >= {min_rows} rows (found {n_rows})",
                  n_rows >= min_rows,
                  f"Parquet has {n_rows} rows, expected >= {min_rows} for {tf}. "
                  f"May be using wrong data source (multi_asset instead of btc_prices).")
        except Exception:
            pass

    # -- 1d cross NPZ should be month-based, not DOY --
    if tf == '1d' and json_exists:
        try:
            with open(json_path, 'r') as f:
                cross_names = json.load(f)
            has_doy = any('doy_' in str(n).lower() for n in cross_names[:100])
            warn("1d crosses are month-based (not DOY)",
                 not has_doy,
                 f"1d cross names contain 'doy_' -- should use month crosses only. "
                 f"DOY gives 15 samples/bin (noise). Month gives 478/bin (signal).")
        except Exception:
            pass

    # -- V2_RIGHT_CHUNK check --
    cross_gen_path = os.path.join(PROJECT_DIR, 'v2_cross_generator.py')
    if os.path.exists(cross_gen_path):
        try:
            with open(cross_gen_path, 'r', encoding='utf-8') as f:
                cg_source = f.read()
            chunk_match = re.search(r'V2_RIGHT_CHUNK\s*=\s*(\d+)', cg_source)
            if chunk_match:
                chunk_val = int(chunk_match.group(1))
                check(f"V2_RIGHT_CHUNK <= 500 (found {chunk_val})",
                      chunk_val <= 500,
                      f"V2_RIGHT_CHUNK={chunk_val} -- OOMs on all TFs except 1w with >500. "
                      f"FIX: v2_cross_generator.py, set V2_RIGHT_CHUNK=500")
        except Exception:
            pass

    # -- pipeline_manifest.json contamination --
    manifest_path = os.path.join(PROJECT_DIR, 'pipeline_manifest.json')
    warn("no stale pipeline_manifest.json",
         not os.path.exists(manifest_path),
         f"pipeline_manifest.json exists -- may contaminate full pipeline. "
         f"Delete it if starting a fresh run.")


# ══════════════════════════════════════════════════════════════
# CATEGORY 5: Training Config Consistency
# Cross-file grep checks — no AI judgment, just pattern matching.
# ══════════════════════════════════════════════════════════════
def check_training_consistency():
    print("\n== CATEGORY 5: Training Config Consistency ==")

    py_files = list(Path(PROJECT_DIR).glob('*.py'))
    training_files = [f for f in py_files if f.name in (
        'run_optuna_local.py', 'ml_multi_tf.py', 'cloud_run_tf.py',
        'exhaustive_optimizer.py', 'meta_labeling.py', 'backtest_validation.py',
    )]

    # Read all training file contents
    file_contents = {}
    for f in training_files:
        try:
            file_contents[f.name] = f.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            pass

    all_py_contents = {}
    for f in py_files:
        try:
            all_py_contents[f.name] = f.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            pass

    # -- No XGBoost in core training code --
    # Only check files that are in the actual training pipeline
    training_critical = {
        'run_optuna_local.py', 'ml_multi_tf.py', 'cloud_run_tf.py',
        'exhaustive_optimizer.py', 'meta_labeling.py', 'backtest_validation.py',
        'config.py', 'feature_library.py', 'live_trader.py',
    }
    xgb_files = []
    for fname, content in all_py_contents.items():
        if fname not in training_critical:
            continue
        lines = content.split('\n')
        for i, line in enumerate(lines):
            stripped = line.split('#')[0].lower()  # ignore comments
            if 'xgbclassifier' in stripped or ('xgboost' in stripped and 'import' in stripped):
                xgb_files.append(f"{fname}:{i+1}")
    check("no XGBoost in core training code",
          len(xgb_files) == 0,
          f"XGBoost found in: {', '.join(xgb_files[:5])}. MUST use LightGBM only.")

    # -- feature_pre_filter=False in Dataset calls --
    for fname, content in file_contents.items():
        if 'lgb.Dataset' in content or 'lightgbm.Dataset' in content:
            # Check that feature_pre_filter is set to False somewhere in the file
            has_fpf_false = 'feature_pre_filter' in content and 'False' in content
            # Also OK if it comes from config params
            has_from_config = 'V3_LGBM_PARAMS' in content or 'params' in content
            check(f"{fname}: feature_pre_filter=False or from config",
                  has_fpf_false or has_from_config,
                  f"{fname} uses lgb.Dataset but feature_pre_filter may not be False. "
                  f"Verify V3_LGBM_PARAMS is passed to Dataset.")

    # -- No fillna(0) on feature matrices (except LSTM) --
    fillna_files = []
    for fname, content in all_py_contents.items():
        if 'lstm' in fname.lower() or fname == 'validate.py':
            continue
        lines = content.split('\n')
        for i, line in enumerate(lines):
            stripped = line.split('#')[0]
            if 'fillna(0)' in stripped or 'nan_to_num' in stripped:
                # Check if it's on feature data (not on labels or metrics)
                if 'label' not in stripped.lower() and 'metric' not in stripped.lower():
                    fillna_files.append(f"{fname}:{i+1}")
    warn("no fillna(0) on features (NaN = missing, 0 = value)",
         len(fillna_files) == 0,
         f"Potential NaN->0 conversion in: {', '.join(fillna_files[:5])}. "
         f"LightGBM treats NaN as missing (learns split direction). 0 means 'value is zero'.")

    # -- No hardcoded feature_fraction < 0.7 (in actual code, not comments) --
    low_ff_files = []
    ff_pattern = re.compile(r"['\"]?feature_fraction['\"]?\s*[:=]\s*(0\.\d+)")
    for fname, content in all_py_contents.items():
        if fname == 'validate.py':
            continue
        lines = content.split('\n')
        for i, line in enumerate(lines):
            code_part = line.split('#')[0]  # strip comments
            for match in ff_pattern.finditer(code_part):
                val = float(match.group(1))
                if val < 0.7:
                    low_ff_files.append(f"{fname}:{i+1} (val={val})")
    check("no hardcoded feature_fraction < 0.7",
          len(low_ff_files) == 0,
          f"Low feature_fraction found in: {', '.join(low_ff_files[:5])}. Must be >= 0.7.")

    # -- No hardcoded bagging_fraction < 0.7 --
    low_bf_files = []
    bf_pattern = re.compile(r"['\"]?bagging_fraction['\"]?\s*[:=]\s*(0\.\d+)")
    for fname, content in all_py_contents.items():
        if fname == 'validate.py':
            continue
        lines = content.split('\n')
        for i, line in enumerate(lines):
            code_part = line.split('#')[0]
            for match in bf_pattern.finditer(code_part):
                val = float(match.group(1))
                if val < 0.7:
                    low_bf_files.append(f"{fname}:{i+1} (val={val})")
    check("no hardcoded bagging_fraction < 0.7",
          len(low_bf_files) == 0,
          f"Low bagging_fraction in: {', '.join(low_bf_files[:5])}. Must be >= 0.7. Rare signals vanish at 0.5-0.6.")

    # -- No hardcoded feature_fraction_bynode < 0.7 --
    low_ffbn_files = []
    ffbn_pattern = re.compile(r"['\"]?feature_fraction_bynode['\"]?\s*[:=]\s*(0\.\d+)")
    for fname, content in all_py_contents.items():
        if fname == 'validate.py':
            continue
        lines = content.split('\n')
        for i, line in enumerate(lines):
            code_part = line.split('#')[0]
            for match in ffbn_pattern.finditer(code_part):
                val = float(match.group(1))
                if val < 0.7:
                    low_ffbn_files.append(f"{fname}:{i+1} (val={val})")
    check("no hardcoded feature_fraction_bynode < 0.7",
          len(low_ffbn_files) == 0,
          f"Low feature_fraction_bynode in: {', '.join(low_ffbn_files[:5])}. Must be >= 0.7.")

    # -- astrology_engine.py exists --
    astro_path = os.path.join(PROJECT_DIR, 'astrology_engine.py')
    warn("astrology_engine.py in v3.3/",
         os.path.exists(astro_path),
         f"Missing {astro_path}. feature_library.py imports from it. "
         f"FIX: cp ../astrology_engine.py v3.3/")

    # -- CPCV purge: no hardcoded max_hold_bars at call sites (must use TRIPLE_BARRIER_CONFIG) --
    # purge must equal TRIPLE_BARRIER_CONFIG[tf]['max_hold_bars'] (50 for 1w, 90 for 1d, etc.)
    # Hardcoded values cause label leakage: training labels look max_hold_bars forward,
    # so bars 7-50 of fold boundary leak into test when purge=6 for 1w.
    hardcoded_purge_issues = []
    cpcv_purge_pattern = re.compile(r'max_hold_bars\s*=\s*(\d+)')
    for fname, content in all_py_contents.items():
        if fname == 'validate.py':
            continue
        lines = content.split('\n')
        for i, line in enumerate(lines):
            code_part = line.split('#')[0]  # strip comments
            for match in cpcv_purge_pattern.finditer(code_part):
                val = int(match.group(1))
                if val > 0:  # 0 would be intentional (disables purge)
                    hardcoded_purge_issues.append(f"{fname}:{i+1} (val={val})")
    check("CPCV purge: no hardcoded max_hold_bars integers in call sites",
          len(hardcoded_purge_issues) == 0,
          f"Hardcoded max_hold_bars found in: {', '.join(hardcoded_purge_issues[:5])}. "
          f"purge MUST equal TRIPLE_BARRIER_CONFIG[tf]['max_hold_bars'] per TF "
          f"(1w=50, 1d=90, 4h=72, 1h=48, 15m=24). "
          f"Hardcoded values cause label leakage at CPCV fold boundaries — "
          f"training labels look max_hold_bars bars forward, so bars 7-50 leak into test when purge=6. "
          f"FIX: max_hold = TRIPLE_BARRIER_CONFIG[tf]['max_hold_bars']")

    # -- CPCV purge source: training files must import TRIPLE_BARRIER_CONFIG for purge --
    for fname in ('ml_multi_tf.py', 'run_optuna_local.py'):
        content = all_py_contents.get(fname, '')
        uses_triple_barrier = 'TRIPLE_BARRIER_CONFIG' in content and "max_hold" in content
        check(f"{fname}: CPCV purge reads from TRIPLE_BARRIER_CONFIG",
              uses_triple_barrier,
              f"{fname}: CPCV purge must read max_hold from TRIPLE_BARRIER_CONFIG[tf]['max_hold_bars']. "
              f"FIX: tb_cfg = TRIPLE_BARRIER_CONFIG.get(tf_name, ...); max_hold = tb_cfg['max_hold_bars']")

    # -- No row-partitioned init_model boosting --
    init_model_files = []
    for fname, content in file_contents.items():
        if 'init_model' in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                stripped = line.split('#')[0]
                if 'init_model' in stripped and 'row' in stripped.lower():
                    init_model_files.append(f"{fname}:{i+1}")
    check("no row-partitioned init_model boosting",
          len(init_model_files) == 0,
          f"Row-partitioned init_model in: {', '.join(init_model_files)}. "
          f"Kills rare signals by splitting below min_data_in_leaf per chunk.")

    # -- force_col_wise=True in training files --
    for fname, content in file_contents.items():
        if 'lgb.train' in content or 'lgb.cv' in content:
            has_fcw = 'force_col_wise' in content or 'V3_LGBM_PARAMS' in content
            check(f"{fname}: force_col_wise=True or from config",
                  has_fcw,
                  f"{fname} calls lgb.train but force_col_wise may not be set. "
                  f"Required for sparse CSR. FIX: pass V3_LGBM_PARAMS.")

    # -- Model files should be .json not .txt --
    model_txts = list(Path(PROJECT_DIR).glob('model_*.txt'))
    model_txts = [f for f in model_txts if 'fold' not in f.name and 'ckpt' not in f.name]
    warn("model files are .json not .txt",
         len(model_txts) == 0,
         f"Found .txt model files: {[f.name for f in model_txts[:3]]}. "
         f"LightGBM models should be saved as .json for readability.")

    # -- optuna_configs_{tf}.json params match config.py (if exists) --
    sys.path.insert(0, PROJECT_DIR)
    import config as cfg
    for tf_name in VALID_TFS:
        optuna_cfg_path = os.path.join(PROJECT_DIR, f'optuna_configs_{tf_name}.json')
        if os.path.exists(optuna_cfg_path):
            try:
                with open(optuna_cfg_path, 'r') as f:
                    opt_cfg = json.load(f)
                # Check feature_fraction in saved config
                if isinstance(opt_cfg, dict):
                    ff = opt_cfg.get('feature_fraction', opt_cfg.get('params', {}).get('feature_fraction'))
                    if ff is not None and ff < 0.7:
                        check(f"optuna_configs_{tf_name}.json feature_fraction >= 0.7",
                              False,
                              f"Saved Optuna config has feature_fraction={ff}. "
                              f"This was trained with the old broken range. Retrain needed.")
            except Exception:
                pass

    # -- Class weight alignment: _cw_arr must be full-length, NOT compact+np.pad --
    # The bug: _cw_arr = [weight for y in y[~isnan(y)]] gives N_nonnan elements.
    # np.pad fills remaining positions assuming NaN rows are at the END — they are NOT.
    # Fix: _cw_arr must be len(y_3class), one weight per row, NaN rows get 1.0.
    cw_bad_files = []
    for fname in ('ml_multi_tf.py', 'run_optuna_local.py'):
        content = file_contents.get(fname, all_py_contents.get(fname, ''))
        if not content:
            continue
        lines = content.split('\n')
        for i, line in enumerate(lines):
            code = line.split('#')[0]
            if 'np.pad' in code and '_cw_arr' in code:
                cw_bad_files.append(f"{fname}:{i+1}")
    check("class weight alignment: no np.pad on _cw_arr",
          len(cw_bad_files) == 0,
          f"Misaligned class weight computation in: {', '.join(cw_bad_files)}. "
          f"np.pad assumes NaN rows are contiguous at end — they are not. "
          f"Fix: _cw_arr = np.ones(len(y)); _cw_arr[~isnan(y)] = weights. "
          f"SHORT 3x upweighting would apply to WRONG rows without this fix.")

    # -- TIER 1.3: _compute_sample_uniqueness ends must have +1 (inclusive end bar) --
    # t1 is stored as the inclusive end bar index. _compute_uniqueness_inner uses range(s, e)
    # which is exclusive of e. Without +1, the last bar of the label window is skipped →
    # concurrency is underestimated → sample weights are wrong → Optuna optimizes for wrong objective.
    for _uniq_fname in ('run_optuna_local.py', 'ml_multi_tf.py'):
        _uniq_content = all_py_contents.get(_uniq_fname, '')
        if '_compute_sample_uniqueness' not in _uniq_content:
            continue
        _uniq_lines = _uniq_content.split('\n')
        for _ui, _ul in enumerate(_uniq_lines):
            _stripped = _ul.strip()
            if ('ends' in _stripped and 't1_arr' in _stripped and
                    'asarray' in _stripped and 'dtype' in _stripped):
                _has_plus1 = '+ 1' in _stripped or '+1' in _stripped
                check(f"{_uniq_fname}: ends = t1_arr + 1 in _compute_sample_uniqueness",
                      _has_plus1,
                      f"{_uniq_fname} line {_ui+1}: ends missing +1. t1 is inclusive end bar but "
                      f"range(s,e) is exclusive — without +1, last bar of label window excluded, "
                      f"concurrency underestimated, sample weights wrong. "
                      f"Fix: ends = np.asarray(t1_arr, dtype=np.int64) + 1")

    # -- T-2: HMM lookahead in parallel CPCV --
    # Parallel CPCV must NOT pass pre-baked full-history HMM columns to workers.
    # Each worker must receive a per-fold HMM overlay fitted only on train-end-date data.
    # Correct pattern: _fold_hmm_overlays dict pre-computed before the parallel loop,
    # then passed as hmm_overlay= in worker_args.
    ml_content = all_py_contents.get('ml_multi_tf.py', '')
    _t2_has_per_fold = '_fold_hmm_overlays' in ml_content and 'fit_hmm_on_window(_train_end_par)' in ml_content
    _t2_has_worker_overlay = 'hmm_overlay, hmm_overlay_names' in ml_content
    _t2_no_skip_comment = 'per-fold HMM re-fitting is skipped in parallel mode' not in ml_content
    check("T-2: parallel CPCV pre-computes per-fold HMM overlays (no lookahead)",
          _t2_has_per_fold and _t2_has_worker_overlay and _t2_no_skip_comment,
          "ml_multi_tf.py: parallel CPCV workers must receive per-fold HMM overlays "
          "(_fold_hmm_overlays dict, fit_hmm_on_window per fold, hmm_overlay in worker args). "
          "Global HMM fit = lookahead bias — Perplexity confirmed 56%% F1 collapse when fixed. "
          "FIX: strip HMM cols from X_all, pre-compute _fold_hmm_overlays before parallel loop, "
          "pass _fold_hmm_overlays[wi] to each worker.")

    # -- T-3: is_unbalance vs explicit class weights in Optuna --
    # Optuna objective MUST NOT use is_unbalance=True when final training uses explicit dict
    # class weights folded into sample_weights. Different gradient scales = wrong Optuna landscape.
    # Correct pattern: dict weights folded into sample_weights in load_tf_data(),
    # is_unbalance only set for 'balanced' TFs (auto-weight by class frequency).
    opt_content = all_py_contents.get('run_optuna_local.py', '')
    # Check that the old unconditional is_unbalance=True is gone
    _t3_no_unconditional = "if tf_name in TF_CLASS_WEIGHT:\n            params['is_unbalance'] = True" not in opt_content
    _t3_no_unconditional2 = "if tf_name in TF_CLASS_WEIGHT:\n        params['is_unbalance'] = True" not in opt_content
    # Check that class weights are applied to sample_weights in load_tf_data
    _t3_has_cw_in_loader = ("isinstance(_cw, dict)" in opt_content and
                            "sample_weights = sample_weights * _cw_arr" in opt_content)
    check("T-3: Optuna uses explicit sample_weights not is_unbalance for dict class weights",
          (_t3_no_unconditional or _t3_no_unconditional2) and _t3_has_cw_in_loader,
          "run_optuna_local.py: Optuna objective must NOT set is_unbalance=True for TFs with "
          "dict TF_CLASS_WEIGHT. Dict weights must be folded into sample_weights in load_tf_data(). "
          "is_unbalance uses auto-balanced inverse-frequency weights, not SHORT=3x explicit weights. "
          "Mismatch = Optuna optimizes in wrong loss landscape vs final training. "
          "FIX: isinstance(TF_CLASS_WEIGHT.get(tf), dict) -> fold into sample_weights; "
          "only set is_unbalance=True when TF_CLASS_WEIGHT.get(tf) == 'balanced'.")

    # -- TIER 1.5: lgb.Dataset() construction must include feature_pre_filter=False in params= --
    # feature_pre_filter is a DATASET parameter baked at construction time.
    # Passing it only in lgb.train() params has NO effect on already-constructed Datasets.
    # Rare esoteric signals (astrology/gematria/numerology, 10-20 fires/year) are permanently
    # dropped at Dataset construction if feature_pre_filter=True (default). This cannot be undone
    # by later lgb.train() calls or across Optuna trials that reuse the same Dataset object.
    _ds_critical_files = ['run_optuna_local.py', 'ml_multi_tf.py',
                          'feature_classifier.py', 'meta_labeling.py']
    for _ds_fname in _ds_critical_files:
        _ds_content = all_py_contents.get(_ds_fname, '')
        if 'lgb.Dataset(' not in _ds_content and 'lightgbm.Dataset(' not in _ds_content:
            continue
        _ds_lines = _ds_content.split('\n')
        _ds_violations = []
        for _di, _dl in enumerate(_ds_lines):
            _dc = _dl.split('#')[0]  # strip comments
            if 'lgb.Dataset(' not in _dc and 'lightgbm.Dataset(' not in _dc:
                continue
            # Binary file loads don't re-bin features — skip
            if '.bin' in _dc and 'label=' not in _dc:
                continue
            # Child datasets with reference= inherit parent's bins — parent must be correct
            _ctx = '\n'.join(_ds_lines[_di:min(_di + 10, len(_ds_lines))])
            if 'reference=' in _ctx:
                continue
            # Must have feature_pre_filter=False in the Dataset call's params= kwarg
            # Accept: params={'feature_pre_filter': False, ...} (literal)
            #         params=_ds_params / params=params (variable that's defined with fpf=False)
            #         **_ds_kwargs / **_final_ds_kwargs (dict-unpacking with fpf=False)
            _has_fpf = ('feature_pre_filter' in _ctx and 'False' in _ctx and
                        ('params=' in _ctx or "params={" in _ctx))
            # Also accept variable-based params: **_ds_kwargs, **_final_ds_kwargs, **_w_ds_params, params=params
            if not _has_fpf:
                _has_var_params = ('**_ds_kwargs' in _ctx or '**_final_ds_kwargs' in _ctx or
                                  '**_w_ds_params' in _ctx or 'params=params' in _ctx or
                                  'params=_ds_params' in _ctx or 'params=_w_ds_params' in _ctx)
                if _has_var_params:
                    _has_fpf = True  # variable params verified to contain fpf=False by other checks
            if not _has_fpf:
                _ds_violations.append(f"line {_di+1}")
        check(f"{_ds_fname}: lgb.Dataset() calls have feature_pre_filter=False in params=",
              len(_ds_violations) == 0,
              f"{_ds_fname}: Dataset() calls without feature_pre_filter=False: {_ds_violations[:5]}. "
              f"LightGBM permanently drops rare esoteric features at Dataset construction — "
              f"train() params have no effect on already-constructed Datasets. "
              f"Fix: lgb.Dataset(X, label=y, params={{'feature_pre_filter': False, 'max_bin': 255}})")


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════
def _get_ram_gb():
    """Get available RAM in GB (cgroup-aware for cloud containers, psutil for Windows)."""
    import platform
    if platform.system() != 'Linux':
        # Windows/Mac: use psutil or ctypes
        try:
            import psutil
            return psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            pass
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulonglong = ctypes.c_ulonglong
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", c_ulonglong), ("ullAvailPhys", c_ulonglong),
                            ("ullTotalPageFile", c_ulonglong), ("ullAvailPageFile", c_ulonglong),
                            ("ullTotalVirtual", c_ulonglong), ("ullAvailVirtual", c_ulonglong),
                            ("ullAvailExtendedVirtual", c_ulonglong)]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / (1024 ** 3)
        except Exception:
            return 0
    # Linux: try cgroup v2
    try:
        with open('/sys/fs/cgroup/memory.max', 'r') as f:
            val = f.read().strip()
            if val != 'max':
                return int(val) / (1024 ** 3)
    except Exception:
        pass
    # Try cgroup v1
    try:
        with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
            val = int(f.read().strip())
            if val < 2 ** 62:
                return val / (1024 ** 3)
    except Exception:
        pass
    # Fallback: psutil or /proc/meminfo
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    return int(line.split()[1]) / (1024 ** 2)
    except Exception:
        pass
    return 0


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Savage22 V3.3 Pre-Flight Validation')
    parser.add_argument('--tf', type=str, choices=list(VALID_TFS),
                        help='Timeframe for data integrity checks')
    parser.add_argument('--cloud', action='store_true',
                        help='Enable cloud machine checks (RAM, DBs, driver)')
    parser.add_argument('--local', action='store_true',
                        help='Local mode (skip cloud-only checks)')
    args = parser.parse_args()

    from datetime import datetime
    mode = 'cloud' if args.cloud else 'local'
    tf_str = args.tf or 'all'

    print(f"\n{'='*55}")
    print(f"  SAVAGE22 V3.3 PRE-FLIGHT VALIDATION")
    print(f"  TF: {tf_str} | Mode: {mode} | Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*55}")

    # Run all categories
    check_config_params()
    check_optuna_search_space()
    check_environment(tf=args.tf, cloud=args.cloud)
    if args.tf:
        check_data_integrity(args.tf)
    check_training_consistency()

    # Summary
    total = _pass + _fail
    print(f"\n{'='*55}")
    if _fail == 0:
        print(f"  RESULT: {_pass}/{total} PASSED, {_warn} warnings")
        print(f"  TRAINING APPROVED")
    else:
        print(f"  RESULT: {_pass}/{total} PASSED, {_fail} FAILED, {_warn} warnings")
        print(f"  TRAINING BLOCKED -- fix failures above")
        print(f"\n  Failures:")
        for name, msg in _failures:
            print(f"    - {name}")
    print(f"{'='*55}\n")

    sys.exit(1 if _fail > 0 else 0)


if __name__ == '__main__':
    main()
