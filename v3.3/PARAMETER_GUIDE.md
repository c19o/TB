# Savage22 V3.3 — Comprehensive Parameter Reference & Per-TF Tuning Guide

**Single source of truth**: `validate.py` (see runtime count via `python validate.py`).
If this document and validate.py disagree, **validate.py wins**.

---

## 1. SACRED PARAMETERS (NEVER CHANGE)

These parameters protect the matrix thesis: rare esoteric signals ARE the edge.
Changing any of these kills rare signal detection silently.

| Parameter | Value | Why Sacred | What Breaks If Changed |
|-----------|-------|------------|----------------------|
| `feature_fraction` | >= 0.7 (default 0.9) | Each tree must see 70%+ of features. Below 0.7, rare cross signals (firing 2-3x/year) are randomly excluded from too many trees and never get a split | Rare esoteric crosses become invisible. 1w AUC drops silently — no error, just worse predictions |
| `feature_fraction_bynode` | >= 0.7 (default 0.8) | Per-node fraction compounds with `feature_fraction`. 0.5 * 0.7 = 0.35 effective — devastating | Same as above but worse: the compounding effect means rare signals almost never get evaluated |
| `feature_pre_filter` | `False` | When `True`, LightGBM silently removes features with zero gain in first pass. Rare signals may show zero gain in early trees but become critical in later tree interactions | Protected esoteric features silently removed at Dataset construction. No warning, no error |
| `bagging_fraction` | >= 0.95 (default 0.95) | P(rare signal in bag) = bf^n. At 0.95: P(10-fire signal) = 59.9%. At 0.8: only 10.7% | Rare signals almost never appear in training bags. Model learns without them |
| `is_enable_sparse` | `True` | CSR sparse input — 2.9M+ features would OOM as dense | OOM crash. Dense matrix for 2.9M features is impossible |
| `max_bin` | 7 | Binary features need 2 bins. 4-tier binarization needs ~5. 7 = safe ceiling. 36x less memory than 255 | Memory explosion (255 bins * 2.9M features), no accuracy gain on binary data |

### Validate.py Enforcement
```
check("feature_fraction >= 0.7")
check("feature_fraction_bynode >= 0.7")
check("feature_pre_filter == False")
check("bagging_fraction >= 0.95")      # floor; config default is 0.95
check("is_enable_sparse == True")
check("max_bin == 7")
```

### Additional Protected Parameters
| Parameter | Value | Enforced By |
|-----------|-------|-------------|
| `boosting_type` | `gbdt` | validate.py |
| `objective` | `multiclass` | validate.py |
| `num_class` | 3 | validate.py |
| `min_data_in_bin` | 1 | validate.py — allows bins with 1 sample (rare signals) |
| `num_threads` | 0 | validate.py — 0 = auto-detect via OpenMP. Never use -1 |
| `force_col_wise` | `True` | validate.py — default for most TFs (15m overrides to row_wise) |
| `bagging_freq` | > 0 | validate.py — LightGBM silently ignores bagging_fraction if 0 |

---

## 2. PER-TF TUNING GUIDE

### 2.1 Weekly (1w) — 819 rows

**Dataset**: Smallest. Every row matters. Every rare signal matters more.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| CPCV groups | (8, 2) | C(8,2)=28 paths, 75% train. Max data per fold for tiny dataset + 50-bar purge |
| CPCV sample paths | 28 (exhaustive) | Only 28 total — use all |
| Optuna Phase 1 trials | 30 | More thorough search needed — tiny data is sensitive to hyperparams |
| Phase 1 LR | 0.20 | Higher LR for faster convergence on tiny data |
| Phase 1 rounds | 40 | Fewer rounds at higher LR prevents overfitting |
| Final LR | 0.1 (override) | Default 0.03 causes validation metric to never improve → ES kills at tree 1 |
| Final rounds | 300 (cap) | Overfits by ~200 rounds at LR=0.03. 300 with ES is safe ceiling |
| ES patience | 50 | On 819 rows, genuine improvements happen early or not at all |
| num_leaves | 15 (cap 31) | Optuna searches [4, 31]. Too many leaves memorizes 819 rows |
| min_data_in_leaf | 2 (floor), 8 (max) | Must catch signals firing 5-10x. Optuna searches [2, 8] |
| max_depth range | (3, 8) | Floor=3 prevents trivially shallow stumps. Cap=8 prevents memorization |
| LR search range | (0.05, 0.3) | Tiny data needs to find its own optimal LR |
| Row subsample | 1.0 | Cannot subsample 819 rows |
| Class weight | SHORT=2x | Without upweighting, model never predicts SHORT |
| Label smoothing | epsilon=0.15 | More smoothing needed for tiny dataset |
| Lean mode | ON | Keeps only SAR/EMA/RSI TA + all esoteric. Drops redundant TA |
| Binary mode | ON | Binary classification (UP/DOWN) |
| force_row_wise | No (col_wise) | Low rows/high features ratio |

**1w Proven Result**: AUC=57.5% CPCV, model AUC=79.3%. Binary mode. All 7 steps PASS.

### 2.2 Daily (1d) — 5,733 rows

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| CPCV groups | (5, 2) | C(5,2)=10 paths, 60% train. Optimal for 3-20 signal fires |
| CPCV sample paths | 10 (exhaustive) | Only 10 total — use all |
| Optuna Phase 1 trials | 25 | Standard |
| Phase 1 LR | 0.15 (default) | Standard convergence |
| Phase 1 rounds | 60 (default) | Standard |
| Final LR | 0.03 (default) | Standard |
| Final rounds | 600 (cap) | Moderate cap — more data than 1w but still moderate |
| num_leaves | 15 (cap 15) | Conservative for moderate data. Optuna searches [4, 15] |
| min_data_in_leaf | 8 (floor), 10 (max) | Lowered to 8 for rare signal headroom |
| max_depth range | (3, 6) | 338K features — cap depth to prevent memorization |
| Row subsample | 1.0 | Use all rows |
| Class weight | DOWN=3x | Binary upweight for DOWN class |
| Label smoothing | epsilon=0.10 (default) | Standard |
| Binary mode | ON | Binary classification |

### 2.3 Four-Hour (4h) — 23,000 rows

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| CPCV groups | (10, 2) | C(10,2)=45 paths, 80% train. Sample 30 paths |
| CPCV sample paths | 30 | Deterministic sample from 45 (seed=42) |
| Optuna Phase 1 trials | 25 | Standard |
| Final LR | 0.03 (default) | Standard |
| Final rounds | 800 (default) | Full rounds — enough data to support it |
| num_leaves | 31 (cap 31) | Standard — EFB ratio 0.92:1 |
| min_data_in_leaf | 8 (floor), 10 (max) | Lowered to 8 |
| max_depth range | (2, 8) (default) | Standard |
| Row subsample | 1.0 | Sparse fix eliminates OOM — use all data |
| Class weight | SHORT=2x | Insurance (v3.2 4h had SHORT accuracy issues) |
| Label smoothing | epsilon=0.10 (default) | Standard |
| Memory | Sparse works fine | No memmap/tcmalloc needed |

### 2.4 One-Hour (1h) — 75,000 rows

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| CPCV groups | (10, 2) | C(10,2)=45 paths, 80% train. Sample 30 paths |
| CPCV sample paths | 30 | Deterministic sample from 45 (seed=42) |
| Optuna Phase 1 trials | 25 | Standard |
| Final LR | 0.03 (default) | Standard |
| Final rounds | 800 (default) | Full rounds |
| num_leaves | 63 (cap 63) | Can handle complexity — EFB ratio 3.8:1 |
| min_data_in_leaf | 8 (floor), 10 (max) | Standard |
| Row subsample | 0.50 | 75K → ~38K for Optuna speed. Final model uses ALL rows |
| Label smoothing | epsilon=0.10 (default) | Standard |
| Memory | **May need memmap or tcmalloc** | Cross gen can exhaust memory |

### 2.5 Fifteen-Minute (15m) — 294,000 rows

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| CPCV groups | (10, 2) | C(10,2)=45 paths, 80% train. Sample 30 paths |
| CPCV sample paths | 30 | Deterministic sample from 45 (seed=42) |
| Optuna Phase 1 trials | 30 | More trials — largest dataset, more room to explore |
| Final LR | 0.03 (default) | Standard |
| Final rounds | 800 (default) | Full rounds — enough data |
| num_leaves | 127 (cap 127) | Deep trees viable — EFB ratio 6.9:1 |
| min_data_in_leaf | 8 (floor), 10 (max) | Standard |
| Row subsample | 0.25 | 294K → ~74K for Optuna speed. Final model uses ALL rows |
| force_row_wise | **Yes** | 294K rows / 23K EFB bundles = 12.8. Row-wise is faster |
| Label smoothing | epsilon=0.10 (default) | Standard |
| Memory | **CPU cross-gen only. May need memmap or tcmalloc** | LARGEST dataset. OOMs on 44GB A40 GPU. Confirm cloud machine target with user before runs |

---

## 3. OPTUNA SEARCH SPACE RATIONALE

### 3.1 Phase Architecture

```
Phase 1: 2 seeded + 8 random + 15 TPE = 25 trials (per-TF overrides available)
  → 2-fold CPCV, fast LR (0.15), 60 rounds max, ES patience=15
  → Purpose: rapid coarse search

Validation Gate: Top-3 from Phase 1 re-evaluated
  → 4-fold CPCV, 200 rounds, LR=0.08, ES patience=50
  → Purpose: confirm signal isn't noise. More folds = more statistical power

Final Retrain: Best validated params
  → Full CPCV K=2, 800 rounds (per-TF caps), LR=0.03
  → Purpose: production model with all data
```

### 3.2 Parameter Ranges & Consequences

| Parameter | Range | Why This Range | If Too Low | If Too High |
|-----------|-------|----------------|------------|-------------|
| `num_leaves` | [4, TF cap] | Cap scales with data size. 4 minimum = v3.2 best was 7 | Underfitting — can't capture interactions | Memorization on small TFs |
| `min_data_in_leaf` | [2-8, 8-10] | Must be <= rare signal frequency (10-20 firings) | Overfitting on noise | Rare signals invisible — can't create leaf with fewer samples |
| `feature_fraction` | [0.7, 1.0] | Floor=0.7 is sacred. 1.0 = deterministic feature selection | Below 0.7: rare crosses randomly excluded | No downside at 1.0 |
| `feature_fraction_bynode` | [0.7, 1.0] | Same as above — compounds with feature_fraction | Effective fraction drops below threshold | No downside at 1.0 |
| `bagging_fraction` | [0.95, 1.0] | Floor=0.95. P(10-fire in bag@0.95) = 59.9% | Below 0.95: rare signals rarely in training bags | 1.0 = no bagging (deterministic) |
| `lambda_l1` | [1e-4, 4.0] log | Capped at 4.0 — values in [1, 100] zeroed rare signals firing ≤15 times | Near-zero: no L1 regularization | >4.0: rare signal leaf values shrunk to zero |
| `lambda_l2` | [1e-4, 10.0] log | Log-scale, mass near zero. Capped at 10.0 | Near-zero: no L2 regularization | >10.0: over-regularized, all signals dampened |
| `min_gain_to_split` | [0.0, 5.0] | 0.0 allows any gain. 5.0 = conservative | 0.0: more splits, potential noise | >5.0: many valid rare splits blocked |
| `max_depth` | [TF lo, TF hi] | 1w: [3,8], 1d: [3,6], others: [2,8] | Shallow stumps can't capture interactions | Deep trees memorize small datasets |
| `extra_trees` | [True, False] | Categorical. Extra Trees = random split points | False: standard greedy splits | True: more randomization, can help generalization |
| `learning_rate` | TF-specific | 1w: searchable [0.05, 0.3]. Others: fixed per phase | Too low on tiny data: ES kills before learning | Too high: overshoots, unstable |

### 3.3 Validate.py Enforcement on Search Space
```
check("Optuna feature_fraction lower >= 0.7")
check("Optuna feature_fraction upper <= 1.0")
check("Optuna feature_fraction_bynode lower >= 0.7")
check("Optuna bagging_fraction lower >= 0.95")
check("Optuna lambda_l1 upper <= 4.0")
check("Optuna lambda_l2 upper <= 10.0")
check("Optuna num_leaves lower >= 4")
check("Optuna min_gain_to_split lower >= 0.0")
check("Optuna min_gain_to_split upper <= 5.0")
check("Optuna max_depth upper <= 12")
check("Phase 1 LR >= 0.10")
check("Phase 1 ES patience >= 10")
check("no xgboost in Optuna search")
```

---

## 4. EARLY STOPPING CRITERIA

### 4.1 During Optuna Phase 1
- **ES patience**: 15 rounds (aggressive — we're searching, not training)
- **Max rounds**: 60 (per-TF overrides: 1w=40)
- **LR**: 0.15 (per-TF: 1w=0.20) — high LR means convergence or failure is fast

### 4.2 During Validation Gate
- **ES patience**: 50 rounds (patient — rare signals may need many rounds to show)
- **Max rounds**: 200
- **LR**: 0.08

### 4.3 During Final Retrain
- **ES patience**: Dynamic formula: `max(50, int(100 * (0.1 / lr)))`. At LR=0.03 → 333, at LR=0.1 → 100
- **Per-TF override**: 1w=50 (819 rows — don't waste rounds past convergence)
- **Max rounds**: 800 (per-TF: 1w=300, 1d=600)

### 4.4 Pipeline-Level Quality Gates
| Gate | Threshold | Action |
|------|-----------|--------|
| After trial 1 | AUC > 0.51 | If not, investigate data/features before continuing |
| After 10 trials | AUC > 0.53 | If not, stop and investigate — something structural is wrong |
| Train/val gap | < 0.15 | If gap > 0.15, overfitting. Investigate regularization params |
| Best vs worst fold | Spread < 0.10 | Large spread = unstable model. Check if single fold is an outlier |

### 4.5 Historical Benchmarks
- **1w**: AUC=57.5% CPCV, model AUC=79.3% (binary mode). All 7 steps PASS.
- **Other TFs**: Training pending (1d next in priority stack).

---

## 5. CROSS-GENERATION & MEMORY CONSIDERATIONS

### 5.1 Per-TF Memory Requirements
| TF | Rows | Features | Cross Features | Memory Notes |
|----|------|----------|----------------|--------------|
| 1w | 819 | ~600 base | ~600 crosses | Fast either way. Sparse fine |
| 1d | 5,733 | ~23K base | ~23K crosses | Sparse fine |
| 4h | 23,000 | ~2.9M base | ~2.9M crosses | Sparse fine. RIGHT_CHUNK=500 |
| 1h | 75,000 | ~5M base | ~5M crosses | **May need memmap/tcmalloc** |
| 15m | 294,000 | ~10M base | ~10M crosses | **CPU-only cross gen. Memmap likely needed** |

### 5.2 Critical Cross-Gen Settings
- `V2_RIGHT_CHUNK=500` — Auto=2000 OOMs on all TFs except 1w
- `OMP_NUM_THREADS=4` — Prevents thread exhaustion
- GPU < 20GB VRAM → use CPU for cross gen (faster + avoids CUDA 13 crashes)
- `int64 indptr` for CSR matrices — handles NNZ > 2^31

### 5.3 EFB Pre-Bundle
All TFs: `EFB_PREBUNDLE_ENABLED = True`
- Binary features packed 127/bundle → 79K bundles instead of 10M histograms
- Biggest win on 15m: 128x histogram reduction

---

## 6. PROTECTED FEATURE PREFIXES

These prefixes are **NEVER pruned, filtered, or removed** regardless of gain scores.
This is the core philosophy: esoteric signals ARE the edge.

See `config.py:PROTECTED_FEATURE_PREFIXES` for the full list (~70 prefixes covering):
- Gematria (`gem_`, `gematria`, `digital_root`)
- Astrology (`moon_`, `eclipse`, `vedic_`, `bazi_`, `tzolkin_`, `arabic_lot`)
- Space weather (`sw_`, `schumann_`)
- Numerology (`master_`, `angel`, `palindrome`, `vortex_`, `sephirah`)
- Social signals (`tweet`, `news_gem`, `news_sentiment`)
- On-chain (`onchain`, `funding`, `oi_`, `whale`)
- Cross features (`cross_`, `dx_`, `ax_`, `mx_`, `vx_`)
- Calendar (`hebrew_`, `shmita`, `chinese_new_year`, `diwali_`, `ramadan_`)
- New V3.3 (`bio_`, `rahu_`, `ketu_`, `fib_`, `gann_`)

---

## 7. QUICK REFERENCE: CONFIG VARIABLE LOCATIONS

| What | Variable | File:Line |
|------|----------|-----------|
| Sacred LightGBM params | `V3_LGBM_PARAMS` | config.py:495 |
| Per-TF min_data_in_leaf | `TF_MIN_DATA_IN_LEAF` | config.py:530 |
| Per-TF num_leaves | `TF_NUM_LEAVES` | config.py:591 |
| Per-TF CPCV groups | `TF_CPCV_GROUPS` | config.py:574 |
| CPCV sampling | `CPCV_SAMPLE_PATHS` | config.py:586 |
| Optuna phase config | `OPTUNA_PHASE1_*` | config.py:639 |
| Per-TF Optuna trials | `OPTUNA_TF_PHASE1_TRIALS` | config.py:666 |
| Per-TF LR overrides | `TF_LEARNING_RATE` | config.py:602 |
| Per-TF round caps | `TF_NUM_BOOST_ROUND` | config.py:609 |
| Optuna search space | `run_optuna_local.py:646` | Actual suggest_* calls |
| Protected prefixes | `PROTECTED_FEATURE_PREFIXES` | config.py:194 |
| Label smoothing | `LABEL_SMOOTHING_*` | config.py:710 |
| Class weights | `TF_CLASS_WEIGHT` | config.py:565 |
| Skip features | `SKIP_FEATURES_BY_TF` | config.py:347 |
| Lean 1w config | `LEAN_1W_MODE` | config.py:359 |
| All validation checks | `validate.py` | 97 checks (run `python validate.py` for current count) |

---

*Last updated: 2026-04-01. Check count updated from 74→97 per validate.py audit.*
*Consult validate.py as authoritative source for all numerical constraints.*
