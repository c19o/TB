# V3.3 Session Resume — 2026-03-30 (Optimization Company + 1w Training)

## INSTRUCTION TO NEW SESSION
Read this file completely. Then read v3.3/CLAUDE.md. Resume from where we left off — bugs need fixing, features need adding, then retrain 1w on the running cloud machine.

---

## ACTIVE MACHINE (STILL RUNNING — $0.54/hr)
- **Instance 33852303** — Ontario CA, 1x RTX 5090, EPYC 7B12 128c, 258GB RAM
- **SSH**: `ssh -p 12302 root@ssh7.vast.ai`
- **Status**: Training KILLED mid-run (accuracy was bad due to 3 bugs). Machine still alive.
- **Files deployed**: All .py symlinked in /workspace/, 24+ .db files, kp_history_gfz.txt
- **EVALUATE BEFORE DESTROYING** — machine has partial results

---

## WHAT WAS DONE THIS SESSION

### Phase 1: Optimization Company (COMPLETED)
- **Wave 1**: 13 read-only analysts found **5 CRITICAL + 16 HIGH** issues
- **Wave 2**: 12 fix agents implemented all fixes on worktree branches
- **Audit Company 2**: 25 auditors verified work. Matrix thesis verified clean.
- **Perplexity Team**: 10 agents (5 fixers + 5 researchers) found:
  - SMOKING GUN: `deterministic=True` in config.py forcing single-threaded histograms (killing ALL GPU parallelism)
  - `max_bin=7` instead of 255 (36x less histogram memory for binary features)
  - `OMP_NUM_THREADS=4` hardcoded, throttling cross gen to 3% of cores
  - cuSPARSE for cross gen could give 5-15x speedup
  - CUDA bitpack kernel: 0.5-2 sec cross gen (vs 30-120s current)
  - Multi-GPU Sobol optimizer implemented (7 GPUs parallel)
  - Assembly-line pipeline orchestrator built (overlap CPU/GPU phases)
  - Complete Linux tuning script (cloud_setup.sh)
  - LightGBM compile from source with -O3 -march=znver4 for AVX-512

### Phase 2: Merge (COMPLETED)
4 branches merged into v3.3-clean:
1. `ceo/backend-dev-c45180ac` — CUDA fork fixes (GPU dead-code removed, ballot_sync, error enum, SM_120)
2. `ceo/backend-dev-55d694ab` — Optimizer GPU fixes (multi-GPU Sobol, spawn context, ALLOW_CPU removal)
3. `ceo/backend-dev-552c4cfd` — Optuna fixes (WilcoxonPruner, n_jobs divisor, validation gate, cache staleness)
4. `ceo/backend-dev-1abf748a` — CPCV fixes (dense guard, thread cap, timeout, SharedMemory)
- Pushed to GitHub: https://github.com/c19o/TB-3.3 (remote: tb33)

### Phase 3: 1w Training Run (FAILED — 3 bugs found)
**3 bugs blocking proper training:**
1. **Optuna crashes on dense data** — `runtime_checks.py:51` calls `.nnz` on dense ndarray. 1w has no cross features = dense. Optuna never ran.
2. **LSTM fails on RTX 5090 (sm_120)** — PyTorch CUDA incompatible with SM 120.
3. **--no-parallel-splits breaks downstream** — optimizer, PBO, meta, audit crash on unrecognized CLI arg.

**1w CPCV results (unoptimized defaults, bugs):**
- Acc=37.8%, PrecL=18.4%, PrecS=35.5% (10 paths)
- 7/10 paths Trees=1, ZERO esoteric features active
- NOT TRADABLE — needs Optuna + feature fixes + param tuning

### Phase 4: Expert Analysis (COMPLETED)
**Astrology Expert:** 4 dead features, no eclipse/natal transit features called, snapshot bias on weekly
**Numerology Expert:** No week-of-year DR, no month DR, no halving cycle features, no date gematria
**Feature Trimming:** 10 constant features to remove, 5+ to add (week-of-year, quarter, halving, eclipse, etc.)
**Feature Importance:** 96/3665 active (2.6%), all TA, zero esoteric — expected for 819 rows per Perplexity

### Phase 5: Training Parameter Team (IN PROGRESS — 5 agents were running)
Analyzing: LightGBM hyperparams, Optuna search space, CPCV config, class weights, trade optimizer params

### Phase 6: Bug Fix + Feature Add (IN PROGRESS — 2 agents were running)
1. Bug fixer — fixing 3 bugs (dense .nnz, LSTM sm_120, --no-parallel-splits)
2. Feature engineer — adding 10 weekly features, trimming 10 constant ones

---

## WHAT NEEDS TO BE DONE NEXT SESSION

### Priority 1: Fix 3 Bugs
1. `runtime_checks.py:51` — add `scipy.sparse.issparse(X)` guard before `.nnz`
2. LSTM + RTX 5090 — upgrade PyTorch or graceful skip with WARNING
3. Remove `--no-parallel-splits` CLI arg, use env var instead

### Priority 2: Add Weekly Features (feature_library.py)
- TRIM: 10 constant features for 1w (hour_sin/cos, dow_sin/cos, day_of_week, is_monday, is_friday, is_weekend, day_of_month, is_month_end)
- ADD: week_of_year sin/cos, week_digital_root, month_digital_root, quarter sin/cos, year_in_halving_cycle, weeks_since_halving, eclipse window, BTC natal transit, Jupiter-Saturn regime, Mercury retrograde days

### Priority 3: Apply Config Overrides (CRITICAL — not yet in config.py)
- `deterministic=False` (currently True — kills GPU parallelism)
- `max_bin=7` (currently 255 — 36x waste for binary features)
- `OMP_NUM_THREADS=128` (currently hardcoded 4 in v2_cross_generator.py:72)

### Priority 4: Tune 1w Parameters
- Get parameter team results (5 agents were analyzing)
- Likely: higher LR (0.1+), fewer CPCV groups (3,1), lower ES patience, adjusted class weights

### Priority 5: Retrain 1w on Cloud Machine
- Machine 33852303 STILL RUNNING at $0.54/hr — DESTROY IF NOT NEEDED SOON
- After fixes: SCP updated code, rerun

### Priority 6: Move to 1d
- After 1w completes, move to 1d (same machine or bigger)

---

## REVISED ETAs (All Optimizations Applied)

### 1x RTX 5090 + 128c EPYC
| TF | TOTAL | Peak RAM | Peak VRAM |
|----|-------|----------|-----------|
| 1w | 3-5 min | 4 GB | 4 GB |
| 1d | 16-24 min | 16 GB | 8 GB |
| 4h | ~20 min | 24 GB | 10 GB |
| 1h | 65-107 min | 50 GB | 10 GB |
| 15m | 5-7.5 hr | 65 GB | 10 GB |

### 8x RTX 5090 + 128c EPYC
| TF | TOTAL | Peak RAM |
|----|-------|----------|
| 1w | 2-3.5 min | 4 GB |
| 1d | 10-16 min | 16 GB |
| 4h | ~10 min | 24 GB |
| 1h | 55-92 min | 50 GB |
| 15m | 3.5-5.5 hr | 65 GB |
| **ALL 5** | **~6-8 hr** | |

256GB RAM is plenty. 1TB overkill.

---

## WORKTREES TO PRUNE
~50 worktrees exist from CEO agents. Run `ceo_prune_worktrees` and delete all `ceo/*` branches.

## CEO SESSION STATS
- ~200+ sessions this conversation
- Total spend: ~$200+

## GIT STATE
- Branch: v3.3-clean
- Latest commit: 7bfea64 (dense→sparse CPCV fix)
- Pushed to: tb33 (github.com/c19o/TB-3.3)
- Bug fixer + feature engineer agents may have uncommitted local changes — check `git status`

## BEHAVIORAL RULES (REINFORCED)
1. Claude Max = unlimited. No budget/turn caps on CEO agents.
2. Worktrees OK (source-only repo, lightweight)
3. ALL agents consult Perplexity with matrix thesis context
4. NEVER solo debug — launch company
5. Cut training short when results are clearly bad
6. Document EVERYTHING scientifically
7. 1w/1d need TF-specific features (month-of-year not day-of-year, long-term astrology cycles)
