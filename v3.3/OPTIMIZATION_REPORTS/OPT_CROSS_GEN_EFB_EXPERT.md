# Expert Audit: Cross Generator + EFB Pre-Bundler

**Auditor:** Sparse Matrix / Cross Generation Expert
**Date:** 2026-03-30
**Files:** `v2_cross_generator.py`, `efb_prebundler.py`, `config.py`
**Matrix thesis:** 2.9M features from cross steps on esoteric signals. Sparse CSR. sparse-dot-mkl. RIGHT_CHUNK=500. ALL features must reach the model.

---

## 1. CROSS GENERATOR AUDIT

### 1A. Cross Steps Inventory

**FINDING: 12 steps present, not 13.** The module docstring claims "3x regime-aware DOY crosses" as a 13th type, but line 2149 reads: `# Cross 13: REMOVED -- Regime-aware DOY was redundant with dx_ (Cross 1)`. The docstring at line 21 is stale.

| Step | Prefix | Left | Right | Status |
|------|--------|------|-------|--------|
| 1 | `dx` | DOY/month windows | ALL contexts | OK |
| 2 | `ax` | Astro | TA | OK |
| 3 | `ax2` | Multi-astro combos (max 50) | TA | OK |
| 4 | `ta2` | Multi-TA combos (max 30) | DOY + astro | OK |
| 5 | `ex2` | Esoteric | TA | OK |
| 6 | `sw` | Space weather | TA | OK |
| 7 | `hod` | Session/hour-of-day | TA + astro | OK |
| 8 | `mx` | Macro | TA | OK |
| 9 | `vx` | Volatility | TA + DOY | OK |
| 10 | `asp` | Aspects | TA | OK |
| 11 | `pn` | Price numerology | TA | OK |
| 12 | `mn` | Moon signs | TA | OK |

**Verdict:** All intended cross types are present. Step 13 (regime-aware DOY) was intentionally removed as redundant. Docstring at line 21 should be updated.

### 1B. Co-occurrence Threshold

**FINDING: Threshold is MIN_CO_OCCURRENCE=3, down from 8. Correct.**

- Line 354: `MIN_CO_OCCURRENCE = int(_env_co_occur) if _env_co_occur else 3`
- Comment: "Lowered from 8: matches min_data_in_leaf=3, preserves rare esoteric crosses"
- Override via `V2_MIN_CO_OCCURRENCE` env var
- **Analysis:** A cross feature firing <3 times cannot reliably appear in both train AND validation splits of 5-fold CPCV. threshold=3 is the mathematical floor -- this is correct. Threshold=8 was overly conservative and would silently prune rare esoteric signals (exactly what the matrix thesis forbids).

**Verdict: PASS.** Threshold is at the minimum viable level for CPCV validation integrity. No rare signals silently killed.

### 1C. sparse-dot-mkl Configuration

**FINDING: Correctly configured with threadpoolctl boost.**

- Import at line 52-55: `from sparse_dot_mkl import dot_product_mkl`
- Wrapper `_mkl_dot()` at line 146-153 uses `threadpool_limits` to temporarily boost MKL threads from the default OMP_NUM_THREADS=4 to `max(4, cpu_count // 2)`.
- Rationale: OMP_NUM_THREADS=4 is set globally to prevent Numba prange thread exhaustion on 128+ core machines, but this would starve MKL SpGEMM. The scoped override is correct.
- `cast=True` passed to handle mixed dtypes safely.

**Verdict: PASS.** MKL is properly integrated with the thread-count workaround.

### 1D. Silent Feature Drops

#### BUG-1: NameError in pair sorting (CRITICAL -- crash, not silent drop)

**Lines 1045 and 1326 reference `co_occur` which is NOT in scope.**

In `_gpu_cross_chunk()` (line 1037):
```python
valid_pairs, _co_method = _compute_cooccurrence_pairs(left_mat, right_mat, min_nonzero)
# ...
pair_nnz = co_occur[valid_pairs[:, 0], valid_pairs[:, 1]]  # NameError!
```

In `_cpu_cross_chunk()` (line 1319):
```python
valid_pairs, _co_method = _compute_cooccurrence_pairs(left_mat, right_mat, min_nonzero)
# ...
pair_nnz = co_occur[valid_pairs[:, 0], valid_pairs[:, 1]]  # NameError!
```

`_compute_cooccurrence_pairs()` returns `(valid_pairs, method)` but does NOT return the `co_occur` matrix. The `co_occur` variable is local to that function.

**Impact:** These two functions will crash with `NameError: name 'co_occur' is not defined` at runtime. The sort-by-nnz optimization cannot execute.

**Note:** `_numba_cross_chunk()` does NOT call `_compute_cooccurrence_pairs()` -- it computes `co_occur` inline (lines 1209-1229), so it IS in scope there. No bug in the Numba path.

**Fix:** Either (a) return `co_occur` from `_compute_cooccurrence_pairs` as a 3rd element, or (b) move the nnz-sort logic INTO `_compute_cooccurrence_pairs`.

#### BUG-2: Name truncation causes silent feature drops

**Lines 1053, 1243, 1330:** Feature names are built with `[:40]` truncation:
```python
f'{prefix}_{left_names[int(p[0])][:40]}_{right_names[int(p[1])][:40]}'
```

**Lines 2379-2392:** Dedup removes features with duplicate names:
```python
seen = set()
dedup_indices = []
for i, n in enumerate(cross_names):
    s = str(n)
    if s in seen:
        continue
    seen.add(s)
    dedup_indices.append(i)
if len(dedup_indices) < len(cross_names):
    n_dups = len(cross_names) - len(dedup_indices)
    log(f"  Deduplicating {n_dups} cross names from truncation collisions")
    cross_names = [str(cross_names[i]) for i in dedup_indices]
    sparse_mat = sparse_mat[:, dedup_indices]
```

If two different cross pairs produce names that truncate to the same string, the SECOND feature is silently dropped from the matrix. The log message acknowledges this, but it is a **violation of the matrix thesis** ("ALL features must reach the model").

**Impact:** Depends on name length distribution. With 40-char truncation on each component plus a prefix, collisions are unlikely for most features but possible for long esoteric signal names like `fibonacci_time_cycle_golden_ratio_extended_...`.

**Fix:** Use a hash suffix or sequential index to guarantee uniqueness: `f'{prefix}_{left[:40]}_{right[:40]}_{i}'`.

#### Issue-3: Binarization filter on binary columns

**Line 666-667:**
```python
b = (vals > 0).astype(np.float32)
if 5 < np.nansum(b) < N * 0.98:
```

Binary/ternary columns that fire fewer than 6 times or more than 98% of rows are silently excluded from contexts. This means an ultra-rare binary signal that fires only 3-5 times will never become a context for crossing.

**Impact:** Low. These signals still appear as LEFT-side signals in their respective group crosses (esoteric, astro, etc.). They just can't serve as RIGHT-side contexts. And a context that fires <=5 times would produce crosses that almost never fire, failing the co-occurrence threshold anyway.

**Verdict: Acceptable.** Not a matrix thesis violation because the signal itself is preserved; only its role as a cross context is filtered.

#### Issue-4: Multi-valued column mask_sum threshold

**Line 696:** `if mask_sums[base + t] > 5:` -- same 5-row minimum for 4-tier binarized masks. Same analysis as Issue-3.

**Verdict: Acceptable** for same reasons.

---

## 2. EFB PRE-BUNDLER AUDIT

### 2A. Zero Features Dropped

**FINDING: PASS -- all features accounted for.**

Flow:
1. `_classify_binary_columns()` splits columns into binary (0/1 only) and non-binary
2. ALL binary columns enter tiered packing (ultra_rare, rare, moderate, common)
3. ALL non-binary columns are passed through unchanged (lines 365-369)
4. Output columns = bundles + passthrough non-binary

The mapping stats at line 273-283 track:
- `binary_features + non_binary_features == total_input_features` (always true by construction)
- `total_bundled_features` = sum of features across all bundles

All-zero columns (classified as binary) land in a bundle but contribute 0 to the encoded value. The feature slot is still reserved, and the mapping still records it. This is correct -- the feature exists but has no signal.

**Verdict: PASS.** Zero features dropped in bundling.

### 2B. Collision-Free Encoding

**FINDING: PASS for bitmap-checked tiers. KNOWN TRADE-OFF for ultra-rare tier.**

**Encoding scheme (line 227):**
```python
bundled[fire_rows, bundle_idx] += offset + 1  # offset = 2 * slot
```

Values per slot: slot 0 = 1, slot 1 = 3, slot 2 = 5, ..., slot 126 = 253.

**Bitmap-checked tiers (rare, moderate, common):**
`_pack_with_bitmap_check()` ensures no two features in the same bundle fire on the same row (line 180: set intersection check). Therefore at most ONE feature fires per row, producing exactly one of the odd values {1, 3, 5, ..., 253}. The value uniquely identifies which feature fired. **Collision-free by construction.**

**Ultra-rare tier (<0.1% density):**
`_pack_blindly()` skips collision checking. When two ultra-rare features DO fire on the same row, the additive encoding produces ambiguous sums. Example: slot 0 + slot 4 = 1 + 9 = 10, which does not correspond to any single slot value. However, it also cannot be confused with any OTHER single-slot value (all single values are odd; multi-slot sums can be even). The ambiguity is between WHICH combination of slots fired, not whether one vs. many fired.

Docstring at lines 133-139 quantifies: P(any collision in a 127-feature bundle) < 0.8% for ultra-rare features. This is a design trade-off: blind packing is much faster, and the information loss from occasional collisions on ultra-rare features is negligible.

**Verdict: PASS with caveat.** Bitmap-checked tiers are collision-free. Ultra-rare tier has documented ~0.8% collision rate, acceptable given the density constraint.

### 2C. Reversible for Feature Importance

**FINDING: PASS -- full mapping stored.**

`_build_mapping()` (lines 232-285) creates a JSON mapping with:
- `bundles[].features[].name` -- original feature name
- `bundles[].features[].original_col_idx` -- original column index
- `bundles[].features[].slot` -- position within bundle
- `bundles[].features[].offset` -- encoding offset value
- `non_binary_passthrough[]` -- non-binary features with original indices

Saved to `v2_efb_mapping_{symbol}_{tf}.json`. This is sufficient to reverse-map SHAP importance from bundle columns back to individual features. For bitmap-checked bundles (single feature per row), SHAP on the bundle value directly maps to the firing feature. For ultra-rare bundles with rare collisions, approximate reverse-mapping is possible.

**Verdict: PASS.**

### 2D. Double-Bundling with LightGBM Internal EFB

**FINDING: RISK EXISTS -- depends on training code configuration.**

The pre-bundler produces a dense integer matrix and the docstring says to use `enable_bundle=False` (line 16). Config at `config.py:365`:
```python
# enable_bundle=False in LightGBM when pre-bundled (already done externally).
EFB_PREBUNDLE_ENABLED = { '1w': True, '1d': True, '4h': True, '1h': True, '15m': True }
```

The comment says to use `enable_bundle=False`, but the actual `V3_LGBM_PARAMS` in config.py does NOT set `enable_bundle=False`. The CLAUDE.md rule says "EFB (enable_bundle) ALWAYS True for ALL timeframes."

**If the training code does not set `enable_bundle=False` when consuming pre-bundled data, LightGBM will DOUBLE-BUNDLE:** it will run its internal EFB on the already-bundled integer columns, potentially re-bundling them and destroying the encoding.

However, the pre-bundled values are NOT binary -- they are integer-encoded (0, 1, 3, 5, ...). LightGBM's internal EFB primarily bundles features that are mutually exclusive (low conflict). Pre-bundled columns are dense with many unique values, so LightGBM's EFB would likely NOT bundle them further (they'd have high conflict counts). But this is not guaranteed.

**The training code MUST explicitly set `enable_bundle=False` when using pre-bundled data.** This was not verified in this audit scope (training code not reviewed), but the config.py comment documents the intent.

**Verdict: CONDITIONAL PASS.** The pre-bundler itself is correct, but correct usage depends on the training code honoring `enable_bundle=False`. The current CLAUDE.md rule ("EFB ALWAYS True") contradicts the pre-bundler's requirement. This contradiction is documented in `EXPERT_LIGHTGBM_EFB.md` line 18 and should be resolved.

---

## 3. BUGS REQUIRING IMMEDIATE FIX

### BUG-1 (CRITICAL): NameError in _gpu_cross_chunk and _cpu_cross_chunk

**File:** `v2_cross_generator.py`
**Lines:** 1045, 1326
**Severity:** CRITICAL -- crashes at runtime
**Description:** `co_occur` variable referenced but not in scope. `_compute_cooccurrence_pairs()` returns `(valid_pairs, method_string)`, not the co-occurrence matrix.
**Impact:** Both GPU and CPU cross chunk functions crash with NameError when attempting to sort pairs by co-occurrence count. The Numba path (`_numba_cross_chunk`) is NOT affected (it computes co_occur inline).
**Fix:** Modify `_compute_cooccurrence_pairs()` to return `co_occur` as a third element, or move the sort into that function.

### BUG-2 (MEDIUM): Stale docstring claims 13 cross types

**File:** `v2_cross_generator.py`
**Line:** 21
**Description:** Docstring lists "3x regime-aware DOY crosses" but Cross 13 was removed at line 2149.
**Fix:** Update docstring to list 12 cross types.

### BUG-3 (LOW-MEDIUM): Name truncation dedup drops features

**File:** `v2_cross_generator.py`
**Lines:** 2379-2392
**Description:** Features with colliding truncated names are silently removed from the matrix. Violates "ALL features must reach the model."
**Fix:** Append unique suffix (e.g., index) to prevent name collisions.

---

## 4. CONFIGURATION CONTRADICTION

**CLAUDE.md line 47:** "EFB (enable_bundle) ALWAYS True for ALL timeframes"
**config.py line 365:** "enable_bundle=False in LightGBM when pre-bundled"
**EFB_PREBUNDLE_ENABLED:** True for all 5 TFs

These three statements are mutually contradictory. If pre-bundling is enabled for all TFs, then `enable_bundle` should be False for all TFs (to prevent double-bundling). The CLAUDE.md rule predates the pre-bundler and should be updated to: "EFB pre-bundling ALWAYS enabled. LightGBM `enable_bundle=False` when consuming pre-bundled data."

---

## 5. SUMMARY

| Area | Verdict | Notes |
|------|---------|-------|
| Cross steps (12/12) | PASS | All intended types present. Cross 13 intentionally removed. |
| Co-occurrence threshold | PASS | MIN_CO_OCCURRENCE=3 matches min_data_in_leaf. Rare signals preserved. |
| sparse-dot-mkl | PASS | Correctly configured with threadpoolctl boost. |
| Silent feature drops | **FAIL** | BUG-1 (NameError crash), BUG-3 (truncation dedup). |
| EFB zero features dropped | PASS | All features land in bundles or passthrough. |
| EFB collision-free | PASS (with caveat) | Bitmap-checked tiers = collision-free. Ultra-rare = 0.8% collision rate (documented). |
| EFB reversible | PASS | Full mapping JSON with names, indices, slots, offsets. |
| EFB double-bundling | CONDITIONAL PASS | Requires training code to set enable_bundle=False. Config contradiction exists. |

**Priority fixes:**
1. **BUG-1:** Fix `co_occur` NameError in `_gpu_cross_chunk` and `_cpu_cross_chunk` (CRASH)
2. **BUG-3:** Add unique suffix to cross feature names to prevent truncation dedup drops
3. **Config:** Resolve CLAUDE.md vs config.py contradiction on enable_bundle
