# EXPERT: LightGBM EFB at 10M+ Sparse Binary Cross-Features

**Date:** 2026-03-30
**Context:** LightGBM with 2-10M sparse binary cross-features (pure 0/1), EFB conflict graph intractable at scale, need bundling optimization. All features must be preserved, no filtering.

---

## 1. Current State in V3.3

| TF | Rows | Feature Count | EFB Status | Problem |
|----|------|---------------|------------|---------|
| 1w | 818 | ~2.2M | enable_bundle=True (default) | Works — small enough for greedy scan |
| 1d | 5,727 | ~5-6M | enable_bundle=True (default) | Slow but completes |
| 4h | 4,380 | ~3-4M | enable_bundle=True (default) | Works |
| 1h | 17,520 | ~7-8M | Should be False | **Intractable — hangs on EFB scan** |
| 15m | 227,577 | ~10M+ | Should be False | **Intractable — hangs on EFB scan** |

**Critical finding:** `config.py` does NOT set `enable_bundle` at all (LightGBM defaults to True). CLAUDE.md says "EFB ALWAYS True for ALL timeframes." But TRAINING_15M.md and TRAINING_1H.md say enable_bundle=False is MANDATORY for >1M features with >40K rows. **These contradict each other.** The training docs are correct — EFB at 10M features is intractable.

**Impact of enable_bundle=False:** Without EFB, LightGBM builds one histogram per feature. For 10M features, that means 10M histograms per tree level instead of ~78K bundles. This is the single largest training speed bottleneck for 1h/15m timeframes.

---

## 2. How EFB Works (Source Code Analysis)

### Source Path
- `src/io/dataset.cpp`: `FastFeatureBundling()` / `FindGroups()` / `GetConflictCount()`
- `include/LightGBM/feature_group.h`: bundled feature representation, bin offsets
- `include/LightGBM/bin.h`: `BinMapper`, `Bin`, `MultiValBin`

### Algorithm (GreedyBundling)
1. Compute per-feature nonzero counts
2. Sort features by nnz descending (proxy for graph degree)
3. For each feature, scan all existing groups to find candidates
4. Test up to 100 candidate groups via `GetConflictCount()` (row-mark bitmap intersection)
5. Accept first group where conflict count < threshold and total bins < max_bin_per_group
6. If no group fits, create new group

### Complexity
- **Sorting:** O(F log F)
- **Per-feature scan:** O(G) where G = current group count
- **Conflict test:** O(k_f) per candidate (k_f = nnz of feature f), up to 100 candidates
- **Total:** O(F log F + sum_f(G_f + 100 * k_f))
- **Worst case:** O(F^2) when bundling fails and G grows with F
- **Memory:** O(G * S) bits for per-group conflict marks (S = total_sample_cnt)

### No Binary Fast Path
There is **no special-case code** in LightGBM for `max_bin=2` or binary features. The EFB algorithm treats all features identically regardless of bin count. This is a missed optimization opportunity.

---

## 3. Binary Feature Bundle Capacity

For pure binary (0/1) features with `max_bin=255`:
- Each binary feature consumes **2 bins** in the bundle (bin 0 = off, bin 1 = on)
- EFB uses offset-based encoding: feature k in bundle maps to bins [2k, 2k+1]
- **Maximum per bundle:** floor(255 / 2) = **127 binary features**
- **Theoretical minimum bundles for 10M features:** ceil(10M / 127) = **78,741 bundles**

This is a **128x reduction** from 10M histograms to ~79K histograms per tree level.

---

## 4. Why the Conflict Graph is Simpler for Binary Features

For general EFB, two features conflict if they are both nonzero on the same row. For pure binary features:
- Each feature is a **sparse set of row IDs** where the feature = 1
- Conflict = set intersection of two row-ID sets
- Two features are **exclusive** (can bundle) when their row-sets are disjoint
- Binary features are typically very sparse (~0.3% density for cross-features)

**Key insight:** At 99.7% sparsity, most feature pairs have zero overlap. The conflict graph is extremely sparse — most edges are absent. This makes the bundling problem much easier than the general case, but LightGBM's code doesn't exploit this.

---

## 5. The Solution: External Pre-Bundling

Since LightGBM has no binary-optimized EFB path, the correct approach is to **pre-bundle features externally** before passing to LightGBM with `enable_bundle=False`.

### Algorithm: Greedy Sparse Bitmap Bundler

```
Input: CSC sparse matrix (10M columns, binary 0/1)
Output: Bundled integer matrix (~79K columns)

1. Extract per-column nonzero row indices from CSC format
2. Sort columns by nnz DESCENDING (dense features first = harder to pack)
3. Initialize empty bundle list

For each column c (in sorted order):
    row_set_c = set of rows where column c = 1

    For each existing bundle b:
        if bundle_b.member_count >= 127:
            skip  # bundle full
        if row_set_c INTERSECT bundle_b.occupied_rows == EMPTY:
            # No conflict — add to this bundle
            bundle_b.add(c, row_set_c)
            goto next_column

    # No compatible bundle found — create new one
    new_bundle(c, row_set_c)

4. Encode: For each bundle, create integer column where:
   - 0 = no feature active in this bundle for this row
   - k = feature k is active (k = 1..127)

5. Output as dense integer matrix (79K columns, max_bin=255)
```

### Optimizations for Scale

| Optimization | Technique | Speedup |
|---|---|---|
| Row-set storage | Roaring bitmaps or compressed bitsets | 10-50x vs set() |
| Candidate search | Only test bundles with compatible row-block signatures | 100x fewer comparisons |
| Parallelism | Partition features into frequency tiers, bundle within tiers | Linear with cores |
| Early termination | If bundle has < threshold occupied rows, always accept sparse features | Skip intersection test |
| Namespace grouping | Bundle within feature families (same left/right source) first | Exploits structural exclusivity |

### Expected Performance

| Metric | Without EFB (current 1h/15m) | With Pre-Bundling |
|---|---|---|
| Features at training | 10M | ~79K bundles |
| Histogram memory per level | 10M * 16B = 160MB | 79K * 16B = 1.3MB |
| Histogram build time | O(nnz * 10M bins) | O(nnz * 79K bins) |
| Training speedup | 1x (baseline) | **~128x** |

---

## 6. Alternative: Namespace Categorical Encoding

Since cross-features come from structured templates (e.g., `dx_featureA_x_featureB`), an alternative to EFB is to reconstruct the original categorical interaction:

- Group all crosses that share the same LEFT source feature
- Encode as one categorical column: "which RIGHT feature was active?"
- LightGBM handles categorical splits in O(k log k) where k = categories

**Example:**
- `dx_rsi_14_high_x_macd_bullish`, `dx_rsi_14_high_x_bb_squeeze`, `dx_rsi_14_high_x_moon_full`
- Become one categorical: `dx_rsi_14_high_cross` with values {none, macd_bullish, bb_squeeze, moon_full}

**Advantage:** Perfect information preservation, O(1) per original feature.
**Disadvantage:** Requires cross-gen metadata to reconstruct groupings. Multi-hot rows (where multiple crosses fire for the same left source) need special handling.

---

## 7. Alternative: Density-Tier Bundling (No Conflict Graph)

For binary features at 99.7% sparsity, a simpler approach that avoids ANY conflict graph:

1. **Sort features by density** (nnz / n_rows)
2. **Tier 1 (ultra-rare, nnz < 10):** Pack 127 per bundle blindly — collision probability < 0.001%
3. **Tier 2 (rare, nnz 10-100):** Pack with quick bitmap check
4. **Tier 3 (moderate, nnz 100-1000):** Full bitmap intersection test
5. **Tier 4 (dense, nnz > 1000):** These are base features, not cross-features. Keep separate.

At 99.7% sparsity, most features are Tier 1-2 where collision probability is negligible. This makes bundling essentially O(F) instead of O(F * G).

---

## 8. Recommended Implementation Plan

### Phase 1: Immediate Fix (enable_bundle=False for 1h/15m)
Add to `config.py`:
```python
# TFs that disable EFB (conflict graph intractable at high feature counts)
TF_DISABLE_EFB = frozenset(['1h', '15m'])
```
And in training code, set `enable_bundle=False` when TF in TF_DISABLE_EFB.

**This is already documented as needed but NOT implemented in code.**

### Phase 2: External Pre-Bundler (offline, before training)
Build `prebundle_binary.py` that:
1. Loads the CSR cross-feature matrix
2. Runs greedy sparse bitmap bundling (density-tier approach)
3. Outputs bundled integer matrix + mapping file
4. Training code loads pre-bundled matrix with `enable_bundle=False`

**Expected reduction:** 10M features -> ~79K bundles -> **128x fewer histograms**

### Phase 3: LightGBM Fork Enhancement (optional)
Add binary-feature fast path to `FastFeatureBundling()` in the GPU fork:
- Detect when all features have max_bin=2
- Use roaring bitmap conflict detection instead of row-mark vectors
- Skip the 100-candidate cap (bitmap intersection is cheap)

---

## 9. Key Parameters After Bundling

```python
# After external pre-bundling:
params = {
    "enable_bundle": False,       # Already bundled externally
    "is_enable_sparse": False,    # Bundled matrix is dense integer
    "force_col_wise": True,       # Still many columns (~79K)
    "max_bin": 255,               # Matches bundle encoding (127 features * 2 bins)
    "feature_pre_filter": False,  # NEVER filter — all bundles carry information
    "min_data_in_bin": 1,         # Rare signals preserved within bundles
}
```

---

## 10. Risk Assessment

| Risk | Mitigation |
|---|---|
| Bundling introduces aliasing (two features in same bundle, same row) | Use exact (zero-conflict) bundling only. Verify intersection = 0. |
| Bundle encoding loses feature identity for SHAP | Maintain bundle->feature mapping file. Unpack SHAP values post-hoc. |
| Pre-bundling adds pipeline step | Cache bundles per TF. Only rebuild when cross-gen changes. |
| Different bundle assignments per CPCV fold | Bundle on FULL dataset before splitting. Bundles are data-independent (structural). |
| max_bin=255 limits to 127 features/bundle | Sufficient. Could use max_bin=511 for 255/bundle if needed (LightGBM supports up to 2^31). |

---

## 11. Summary

| Question | Answer |
|---|---|
| Is the conflict graph simpler for binary features? | **Yes.** Conflict = set intersection of sparse row-ID lists. At 99.7% sparsity, most pairs are non-conflicting. |
| Can we pre-compute bundles offline? | **Yes.** Features are static binary — bundle once, reuse across folds and Optuna trials. |
| Can we bypass the conflict graph entirely? | **Yes.** Density-tier bundling: ultra-rare features can be packed blindly (collision prob < 0.001%). |
| Time complexity for binary-only vs mixed EFB? | **Binary:** O(F) with density-tier. **Mixed (LightGBM internal):** O(F^2) worst case. |
| enable_bundle=False impact? | **10M histograms vs ~79K bundles = 128x more work per tree level.** This is the #1 training bottleneck for 1h/15m. |
| Can we fix this? | **Yes.** External pre-bundler reduces 10M features to ~79K bundles, then train with enable_bundle=False on the bundled representation. |

---

## Sources
- LightGBM NeurIPS Paper: "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" (Ke et al., 2017)
- LightGBM source: `src/io/dataset.cpp` (`FastFeatureBundling`, `FindGroups`, `GetConflictCount`)
- LightGBM source: `include/LightGBM/feature_group.h` (bundle representation)
- LightGBM Parameters docs: `enable_bundle`, `is_enable_sparse`, `force_col_wise`, `max_bin`
- ApX ML: "LightGBM's Exclusive Feature Bundling"
- Towards AI: "Understanding LightGBM: GOSS and EFB"
