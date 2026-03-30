#!/usr/bin/env python3
"""
inference_crosses.py — Compute 2.9M cross features for ONE bar in ~20ms.

At training time: save_inference_artifacts() stores thresholds + cross index pairs.
At inference time: InferenceCrossComputer loads artifacts, computes crosses per bar.

Usage:
    # Training (add to v2_cross_generator.py after cross gen):
    from inference_crosses import save_inference_artifacts
    save_inference_artifacts(ctx_names, ctx_arrays, all_cross_names, df, tf, output_dir)

    # Inference (in live_trader.py):
    from inference_crosses import InferenceCrossComputer
    xc = InferenceCrossComputer('1h')  # loads artifacts once
    full_features = xc.compute(base_feature_values)  # ~20ms per bar
"""
import os
import numpy as np
import json
import time

# Directory for inference artifacts — override with V30_DATA_DIR env var
ARTIFACT_DIR = os.environ.get("V30_DATA_DIR", os.path.dirname(os.path.abspath(__file__)))

# Cross type prefixes — must match v2_cross_generator.py gpu_batch_cross calls
CROSS_PREFIXES = ('dx_', 'ax_', 'ax2_', 'ta2_', 'ex2_', 'sw_', 'hod_',
                  'mx_', 'vx_', 'asp_', 'mn_', 'pn_')


def save_inference_artifacts(ctx_names, ctx_arrays, cross_names, df, tf, output_dir=None,
                             combo_formulas=None):
    """
    Save everything needed for inference cross computation.
    Call this ONCE after training cross generation completes.

    Saves:
    - inference_{tf}_thresholds.json: per-column binarization thresholds
    - inference_{tf}_cross_pairs.npz: (left_idx, right_idx) for each cross
    - inference_{tf}_ctx_names.json: ordered list of context signal names
    - inference_{tf}_base_cols.json: ordered list of base feature column names
    - inference_{tf}_cross_names.json: ordered cross names matching index pairs
    - inference_{tf}_combo_formulas.json: {combo_name: [comp_a, comp_b]} for I-2 fix
    """
    out = output_dir or ARTIFACT_DIR
    t0 = time.time()

    # 1. Save base column order (needed to map feature values to indices)
    base_cols = list(df.columns)
    with open(os.path.join(out, f'inference_{tf}_base_cols.json'), 'w') as f:
        json.dump(base_cols, f)

    # 2. Save context names.
    # Extend with combo context names (ax2_/ta2_ prefixes) so that cross pair
    # resolution can find them in trunc_to_original during step 4 below.
    # Combo contexts are not real df columns — they're ANDs of two parent signals —
    # so they have no threshold entry; inference_crosses.py resolves them via formulas.
    existing_ctx_set = set(ctx_names)
    extended_ctx_names = list(ctx_names)
    if combo_formulas:
        for cname in combo_formulas:
            if cname not in existing_ctx_set:
                extended_ctx_names.append(cname)
                existing_ctx_set.add(cname)

    with open(os.path.join(out, f'inference_{tf}_ctx_names.json'), 'w') as f:
        json.dump(extended_ctx_names, f)

    # 2b. Save combo formulas for inference reconstruction (I-2 fix)
    if combo_formulas:
        with open(os.path.join(out, f'inference_{tf}_combo_formulas.json'), 'w') as f:
            json.dump(combo_formulas, f, indent=2)
        print(f"  Saved {len(combo_formulas)} combo formulas")

    # 3. Save binarization thresholds
    # For each context, store: (source_col, tier, threshold_value)
    # tier: 'binary' (threshold=0), 'XH' (q95), 'H' (q75), 'L' (q25), 'XL' (q5)
    thresholds = {}
    skip_pre = ('tx_', 'px_', 'ex_', 'dx_', 'cross_', 'next_', 'target_',
                'doy_', 'ax_', 'ax2_', 'ta2_', 'ex2_', 'sw_', 'hod_',
                'mx_', 'vx_', 'pn_', 'seq_', 'roc_', 'harm_')
    skip_ex = {'timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume',
               'trades', 'taker_buy_volume', 'taker_buy_quote', 'triple_barrier_label',
               'open_time', 'open_time_ms'}

    for col in df.columns:
        if col.startswith(skip_pre) or col in skip_ex:
            continue
        vals = df[col].values.astype(np.float64)
        vals_clean = vals[~np.isnan(vals)]
        uniq = np.unique(vals_clean)
        if len(uniq) <= 1:
            continue

        if len(uniq) <= 3:
            # Binary: threshold = 0
            thresholds[col] = {'type': 'binary', 'threshold': 0.0}
        else:
            nz = vals_clean[vals_clean != 0] if np.sum(vals_clean != 0) > 100 else vals_clean
            q95, q75, q25, q5 = np.percentile(nz, [95, 75, 25, 5])
            thresholds[col] = {
                'type': '4tier',
                'q95': float(q95), 'q75': float(q75),
                'q25': float(q25), 'q5': float(q5),
            }

    with open(os.path.join(out, f'inference_{tf}_thresholds.json'), 'w') as f:
        json.dump(thresholds, f, indent=2)

    # 4. Save cross index pairs
    # For each cross feature, record which two context indices are ANDed.
    # Cross name format: "{prefix}_{left_name[:40]}_{right_name[:40]}"
    # Context names contain underscores, so we can't split on '_'.
    # Strategy: build a lookup from truncated names to indices, then for each
    # cross body try all possible split positions (at each '_') to find a
    # valid (left, right) pair. O(n_crosses * avg_underscores) ~ 20 seconds.
    # Use extended_ctx_names so combo contexts (ax2_/ta2_) are included in the lookup.
    ctx_idx = {name: i for i, name in enumerate(extended_ctx_names)}

    # Build set of truncated context names -> original name for O(1) lookup
    # Multiple originals could truncate to the same 40-char string; use first match.
    trunc_to_original = {}
    for name in extended_ctx_names:
        t = name[:40]
        if t not in trunc_to_original:
            trunc_to_original[t] = name

    left_indices = []
    right_indices = []
    valid_cross_names = []
    _unmatched = 0

    for cname in cross_names:
        # Strip the cross type prefix
        body = cname
        for pfx in CROSS_PREFIXES:
            if cname.startswith(pfx):
                body = cname[len(pfx):]
                break

        # Try every '_' position as a potential split between left and right.
        # Use longest-left-match (reverse iteration) to resolve ambiguity.
        matched = False
        # Find all underscore positions in body
        uscore_positions = [i for i, ch in enumerate(body) if ch == '_']
        # Try from rightmost underscore backwards (longest left match first)
        for split_pos in reversed(uscore_positions):
            left_part = body[:split_pos]
            right_part = body[split_pos + 1:]
            if left_part in trunc_to_original and right_part in trunc_to_original:
                left_orig = trunc_to_original[left_part]
                right_orig = trunc_to_original[right_part]
                left_indices.append(ctx_idx[left_orig])
                right_indices.append(ctx_idx[right_orig])
                valid_cross_names.append(cname)
                matched = True
                break

        if not matched:
            _unmatched += 1

    left_arr = np.array(left_indices, dtype=np.int32)
    right_arr = np.array(right_indices, dtype=np.int32)

    np.savez_compressed(
        os.path.join(out, f'inference_{tf}_cross_pairs.npz'),
        left=left_arr,
        right=right_arr,
    )

    # Save the valid cross names (matches the index pairs)
    with open(os.path.join(out, f'inference_{tf}_cross_names.json'), 'w') as f:
        json.dump(valid_cross_names, f)

    elapsed = time.time() - t0
    print(f"  Saved inference artifacts for {tf}: {len(thresholds)} thresholds, "
          f"{len(valid_cross_names)}/{len(cross_names)} cross pairs "
          f"({_unmatched} unmatched, {elapsed:.1f}s)")


class InferenceCrossComputer:
    """
    Compute cross features for a single bar at inference time.
    Load once at startup, call compute() per bar (~20ms).
    """

    def __init__(self, tf, artifact_dir=None):
        art_dir = artifact_dir or ARTIFACT_DIR
        self.tf = tf

        # Load thresholds
        with open(os.path.join(art_dir, f'inference_{tf}_thresholds.json')) as f:
            self.thresholds = json.load(f)

        # Load context names
        with open(os.path.join(art_dir, f'inference_{tf}_ctx_names.json')) as f:
            self.ctx_names = json.load(f)

        # Load base column order
        with open(os.path.join(art_dir, f'inference_{tf}_base_cols.json')) as f:
            self.base_cols = json.load(f)

        # Load cross index pairs
        pairs = np.load(os.path.join(art_dir, f'inference_{tf}_cross_pairs.npz'))
        self.left_idx = pairs['left']
        self.right_idx = pairs['right']

        # Load cross names
        with open(os.path.join(art_dir, f'inference_{tf}_cross_names.json')) as f:
            self.cross_names = json.load(f)

        # Load combo formulas (I-2 fix): {combo_name: [comp_a, comp_b]}
        # Written by save_inference_artifacts() when combo_formulas != None.
        # Absent on artifacts saved before this fix — defaults to empty dict.
        combo_path = os.path.join(art_dir, f'inference_{tf}_combo_formulas.json')
        if os.path.exists(combo_path):
            with open(combo_path) as f:
                self.combo_formulas = json.load(f)
        else:
            self.combo_formulas = {}

        # Pre-build fast ctx_name -> index lookup (used in combo resolution)
        self._ctx_name_to_idx = {name: i for i, name in enumerate(self.ctx_names)}

        # Pre-build column -> threshold mapping for fast binarization
        self._build_binarize_map()

        n_combos = sum(1 for _, comp, _ in self.ctx_producers if comp == 'combo')
        n_special = sum(1 for _, comp, _ in self.ctx_producers if comp == 'special')
        print(f"InferenceCrossComputer({tf}): {len(self.thresholds)} cols, "
              f"{len(self.ctx_names)} contexts ({n_combos} combos resolved, "
              f"{n_special} unresolvable), {len(self.cross_names)} crosses loaded")

    def _build_binarize_map(self):
        """Pre-compute the mapping from base columns to context signals."""
        # For each context name, figure out which base column + threshold produces it
        self.ctx_producers = []  # list of (col_name, comparison, threshold)

        for ctx_name in self.ctx_names:
            # Parse context name back to column + tier
            # Format: "{col}_XH", "{col}_H", "{col}_L", "{col}_XL", or "{col}" (binary)
            # DOY windows: "dw_{number}" — need day_of_year feature
            found = False

            # Check for DOY window contexts (dw_1 through dw_365)
            if ctx_name.startswith('dw_'):
                try:
                    doy_center = int(ctx_name[3:])
                    # DOY window is +-2 days around center
                    self.ctx_producers.append((ctx_name, 'doy', doy_center))
                    found = True
                except ValueError:
                    pass

            # Check for regime-aware DOY: "dw_{num}_B", "dw_{num}_R", "dw_{num}_S"
            if not found and ctx_name.startswith('dw_') and ctx_name[-2:] in ('_B', '_R', '_S'):
                try:
                    doy_center = int(ctx_name[3:-2])
                    regime_tag = ctx_name[-1]  # B, R, or S
                    self.ctx_producers.append((ctx_name, 'regime_doy', (doy_center, regime_tag)))
                    found = True
                except ValueError:
                    pass

            if not found:
                # Check 4-tier suffixes
                for suffix, comp, tier_key in [
                    ('_XH', '>', 'q95'), ('_H', '>', 'q75'),
                    ('_L', '<', 'q25'), ('_XL', '<', 'q5'),
                ]:
                    if ctx_name.endswith(suffix):
                        col = ctx_name[:-len(suffix)]
                        if col in self.thresholds and self.thresholds[col]['type'] == '4tier':
                            thresh = self.thresholds[col][tier_key]
                            self.ctx_producers.append((col, comp, thresh))
                            found = True
                            break

            if not found:
                # Binary context (no suffix, or col itself is the context)
                col = ctx_name
                if col in self.thresholds and self.thresholds[col]['type'] == 'binary':
                    self.ctx_producers.append((col, '>', 0.0))
                elif ctx_name in self.combo_formulas:
                    # I-2 fix: multi-signal combo with known formula.
                    # Store as ('combo', (comp_a, comp_b)) — resolved in compute() pass 2.
                    comp_a, comp_b = self.combo_formulas[ctx_name]
                    self.ctx_producers.append((ctx_name, 'combo', (comp_a, comp_b)))
                    found = True
                else:
                    # Unknown context — no threshold and no formula.
                    # Conservative: ctx_binary stays 0 (model sees combo didn't fire).
                    self.ctx_producers.append((ctx_name, 'special', None))

    def compute(self, base_features_dict, day_of_year=None, regime=None):
        """
        Compute cross features for a single bar.

        Args:
            base_features_dict: dict of {feature_name: value} for one bar
            day_of_year: int (1-365), extracted from bar timestamp for DOY windows
            regime: str ('bull', 'bear', 'sideways') for regime-aware DOY crosses

        Returns:
            cross_values: np.array of shape (n_crosses,), dtype float32
            elapsed_ms: float
        """
        t0 = time.time()

        # Step 1: Binarize base features into contexts (~1ms)
        ctx_binary = np.zeros(len(self.ctx_names), dtype=np.uint8)
        # I-4: track which contexts had NaN/missing inputs (NaN ≠ 0 in LightGBM)
        ctx_nan = np.zeros(len(self.ctx_names), dtype=bool)

        # Pass 1a: compute all non-combo contexts
        for i, (col, comp, thresh) in enumerate(self.ctx_producers):
            if comp in ('special', 'combo'):
                # 'special': unknown — stays 0/nan (handled after pass 1a)
                # 'combo': resolved in pass 1b below
                continue
            elif comp == 'doy':
                # DOY window: check if current day_of_year is within +-2 of center
                if day_of_year is not None:
                    center = thresh  # doy_center stored as thresh
                    # Window wraps around year boundary
                    targets = set((center + offset - 1) % 365 + 1 for offset in range(-2, 3))
                    ctx_binary[i] = 1 if day_of_year in targets else 0
                # else: day_of_year unknown — ctx_binary stays 0 (not NaN: DOY is always knowable)
            elif comp == 'regime_doy':
                # Regime-aware DOY: check DOY window AND regime match
                if day_of_year is not None and regime is not None:
                    doy_center, regime_tag = thresh
                    regime_map = {'B': 'bull', 'R': 'bear', 'S': 'sideways'}
                    targets = set((doy_center + offset - 1) % 365 + 1 for offset in range(-2, 3))
                    if day_of_year in targets and regime == regime_map.get(regime_tag):
                        ctx_binary[i] = 1
            else:
                val = base_features_dict.get(col, np.nan)
                if isinstance(val, float) and np.isnan(val):
                    # I-4: NaN input → mark context as NaN (missing), not 0 (false)
                    ctx_nan[i] = True
                    continue
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    ctx_nan[i] = True
                    continue
                if comp == '>':
                    ctx_binary[i] = 1 if val > thresh else 0
                elif comp == '<':
                    ctx_binary[i] = 1 if val < thresh else 0

        # Pass 1b: resolve combo contexts from their component signals (I-2)
        # Components are regular contexts already computed in pass 1a.
        for i, (col, comp, thresh) in enumerate(self.ctx_producers):
            if comp != 'combo':
                continue
            comp_a, comp_b = thresh
            idx_a = self._ctx_name_to_idx.get(comp_a)
            idx_b = self._ctx_name_to_idx.get(comp_b)
            if idx_a is None or idx_b is None:
                # Component signal not found in ctx_names — treat as missing
                ctx_nan[i] = True
                continue
            if ctx_nan[idx_a] or ctx_nan[idx_b]:
                # NaN propagates: if either parent is missing, combo is missing
                ctx_nan[i] = True
                continue
            ctx_binary[i] = ctx_binary[idx_a] & ctx_binary[idx_b]

        # Step 2: Compute crosses via index lookup (~16ms for 2.9M)
        crosses = (ctx_binary[self.left_idx] & ctx_binary[self.right_idx]).astype(np.float32)

        # I-4: propagate NaN to crosses where either input context had NaN input.
        # LightGBM treats NaN as missing (uses learned split direction) — correct.
        # 0 would be treated as "cross definitely didn't fire" — incorrect for NaN inputs.
        nan_cross_mask = ctx_nan[self.left_idx] | ctx_nan[self.right_idx]
        if nan_cross_mask.any():
            crosses[nan_cross_mask] = np.nan

        elapsed_ms = (time.time() - t0) * 1000
        return crosses, elapsed_ms

    def get_cross_feature_names(self):
        """Return ordered list of cross feature names (matches compute() output order)."""
        return self.cross_names

    def get_full_feature_vector(self, base_features_dict, day_of_year=None, regime=None):
        """
        Get complete feature vector (base + crosses) for LightGBM inference.

        Returns:
            features: np.array, feature_names: list, elapsed_ms: float
        """
        # Base features in correct order
        base_vals = np.array([base_features_dict.get(col, np.nan)
                              for col in self.base_cols], dtype=np.float32)

        # Cross features
        crosses, elapsed_ms = self.compute(base_features_dict, day_of_year, regime)

        # Concatenate
        full = np.concatenate([base_vals, crosses])
        return full, self.base_cols + self.cross_names, elapsed_ms


if __name__ == '__main__':
    # Quick test with synthetic data
    print("=== InferenceCrossComputer Test ===")
    print("This module requires training artifacts.")
    print("Run save_inference_artifacts() during training first.")
    print()
    print("Usage:")
    print("  # At training time:")
    print("  from inference_crosses import save_inference_artifacts")
    print("  save_inference_artifacts(ctx_names, ctx_arrays, cross_names, df, '1h')")
    print()
    print("  # At inference time:")
    print("  from inference_crosses import InferenceCrossComputer")
    print("  xc = InferenceCrossComputer('1h')")
    print("  crosses, ms = xc.compute(feature_dict, day_of_year=100)")
    print("  print(f'{len(crosses)} crosses in {ms:.1f}ms')")
