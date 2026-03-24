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

# Directory for inference artifacts
ARTIFACT_DIR = os.path.dirname(os.path.abspath(__file__))


def save_inference_artifacts(ctx_names, ctx_arrays, cross_names, df, tf, output_dir=None):
    """
    Save everything needed for inference cross computation.
    Call this ONCE after training cross generation completes.

    Saves:
    - inference_{tf}_thresholds.json: per-column binarization thresholds
    - inference_{tf}_cross_pairs.npz: (left_idx, right_idx) for each cross
    - inference_{tf}_ctx_names.json: ordered list of context signal names
    - inference_{tf}_base_cols.json: ordered list of base feature column names
    """
    out = output_dir or ARTIFACT_DIR
    t0 = time.time()

    # 1. Save base column order (needed to map feature values to indices)
    base_cols = list(df.columns)
    with open(os.path.join(out, f'inference_{tf}_base_cols.json'), 'w') as f:
        json.dump(base_cols, f)

    # 2. Save context names (ordered, matches ctx_arrays)
    with open(os.path.join(out, f'inference_{tf}_ctx_names.json'), 'w') as f:
        json.dump(ctx_names, f)

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
    # For each cross feature, record which two context indices are ANDed
    # Build context name → index mapping
    ctx_idx = {name: i for i, name in enumerate(ctx_names)}

    # Parse cross names to extract left/right context
    # Cross name format: "{prefix}_{left_ctx}__x__{right_ctx}" or similar
    # We need to map each cross to (left_ctx_idx, right_ctx_idx)
    left_indices = []
    right_indices = []
    valid_cross_names = []

    for cname in cross_names:
        # Try to find the two context names in the cross name
        # Cross names are like "dx_doy_100__x__rsi_14_H" or "ax_mercury_retro__x__bb_pctb_XH"
        parts = cname.split('__x__')
        if len(parts) == 2:
            # Remove prefix (dx_, ax_, etc.)
            left_part = parts[0]
            right_part = parts[1]
            # Strip the cross type prefix
            for prefix in ['dx_', 'ax_', 'ax2_', 'ta2_', 'ex2_', 'mx_', 'vx_',
                           'asp_', 'mn_', 'pn_', 'hod_', 'rdx_']:
                if left_part.startswith(prefix):
                    left_part = left_part[len(prefix):]
                    break

            if left_part in ctx_idx and right_part in ctx_idx:
                left_indices.append(ctx_idx[left_part])
                right_indices.append(ctx_idx[right_part])
                valid_cross_names.append(cname)

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
          f"{len(valid_cross_names)}/{len(cross_names)} cross pairs ({elapsed:.1f}s)")


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

        # Pre-build column → threshold mapping for fast binarization
        self._build_binarize_map()

        print(f"InferenceCrossComputer({tf}): {len(self.thresholds)} cols, "
              f"{len(self.ctx_names)} contexts, {len(self.cross_names)} crosses loaded")

    def _build_binarize_map(self):
        """Pre-compute the mapping from base columns to context signals."""
        # For each context name, figure out which base column + threshold produces it
        self.ctx_producers = []  # list of (col_name, comparison, threshold)

        for ctx_name in self.ctx_names:
            # Parse context name back to column + tier
            # Format: "{col}_XH", "{col}_H", "{col}_L", "{col}_XL", or "{col}" (binary)
            found = False
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
                else:
                    # DOY window or special context — handle separately
                    self.ctx_producers.append((ctx_name, 'special', None))

    def compute(self, base_features_dict):
        """
        Compute cross features for a single bar.

        Args:
            base_features_dict: dict of {feature_name: value} for one bar

        Returns:
            cross_values: np.array of shape (n_crosses,), dtype float32
        """
        t0 = time.time()

        # Step 1: Binarize base features into contexts (~1ms)
        ctx_binary = np.zeros(len(self.ctx_names), dtype=np.uint8)
        for i, (col, comp, thresh) in enumerate(self.ctx_producers):
            if comp == 'special':
                continue  # DOY windows handled separately
            val = base_features_dict.get(col, np.nan)
            if np.isnan(val):
                continue
            if comp == '>':
                ctx_binary[i] = 1 if val > thresh else 0
            elif comp == '<':
                ctx_binary[i] = 1 if val < thresh else 0

        # Step 2: Compute crosses via index lookup (~16ms for 2.9M)
        crosses = (ctx_binary[self.left_idx] & ctx_binary[self.right_idx]).astype(np.float32)

        elapsed_ms = (time.time() - t0) * 1000
        return crosses, elapsed_ms

    def get_full_feature_vector(self, base_features_dict):
        """
        Get complete feature vector (base + crosses) for LightGBM inference.

        Returns:
            features: np.array, cross_names: list
        """
        # Base features in correct order
        base_vals = np.array([base_features_dict.get(col, np.nan)
                              for col in self.base_cols], dtype=np.float32)

        # Cross features
        crosses, elapsed_ms = self.compute(base_features_dict)

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
    print("  crosses, ms = xc.compute(feature_dict)")
    print("  print(f'{len(crosses)} crosses in {ms:.1f}ms')")
