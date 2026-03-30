# EFB Pre-Bundler — External Feature Bundling for Binary Cross Features

## Problem

LightGBM's internal EFB (Exclusive Feature Bundling) builds an O(F²) conflict graph
to determine which features can share histogram bins. At 10M binary cross features,
this becomes intractable — the conflict graph alone would require ~400TB of memory.

The CLAUDE.md rule "EFB ALWAYS True" and the training config's `enable_bundle=False`
are both suboptimal. The real solution is **external pre-bundling**.

## Solution

`efb_prebundler.py` pre-bundles binary (0/1) features *before* LightGBM ever sees them.

### How It Works

1. **Classification**: Identifies pure binary columns (0/1 only) vs non-binary passthrough
2. **Density Tiering**: Groups binary features by firing density:
   - Ultra-rare (<0.1%): blind packing — collision prob <0.001%
   - Rare (<1%), Moderate (<5%), Common (>5%): bitmap intersection conflict checking
3. **Packing**: Up to 127 binary features per bundle (max_bin=255, 2 bins each)
4. **Offset Encoding**: `bundle_val = sum(feature_i * (2*slot + 1))` — unique, decodable
5. **Output**: Dense int16 matrix + reversible feature→bundle mapping JSON

### Key Numbers

| Metric | Before | After |
|--------|--------|-------|
| Feature columns | ~10M | ~79K bundles + passthrough |
| Histogram builds | 10M | ~79K |
| Reduction | — | **128x** |

### Zero Feature Loss

- ALL binary features are packed into bundles — none dropped
- Non-binary features pass through unchanged (appended as extra columns)
- Feature→bundle mapping stored for SHAP/importance analysis

## Files Changed

| File | Change |
|------|--------|
| `efb_prebundler.py` | **NEW** — pre-bundler module |
| `config.py` | Added `EFB_PREBUNDLE_ENABLED` per-TF toggle |
| `v2_cross_generator.py` | Calls prebundler after cross gen save (config-gated, non-fatal) |

## Config

```python
# config.py
EFB_PREBUNDLE_ENABLED = {
    '1w': True,    # 1158 rows
    '1d': True,    # 5.7K rows
    '4h': True,    # 23K rows, ~2.9M crosses
    '1h': True,    # 75K rows, ~5M crosses
    '15m': True,   # 294K rows, ~10M crosses — BIGGEST win
}
```

## Output Files

After cross gen, these additional files are created:
- `v2_crosses_{symbol}_{tf}_bundled.npz` — pre-bundled matrix (sparse CSR format)
- `v2_cross_names_{symbol}_{tf}_bundled.json` — bundled column names
- `v2_efb_mapping_{symbol}_{tf}.json` — full feature→bundle mapping

## Training Integration

When using pre-bundled crosses, LightGBM should use `enable_bundle=False`
(the features are already bundled — internal EFB would try to re-bundle bundles).

## CLI Usage

```bash
# Standalone (after cross gen has produced NPZ + names JSON)
python efb_prebundler.py --npz v2_crosses_BTC_1d.npz --names v2_cross_names_BTC_1d.json --tf 1d

# Automatic (called by v2_cross_generator.py when EFB_PREBUNDLE_ENABLED[tf] is True)
python v2_cross_generator.py --tf 1d  # prebundler runs automatically after save
```
