#!/usr/bin/env python
"""
gpu_cross_builder.py — GPU-accelerated cross feature generation
================================================================
Runs AFTER base feature build. Takes the parquet with base features,
generates all DOY x ALL context crosses using CuPy batch GPU multiplication.

Usage:
  python gpu_cross_builder.py --tf 1h
  python gpu_cross_builder.py --tf 1h 4h 15m 5m 1w 1d --gpu 0

What it does:
1. Loads features_{tf}.parquet (base features, no DOY crosses)
2. Creates DOY 1-365 flags if missing
3. Binarizes ALL base columns as contexts (binary direct, continuous 80/20 pct)
4. Batch GPU multiply: DOY matrix x context matrix = 135K+ crosses
5. Adds tx_doy x bull/bear/VWAP/range
6. Integrates px_sys_ from systematic_cross_results CSVs
7. Saves expanded parquet
"""
import os, sys, io, time, argparse, warnings
import numpy as np
import pandas as pd
if os.name == 'nt':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--tf', nargs='+', default=['1h'])
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

try:
    import cupy as cp
    cp.cuda.Device(args.gpu).use()
    GPU = True
    pool = cp.get_default_memory_pool()
    print(f'[GPU {args.gpu}] CuPy ready, VRAM free: {cp.cuda.runtime.memGetInfo()[0]/1e9:.1f} GB')
except Exception as e:
    GPU = False
    print(f'[CPU fallback] {e}')


def build_crosses(tf):
    t0 = time.time()
    path = f'features_{tf}.parquet'
    if not os.path.exists(path):
        print(f'  [SKIP] {path} not found')
        return

    print(f'\n{"="*80}')
    print(f'  GPU CROSS BUILDER — {tf.upper()} (GPU={GPU})')
    print(f'{"="*80}')

    df = pd.read_parquet(path)
    N = len(df)
    print(f'  Loaded: {N:,} rows x {len(df.columns):,} cols ({time.time()-t0:.1f}s)')

    # ═══ CREATE DOY FLAGS ═══
    doy_cols = [c for c in df.columns if c.startswith('doy_') and c[4:].isdigit()]
    if len(doy_cols) < 300:
        print(f'  Creating DOY 1-365 flags...')
        if 'day_of_year' in df.columns:
            doy_vals = pd.to_numeric(df['day_of_year'], errors='coerce').values
        elif hasattr(df.index, 'dayofyear'):
            doy_vals = df.index.dayofyear.values
        else:
            print(f'  [ERROR] No DOY source')
            return
        for d in range(1, 366):
            df[f'doy_{d}'] = (doy_vals == d).astype(np.int8)
        doy_cols = [f'doy_{d}' for d in range(1, 366)]
    print(f'  DOY flags: {len(doy_cols)}')

    # ═══ IDENTIFY + BINARIZE CONTEXTS ═══
    t1 = time.time()
    skip_pre = ('tx_', 'px_', 'ex_', 'dx_', 'cross_', 'next_', 'target_', 'doy_')
    skip_ex = {'timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume',
               'trades', 'taker_buy_volume', 'taker_buy_quote', 'triple_barrier_label',
               'open_time'}

    ctx_names = []
    ctx_arrays = []

    for col in df.columns:
        if col.startswith(skip_pre) or col in skip_ex:
            continue
        vals = pd.to_numeric(df[col], errors='coerce').values.astype(np.float32)
        nan_mask = np.isnan(vals)
        valid = vals[~nan_mask]
        uniq = np.unique(valid)
        if len(uniq) <= 1:
            continue
        if len(uniq) <= 3:
            b = np.where(nan_mask, np.float32(np.nan), (vals > 0).astype(np.float32))
            if np.nansum(b) > 5 and np.nansum(b) < N * 0.98:
                ctx_names.append(col)
                ctx_arrays.append(b)
        else:
            try:
                nz = valid[valid != 0] if (valid != 0).sum() > 100 else valid
                q80, q20 = np.percentile(nz, 80), np.percentile(nz, 20)
                h = np.where(nan_mask, np.float32(np.nan), (vals > q80).astype(np.float32))
                lo = np.where(nan_mask, np.float32(np.nan), (vals < q20).astype(np.float32))
                if np.nansum(h) > 5:
                    ctx_names.append(f'{col}_H')
                    ctx_arrays.append(h)
                if np.nansum(lo) > 5:
                    ctx_names.append(f'{col}_L')
                    ctx_arrays.append(lo)
            except:
                pass

    n_ctx = len(ctx_names)
    print(f'  Contexts: {n_ctx} ({time.time()-t1:.1f}s)')

    # ═══ GPU BATCH MULTIPLY: DOY x CONTEXTS ═══
    print(f'  Generating {len(doy_cols)} x {n_ctx} = {len(doy_cols)*n_ctx:,} crosses...')
    t2 = time.time()

    doy_mat = np.column_stack([df[c].values.astype(np.float32) for c in doy_cols])  # (N, 365)
    ctx_mat = np.column_stack(ctx_arrays)  # (N, n_ctx)

    n_created = 0
    BATCH = 25  # DOYs per GPU batch — balance memory vs speed

    if GPU:
        ctx_gpu = cp.asarray(ctx_mat)  # (N, n_ctx) — stays on GPU

        for b_start in range(0, len(doy_cols), BATCH):
            b_end = min(b_start + BATCH, len(doy_cols))
            doy_batch = cp.asarray(doy_mat[:, b_start:b_end])  # (N, batch)

            # GPU outer product: (N, batch, 1) * (N, 1, n_ctx) = (N, batch, n_ctx)
            crosses = doy_batch[:, :, None] * ctx_gpu[:, None, :]

            # Find non-empty crosses and store
            sums = cp.nansum(crosses, axis=0)  # (batch, n_ctx)
            nonzero_mask = cp.asnumpy(sums > 0)

            crosses_cpu = cp.asnumpy(crosses)

            for i in range(b_end - b_start):
                d = b_start + i + 1
                for j in range(n_ctx):
                    if nonzero_mask[i, j]:
                        cn = ctx_names[j][:28]
                        df[f'dx_{d}_{cn}'] = crosses_cpu[:, i, j]
                        n_created += 1

            del crosses, doy_batch, sums
            pool.free_all_blocks()

            elapsed = time.time() - t2
            pct = b_end / len(doy_cols) * 100
            rate = n_created / max(elapsed, 0.01)
            print(f'    [{pct:5.1f}%] DOY {b_start+1}-{b_end} | {n_created:,} crosses | {rate:,.0f}/sec | {elapsed:.0f}s')

        del ctx_gpu
        pool.free_all_blocks()
    else:
        # CPU vectorized fallback
        for d_idx in range(len(doy_cols)):
            d = d_idx + 1
            dv = doy_mat[:, d_idx]
            if dv.sum() == 0:
                continue
            for j in range(n_ctx):
                cross = dv * ctx_mat[:, j]
                if np.nansum(cross) > 0:
                    cn = ctx_names[j][:28]
                    df[f'dx_{d}_{cn}'] = cross
                    n_created += 1
            if (d_idx + 1) % 50 == 0:
                print(f'    [{(d_idx+1)/365*100:5.1f}%] DOY {d_idx+1}/365 | {n_created:,}')

    print(f'  DOY x ALL: {n_created:,} crosses ({time.time()-t2:.1f}s)')

    # ═══ TREND CROSSES: DOY x bull/bear/VWAP/range ═══
    t3 = time.time()
    # Bull/bear
    bull = None
    for src in ({'5m': 'h4_trend', '15m': 'h4_trend', '1h': 'd_trend', '4h': 'w_trend'}.get(tf, ''), 'ema50_rising'):
        if src and src in df.columns:
            bull = pd.to_numeric(df[src], errors='coerce').values.astype(np.float32)
            break
    if bull is not None:
        bear = 1.0 - bull
        for d in range(1, 366):
            dv = doy_mat[:, d-1]
            if dv.sum() > 0:
                df[f'tx_doy_{d}_xb'] = dv * bull
                df[f'tx_doy_{d}_xr'] = dv * bear

    # VWAP
    for vc in ('avwap_position', 'close_vs_vwap'):
        if vc in df.columns:
            v = pd.to_numeric(df[vc], errors='coerce').values.astype(np.float32)
            v_nan = np.isnan(v)
            av = np.where(v_nan, np.float32(np.nan), (v > 0).astype(np.float32))
            bv = np.where(v_nan, np.float32(np.nan), (v <= 0).astype(np.float32))
            for d in range(1, 366):
                dv = doy_mat[:, d-1]
                if dv.sum() > 0:
                    df[f'tx_doy_{d}_xav'] = dv * av
                    df[f'tx_doy_{d}_xbv'] = dv * bv
            break

    # Range
    if 'range_position' in df.columns:
        rp = pd.to_numeric(df['range_position'], errors='coerce').values.astype(np.float32)
        rp_nan = np.isnan(rp)
        rt = np.where(rp_nan, np.float32(np.nan), (rp > 0.75).astype(np.float32))
        rb = np.where(rp_nan, np.float32(np.nan), (rp < 0.25).astype(np.float32))
        for d in range(1, 366):
            dv = doy_mat[:, d-1]
            if dv.sum() > 0:
                df[f'tx_doy_{d}_xrt'] = dv * rt
                df[f'tx_doy_{d}_xrb'] = dv * rb

    tx_count = sum(1 for c in df.columns if c.startswith('tx_doy_'))
    print(f'  Trend crosses: {tx_count:,} ({time.time()-t3:.1f}s)')

    # ═══ SYSTEMATIC CROSS SURVIVORS (px_sys_) ═══
    t4 = time.time()
    px_count = 0
    for csv_f in sorted(f for f in os.listdir('.') if f.startswith('systematic_cross_results_') and f.endswith('.csv') and 'all' not in f):
        try:
            sv = pd.read_csv(csv_f)
            sv = sv.sort_values('confidence', ascending=False).drop_duplicates(subset=['signal', 'context'], keep='first')
            for _, row in sv.iterrows():
                sn, cn = row['signal'], row['context']
                fn = f'px_{sn[:18]}_{cn[:22]}'
                if fn in df.columns or sn not in df.columns:
                    continue
                sa = pd.to_numeric(df[sn], errors='coerce').values.astype(np.float32)
                # Context
                if cn.endswith('_HIGH'):
                    base = cn[:-5]
                    if base not in df.columns: continue
                    s = pd.to_numeric(df[base], errors='coerce').values.astype(np.float32)
                    s_valid = s[~np.isnan(s)]
                    s_nz = s_valid[s_valid!=0] if (s_valid!=0).sum()>10 else s_valid
                    s_nan = np.isnan(s)
                    ca = np.where(s_nan, np.float32(np.nan), (s > np.percentile(s_nz, 80)).astype(np.float32))
                elif cn.endswith('_LOW'):
                    base = cn[:-4]
                    if base not in df.columns: continue
                    s = pd.to_numeric(df[base], errors='coerce').values.astype(np.float32)
                    s_valid = s[~np.isnan(s)]
                    s_nz = s_valid[s_valid!=0] if (s_valid!=0).sum()>10 else s_valid
                    s_nan = np.isnan(s)
                    ca = np.where(s_nan, np.float32(np.nan), (s < np.percentile(s_nz, 20)).astype(np.float32))
                else:
                    if cn not in df.columns: continue
                    ca = pd.to_numeric(df[cn], errors='coerce').values.astype(np.float32)
                cross = sa * ca
                if np.nansum(cross) > 0:
                    df[fn] = cross
                    px_count += 1
        except Exception as e:
            print(f'  [WARN] {csv_f}: {e}')
    print(f'  Systematic (px_): {px_count:,} ({time.time()-t4:.1f}s)')

    # ═══ SAVE ═══
    total_cols = len(df.columns)
    print(f'\n  FINAL: {N:,} rows x {total_cols:,} cols')
    out = f'features_{tf}.parquet'
    print(f'  Saving {out}...')
    df.to_parquet(out, engine='pyarrow', compression='snappy')
    sz = os.path.getsize(out) / 1e9
    print(f'  Saved: {sz:.2f} GB | Total time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    print(f'GPU Cross Builder | TFs: {args.tf} | GPU: {args.gpu}')
    for tf in args.tf:
        build_crosses(tf)
    print('\nALL DONE!')
