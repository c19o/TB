"""
Post-rebuild verification: ensures no features were silently dropped.

Checks:
1. tx_ columns exist and have non-zero values
2. Critical dependencies (ema50_rising, etc.) present
3. Cross-TF consistency
4. Minimum feature count

Usage:
    python verify_rebuild.py
"""
import sqlite3
import os
import sys

TFS = ['15m', '1h', '4h', '1d', '1w']
MIN_FEATURES = 850  # expected minimum per TF

# tx_ columns expected ONLY for sub-daily TFs (session features)
SUB_DAILY_ONLY = {'tx_hour_4_x_bull', 'tx_hour_4_x_bear',
                  'tx_asia_session_x_bull', 'tx_asia_session_x_bear',
                  'tx_london_session_x_bull', 'tx_london_session_x_bear',
                  'tx_ny_session_x_bull', 'tx_ny_session_x_bear'}


def verify_db(tf, db_path):
    """Verify a single feature DB. Returns (ok, warnings, errors)."""
    warnings = []
    errors = []

    if not os.path.exists(db_path):
        return False, [], [f'{tf}: DB not found at {db_path}']

    conn = sqlite3.connect(db_path)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    if not tables:
        conn.close()
        return False, [], [f'{tf}: No tables in DB']

    table = tables[0]
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    tx_cols = sorted([c for c in cols if c.startswith('tx_')])

    # Check minimum feature count
    if len(cols) < MIN_FEATURES:
        errors.append(f'{tf}: Only {len(cols)} features (expected >= {MIN_FEATURES})')

    # Check critical dependencies
    for critical in ['ema50_rising', 'ema50_declining', 'ema50_slope', 'close']:
        if critical not in cols:
            errors.append(f'{tf}: MISSING critical column: {critical}')

    # Check HTF trend columns for sub-daily
    if tf == '15m' and 'h4_trend' not in cols:
        warnings.append(f'{tf}: Missing h4_trend (HTF regime will fall back to same-TF)')
    if tf == '1h' and 'd_trend' not in cols:
        warnings.append(f'{tf}: Missing d_trend (HTF regime will fall back to same-TF)')
    if tf == '4h' and 'w_trend' not in cols:
        warnings.append(f'{tf}: Missing w_trend (HTF regime will fall back to same-TF)')

    # Check tx_ columns have non-zero values
    all_zero_tx = []
    for tx in tx_cols:
        try:
            result = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE [{tx}] != 0 AND [{tx}] IS NOT NULL"
            ).fetchone()[0]
            if result == 0:
                all_zero_tx.append(tx)
        except Exception as e:
            warnings.append(f'{tf}: Error checking {tx}: {e}')

    if all_zero_tx:
        warnings.append(f'{tf}: {len(all_zero_tx)} tx_ columns are ALL ZERO: '
                        f'{", ".join(all_zero_tx[:10])}{"..." if len(all_zero_tx) > 10 else ""}')

    conn.close()

    ok = len(errors) == 0
    return ok, warnings, errors, len(cols), len(tx_cols), row_count


def verify_all():
    """Verify all 6 TF databases."""
    print(f"\n{'='*70}")
    print(f"POST-REBUILD VERIFICATION")
    print(f"{'='*70}\n")

    all_ok = True
    all_tx = {}  # tf -> set of tx_ columns
    results = []

    for tf in TFS:
        db_path = f'features_{tf}.db'
        result = verify_db(tf, db_path)

        if len(result) == 3:
            ok, warnings, errors = result
            total = tx_count = rows = 0
        else:
            ok, warnings, errors, total, tx_count, rows = result

        all_tx[tf] = set()
        if ok or total > 0:
            conn = sqlite3.connect(db_path)
            table = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()][0]
            all_tx[tf] = {c for c in
                [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
                if c.startswith('tx_')}
            conn.close()

        status = 'OK' if ok else 'FAIL'
        warn_str = f' ({len(warnings)} warnings)' if warnings else ''
        print(f"  {tf:>3s}: {total:4d} features, {tx_count:3d} tx_, {rows:>7,d} rows — {status}{warn_str}")

        for e in errors:
            print(f"       ERROR: {e}")
        for w in warnings:
            print(f"       WARN:  {w}")

        if not ok:
            all_ok = False
        results.append((tf, ok, total, tx_count, rows))

    # Cross-TF consistency check
    print(f"\n{'='*70}")
    print(f"CROSS-TF CONSISTENCY")
    print(f"{'='*70}\n")

    # Use 1H as reference (most complete)
    ref_tf = '1h'
    ref_tx = all_tx.get(ref_tf, set())

    for tf in TFS:
        if tf == ref_tf:
            continue
        tf_tx = all_tx.get(tf, set())
        in_ref_not_tf = ref_tx - tf_tx - SUB_DAILY_ONLY
        in_tf_not_ref = tf_tx - ref_tx - SUB_DAILY_ONLY

        if in_ref_not_tf:
            # Filter out HTF-specific columns that won't exist for daily/weekly
            if tf in ('1d', '1w'):
                # These TFs don't have session features — that's expected
                in_ref_not_tf = {c for c in in_ref_not_tf
                                 if not any(s in c for s in ['session', 'hour_4'])}
            if in_ref_not_tf:
                print(f"  {tf}: Missing {len(in_ref_not_tf)} tx_ columns that 1H has:")
                for c in sorted(in_ref_not_tf)[:5]:
                    print(f"       - {c}")
                if len(in_ref_not_tf) > 5:
                    print(f"       ... and {len(in_ref_not_tf) - 5} more")

        if in_tf_not_ref:
            print(f"  {tf}: Has {len(in_tf_not_ref)} EXTRA tx_ columns not in 1H:")
            for c in sorted(in_tf_not_ref)[:5]:
                print(f"       + {c}")

    if not any(in_ref_not_tf for tf in TFS if tf != ref_tf):
        print(f"  All TFs consistent with 1H reference (accounting for TF-specific features)")

    # Summary
    print(f"\n{'='*70}")
    if all_ok:
        print(f"VERIFICATION PASSED — all {len(TFS)} DBs OK")
    else:
        failed = [tf for tf, ok, *_ in results if not ok]
        print(f"VERIFICATION FAILED — {len(failed)} DBs have errors: {', '.join(failed)}")
    print(f"{'='*70}\n")

    return all_ok


if __name__ == '__main__':
    ok = verify_all()
    sys.exit(0 if ok else 1)
