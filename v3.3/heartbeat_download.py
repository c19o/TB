"""
HEARTBEAT — GCP Data Downloader + BTC Price Fetcher + Correlation Analysis
Downloads Global Consciousness Project RNG data and correlates with BTC price action.
"""
import os
import sys
import gzip
import csv
import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
import math
import threading

_print_lock = threading.Lock()
_download_counter = {"done": 0, "fail": 0, "skip": 0}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heartbeat_data")
GCP_DIR = os.path.join(DATA_DIR, "gcp")
BTC_DIR = os.path.join(DATA_DIR, "btc")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

os.makedirs(GCP_DIR, exist_ok=True)
os.makedirs(BTC_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# STEP 1: Download GCP daily CSV files
# ============================================================

def download_gcp_day(date_str):
    """Download one day of GCP basket data. date_str = 'YYYY-MM-DD'"""
    year = date_str[:4]
    filename = f"basketdata-{date_str}.csv.gz"
    local_gz = os.path.join(GCP_DIR, f"tmp_{date_str}.csv.gz")
    local_csv = os.path.join(GCP_DIR, f"basketdata-{date_str}.csv")

    if os.path.exists(local_csv):
        _download_counter["skip"] += 1
        return local_csv  # already have it

    url = f"https://global-mind.org/data/eggsummary/{year}/{filename}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()

        with open(local_gz, "wb") as f:
            f.write(data)

        # Decompress
        with gzip.open(local_gz, "rb") as gz:
            csv_data = gz.read()
        with open(local_csv, "wb") as f:
            f.write(csv_data)

        try:
            os.remove(local_gz)
        except OSError:
            pass

        _download_counter["done"] += 1
        total = _download_counter["done"] + _download_counter["fail"]
        if total % 20 == 0:
            with _print_lock:
                print(f"  Downloaded {_download_counter['done']} | "
                      f"Failed {_download_counter['fail']} | "
                      f"Skipped {_download_counter['skip']} | "
                      f"Latest: {date_str}", flush=True)
        return local_csv

    except urllib.error.HTTPError as e:
        _download_counter["fail"] += 1
        return None
    except Exception as e:
        _download_counter["fail"] += 1
        return None


def download_gcp_range(start_date, end_date, max_workers=20):
    """Download GCP data for a date range using parallel threads."""
    # Build list of all dates
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    # Reset counters
    _download_counter["done"] = 0
    _download_counter["fail"] = 0
    _download_counter["skip"] = 0

    print(f"  Total dates to process: {len(dates)} | Workers: {max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_gcp_day, d): d for d in dates}
        for future in as_completed(futures):
            pass  # results handled inside download_gcp_day

    total_new = _download_counter["done"]
    total_skip = _download_counter["skip"]
    total_fail = _download_counter["fail"]
    print(f"\n  GCP download complete: {total_new} new + {total_skip} cached = "
          f"{total_new + total_skip} total | {total_fail} failed")
    return total_new + total_skip


# ============================================================
# STEP 2: Parse GCP CSV into hourly deviation stats
# ============================================================

def parse_gcp_csv(filepath):
    """
    Parse a GCP basket CSV file.
    Returns list of (unix_timestamp, [egg_values]) per second.
    """
    records = []
    trial_size = 200  # default

    try:
        with open(filepath, "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if not parts:
                    continue

                rec_type = parts[0].strip()

                if rec_type == "10":
                    # Protocol: samples_per_rec, sec_per_rec, rec_per_pkt, trial_size
                    if len(parts) >= 5:
                        try:
                            trial_size = int(parts[4].strip())
                        except ValueError:
                            pass

                elif rec_type == "13":
                    # Data record: type, unix_time, civil_time, egg1, egg2, ...
                    if len(parts) >= 4:
                        try:
                            unix_time = float(parts[1].strip())
                            egg_values = []
                            for v in parts[3:]:  # skip type, time, civil_time
                                v = v.strip()
                                if v:
                                    egg_values.append(int(v))
                            if egg_values:
                                records.append((unix_time, egg_values, trial_size))
                        except (ValueError, IndexError):
                            pass

    except Exception as e:
        print(f"  Error parsing {filepath}: {e}")

    return records


def compute_hourly_deviation(records):
    """
    Compute hourly deviation statistics from per-second GCP records.

    For each egg, each trial of N bits should sum to ~N/2 (100 for 200-bit trials).
    The Z-score measures how far the actual sum deviates from expected.

    Returns dict of {hour_timestamp: {mean_z, max_z, variance, n_eggs, n_samples}}
    """
    hourly = defaultdict(list)

    for unix_time, egg_values, trial_size in records:
        # Round down to hour
        hour_ts = int(unix_time) - (int(unix_time) % 3600)

        expected = trial_size / 2.0
        std_dev = math.sqrt(trial_size / 4.0)  # std of binomial(N, 0.5)

        z_scores = []
        for val in egg_values:
            z = (val - expected) / std_dev
            z_scores.append(z)

        if z_scores:
            # Network variance: sum of squared z-scores / n_eggs
            # Under null hypothesis, this should be ~1.0
            net_var = sum(z**2 for z in z_scores) / len(z_scores)
            hourly[hour_ts].append(net_var)

    # Aggregate per hour
    result = {}
    for hour_ts, variances in hourly.items():
        if len(variances) < 100:  # need reasonable sample
            continue
        mean_var = statistics.mean(variances)
        max_var = max(variances)
        result[hour_ts] = {
            "mean_var": round(mean_var, 6),
            "max_var": round(max_var, 4),
            "std_var": round(statistics.stdev(variances), 6) if len(variances) > 1 else 0,
            "n_samples": len(variances),
            "deviation": round(mean_var - 1.0, 6),  # positive = above chance
        }

    return result


# ============================================================
# STEP 3: Download BTC price data (hourly candles)
# ============================================================

def load_btc_hourly(start_ts, end_ts):
    """Load hourly BTC candles from local btc_prices.db."""
    import sqlite3

    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "btc_prices.db")
    print(f"  Loading BTC 1h data from {db_path}...")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # open_time is in milliseconds
    start_ms = start_ts * 1000
    end_ms = end_ts * 1000

    cur.execute("""
        SELECT open_time, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol = 'BTC/USDT' AND timeframe = '1h'
        AND open_time >= ? AND open_time <= ?
        ORDER BY open_time
    """, (start_ms, end_ms))

    rows = cur.fetchall()
    conn.close()

    candles = []
    for r in rows:
        candles.append({
            "ts": r[0] // 1000,  # convert ms to seconds
            "open": r[1],
            "high": r[2],
            "low": r[3],
            "close": r[4],
            "volume": r[5],
        })

    print(f"  BTC loaded: {len(candles)} hourly candles")
    return candles


# ============================================================
# STEP 4: Correlation Analysis
# ============================================================

def correlate(gcp_hourly, btc_candles):
    """
    Cross-correlate GCP deviation with BTC price action.
    """
    # Index BTC by hour timestamp
    btc_by_hour = {}
    for c in btc_candles:
        hour_ts = c["ts"] - (c["ts"] % 3600)
        btc_by_hour[hour_ts] = c

    # Build paired dataset
    paired = []
    for hour_ts, gcp in sorted(gcp_hourly.items()):
        if hour_ts in btc_by_hour:
            btc = btc_by_hour[hour_ts]
            pct_change = ((btc["close"] - btc["open"]) / btc["open"]) * 100
            abs_change = abs(pct_change)
            paired.append({
                "ts": hour_ts,
                "dt": datetime.utcfromtimestamp(hour_ts).strftime("%Y-%m-%d %H:00"),
                "gcp_deviation": gcp["deviation"],
                "gcp_mean_var": gcp["mean_var"],
                "gcp_max_var": gcp["max_var"],
                "btc_pct_change": round(pct_change, 4),
                "btc_abs_change": round(abs_change, 4),
                "btc_volume": btc["volume"],
                "btc_range": round(((btc["high"] - btc["low"]) / btc["open"]) * 100, 4),
            })

    if len(paired) < 50:
        print(f"\n  Only {len(paired)} paired hours — not enough data for correlation")
        return paired

    print(f"\n  Paired dataset: {len(paired)} hours with both GCP + BTC data")

    # Pearson correlation
    def pearson(x_vals, y_vals):
        n = len(x_vals)
        if n < 10:
            return 0, 0
        mean_x = sum(x_vals) / n
        mean_y = sum(y_vals) / n
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals)) / n
        std_x = math.sqrt(sum((x - mean_x)**2 for x in x_vals) / n)
        std_y = math.sqrt(sum((y - mean_y)**2 for y in y_vals) / n)
        if std_x == 0 or std_y == 0:
            return 0, 0
        r = cov / (std_x * std_y)
        # t-test for significance
        if abs(r) >= 1:
            t_stat = float("inf")
        else:
            t_stat = r * math.sqrt((n - 2) / (1 - r**2))
        return round(r, 6), round(t_stat, 4)

    gcp_dev = [p["gcp_deviation"] for p in paired]
    gcp_max = [p["gcp_max_var"] for p in paired]
    btc_pct = [p["btc_pct_change"] for p in paired]
    btc_abs = [p["btc_abs_change"] for p in paired]
    btc_vol = [p["btc_volume"] for p in paired]
    btc_range = [p["btc_range"] for p in paired]

    print("\n  ===========================================================")
    print("  CORRELATION RESULTS: GCP Deviation vs BTC Price Action")
    print("  ===========================================================")

    tests = [
        ("GCP deviation  vs  BTC % change (direction)", gcp_dev, btc_pct),
        ("GCP deviation  vs  BTC |% change| (volatility)", gcp_dev, btc_abs),
        ("GCP deviation  vs  BTC volume", gcp_dev, btc_vol),
        ("GCP deviation  vs  BTC range (high-low %)", gcp_dev, btc_range),
        ("GCP max_var    vs  BTC |% change| (volatility)", gcp_max, btc_abs),
        ("GCP max_var    vs  BTC volume", gcp_max, btc_vol),
        ("GCP max_var    vs  BTC range (high-low %)", gcp_max, btc_range),
    ]

    for label, x, y in tests:
        r, t = pearson(x, y)
        sig = "***" if abs(t) > 3.29 else "**" if abs(t) > 2.58 else "*" if abs(t) > 1.96 else ""
        print(f"  {label}")
        print(f"    r = {r:+.6f}   t = {t:+.4f}  {sig}")
        print()

    print("  Significance: * p<0.05  ** p<0.01  *** p<0.001")
    print("  ===========================================================")

    # Quintile analysis: split GCP deviation into 5 buckets, show avg BTC stats per bucket
    print("\n  --- QUINTILE ANALYSIS: GCP Deviation Buckets -> BTC Behavior ---")
    sorted_pairs = sorted(paired, key=lambda p: p["gcp_deviation"])
    q_size = len(sorted_pairs) // 5

    for q in range(5):
        start = q * q_size
        end = start + q_size if q < 4 else len(sorted_pairs)
        bucket = sorted_pairs[start:end]

        avg_dev = statistics.mean([p["gcp_deviation"] for p in bucket])
        avg_abs = statistics.mean([p["btc_abs_change"] for p in bucket])
        avg_vol = statistics.mean([p["btc_volume"] for p in bucket])
        avg_range = statistics.mean([p["btc_range"] for p in bucket])
        avg_pct = statistics.mean([p["btc_pct_change"] for p in bucket])

        label = ["LOWEST", "LOW", "MIDDLE", "HIGH", "HIGHEST"][q]
        print(f"  Q{q+1} ({label:>7}) | GCP dev: {avg_dev:+.4f} | "
              f"BTC |chg|: {avg_abs:.3f}% | range: {avg_range:.3f}% | "
              f"vol: {avg_vol:,.0f} | bias: {avg_pct:+.4f}%")

    print()

    # ── DIRECTIONAL ANALYSIS (the key question) ──
    print("\n  --- DIRECTIONAL ANALYSIS: Does GCP deviation predict BTC direction? ---")
    print("  (This is what matters — not volatility, DIRECTION)\n")

    # Test 1: When GCP deviation is high positive vs high negative, which way does BTC go?
    # Split into: rising deviation (above median) vs falling deviation (below median)
    median_dev = statistics.median(gcp_dev)

    high_dev_hours = [p for p in paired if p["gcp_deviation"] > median_dev]
    low_dev_hours = [p for p in paired if p["gcp_deviation"] <= median_dev]

    high_up = sum(1 for p in high_dev_hours if p["btc_pct_change"] > 0)
    high_down = sum(1 for p in high_dev_hours if p["btc_pct_change"] < 0)
    low_up = sum(1 for p in low_dev_hours if p["btc_pct_change"] > 0)
    low_down = sum(1 for p in low_dev_hours if p["btc_pct_change"] < 0)

    high_pct_up = high_up / max(len(high_dev_hours), 1) * 100
    low_pct_up = low_up / max(len(low_dev_hours), 1) * 100

    print(f"  HIGH GCP deviation hours: {high_up} up / {high_down} down ({high_pct_up:.1f}% bullish)")
    print(f"  LOW  GCP deviation hours: {low_up} up / {low_down} down ({low_pct_up:.1f}% bullish)")
    print(f"  Directional edge: {high_pct_up - low_pct_up:+.2f}% difference")
    print()

    # Test 2: Rate of change of GCP deviation (is the field SHIFTING?)
    # Compare deviation at hour N vs hour N-1, see if the CHANGE predicts BTC direction
    print("  --- RATE OF CHANGE: Does a SHIFT in field deviation predict BTC direction? ---\n")
    sorted_paired = sorted(paired, key=lambda p: p["ts"])
    shift_pairs = []
    for i in range(1, len(sorted_paired)):
        prev = sorted_paired[i-1]
        curr = sorted_paired[i]
        if curr["ts"] - prev["ts"] == 3600:  # consecutive hours only
            shift = curr["gcp_deviation"] - prev["gcp_deviation"]
            shift_pairs.append({
                "gcp_shift": shift,
                "btc_pct_change": curr["btc_pct_change"],
                "btc_next_pct": sorted_paired[i]["btc_pct_change"],
            })

    if shift_pairs:
        rising_field = [p for p in shift_pairs if p["gcp_shift"] > 0]
        falling_field = [p for p in shift_pairs if p["gcp_shift"] < 0]

        rise_up = sum(1 for p in rising_field if p["btc_pct_change"] > 0)
        rise_down = sum(1 for p in rising_field if p["btc_pct_change"] < 0)
        fall_up = sum(1 for p in falling_field if p["btc_pct_change"] > 0)
        fall_down = sum(1 for p in falling_field if p["btc_pct_change"] < 0)

        rise_pct = rise_up / max(len(rising_field), 1) * 100
        fall_pct = fall_up / max(len(falling_field), 1) * 100

        print(f"  RISING field (deviation increasing):  {rise_up} up / {rise_down} down ({rise_pct:.1f}% bullish)")
        print(f"  FALLING field (deviation decreasing): {fall_up} up / {fall_down} down ({fall_pct:.1f}% bullish)")
        print(f"  Directional edge: {rise_pct - fall_pct:+.2f}% difference")

        r_shift, t_shift = pearson(
            [p["gcp_shift"] for p in shift_pairs],
            [p["btc_pct_change"] for p in shift_pairs]
        )
        sig = "***" if abs(t_shift) > 3.29 else "**" if abs(t_shift) > 2.58 else "*" if abs(t_shift) > 1.96 else ""
        print(f"  Pearson (GCP shift vs BTC direction): r = {r_shift:+.6f}  t = {t_shift:+.4f}  {sig}")
    print()

    # Test 3: Lagged analysis — does GCP deviation NOW predict BTC direction 1-4 hours LATER?
    print("  --- LAGGED ANALYSIS: Does GCP NOW predict BTC direction LATER? ---\n")
    ts_index = {p["ts"]: p for p in paired}

    for lag in [1, 2, 3, 4, 6, 12, 24]:
        lag_pairs_up = 0
        lag_pairs_down = 0
        high_dev_up = 0
        high_dev_down = 0
        low_dev_up = 0
        low_dev_down = 0
        high_dev_mean_pct = []
        low_dev_mean_pct = []

        for p in paired:
            future_ts = p["ts"] + lag * 3600
            if future_ts in ts_index:
                future = ts_index[future_ts]
                if p["gcp_deviation"] > median_dev:
                    if future["btc_pct_change"] > 0:
                        high_dev_up += 1
                    else:
                        high_dev_down += 1
                    high_dev_mean_pct.append(future["btc_pct_change"])
                else:
                    if future["btc_pct_change"] > 0:
                        low_dev_up += 1
                    else:
                        low_dev_down += 1
                    low_dev_mean_pct.append(future["btc_pct_change"])

        total_high = high_dev_up + high_dev_down
        total_low = low_dev_up + low_dev_down

        if total_high > 0 and total_low > 0:
            h_pct = high_dev_up / total_high * 100
            l_pct = low_dev_up / total_low * 100
            h_mean = statistics.mean(high_dev_mean_pct) if high_dev_mean_pct else 0
            l_mean = statistics.mean(low_dev_mean_pct) if low_dev_mean_pct else 0
            edge = h_pct - l_pct
            flag = " <-- SIGNAL" if abs(edge) > 2.0 else ""
            print(f"  Lag {lag:>2}h | HIGH dev: {h_pct:.1f}% up (avg {h_mean:+.4f}%) | "
                  f"LOW dev: {l_pct:.1f}% up (avg {l_mean:+.4f}%) | "
                  f"edge: {edge:+.2f}%{flag}")

    print()

    # Test 4: Extreme deviation events — what happens to BTC in next 24h?
    print("  --- EXTREME EVENTS: Top 1% GCP deviation hours -> BTC next 24h ---\n")
    dev_threshold = sorted(gcp_dev, reverse=True)[max(1, len(gcp_dev) // 100)]

    extreme_events = []
    for p in paired:
        if p["gcp_deviation"] >= dev_threshold:
            # Look at BTC over next 24 hours
            future_changes = []
            for lag in range(1, 25):
                future_ts = p["ts"] + lag * 3600
                if future_ts in ts_index:
                    future_changes.append(ts_index[future_ts]["btc_pct_change"])

            if len(future_changes) >= 12:
                cumulative = sum(future_changes)
                extreme_events.append({
                    "dt": p["dt"],
                    "gcp_dev": p["gcp_deviation"],
                    "btc_24h_cumulative": round(cumulative, 4),
                    "direction": "UP" if cumulative > 0 else "DOWN",
                })

    if extreme_events:
        up_count = sum(1 for e in extreme_events if e["direction"] == "UP")
        down_count = sum(1 for e in extreme_events if e["direction"] == "DOWN")
        avg_move = statistics.mean([e["btc_24h_cumulative"] for e in extreme_events])

        print(f"  {len(extreme_events)} extreme GCP events found (top 1% deviation)")
        print(f"  BTC next 24h: {up_count} up / {down_count} down")
        print(f"  Average 24h cumulative move: {avg_move:+.4f}%")
        print(f"  Win rate (if betting direction of avg): {max(up_count, down_count)/len(extreme_events)*100:.1f}%")
        print()
        print("  Sample extreme events:")
        for e in extreme_events[:20]:
            print(f"    {e['dt']} | GCP dev: {e['gcp_dev']:+.4f} | BTC 24h: {e['btc_24h_cumulative']:+.4f}% {e['direction']}")
    print()

    # Save full paired dataset
    output_file = os.path.join(RESULTS_DIR, "gcp_btc_correlation.json")
    with open(output_file, "w") as f:
        json.dump(paired, f, indent=2)
    print(f"  Full dataset saved: {output_file}")

    # Save CSV too
    csv_file = os.path.join(RESULTS_DIR, "gcp_btc_correlation.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=paired[0].keys())
        writer.writeheader()
        writer.writerows(paired)
    print(f"  CSV saved: {csv_file}")

    return paired


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n  === HEARTBEAT - Collective Consciousness Field Reader ===")
    print("  === GCP RNG Network x BTC Price Correlation Analysis  ===\n")

    # BTC on Binance starts around 2017-08-17
    # Go back as far as we can with both datasets
    start = datetime(2019, 1, 1)
    end = datetime(2025, 12, 31)

    # ── Download GCP data ──
    print("  [1/4] Downloading GCP RNG data...")
    download_gcp_range(start, end)

    # ── Parse GCP into hourly stats (with cache) ──
    cache_file = os.path.join(RESULTS_DIR, "gcp_hourly_cache.json")
    if os.path.exists(cache_file):
        print("\n  [2/4] Loading cached GCP hourly stats...")
        with open(cache_file, "r") as f:
            raw = json.load(f)
        all_gcp_hourly = {int(k): v for k, v in raw.items()}
        print(f"  GCP loaded from cache: {len(all_gcp_hourly)} hourly data points")
    else:
        print("\n  [2/4] Parsing GCP data into hourly deviation stats...")
        all_gcp_hourly = {}
        gcp_files = sorted([f for f in os.listdir(GCP_DIR) if f.endswith(".csv")])
        for i, fname in enumerate(gcp_files):
            if (i + 1) % 100 == 0:
                print(f"    Parsed {i+1}/{len(gcp_files)} files...", flush=True)
            records = parse_gcp_csv(os.path.join(GCP_DIR, fname))
            hourly = compute_hourly_deviation(records)
            all_gcp_hourly.update(hourly)

        print(f"  GCP parsed: {len(all_gcp_hourly)} hourly data points")
        # Save cache
        with open(cache_file, "w") as f:
            json.dump(all_gcp_hourly, f)
        print(f"  Cached to {cache_file}")

    if not all_gcp_hourly:
        print("\n  ERROR: No GCP data parsed. Check downloads.")
        sys.exit(1)

    # ── Download BTC data ──
    print("\n  [3/4] Downloading BTC hourly price data...")
    min_ts = min(all_gcp_hourly.keys())
    max_ts = max(all_gcp_hourly.keys())
    btc_candles = load_btc_hourly(min_ts, max_ts)

    if not btc_candles:
        print("\n  ERROR: No BTC data downloaded.")
        sys.exit(1)

    # ── Correlate ──
    print("\n  [4/4] Running correlation analysis...")
    correlate(all_gcp_hourly, btc_candles)

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
