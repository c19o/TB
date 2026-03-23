"""Quick optimizer — tests key combos across 3 market regimes, saves all to CSV."""
import sys, io, csv, os, time
if os.name == 'nt':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from datetime import datetime

import simulate_ultimate_v2 as sim

RESULTS_FILE = "optimization_results.csv"

# Clear previous
with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "tfs", "risk_1d", "risk_4h", "risk_1h",
                "regime", "roi", "pf", "dd", "trades", "wr", "final", "score", "notes"])

def run_test(test_id, tfs, risks, stops, rrs, regime_end, regime_name, notes=""):
    sim._ACTIVE_TFS = set(tfs)
    sim._SIGNAL_RISK_TIERS = getattr(sim, "_SIGNAL_RISK_TIERS", {})
    for tf in ["1D", "4H", "1H", "15M"]:
        cfg = sim.TF_CONFIG[tf]
        cfg["risk_pct"] = risks.get(tf, cfg["risk_pct"])
        cfg["stop_atr_mult"] = stops.get(tf, cfg["stop_atr_mult"])
        cfg["rr_ratio"] = rrs.get(tf, cfg["rr_ratio"])
        cfg["trailing_atr"] = cfg["stop_atr_mult"] * 1.5
    try:
        r = sim.run_simulation(sim_end_date=regime_end, sim_days=180)
    except Exception as e:
        print(f"  [{test_id}] ERROR: {e}")
        return None
    if not r or r["trades"] == 0:
        print(f"  [{test_id}] No trades")
        return None
    roi = r["roi"]
    pf = min(r["profit_factor"], 10)
    dd = r["max_dd"]
    score = roi * (1 - dd / 100) * pf
    print(f"  [{test_id:>3}] {regime_name:<12} TF={'+'.join(tfs):<12} "
          f"ROI={roi:>+7.1f}%  PF={pf:.2f}  DD={dd:>5.1f}%  "
          f"trades={r['trades']:>5}  ${r['balance']:>8,.0f}  score={score:>7.0f}  {notes}")
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([test_id, "+".join(tfs), risks.get("1D", 0), risks.get("4H", 0),
                     risks.get("1H", 0), regime_name,
                     f"{roi:.1f}", f"{pf:.2f}", f"{dd:.1f}",
                     r["trades"], f"{r['win_rate']:.1f}", f"{r['balance']:.2f}",
                     f"{score:.0f}", notes])
    return r

# 3 market regimes
REGIMES = [
    (datetime(2026, 3, 16), "RECENT"),
    (datetime(2022, 6, 18), "BEAR_2022"),
    (datetime(2024, 12, 17), "BULL_2024"),
]
default_stops = {"1D": 1.0, "4H": 0.7, "1H": 0.5, "15M": 0.3}
default_rrs = {"1D": 1.5, "4H": 2.0, "1H": 2.0, "15M": 2.5}

tid = 0
start = time.time()

# ======== PHASE 1: TF combos x 3 regimes ========
print("=" * 100)
print("  PHASE 1: TIMEFRAME COMBINATIONS x 3 MARKET REGIMES")
print("=" * 100)
tf_combos = [["1D"], ["4H"], ["1H"], ["1D", "4H"], ["1D", "4H", "1H"],
             ["1D", "4H", "1H", "15M"], ["4H", "1H"]]
for rend, rname in REGIMES:
    print(f"\n  --- {rname} ---")
    for tfs in tf_combos:
        tid += 1
        run_test(tid, tfs, {"1D": 0.03, "4H": 0.02, "1H": 0.01, "15M": 0.005},
                 default_stops, default_rrs, rend, rname, f"TF={'+'.join(tfs)}")

# ======== PHASE 2: Uniform risk ========
print("\n" + "=" * 100)
print("  PHASE 2: UNIFORM RISK (same % all TFs)")
print("=" * 100)
for risk_pct in [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]:
    for rend, rname in REGIMES:
        tid += 1
        risks = {"1D": risk_pct, "4H": risk_pct, "1H": risk_pct}
        run_test(tid, ["1D", "4H", "1H"], risks, default_stops, default_rrs,
                 rend, rname, f"uniform_{risk_pct*100:.0f}%")

# ======== PHASE 3: Tiered risk ========
print("\n" + "=" * 100)
print("  PHASE 3: TIERED RISK COMBOS")
print("=" * 100)
risk_combos = [
    ({"1D": 0.03, "4H": 0.02, "1H": 0.01}, "3/2/1 default"),
    ({"1D": 0.05, "4H": 0.03, "1H": 0.02}, "5/3/2 aggressive"),
    ({"1D": 0.07, "4H": 0.04, "1H": 0.02}, "7/4/2 high"),
    ({"1D": 0.10, "4H": 0.05, "1H": 0.03}, "10/5/3 max"),
    ({"1D": 0.05, "4H": 0.05, "1H": 0.03}, "5/5/3 equal-high"),
    ({"1D": 0.03, "4H": 0.03, "1H": 0.03}, "3/3/3 flat"),
    ({"1D": 0.04, "4H": 0.04, "1H": 0.02}, "4/4/2 balanced"),
    ({"1D": 0.05, "4H": 0.02, "1H": 0.01}, "5/2/1 1D-heavy"),
]
for risks, label in risk_combos:
    for rend, rname in REGIMES:
        tid += 1
        run_test(tid, ["1D", "4H", "1H"], risks, default_stops, default_rrs,
                 rend, rname, label)

# ======== PHASE 4: Stop/RR sweep ========
print("\n" + "=" * 100)
print("  PHASE 4: STOP & R:R SWEEP")
print("=" * 100)
best_risks = {"1D": 0.05, "4H": 0.03, "1H": 0.02}
stop_rr_combos = [
    ({"1D": 0.8, "4H": 0.5, "1H": 0.3}, {"1D": 1.5, "4H": 2.0, "1H": 2.0}, "tight+default_RR"),
    ({"1D": 1.0, "4H": 0.7, "1H": 0.5}, {"1D": 1.5, "4H": 2.0, "1H": 2.0}, "default+default_RR"),
    ({"1D": 1.2, "4H": 1.0, "1H": 0.7}, {"1D": 1.5, "4H": 2.0, "1H": 2.0}, "wide+default_RR"),
    ({"1D": 1.0, "4H": 0.7, "1H": 0.5}, {"1D": 2.0, "4H": 2.5, "1H": 2.5}, "default+high_RR"),
    ({"1D": 1.0, "4H": 0.7, "1H": 0.5}, {"1D": 2.5, "4H": 3.0, "1H": 3.0}, "default+vhigh_RR"),
    ({"1D": 0.8, "4H": 0.5, "1H": 0.3}, {"1D": 2.0, "4H": 3.0, "1H": 3.0}, "tight+high_RR"),
    ({"1D": 1.5, "4H": 0.7, "1H": 0.5}, {"1D": 1.5, "4H": 3.0, "1H": 2.0}, "1D_wide+mixed_RR"),
]
for stops, rrs, label in stop_rr_combos:
    for rend, rname in REGIMES:
        tid += 1
        run_test(tid, ["1D", "4H", "1H"], best_risks, stops, rrs, rend, rname, label)

# ======== PHASE 5: Per-signal tiered risk ========
print("\n" + "=" * 100)
print("  PHASE 5: PER-SIGNAL RISK TIERS (goldmine=5-10%)")
print("=" * 100)
sim._SIGNAL_RISK_TIERS = {
    "4H_113_LONG": 0.08, "4H_Gold_tweet_SHORT": 0.07,
    "4H_223_322_SHORT": 0.07, "4H_Convergence5": 0.10,
    "4H_Purple_image_TEST": 0.06, "1D_Elon_LONG": 0.10,
    "4H_Misdirection_BEARISH": 0.07, "1D_Wednesday_SHORT": 0.06,
    "4H_Day13_LONG": 0.05, "1D_Day13_LONG": 0.05,
    "4H_Elon_LONG": 0.05, "4H_TweetGematria_SHORT": 0.04,
    "1H_Misdirection_BEARISH": 0.04, "1D_Red_tweet_LONG": 0.05,
}
for rend, rname in REGIMES:
    tid += 1
    run_test(tid, ["1D", "4H", "1H"], best_risks, default_stops, default_rrs,
             rend, rname, "TIERED goldmine=5-10%")
sim._SIGNAL_RISK_TIERS = {}

# ======== PHASE 6: 1D+4H only (no 1H noise) ========
print("\n" + "=" * 100)
print("  PHASE 6: 1D+4H ONLY (no 1H noise)")
print("=" * 100)
for risks, label in risk_combos[:5]:
    for rend, rname in REGIMES:
        tid += 1
        run_test(tid, ["1D", "4H"], risks, default_stops, default_rrs, rend, rname, f"1D4H_{label}")

# ======== RESULTS ========
elapsed = time.time() - start
print(f"\n{'=' * 100}")
print(f"  COMPLETE: {tid} configs tested in {elapsed/60:.1f} minutes")
print(f"  Results: {RESULTS_FILE}")
print(f"{'=' * 100}")

# Sort and display top results
results = []
with open(RESULTS_FILE, "r") as f:
    for row in csv.DictReader(f):
        results.append(row)
results.sort(key=lambda x: -float(x.get("score", "0")))

print(f"\n  TOP 15 CONFIGS BY SCORE:")
print(f"  {'#':<3} {'TFs':<12} {'Regime':<12} {'ROI':>8} {'PF':>6} {'DD':>6} {'Trades':>7} {'WR':>6} {'Final$':>10} {'Score':>8}  Notes")
print("  " + "-" * 105)
for i, r in enumerate(results[:15]):
    print(f"  {i+1:<3} {r['tfs']:<12} {r['regime']:<12} {r['roi']:>7}% {r['pf']:>6} "
          f"{r['dd']:>5}% {r['trades']:>7} {r['wr']:>5}% ${float(r['final']):>9,.0f} "
          f"{r['score']:>8}  {r['notes']}")

# Cross-regime analysis
print(f"\n  CROSS-REGIME PERFORMANCE (configs that work in ALL markets):")
# Group by notes (config description)
from collections import defaultdict
by_config = defaultdict(list)
for r in results:
    by_config[r["notes"]].append(r)

cross_scores = []
for config, rows in by_config.items():
    regimes_tested = set(r["regime"] for r in rows)
    if len(regimes_tested) >= 3:
        avg_roi = sum(float(r["roi"]) for r in rows) / len(rows)
        avg_dd = sum(float(r["dd"]) for r in rows) / len(rows)
        avg_pf = sum(float(r["pf"]) for r in rows) / len(rows)
        min_roi = min(float(r["roi"]) for r in rows)
        cross_score = avg_roi * (1 - avg_dd / 100) * avg_pf
        cross_scores.append((config, avg_roi, avg_pf, avg_dd, min_roi, cross_score, rows))

cross_scores.sort(key=lambda x: -x[5])
print(f"  {'Config':<30} {'Avg ROI':>8} {'Avg PF':>7} {'Avg DD':>7} {'Min ROI':>8} {'Cross Score':>12}")
print("  " + "-" * 85)
for config, avg_roi, avg_pf, avg_dd, min_roi, cs, rows in cross_scores[:10]:
    print(f"  {config:<30} {avg_roi:>+7.1f}% {avg_pf:>6.2f} {avg_dd:>6.1f}% {min_roi:>+7.1f}% {cs:>11.0f}")
    for r in rows:
        print(f"    {r['regime']:<12} ROI={r['roi']:>7}% PF={r['pf']:>6} DD={r['dd']:>5}%")
