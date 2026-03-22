"""
JSON API bridge for the Next.js dashboard.
Called by the dashboard API routes via child_process.execSync.

Usage:
  python api_bridge.py score 2026-03-16
  python api_bridge.py ta 1d
  python api_bridge.py manipulation 2026-03-16
  python api_bridge.py coins
"""
import sys
import os
import io
import json
import sqlite3
from datetime import datetime

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BTC_DB


def cmd_score(date_str):
    from unified_engine import UnifiedSignal

    dt = datetime.strptime(date_str, "%Y-%m-%d")

    # Get current BTC price
    btc_price = None
    try:
        conn = sqlite3.connect(BTC_DB)
        row = conn.execute(
            "SELECT close FROM ohlcv WHERE symbol='BTC/USDT' AND timeframe='1d' ORDER BY open_time DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if row:
            btc_price = row[0]
    except Exception:
        pass

    sig = UnifiedSignal(dt, btc_price=btc_price)
    data = sig.summary_dict()

    # Flatten for the dashboard
    result = {
        "score": round(data["final_score"], 2),
        "confidence": round(data["confidence"], 1),
        "action": data["action"],
        "volatility_alert": data["volatility_alert"],
        "inversion_setup": data["inversion_setup"],
        "signal_count": data["signal_count"],
        "bearish_count": data["bearish_signals"],
        "bullish_count": data["bullish_signals"],
        "neutral_count": data["neutral_signals"],
        "signals": data["signals"],
        "date": date_str,
    }

    # TA fields if present
    if "ta_bias" in data:
        result["ta_bias"] = data["ta_bias"]
        result["ta_confidence"] = data["ta_confidence"]
        result["wyckoff_phase"] = data.get("wyckoff_phase", "unknown")
        result["elliott_wave"] = data.get("elliott_wave", "unknown")
        result["gann_direction"] = data.get("gann_direction", "neutral")

    # Group signals by direction for component breakdown
    bearish_weight = sum(abs(s["weight"]) for s in data["signals"] if s["direction"] < 0)
    bullish_weight = sum(abs(s["weight"]) for s in data["signals"] if s["direction"] > 0)

    # Derive component scores for the UI
    num_signals = [s for s in data["signals"] if s["name"].startswith(("date_reduction", "day_of_year", "moon", "zodiac", "planetary", "master_num", "caution", "pump", "fibonacci", "angel", "clock", "mirror", "digit_sum"))]
    tech_signals = [s for s in data["signals"] if s["name"].startswith(("ta_", "wyckoff", "elliott", "gann", "sma", "ema", "rsi", "macd", "momentum"))]
    tweet_signals = [s for s in data["signals"] if s["name"].startswith(("tweet", "gematria"))]
    manip_signals = [s for s in data["signals"] if s["name"].startswith(("convergence", "ritual", "misdirection", "phase_shift", "fractal"))]

    def component_score(sigs):
        if not sigs:
            return 0
        return round(sum(s["weight"] * s["direction"] for s in sigs), 1)

    result["components"] = {
        "numerology": component_score(num_signals),
        "technical": component_score(tech_signals),
        "tweets": component_score(tweet_signals),
        "manipulation": component_score(manip_signals),
    }

    # Map to existing UI fields
    result["threat_level"] = (
        "CRITICAL" if result["score"] <= -7 else
        "HIGH" if result["score"] <= -4 else
        "MODERATE" if result["score"] < 0 else
        "LOW"
    )
    result["inversion_warning"] = data["inversion_setup"]
    result["phase_shift"] = data["volatility_alert"]

    print(json.dumps(result))


def cmd_ta(timeframe):
    import numpy as np
    from technical_engine import full_ta_analysis

    result = full_ta_analysis(timeframe=timeframe)

    # Make serializable: convert any non-serializable types
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            v = float(obj)
            if v != v:  # NaN
                return None
            return round(v, 6)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, float):
            if obj != obj:  # NaN
                return None
            return round(obj, 6)
        elif isinstance(obj, np.ndarray):
            return make_serializable(obj.tolist())
        elif isinstance(obj, set):
            return list(obj)
        return obj

    print(json.dumps(make_serializable(result)))


def cmd_manipulation(date_str):
    from manipulation_detector import full_manipulation_scan

    dt = datetime.strptime(date_str, "%Y-%m-%d")

    # Get BTC price
    btc_price = None
    btc_volume = None
    try:
        conn = sqlite3.connect(BTC_DB)
        row = conn.execute(
            "SELECT close, volume FROM ohlcv WHERE symbol='BTC/USDT' AND timeframe='1d' ORDER BY open_time DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if row:
            btc_price = row[0]
            btc_volume = row[1]
    except Exception:
        pass

    result = full_manipulation_scan(dt, btc_price=btc_price, btc_volume=btc_volume)

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, float):
            if obj != obj:
                return None
            return round(obj, 6)
        elif isinstance(obj, set):
            return list(obj)
        return obj

    print(json.dumps(make_serializable(result)))


def cmd_coins():
    """Return available coins with latest price and 24h change."""
    try:
        conn = sqlite3.connect(BTC_DB)
        # Get distinct symbols
        symbols = conn.execute(
            "SELECT DISTINCT symbol FROM ohlcv WHERE timeframe = '1d' ORDER BY symbol"
        ).fetchall()

        coins = []
        for (sym,) in symbols:
            # Get latest 2 candles for price + change
            rows = conn.execute(
                "SELECT close FROM ohlcv WHERE symbol = ? AND timeframe = '1d' ORDER BY open_time DESC LIMIT 2",
                (sym,)
            ).fetchall()
            if rows:
                price = rows[0][0]
                change = 0
                if len(rows) >= 2 and rows[1][0] > 0:
                    change = ((rows[0][0] - rows[1][0]) / rows[1][0]) * 100
                coins.append({
                    "symbol": sym,
                    "price": round(price, 8),
                    "change24h": round(change, 2),
                })

        conn.close()
        print(json.dumps({"coins": coins}))
    except Exception as e:
        print(json.dumps({"coins": [], "error": str(e)}))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python api_bridge.py <command> [args]"}))
        sys.exit(1)

    command = sys.argv[1]

    try:
        if command == "score":
            date_str = sys.argv[2] if len(sys.argv) > 2 else datetime.utcnow().strftime("%Y-%m-%d")
            cmd_score(date_str)
        elif command == "ta":
            tf = sys.argv[2] if len(sys.argv) > 2 else "1d"
            cmd_ta(tf)
        elif command == "manipulation":
            date_str = sys.argv[2] if len(sys.argv) > 2 else datetime.utcnow().strftime("%Y-%m-%d")
            cmd_manipulation(date_str)
        elif command == "coins":
            cmd_coins()
        else:
            print(json.dumps({"error": f"Unknown command: {command}"}))
            sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
