"""
Savage22 Crypto Matrix — Configuration
"""
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Canonical database directory — all streamers and readers use this
# Override with SAVAGE22_DB_DIR env var when running from non-root dirs
DB_DIR = os.environ.get('SAVAGE22_DB_DIR', PROJECT_DIR)

SAVAGE_DB = os.path.join(PROJECT_DIR, "savage22.db")
BTC_DB = os.path.join(PROJECT_DIR, "btc_prices.db")
CHROMA_DIR = os.path.join(PROJECT_DIR, "chroma_db")

# Esoteric KB for cross-referencing
ESOTERIC_KB_DIR = r"C:\Users\C\Desktop\MY GOOGLE DRIVE\Orgonite master"

# Crypto symbols to detect in messages
SYMBOLS = ["BTC", "XRP", "ETH", "XDC", "XLM", "QNT", "HBAR", "ADA", "SOL",
           "DOGE", "LINK", "DOT", "MATIC", "AVAX", "UNI", "ALGO", "VET"]

# Signal type keywords for auto-tagging
SIGNAL_KEYWORDS = {
    "gematria": ["gematria", "gematrix", "ordinal", "reverse ordinal", "cipher",
                 "english extended", "jewish", "sumerian", "isopsephy"],
    "numerology": ["numerology", "master number", "angel number", "vibration",
                   "11:11", "22", "33", "44", "55", "66", "77", "88", "99",
                   "111", "222", "333", "444", "555", "666", "777", "888", "999",
                   "dr matrix", "magic number", "pythagorean", "chaldean"],
    "astrology": ["astrology", "zodiac", "mercury retrograde", "full moon",
                  "new moon", "eclipse", "equinox", "solstice", "saturn",
                  "jupiter", "mars", "venus", "conjunction", "opposition",
                  "natal", "transit", "horoscope"],
    "technical": ["support", "resistance", "breakout", "breakdown", "fibonacci",
                  "fib", "ema", "sma", "rsi", "macd", "bollinger", "volume",
                  "divergence", "bull flag", "bear flag", "head and shoulders",
                  "cup and handle", "wedge", "triangle", "channel",
                  "long", "short", "buy", "sell", "entry", "target", "stop loss",
                  "take profit", "tp", "sl"],
    "decode": ["decode", "decoded", "decoding", "hidden message", "symbolism",
               "riddle", "riddler", "hint", "clue", "signal", "tweet",
               "elon", "musk", "schwartz", "dorsey"],
}

# Side detection patterns
LONG_PATTERNS = ["long", "buy", "bull", "pump", "moon", "breakout", "upside",
                 "accumulate", "load up", "going up", "higher", "green"]
SHORT_PATTERNS = ["short", "sell", "bear", "dump", "crash", "breakdown",
                  "downside", "drop", "lower", "red", "flush"]
