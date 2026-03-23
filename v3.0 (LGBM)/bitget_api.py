"""
Bitget API Integration for Crypto Matrix Trading Bot
=====================================================
Supports both paper mode (virtual balance) and live mode (real API calls).
Switch between modes with PAPER_MODE flag.

API docs: https://www.bitget.com/api-doc/
Uses Bitget's mix (futures) endpoints for leveraged trading.
"""
import os
import sys
import io
import time
import json
import hmac
import hashlib
import base64
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime, timezone

if os.name == 'nt':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PAPER_MODE = True  # Set to False for LIVE trading (use with extreme caution!)

# API credentials from environment variables
API_KEY = os.environ.get("BITGET_API_KEY", "")
API_SECRET = os.environ.get("BITGET_API_SECRET", os.environ.get("BITGET_SECRET", ""))
API_PASSPHRASE = os.environ.get("BITGET_API_PASSPHRASE", os.environ.get("BITGET_PASSPHRASE", ""))

# Bitget API base URLs
BASE_URL = "https://api.bitget.com"
# For demo/testnet (if available):
DEMO_BASE_URL = "https://api.bitget.com"  # Bitget uses same endpoint with demo flag

# Default trading pair
SYMBOL = "BTCUSDT_UMCBL"  # USDT-M perpetual contract
PRODUCT_TYPE = "USDT-FUTURES"

# ---------------------------------------------------------------------------
# Authentication helpers
# ---------------------------------------------------------------------------
def _get_timestamp():
    """Get current timestamp in milliseconds."""
    return str(int(time.time() * 1000))


def _sign(timestamp, method, request_path, body=""):
    """Create HMAC-SHA256 signature for Bitget API."""
    message = timestamp + method.upper() + request_path + (body if body else "")
    mac = hmac.new(
        API_SECRET.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    )
    return base64.b64encode(mac.digest()).decode("utf-8")


def _headers(method, request_path, body=""):
    """Build authenticated headers."""
    timestamp = _get_timestamp()
    sign = _sign(timestamp, method, request_path, body)
    return {
        "ACCESS-KEY": API_KEY,
        "ACCESS-SIGN": sign,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-PASSPHRASE": API_PASSPHRASE,
        "Content-Type": "application/json",
        "locale": "en-US",
    }


def _request(method, path, params=None, body=None, authenticated=False):
    """Make HTTP request to Bitget API."""
    url = BASE_URL + path
    body_str = ""

    if method == "GET" and params:
        query = urllib.parse.urlencode(params)
        url += "?" + query
        request_path = path + "?" + query
    else:
        request_path = path
        if body:
            body_str = json.dumps(body)

    if authenticated:
        headers = _headers(method, request_path, body_str)
    else:
        headers = {"Content-Type": "application/json"}

    req = urllib.request.Request(
        url,
        data=body_str.encode("utf-8") if body_str else None,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            if data.get("code") == "00000":
                return data.get("data", data)
            else:
                print(f"[BITGET API ERROR] {data.get('code')}: {data.get('msg')}")
                return None
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        print(f"[BITGET HTTP ERROR] {e.code}: {error_body}")
        return None
    except Exception as e:
        print(f"[BITGET ERROR] {e}")
        return None


# ---------------------------------------------------------------------------
# Virtual Account (for paper mode)
# ---------------------------------------------------------------------------
class VirtualAccount:
    """Simulates a trading account for paper trading."""

    def __init__(self, initial_balance=1000.0):
        self.balance = initial_balance
        self.positions = {}  # symbol -> position info
        self.trade_history = []
        self._next_id = 1

    def get_balance(self):
        return {
            "available": self.balance,
            "frozen": sum(p.get("margin", 0) for p in self.positions.values()),
            "total": self.balance + sum(p.get("margin", 0) for p in self.positions.values()),
        }

    def open_position(self, symbol, side, size, leverage, entry_price, stop_loss=None, take_profit=None):
        """Open a paper position."""
        margin = size / leverage
        if margin > self.balance:
            return None, "Insufficient balance"

        self.balance -= margin
        pos_id = str(self._next_id)
        self._next_id += 1

        self.positions[pos_id] = {
            "id": pos_id,
            "symbol": symbol,
            "side": side,
            "size": size,
            "leverage": leverage,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "margin": margin,
            "opened_at": datetime.utcnow().isoformat(),
            "unrealized_pnl": 0.0,
        }
        return pos_id, "OK"

    def close_position(self, pos_id, exit_price):
        """Close a paper position."""
        pos = self.positions.get(pos_id)
        if not pos:
            return None, "Position not found"

        if pos["side"] == "long":
            pnl = ((exit_price - pos["entry_price"]) / pos["entry_price"]) * pos["size"]
        else:
            pnl = ((pos["entry_price"] - exit_price) / pos["entry_price"]) * pos["size"]

        self.balance += pos["margin"] + pnl
        self.trade_history.append({
            **pos,
            "exit_price": exit_price,
            "pnl": pnl,
            "closed_at": datetime.utcnow().isoformat(),
        })
        del self.positions[pos_id]
        return pnl, "OK"

    def get_positions(self):
        return list(self.positions.values())

    def update_unrealized(self, current_price):
        """Update unrealized P&L for all positions."""
        for pos in self.positions.values():
            if pos["side"] == "long":
                pos["unrealized_pnl"] = ((current_price - pos["entry_price"]) / pos["entry_price"]) * pos["size"]
            else:
                pos["unrealized_pnl"] = ((pos["entry_price"] - current_price) / pos["entry_price"]) * pos["size"]


# Global virtual account instance
_virtual_account = VirtualAccount()


# ---------------------------------------------------------------------------
# Public API Functions
# ---------------------------------------------------------------------------
def get_account_balance():
    """Get USDT balance for futures account."""
    if PAPER_MODE:
        return _virtual_account.get_balance()

    data = _request("GET", "/api/v2/mix/account/account",
                    params={"symbol": SYMBOL, "productType": PRODUCT_TYPE},
                    authenticated=True)
    if data:
        return {
            "available": float(data.get("available", 0)),
            "frozen": float(data.get("frozen", 0)),
            "total": float(data.get("accountEquity", 0)),
        }
    return None


def place_order(symbol="BTCUSDT", side="long", size=0.001, leverage=10,
                stop_loss=None, take_profit=None):
    """Open a position (market order)."""
    if PAPER_MODE:
        price = get_ticker(symbol)
        if not price:
            return None, "Cannot get price"
        pos_id, msg = _virtual_account.open_position(
            symbol, side, size * price, leverage, price, stop_loss, take_profit
        )
        return {"orderId": pos_id, "price": price}, msg

    # Live mode
    _request("POST", "/api/v2/mix/account/set-leverage",
             body={
                 "symbol": SYMBOL,
                 "productType": PRODUCT_TYPE,
                 "marginCoin": "USDT",
                 "leverage": str(leverage),
             },
             authenticated=True)

    order_body = {
        "symbol": SYMBOL,
        "productType": PRODUCT_TYPE,
        "marginMode": "crossed",
        "marginCoin": "USDT",
        "size": str(size),
        "side": "buy" if side == "long" else "sell",
        "tradeSide": "open",
        "orderType": "market",
    }

    # Add SL/TP if provided
    if stop_loss:
        order_body["presetStopLossPrice"] = str(stop_loss)
    if take_profit:
        order_body["presetTakeProfitPrice"] = str(take_profit)

    data = _request("POST", "/api/v2/mix/order/place-order",
                    body=order_body, authenticated=True)
    if data:
        return {"orderId": data.get("orderId")}, "OK"
    return None, "Order failed"


def close_position(symbol="BTCUSDT", pos_id=None):
    """Close an open position."""
    if PAPER_MODE:
        price = get_ticker(symbol)
        if not price:
            return None, "Cannot get price"
        if pos_id:
            pnl, msg = _virtual_account.close_position(pos_id, price)
            return {"pnl": pnl, "price": price}, msg
        # Close all positions for symbol
        for pid, pos in list(_virtual_account.positions.items()):
            if pos["symbol"] == symbol:
                _virtual_account.close_position(pid, price)
        return {"price": price}, "All closed"

    # Live mode — close via flash close
    data = _request("POST", "/api/v2/mix/order/close-positions",
                    body={
                        "symbol": SYMBOL,
                        "productType": PRODUCT_TYPE,
                    },
                    authenticated=True)
    return data, "OK" if data else "Failed"


def get_open_positions():
    """List all open positions."""
    if PAPER_MODE:
        return _virtual_account.get_positions()

    data = _request("GET", "/api/v2/mix/position/all-position",
                    params={"productType": PRODUCT_TYPE},
                    authenticated=True)
    if data:
        positions = []
        for p in data:
            if float(p.get("total", 0)) > 0:
                positions.append({
                    "symbol": p.get("symbol"),
                    "side": "long" if p.get("holdSide") == "long" else "short",
                    "size": float(p.get("total", 0)),
                    "entry_price": float(p.get("averageOpenPrice", 0)),
                    "leverage": float(p.get("leverage", 1)),
                    "unrealized_pnl": float(p.get("unrealizedPL", 0)),
                    "margin": float(p.get("margin", 0)),
                })
        return positions
    return []


def get_ticker(symbol="BTCUSDT"):
    """Get current price for a symbol."""
    # Use Binance.US for public price (no auth needed)
    try:
        url = f"https://api.binance.us/api/v3/ticker/price?symbol={symbol}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return float(data["price"])
    except Exception:
        pass

    # Fallback to Bitget
    data = _request("GET", "/api/v2/mix/market/ticker",
                    params={"symbol": SYMBOL, "productType": PRODUCT_TYPE})
    if data and isinstance(data, list) and len(data) > 0:
        return float(data[0].get("lastPr", 0))
    return None


def get_klines(symbol="BTCUSDT", interval="4h", limit=100):
    """Get candle data."""
    # Use Binance.US for candles (public, no auth)
    interval_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h",
                    "4h": "4h", "1d": "1d", "1w": "1w"}
    binance_interval = interval_map.get(interval, interval)

    try:
        url = (f"https://api.binance.us/api/v3/klines"
               f"?symbol={symbol}&interval={binance_interval}&limit={limit}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            candles = []
            for k in data:
                candles.append({
                    "time": k[0],
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                })
            return candles
    except Exception as e:
        print(f"[KLINES ERROR] {e}")
        return []


# ---------------------------------------------------------------------------
# Status / Info
# ---------------------------------------------------------------------------
def print_status():
    """Print current account status."""
    bal = get_account_balance()
    price = get_ticker()
    positions = get_open_positions()

    mode = "PAPER" if PAPER_MODE else "LIVE"
    print(f"\n{'=' * 60}")
    print(f"  BITGET API STATUS [{mode} MODE]")
    print(f"{'=' * 60}")

    if bal:
        print(f"  Balance:    ${bal['available']:.2f} available / ${bal['total']:.2f} total")
    if price:
        print(f"  BTC Price:  ${price:,.2f}")
    print(f"  Positions:  {len(positions)}")
    for p in positions:
        print(f"    {p.get('symbol', 'BTC')} {p['side']} "
              f"size={p.get('size', 0):.4f} entry=${p.get('entry_price', 0):.2f} "
              f"lev={p.get('leverage', 1)}x P&L=${p.get('unrealized_pnl', 0):.2f}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print_status()

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "balance":
            print(json.dumps(get_account_balance(), indent=2))
        elif cmd == "price":
            p = get_ticker()
            print(f"BTC/USDT: ${p:,.2f}" if p else "Failed to get price")
        elif cmd == "positions":
            pos = get_open_positions()
            print(json.dumps(pos, indent=2, default=str))
        elif cmd == "klines":
            interval = sys.argv[2] if len(sys.argv) > 2 else "4h"
            limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10
            klines = get_klines(interval=interval, limit=limit)
            for k in klines[-5:]:
                print(f"  {datetime.utcfromtimestamp(k['time']/1000).strftime('%Y-%m-%d %H:%M')} "
                      f"O={k['open']:.2f} H={k['high']:.2f} L={k['low']:.2f} C={k['close']:.2f} V={k['volume']:.0f}")
        elif cmd == "test-order":
            # Place a test paper order
            result, msg = place_order("BTCUSDT", "long", 0.001, 10)
            print(f"Order result: {result}, msg: {msg}")
        else:
            print(f"Unknown command: {cmd}")
            print("Commands: balance, price, positions, klines [interval] [limit], test-order")
