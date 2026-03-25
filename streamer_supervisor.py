"""
Streamer Supervisor — manages all 7 data streamer processes for Savage22.

Usage:
    python streamer_supervisor.py            # Start the supervisor daemon
    python streamer_supervisor.py --status   # Check status of all streamers

Manages: tweet, news, crypto, sports, macro, space_weather, download_btc
Auto-restarts crashed processes with exponential backoff.
Checks data freshness every 5 minutes.
"""

import sys
import os
import time
import signal
import subprocess
import threading
import sqlite3
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    from config import DB_DIR as _DB_DIR
except ImportError:
    _DB_DIR = BASE_DIR
LOG_DIR = os.path.join(BASE_DIR, "logs")
STATUS_FILE = os.path.join(BASE_DIR, "supervisor_status.json")

BACKOFF_SCHEDULE = [5, 15, 60, 300]  # seconds between restarts

STREAMERS = {
    "tweet_streamer": {
        "cmd": [sys.executable, "-u", os.path.join(BASE_DIR, "tweet_streamer.py")],
        "db": os.path.join(_DB_DIR, "tweets.db"),
        "freshness_query": "SELECT MAX(ts_unix) FROM tweets",
        "freshness_type": "unix",
        "stale_hours": 1,
    },
    "news_streamer": {
        "cmd": [sys.executable, "-u", os.path.join(BASE_DIR, "news_streamer.py")],
        "db": os.path.join(_DB_DIR, "news_articles.db"),
        "freshness_query": "SELECT MAX(ts_unix) FROM streamer_articles",
        "freshness_type": "unix",
        "stale_hours": 1,
    },
    "crypto_streamer": {
        "cmd": [sys.executable, "-u", os.path.join(BASE_DIR, "crypto_streamer.py")],
        "db": os.path.join(_DB_DIR, "onchain_data.db"),
        "freshness_query": "SELECT MAX(timestamp) FROM onchain_data",
        "freshness_type": "iso",
        "stale_hours": 1,
    },
    "sports_streamer": {
        "cmd": [sys.executable, "-u", os.path.join(BASE_DIR, "sports_streamer.py")],
        "db": os.path.join(_DB_DIR, "sports_results.db"),
        "freshness_query": "SELECT MAX(inserted_at) FROM games",
        "freshness_type": "iso",
        "stale_hours": 4,
    },
    "macro_streamer": {
        "cmd": [sys.executable, "-u", os.path.join(BASE_DIR, "macro_streamer.py")],
        "db": os.path.join(_DB_DIR, "macro_data.db"),
        "freshness_query": "SELECT MAX(date) FROM macro_data",
        "freshness_type": "date",
        "stale_hours": 4,
    },
    "space_weather_streamer": {
        "cmd": [sys.executable, "-u", os.path.join(BASE_DIR, "space_weather_streamer.py")],
        "db": os.path.join(_DB_DIR, "space_weather.db"),
        "freshness_query": "SELECT MAX(timestamp) FROM space_weather",
        "freshness_type": "unix",
        "stale_hours": 4,
    },
    "download_btc": {
        "cmd": [sys.executable, "-u", os.path.join(BASE_DIR, "download_btc.py"), "--daemon"],
        "db": os.path.join(_DB_DIR, "btc_prices.db"),
        "freshness_query": "SELECT MAX(open_time) FROM ohlcv",
        "freshness_type": "unix_ms",
        "stale_hours": 1,
    },
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("supervisor")
logger.setLevel(logging.DEBUG)

# File handler — supervisor.log
fh = logging.FileHandler(os.path.join(BASE_DIR, "supervisor.log"), encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(fh)

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(ch)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now_utc():
    return datetime.now(timezone.utc)


def ts_age_hours(ts_value, ts_type):
    """Return age in hours for a timestamp value, or None if unparseable."""
    try:
        if ts_value is None:
            return None
        now = time.time()
        if ts_type == "unix":
            return (now - float(ts_value)) / 3600
        elif ts_type == "unix_ms":
            return (now - float(ts_value) / 1000) / 3600
        elif ts_type == "iso":
            # Try multiple ISO formats
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z",
                        "%Y-%m-%dT%H:%M:%S+00:00", "%Y-%m-%d %H:%M:%S.%f+00:00",
                        "%Y-%m-%dT%H:%M:%S.%f+00:00"):
                try:
                    dt = datetime.strptime(str(ts_value), fmt)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return (now - dt.timestamp()) / 3600
                except ValueError:
                    continue
            # Fallback: try just the date portion
            dt = datetime.strptime(str(ts_value)[:19], "%Y-%m-%d %H:%M:%S")
            dt = dt.replace(tzinfo=timezone.utc)
            return (now - dt.timestamp()) / 3600
        elif ts_type == "date":
            dt = datetime.strptime(str(ts_value), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            return (now - dt.timestamp()) / 3600
    except Exception:
        return None


def check_db_freshness(name, cfg):
    """Check data freshness for a streamer's DB. Returns (age_hours, stale_bool)."""
    db_path = cfg["db"]
    if not os.path.exists(db_path):
        return None, True
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        cur = conn.cursor()
        cur.execute(cfg["freshness_query"])
        val = cur.fetchone()[0]
        conn.close()
        age = ts_age_hours(val, cfg["freshness_type"])
        if age is None:
            return None, True
        return age, age > cfg["stale_hours"]
    except Exception as e:
        logger.debug(f"Freshness check error for {name}: {e}")
        return None, True


# ---------------------------------------------------------------------------
# Streamer Process Wrapper
# ---------------------------------------------------------------------------

class StreamerProcess:
    def __init__(self, name, cfg):
        self.name = name
        self.cfg = cfg
        self.process = None
        self.log_file = None
        self.started_at = None
        self.restart_count = 0
        self.last_exit_code = None
        self.lock = threading.Lock()

    @property
    def backoff(self):
        idx = min(self.restart_count, len(BACKOFF_SCHEDULE) - 1)
        return BACKOFF_SCHEDULE[idx]

    @property
    def is_alive(self):
        with self.lock:
            return self.process is not None and self.process.poll() is None

    @property
    def pid(self):
        with self.lock:
            if self.process:
                return self.process.pid
        return None

    @property
    def uptime_str(self):
        if self.started_at is None:
            return "never started"
        delta = now_utc() - self.started_at
        secs = int(delta.total_seconds())
        if secs < 60:
            return f"{secs}s"
        elif secs < 3600:
            return f"{secs // 60}m {secs % 60}s"
        else:
            return f"{secs // 3600}h {(secs % 3600) // 60}m"

    def start(self):
        """Launch the streamer subprocess."""
        with self.lock:
            log_path = os.path.join(LOG_DIR, f"{self.name}.log")
            self.log_file = open(log_path, "a", encoding="utf-8", buffering=1)
            try:
                self.process = subprocess.Popen(
                    self.cfg["cmd"],
                    stdout=self.log_file,
                    stderr=subprocess.STDOUT,
                    cwd=BASE_DIR,
                    # CREATE_NEW_PROCESS_GROUP on Windows so we can signal properly
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
                )
                self.started_at = now_utc()
                logger.info(f"[{self.name}] Started (PID {self.process.pid})")
            except Exception as e:
                logger.error(f"[{self.name}] Failed to start: {e}")
                self.log_file.close()
                self.log_file = None

    def stop(self):
        """Gracefully stop the streamer."""
        with self.lock:
            if self.process and self.process.poll() is None:
                logger.info(f"[{self.name}] Sending terminate (PID {self.process.pid})")
                try:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"[{self.name}] Did not exit in 10s, killing")
                        self.process.kill()
                        self.process.wait(timeout=5)
                except Exception as e:
                    logger.error(f"[{self.name}] Error stopping: {e}")
            if self.log_file:
                try:
                    self.log_file.close()
                except Exception:
                    pass
                self.log_file = None

    def check_and_restart(self):
        """Check if process died; if so, restart with backoff. Returns True if restarted."""
        with self.lock:
            if self.process is None:
                return False
            rc = self.process.poll()
            if rc is None:
                return False  # still running
            self.last_exit_code = rc
            # Close old log file handle
            if self.log_file:
                try:
                    self.log_file.close()
                except Exception:
                    pass
                self.log_file = None

        # Process died — log and schedule restart
        logger.warning(f"[{self.name}] Crashed (exit code {self.last_exit_code}), "
                       f"restart #{self.restart_count + 1} in {self.backoff}s")
        time.sleep(self.backoff)
        self.restart_count += 1
        self.start()
        return True

    def reset_backoff(self):
        """Reset backoff counter (call when process has been running stably)."""
        self.restart_count = 0

    def status_dict(self):
        age, stale = check_db_freshness(self.name, self.cfg)
        return {
            "name": self.name,
            "pid": self.pid,
            "alive": self.is_alive,
            "uptime": self.uptime_str,
            "restart_count": self.restart_count,
            "last_exit_code": self.last_exit_code,
            "db_age_hours": round(age, 2) if age is not None else None,
            "db_stale": stale,
            "stale_threshold_hours": self.cfg["stale_hours"],
        }


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

class Supervisor:
    def __init__(self):
        self.streamers = {}
        self.shutdown_event = threading.Event()
        for name, cfg in STREAMERS.items():
            self.streamers[name] = StreamerProcess(name, cfg)

    def start_all(self):
        logger.info("=" * 60)
        logger.info("Supervisor starting all streamers")
        logger.info("=" * 60)
        for sp in self.streamers.values():
            sp.start()
            time.sleep(0.5)  # slight stagger to avoid thundering herd

    def stop_all(self):
        logger.info("Supervisor shutting down all streamers")
        threads = []
        for sp in self.streamers.values():
            t = threading.Thread(target=sp.stop, daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=15)
        logger.info("All streamers stopped")

    def monitor_loop(self):
        """Main monitoring loop — checks processes and data freshness."""
        STABLE_THRESHOLD = 600  # 10 min of uptime = reset backoff
        last_freshness_check = 0
        FRESHNESS_INTERVAL = 300  # 5 minutes

        while not self.shutdown_event.is_set():
            # Check each process
            for sp in self.streamers.values():
                if self.shutdown_event.is_set():
                    break
                if not sp.is_alive:
                    sp.check_and_restart()
                else:
                    # Reset backoff if running stably
                    if sp.started_at and (now_utc() - sp.started_at).total_seconds() > STABLE_THRESHOLD:
                        if sp.restart_count > 0:
                            logger.info(f"[{sp.name}] Stable for >{STABLE_THRESHOLD}s, resetting backoff")
                            sp.reset_backoff()

            # Periodic freshness check
            if time.time() - last_freshness_check > FRESHNESS_INTERVAL:
                last_freshness_check = time.time()
                self._check_freshness()
                self._write_status()

            self.shutdown_event.wait(timeout=5)

    def _check_freshness(self):
        logger.info("--- Data freshness check ---")
        for name, sp in self.streamers.items():
            age, stale = check_db_freshness(name, sp.cfg)
            if age is None:
                logger.warning(f"  [{name}] STALE — could not read DB or no data")
            elif stale:
                logger.warning(f"  [{name}] STALE — last data {age:.1f}h ago (threshold: {sp.cfg['stale_hours']}h)")
            else:
                logger.info(f"  [{name}] OK — last data {age:.1f}h ago")

    def _write_status(self):
        """Write status JSON for --status command to read."""
        status = {
            "updated": now_utc().isoformat(),
            "streamers": {name: sp.status_dict() for name, sp in self.streamers.items()},
        }
        try:
            with open(STATUS_FILE, "w", encoding="utf-8") as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to write status file: {e}")

    def run(self):
        """Main entry point for daemon mode."""
        # Register signal handlers
        def handle_shutdown(signum, frame):
            sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
            logger.info(f"Received {sig_name}, shutting down...")
            self.shutdown_event.set()

        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)
        # Windows doesn't have SIGHUP, but handle it on Linux
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, handle_shutdown)

        self.start_all()

        try:
            self.monitor_loop()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received")
        finally:
            self.stop_all()
            self._write_status()
            logger.info("Supervisor exited cleanly")


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------

def print_status():
    """Print streamer status from the status file."""
    if not os.path.exists(STATUS_FILE):
        print("No status file found. Is the supervisor running?")
        print(f"  Expected: {STATUS_FILE}")
        # Fall back to live DB check
        print("\nLive DB freshness check:")
        for name, cfg in STREAMERS.items():
            age, stale = check_db_freshness(name, cfg)
            tag = "STALE" if stale else "OK"
            age_str = f"{age:.1f}h" if age is not None else "N/A"
            print(f"  {name:30s}  {tag:5s}  age={age_str}  (threshold: {cfg['stale_hours']}h)")
        return

    with open(STATUS_FILE, "r", encoding="utf-8") as f:
        status = json.load(f)

    print(f"Supervisor status as of {status['updated']}")
    print("-" * 95)
    print(f"{'Streamer':<28s} {'PID':>7s} {'Alive':>6s} {'Uptime':>10s} {'Restarts':>9s} {'DB Age':>8s} {'Fresh':>6s}")
    print("-" * 95)

    for name, info in status["streamers"].items():
        pid_str = str(info["pid"]) if info["pid"] else "-"
        alive_str = "YES" if info["alive"] else "NO"
        age_str = f"{info['db_age_hours']:.1f}h" if info["db_age_hours"] is not None else "N/A"
        fresh_str = "OK" if not info["db_stale"] else "STALE"
        print(f"  {name:<26s} {pid_str:>7s} {alive_str:>6s} {info['uptime']:>10s} "
              f"{info['restart_count']:>9d} {age_str:>8s} {fresh_str:>6s}")

    print("-" * 95)

    # Also do a live freshness check
    print("\nLive DB freshness (current):")
    for name, cfg in STREAMERS.items():
        age, stale = check_db_freshness(name, cfg)
        tag = "STALE" if stale else "OK"
        age_str = f"{age:.1f}h" if age is not None else "N/A"
        print(f"  {name:30s}  {tag:5s}  age={age_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--status" in sys.argv:
        print_status()
    else:
        supervisor = Supervisor()
        supervisor.run()
