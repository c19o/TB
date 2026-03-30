"""
V2 Streamer Supervisor — manages all 7 data streamer processes for Savage22 V2.

Usage:
    python streamer_supervisor.py                    # Start all streamers
    python streamer_supervisor.py --status           # Check status from JSON
    python streamer_supervisor.py --exclude tweet    # Skip tweet_streamer

Manages: crypto, tweet, news, sports, space_weather, macro, v2_easy_streamers
Auto-restarts crashed processes with exponential backoff.
Checks data freshness every 5 minutes.
Writes supervisor_status.json for dashboard consumption.
"""

import sys
import os
import time
import signal
import subprocess
import threading
import sqlite3
import json
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
STATUS_FILE = os.path.join(BASE_DIR, "supervisor_status.json")

BACKOFF_SCHEDULE = [5, 15, 60, 300]  # seconds between restarts

STREAMERS = {
    "crypto_streamer": {
        "script": "crypto_streamer.py",
        "db": os.path.join(BASE_DIR, "onchain_data.db"),
        "freshness_query": "SELECT MAX(timestamp) FROM onchain_data",
        "freshness_type": "iso",
        "stale_minutes": 60,
    },
    "tweet_streamer": {
        "script": "tweet_streamer.py",
        "db": os.path.join(BASE_DIR, "tweets.db"),
        "freshness_query": "SELECT MAX(ts_unix) FROM tweets",
        "freshness_type": "unix",
        "stale_minutes": 15,
    },
    "news_streamer": {
        "script": "news_streamer.py",
        "db": os.path.join(BASE_DIR, "news_articles.db"),
        "freshness_query": "SELECT MAX(ts_unix) FROM streamer_articles",
        "freshness_type": "unix",
        "stale_minutes": 60,
    },
    "sports_streamer": {
        "script": "sports_streamer.py",
        "db": os.path.join(BASE_DIR, "sports_results.db"),
        "freshness_query": "SELECT MAX(inserted_at) FROM games",
        "freshness_type": "iso",
        "stale_minutes": 240,
    },
    "space_weather_streamer": {
        "script": "space_weather_streamer.py",
        "db": os.path.join(BASE_DIR, "space_weather.db"),
        "freshness_query": "SELECT MAX(timestamp) FROM space_weather",
        "freshness_type": "unix",
        "stale_minutes": 240,
    },
    "macro_streamer": {
        "script": "macro_streamer.py",
        "db": os.path.join(BASE_DIR, "macro_data.db"),
        "freshness_query": "SELECT MAX(date) FROM macro_data",
        "freshness_type": "date",
        "stale_minutes": 240,
    },
    "v2_easy_streamers": {
        "script": "v2_easy_streamers.py",
        "db": os.path.join(BASE_DIR, "v2_signals.db"),
        "freshness_query": "SELECT MAX(date) FROM defi_tvl",
        "freshness_type": "date",
        "stale_minutes": 1440,  # daily data, 24h threshold
    },
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("v2_supervisor")
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


def ts_age_minutes(ts_value, ts_type):
    """Return age in minutes for a timestamp value, or None if unparseable."""
    try:
        if ts_value is None:
            return None
        now = time.time()
        if ts_type == "unix":
            return (now - float(ts_value)) / 60
        elif ts_type == "unix_ms":
            return (now - float(ts_value) / 1000) / 60
        elif ts_type == "iso":
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                        "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S+00:00",
                        "%Y-%m-%d %H:%M:%S.%f+00:00", "%Y-%m-%dT%H:%M:%S.%f+00:00"):
                try:
                    dt = datetime.strptime(str(ts_value), fmt)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return (now - dt.timestamp()) / 60
                except ValueError:
                    continue
            # Fallback: try just first 19 chars
            dt = datetime.strptime(str(ts_value)[:19], "%Y-%m-%d %H:%M:%S")
            dt = dt.replace(tzinfo=timezone.utc)
            return (now - dt.timestamp()) / 60
        elif ts_type == "date":
            dt = datetime.strptime(str(ts_value), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            return (now - dt.timestamp()) / 60
    except Exception:
        return None


def check_db_freshness(name, cfg):
    """Check data freshness for a streamer's DB. Returns (age_minutes, stale_bool, last_ts_str)."""
    db_path = cfg["db"]
    if not os.path.exists(db_path):
        return None, True, None
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        cur = conn.cursor()
        cur.execute(cfg["freshness_query"])
        val = cur.fetchone()[0]
        conn.close()
        age = ts_age_minutes(val, cfg["freshness_type"])
        if age is None:
            return None, True, None
        # Build a human-readable last_data timestamp
        last_ts = str(val) if val is not None else None
        return age, age > cfg["stale_minutes"], last_ts
    except Exception as e:
        logger.debug(f"Freshness check error for {name}: {e}")
        return None, True, None


def format_age(minutes):
    """Format age in minutes to human-readable string."""
    if minutes is None:
        return "N/A"
    if minutes < 60:
        return f"{minutes:.0f}m"
    elif minutes < 1440:
        return f"{minutes / 60:.1f}h"
    else:
        return f"{minutes / 1440:.1f}d"


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
        """Launch the streamer subprocess with PYTHONUNBUFFERED=1."""
        with self.lock:
            log_path = os.path.join(LOG_DIR, f"{self.name}.log")
            self.log_file = open(log_path, "a", encoding="utf-8", buffering=1)

            script_path = os.path.join(BASE_DIR, self.cfg["script"])
            cmd = [sys.executable, "-u", script_path]

            # Ensure PYTHONUNBUFFERED=1 in env
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            try:
                kwargs = {
                    "stdout": self.log_file,
                    "stderr": subprocess.STDOUT,
                    "cwd": BASE_DIR,
                    "env": env,
                }
                # Windows: CREATE_NEW_PROCESS_GROUP for proper signal handling
                if sys.platform == "win32":
                    kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

                self.process = subprocess.Popen(cmd, **kwargs)
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
            if self.log_file:
                try:
                    self.log_file.close()
                except Exception:
                    pass
                self.log_file = None

        # Process died — log and schedule restart
        wait = self.backoff
        logger.warning(f"[{self.name}] Crashed (exit code {self.last_exit_code}), "
                       f"restart #{self.restart_count + 1} in {wait}s")
        time.sleep(wait)
        self.restart_count += 1
        self.start()
        return True

    def reset_backoff(self):
        """Reset backoff counter (call when process has been running stably)."""
        self.restart_count = 0

    def status_dict(self):
        """Build status dict for JSON output."""
        age, stale, last_ts = check_db_freshness(self.name, self.cfg)
        return {
            "pid": self.pid,
            "status": "running" if self.is_alive else "stopped",
            "restarts": self.restart_count,
            "last_data": last_ts,
            "uptime": self.uptime_str,
            "last_exit_code": self.last_exit_code,
            "data_age": format_age(age),
            "data_stale": stale,
            "stale_threshold_minutes": self.cfg["stale_minutes"],
        }


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

class Supervisor:
    def __init__(self, exclude=None):
        self.streamers = {}
        self.shutdown_event = threading.Event()
        self.started_at = now_utc()

        exclude = set(exclude or [])
        for name, cfg in STREAMERS.items():
            # Match by full name or partial (e.g. "tweet" matches "tweet_streamer")
            skip = False
            for ex in exclude:
                ex_lower = ex.lower().replace(".py", "")
                if ex_lower in name.lower():
                    skip = True
                    break
            if skip:
                logger.info(f"[{name}] Excluded via --exclude")
                continue
            self.streamers[name] = StreamerProcess(name, cfg)

    def start_all(self):
        logger.info("=" * 60)
        logger.info("V2 Streamer Supervisor starting")
        logger.info(f"Managing {len(self.streamers)} streamers: {', '.join(self.streamers.keys())}")
        logger.info("=" * 60)
        for sp in self.streamers.values():
            sp.start()
            time.sleep(0.5)  # slight stagger to avoid thundering herd

    def stop_all(self):
        logger.info("Supervisor shutting down all streamers...")
        threads = []
        for sp in self.streamers.values():
            t = threading.Thread(target=sp.stop, daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=15)
        logger.info("All streamers stopped")

    def monitor_loop(self):
        """Main monitoring loop — health checks every 30s, freshness every 5 min."""
        HEALTH_INTERVAL = 30       # seconds between health checks
        FRESHNESS_INTERVAL = 300   # seconds between freshness checks
        STABLE_THRESHOLD = 600     # 10 min uptime = reset backoff

        last_health = 0
        last_freshness = 0

        while not self.shutdown_event.is_set():
            now = time.time()

            # Health check every 30 seconds
            if now - last_health >= HEALTH_INTERVAL:
                last_health = now
                for sp in self.streamers.values():
                    if self.shutdown_event.is_set():
                        break
                    if not sp.is_alive:
                        sp.check_and_restart()
                    else:
                        # Reset backoff if running stably for 10+ minutes
                        if (sp.started_at and sp.restart_count > 0
                                and (now_utc() - sp.started_at).total_seconds() > STABLE_THRESHOLD):
                            logger.info(f"[{sp.name}] Stable for >{STABLE_THRESHOLD}s, resetting backoff")
                            sp.reset_backoff()

            # Freshness check every 5 minutes
            if now - last_freshness >= FRESHNESS_INTERVAL:
                last_freshness = now
                self._check_freshness()
                self._write_status()

            # Sleep in small increments so shutdown is responsive
            self.shutdown_event.wait(timeout=5)

    def _check_freshness(self):
        logger.info("--- Data freshness check ---")
        for name, sp in self.streamers.items():
            age, stale, last_ts = check_db_freshness(name, sp.cfg)
            age_str = format_age(age)
            threshold_str = format_age(sp.cfg["stale_minutes"])
            if age is None:
                logger.warning(f"  [{name}] STALE -- could not read DB or no data")
            elif stale:
                logger.warning(f"  [{name}] STALE -- last data {age_str} ago (threshold: {threshold_str})")
            else:
                logger.info(f"  [{name}] OK -- last data {age_str} ago")

    def _write_status(self):
        """Write status JSON for dashboard consumption."""
        status = {
            "started_at": self.started_at.isoformat(),
            "updated_at": now_utc().isoformat(),
            "streamers": {name: sp.status_dict() for name, sp in self.streamers.items()},
        }
        try:
            tmp = STATUS_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(status, f, indent=2)
            # Atomic rename (works on Windows with same drive)
            if os.path.exists(STATUS_FILE):
                os.remove(STATUS_FILE)
            os.rename(tmp, STATUS_FILE)
        except Exception as e:
            logger.debug(f"Failed to write status file: {e}")

    def run(self):
        """Main entry point."""
        # Register signal handlers
        def handle_shutdown(signum, frame):
            sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
            logger.info(f"Received {sig_name}, shutting down...")
            self.shutdown_event.set()

        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)
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
# Status display (--status)
# ---------------------------------------------------------------------------

def print_status():
    """Print streamer status from the status JSON file."""
    if not os.path.exists(STATUS_FILE):
        print("No status file found. Is the supervisor running?")
        print(f"  Expected: {STATUS_FILE}")
        print("\nLive DB freshness check:")
        for name, cfg in STREAMERS.items():
            age, stale, last_ts = check_db_freshness(name, cfg)
            tag = "STALE" if stale else "OK"
            age_str = format_age(age)
            threshold_str = format_age(cfg["stale_minutes"])
            print(f"  {name:30s}  {tag:5s}  age={age_str:>8s}  (threshold: {threshold_str})")
        return

    with open(STATUS_FILE, "r", encoding="utf-8") as f:
        status = json.load(f)

    print(f"V2 Supervisor Status")
    print(f"  Started:  {status.get('started_at', '?')}")
    print(f"  Updated:  {status.get('updated_at', '?')}")
    print("-" * 100)
    print(f"  {'Streamer':<26s} {'PID':>7s} {'Status':>8s} {'Uptime':>10s} "
          f"{'Restarts':>9s} {'Data Age':>10s} {'Fresh':>6s}")
    print("-" * 100)

    for name, info in status["streamers"].items():
        pid_str = str(info.get("pid", "-")) if info.get("pid") else "-"
        status_str = info.get("status", "?")
        uptime_str = info.get("uptime", "?")
        restarts = info.get("restarts", 0)
        age_str = info.get("data_age", "N/A")
        fresh_str = "OK" if not info.get("data_stale", True) else "STALE"
        print(f"  {name:<26s} {pid_str:>7s} {status_str:>8s} {uptime_str:>10s} "
              f"{restarts:>9d} {age_str:>10s} {fresh_str:>6s}")

    print("-" * 100)

    # Live freshness
    print("\nLive DB freshness (current):")
    for name, cfg in STREAMERS.items():
        age, stale, last_ts = check_db_freshness(name, cfg)
        tag = "STALE" if stale else "OK"
        age_str = format_age(age)
        print(f"  {name:30s}  {tag:5s}  age={age_str:>8s}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V2 Streamer Supervisor")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    parser.add_argument("--exclude", nargs="*", default=[], help="Streamers to skip (partial match)")
    args = parser.parse_args()

    if args.status:
        print_status()
    else:
        supervisor = Supervisor(exclude=args.exclude)
        supervisor.run()
