#!/usr/bin/env python3
"""
model_server.py — Blue/Green Model Inference Server with Shadow Mode

FEATURES:
- Loads models from symlink registry (active/shadow/previous)
- Always serves from ACTIVE model
- Runs SHADOW model in parallel on configurable % of traffic
- Logs shadow predictions for offline validation (48hr soak test)
- Atomic model reloading without service restart
- Per-timeframe model management

USAGE:
    # Start server for a single timeframe
    python model_server.py --tf 1w --shadow-pct 0.1

    # Start server for all timeframes
    python model_server.py --all --shadow-pct 0.1

    # In Python (programmatic usage)
    from model_server import ModelServer
    server = ModelServer(tf='1w', shadow_pct=0.1)
    prediction = server.predict(features, request_id='req123')

SHADOW MODE:
- Shadow model runs on shadow_pct % of requests (default 10%)
- Shadow predictions are logged to models/{tf}/shadow_log.jsonl
- Does NOT affect served predictions (always from active)
- Used for 48hr soak testing before promotion

PROMOTION:
- After 48hr shadow soak, analyze shadow_log.jsonl
- If shadow performs well, promote: ./deploy_model.py promote --tf {tf}
- Server auto-reloads models (checks symlinks every 60s)

DIRECTORY LAYOUT:
    models/
      1w/
        active -> versions/lgbm_1w_v20260401_.../
        shadow -> versions/lgbm_1w_v20260328_.../
        shadow_log.jsonl
        shadow_metrics.json
      ...

LOG FORMAT (shadow_log.jsonl):
    {
      "timestamp": "2026-04-01T10:30:00Z",
      "request_id": "req123",
      "tf": "1w",
      "active_version": "lgbm_1w_v20260401_...",
      "shadow_version": "lgbm_1w_v20260328_...",
      "active_pred": [0.2, 0.5, 0.3],  # 3-class probabilities
      "shadow_pred": [0.25, 0.45, 0.3],
      "delta_l1": 0.15  # L1 distance between predictions
    }

"""
import os
import sys
import json
import time
import hashlib
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

try:
    import lightgbm as lgb
    import numpy as np
except ImportError as e:
    print(f"ERROR: Required package not installed: {e}")
    print("Install with: pip install lightgbm numpy")
    sys.exit(1)

# Constants
MODELS_BASE_DIR = Path(__file__).parent.parent / 'models'
RELOAD_INTERVAL_SEC = 60  # Check for model updates every 60s
VALID_TIMEFRAMES = ['1w', '1d', '4h', '1h', '15m']


class ModelSlot:
    """Represents a single model slot (active/shadow/previous)."""

    def __init__(self, tf: str, slot_name: str):
        self.tf = tf
        self.slot_name = slot_name
        self.tf_dir = MODELS_BASE_DIR / tf
        self.symlink_path = self.tf_dir / slot_name
        self.model: Optional[lgb.Booster] = None
        self.version_id: Optional[str] = None
        self.features: Optional[List[str]] = None
        self.last_load_time: float = 0

    def get_current_version(self) -> Optional[str]:
        """Get the version ID currently pointed to by the symlink."""
        if not self.symlink_path.exists() or not self.symlink_path.is_symlink():
            return None
        target = os.readlink(self.symlink_path)
        return Path(target).name

    def needs_reload(self) -> bool:
        """Check if model needs to be reloaded."""
        current_version = self.get_current_version()

        # No symlink = no model
        if current_version is None:
            return self.model is not None  # Unload if we had a model

        # Version changed = reload
        if current_version != self.version_id:
            return True

        # Periodically reload to catch file changes
        if time.time() - self.last_load_time > RELOAD_INTERVAL_SEC:
            return True

        return False

    def load(self) -> bool:
        """Load (or reload) the model. Returns True if successful."""
        current_version = self.get_current_version()

        # No symlink = unload
        if current_version is None:
            if self.model is not None:
                print(f"  [{self.tf}:{self.slot_name}] Unloading (symlink removed)")
                self.model = None
                self.version_id = None
                self.features = None
            return False

        # Load model
        model_path = self.symlink_path / 'model.json'
        if not model_path.exists():
            print(f"  [{self.tf}:{self.slot_name}] ERROR: model.json not found in {current_version}")
            return False

        try:
            # Load model
            model = lgb.Booster(model_file=str(model_path))

            # Load features if available
            features_path = self.symlink_path / 'features_all.json'
            features = None
            if features_path.exists():
                with open(features_path, 'r') as f:
                    features = json.load(f)

            # Update state
            self.model = model
            self.version_id = current_version
            self.features = features
            self.last_load_time = time.time()

            print(f"  [{self.tf}:{self.slot_name}] Loaded: {current_version}")
            if features:
                print(f"    Features: {len(features):,}")

            return True

        except Exception as e:
            print(f"  [{self.tf}:{self.slot_name}] ERROR loading {current_version}: {e}")
            return False

    def predict(self, features: np.ndarray) -> Optional[np.ndarray]:
        """Get prediction from this model slot. Returns None if no model loaded."""
        if self.model is None:
            return None

        try:
            return self.model.predict(features)
        except Exception as e:
            print(f"  [{self.tf}:{self.slot_name}] Prediction error: {e}")
            return None


class ModelServer:
    """Blue/Green model server with shadow mode testing."""

    def __init__(self, tf: str, shadow_pct: float = 0.1, auto_reload: bool = True):
        if tf not in VALID_TIMEFRAMES:
            raise ValueError(f"Invalid timeframe '{tf}'. Valid: {VALID_TIMEFRAMES}")

        self.tf = tf
        self.shadow_pct = max(0.0, min(1.0, shadow_pct))  # Clamp to [0, 1]
        self.auto_reload = auto_reload

        # Create model slots
        self.active = ModelSlot(tf, 'active')
        self.shadow = ModelSlot(tf, 'shadow')
        self.previous = ModelSlot(tf, 'previous')

        # Shadow logging
        self.tf_dir = MODELS_BASE_DIR / tf
        self.tf_dir.mkdir(parents=True, exist_ok=True)
        self.shadow_log_path = self.tf_dir / 'shadow_log.jsonl'
        self.shadow_metrics_path = self.tf_dir / 'shadow_metrics.json'

        # Shadow metrics tracking
        self.shadow_request_count = 0
        self.shadow_comparison_count = 0
        self.shadow_total_l1_delta = 0.0

        # Initial load
        print(f"\n[{self.tf}] Initializing model server...")
        self.reload_if_needed()

        # Start auto-reload thread if enabled
        if self.auto_reload:
            import threading
            self.reload_thread = threading.Thread(target=self._auto_reload_loop, daemon=True)
            self.reload_thread.start()

    def reload_if_needed(self):
        """Check and reload models if needed."""
        reloaded = []

        if self.active.needs_reload():
            if self.active.load():
                reloaded.append('active')

        if self.shadow.needs_reload():
            if self.shadow.load():
                reloaded.append('shadow')

        if self.previous.needs_reload():
            if self.previous.load():
                reloaded.append('previous')

        if reloaded:
            print(f"[{self.tf}] Reloaded: {', '.join(reloaded)}")

    def _auto_reload_loop(self):
        """Background thread that periodically reloads models."""
        while True:
            time.sleep(RELOAD_INTERVAL_SEC)
            try:
                self.reload_if_needed()
            except Exception as e:
                print(f"[{self.tf}] Auto-reload error: {e}")

    def predict(self, features: np.ndarray, request_id: str = None) -> Optional[np.ndarray]:
        """
        Get prediction from ACTIVE model.
        Optionally runs SHADOW model in parallel for logging (shadow mode).

        Args:
            features: Feature array (sparse CSR or dense)
            request_id: Optional request ID for logging

        Returns:
            Prediction from ACTIVE model (3-class probabilities)
        """
        # Always serve from active
        active_pred = self.active.predict(features)
        if active_pred is None:
            print(f"[{self.tf}] ERROR: Active model prediction failed")
            return None

        # Shadow mode: run shadow model on a % of requests
        self.shadow_request_count += 1

        if self.shadow.model is not None and self.shadow_pct > 0:
            # Probabilistic sampling
            import random
            if random.random() < self.shadow_pct:
                shadow_pred = self.shadow.predict(features)

                if shadow_pred is not None:
                    # Log shadow comparison
                    self._log_shadow_comparison(
                        request_id=request_id or f"req{self.shadow_request_count}",
                        active_pred=active_pred,
                        shadow_pred=shadow_pred
                    )

        return active_pred

    def _log_shadow_comparison(self, request_id: str, active_pred: np.ndarray, shadow_pred: np.ndarray):
        """Log a shadow mode prediction comparison."""
        # Compute L1 delta
        delta_l1 = float(np.abs(active_pred - shadow_pred).sum())

        # Create log entry
        log_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'request_id': request_id,
            'tf': self.tf,
            'active_version': self.active.version_id,
            'shadow_version': self.shadow.version_id,
            'active_pred': active_pred.tolist(),
            'shadow_pred': shadow_pred.tolist(),
            'delta_l1': delta_l1
        }

        # Append to log file
        try:
            with open(self.shadow_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

            # Update metrics
            self.shadow_comparison_count += 1
            self.shadow_total_l1_delta += delta_l1

            # Periodically save metrics summary
            if self.shadow_comparison_count % 100 == 0:
                self._save_shadow_metrics()

        except Exception as e:
            print(f"[{self.tf}] Shadow logging error: {e}")

    def _save_shadow_metrics(self):
        """Save shadow mode metrics summary."""
        if self.shadow_comparison_count == 0:
            return

        avg_delta = self.shadow_total_l1_delta / self.shadow_comparison_count

        metrics = {
            'tf': self.tf,
            'active_version': self.active.version_id,
            'shadow_version': self.shadow.version_id,
            'total_requests': self.shadow_request_count,
            'shadow_comparisons': self.shadow_comparison_count,
            'shadow_pct_actual': self.shadow_comparison_count / self.shadow_request_count,
            'avg_l1_delta': avg_delta,
            'last_updated': datetime.now(timezone.utc).isoformat()
        }

        try:
            with open(self.shadow_metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            print(f"[{self.tf}] Metrics save error: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current server status."""
        return {
            'tf': self.tf,
            'shadow_pct': self.shadow_pct,
            'active': {
                'version': self.active.version_id,
                'loaded': self.active.model is not None,
                'features': len(self.active.features) if self.active.features else 0
            },
            'shadow': {
                'version': self.shadow.version_id,
                'loaded': self.shadow.model is not None,
                'features': len(self.shadow.features) if self.shadow.features else 0
            },
            'previous': {
                'version': self.previous.version_id,
                'loaded': self.previous.model is not None
            },
            'shadow_stats': {
                'total_requests': self.shadow_request_count,
                'shadow_comparisons': self.shadow_comparison_count,
                'avg_l1_delta': self.shadow_total_l1_delta / self.shadow_comparison_count if self.shadow_comparison_count > 0 else 0.0
            }
        }


def main():
    parser = argparse.ArgumentParser(description='V3.3 Blue/Green Model Inference Server')
    parser.add_argument('--tf', choices=VALID_TIMEFRAMES, help='Timeframe to serve')
    parser.add_argument('--all', action='store_true', help='Serve all timeframes')
    parser.add_argument('--shadow-pct', type=float, default=0.1,
                       help='Shadow mode traffic percentage (0.0-1.0, default 0.1)')
    parser.add_argument('--status', action='store_true', help='Print status and exit')

    args = parser.parse_args()

    if not args.tf and not args.all:
        parser.print_help()
        sys.exit(1)

    # Determine which timeframes to serve
    timeframes = VALID_TIMEFRAMES if args.all else [args.tf]

    # Create servers
    servers = {}
    for tf in timeframes:
        servers[tf] = ModelServer(tf, shadow_pct=args.shadow_pct)

    # Status mode
    if args.status:
        for tf, server in servers.items():
            print(f"\n=== {tf} ===")
            status = server.get_status()
            print(json.dumps(status, indent=2))
        sys.exit(0)

    # Keep alive
    print(f"\n✅ Model server running for: {', '.join(timeframes)}")
    print(f"   Shadow mode: {args.shadow_pct*100:.1f}% of traffic")
    print(f"   Auto-reload: every {RELOAD_INTERVAL_SEC}s")
    print("\nPress Ctrl+C to stop\n")

    try:
        while True:
            time.sleep(10)
            # Periodically print status
            for tf, server in servers.items():
                status = server.get_status()
                print(f"[{tf}] Requests: {status['shadow_stats']['total_requests']}, "
                      f"Shadow: {status['shadow_stats']['shadow_comparisons']}, "
                      f"Avg Δ: {status['shadow_stats']['avg_l1_delta']:.4f}")
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        sys.exit(0)


if __name__ == '__main__':
    main()
