#!/bin/bash
# v3.3 Cloud Setup — no Docker needed
# Usage: scp this + code.tar.gz + dbs.tar.gz to machine, then run: bash setup.sh

set -e

echo "=== v3.3 Setup ==="
echo "Cores: $(nproc) | RAM: $(free -g | awk '/Mem/{print $2}')GB"

# Install deps
echo ">>> Installing Python packages..."
pip install -q xgboost lightgbm scikit-learn scipy ephem astropy pytz joblib 2>&1 | tail -3

# Unpack code
if [ -f /workspace/code.tar.gz ]; then
    echo ">>> Unpacking code..."
    cd /workspace
    tar xzf code.tar.gz
    echo "Code ready: $(ls /workspace/v3.2_2.9M_Features/*.py | wc -l) Python files"
fi

# Unpack DBs
if [ -f /workspace/dbs.tar.gz ]; then
    echo ">>> Unpacking databases..."
    cd /workspace
    tar xzf dbs.tar.gz
    echo "DBs ready: $(ls /workspace/v3.2_2.9M_Features/*.db 2>/dev/null | wc -l) databases"
fi

# Symlinks for legacy paths
ln -sf /workspace/v3.2_2.9M_Features/*.db /workspace/ 2>/dev/null || true

echo ""
echo "=== READY ==="
echo "Run: cd /workspace/v3.2_2.9M_Features && python -u cloud_run_tf.py --symbol BTC --tf 1w"
