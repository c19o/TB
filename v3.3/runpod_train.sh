#!/bin/bash
# runpod_train.sh -- Manual RunPod training wrapper for Savage22
# Usage:
#   bash runpod_train.sh              # Full pipeline
#   bash runpod_train.sh --dry-run    # Show plan only
#   bash runpod_train.sh --skip-upload # Skip data upload
#
# Prerequisites:
#   pip install runpod paramiko scp python-dotenv

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo "Savage22 RunPod Training Pipeline"
echo "============================================================"
echo "Project dir: $SCRIPT_DIR"
echo "Time: $(date)"
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    echo "ERROR: python not found in PATH"
    exit 1
fi

# Check required packages
python -c "import runpod" 2>/dev/null || {
    echo "Installing runpod SDK..."
    pip install runpod
}

python -c "from dotenv import load_dotenv" 2>/dev/null || {
    echo "Installing python-dotenv..."
    pip install python-dotenv
}

# Check .env
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo "ERROR: .env file not found in $SCRIPT_DIR"
    exit 1
fi

# Check SSH key
SSH_KEY="$HOME/.ssh/id_ed25519"
if [ ! -f "$SSH_KEY" ]; then
    echo "WARNING: SSH key not found at $SSH_KEY"
    echo "Make sure your public key is added to RunPod console (Settings -> SSH Keys)"
fi

# Check that feature databases exist
echo "Checking data files..."
MISSING=0
for DB in features_5m.db features_15m.db features_1h.db features_4h.db features_1d.db features_1w.db btc_prices.db; do
    if [ -f "$SCRIPT_DIR/$DB" ]; then
        SIZE=$(du -h "$SCRIPT_DIR/$DB" | cut -f1)
        echo "  [OK] $DB ($SIZE)"
    else
        echo "  [MISSING] $DB"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -gt 0 ]; then
    echo ""
    echo "WARNING: $MISSING database files missing. Training may fail."
    if [ "$1" != "--dry-run" ]; then
        read -p "Continue anyway? (y/N): " RESP
        if [ "$RESP" != "y" ] && [ "$RESP" != "Y" ]; then
            echo "Aborted."
            exit 1
        fi
    fi
fi

echo ""
echo "Starting pipeline..."
echo ""

# Run the Python script with all arguments
python "$SCRIPT_DIR/runpod_train.py" "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "Pipeline complete!"
    echo "Results: $SCRIPT_DIR/runpod_output/"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "Pipeline exited with code $EXIT_CODE"
    echo "Check https://www.console.runpod.io/pods for orphaned pods!"
    echo "============================================================"
fi

exit $EXIT_CODE
