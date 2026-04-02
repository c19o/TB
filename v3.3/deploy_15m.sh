#!/bin/bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
python -X utf8 "$PROJECT_DIR/v3.3/deploy_tf.py" --tf 15m "$@"
