#!/bin/bash
#
# rollback.sh — Quick rollback wrapper for deploy_model.py
#
# USAGE:
#   ./rollback.sh 1w           # Rollback 1w to previous version
#   ./rollback.sh 1w shadow    # Rollback 1w to shadow version
#   ./rollback.sh all          # Rollback ALL timeframes to previous
#
# SAFETY:
# - Always prompts for confirmation before rollback
# - Writes to audit log
# - Atomic symlink swap (no downtime)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_SCRIPT="$SCRIPT_DIR/deploy_model.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

TF="${1:-}"
TARGET="${2:-previous}"
VALID_TFS=("1w" "1d" "4h" "1h" "15m" "all")

usage() {
    echo "Usage: $0 <timeframe> [target]"
    echo ""
    echo "Arguments:"
    echo "  timeframe   Timeframe to rollback (1w, 1d, 4h, 1h, 15m, all)"
    echo "  target      Rollback target (previous or shadow, default: previous)"
    echo ""
    echo "Examples:"
    echo "  $0 1w           # Rollback 1w to previous version"
    echo "  $0 1w shadow    # Rollback 1w to shadow version"
    echo "  $0 all          # Rollback ALL timeframes to previous"
    exit 1
}

# Check arguments
if [[ -z "$TF" ]]; then
    usage
fi

if [[ ! " ${VALID_TFS[@]} " =~ " ${TF} " ]]; then
    echo -e "${RED}❌ ERROR: Invalid timeframe '$TF'${NC}"
    echo "Valid timeframes: ${VALID_TFS[@]}"
    exit 1
fi

if [[ "$TARGET" != "previous" && "$TARGET" != "shadow" ]]; then
    echo -e "${RED}❌ ERROR: Invalid target '$TARGET'${NC}"
    echo "Valid targets: previous, shadow"
    exit 1
fi

# Rollback a single timeframe
rollback_tf() {
    local tf="$1"
    local target="$2"

    echo -e "${YELLOW}🔁 Preparing to rollback ${tf} to ${target}...${NC}"

    # Show current state
    python "$DEPLOY_SCRIPT" list --tf "$tf"

    # Confirm
    read -p "Are you sure you want to rollback $tf to $target? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}❌ Rollback cancelled${NC}"
        exit 1
    fi

    # Execute rollback
    python "$DEPLOY_SCRIPT" rollback --tf "$tf" --target "$target"

    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✅ Rollback complete for ${tf}${NC}"
        echo -e "${YELLOW}⚠️  IMPORTANT: Restart inference service to load new model${NC}"
    else
        echo -e "${RED}❌ Rollback failed for ${tf}${NC}"
        exit 1
    fi
}

# Rollback all timeframes
rollback_all() {
    local target="$1"
    local tfs=("1w" "1d" "4h" "1h" "15m")

    echo -e "${YELLOW}🔁 Preparing to rollback ALL timeframes to ${target}...${NC}"
    echo ""

    # Show current state for all timeframes
    for tf in "${tfs[@]}"; do
        echo -e "${YELLOW}== $tf ==${NC}"
        python "$DEPLOY_SCRIPT" list --tf "$tf"
        echo ""
    done

    # Confirm
    read -p "Are you sure you want to rollback ALL timeframes to $target? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}❌ Rollback cancelled${NC}"
        exit 1
    fi

    # Execute rollback for each timeframe
    for tf in "${tfs[@]}"; do
        echo -e "${YELLOW}== Rolling back $tf ==${NC}"
        python "$DEPLOY_SCRIPT" rollback --tf "$tf" --target "$target"
        echo ""
    done

    echo -e "${GREEN}✅ Rollback complete for ALL timeframes${NC}"
    echo -e "${YELLOW}⚠️  IMPORTANT: Restart inference service to load new models${NC}"
}

# Main execution
if [[ "$TF" == "all" ]]; then
    rollback_all "$TARGET"
else
    rollback_tf "$TF" "$TARGET"
fi
