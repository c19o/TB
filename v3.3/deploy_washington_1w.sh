#!/bin/bash
# ============================================================================
# deploy_washington_1w.sh - Deploy 1w training on approved Washington machine.
# ============================================================================
# This script does NOT rent the machine. Set INSTANCE_ID to a pre-rented instance.
# Usage:
#   INSTANCE_ID=<instance_id> ./v3.3/deploy_washington_1w.sh [--dry-run]
#
# Sequence:
#   1. build_features_v2.py
#   2. v2_cross_generator.py
#   3. ml_multi_tf.py baseline
#   4. run_optuna_local.py
#   5. ml_multi_tf.py retrain
#   6. exhaustive_optimizer.py
#
# This intentionally avoids cloud_run_tf.py one-shot orchestration for 1w auditability.
# ============================================================================

set -euo pipefail

INSTANCE_ID="${INSTANCE_ID:-}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
V33_DIR="$PROJECT_DIR/v3.3"
REMOTE_DIR="/workspace"
REMOTE_RELEASES_DIR="$REMOTE_DIR/releases"
REMOTE_RUNS_DIR="$REMOTE_DIR/runs"
REMOTE_ARTIFACTS_DIR="$REMOTE_DIR/artifacts"
REMOTE_CACHE_DIR="$REMOTE_DIR/cache"
TF="1w"
SESSION_NAME="train_1w"
RUN_ID="1w-${INSTANCE_ID}-$(date -u +%Y%m%dT%H%M%SZ)"
REMOTE_RELEASE_DIR="$REMOTE_RELEASES_DIR/v3.3_${RUN_ID}"
REMOTE_RELEASE_STAGING="${REMOTE_RELEASE_DIR}.staging"
REMOTE_CURRENT_LINK="$REMOTE_DIR/current_v3.3"
REMOTE_RUN_DIR="$REMOTE_RUNS_DIR/$RUN_ID"
REMOTE_ARTIFACT_ROOT="$REMOTE_ARTIFACTS_DIR/$RUN_ID"
REMOTE_HB="$REMOTE_RUN_DIR/cloud_run_1w_heartbeat.json"

TARGET_TEMPLATE_ID="33923286"
TARGET_REGION="Washington, US"
SSH_HOST_OVERRIDE="${SSH_HOST_OVERRIDE:-}"
SSH_PORT_OVERRIDE="${SSH_PORT_OVERRIDE:-}"
ALLOW_TEMPLATE_MISMATCH="${ALLOW_TEMPLATE_MISMATCH:-0}"
ALLOW_REGION_MISMATCH="${ALLOW_REGION_MISMATCH:-0}"
SSH_BIN="${SSH_BIN:-}"
SCP_BIN="${SCP_BIN:-}"
USE_GCS_SHARED_DB="${USE_GCS_SHARED_DB:-1}"
GCS_PROJECT_ID="${GCS_PROJECT_ID:-tbtb-492116}"
GCS_BUCKET="${GCS_BUCKET:-tbtb}"
GCS_PREFIX="${GCS_PREFIX:-savage22/shared-db/latest}"
GCS_KEY_PATH="${GCS_KEY_PATH:-}"
GCS_SHARED_SEED_PROJECT_ID=""
GCS_SHARED_SEED_BUCKET=""
GCS_SHARED_SEED_PREFIX=""
GCS_SHARED_SEED_MANIFEST_SHA256=""
GCS_SHARED_SEED_FILE_COUNT=""
REMOTE_GCS_KEY_PATH="$REMOTE_RUN_DIR/gcs_service_account.json"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

DRY_RUN=false
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=true
done

LOCAL_PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$LOCAL_PYTHON_BIN" ]]; then
    if command -v python >/dev/null 2>&1; then
        LOCAL_PYTHON_BIN="$(command -v python)"
    elif command -v python3 >/dev/null 2>&1; then
        LOCAL_PYTHON_BIN="$(command -v python3)"
    else
        echo "ERROR: python/python3 not found on local machine"
        exit 1
    fi
fi

LOCAL_VASTAI_BIN="${VASTAI_BIN:-}"
if [[ -z "$LOCAL_VASTAI_BIN" ]]; then
    if command -v vastai >/dev/null 2>&1; then
        LOCAL_VASTAI_BIN="$(command -v vastai)"
    else
        for candidate in \
            "/c/Users/C/AppData/Local/Programs/Python/Python312/Scripts/vastai.exe" \
            "/mnt/c/Users/C/AppData/Local/Programs/Python/Python312/Scripts/vastai.exe"
        do
            if [[ -x "$candidate" ]]; then
                LOCAL_VASTAI_BIN="$candidate"
                break
            fi
        done
    fi
fi
if [[ -z "$LOCAL_VASTAI_BIN" ]]; then
    echo "ERROR: vastai CLI not found on local machine"
    exit 1
fi

if [[ -z "$INSTANCE_ID" ]]; then
    echo "ERROR: set INSTANCE_ID to the rented Washington machine id"
    exit 1
fi

if [[ -z "$SSH_BIN" ]]; then
    for candidate in \
        "/mnt/c/WINDOWS/System32/OpenSSH/ssh.exe" \
        "/c/WINDOWS/System32/OpenSSH/ssh.exe"
    do
        if [[ -x "$candidate" ]]; then
            SSH_BIN="$candidate"
            break
        fi
    done
    [[ -n "$SSH_BIN" ]] || SSH_BIN="ssh"
fi

if [[ -z "$SCP_BIN" ]]; then
    for candidate in \
        "/mnt/c/WINDOWS/System32/OpenSSH/scp.exe" \
        "/c/WINDOWS/System32/OpenSSH/scp.exe"
    do
        if [[ -x "$candidate" ]]; then
            SCP_BIN="$candidate"
            break
        fi
    done
    [[ -n "$SCP_BIN" ]] || SCP_BIN="scp"
fi

log_step()  { echo -e "${GREEN}[STEP]${NC} $1"; }
log_info()  { echo -e "${CYAN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_header() {
    echo ""
    echo -e "${BOLD}${CYAN}============================================================${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BOLD}${CYAN}============================================================${NC}"
    echo ""
}

remote() {
    "$SSH_BIN" -p "$SSH_PORT" -o ConnectTimeout=15 -o StrictHostKeyChecking=no "root@$SSH_HOST" "$@"
}

upload() {
    local src="$1"
    local dst="$2"
    local scp_src="$src"
    if [[ "$SCP_BIN" == *.exe ]]; then
        if [[ "$scp_src" =~ ^/mnt/([a-zA-Z])/(.*)$ ]]; then
            local drive="${BASH_REMATCH[1]}"
            local rest="${BASH_REMATCH[2]//\//\\}"
            scp_src="${drive^^}:\\$rest"
        elif [[ "$scp_src" =~ ^/([a-zA-Z])/(.*)$ ]]; then
            local drive="${BASH_REMATCH[1]}"
            local rest="${BASH_REMATCH[2]//\//\\}"
            scp_src="${drive^^}:\\$rest"
        fi
    fi
    "$SCP_BIN" -P "$SSH_PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=15 "$scp_src" "root@$SSH_HOST:$dst"
}

declare -A UPLOADED_REMOTE=()

upload_once() {
    local src="$1"
    local dst="$2"
    local key="$dst$(basename "$src")"
    if [[ -n "${UPLOADED_REMOTE[$key]:-}" ]]; then
        return 0
    fi
    upload "$src" "$dst"
    UPLOADED_REMOTE["$key"]=1
}

validate_local_artifacts() {
    local required=(
        "$V33_DIR/CLOUD_1W_LAUNCH_CONTRACT.md"
        "$V33_DIR/CLOUD_TARGET_MACHINE.md"
        "$V33_DIR/deploy_manifest.py"
        "$V33_DIR/deploy_verify.py"
        "$V33_DIR/gcs_shared_seed.py"
        "$V33_DIR/deploy_model.py"
        "$V33_DIR/deploy_sichuan.sh"
        "$V33_DIR/deploy_washington_1w.sh"
        "$V33_DIR/WEEKLY_1W_ARTIFACT_CONTRACT.json"
        "$V33_DIR/PRODUCTION_READINESS.md"
        "$V33_DIR/run_optuna_local.py"
        "$V33_DIR/ml_multi_tf.py"
        "$V33_DIR/exhaustive_optimizer.py"
        "$V33_DIR/v2_cross_generator.py"
        "$V33_DIR/build_features_v2.py"
        "$PROJECT_DIR/kp_history_gfz.txt"
    )

    local missing=()
    for f in "${required[@]}"; do
        [[ -f "$f" ]] || missing+=("$f")
    done

    if (( ${#missing[@]} > 0 )); then
        log_error "Missing required local artifacts:"
        for f in "${missing[@]}"; do
            echo "  - $f"
        done
        exit 1
    fi
}

generate_deploy_manifest() {
    log_step "Generating fresh deploy manifest from current local code"
    (
        cd "$V33_DIR"
        "$LOCAL_PYTHON_BIN" deploy_manifest.py
    )
    [[ -f "$V33_DIR/deploy_manifest.json" ]] || {
        log_error "deploy_manifest.py did not produce $V33_DIR/deploy_manifest.json"
        exit 1
    }
}

resolve_gcs_key_path() {
    if [[ -n "$GCS_KEY_PATH" && -f "$GCS_KEY_PATH" ]]; then
        return 0
    fi
    for candidate in \
        "/mnt/c/Users/C/Desktop/tbtb-492116-586524867ff4.json" \
        "/c/Users/C/Desktop/tbtb-492116-586524867ff4.json"
    do
        if [[ -f "$candidate" ]]; then
            GCS_KEY_PATH="$candidate"
            return 0
        fi
    done
    log_error "GCS key file not found. Set GCS_KEY_PATH to the service account JSON."
    exit 1
}

sync_shared_db_seed_to_gcs() {
    [[ "$USE_GCS_SHARED_DB" == "1" ]] || return 0
    resolve_gcs_key_path
    log_step "Syncing shared DB seed to gs://$GCS_BUCKET/$GCS_PREFIX"
    local manifest_out
    manifest_out="$(mktemp "${TMPDIR:-/tmp}/savage22_shared_db_manifest.XXXXXX.json")"
    trap 'rm -f "$manifest_out"' RETURN
    "$LOCAL_PYTHON_BIN" "$V33_DIR/gcs_shared_seed.py" upload \
        --bucket "$GCS_BUCKET" \
        --prefix "$GCS_PREFIX" \
        --key-file "$GCS_KEY_PATH" \
        --project-root "$PROJECT_DIR" \
        --manifest-out "$manifest_out"

    IFS=$'\t' read -r GCS_SHARED_SEED_PROJECT_ID GCS_SHARED_SEED_BUCKET GCS_SHARED_SEED_PREFIX GCS_SHARED_SEED_MANIFEST_SHA256 GCS_SHARED_SEED_FILE_COUNT < <(
        "$LOCAL_PYTHON_BIN" - "$manifest_out" <<'PY'
import hashlib
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
digest = hashlib.sha256(path.read_bytes()).hexdigest()
values = [
    data.get("project_id", ""),
    data.get("bucket", ""),
    data.get("prefix", ""),
    digest,
    str(len(data.get("files", {}))),
]
print("\t".join(values))
PY
    )
    rm -f "$manifest_out"
    trap - RETURN

    export GCS_SHARED_SEED_PROJECT_ID GCS_SHARED_SEED_BUCKET GCS_SHARED_SEED_PREFIX
    export GCS_SHARED_SEED_MANIFEST_SHA256 GCS_SHARED_SEED_FILE_COUNT
}

should_ship_release_file() {
    local base="$1"
    case "$base" in
        deploy_manifest.json|WEEKLY_1W_ARTIFACT_CONTRACT.json)
            return 0
            ;;
        *.py|*.md|*.sh)
            ;;
        *)
            return 1
            ;;
    esac

    case "$base" in
        _cross_checkpoint_*|cpcv_oos_*|feature_importance_*|features_*|inference_*|lgbm_dataset_*|lgbm_parent_*|meta_model_*|ml_multi_tf_configs*|model_*|optuna_model_*|optuna_search*|pipeline_manifest*|shap_analysis_*|validation_report_*|v2_cross_names_*|v2_crosses_*)
            return 1
            ;;
    esac

    return 0
}

upload_release_bundle() {
    for f in "$V33_DIR"/*; do
        [[ -f "$f" ]] || continue
        local base
        base="$(basename "$f")"
        if should_ship_release_file "$base"; then
            upload_once "$f" "$REMOTE_RELEASE_STAGING/"
        fi
    done
}

write_release_manifest() {
    local release_dir="$1"
    local current_link="${2:-}"
    remote "RELEASE_DIR='$release_dir' CURRENT_LINK='$current_link' RUN_ID='$RUN_ID' INSTANCE_ID='$INSTANCE_ID' SESSION_NAME='$SESSION_NAME' ARTIFACT_ROOT='$REMOTE_ARTIFACT_ROOT' RUN_ROOT='$REMOTE_RUN_DIR' HEARTBEAT_FILE='$REMOTE_HB' SHARED_DB_ROOT='/workspace' SEED_PROJECT_ID='$GCS_SHARED_SEED_PROJECT_ID' SEED_BUCKET='$GCS_SHARED_SEED_BUCKET' SEED_PREFIX='$GCS_SHARED_SEED_PREFIX' SEED_MANIFEST_SHA256='$GCS_SHARED_SEED_MANIFEST_SHA256' SEED_FILE_COUNT='$GCS_SHARED_SEED_FILE_COUNT' python - <<'PY'
import json
import os
import pathlib
import time

payload = {
    "run_id": os.environ["RUN_ID"],
    "release_dir": os.environ["RELEASE_DIR"],
    "run_dir": os.environ["RUN_ROOT"],
    "artifact_root": os.environ["ARTIFACT_ROOT"],
    "shared_db_root": os.environ["SHARED_DB_ROOT"],
    "heartbeat_path": os.environ["HEARTBEAT_FILE"],
}
current_link = os.environ.get("CURRENT_LINK", "")
if current_link:
    payload["current_link"] = current_link
seed = {}
for env_key, payload_key in (
    ("SEED_PROJECT_ID", "project_id"),
    ("SEED_BUCKET", "bucket"),
    ("SEED_PREFIX", "prefix"),
    ("SEED_MANIFEST_SHA256", "manifest_sha256"),
    ("SEED_FILE_COUNT", "file_count"),
):
    value = os.environ.get(env_key, "")
    if value:
        if payload_key == "file_count":
            try:
                seed[payload_key] = int(value)
            except Exception:
                seed[payload_key] = value
        else:
            seed[payload_key] = value
if seed:
    payload["shared_db_seed"] = seed
path = pathlib.Path(os.environ["RUN_ROOT"]) / "release_manifest.json"
path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY"
}

resolve_instance() {
    if [[ -n "$SSH_HOST_OVERRIDE" && -n "$SSH_PORT_OVERRIDE" ]]; then
        SSH_HOST="$SSH_HOST_OVERRIDE"
        SSH_PORT="$SSH_PORT_OVERRIDE"
        ACTUAL_STATUS="running"
        ACTUAL_REGION="${TARGET_REGION}"
        ACTUAL_TEMPLATE_ID="${TARGET_TEMPLATE_ID}"
        ACTUAL_GPU_NAME="RTX 4090"
        GPU_COUNT="2"
        return 0
    fi

    INSTANCE_JSON=$("$LOCAL_VASTAI_BIN" show instances --raw 2>/dev/null || echo "[]")
    if [[ ! "$INSTANCE_JSON" =~ ^\[.*\]$ ]]; then
        log_error "vastai CLI did not return JSON"
        exit 1
    fi

    INSTANCE_LINE=$(printf '%s\n' "$INSTANCE_JSON" | "$LOCAL_PYTHON_BIN" - "$INSTANCE_ID" <<'PY'
import json
import sys

target = str(sys.argv[1])
instances = json.load(sys.stdin)
match = None
for inst in instances:
    if str(inst.get("id", "")) == target:
        match = inst
        break

if not match:
    sys.exit(1)

def pick(inst, *keys):
    for key in keys:
        cur = inst
        ok = True
        for part in key.split("."):
            if isinstance(cur, dict) and part in cur and cur[part] not in (None, "", []):
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            if isinstance(cur, list) and cur:
                cur = cur[0]
            if cur is not None:
                return str(cur)
    return ""

ssh_host = pick(
    match,
    "ssh_host",
    "ssh.hostname",
)
ssh_port = pick(
    match,
    "ssh_port",
    "port",
)
actual_status = pick(
    match,
    "actual_status",
    "status",
    "state",
)
region = pick(
    match,
    "location",
    "datacenter",
    "dc_name",
    "region",
    "city",
)
template_id = pick(
    match,
    "template_id",
    "ask_contract_id",
    "template.id",
    "contract_id",
    "ask.id",
)
gpu_name = pick(
    match,
    "gpu_name",
    "gpus.0.name",
    "gpus.0.gpu_name",
)
gpu_count = pick(
    match,
    "num_gpus",
    "gpus",
    "gpus_count",
)
if gpu_count.isdigit():
    gpu_count = str(gpu_count)
else:
    try:
        n = int(len(match.get("gpus", [])))
        gpu_count = str(n)
    except Exception:
        gpu_count = ""

print("\n".join([ssh_host, ssh_port, actual_status, region, template_id, gpu_name, gpu_count]))
PY
    )

    if [[ -z "$INSTANCE_LINE" ]]; then
        log_error "Instance $INSTANCE_ID not found in vastai list"
        exit 1
    fi

    IFS=$'\n' read -r SSH_HOST SSH_PORT ACTUAL_STATUS ACTUAL_REGION ACTUAL_TEMPLATE_ID ACTUAL_GPU_NAME GPU_COUNT <<< "$INSTANCE_LINE"
}

validate_machine() {
    if [[ -z "$SSH_HOST" || -z "$SSH_PORT" ]]; then
        log_error "Could not resolve SSH host/port for instance $INSTANCE_ID"
        exit 1
    fi

    if [[ -z "$ACTUAL_STATUS" ]]; then
        log_warn "Could not resolve instance status; treating as non-production-safe"
        exit 1
    fi

    if [[ "${ACTUAL_STATUS,,}" != running* && "${ACTUAL_STATUS,,}" != active* ]]; then
        log_error "Instance $INSTANCE_ID not in RUNNING state: $ACTUAL_STATUS"
        exit 1
    fi

    if [[ "$ACTUAL_TEMPLATE_ID" != "$TARGET_TEMPLATE_ID" && "$ALLOW_TEMPLATE_MISMATCH" != "1" ]]; then
        log_error "Template mismatch for instance $INSTANCE_ID"
        log_error "Expected template: $TARGET_TEMPLATE_ID"
        log_error "Detected template: ${ACTUAL_TEMPLATE_ID:-<none>}"
        exit 1
    fi

    if [[ "${ACTUAL_REGION,,}" != *"washington"* && "$ALLOW_REGION_MISMATCH" != "1" ]]; then
        log_error "Region mismatch. Expected Washington, US. Detected: ${ACTUAL_REGION:-<none>}"
        exit 1
    fi

    if [[ "$GPU_COUNT" != "2" ]]; then
        log_error "Expected 2 GPUs. Detected count: ${GPU_COUNT:-<none>}"
        exit 1
    fi

    if [[ "${ACTUAL_GPU_NAME,,}" != *"rtx 4090"* && "${ACTUAL_GPU_NAME,,}" != *"4090"* ]]; then
        log_error "Expected RTX 4090 GPUs. Detected GPU: ${ACTUAL_GPU_NAME:-<none>}"
        exit 1
    fi
}

REMOTE_ENV_BASE="PYTHONUNBUFFERED=1 V30_DATA_DIR=$REMOTE_ARTIFACT_ROOT SAVAGE22_ARTIFACT_DIR=$REMOTE_ARTIFACT_ROOT SAVAGE22_RUN_DIR=$REMOTE_RUN_DIR SAVAGE22_DB_DIR=/workspace SAVAGE22_V1_DIR=/workspace"
REMOTE_ALLOW_CPU_DETECT='if python -c '"'"'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("cudf") else 1)'"'"' >/dev/null 2>&1; then unset ALLOW_CPU; else export ALLOW_CPU=1; fi'
REMOTE_1W_SEQUENCE=$(cat <<'EOF'
set -euo pipefail
RELEASE_DIR="'"$REMOTE_RELEASE_DIR"'"
RUN_DIR="'"$REMOTE_RUN_DIR"'"
ARTIFACT_ROOT="'"$REMOTE_ARTIFACT_ROOT"'"
LOG_DIR="$RUN_DIR/logs"
HEARTBEAT_FILE="'"$REMOTE_HB"'"
ARTIFACT_CONTRACT="$RELEASE_DIR/WEEKLY_1W_ARTIFACT_CONTRACT.json"
mkdir -p "$RUN_DIR" "$LOG_DIR" "$ARTIFACT_ROOT" "'"$REMOTE_CACHE_DIR"'"
write_heartbeat() {
  local phase="${1:-unknown}"
  local status="${2:-running}"
  local detail="${3:-}"
  HEARTBEAT_PHASE="$phase" HEARTBEAT_STATUS="$status" HEARTBEAT_DETAIL="$detail" python - <<'PY'
import json
import os
import pathlib
import time

path = pathlib.Path("'"$REMOTE_HB"'")
contract_path = pathlib.Path("'"$REMOTE_RELEASE_DIR"'/WEEKLY_1W_ARTIFACT_CONTRACT.json")
phase = os.environ.get("HEARTBEAT_PHASE", "")
status = os.environ.get("HEARTBEAT_STATUS", "")
contract = {}
expected = []
phase_seq = None
if contract_path.exists():
    try:
        contract = json.loads(contract_path.read_text())
        phase_cfg = contract.get("phases", {}).get(phase, {})
        expected = phase_cfg.get("required_artifacts", [])
        phase_seq = phase_cfg.get("phase_seq")
    except Exception:
        pass
payload = {
    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "tf": "1w",
    "run_id": "'"$RUN_ID"'",
    "session_name": "'"$SESSION_NAME"'",
    "owner": "deploy_washington_1w.sh",
    "instance_id": "'"$INSTANCE_ID"'",
    "code_root": "'"$REMOTE_RELEASE_DIR"'",
    "shared_db_root": "/workspace",
    "run_root": "'"$REMOTE_RUN_DIR"'",
    "artifact_root": "'"$REMOTE_ARTIFACT_ROOT"'",
    "heartbeat_path": str(path),
    "phase": phase,
    "phase_seq": phase_seq,
    "status": status,
    "detail": os.environ.get("HEARTBEAT_DETAIL", ""),
    "artifact_contract": str(contract_path),
    "release_manifest": "'"$REMOTE_RUN_DIR"'/release_manifest.json",
    "expected_artifacts": expected,
}
try:
    path.write_text(json.dumps(payload, sort_keys=True))
except Exception:
    pass
PY
}
validate_phase_artifacts() {
  local phase="${1:-unknown}"
  local min_epoch="${2:-0}"
  HEARTBEAT_PHASE="$phase" HEARTBEAT_PHASE_START_EPOCH="$min_epoch" python - <<'PY'
import json
import os
import pathlib
import sys

phase = os.environ["HEARTBEAT_PHASE"]
min_epoch = float(os.environ.get("HEARTBEAT_PHASE_START_EPOCH", "0") or "0")
contract_path = pathlib.Path("'"$REMOTE_RELEASE_DIR"'/WEEKLY_1W_ARTIFACT_CONTRACT.json")
if not contract_path.exists():
    sys.exit(0)
contract = json.loads(contract_path.read_text())
required = contract.get("phases", {}).get(phase, {}).get("required_artifacts", [])
missing = []
artifact_root = pathlib.Path("'"$REMOTE_ARTIFACT_ROOT"'").resolve()
for name in required:
    path = (artifact_root / name).resolve()
    try:
        path.relative_to(artifact_root)
    except ValueError:
        missing.append(name)
        continue
    if not path.exists():
        missing.append(name)
        continue
    if min_epoch > 0 and path.stat().st_mtime < min_epoch:
        missing.append(name)
if missing:
    print("MISSING_ARTIFACTS:" + ",".join(missing))
    sys.exit(1)
PY
}
clean_stale_artifacts() {
  rm -f "$ARTIFACT_ROOT/features_BTC_1w.parquet"
  rm -f "$ARTIFACT_ROOT/v2_crosses_BTC_1w.npz"
  rm -f "$ARTIFACT_ROOT/v2_cross_names_BTC_1w.json"
  rm -f "$ARTIFACT_ROOT/model_1w.json"
  rm -f "$ARTIFACT_ROOT/model_1w_cpcv_backup.json"
  rm -f "$ARTIFACT_ROOT/optuna_model_1w.json"
  rm -f "$ARTIFACT_ROOT/optuna_configs_1w.json"
  rm -f "$ARTIFACT_ROOT/cpcv_oos_predictions_1w.pkl"
  rm -f "$ARTIFACT_ROOT/feature_importance_stability_1w.json"
  rm -f "$ARTIFACT_ROOT/platt_1w.pkl"
  rm -f "$ARTIFACT_ROOT/lgbm_dataset_1w.bin"
  rm -f "$ARTIFACT_ROOT/lgbm_parent_1w.bin"
  rm -rf "$ARTIFACT_ROOT/_idx"
  rm -f "$HEARTBEAT_FILE"
  rm -f "$LOG_DIR"/step*_1w.log
}
preflight_layout() {
  [[ -d "$RELEASE_DIR" ]] || { echo "missing release dir: $RELEASE_DIR"; exit 1; }
  [[ -f "$ARTIFACT_CONTRACT" ]] || { echo "missing artifact contract: $ARTIFACT_CONTRACT"; exit 1; }
  [[ -d "$RUN_DIR" ]] || { echo "missing run dir: $RUN_DIR"; exit 1; }
  [[ -d "$ARTIFACT_ROOT" ]] || { echo "missing artifact root: $ARTIFACT_ROOT"; exit 1; }
  touch "$RUN_DIR/.write_test"
  rm -f "$RUN_DIR/.write_test"
  touch "$ARTIFACT_ROOT/.write_test"
  rm -f "$ARTIFACT_ROOT/.write_test"
}
trap 'write_heartbeat "${CURRENT_PHASE:-unknown}" "failed" "command failed on $(printf "%s" "$BASH_COMMAND")"' ERR

write_heartbeat "step0_preflight" "running" "Explicit 1w sequence prepared"
cd "$RELEASE_DIR"
$REMOTE_ALLOW_CPU_DETECT
export PYTHONUNBUFFERED=1
export V30_DATA_DIR="$ARTIFACT_ROOT"
export SAVAGE22_ARTIFACT_DIR="$ARTIFACT_ROOT"
export SAVAGE22_RUN_DIR="$RUN_DIR"
export SAVAGE22_DB_DIR=/workspace
export SAVAGE22_V1_DIR=/workspace
export OPTUNA_SKIP_FINAL_RETRAIN=1
export V3_HOT_PATH_TRAINING=1
export V3_RUN_FI_STABILITY=0
export V3_RUN_ADVANCED_FI=0
export V3_CHECKPOINT_PERIOD=200
preflight_layout
clean_stale_artifacts
write_heartbeat "step0_preflight" "validated" "4-root layout validated and artifact root cleared"

CURRENT_PHASE="step1_features"
PHASE_START_TS=$(date +%s)
write_heartbeat "$CURRENT_PHASE" "running" "Building weekly features"
python -u build_features_v2.py --symbol BTC --tf 1w | tee "$LOG_DIR/step1_build_features_1w.log"
validate_phase_artifacts "$CURRENT_PHASE" "$PHASE_START_TS"
write_heartbeat "$CURRENT_PHASE" "validated" "features_BTC_1w.parquet present"

CURRENT_PHASE="step2_crosses"
PHASE_START_TS=$(date +%s)
write_heartbeat "$CURRENT_PHASE" "running" "Generating weekly sparse crosses"
python -u v2_cross_generator.py --tf 1w --symbol BTC --save-sparse | tee "$LOG_DIR/step2_crosses_1w.log"
validate_phase_artifacts "$CURRENT_PHASE" "$PHASE_START_TS"
write_heartbeat "$CURRENT_PHASE" "validated" "cross artifacts present"

CURRENT_PHASE="step3_baseline"
PHASE_START_TS=$(date +%s)
write_heartbeat "$CURRENT_PHASE" "running" "Training weekly baseline model"
python -u ml_multi_tf.py --tf 1w --boost-rounds 800 | tee "$LOG_DIR/step3_train_baseline_1w.log"
validate_phase_artifacts "$CURRENT_PHASE" "$PHASE_START_TS"
write_heartbeat "$CURRENT_PHASE" "validated" "baseline model and CPCV artifacts present"

CURRENT_PHASE="step4_optuna"
PHASE_START_TS=$(date +%s)
write_heartbeat "$CURRENT_PHASE" "running" "Running weekly Optuna search"
python -u run_optuna_local.py --tf 1w --search-only | tee "$LOG_DIR/step4_optuna_1w.log"
validate_phase_artifacts "$CURRENT_PHASE" "$PHASE_START_TS"
write_heartbeat "$CURRENT_PHASE" "validated" "optuna configs present"

CURRENT_PHASE="step5_retrain"
PHASE_START_TS=$(date +%s)
write_heartbeat "$CURRENT_PHASE" "running" "Retraining weekly model with tuned params"
python -u ml_multi_tf.py --tf 1w --boost-rounds 800 | tee "$LOG_DIR/step5_retrain_optuna_1w.log"
validate_phase_artifacts "$CURRENT_PHASE" "$PHASE_START_TS"
write_heartbeat "$CURRENT_PHASE" "validated" "retrained model artifacts present"

CURRENT_PHASE="step6_optimizer"
PHASE_START_TS=$(date +%s)
write_heartbeat "$CURRENT_PHASE" "running" "Running weekly execution optimizer"
python -u exhaustive_optimizer.py --tf 1w --n-trials 200 | tee "$LOG_DIR/step6_optimizer_1w.log"
validate_phase_artifacts "$CURRENT_PHASE" "$PHASE_START_TS"
write_heartbeat "$CURRENT_PHASE" "validated" "optimizer output present"
CURRENT_PHASE="complete"
validate_phase_artifacts "$CURRENT_PHASE"
write_heartbeat "$CURRENT_PHASE" "complete" "1w sequence finished with required artifacts"
EOF
)

log_header "PHASE 0: Resolve and Validate Machine $INSTANCE_ID"
validate_local_artifacts
generate_deploy_manifest
sync_shared_db_seed_to_gcs
resolve_instance
validate_machine

log_header "WASHINGTON 2x RTX 4090 DEPLOYMENT"
echo -e "  Instance: ${BOLD}$INSTANCE_ID${NC}"
echo -e "  SSH:      ${BOLD}root@$SSH_HOST:$SSH_PORT${NC}"
echo -e "  Status:   ${BOLD}$ACTUAL_STATUS${NC}"
echo -e "  Region:   ${BOLD}${ACTUAL_REGION:-unknown}${NC}"
echo -e "  Template: ${BOLD}${ACTUAL_TEMPLATE_ID:-unknown}${NC}"
echo -e "  GPU:      ${BOLD}${GPU_COUNT:-?}x ${ACTUAL_GPU_NAME:-unknown}${NC}"
echo -e "  Target:   ${BOLD}$TF${NC}"
echo -e "  Project:  ${BOLD}$PROJECT_DIR${NC}"
echo -e "  Dry run:  ${BOLD}$DRY_RUN${NC}"
echo ""

if $DRY_RUN; then
    log_warn "DRY RUN - no files will be transferred or commands executed"
fi

log_header "PHASE 1: Upload Current v3.3 Tree"
if ! $DRY_RUN; then
    remote "rm -rf '$REMOTE_RELEASE_STAGING' && mkdir -p '$REMOTE_RELEASE_STAGING' '$REMOTE_RELEASES_DIR' '$REMOTE_RUN_DIR' '$REMOTE_ARTIFACT_ROOT' '$REMOTE_ARTIFACTS_DIR' '$REMOTE_CACHE_DIR'"
    upload_release_bundle

    if [[ "$USE_GCS_SHARED_DB" == "1" ]]; then
        upload_once "$GCS_KEY_PATH" "$REMOTE_RUN_DIR/"
        remote "cd '$REMOTE_RELEASE_STAGING' && trap 'rm -f \"$REMOTE_GCS_KEY_PATH\"' EXIT && python - <<'PY'
import importlib.util, subprocess, sys
mods = ('google.cloud.storage',)
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'google-cloud-storage'])
PY"
        remote "cd '$REMOTE_RELEASE_STAGING' && trap 'rm -f \"$REMOTE_GCS_KEY_PATH\"' EXIT && PYTHONUNBUFFERED=1 python -u gcs_shared_seed.py download --bucket '$GCS_BUCKET' --prefix '$GCS_PREFIX' --key-file '$REMOTE_GCS_KEY_PATH' --dest '$REMOTE_DIR'"
    else
        if [[ -f "$PROJECT_DIR/kp_history_gfz.txt" ]]; then
            upload_once "$PROJECT_DIR/kp_history_gfz.txt" "$REMOTE_DIR/"
        fi
        for f in "$PROJECT_DIR"/*.db "$V33_DIR"/multi_asset_prices.db "$V33_DIR"/v2_signals.db "$V33_DIR"/llm_cache.db; do
            [[ -f "$f" ]] || continue
            upload_once "$f" "$REMOTE_DIR/"
        done
    fi
    write_release_manifest "$REMOTE_RELEASE_STAGING"
    log_step "Upload complete to staging: $REMOTE_RELEASE_STAGING"
else
    log_info "[DRY RUN] Would upload repo files and databases"
fi

log_header "PHASE 2: Remote Preflight"
if ! $DRY_RUN; then
    remote "cd '$REMOTE_RELEASE_STAGING' && mkdir -p '$REMOTE_RUN_DIR' '$REMOTE_RUN_DIR/logs' '$REMOTE_ARTIFACT_ROOT' '$REMOTE_CACHE_DIR' && touch '$REMOTE_RUN_DIR/.write_test' && rm -f '$REMOTE_RUN_DIR/.write_test' && touch '$REMOTE_ARTIFACT_ROOT/.write_test' && rm -f '$REMOTE_ARTIFACT_ROOT/.write_test'"
    write_release_manifest "$REMOTE_RELEASE_STAGING"
    remote "cd '$REMOTE_RELEASE_STAGING' && $REMOTE_ALLOW_CPU_DETECT && $REMOTE_ENV_BASE python -u deploy_verify.py --tf $TF --allow-staged-release"
    remote "cd '$REMOTE_RELEASE_STAGING' && $REMOTE_ALLOW_CPU_DETECT && $REMOTE_ENV_BASE python -u validate.py --tf $TF --cloud"
    remote "cd '$REMOTE_RELEASE_STAGING' && $REMOTE_ALLOW_CPU_DETECT && $REMOTE_ENV_BASE python -u test_pipeline_plumbing.py --tf $TF"
    log_step "Remote preflight passed"
else
    log_info "[DRY RUN] Would run deploy_verify.py, validate.py, and test_pipeline_plumbing.py"
fi

log_header "PHASE 3: Promote Verified Release"
if ! $DRY_RUN; then
    remote "rm -rf '$REMOTE_RELEASE_DIR' && mv '$REMOTE_RELEASE_STAGING' '$REMOTE_RELEASE_DIR' && ln -sfn '$REMOTE_RELEASE_DIR' '$REMOTE_CURRENT_LINK'"
    write_release_manifest "$REMOTE_RELEASE_DIR" "$REMOTE_CURRENT_LINK"
    log_step "Promoted verified release to $REMOTE_RELEASE_DIR"
else
    log_info "[DRY RUN] Would promote staging release to $REMOTE_RELEASE_DIR"
fi

log_header "PHASE 4: Launch Explicit 1W Sequence"
if ! $DRY_RUN; then
    remote "tmux kill-session -t train_1w 2>/dev/null || true"
    remote "tmux kill-session -t train_1w_remaining 2>/dev/null || true"
    remote "cat > /tmp/washington_1w_sequence.sh <<'EOS'
$REMOTE_1W_SEQUENCE
EOS
bash -n /tmp/washington_1w_sequence.sh"
    remote "tmux new-session -d -s $SESSION_NAME 'bash /tmp/washington_1w_sequence.sh'"
    log_step "Launched tmux session $SESSION_NAME (run_id=$RUN_ID)"
else
    log_info "[DRY RUN] Would launch tmux session $SESSION_NAME"
fi

log_header "NEXT ACTIONS"
echo "1) Monitor log and heartbeat:"
echo "   ssh -p $SSH_PORT root@$SSH_HOST -t 'tmux attach -t $SESSION_NAME'"
echo "   ssh -p $SSH_PORT root@$SSH_HOST \"watch -n 20 'cat $REMOTE_HB'\""
echo "2) Download checkpoint artifacts after each step from $REMOTE_ARTIFACT_ROOT and logs from $REMOTE_RUN_DIR/logs:"
echo "   - features_BTC_1w.parquet, v2_crosses_BTC_1w.npz, v2_cross_names_BTC_1w.json"
echo "   - model_1w.json, model_1w_cpcv_backup.json, optuna_configs_1w.json, meta_model_1w.pkl"
echo "   - lgbm_parent_1w.bin, lgbm_dataset_1w.bin, shap_analysis_1w.json"
echo "   - cpcv_oos_predictions_1w.pkl"
echo "   - platt_1w.pkl, $(basename "$REMOTE_HB"), release_manifest.json"
echo "3) Final retrain policy is logged in run_optuna_local.py:"
echo "   - FINAL_RETRAIN_PARALLEL_POLICY (default: auto)"
echo "   - FINAL_RETRAIN_PARALLEL_MIN_ROWS (default: 512)"
echo "   - OPTUNA_FINAL_RETRAIN_MAX_PARALLEL_FOLDS (default: auto)"
echo "4) Audit 1w completion and quality before lower-timeframe launch."
echo "5) Heartbeat truth: trust status=validated/complete only when expected_artifacts resolve inside $REMOTE_ARTIFACT_ROOT."
echo "6) Protocol: maintain 4 roots per run: code=$REMOTE_RELEASE_DIR db=/workspace run=$REMOTE_RUN_DIR artifacts=$REMOTE_ARTIFACT_ROOT."
echo "7) If heartbeat stops updating for > 6 min, treat as stalled run and inspect the active tmux session."
