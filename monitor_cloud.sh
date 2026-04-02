#!/bin/bash
# ============================================================================
# monitor_cloud.sh — Companion monitor for vast.ai training pipeline
# ============================================================================
#
# Shows GPU utilization, pipeline progress, cost, and tails the log.
#
# Usage:
#   ./monitor_cloud.sh SSH_HOST SSH_PORT DPH [--tail|--gpu|--status|--cost|--full]
#
# Modes:
#   --tail     Tail the pipeline log (default)
#   --gpu      Watch GPU utilization (refreshes every 2s)
#   --status   Show pipeline manifest status (one-shot)
#   --cost     Show elapsed time and cost estimate (one-shot)
#   --full     Full dashboard: status + GPU + cost + tail
#
# Examples:
#   ./monitor_cloud.sh root@ssh7.vast.ai 13562 2.88
#   ./monitor_cloud.sh root@ssh7.vast.ai 13562 2.88 --gpu
#   ./monitor_cloud.sh root@ssh7.vast.ai 13562 2.88 --full
#
# ============================================================================

set -euo pipefail

# ── Color codes ──────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Parse arguments ──────────────────────────────────────────
if [[ $# -lt 3 ]]; then
    echo -e "${RED}ERROR: Missing required arguments${NC}"
    echo ""
    echo "Usage: $0 SSH_HOST SSH_PORT DPH [--tail|--gpu|--status|--cost|--full]"
    echo ""
    echo "Modes:"
    echo "  --tail     Tail pipeline log (default)"
    echo "  --gpu      Watch GPU utilization"
    echo "  --status   Show pipeline manifest"
    echo "  --cost     Show elapsed time + cost"
    echo "  --full     Full dashboard (status + GPU + cost + tail)"
    exit 1
fi

SSH_HOST="$1"
SSH_PORT="$2"
DPH="$3"
MODE="${4:---tail}"

SSH_CMD="ssh -p $SSH_PORT -o ConnectTimeout=10 -o StrictHostKeyChecking=no"
REMOTE_DIR="/workspace"

# ── Helper functions ─────────────────────────────────────────

separator() {
    echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
}

check_connection() {
    if ! $SSH_CMD "$SSH_HOST" 'echo ok' &>/dev/null; then
        echo -e "${RED}ERROR: Cannot connect to $SSH_HOST:$SSH_PORT${NC}"
        echo "Is the instance running?"
        exit 1
    fi
}

# ── GPU Utilization ──────────────────────────────────────────

show_gpu() {
    echo -e "${BOLD}${CYAN}  GPU UTILIZATION${NC}"
    separator
    $SSH_CMD "$SSH_HOST" bash -c "'
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | while IFS=\",\" read -r idx name util mem_used mem_total temp power; do
            idx=\$(echo \$idx | xargs)
            name=\$(echo \$name | xargs)
            util=\$(echo \$util | xargs)
            mem_used=\$(echo \$mem_used | xargs)
            mem_total=\$(echo \$mem_total | xargs)
            temp=\$(echo \$temp | xargs)
            power=\$(echo \$power | xargs)

            # Build utilization bar
            bar_len=20
            filled=\$(echo \"\$util * \$bar_len / 100\" | bc 2>/dev/null || echo 0)
            bar=\"\"
            for ((i=0; i<filled; i++)); do bar+=\"#\"; done
            for ((i=filled; i<bar_len; i++)); do bar+=\".\"; done

            printf \"  GPU %s %-20s [%s] %3s%% | VRAM %5s/%5s MB | %s C | %s W\n\" \
                \"\$idx\" \"\$name\" \"\$bar\" \"\$util\" \"\$mem_used\" \"\$mem_total\" \"\$temp\" \"\$power\"
        done
    '" 2>/dev/null || echo "  nvidia-smi unavailable"
    echo ""
}

# ── RAM Utilization ──────────────────────────────────────────

show_ram() {
    echo -e "${BOLD}${CYAN}  SYSTEM RAM${NC}"
    separator
    $SSH_CMD "$SSH_HOST" bash -c "'
        if command -v free &>/dev/null; then
            free -h | head -2
        elif [[ -f /proc/meminfo ]]; then
            total=\$(grep MemTotal /proc/meminfo | awk \"{print \\\$2}\")
            avail=\$(grep MemAvailable /proc/meminfo | awk \"{print \\\$2}\")
            used=\$((total - avail))
            printf \"  Used: %d GB / %d GB (%.0f%%)\n\" \$((used/1048576)) \$((total/1048576)) \$((used*100/total))
        fi
    '" 2>/dev/null || echo "  RAM info unavailable"
    echo ""
}

# ── Pipeline Manifest Status ────────────────────────────────

show_status() {
    echo -e "${BOLD}${CYAN}  PIPELINE MANIFEST STATUS${NC}"
    separator
    $SSH_CMD "$SSH_HOST" bash -c "'
        MANIFEST=\"$REMOTE_DIR/pipeline_manifest.json\"
        if [[ -f \"\$MANIFEST\" ]]; then
            python3 -c \"
import json, sys
with open(sys.argv[1]) as f:
    m = json.load(f)
print()
# Count by status
total = len(m) if isinstance(m, list) else len(m.get(\"steps\", m))
if isinstance(m, dict):
    # Handle dict-style manifest
    for key, val in m.items():
        if isinstance(val, dict):
            status = val.get(\"status\", \"unknown\")
            elapsed = val.get(\"elapsed_s\", 0)
            if status == \"OK\" or status == \"completed\":
                print(f\"  [DONE] {key:40s} ({elapsed:.0f}s)\")
            elif status == \"running\" or status == \"in_progress\":
                print(f\"  [ >> ] {key:40s} (running...)\")
            elif \"FAIL\" in str(status).upper():
                print(f\"  [FAIL] {key:40s} ({status})\")
            else:
                print(f\"  [    ] {key:40s} ({status})\")
        elif isinstance(val, str):
            print(f\"  {key}: {val}\")
    # Summary
    done_count = sum(1 for v in m.values() if isinstance(v, dict) and v.get(\"status\") in (\"OK\",\"completed\"))
    fail_count = sum(1 for v in m.values() if isinstance(v, dict) and \"FAIL\" in str(v.get(\"status\",\"\")).upper())
    remaining = sum(1 for v in m.values() if isinstance(v, dict)) - done_count - fail_count
    print()
    print(f\"  Summary: {done_count} done, {fail_count} failed, {remaining} remaining\")
elif isinstance(m, list):
    for item in m:
        if isinstance(item, dict):
            name = item.get(\"name\", item.get(\"step\", \"?\"))
            status = item.get(\"status\", \"pending\")
            elapsed = item.get(\"elapsed_s\", 0)
            if status in (\"OK\", \"completed\"):
                print(f\"  [DONE] {name:40s} ({elapsed:.0f}s)\")
            elif status in (\"running\", \"in_progress\"):
                print(f\"  [ >> ] {name:40s} (running...)\")
            elif \"FAIL\" in str(status).upper():
                print(f\"  [FAIL] {name:40s} ({status})\")
            else:
                print(f\"  [    ] {name:40s} ({status})\")
\" \"\$MANIFEST\" 2>/dev/null || cat \"\$MANIFEST\" | python3 -m json.tool 2>/dev/null || cat \"\$MANIFEST\"
        else
            echo \"  No pipeline_manifest.json found — pipeline may not have started yet\"
        fi
    '" 2>/dev/null || echo "  Could not read manifest"
    echo ""
}

# ── Cost Estimate ────────────────────────────────────────────

show_cost() {
    echo -e "${BOLD}${CYAN}  COST ESTIMATE${NC}"
    separator

    # Get tmux session start time from the pipeline log
    COST_INFO=$($SSH_CMD "$SSH_HOST" bash -c "'
        LOG=\"$REMOTE_DIR/pipeline.log\"
        if [[ -f \"\$LOG\" ]]; then
            # Get first and last timestamps from the log
            first_line=\$(head -1 \"\$LOG\" 2>/dev/null)
            last_line=\$(tail -1 \"\$LOG\" 2>/dev/null)
            line_count=\$(wc -l < \"\$LOG\" 2>/dev/null)

            # Try to extract elapsed time from heartbeat format [HH:MM:SS]
            last_time=\$(echo \"\$last_line\" | grep -oP \"\\[\\K[0-9]+:[0-9]+:[0-9]+\" | head -1)
            if [[ -n \"\$last_time\" ]]; then
                IFS=: read -r h m s <<< \"\$last_time\"
                total_sec=\$((10#\$h * 3600 + 10#\$m * 60 + 10#\$s))
                echo \"ELAPSED_SEC=\$total_sec\"
                echo \"LOG_LINES=\$line_count\"
            else
                # Fallback: use file modification time
                if stat --version &>/dev/null 2>&1; then
                    mod_epoch=\$(stat -c %Y \"\$LOG\")
                    create_epoch=\$(stat -c %W \"\$LOG\" 2>/dev/null || stat -c %Y \"\$LOG\")
                else
                    mod_epoch=\$(stat -f %m \"\$LOG\")
                    create_epoch=\$(stat -f %B \"\$LOG\" 2>/dev/null || stat -f %m \"\$LOG\")
                fi
                now_epoch=\$(date +%s)
                elapsed=\$((now_epoch - create_epoch))
                echo \"ELAPSED_SEC=\$elapsed\"
                echo \"LOG_LINES=\$line_count\"
            fi
        else
            echo \"ELAPSED_SEC=0\"
            echo \"LOG_LINES=0\"
            echo \"NO_LOG=1\"
        fi
    '" 2>/dev/null) || true

    # Parse the response
    ELAPSED_SEC=$(echo "$COST_INFO" | grep 'ELAPSED_SEC=' | head -1 | cut -d= -f2)
    LOG_LINES=$(echo "$COST_INFO" | grep 'LOG_LINES=' | head -1 | cut -d= -f2)
    NO_LOG=$(echo "$COST_INFO" | grep 'NO_LOG=' | head -1 | cut -d= -f2)

    ELAPSED_SEC=${ELAPSED_SEC:-0}
    LOG_LINES=${LOG_LINES:-0}

    if [[ "${NO_LOG:-0}" == "1" ]]; then
        echo -e "  ${YELLOW}No pipeline.log found — pipeline may not have started${NC}"
    else
        ELAPSED_H=$(echo "scale=2; $ELAPSED_SEC / 3600" | bc)
        ELAPSED_MIN=$(echo "scale=1; $ELAPSED_SEC / 60" | bc)
        COST_SO_FAR=$(echo "scale=2; $ELAPSED_H * $DPH" | bc)

        echo -e "  Elapsed:     ${BOLD}${ELAPSED_MIN} minutes${NC} (${ELAPSED_H}h)"
        echo -e "  Rate:        ${BOLD}\$$DPH/hr${NC}"
        echo -e "  Cost so far: ${BOLD}\$$COST_SO_FAR${NC}"
        echo -e "  Log lines:   ${BOLD}$LOG_LINES${NC}"

        # Estimate remaining (rough: 90 min total pipeline)
        EST_TOTAL=5400  # 90 min in seconds
        if (( ELAPSED_SEC > 0 && ELAPSED_SEC < EST_TOTAL )); then
            REMAINING=$((EST_TOTAL - ELAPSED_SEC))
            REMAINING_MIN=$(echo "scale=1; $REMAINING / 60" | bc)
            TOTAL_COST=$(echo "scale=2; $EST_TOTAL / 3600 * $DPH" | bc)
            echo -e "  Est. remaining: ~${REMAINING_MIN} min"
            echo -e "  Est. total cost: ~\$$TOTAL_COST"
        fi
    fi
    echo ""
}

# ── Active Processes ─────────────────────────────────────────

show_processes() {
    echo -e "${BOLD}${CYAN}  ACTIVE PYTHON PROCESSES${NC}"
    separator
    $SSH_CMD "$SSH_HOST" bash -c "'
        ps aux | grep \"[p]ython\" | grep -v grep | awk \"{printf \\\"  PID %6s | CPU %5s%% | MEM %5s%% | %s\\n\\\", \\\$2, \\\$3, \\\$4, \\\$11}\"
    '" 2>/dev/null || echo "  No Python processes running"
    echo ""
}

# ── Tmux Status ──────────────────────────────────────────────

show_tmux() {
    echo -e "${BOLD}${CYAN}  TMUX SESSIONS${NC}"
    separator
    $SSH_CMD "$SSH_HOST" 'tmux list-sessions 2>/dev/null || echo "  No tmux sessions"' 2>/dev/null
    echo ""
}

# ── Last N Log Lines ─────────────────────────────────────────

show_recent_log() {
    local n=${1:-30}
    echo -e "${BOLD}${CYAN}  LAST $n LOG LINES${NC}"
    separator
    $SSH_CMD "$SSH_HOST" "tail -n $n $REMOTE_DIR/pipeline.log 2>/dev/null || echo '  No pipeline.log found'" 2>/dev/null
    echo ""
}

# ── Disk Usage ───────────────────────────────────────────────

show_disk() {
    echo -e "${BOLD}${CYAN}  DISK USAGE${NC}"
    separator
    $SSH_CMD "$SSH_HOST" bash -c "'
        echo \"  /workspace:\"
        du -sh $REMOTE_DIR/*.db 2>/dev/null | sort -rh | head -5 | sed \"s/^/    /\"
        echo \"  ---\"
        du -sh $REMOTE_DIR/*.parquet 2>/dev/null | sort -rh | head -5 | sed \"s/^/    /\"
        echo \"  ---\"
        du -sh $REMOTE_DIR/ 2>/dev/null | sed \"s/^/    Total: /\"
        echo \"\"
        df -h $REMOTE_DIR 2>/dev/null | tail -1 | awk \"{printf \\\"    Disk: %s used / %s total (%s)\\n\\\", \\\$3, \\\$2, \\\$5}\"
    '" 2>/dev/null || echo "  Disk info unavailable"
    echo ""
}

# ============================================================
# MAIN — Route by mode
# ============================================================

echo ""
echo -e "${BOLD}${CYAN}  SAVAGE22 CLOUD MONITOR — $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "  ${BOLD}Host:${NC} $SSH_HOST:$SSH_PORT  ${BOLD}Rate:${NC} \$$DPH/hr"
echo ""

# Quick connection check
check_connection

case "$MODE" in
    --tail)
        show_cost
        echo -e "${GREEN}  Tailing pipeline.log (Ctrl+C to stop)...${NC}"
        separator
        $SSH_CMD "$SSH_HOST" "tail -f $REMOTE_DIR/pipeline.log" 2>/dev/null \
            || echo -e "${RED}  No pipeline.log found${NC}"
        ;;

    --gpu)
        echo -e "${GREEN}  Watching GPU utilization (Ctrl+C to stop)...${NC}"
        separator
        $SSH_CMD "$SSH_HOST" 'watch -n 2 nvidia-smi' 2>/dev/null \
            || {
                # Fallback: manual loop
                while true; do
                    clear
                    echo -e "${BOLD}${CYAN}  GPU MONITOR — $(date '+%H:%M:%S')${NC}"
                    separator
                    show_gpu
                    show_ram
                    show_cost
                    sleep 3
                done
            }
        ;;

    --status)
        show_status
        show_cost
        show_tmux
        show_processes
        ;;

    --cost)
        show_cost
        ;;

    --full)
        show_gpu
        show_ram
        show_cost
        show_status
        show_tmux
        show_processes
        show_disk
        show_recent_log 20
        ;;

    --loop)
        # Continuous dashboard refresh every 15s
        while true; do
            clear
            echo ""
            echo -e "${BOLD}${CYAN}  SAVAGE22 CLOUD DASHBOARD — $(date '+%Y-%m-%d %H:%M:%S')${NC}"
            echo -e "  ${BOLD}Host:${NC} $SSH_HOST:$SSH_PORT  ${BOLD}Rate:${NC} \$$DPH/hr"
            echo ""
            show_gpu
            show_ram
            show_cost
            show_status
            show_processes
            show_recent_log 10
            echo -e "  ${YELLOW}Refreshing in 15s... (Ctrl+C to stop)${NC}"
            sleep 15
        done
        ;;

    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo "Valid modes: --tail --gpu --status --cost --full --loop"
        exit 1
        ;;
esac
