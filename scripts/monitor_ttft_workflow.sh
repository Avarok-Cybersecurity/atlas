#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/workspace/atlas"
TASK_DIR="$REPO_ROOT/tasks/codex_agent_2026-03-23"
OPS_LOG="$TASK_DIR/ttft-overnight-ops-2026-03-23.md"
RUN_LOG="$TASK_DIR/ttft-overnight-codex.log"
LAST_MSG="$TASK_DIR/ttft-overnight-last-message.txt"
STATE_FILE="$TASK_DIR/ttft-monitor-state.env"
MARKER="ATLAS_TTFT_OVERNIGHT_2026_03_23"
CUTOFF="2026-03-23 11:00:00"

timestamp() {
  TZ=America/New_York date '+%Y-%m-%d %H:%M:%S %Z'
}

now_epoch=$(date +%s)
cutoff_epoch=$(TZ=America/New_York date -d "$CUTOFF" +%s)

mkdir -p "$TASK_DIR"

if [ "$now_epoch" -ge "$cutoff_epoch" ]; then
  if crontab -l 2>/dev/null | grep -q "$MARKER"; then
    tmp=$(mktemp)
    crontab -l 2>/dev/null | grep -v "$MARKER" >"$tmp" || true
    crontab "$tmp"
    rm -f "$tmp"
  fi
  {
    echo ""
    echo "### $(timestamp)"
    echo ""
    echo "- Overnight cutoff reached (`$CUTOFF` America/New_York)."
    echo "- Removed cron entry `$MARKER`."
  } >>"$OPS_LOG"
  exit 0
fi

cd "$REPO_ROOT"

codex_lines=$(ps -eo pid,ppid,etime,cmd | grep -E '/codex exec --dangerously-bypass-approvals-and-sandbox -C /workspace/atlas' | grep -v grep || true)
mcp_lines=$(ps -eo pid,ppid,etime,cmd | grep -E 'npm exec @steipete/claude-code-mcp@latest|claude-code-mcp$' | grep -v grep || true)
claude_lines=$(ps -eo pid,ppid,etime,cmd | grep -E '/workspace/.local/bin/claude( |$)|/workspace/.local/bin/claude-connect( |$)' | grep -v grep || true)

fingerprint=$(
  {
    git status --short -- \
      crates/spark-model/src/model.rs \
      crates/spark-model/src/traits.rs \
      crates/spark-runtime/src/buffers.rs \
      crates/spark-runtime/src/gpu.rs \
      crates/spark-runtime/src/cuda_backend.rs \
      crates/spark-server/src/scheduler.rs
    git diff --stat -- \
      crates/spark-model/src/model.rs \
      crates/spark-model/src/traits.rs \
      crates/spark-runtime/src/buffers.rs \
      crates/spark-runtime/src/gpu.rs \
      crates/spark-runtime/src/cuda_backend.rs \
      crates/spark-server/src/scheduler.rs
  } | sha256sum | awk '{print $1}'
)

last_fp=""
stale_count=0
last_nudge_epoch=0
if [ -f "$STATE_FILE" ]; then
  # shellcheck disable=SC1090
  . "$STATE_FILE"
fi

status="healthy"
action="none"

if [ -z "$codex_lines" ]; then
  status="workflow_down"
  pid=$("$REPO_ROOT/scripts/start_ttft_workflow.sh")
  action="restarted_codex_pid_$pid"
  sleep 2
  codex_lines=$(ps -eo pid,ppid,etime,cmd | grep -E '/codex exec --dangerously-bypass-approvals-and-sandbox -C /workspace/atlas' | grep -v grep || true)
  mcp_lines=$(ps -eo pid,ppid,etime,cmd | grep -E 'npm exec @steipete/claude-code-mcp@latest|claude-code-mcp$' | grep -v grep || true)
  claude_lines=$(ps -eo pid,ppid,etime,cmd | grep -E '/workspace/.local/bin/claude( |$)|/workspace/.local/bin/claude-connect( |$)' | grep -v grep || true)
elif [ -z "$mcp_lines" ]; then
  status="degraded_no_mcp"
fi

if [ -n "$codex_lines" ]; then
  if [ "$fingerprint" = "${last_fp:-}" ]; then
    stale_count=$(( ${stale_count:-0} + 1 ))
  else
    stale_count=0
  fi
else
  stale_count=0
fi

if [ "$status" = "healthy" ] && [ -n "$codex_lines" ] && [ "$stale_count" -ge 3 ]; then
  status="stale_no_progress"
  if [ $(( now_epoch - ${last_nudge_epoch:-0} )) -ge 5400 ]; then
    pid=$("$REPO_ROOT/scripts/start_ttft_workflow.sh" nudge)
    action="nudged_codex_pid_$pid"
    last_nudge_epoch=$now_epoch
  else
    action="nudge_skipped_recently"
  fi
fi

cat >"$STATE_FILE" <<EOF
last_fp="$fingerprint"
stale_count=$stale_count
last_nudge_epoch=${last_nudge_epoch:-0}
EOF

{
  echo ""
  echo "### $(timestamp)"
  echo ""
  echo "- Status: $status"
  echo "- Action: $action"
  echo "- Fingerprint: $fingerprint"
  echo "- Stale count: $stale_count"
  echo "- Codex processes:"
  if [ -n "$codex_lines" ]; then
    echo '```text'
    echo "$codex_lines"
    echo '```'
  else
    echo '  - none'
  fi
  echo "- Claude MCP processes:"
  if [ -n "$mcp_lines" ]; then
    echo '```text'
    echo "$mcp_lines"
    echo '```'
  else
    echo '  - none'
  fi
  echo "- Claude CLI processes:"
  if [ -n "$claude_lines" ]; then
    echo '```text'
    echo "$claude_lines"
    echo '```'
  else
    echo '  - none'
  fi
  if [ -f "$RUN_LOG" ]; then
    echo "- Recent Codex log tail:"
    echo '```text'
    tail -n 20 "$RUN_LOG" || true
    echo '```'
  fi
  if [ -f "$LAST_MSG" ]; then
    echo "- Last Codex message tail:"
    echo '```text'
    tail -n 20 "$LAST_MSG" || true
    echo '```'
  fi
} >>"$OPS_LOG"
