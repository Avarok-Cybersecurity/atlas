#!/usr/bin/env bash
set -euo pipefail

# Repository root (override via REPO_ROOT env). Defaults to the directory
# this script lives in two levels up.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TASK_DIR="${TASK_DIR:-$REPO_ROOT/tasks/codex_agent_2026-03-23}"
# Codex CLI binary (override via CODEX_BIN env). Default: discover via PATH.
CODEX_BIN="${CODEX_BIN:-$(command -v codex 2>/dev/null || echo codex)}"
# Claude Code CLI binary (override via CLAUDE_BIN env).
CLAUDE_BIN="${CLAUDE_BIN:-$(command -v claude 2>/dev/null || echo claude)}"
RUN_LOG="$TASK_DIR/ttft-overnight-codex.log"
LAST_MSG="$TASK_DIR/ttft-overnight-last-message.txt"
MODE="${1:-main}"

MAIN_PROMPT=$(cat <<'EOF'
Atlas repo. Tonight's primary goal is to reduce long-context TTFT for 16K+ prompts, especially 32K and 64K, toward roughly 20% of the former latency baseline while preserving TPOT and coherence.

Keep your role as planner/reviewer.
Use claude_workhorse as the implementation workhorse only for bounded slices.
First confirm the current chunked-prefill metadata hot path in crates/spark-model/src/model.rs.
Do not touch crates/spark-server/src/scheduler.rs unless forced.

Bias toward low-risk wins first:
- remove per-chunk waste
- reduce metadata rebuild/upload overhead
- audit stream ownership

Do not lead with chunk-size changes.

Before any rebuild inspect the diff and check OOM risk.
If runtime behavior changed, rebuild from scratch with no cache.

Validate with:
- long-context TTFT
- 256/256 TPOT
- 1024/1024 TPOT
- ISL=1024 conc=8 TPOT
- ISL=1024 conc=16 TPOT
- representative long-context coherence

Commit only after a slice is validated.
EOF
)

NUDGE_PROMPT=$(cat <<'EOF'
Atlas repo. This is an overnight rescue/nudge run for the long-context TTFT effort.

The standing goal remains:
- reduce long-context TTFT for 16K+ prompts, especially 32K and 64K, toward roughly 20% of the former latency baseline
- preserve TPOT and coherence

First inspect:
- tasks/codex_agent_2026-03-23/ttft-overnight-ops-2026-03-23.md
- current worktree diff
- the existing TTFT slice around chunked-prefill metadata in crates/spark-model/src/model.rs

If another Codex session is already making concrete progress, do not duplicate work blindly.
Instead, pick the next smallest helpful action:
- unblock a stall
- narrow the next implementation slice
- run the next required validation step
- or document the current blocker and exact next move

Keep scheduler.rs out unless forced.
Do not lead with chunk-size changes.
Use claude_workhorse only for bounded implementation slices.
EOF
)

case "$MODE" in
  main) PROMPT="$MAIN_PROMPT" ;;
  nudge) PROMPT="$NUDGE_PROMPT" ;;
  *) echo "unknown mode: $MODE" >&2; exit 1 ;;
esac

mkdir -p "$TASK_DIR"
touch "$RUN_LOG"

nohup "$CODEX_BIN" exec \
  --dangerously-bypass-approvals-and-sandbox \
  -C "$REPO_ROOT" \
  -c 'mcp_servers.claude_workhorse.command="npx"' \
  -c 'mcp_servers.claude_workhorse.args=["-y","@steipete/claude-code-mcp@latest"]' \
  -c 'mcp_servers.claude_workhorse.startup_timeout_sec=20' \
  -c 'mcp_servers.claude_workhorse.tool_timeout_sec=900' \
  -c "mcp_servers.claude_workhorse.env.CLAUDE_CLI_NAME=\"$CLAUDE_BIN\"" \
  -o "$LAST_MSG" \
  "$PROMPT" >>"$RUN_LOG" 2>&1 &

echo $!
