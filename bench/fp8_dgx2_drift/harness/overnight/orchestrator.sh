#!/usr/bin/env bash
# Overnight graduated-completion orchestrator (12h budget).
#
# Walks the graduated ladder easy->hard. For each level it runs BOTH clients
# (opencode + Claude Code) via run_level.py and checks completion (cargo
# build+test). On PASS it git-commits (iteration name + prompt) and advances.
# On FAIL it leaves the bug sentinel (/workspace/overnight_BUG.json) and WAITS
# for the orchestrating agent to either fix-and-RESUME or SKIP — auto-skipping
# after OVN_FIX_WAIT so it never hangs forever while unattended.
#
# Signals (touched by the agent on wakeup):
#   /workspace/overnight_RESUME  -> retry the current (failed) level
#   /workspace/overnight_SKIP    -> give up on this level, advance
set -uo pipefail

H="$(cd "$(dirname "$0")/.." && pwd)"           # harness dir
OVN="$H/overnight"
REPO=/workspace/atlas-mtp
LOG=/workspace/overnight.log
RESULTS="$OVN/results.jsonl"
BUG=/workspace/overnight_BUG.json
RESUME=/workspace/overnight_RESUME
SKIP=/workspace/overnight_SKIP
DONE=/workspace/overnight_DONE

DURATION_H="${OVN_DURATION_H:-12}"
DEADLINE=$(( $(date +%s) + DURATION_H*3600 ))
FIX_WAIT="${OVN_FIX_WAIT:-5400}"                 # 90 min max wait for a fix
NLEVELS=$(grep -c . "$H/prompts/graduated.jsonl")

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

commit_pass(){
  local lvl="$1" name="$2"
  local pfirst; pfirst=$(python3 -c "import json,sys
for l in open('$H/prompts/graduated.jsonl'):
    d=json.loads(l)
    if d['level']==$lvl: print(d['prompt'].splitlines()[0][:100]); break")
  cd "$REPO"
  git add -A bench/fp8_dgx2_drift/harness >/dev/null 2>&1 || true
  # include any source the agent fixed for this iteration
  git add -A crates kernels 2>/dev/null || true
  git -c user.name='Azeez Ishaqui' -c user.email='debaterishaqui@gmail.com' \
    commit -q -m "overnight L${lvl} ${name}: PASS [opencode+claude-code]

Prompt: ${pfirst}

Co-Authored-By: Azeez Ishaqui <debaterishaqui@gmail.com>" >/dev/null 2>&1 \
    && log "committed L${lvl} ${name} PASS ($(git rev-parse --short HEAD))" \
    || log "nothing to commit for L${lvl} (clean tree)"
}

log "=== OVERNIGHT ORCHESTRATOR START — ${NLEVELS} levels, ${DURATION_H}h budget, deadline $(date -d @${DEADLINE} '+%H:%M' 2>/dev/null||echo +${DURATION_H}h) ==="
rm -f "$RESUME" "$SKIP" "$DONE"

for (( lvl=1; lvl<=NLEVELS; lvl++ )); do
  name=$(python3 -c "import json
for l in open('$H/prompts/graduated.jsonl'):
    d=json.loads(l)
    if d['level']==$lvl: print(d['name']); break")
  while :; do
    if (( $(date +%s) >= DEADLINE )); then log "DEADLINE reached — stopping at L${lvl}"; touch "$DONE"; exit 0; fi
    log "--- L${lvl} ${name}: running both clients ---"
    rm -f "$BUG"
    if python3 "$OVN/run_level.py" "$lvl" >>"$LOG" 2>&1; then
      log "L${lvl} ${name}: PASS"
      commit_pass "$lvl" "$name"
      break                                       # advance to next level
    fi
    log "L${lvl} ${name}: FAIL — bug sentinel written; waiting up to $((FIX_WAIT/60))m for RESUME/SKIP"
    waited=0
    while :; do
      [[ -f "$RESUME" ]] && { rm -f "$RESUME"; log "RESUME signalled — retrying L${lvl}"; break; }
      [[ -f "$SKIP" ]]   && { rm -f "$SKIP"; log "SKIP signalled — advancing past L${lvl}"; break 2; }
      if (( waited >= FIX_WAIT )); then log "fix-wait timeout — auto-SKIP L${lvl}"; break 2; fi
      if (( $(date +%s) >= DEADLINE )); then log "DEADLINE during fix-wait — stop"; touch "$DONE"; exit 0; fi
      sleep 30; waited=$((waited+30))
    done
    # RESUME path falls through to re-run the level
  done
done
log "=== ALL ${NLEVELS} LEVELS PROCESSED — overnight complete ==="
touch "$DONE"
