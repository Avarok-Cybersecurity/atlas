#!/bin/bash
# MoE row-cap sweep: does raising the cap pull the (100%-firing) overflow tier
# back into the single fused launch, and does prefill move?
#
# ONE variable: ATLAS_EXL3_MOE_ROWS_PER_EXPERT (1024 = shipped default, then
# 2048, 4096). Same binary, same preset, tier telemetry on in every arm so each
# arm reports its OWN overflow rate — the arm proves itself.
#
# Cost side (moe_prefill_cap.rs arithmetic, C=6, hidden 2560, inter 640):
#   1024 -> 78.6 MB temp slabs, 2048 -> 157.3 MB, 4096 -> 314.6 MB,
# against the 419 MB deterministic slot slab already paid.
#
# Counterbalanced: 1024, 2048, 4096, then 4096, 2048, 1024 — a straight ladder
# cannot separate the knob from drift, and this box has produced exactly that
# artifact before.
set -uo pipefail
cd /home/ms/atlas/.claude/worktrees/exl3-research
D=/home/ms/.claude/jobs/5a7bd33d/tmp/cap_sweep
mkdir -p "$D"
PORT=8891

cleanup() { pkill -f "spark-exl[3] serve" >/dev/null 2>&1; sleep 5; }
trap cleanup EXIT
cleanup

run_arm() {
  local cap="$1" tag="$2"
  echo "=== CAP $cap ($tag)  $(date +%T)"
  ATLAS_EXL3_MOE_TIER_STATS=1 MOE_ROWS=$cap \
  BIN=/home/ms/spark-exl3 CTX=32768 SEQS=4 PORT=$PORT \
    setsid .research/exl3_decode_perf/run_exl3_single.sh > "$D/serve_$tag.log" 2>&1 &
  local ok=0
  for i in $(seq 1 900); do
    curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -aq qwen3.8-flash-next && { echo "READY ~${i}s"; ok=1; break; }
    if [ "$i" -gt 30 ] && ! pgrep -f "spark-exl[3] serve" >/dev/null; then echo "SERVER EXITED ~${i}s"; break; fi
    sleep 1
  done
  if [ "$ok" != 1 ]; then echo "--- serve tail:"; tail -25 "$D/serve_$tag.log"; cleanup; return 1; fi

  python3 -u .research/exl3_decode_perf/measure_prefill.py --port $PORT \
    --tokens 4000 8000 11000 --repeats 3 > "$D/prefill_$tag.log" 2>&1
  grep -a SUMMARY "$D/prefill_$tag.log"
  echo "--- tier stats (this arm):"
  grep -a "EXL3 MoE tier stats" "$D/serve_$tag.log" | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-235 | tail -1
  echo "--- resolved cap (arm proof):"
  grep -aiE "rows.per.expert|row cap" "$D/serve_$tag.log" | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-190 | head -2
  cleanup
}

run_arm 1024 a1
run_arm 2048 b1
run_arm 4096 c1
run_arm 4096 c2
run_arm 2048 b2
run_arm 1024 a2
echo "=== ALL DONE $(date +%T)"
