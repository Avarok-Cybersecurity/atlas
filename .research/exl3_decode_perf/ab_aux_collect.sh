#!/bin/bash
# A/B: batched Marconi aux collect vs the legacy per-layer draining collect.
#
# ONE variable: ATLAS_AUX_COLLECT_BATCHED (0 = legacy, unset = batched). Same
# binary, same preset, same checkpoint, same harness params — so a difference
# here is the collect and nothing else.
#
# Counterbalanced B A A B: a straight A,B pair cannot separate the arm from
# drift (thermal, page-cache, allocator), and this box has produced exactly
# that artifact before (the admission-window "4.5x" that a 5-rep control
# erased).
#
# The collect fires on chunk-boundary snapshot saves and every
# ATLAS_DECODE_CKPT_BLOCKS (4) blocks = 64 decode tokens, and the QSA half of
# each blob is O(context). So the effect must be looked for at LONG context;
# a 300-token probe is where this hides, not where it shows.
set -uo pipefail
cd "$(dirname "$0")/../.."
D=/home/ms/.claude/jobs/5a7bd33d/tmp/aux_ab
mkdir -p "$D"
BIN="${BIN:-/home/ms/spark-exl3}"
PORT=8891
CTX="${CTX:-32768}"
REPS="${REPS:-3}"

cleanup() { pkill -f "spark-exl[3] serve" >/dev/null 2>&1; sleep 5; }
trap cleanup EXIT
cleanup

run_arm() {
  local arm="$1" tag="$2"
  echo "=== ARM $arm ($tag)  $(date +%T)"
  if [ "$arm" = "legacy" ]; then export ATLAS_AUX_COLLECT_BATCHED=0; else unset ATLAS_AUX_COLLECT_BATCHED; fi
  BIN=$BIN CTX=$CTX SEQS=4 PORT=$PORT setsid .research/exl3_decode_perf/run_exl3_single.sh \
    > "$D/serve_$tag.log" 2>&1 &
  local ok=0
  for i in $(seq 1 900); do
    curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -aq qwen3.8-flash-next && { echo "READY ~${i}s"; ok=1; break; }
    if [ "$i" -gt 30 ] && ! pgrep -f "spark-exl[3] serve" >/dev/null; then echo "SERVER EXITED ~${i}s"; break; fi
    sleep 1
  done
  if [ "$ok" != 1 ]; then echo "--- serve tail:"; tail -25 "$D/serve_$tag.log"; return 1; fi

  # Arm-liveness proof: the server must SAY which path it took. Env alone is
  # an intention, not evidence.
  echo "--- arm proof: aux_batched=${ATLAS_AUX_COLLECT_BATCHED:-unset}"

  # Prefill: the collect also runs at every chunk-boundary snapshot save.
  python3 -u .research/exl3_decode_perf/measure_prefill.py --port $PORT \
    --tokens 8000 11000 --repeats "$REPS" > "$D/prefill_$tag.log" 2>&1
  # Decode at LONG context, enough tokens for several 64-token checkpoints.
  python3 -u .research/exl3_decode_perf/measure_concurrency.py --port $PORT \
    --concurrency 1 --prompt-tokens 16000 --max-tokens 400 --repeats "$REPS" \
    --salt 424242 --label "aux_$tag" > "$D/decode_$tag.log" 2>&1
  grep -a "aux collect:" "$D/serve_$tag.log" | sed 's/\x1b\[[0-9;]*m//g' | tail -2
  cleanup
  echo "=== ARM $arm done $(date +%T)"
}

run_arm batched b1
run_arm legacy  a1
run_arm legacy  a2
run_arm batched b2
echo "ALL DONE — logs in $D"
