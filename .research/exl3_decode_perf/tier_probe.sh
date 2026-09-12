#!/bin/bash
# Does the EXL3 MoE overflow tier ever fire at the shipped cap (1024)?
#
# Counter-only telemetry (ATLAS_EXL3_MOE_TIER_STATS=1) on the host side of the
# tier select, which already reads expert_offsets — so this measures the real
# routing distribution at serving shapes with no extra device work.
#
# Two prompt lengths x 3 reps, plus a long-context decode leg so the sweep
# covers the chunk sizes prefill actually sees.
set -uo pipefail
cd /home/ms/atlas/.claude/worktrees/exl3-research
D=/home/ms/.claude/jobs/5a7bd33d/tmp/tier_probe
mkdir -p "$D"
PORT=8891

cleanup() { pkill -f "spark-exl[3] serve" >/dev/null 2>&1; sleep 5; }
trap cleanup EXIT
cleanup

echo "=== BUILD  $(date +%T)"
export PATH=/usr/local/cuda/bin:$PATH
export CUTLASS_HOME=/home/ms/cutlass FLASHINFER_HOME=/home/ms/flashinfer
export RUSTFLAGS="-L native=/home/ms/nccl/build/lib -L/usr/local/cuda/lib64"
export ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.8-flash-next ATLAS_TARGET_QUANT=nvfp4
cargo build --release -p spark-server --bin spark > "$D/build.log" 2>&1
rc=$?; echo "build exit=$rc"
[ "$rc" = 0 ] || { tail -25 "$D/build.log"; exit 1; }
cp -f target/release/spark /home/ms/spark-exl3

echo "=== SERVE (tier stats on)  $(date +%T)"
ATLAS_EXL3_MOE_TIER_STATS=1 BIN=/home/ms/spark-exl3 CTX=32768 SEQS=4 PORT=$PORT \
  setsid .research/exl3_decode_perf/run_exl3_single.sh > "$D/serve.log" 2>&1 &
ok=0
for i in $(seq 1 900); do
  curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -aq qwen3.8-flash-next && { echo "READY ~${i}s"; ok=1; break; }
  if [ "$i" -gt 30 ] && ! pgrep -f "spark-exl[3] serve" >/dev/null; then echo "SERVER EXITED ~${i}s"; break; fi
  sleep 1
done
[ "$ok" = 1 ] || { tail -30 "$D/serve.log"; exit 1; }

echo "=== PREFILL SWEEP  $(date +%T)"
python3 -u .research/exl3_decode_perf/measure_prefill.py --port $PORT \
  --tokens 4000 8000 11000 --repeats 3 > "$D/prefill.log" 2>&1
grep -a SUMMARY "$D/prefill.log"

echo "=== LONG-CONTEXT DECODE LEG  $(date +%T)"
python3 -u .research/exl3_decode_perf/measure_concurrency.py --port $PORT \
  --concurrency 2 --prompt-tokens 16000 --max-tokens 200 --repeats 2 \
  --salt 424242 --label tier > "$D/decode.log" 2>&1

echo "=== TIER STATS (all summaries):"
grep -a "EXL3 MoE tier stats" "$D/serve.log" | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-230
echo "=== DONE $(date +%T)"
