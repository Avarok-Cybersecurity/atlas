#!/bin/bash
# Run ONE arm end to end, fully detached-safe:
#   $1 = tag (e.g. A1)   $2 = on|off (prefix caching)
# boot serve -> wait for ready -> clear sandbox -> bench (3 iters) -> preserve
# trajectories -> kill serve.
set -uo pipefail
TAG=$1
ARM=$2
OUT=/home/ms/.claude/jobs/5a7bd33d/tmp/mangle2
SB=/home/ms/.atlas/runs/agentic-webserver/sandbox

echo "=== ARM $TAG prefix=$ARM start $(date -u +%FT%TZ) ==="

nohup setsid stdbuf -oL -eL "$OUT/boot.sh" "$ARM" > "$OUT/serve.$TAG.log" 2>&1 < /dev/null &
sleep 2
SPID=$(pgrep -f "exl3-research/target/release/spark serv[e]" | head -1)
echo "serve pid $SPID"

# wait up to 10 min for /v1/models to answer
ready=0
for i in $(seq 1 120); do
  if curl -s --max-time 4 http://127.0.0.1:8890/v1/models | grep -a -q qwen4exp-exl3; then ready=1; break; fi
  if ! pgrep -f "exl3-research/target/release/spark serv[e]" >/dev/null; then
    echo "SERVE DIED during load at $(date -u +%FT%TZ)"; exit 4
  fi
  sleep 5
done
[ "$ready" = 1 ] || { echo "serve never became ready"; exit 5; }
echo "serve ready $(date -u +%FT%TZ)"
BENCH_START=$(date -u +%FT%TZ)

rm -f "$SB"/*.trajectory.txt

export LD_LIBRARY_PATH="/home/ms/atlas-gdn-libs:/home/ms/nccl/build/lib:${LD_LIBRARY_PATH:-}"
export ATLAS_AGENTIC_SAMPLING=model-card
export ATLAS_NO_HW_PRECHECK=1

/home/ms/atlas/.claude/worktrees/exl3-research/target/release/spark benchmark run agentic-webserver \
  --yes \
  --url http://127.0.0.1:8890 \
  --model qwen4exp-exl3 \
  --param iterations=3 \
  --param wall_budget_s=9000 \
  --no-fail-on-verdict \
  --format json \
  > "$OUT/$TAG.json" 2> "$OUT/$TAG.progress.log"
rc=$?
BENCH_END=$(date -u +%FT%TZ)
echo "benchmark exit code: $rc"
echo "window $BENCH_START .. $BENCH_END" > "$OUT/$TAG.window.txt"

mkdir -p "$OUT/$TAG-trajectories"
rm -f "$OUT/$TAG-trajectories"/*
cp "$SB"/*.trajectory.txt "$OUT/$TAG-trajectories/" 2>/dev/null
echo "trajectories: $(ls "$OUT/$TAG-trajectories" | wc -l)"

pkill -f "exl3-research/target/release/spark serv[e]"
sleep 20
pgrep -f "exl3-research/target/release/spark serv[e]" && { pkill -9 -f "exl3-research/target/release/spark serv[e]"; sleep 15; }
echo "=== ARM $TAG done $(date -u +%FT%TZ) ==="
free -g
