#!/bin/bash
# One agentic-webserver iteration against the fixed binary (PR #834 agentic profile).
set -u
D=/home/ms/.claude/jobs/5a7bd33d/tmp/exl3bench
TAG=${1:-agentic_fix}
LOG=$D/serve_${TAG}.log
SB=/home/ms/.atlas/runs/agentic-webserver/sandbox
echo "=== $TAG start $(date -u +%FT%TZ)"
setsid $D/serve_exl3_fix_agentic.sh 8888 > "$LOG" 2>&1 < /dev/null &
for i in $(seq 1 900); do
  if curl -s -m 2 http://127.0.0.1:8888/v1/models | grep -aq qwen3.8-flash-next; then echo "READY after ~${i}s"; break; fi
  if ! pgrep -f "serv[e] --model-from-path.*8888" >/dev/null; then echo "SERVER EXITED"; tail -20 "$LOG" | cut -c1-200; exit 1; fi
  sleep 1
done
rm -f "$SB"/*.trajectory.txt 2>/dev/null
export ATLAS_AGENTIC_PRESERVE_THINKING=1 ATLAS_NO_HW_PRECHECK=1; [ "${SAMPLING:-greedy}" = model-card ] && export ATLAS_AGENTIC_SAMPLING=model-card
export LD_LIBRARY_PATH=/home/ms/nccl/build/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
START=$(date +%s)
$D/spark-sharedgemv benchmark run agentic-webserver --yes --url http://127.0.0.1:8888 --model qwen3.8-flash-next \
  --param iterations=1 --param wall_budget_s=1000 --no-fail-on-verdict --format json \
  > $D/${TAG}.json 2> $D/${TAG}.progress.log
echo "benchmark exit=$? wall=$(( $(date +%s) - START ))s"
mkdir -p $D/${TAG}-trajectories; cp "$SB"/*.trajectory.txt $D/${TAG}-trajectories/ 2>/dev/null; ls $D/${TAG}-trajectories | wc -l
ls -t /home/ms/.atlas/runs/agentic-webserver/run-*.json | head -1
pkill -f "serv[e] --model-from-path.*8888"; sleep 8; echo "=== $TAG done $(date -u +%FT%TZ)"
