#!/bin/bash
# Queueing under a mixed-length burst: does SLAI (+ the 100 ms co-dispatch
# window) redistribute latency better than FIFO? Prefill is saturated
# (~630 tok/s, measured), so no arm can create throughput — the question is
# purely how the fixed service capacity is allocated across waiting requests.
#
#   fifo      --scheduling-policy fifo      (today's preset default)
#   slai      --scheduling-policy slai      (shortest-pending-first + TBT-aware
#                                            prefill deferral, --tbt-deadline-ms 100)
#   slai_cd   slai + ATLAS_PREFILL_CODISPATCH=1 (100 ms admission window, so a
#                                            burst is admitted as one cohort)
#
# Burst: one ~8000-token prompt submitted FIRST, then 3 x ~600-token — FIFO's
# worst case. All timings CLIENT-SIDE from submission (the server's TTFT starts
# at processing and hides queue wait). Prefix caching ON in every arm.
set -u
cd /home/ms/atlas/.claude/worktrees/exl3-research || exit 2
export LD_LIBRARY_PATH=/home/ms/nccl/build/lib:/usr/local/cuda/lib64 RUST_LOG=info ATLAS_NO_HW_PRECHECK=1
BIN=${BIN:-target/release/spark}
PORT=8899; CKPT=/tank/exl3-ckpt/qwen38-flash-next-4.05bpw
PAT="serv[e] qwen3.8-flash-next-exl3.*--port $PORT"
OUT=${OUT:-.research/exl3_decode_perf/ab_queueing_$(date +%Y%m%dT%H%M%S)}; mkdir -p "$OUT"
ARMS=${*:-"fifo slai slai_cd slai_win fifo_win"}
pgrep -f "spark serv[e]" >/dev/null && { echo "REFUSING: another spark serve is running"; exit 2; }
echo "FINGERPRINT ab_queueing_policy bin=$(sha256sum $BIN | cut -c1-16) git=$(git rev-parse --short HEAD) date=$(date -u +%FT%TZ) host=$(hostname) arms=$ARMS" | tee "$OUT/fingerprint.txt"
stop() { pkill -f "$PAT" 2>/dev/null; for i in $(seq 1 60); do pgrep -f "$PAT" >/dev/null || break; sleep 1; done; sleep 3; }
trap stop EXIT
for arm in $ARMS; do
  case $arm in
    fifo)    ENVS=""; XARGS="--scheduling-policy fifo" ;;
    slai)    ENVS=""; XARGS="--scheduling-policy slai --tbt-deadline-ms 100" ;;
    slai_cd) ENVS="ATLAS_PREFILL_CODISPATCH=1"; XARGS="--scheduling-policy slai --tbt-deadline-ms 100" ;;
    # The arm the old flag made impossible: a 100 ms admission window WITHOUT
    # arming the batched dispatch, so SLAI's shortest-pending-first has a real
    # queue to order instead of an empty one or a pre-fused cohort.
    slai_win) ENVS="ATLAS_PREFILL_ADMISSION_WINDOW_MS=100"; XARGS="--scheduling-policy slai --tbt-deadline-ms 100" ;;
    fifo_win) ENVS="ATLAS_PREFILL_ADMISSION_WINDOW_MS=100"; XARGS="--scheduling-policy fifo" ;;
    *) echo "unknown arm $arm"; exit 2 ;;
  esac
  echo "=== arm $arm env=[${ENVS:-<none>}] args=[$XARGS] $(date +%T)" | tee -a "$OUT/summary.txt"
  ( env $ENVS setsid "$BIN" serve qwen3.8-flash-next-exl3 --model-from-path "$CKPT" --bind 127.0.0.1 --port $PORT --no-tui $XARGS > "$OUT/serve_$arm.log" 2>&1 < /dev/null & )
  ok=0
  for i in $(seq 1 300); do
    curl -s -m 2 http://127.0.0.1:$PORT/v1/models | grep -aq qwen3.8-flash-next && { echo "READY ~${i}s" | tee -a "$OUT/summary.txt"; ok=1; break; }
    pgrep -f "$PAT" >/dev/null || { echo "SERVER EXITED" | tee -a "$OUT/summary.txt"; tail -12 "$OUT/serve_$arm.log" | cut -c1-200; exit 1; }
    sleep 1
  done
  [ $ok = 1 ] || exit 1
  # Arm proof: the scheduler names its policy at startup.
  sed 's/\x1b\[[0-9;]*m//g' "$OUT/serve_$arm.log" | grep -aE "Scheduling policy|Scheduler started" | cut -c1-190 | tee -a "$OUT/summary.txt"
  python3 -u .research/exl3_decode_perf/probe_queueing.py --port $PORT --label "$arm" --repeats 2 2>&1 | tee "$OUT/burst_$arm.txt" | grep -aE "^  rep|^SUMMARY" | tee -a "$OUT/summary.txt"
  stop
done
rm -f "$OUT"/serve_*.log
echo "=== records $OUT"; grep -aE "^=== arm|^SUMMARY|Scheduling policy" "$OUT/summary.txt"
