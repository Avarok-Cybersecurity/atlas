#!/bin/bash
# Two cross-stream serialization levers on the EXL3 preset, one variable per arm.
# Prefix caching stays ON in every arm (house rule; the warm cell needs it).
#
#   base    defaults: PLE fault workers 64 (the measured drive knee),
#           decode checkpoint every 4 blocks (64 generated tokens)
#   ple16   ATLAS_PLE_FAULT_WORKERS=16   — the pre-2026-09-06 cap. `resolve`
#           holds the PLE table mutex while faulting, so if that lock is a real
#           contention point the new default should help MORE at C=4 than C=1.
#   ckpt16  ATLAS_DECODE_CKPT_BLOCKS=16  — a snapshot every 256 generated
#           tokens instead of 64. Decode-time snapshot stalls measured ~20-25%
#           of decode at C=1; at C=4 there are 4x as many. The cost is coarser
#           Marconi anchors, which the WARM cell is here to catch.
#
# Cells per arm (all prompts unique/salted so the cold cells are honest):
#   cold1 / cold4  ~3000-token prompts, max_tokens=16  -> TTFT (PLE-miss heavy)
#   dec1  / dec4   ~600-token prompts,  max_tokens=400 -> decode tok/s (ckpt)
#   warm4          cold4's prompts again + a short suffix -> TTFT (anchor cost)
set -u
cd /home/ms/atlas/.claude/worktrees/exl3-research || exit 2
export LD_LIBRARY_PATH=/home/ms/nccl/build/lib:/usr/local/cuda/lib64 RUST_LOG=info ATLAS_NO_HW_PRECHECK=1
BIN=${BIN:-target/release/spark}
PORT=8899; CKPT=/tank/exl3-ckpt/qwen38-flash-next-4.05bpw
PAT="serv[e] qwen3.8-flash-next-exl3.*--port $PORT"
OUT=${OUT:-.research/exl3_decode_perf/ab_conc_serial_$(date +%Y%m%dT%H%M%S)}; mkdir -p "$OUT"
ARMS=${*:-"base ple16 ckpt16"}
pgrep -f "spark serv[e]" >/dev/null && { echo "REFUSING: another spark serve is running"; exit 2; }
echo "FINGERPRINT ab_concurrency_serialization bin=$(sha256sum $BIN | cut -c1-16) git=$(git rev-parse --short HEAD) date=$(date -u +%FT%TZ) host=$(hostname) arms=$ARMS" | tee "$OUT/fingerprint.txt"
stop() { pkill -f "$PAT" 2>/dev/null; for i in $(seq 1 60); do pgrep -f "$PAT" >/dev/null || break; sleep 1; done; sleep 3; }
trap stop EXIT
for arm in $ARMS; do
  case $arm in
    base)   ENVS="" ;;
    ple16)  ENVS="ATLAS_PLE_FAULT_WORKERS=16" ;;
    ckpt16) ENVS="ATLAS_DECODE_CKPT_BLOCKS=16" ;;
    *) echo "unknown arm $arm"; exit 2 ;;
  esac
  echo "=== arm $arm env=[${ENVS:-<none>}] $(date +%T)" | tee -a "$OUT/summary.txt"
  ( env $ENVS setsid "$BIN" serve qwen3.8-flash-next-exl3 --model-from-path "$CKPT" --bind 127.0.0.1 --port $PORT --no-tui > "$OUT/serve_$arm.log" 2>&1 < /dev/null & )
  ok=0
  for i in $(seq 1 300); do
    curl -s -m 2 http://127.0.0.1:$PORT/v1/models | grep -aq qwen3.8-flash-next && { echo "READY ~${i}s" | tee -a "$OUT/summary.txt"; ok=1; break; }
    pgrep -f "$PAT" >/dev/null || { echo "SERVER EXITED" | tee -a "$OUT/summary.txt"; tail -12 "$OUT/serve_$arm.log" | cut -c1-200; exit 1; }
    sleep 1
  done
  [ $ok = 1 ] || exit 1
  python3 -u .research/exl3_decode_perf/probe_conc_serialization.py --port $PORT --label "$arm" 2>&1 | tee "$OUT/cells_$arm.txt" | grep -aE "^CELL|^SUMMARY" | tee -a "$OUT/summary.txt"
  sed 's/\x1b\[[0-9;]*m//g' "$OUT/serve_$arm.log" | grep -ac "decode-ckpt SAVE" | sed "s/^/  decode-ckpt SAVE lines: /" | tee -a "$OUT/summary.txt"
  sed 's/\x1b\[[0-9;]*m//g' "$OUT/serve_$arm.log" | grep -a "PLE gather" | tail -3 | cut -c1-170 | tee -a "$OUT/summary.txt"
  stop
done
rm -f "$OUT"/serve_*.log
echo "=== records $OUT"; cat "$OUT/summary.txt"
