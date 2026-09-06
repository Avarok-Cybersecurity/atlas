#!/bin/bash
# Merged-binary gate for wip/exl3-research after the round-2 levers landed (fused egress default-on,
# MoE row cap 1024 default, dense reconstruct tier opt-in). One boot per arm of the NAMED PRESET
# (`spark serve qwen3.8-flash-next-exl3`: 128K x 4 seqs, util 0.72, 2 MTP drafts, prefix caching ON,
# reasoning_effort low + preserve_thinking), then on that server, in order:
#   1. measure_prefill.py 8000 11000 x2 (cold, salted prompts)      -> prefill_<arm>.txt
#   2. measure_decode.py x3 (code prompt, 300 tokens, temp 0)        -> decode_<arm>.txt
#   3. 200-token greedy sample + sha256                              -> sample_<arm>.txt
#   4. agentic-webserver benchmark, 1 iteration, model-card sampling -> agentic_<arm>.json / .progress.log
# Arms: default (no extra env) and recon512 (ATLAS_EXL3_DENSE_RECONSTRUCT_ROWS=512). The env is the only
# variable. Usage: bash merged_gate.sh [arms...]   (SPARK_BIN, PORT=8888, OUT override)
set -u
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd)
BIN=${SPARK_BIN:-$ROOT/target/release/spark}
PORT=${PORT:-8888}
CKPT=/tank/exl3-ckpt/qwen38-flash-next-4.05bpw
ARMS=${*:-default recon512}
OUT=${OUT:-$HERE/merged_gate_$(date +%Y%m%dT%H%M%S)}; mkdir -p "$OUT"
export LD_LIBRARY_PATH=/home/ms/nccl/build/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export RUST_LOG=${RUST_LOG:-info} ATLAS_NO_HW_PRECHECK=1 ATLAS_MTP_ACCEPT_DEBUG=1
[ -x "$BIN" ] || { echo "no binary at $BIN"; exit 2; }
pgrep -f "spark serv[e]" >/dev/null && { echo "REFUSING: another spark serve is running"; pgrep -af "spark serv[e]" | cut -c1-160; exit 2; }
GITHEAD=$(cd "$ROOT" && git rev-parse --short HEAD)
echo "FINGERPRINT merged_gate bin=$BIN sha256=$(sha256sum "$BIN" | cut -c1-16) git=$GITHEAD port=$PORT date=$(date -u +%FT%TZ) host=$(hostname) preset=qwen3.8-flash-next-exl3 ckpt=$CKPT arms=$ARMS" | tee "$OUT/fingerprint.txt"
free -g | tee -a "$OUT/fingerprint.txt"
SB=/home/ms/.atlas/runs/agentic-webserver/sandbox
PAT="serv[e] qwen3.8-flash-next-exl3.*--port $PORT"
stop_server() { pkill -f "$PAT" 2>/dev/null; for i in $(seq 1 60); do pgrep -f "$PAT" >/dev/null || break; sleep 1; done; sleep 3; }
trap stop_server EXIT
for arm in $ARMS; do
  echo "=== arm $arm start $(date -u +%FT%TZ)" | tee -a "$OUT/summary.txt"
  case $arm in
    default)   ENVS="" ;;
    recon512)  ENVS="ATLAS_EXL3_DENSE_RECONSTRUCT_ROWS=512" ;;
    recon1024) ENVS="ATLAS_EXL3_DENSE_RECONSTRUCT_ROWS=1024" ;;
    *) echo "unknown arm $arm"; exit 2 ;;
  esac
  echo "env: ${ENVS:-<none>}" | tee -a "$OUT/summary.txt"
  ( cd "$ROOT" && env $ENVS setsid "$BIN" serve qwen3.8-flash-next-exl3 --model-from-path "$CKPT" --bind 127.0.0.1 --port "$PORT" > "$OUT/serve_$arm.log" 2>&1 < /dev/null & )
  ok=0
  for i in $(seq 1 900); do
    curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" | grep -aq qwen3.8-flash-next && { echo "READY after ~${i}s" | tee -a "$OUT/summary.txt"; ok=1; break; }
    pgrep -f "$PAT" >/dev/null || { echo "SERVER EXITED" | tee -a "$OUT/summary.txt"; tail -20 "$OUT/serve_$arm.log" | cut -c1-200; exit 1; }
    sleep 1
  done
  [ $ok = 1 ] || { echo "boot timeout"; exit 1; }
  sed 's/\x1b\[[0-9;]*m//g' "$OUT/serve_$arm.log" | grep -aE "rows/expert|dense decode egress|reconstruct tier" | cut -c1-260 | tee -a "$OUT/summary.txt"
  python3 -u "$HERE/measure_prefill.py" --port "$PORT" --tokens 8000 11000 --repeats 2 > "$OUT/prefill_$arm.txt" 2>&1
  grep -a "SUMMARY" "$OUT/prefill_$arm.txt" | tee -a "$OUT/summary.txt"
  python3 -u "$HERE/measure_decode.py" --port "$PORT" --repeats 3 --max-tokens 300 > "$OUT/decode_$arm.txt" 2>&1
  grep -a "SUMMARY" "$OUT/decode_$arm.txt" | tee -a "$OUT/summary.txt"
  curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' -d '{"model":"qwen3.8-flash-next","temperature":0,"max_tokens":200,"messages":[{"role":"user","content":"Write a Python function that merges two sorted lists into one sorted list, then explain its complexity in two sentences."}]}' | python3 -c 'import sys,json; d=json.load(sys.stdin); m=d["choices"][0]["message"]; print(m.get("reasoning_content") or m.get("reasoning") or ""); print("----"); print(m["content"])' > "$OUT/sample_$arm.txt" 2>&1
  echo "sample $arm: $(wc -c < "$OUT/sample_$arm.txt") bytes sha256 $(sha256sum "$OUT/sample_$arm.txt" | cut -c1-16)" | tee -a "$OUT/summary.txt"
  rm -f "$SB"/*.trajectory.txt 2>/dev/null
  START=$(date +%s)
  ATLAS_AGENTIC_PRESERVE_THINKING=1 ATLAS_AGENTIC_SAMPLING=model-card "$BIN" benchmark run agentic-webserver --yes --url "http://127.0.0.1:$PORT" --model qwen3.8-flash-next \
    --param iterations=1 --param wall_budget_s=1000 --no-fail-on-verdict --format json > "$OUT/agentic_$arm.json" 2> "$OUT/agentic_$arm.progress.log"
  echo "agentic $arm: exit=$? wall=$(( $(date +%s) - START ))s" | tee -a "$OUT/summary.txt"
  python3 - "$OUT/agentic_$arm.json" <<'PY' | tee -a "$OUT/summary.txt"
import json,sys
try:
    d=json.load(open(sys.argv[1]))
except Exception as e:
    print("agentic json unreadable:", e); sys.exit()
def walk(o):
    if isinstance(o,dict):
        for k,v in o.items():
            if k in ("verdict","webserver_ok","followed","turns","pass","passed","score","summary","status","iterations_ok"):
                print(f"  {k}: {str(v)[:160]}")
            walk(v)
    elif isinstance(o,list):
        for x in o[:20]: walk(x)
walk(d)
PY
  mkdir -p "$OUT/agentic_${arm}_trajectories"; cp "$SB"/*.trajectory.txt "$OUT/agentic_${arm}_trajectories/" 2>/dev/null
  grep -ac "MTP accept" "$OUT/serve_$arm.log" | sed 's/^/MTP accept lines: /' | tee -a "$OUT/summary.txt"
  stop_server
  echo "=== arm $arm done $(date -u +%FT%TZ)" | tee -a "$OUT/summary.txt"
done
echo "=== SUMMARY $OUT"; cat "$OUT/summary.txt"
