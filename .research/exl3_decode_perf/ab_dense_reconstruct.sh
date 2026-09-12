#!/bin/bash
# A/B for the EXL3 dense prefill RECONSTRUCT tier (branch
# perf/exl3-dense-reconstruct-prefill-tier, crates/spark-model/src/layers/ops/exl3_dense/reconstruct.rs).
#
# Three arms, ONE variable — ATLAS_EXL3_DENSE_RECONSTRUCT_ROWS (unset / 512 / 1024) — over the
# preset `spark serve qwen3.8-flash-next-exl3` on this branch's release binary. Fresh server per
# arm, same port, back-to-back, box otherwise idle (the script refuses to start while another
# `spark serve` is alive — one Atlas instance at a time).
#
# Per arm:
#   1. boot, wait for /v1/models, snapshot `free -g`, grep the tier's load line (proves the arm is
#      LIVE, not inert — "ARMED ... m >= N rows" or "off (default)")
#   2. measure_prefill.py --tokens 8000 11000 --repeats 2   (cold prefill tok/s; THE metric)
#   3. measure_decode.py --repeats 3 --max-tokens 300        (decode sanity: m<=8 must be untouched;
#      under the preset's 2 MTP drafts read the tokens/wall column, not the streaming gap)
#   4. greedy 200-token sample (non-streaming, temp 0) + sha256 — EXPECTED TO DIFFER between the
#      off arm and the armed arms (BF16-rounded reconstructed weight + different reduction order);
#      recorded so the operator can read both and judge coherence. The two armed arms should be
#      byte-identical to each other (same kernels, only the threshold differs and both are below
#      the 8192-row prefill chunk).
#   5. stop the server, wait for the process to exit
#
# Nothing here is measured yet: every expectation in the module doc is a hypothesis until this runs.
# Usage:  bash .research/exl3_decode_perf/ab_dense_reconstruct.sh [arms...]   (default: off r512 r1024)
#   BIN=/path/to/spark overrides the binary; PORT overrides 8899; REPEATS_PREFILL / REPEATS_DECODE.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
BIN=${BIN:-$ROOT/target/release/spark}
PORT=${PORT:-8899}
OUT=${OUT:-$HERE/ab_dense_reconstruct}
CKPT=/tank/exl3-ckpt/qwen38-flash-next-4.05bpw
ARMS=${*:-off r512 r1024}
mkdir -p "$OUT"
export LD_LIBRARY_PATH=/home/ms/nccl/build/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export RUST_LOG=${RUST_LOG:-info}
export ATLAS_NO_HW_PRECHECK=${ATLAS_NO_HW_PRECHECK:-1}

[ -x "$BIN" ] || { echo "no binary at $BIN (cargo build --release -p spark-server on the branch first)"; exit 2; }
if pgrep -f "spark serv[e]" >/dev/null; then
  echo "REFUSING: another spark serve is running (one Atlas instance at a time):"; pgrep -af "spark serv[e]" | cut -c1-160; exit 2
fi
echo "FINGERPRINT bin=$BIN sha256=$(sha256sum "$BIN" | cut -c1-16) git=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null) port=$PORT date=$(date -u +%FT%TZ) host=$(hostname)" | tee "$OUT/fingerprint.txt"
free -g | tee -a "$OUT/fingerprint.txt"

stop_server() {
  pkill -f "spark serv[e].*--port $PORT" 2>/dev/null
  for _ in $(seq 1 60); do pgrep -f "spark serv[e].*--port $PORT" >/dev/null || break; sleep 1; done
  pgrep -f "spark serv[e].*--port $PORT" >/dev/null && { echo "server still alive, SIGKILL"; pkill -9 -f "spark serv[e].*--port $PORT"; sleep 5; }
}

run_arm() {
  local ARM=$1 LOG="$OUT/serve_${1}.log"
  # The ONE variable. Everything else comes from the preset.
  unset ATLAS_EXL3_DENSE_RECONSTRUCT_ROWS ATLAS_NO_EXL3_DENSE_RECONSTRUCT
  case "$ARM" in
    off) ;;
    r*) export ATLAS_EXL3_DENSE_RECONSTRUCT_ROWS=${ARM#r} ;;
    *) echo "unknown arm $ARM (off | r<rows>)"; return 2 ;;
  esac
  echo "=== arm $ARM start $(date -u +%FT%TZ) ATLAS_EXL3_DENSE_RECONSTRUCT_ROWS=${ATLAS_EXL3_DENSE_RECONSTRUCT_ROWS:-<unset>}" | tee "$OUT/arm_${ARM}.txt"
  setsid "$BIN" serve qwen3.8-flash-next-exl3 --model-from-path "$CKPT" --bind 127.0.0.1 --port "$PORT" \
    > "$LOG" 2>&1 < /dev/null &
  local ready=0
  for i in $(seq 1 1200); do
    if curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then echo "READY after ~${i}s" | tee -a "$OUT/arm_${ARM}.txt"; ready=1; break; fi
    if ! pgrep -f "spark serv[e].*--port $PORT" >/dev/null; then echo "SERVER EXITED" | tee -a "$OUT/arm_${ARM}.txt"; tail -30 "$LOG" | cut -c1-220; return 1; fi
    sleep 1
  done
  [ "$ready" = 1 ] || { echo "not ready after 1200s" | tee -a "$OUT/arm_${ARM}.txt"; stop_server; return 1; }
  local MODEL
  MODEL=$(curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
  MODEL=${MODEL:-qwen3.8-flash-next}
  echo "served model id: $MODEL" | tee -a "$OUT/arm_${ARM}.txt"
  free -g | tee -a "$OUT/arm_${ARM}.txt"
  # Liveness of the arm: the stage logs exactly one of these at load.
  grep -aE "EXL3 dense reconstruct tier" "$LOG" | cut -c1-260 | tee -a "$OUT/arm_${ARM}.txt"
  grep -aE "EXL3 native dense stage allocated" "$LOG" | cut -c1-200 | tee -a "$OUT/arm_${ARM}.txt"
  # Warm-up: one short request so the first measured prefill is not also the first kernel JIT.
  curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"temperature\":0.0,\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}]}" >/dev/null

  # 2. Cold prefill throughput (salted unique prompts; the tier is on the critical path here).
  python3 -u "$HERE/measure_prefill.py" --port "$PORT" --model "$MODEL" --tokens 8000 11000 --repeats "${REPEATS_PREFILL:-2}" \
    > "$OUT/prefill_${ARM}.txt" 2>&1
  cat "$OUT/prefill_${ARM}.txt" | tee -a "$OUT/arm_${ARM}.txt"

  # 3. Decode sanity (the tier must never take m<=8; the preset runs 2 MTP drafts, so tokens/wall).
  python3 -u "$HERE/measure_decode.py" --port "$PORT" --model "$MODEL" --repeats "${REPEATS_DECODE:-3}" --max-tokens 300 \
    > "$OUT/decode_${ARM}.txt" 2>&1
  grep -E "FINGERPRINT|SUMMARY" "$OUT/decode_${ARM}.txt" | tee -a "$OUT/arm_${ARM}.txt"

  # 4. Greedy sample: a LONG prompt so the prefill actually crosses both thresholds (the sample is
  #    the output of a reconstruct-tier prefill), plus the standard short code prompt.
  python3 - "$PORT" "$MODEL" "$OUT/sample_long_${ARM}.txt" <<'PY'
import json, sys, urllib.request, random
port, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
rng = random.Random(20260906)
words = ("kernel stream tensor cache block schedule verify draft router expert layer norm gate highway "
         "carry snapshot commit rewind window hash gather slot arena page token prefix budget latency").split()
body = " ".join(rng.choice(words) for _ in range(1500))
prompt = ("Below is a log of about 1500 words. After it, write a short Rust function that counts word "
          "frequencies in a &str and returns the top 5 as a Vec<(String, usize)>, then list the three most "
          "frequent words in the log.\n\n" + body + "\n\nAnswer:")
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps({"model": model, "temperature": 0.0, "max_tokens": 200,
                     "chat_template_kwargs": {"reasoning_effort": "low"},
                     "messages": [{"role": "user", "content": prompt}]}).encode(),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=900) as r:
    d = json.load(r)
m = d["choices"][0]["message"]
with open(out, "w") as f:
    f.write(f"prompt_tokens={d['usage']['prompt_tokens']} completion_tokens={d['usage']['completion_tokens']}\n")
    f.write(m.get("reasoning_content") or m.get("reasoning") or "")
    f.write("\n=====CONTENT=====\n")
    f.write(m.get("content") or "")
PY
  curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "{
    \"model\":\"$MODEL\",\"temperature\":0.0,\"max_tokens\":200,
    \"chat_template_kwargs\":{\"reasoning_effort\":\"low\"},
    \"messages\":[{\"role\":\"user\",\"content\":\"Write a complete Rust implementation of an LRU cache with generic key and value types, using a HashMap and a doubly linked list of indices into a Vec arena. Include get, put, capacity handling, and unit tests. Explain each design decision briefly in comments.\"}]}" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); m=d["choices"][0]["message"]; print(m.get("reasoning_content") or m.get("reasoning") or ""); print("=====CONTENT====="); print(m.get("content"))' \
    > "$OUT/sample_short_${ARM}.txt" 2>&1
  for s in long short; do
    echo "sample_${s}: $(wc -c < "$OUT/sample_${s}_${ARM}.txt") bytes sha256 $(sha256sum "$OUT/sample_${s}_${ARM}.txt" | cut -c1-16)" | tee -a "$OUT/arm_${ARM}.txt"
  done
  grep -aiE "error|panick|CUDA_ERROR|illegal" "$LOG" | head -5 | cut -c1-200 | tee -a "$OUT/arm_${ARM}.txt"

  # 5. Stop.
  stop_server
  echo "=== arm $ARM done $(date -u +%FT%TZ)" | tee -a "$OUT/arm_${ARM}.txt"
}

for ARM in $ARMS; do run_arm "$ARM"; done

echo; echo "===== SUMMARY (prefill tok/s medians; decode tokens/wall; sample hashes) ====="
for ARM in $ARMS; do
  echo "--- $ARM"
  grep -a "reconstruct tier" "$OUT/arm_${ARM}.txt" | head -1 | cut -c1-160
  grep -a "^SUMMARY" "$OUT/prefill_${ARM}.txt" 2>/dev/null
  grep -a "^SUMMARY" "$OUT/decode_${ARM}.txt" 2>/dev/null
  grep -a "^sample_" "$OUT/arm_${ARM}.txt"
done
echo "Read the module doc before quoting: numerics differ by design; a prefill gain is only real if"
echo "the armed arms' samples are coherent AND the model-card agentic gate passes on the armed binary."
