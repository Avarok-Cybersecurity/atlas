#!/bin/bash
# A/B for the fused BF16 egress on the EXL3 dense decode arm (branch
# perf/exl3-gemm-bf16-egress). The kill switch ATLAS_EXL3_NO_FUSED_EGRESS is the
# ONLY variable: arm "off" sets it (the `_f32_abf16` GEMM + a separate
# `exl3_f32_to_bf16[_2d]` launch per dense projection), arm "on" leaves it unset
# (the `_f32_abf16_obf16` GEMM stores BF16(C) from its epilogue, no egress
# launch). Fresh server per arm, same binary, same flags, back to back.
#
# Two profiles, each measured with the existing measure_decode.py harness
# (code prompt, 300 tokens, temp 0, streaming, 3 repeats) plus a 200-token
# greedy sample that MUST be byte-identical across the two arms (asserted; the
# fused path is bit-exact by construction, so a differing sample is a BUG, not
# a numerics tradeoff):
#
#   mtp    — the named preset VERBATIM:
#            spark serve qwen3.8-flash-next-exl3 --model-from-path <ckpt>
#                  --bind 127.0.0.1 --port 8899
#            (2 MTP drafts, prefix caching ON, 128K x 4 seqs, util 0.72).
#            Under MTP the streaming gap median is meaningless (drafted tokens
#            arrive in bursts) — read decode_tok_s_median (server-attested
#            tokens / wall).
#   serial — the preset cannot drop `--speculative` (a presence flag has no
#            negation), so the serial profile is the existing serve_exl3.sh
#            flag set on the same checkpoint: native MoE+dense+lm_head, no
#            speculation, C=1, 32K ctx, util 0.72, bf16 KV. Read gap_median_ms.
#
# Expected (HYPOTHESIS until this script has run): ~1 launch x ~2.6 us gap +
# ~1.3 us kernel fewer per dense projection (~61 per serial token, ~252 per
# 2-draft MTP step), i.e. the same order as the fused-ingress lever
# (+1.7% serial / +1.2% MTP). Anything the arm is NOT: a numerics change.
#
# Usage (GPU box, no other Atlas instance running — checked):
#   .research/exl3_decode_perf/ab_fused_egress.sh
#   PROFILES=mtp REPEATS=5 .research/exl3_decode_perf/ab_fused_egress.sh
#   SPARK_BIN=/path/to/spark .research/exl3_decode_perf/ab_fused_egress.sh
# Records land in .research/exl3_decode_perf/ab_fused_egress_<stamp>/:
#   measure_<profile>_<arm>.txt, sample_<profile>_<arm>.txt, serve_<profile>_<arm>.log,
#   SUMMARY.txt (tok/s per arm + the byte-equality verdict). Exit 1 on any
#   sample mismatch, arm-not-live, or server failure.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
BIN=${SPARK_BIN:-$ROOT/target/release/spark}
PORT=${PORT:-8899}
CKPT=${CKPT:-/tank/exl3-ckpt/qwen38-flash-next-4.05bpw}
PROFILES=${PROFILES:-"mtp serial"}
REPEATS=${REPEATS:-3}
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=${OUT:-$HERE/ab_fused_egress_$STAMP}
mkdir -p "$OUT"
SUMMARY=$OUT/SUMMARY.txt

export LD_LIBRARY_PATH=/home/ms/nccl/build/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export RUST_LOG=${RUST_LOG:-info}

die() { echo "FATAL: $*" | tee -a "$SUMMARY" >&2; stop_server; exit 1; }

[ -x "$BIN" ] || { echo "no binary at $BIN (build: cargo build --release -p spark-server)"; exit 1; }
if pgrep -af "spark serv[e]" >/dev/null; then
  echo "another Atlas instance is running (one at a time, house rule):"; pgrep -af "spark serv[e]" | cut -c1-120; exit 1
fi
echo "free -g before:"; free -g | head -2

{
  echo "ab_fused_egress $STAMP"
  echo "binary  $BIN"
  echo "sha256  $(sha256sum "$BIN" | cut -c1-16)"
  echo "commit  $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null) ($(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null))"
  echo "host    $(hostname)  port $PORT  ckpt $CKPT  repeats $REPEATS"
  echo
} | tee "$SUMMARY"

stop_server() {
  pkill -f "serv[e].*--port $PORT" 2>/dev/null
  for _ in $(seq 1 30); do pgrep -f "serv[e].*--port $PORT" >/dev/null || break; sleep 1; done
  pgrep -f "serv[e].*--port $PORT" >/dev/null && pkill -9 -f "serv[e].*--port $PORT"
  sleep 3
}

# start_server <profile> <arm>; the arm's env is the only difference between
# two launches of the same profile.
start_server() {
  local profile=$1 arm=$2 log=$OUT/serve_${1}_${2}.log
  local -a env_arm=()
  [ "$arm" = off ] && env_arm=(ATLAS_EXL3_NO_FUSED_EGRESS=1)
  case "$profile" in
    mtp)
      setsid env "${env_arm[@]}" "$BIN" serve qwen3.8-flash-next-exl3 \
        --model-from-path "$CKPT" --bind 127.0.0.1 --port "$PORT" \
        > "$log" 2>&1 < /dev/null &
      ;;
    serial)
      setsid env "${env_arm[@]}" \
        ATLAS_EXL3_NATIVE=1 ATLAS_EXL3_NATIVE_MOE=1 ATLAS_EXL3_NATIVE_DENSE=1 \
        ATLAS_PLE_MAX_TOKENS=9216 ATLAS_PLE_CACHE_SLOTS=4194304 ATLAS_QSA_MAX_TOKENS=32768 \
        ATLAS_INTHINK_TOOL_LEAK_OPENERS=0 ATLAS_NO_HW_PRECHECK=1 \
        "$BIN" serve --model-from-path "$CKPT" \
        --model-name qwen3.8-flash-next --kernel-target qwen3.8-flash-next \
        --world-size 1 --bind 127.0.0.1 --port "$PORT" \
        --max-seq-len 32768 --max-num-seqs 1 --max-batch-size 1 \
        --gpu-memory-utilization 0.72 --kv-cache-dtype bf16 --ssm-cache-slots 16 \
        --request-timeout 1800 --fast-load-prefetch-shards \
        --default-chat-template-kwargs '{"reasoning_effort":"low"}' \
        > "$log" 2>&1 < /dev/null &
      ;;
    *) die "unknown profile $profile" ;;
  esac
  for i in $(seq 1 1200); do
    if curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then echo "READY after ~${i}s ($profile/$arm)"; return 0; fi
    if ! pgrep -f "serv[e].*--port $PORT" >/dev/null; then tail -30 "$log" | cut -c1-200; die "server exited ($profile/$arm)"; fi
    sleep 1
  done
  die "server not ready after 1200s ($profile/$arm)"
}

model_id() {
  curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])'
}

# One arm: boot, measure, sample, verify the arm is the one we think it is, stop.
run_arm() {
  local profile=$1 arm=$2
  start_server "$profile" "$arm"
  local model; model=$(model_id)
  python3 -u "$HERE/measure_decode.py" --port "$PORT" --model "$model" --repeats "$REPEATS" --max-tokens 300 \
    > "$OUT/measure_${profile}_${arm}.txt" 2>&1
  tail -1 "$OUT/measure_${profile}_${arm}.txt"
  curl -s -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{
    "model":"'"$model"'","temperature":0.0,"max_tokens":200,
    "chat_template_kwargs":{"reasoning_effort":"low"},
    "messages":[{"role":"user","content":"Write a complete Rust implementation of an LRU cache with generic key and value types, using a HashMap and a doubly linked list of indices into a Vec arena. Include get, put, capacity handling, and unit tests. Explain each design decision briefly in comments."}]}' \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); m=d["choices"][0]["message"]; print(m.get("reasoning_content") or m.get("reasoning") or ""); print("=====CONTENT====="); print(m.get("content"))' \
    > "$OUT/sample_${profile}_${arm}.txt" 2>&1
  # The arm must be LIVE: the dense arm logs its egress plan once at first
  # dispatch (`EXL3 dense decode egress`). A missing or wrong line means the
  # binary is not this branch's or the env did not reach the process.
  local want; [ "$arm" = on ] && want="fused" || want="separate"
  if ! grep -a "EXL3 dense decode egress" "$OUT/serve_${profile}_${arm}.log" | grep -q "$want"; then
    grep -a "EXL3 dense decode egress" "$OUT/serve_${profile}_${arm}.log" | head -2 | cut -c1-200
    die "arm $profile/$arm is not live (expected '$want' in the egress log line)"
  fi
  stop_server
}

for profile in $PROFILES; do
  for arm in off on; do
    echo "=== $profile / $arm ($(date +%H:%M:%S))" | tee -a "$SUMMARY"
    run_arm "$profile" "$arm"
    tail -1 "$OUT/measure_${profile}_${arm}.txt" | tee -a "$SUMMARY"
    echo "sample sha256 $(sha256sum "$OUT/sample_${profile}_${arm}.txt" | cut -c1-16)  bytes $(wc -c < "$OUT/sample_${profile}_${arm}.txt")" | tee -a "$SUMMARY"
  done
  if cmp -s "$OUT/sample_${profile}_off.txt" "$OUT/sample_${profile}_on.txt"; then
    echo "SAMPLE $profile: BYTE-IDENTICAL across arms" | tee -a "$SUMMARY"
  else
    diff "$OUT/sample_${profile}_off.txt" "$OUT/sample_${profile}_on.txt" | head -20 | tee -a "$SUMMARY"
    die "SAMPLE $profile: arms DIFFER — the fused egress is supposed to be bit-exact; do not ship"
  fi
done

echo | tee -a "$SUMMARY"
echo "free -g after:" | tee -a "$SUMMARY"; free -g | head -2 | tee -a "$SUMMARY"
echo "records: $OUT" | tee -a "$SUMMARY"
