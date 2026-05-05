#!/usr/bin/env bash
# verify-high-speed-swap.sh — live-test harness for the --high-speed-swap
# integration. Spins up two spark servers (with HSS / without HSS), sends
# the same long-context prompt to both, and compares output.
#
# Run after stopping any existing spark server holding the GPU. Usage:
#
#     bash scripts/verify-high-speed-swap.sh \
#         /path/to/qwen3-bf16-model \
#         /mnt/nvme0/atlas-hsw-test
#
# Exits 0 on success (outputs match within tolerance); non-zero on failure.

set -euo pipefail

MODEL=${1:?"usage: verify-high-speed-swap.sh <model> <hsw-dir>"}
HSW_DIR=${2:?"usage: verify-high-speed-swap.sh <model> <hsw-dir>"}

if [[ -z "${SKIP_GPU_CHECK:-}" ]] && nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then
  echo "ERROR: another process is using the GPU; stop it first" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 | head -5 >&2
  exit 1
fi

PORT_HSW=${PORT_HSW:-8889}
PORT_REF=${PORT_REF:-8890}
SPARK=${SPARK:-./target/release/spark}
PROMPT_FILE=${PROMPT_FILE:-/tmp/atlas-hsw-prompt.txt}

# Generate a 30K-token prompt if not provided. Use lorem ipsum repeated
# to deterministic length.
if [[ ! -f "$PROMPT_FILE" ]]; then
  python3 -c "
import sys
chunk = 'The quick brown fox jumps over the lazy dog. ' * 200  # ~9000 chars
target_chars = 100_000  # ~30K tokens at 3.3 chars/token
out = (chunk * (target_chars // len(chunk) + 1))[:target_chars]
sys.stdout.write(out)
" > "$PROMPT_FILE"
fi
PROMPT_BYTES=$(wc -c < "$PROMPT_FILE")
echo ">> Prompt: $PROMPT_FILE ($PROMPT_BYTES bytes)"

mkdir -p "$HSW_DIR"

# --- Run 1: --high-speed-swap on ---
echo ">> Run 1: with --high-speed-swap"
HSW_LOG=/tmp/atlas-hsw-with.log
"$SPARK" serve "$MODEL" \
    --port "$PORT_HSW" \
    --max-seq-len 32768 \
    --high-speed-swap \
    --high-speed-swap-dir "$HSW_DIR" \
    --high-speed-swap-gb 64 \
    --high-speed-swap-resident-blocks 8192 \
    --high-speed-swap-cache-blocks-per-seq 64 \
    --high-speed-swap-rank 32 \
    --high-speed-swap-qd 8 \
    > "$HSW_LOG" 2>&1 &
HSW_PID=$!
trap "kill $HSW_PID 2>/dev/null || true" EXIT

# Wait for server to come up (look for "ready" in log).
for i in $(seq 1 120); do
  if grep -q "Listening on" "$HSW_LOG" 2>/dev/null; then
    break
  fi
  if ! kill -0 $HSW_PID 2>/dev/null; then
    echo "ERROR: HSS server crashed during startup. Tail of log:" >&2
    tail -30 "$HSW_LOG" >&2
    exit 2
  fi
  sleep 1
done

# Sanity-check the startup logs for HSS-specific markers.
echo ">> Validating HSS startup banners..."
for marker in \
    "--high-speed-swap config staged" \
    "HBM cache shrunk to" \
    "BF16-KV"; do
  if ! grep -q "$marker" "$HSW_LOG"; then
    echo "  MISSING marker: '$marker'" >&2
    grep -i 'high.speed.swap\|hss\|kv cache' "$HSW_LOG" | head -10 >&2
    exit 3
  fi
  echo "  ✓ $marker"
done

# Send the prompt.
RESPONSE_HSW=$(curl -s -X POST "http://localhost:$PORT_HSW/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(jq -Rs '{model:"local",max_tokens:64,temperature:0,messages:[{role:"user",content:.}]}' < "$PROMPT_FILE")" \
    | jq -r '.choices[0].message.content // .error.message // "no response"')
echo ">> HSS response: ${RESPONSE_HSW:0:200}"
kill $HSW_PID 2>/dev/null || true
wait $HSW_PID 2>/dev/null || true
sleep 5

# --- Run 2: --high-speed-swap off (baseline) ---
echo ">> Run 2: baseline (no HSS)"
REF_LOG=/tmp/atlas-hsw-without.log
"$SPARK" serve "$MODEL" \
    --port "$PORT_REF" \
    --max-seq-len 32768 \
    > "$REF_LOG" 2>&1 &
REF_PID=$!
trap "kill $REF_PID 2>/dev/null || true" EXIT

for i in $(seq 1 120); do
  if grep -q "Listening on" "$REF_LOG" 2>/dev/null; then
    break
  fi
  if ! kill -0 $REF_PID 2>/dev/null; then
    echo "ERROR: baseline server crashed. Tail of log:" >&2
    tail -30 "$REF_LOG" >&2
    exit 4
  fi
  sleep 1
done

RESPONSE_REF=$(curl -s -X POST "http://localhost:$PORT_REF/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(jq -Rs '{model:"local",max_tokens:64,temperature:0,messages:[{role:"user",content:.}]}' < "$PROMPT_FILE")" \
    | jq -r '.choices[0].message.content // .error.message // "no response"')
echo ">> Baseline response: ${RESPONSE_REF:0:200}"
kill $REF_PID 2>/dev/null || true
wait $REF_PID 2>/dev/null || true

# --- Compare ---
echo ""
echo "===================="
if [[ "$RESPONSE_HSW" == "$RESPONSE_REF" ]]; then
  echo "PASS: HSS and baseline responses are byte-identical."
  exit 0
else
  echo "DIFFERENT outputs."
  echo "HSS    : ${RESPONSE_HSW:0:300}"
  echo "Baseline: ${RESPONSE_REF:0:300}"
  echo ""
  echo "Note: at temperature=0 outputs SHOULD match. Differences indicate either:"
  echo "  (a) the HSS read path has a correctness bug (most likely the per-layer"
  echo "      offload tracking — check disk_last_offloaded_per_layer is incrementing"
  echo "      for every attention layer in the spark log), or"
  echo "  (b) the model uses non-BF16 KV (--high-speed-swap silently falls back to"
  echo "      production paged for non-BF16 layers — see the 'mixed-precision KV'"
  echo "      warning at HSS startup)."
  exit 5
fi
