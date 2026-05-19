#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Start Sehyo/Qwen3.5-35B-A3B-Turbo8 + LiteLLM proxy
# — thinking mode DISABLED globally
# — MTP speculative K=2 (num-drafts 1)
# — Turbo8 quantization for KV cache and MTP (FP8-level with outlier suppression)
#
# Usage: ./start-qwen35-turbo8.sh
#
# Architecture:
#   Copilot / VS Code → LiteLLM (11111) → Atlas spark (9999)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUNCH_MODEL_DIR="/home/sna/ai-projects/lunch-model"
SPARK_BIN="${SCRIPT_DIR}/target/release/spark"

MODEL_ID="RedHatAI/Qwen3.6-35B-A3B-NVFP4"
ATLAS_PORT="${ATLAS_PORT:-9999}"
LITELLM_PORT="${LITE_LLM_PROXY_PORT:-11111}"

# ── sanity checks ─────────────────────────────────────────────────────────────
if [[ ! -x "$SPARK_BIN" ]]; then
    echo "❌ Binary not found: $SPARK_BIN"
    echo "   Build first: ATLAS_TARGET_MODEL=qwen3.6-35b-a3b cargo build --release -p spark-server"
    exit 1
fi

if [[ ! -d "$LUNCH_MODEL_DIR" ]]; then
    echo "❌ lunch-model not found at $LUNCH_MODEL_DIR"
    exit 1
fi

source "$HOME/.cargo/env" 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"

# ── cleanup on exit ───────────────────────────────────────────────────────────
SPARK_PID=""
LITELLM_PID=""
TAIL_PID=""

cleanup() {
    echo ""
    echo "🛑 Stopping Atlas and LiteLLM..."
    [[ -n "$TAIL_PID" ]]    && kill "$TAIL_PID"    2>/dev/null || true
    [[ -n "$LITELLM_PID" ]] && kill "$LITELLM_PID" 2>/dev/null || true
    [[ -n "$SPARK_PID" ]]   && kill "$SPARK_PID"   2>/dev/null || true
    echo "✅ Stopped."
    exit 0
}
trap cleanup EXIT INT TERM

# ── start Atlas spark ─────────────────────────────────────────────────────────
echo "🚀 Starting Atlas spark on port $ATLAS_PORT..."
echo "   Model:      $MODEL_ID"
echo "   Thinking:   DISABLED"
echo "   Speculative: K=2 (num-drafts 1, turbo8)"

"$SPARK_BIN" serve "$MODEL_ID" \
    --port "$ATLAS_PORT" \
    --max-seq-len 60000 \
    --kv-cache-dtype turbo8 \
    --kv-high-precision-layers auto \
    --gpu-memory-utilization 0.45 \
    --scheduling-policy slai \
    --speculative \
    --num-drafts 1 \
    --mtp-quantization nvfp4 \
    --ssm-cache-slots 64 \
    --disable-thinking \
    --call-parser qwen3_coder \
    &
SPARK_PID=$!
echo "   Spark PID: $SPARK_PID"

# ── wait for Atlas to be ready ────────────────────────────────────────────────
echo "⏳ Waiting for Atlas to be ready..."
mkdir -p "${SCRIPT_DIR}/logs"
for i in $(seq 1 120); do
    if curl -sf "http://localhost:${ATLAS_PORT}/v1/models" >/dev/null 2>&1; then
        echo "✅ Atlas is up (${i}s)"
        break
    fi
    if ! kill -0 "$SPARK_PID" 2>/dev/null; then
        echo "❌ Spark exited unexpectedly."
        exit 1
    fi
    sleep 1
done

# ── start LiteLLM proxy ───────────────────────────────────────────────────────
echo "🚀 Starting LiteLLM proxy on port $LITELLM_PORT..."
cd "$LUNCH_MODEL_DIR"
ATLAS_PORT="$ATLAS_PORT" LITE_LLM_PROXY_PORT="$LITELLM_PORT" \
    venv/bin/python3 server_compress.py \
    &> "${SCRIPT_DIR}/logs/litellm.log" &
LITELLM_PID=$!
echo "   LiteLLM PID: $LITELLM_PID  (logs: ${SCRIPT_DIR}/logs/litellm.log)"

sleep 3
if ! kill -0 "$LITELLM_PID" 2>/dev/null; then
    echo "❌ LiteLLM failed to start. Check logs:"
    tail -20 "${SCRIPT_DIR}/logs/litellm.log"
    exit 1
fi

# ── done ──────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════"
echo "✅ Atlas stack running (no-think mode):"
echo "   Atlas spark:   http://localhost:${ATLAS_PORT}/v1"
echo "   LiteLLM proxy: http://localhost:${LITELLM_PORT}/v1"
echo ""
echo "   VS Code Copilot → http://localhost:${LITELLM_PORT}/v1"
echo "   Model: qwen3.5-35b-turbo8"
echo "   KV Cache: turbo8 (FP8-level with outlier suppression)"
echo "════════════════════════════════════════════"
echo "Ctrl-C to stop both services."
echo ""

tail -f "${SCRIPT_DIR}/logs/litellm.log" &
wait "$SPARK_PID"