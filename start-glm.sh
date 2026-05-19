#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Start GadflyII/GLM-4.7-Flash-NVFP4 + LiteLLM proxy via Atlas
# — thinking mode ENABLED
# — MTP speculative DISABLED (Phase 4 not yet implemented)
#
# Usage: ./start-glm.sh
#
# Build first:
#   ATLAS_TARGET_MODEL=glm-4.7-flash cargo build --release -p spark-server
#
# Architecture:
#   Copilot / VS Code → LiteLLM (11111) → Atlas spark (9999)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUNCH_MODEL_DIR="/home/sna/ai-projects/lunch-model"
SPARK_BIN="${SCRIPT_DIR}/target/release/spark"

MODEL_ID="GadflyII/GLM-4.7-Flash-NVFP4"
ATLAS_PORT="${ATLAS_PORT:-9999}"
LITELLM_PORT="${LITE_LLM_PROXY_PORT:-11111}"

# ── sanity checks ─────────────────────────────────────────────────────────────
if [[ ! -x "$SPARK_BIN" ]]; then
    echo "❌ Binary not found: $SPARK_BIN"
    echo "   Build first:"
    echo "   ATLAS_TARGET_MODEL=glm-4.7-flash cargo build --release -p spark-server"
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
echo "   Model:       $MODEL_ID"
echo "   Thinking:    ENABLED"
echo "   Speculative: DISABLED (MTP Phase 4 pending)"

"$SPARK_BIN" serve "$MODEL_ID" \
    --port "$ATLAS_PORT" \
    --max-seq-len 180000 \
    --kv-cache-dtype bf16 \
    --max-batch-size 1 \
    --gpu-memory-utilization 0.85 \
    --scheduling-policy slai \
    &
SPARK_PID=$!
echo "   Spark PID: $SPARK_PID"

# ── wait for Atlas to be ready ────────────────────────────────────────────────
echo "⏳ Waiting for Atlas to be ready..."
mkdir -p "${SCRIPT_DIR}/logs"
for i in $(seq 1 180); do
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
ATLAS_PORT="$ATLAS_PORT" \
    venv/bin/litellm \
        --config "${LUNCH_MODEL_DIR}/lite_llm_config_glm.yaml" \
        --port "$LITELLM_PORT" \
        --host 0.0.0.0 \
    &> "${SCRIPT_DIR}/logs/litellm-glm.log" &
LITELLM_PID=$!
echo "   LiteLLM PID: $LITELLM_PID  (logs: ${SCRIPT_DIR}/logs/litellm-glm.log)"

sleep 3
if ! kill -0 "$LITELLM_PID" 2>/dev/null; then
    echo "❌ LiteLLM failed to start. Check logs:"
    tail -20 "${SCRIPT_DIR}/logs/litellm-glm.log"
    exit 1
fi

# ── done ──────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════"
echo "✅ Atlas stack running (GLM-4.7-Flash):"
echo "   Atlas spark:   http://localhost:${ATLAS_PORT}/v1"
echo "   LiteLLM proxy: http://localhost:${LITELLM_PORT}/v1"
echo ""
echo "   VS Code Copilot → http://localhost:${LITELLM_PORT}/v1"
echo "   Model: glm-4.7-flash-nvfp4"
echo "════════════════════════════════════════════"
echo "Ctrl-C to stop both services."
echo ""

tail -f "${SCRIPT_DIR}/logs/litellm-glm.log" &
TAIL_PID=$!
wait "$SPARK_PID"
