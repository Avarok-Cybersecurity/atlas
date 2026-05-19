#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Atlas — start spark serve + LiteLLM proxy together
#
# Usage:
#   ./start.sh                                    # Qwen3.6-35B (default)
#   ./start.sh Sehyo/Qwen3.5-35B-A3B-NVFP4       # Qwen3.5-35B
#   ./start.sh RedHatAI/Qwen3.6-35B-A3B-NVFP4    # explicit
#   ./start.sh --port 9998 <MODEL_ID>             # custom port
#
# Architecture:
#   Copilot / VS Code → LiteLLM (11111) → Atlas spark (9999)
#
# Stops both services when you Ctrl-C.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUNCH_MODEL_DIR="/home/sna/ai-projects/lunch-model"
SPARK_BIN="${SCRIPT_DIR}/target/release/spark"

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL="RedHatAI/Qwen3.6-35B-A3B-NVFP4"
ATLAS_PORT="${ATLAS_PORT:-9999}"
LITELLM_PORT="${LITE_LLM_PROXY_PORT:-11111}"

# ── parse args ────────────────────────────────────────────────────────────────
MODEL_ID="$DEFAULT_MODEL"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) ATLAS_PORT="$2"; shift 2 ;;
        *) MODEL_ID="$1"; shift ;;
    esac
done

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

# ── make sure Cargo/CUDA env is loaded ───────────────────────────────────────
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
echo "   Model: $MODEL_ID"

"$SPARK_BIN" serve "$MODEL_ID" \
    --port "$ATLAS_PORT" \
    --max-seq-len 60000 \
    --kv-cache-dtype nvfp4 \
    --kv-high-precision-layers auto \
    --gpu-memory-utilization 0.45 \
    --scheduling-policy slai \
    --tool-call-parser qwen3_coder \
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

# ── update LiteLLM config to point at this Atlas port ─────────────────────────
export ATLAS_PORT
export LITE_LLM_PROXY_PORT="$LITELLM_PORT"

# ── start LiteLLM proxy ───────────────────────────────────────────────────────
echo "🚀 Starting LiteLLM proxy on port $LITELLM_PORT..."
cd "$LUNCH_MODEL_DIR"
VENV_PYTHON="venv/bin/python3"
ATLAS_PORT="$ATLAS_PORT" "$VENV_PYTHON" server_compress.py \
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
echo "✅ Atlas stack running:"
echo "   Atlas spark:   http://localhost:${ATLAS_PORT}/v1"
echo "   LiteLLM proxy: http://localhost:${LITELLM_PORT}/v1"
echo ""
echo "   VS Code Copilot → http://localhost:${LITELLM_PORT}/v1"
echo "   Models: qwen3.6-35b-nvfp4 · qwen36flash · qwen36-think · qwen3.5-35b-nvfp4"
echo "════════════════════════════════════════════"
echo "Ctrl-C to stop both services."
echo ""

# Atlas logs stream directly to this terminal.
# LiteLLM is silent unless it errors — tail its log for visibility.
tail -f "${SCRIPT_DIR}/logs/litellm.log" &
TAIL_PID=$!
wait "$SPARK_PID"
