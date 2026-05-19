#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Build (optional) and launch GLM-4.7-Flash-NVFP4 for interactive use.
# No smoke test — server stays up until Ctrl-C.
#
# Usage:
#   ./launch-glm.sh          # start only (binary must exist)
#   ./launch-glm.sh --build  # build first, then start

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPARK_BIN="${SCRIPT_DIR}/target/release/spark"
MODEL_ID="GadflyII/GLM-4.7-Flash-NVFP4"
ATLAS_PORT="${ATLAS_PORT:-9999}"

source "$HOME/.cargo/env" 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"

SPARK_PID=""
cleanup() {
    echo ""
    echo "🛑 Stopping Atlas..."
    [[ -n "$SPARK_PID" ]] && kill "$SPARK_PID" 2>/dev/null || true
    echo "✅ Stopped."
    exit 0
}
trap cleanup EXIT INT TERM

# ── optional build ─────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--build" ]]; then
    echo "🔨 Building spark (GLM target)..."
    ATLAS_TARGET_MODEL=glm-4.7-flash-a3b \
        cargo build --release -p spark-server
elif [[ ! -x "$SPARK_BIN" ]]; then
    echo "❌ Binary not found: $SPARK_BIN"
    echo "   Run with --build to compile first:"
    echo "   ./launch-glm.sh --build"
    exit 1
fi

# ── start Atlas ────────────────────────────────────────────────────────────────
echo "🚀 Starting Atlas on port $ATLAS_PORT..."
echo "   Model: $MODEL_ID"
"$SPARK_BIN" serve "$MODEL_ID" \
    --port "$ATLAS_PORT" \
    --max-seq-len 60000 \
    --kv-cache-dtype bf16 \
    --max-batch-size 1 \
    --gpu-memory-utilization 0.45 \
    --scheduling-policy slai \
    &
SPARK_PID=$!

echo "⏳ Waiting for Atlas to be ready..."
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

echo ""
echo "════════════════════════════════════════════"
echo "   Atlas (GLM-4.7-Flash): http://localhost:${ATLAS_PORT}/v1"
echo "   Ctrl-C to stop."
echo "════════════════════════════════════════════"
echo ""

wait "$SPARK_PID"
