#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Build, start, and smoke-test GLM-4.7-Flash-NVFP4.
# Starts a temporary Atlas instance, runs the test, then shuts it down.
#
# Usage: ./smoke-glm.sh [--diag]
#   --diag   Enable per-layer GLM prefill diagnostics (ATLAS_DIAG_GLM=1)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPARK_BIN="${SCRIPT_DIR}/target/release/spark"
MODEL_ID="GadflyII/GLM-4.7-Flash-NVFP4"
ATLAS_PORT="${ATLAS_PORT:-9999}"
DIAG_MODE=0

for arg in "$@"; do
    [[ "$arg" == "--diag" ]] && DIAG_MODE=1
done

source "$HOME/.cargo/env" 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"

SPARK_PID=""
cleanup() {
    [[ -n "$SPARK_PID" ]] && kill "$SPARK_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── 1. Build ──────────────────────────────────────────────────────────────────
echo "🔨 Building spark (GLM target)..."
ATLAS_TARGET_MODEL=glm-4.7-flash \
    cargo build --release -p spark-server

# ── 2. Start ──────────────────────────────────────────────────────────────────
echo "🚀 Starting Atlas on port $ATLAS_PORT..."
if [[ "$DIAG_MODE" == "1" ]]; then
    export ATLAS_DIAG_GLM=1
    echo "🔬 Diagnostics: ENABLED (ATLAS_DIAG_GLM=1 --profile)"
    PROFILE_FLAG="--profile"
else
    PROFILE_FLAG=""
fi
"$SPARK_BIN" serve "$MODEL_ID" \
    --port "$ATLAS_PORT" \
    --max-seq-len 60000 \
    --kv-cache-dtype bf16 \
    --max-batch-size 1 \
    --gpu-memory-utilization 0.45 \
    --scheduling-policy slai \
    $PROFILE_FLAG \
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

# ── 3. Smoke test ─────────────────────────────────────────────────────────────
echo ""
echo "🧪 Running smoke test (2+2)..."
curl -s http://localhost:"$ATLAS_PORT"/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "glm4_moe_lite",
      "messages": [{"role": "user", "content": "What is 2+2?"}],
      "max_tokens": 20,
      "temperature": 0.01
    }' | python3 -c "
import sys, json
r = json.load(sys.stdin)
content = r['choices'][0]['message']['content']
print('Response:', content)
ok = any(x in content for x in ['4', 'four', 'Four'])
print('✅ PASS' if ok else '❌ FAIL — unexpected output')
sys.exit(0 if ok else 1)
"
