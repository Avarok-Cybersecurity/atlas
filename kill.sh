#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Kill Atlas spark and LiteLLM proxy.
# Usage: ./kill.sh

SPARK_BIN_PATH="/home/sna/ai-projects/atlas/target/release/spark"

killed=0

# ── Atlas spark ───────────────────────────────────────────────────────────────
SPARK_PIDS=$(pgrep -f "$SPARK_BIN_PATH" 2>/dev/null || true)
if [[ -n "$SPARK_PIDS" ]]; then
    echo "🛑 Stopping Atlas spark (PIDs: $SPARK_PIDS)..."
    kill $SPARK_PIDS 2>/dev/null || true
    killed=1
else
    echo "   Atlas spark: not running"
fi

# ── LiteLLM proxy (server_compress.py) ───────────────────────────────────────
LITELLM_PIDS=$(pgrep -f "server_compress.py" 2>/dev/null || true)
if [[ -n "$LITELLM_PIDS" ]]; then
    echo "🛑 Stopping LiteLLM proxy (PIDs: $LITELLM_PIDS)..."
    kill $LITELLM_PIDS 2>/dev/null || true
    killed=1
else
    echo "   LiteLLM proxy: not running"
fi

# ── wait for clean exit ───────────────────────────────────────────────────────
if [[ $killed -eq 1 ]]; then
    sleep 2
    echo "✅ Done."
fi
