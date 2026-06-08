#!/usr/bin/env bash
# Phase 2b replay: feed the same 30-turn / 18920-token prompt to the RNE-image
# Atlas server and capture per-layer ATLAS_NEMO_DUMP at /workspace/atlas-dumps/numdrift/rne/.
#
# Prereqs:
#   - atlas-qwen container running on `atlas-gb10:fp8-dequant-rne` image with
#     `-e ATLAS_NEMO_DUMP=/workspace/atlas-dumps/numdrift/rne` env var
#   - localhost:8888 reachable
#
# Output:
#   /workspace/atlas-dumps/numdrift/rne/atlas_L{0..39}.bin
#   /workspace/atlas-dumps/numdrift/rne/atlas_final_norm.bin
#   /workspace/atlas-dumps/numdrift/rne/atlas_logits.bin
#
# After this completes, run cosine_three_way_phase2b.py for the verdict.

set -euo pipefail

PROBE=/workspace/atlas-dumps/numdrift/atlas_turn11_probe.json
OUT_DIR=/workspace/atlas-dumps/numdrift/rne

if [[ ! -f "$PROBE" ]]; then
    echo "ERROR: probe prompt not found at $PROBE" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "Sending probe request (18920-token prompt) at $(date)"
curl -s -X POST http://localhost:8888/v1/chat/completions \
    -H "Content-Type: application/json" \
    --data-binary @"$PROBE" \
    -o "$OUT_DIR/response.jsonl" \
    || { echo "request failed"; exit 1; }

echo "Server responded; checking ATLAS_NEMO_DUMP output at $OUT_DIR"
ls "$OUT_DIR"/atlas_L*.bin 2>&1 | wc -l
echo "If the count is 40, dump is complete."
echo "Generated tokens preview (first 200 chars of stream):"
head -c 200 "$OUT_DIR/response.jsonl"
echo
