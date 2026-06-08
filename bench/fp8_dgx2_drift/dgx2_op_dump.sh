#!/usr/bin/env bash
# Run atlas-gb10:op-drift on dgx2 with all op-dump env vars enabled.
# Fires the canonical 10382-token prompt via /v1/completions.

set -euo pipefail

CONTAINER=atlas-op-drift
PORT=8888
DUMP_DIR=/workspace/atlas-dumps/op_drift_atlas_dgx2
TOKENS_JSON=/workspace/atlas-mtp/bench/fp8_dgx2_drift/atlas_tokens_dgx2.json

# Cleanup any previous run
sudo docker stop "$CONTAINER" 2>/dev/null || true
sudo docker rm "$CONTAINER" 2>/dev/null || true
sudo rm -rf "$DUMP_DIR"
mkdir -p "$DUMP_DIR"

# ATLAS_GDN_DUMP_LAYERS uses SSM-relative indices (0..29 for A3B).
# Want ALL 30 SSM layers covered, so list 0..29.
SSM_LAYERS=$(seq -s, 0 29)

# ATLAS_OP_DUMP_LAYERS uses ABSOLUTE layer indices for full-attn (3,7,...39).
# Leaving unset means "all layers" — we want full-attn for full-attn-only ops
# but also the input/post norms for SSM layers via ATLAS_OP_DUMP itself.
# Currently the op_dump hooks only fire in qwen3_attention prefill paths,
# so this is effectively just the 10 full-attn layers anyway.
# (No filter — capture every layer that fires.)

sudo docker run -d \
  --name "$CONTAINER" \
  --network host \
  --gpus all \
  --ipc=host \
  -v /workspace/.cache/huggingface:/root/.cache/huggingface \
  -v "$DUMP_DIR":/dump \
  -e ATLAS_OP_DUMP=/dump \
  -e ATLAS_GDN_DUMP=/dump \
  -e ATLAS_GDN_DUMP_LAYERS="$SSM_LAYERS" \
  -e ATLAS_GDN_DUMP_N_SSM=30 \
  -e ATLAS_NEMO_DUMP=/dump \
  -e RUST_LOG=info \
  -e CUDA_VISIBLE_DEVICES=0 \
  atlas-gb10:op-drift \
  serve Qwen/Qwen3.6-35B-A3B-FP8 \
    --port "$PORT" \
    --bind 0.0.0.0 \
    --gpu-memory-utilization 0.85 \
    --max-seq-len 16384 \
    --max-batch-size 1

echo "Container started: $CONTAINER"
echo "Logs: sudo docker logs -f $CONTAINER"
echo "Dumps will appear in: $DUMP_DIR"
