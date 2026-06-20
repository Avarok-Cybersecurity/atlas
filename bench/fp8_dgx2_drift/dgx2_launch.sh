#!/usr/bin/env bash
# Launch Atlas FP8-native server on dgx2 with ATLAS_NEMO_DUMP enabled.
# Mirror the dgx1 atlas-qwen-final config minus the request-dump path.
set -euo pipefail

DUMP_DIR="/workspace/atlas-dumps/fp8native_dgx2"
mkdir -p "${DUMP_DIR}"

# Match dgx1 atlas-qwen-final args: max-seq-len 65536, max-batch 4, gpu-mem 0.88,
# kv fp8, slai, prefix-caching ON.
sudo docker rm -f atlas-dgx2-dump 2>/dev/null || true

sudo docker run -d \
  --name atlas-dgx2-dump \
  --network host \
  --gpus all \
  --ipc=host \
  --security-opt label=disable \
  --runtime nvidia \
  -e RUST_LOG=info \
  -e ATLAS_NEMO_DUMP="${DUMP_DIR}" \
  -e ATLAS_DFLASH_DEBUG_DUMP_FULL=1 \
  -v "/workspace/.cache/huggingface:/root/.cache/huggingface" \
  -v "/workspace:/workspace" \
  atlas-gb10:fp8-much-better \
  serve Qwen/Qwen3.6-35B-A3B-FP8 \
    --port 8888 \
    --bind 0.0.0.0 \
    --max-seq-len 65536 \
    --max-batch-size 4 \
    --gpu-memory-utilization 0.88 \
    --kv-cache-dtype fp8 \
    --scheduling-policy slai \
    --enable-prefix-caching

echo "started; tail logs with: sudo docker logs -f atlas-dgx2-dump"
