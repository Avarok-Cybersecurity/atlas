#!/usr/bin/env bash
# Single-node MiniMax M2 bring-up on GB10.
#
# MiniMax doesn't ship MTP weights in the public checkpoint (confirmed on
# MiniMax-M2 and MiniMax-M2.7), so --speculative is OFF. --kv-cache-dtype
# bf16 for bring-up coherence; can flip to fp8 or nvfp4 once we're sure
# attention is numerically clean end-to-end.
set -euo pipefail

MODEL="${1:-MiniMaxAI/MiniMax-M2}"
IMAGE="${IMAGE:-atlas-gb10:m3e}"
PORT="${PORT:-8888}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-16384}"
CONTAINER="${CONTAINER:-atlas-minimax-bringup}"

echo "=== Atlas MiniMax single-node bring-up ==="
echo "Model:    $MODEL"
echo "Image:    $IMAGE"
echo "Port:     $PORT"
echo "GPU mem:  $GPU_MEM_UTIL"
echo "Max seq:  $MAX_SEQ_LEN"
echo ""

sudo docker rm -f "$CONTAINER" 2>/dev/null || true

sudo docker run -d \
  --name "$CONTAINER" \
  --gpus all \
  --ipc=host \
  --network host \
  -e RUST_LOG=info \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  "$IMAGE" serve "$MODEL" \
    --port "$PORT" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --max-batch-size 1 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --kv-cache-dtype bf16

echo "Container: $CONTAINER"
echo "Monitor:   sudo docker logs -f $CONTAINER"
echo "Health:    curl http://localhost:$PORT/v1/models"
echo "Test:      curl http://localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":40}'"
