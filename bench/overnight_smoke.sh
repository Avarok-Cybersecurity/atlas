#!/usr/bin/env bash
# Overnight smoke test for the atlas-gb10:overnight2 image.
#
# Boots Qwen3.5-35B-A3B-FP8 (Qwen3.6 not in local HF cache), waits for
# readiness, runs bench/qwen36_ttft.py to capture TTFT + decode TPS, then
# stops the container. Output goes to bench/qwen36_ttft_<tag>.json and
# one-line summary to stdout.
#
# Usage:
#   bench/overnight_smoke.sh overnight2-baseline
#   bench/overnight_smoke.sh iteration-N [--baseline bench/qwen36_ttft_overnight2-baseline.json]

set -euo pipefail

TAG="${1:-$(date +%Y%m%d-%H%M)}"
shift || true
IMAGE="${ATLAS_IMAGE:-atlas-gb10:overnight2}"
MODEL="${ATLAS_MODEL:-Qwen/Qwen3.5-35B-A3B-FP8}"
PORT="${ATLAS_PORT:-8888}"
CONTAINER="atlas-smoke"
READY_TIMEOUT="${READY_TIMEOUT:-600}"

cd "$(dirname "$0")/.."

# Clean any prior container.
sudo docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

echo "[smoke] booting $CONTAINER from $IMAGE ..."
sudo docker run -d --name "$CONTAINER" \
  --gpus all --ipc=host --network host \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  "$IMAGE" serve \
    --scheduling-policy slai \
    --max-seq-len 32768 \
    --max-batch-size 1 \
    --tool-call-parser qwen3_coder \
    --kv-cache-dtype fp8 \
    --kv-high-precision-layers auto \
    --gpu-memory-utilization 0.85 \
    --port "$PORT" \
    "$MODEL" >/dev/null

echo "[smoke] waiting up to ${READY_TIMEOUT}s for /v1/models ..."
deadline=$(( $(date +%s) + READY_TIMEOUT ))
while (( $(date +%s) < deadline )); do
    if curl -fs --max-time 2 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "[smoke] ready"
        break
    fi
    # Early-fail on known startup errors.
    if sudo docker logs "$CONTAINER" 2>&1 | grep -qE "Error: Failed|terminate called|panicked at"; then
        echo "[smoke] STARTUP FAILED — dumping last 40 log lines"
        sudo docker logs "$CONTAINER" 2>&1 | tail -40
        sudo docker rm -f "$CONTAINER" >/dev/null
        exit 2
    fi
    sleep 2
done

if ! curl -fs --max-time 2 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[smoke] TIMEOUT waiting for ready"
    sudo docker logs "$CONTAINER" 2>&1 | tail -30
    sudo docker rm -f "$CONTAINER" >/dev/null
    exit 3
fi

echo "[smoke] running benchmark with tag=$TAG"
python3 bench/qwen36_ttft.py --url "http://localhost:${PORT}" --model "$MODEL" --tag "$TAG" "$@"
rc=$?

echo "[smoke] tearing down $CONTAINER"
sudo docker stop "$CONTAINER" >/dev/null
sudo docker rm "$CONTAINER" >/dev/null
exit "$rc"
