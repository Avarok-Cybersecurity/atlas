#!/usr/bin/env bash
# DeepSeek-V4-Flash EP=2 bring-up on 2x GB10 (pure EP, no TP).
#
# DeepSeek-V4 uses num_key_value_heads=1 (MQA), making TP>1 impossible.
# Multi-spark deployments MUST use pure EP (tp-size 1, ep-size 2).
#
# Usage:
#   ./scripts/start-deepseek-ep2.sh [MODEL]
#
# Default model: deepseek-ai/DeepSeek-V4-Flash
#
# Prerequisites:
#   - Two GB10 nodes connected via RoCE (enp1s0f0np0), MTU 9000
#   - Passwordless SSH from head (HEAD_IP env) to worker (WORKER_IP env)
#   - atlas-deepseek-v4:latest Docker image on both nodes
#     Build: docker build -f docker/gb10/deepseek-v4-flash/nvfp4/Dockerfile -t atlas-deepseek-v4 .
#   - Same image tag on BOTH nodes (mixing Atlas versions across ranks
#     causes NCCL to hang at ncclCommInitRank).
#   - Model weights cached on both nodes (~/.cache/huggingface)
#   - RDMA kernel support on host (IB device at /dev/infiniband)

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-}"
MODEL_NAME="${MODEL_NAME:-nvidia/DeepSeek-V4-Flash-NVFP4}"
# Host model directory name (under the HF hub cache on each node). The nvidia
# NVFP4 checkpoint (ships an MTP module) lives in `v4-nvfp4-mtp`.
MODEL_DIR="${MODEL_DIR:-v4-nvfp4-mtp}"
# Explicit per-node host paths (avoids the $HOME footgun: $HOME resolves to
# /workspace in some shells, whose cache dir is empty).
MODEL_HOST_HEAD="${MODEL_HOST_HEAD:-/home/azeez/.cache/huggingface/hub/${MODEL_DIR}}"
MODEL_HOST_WORKER="${MODEL_HOST_WORKER:-/raid/hf-cache/hub/${MODEL_DIR}}"
IMAGE="${IMAGE:-atlas-deepseek-v4:latest}"
HEAD_IP="${HEAD_IP:-127.0.0.1}"
WORKER_IP="${WORKER_IP:-127.0.0.1}"
MASTER_PORT="29500"
PORT="8888"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}"
KV_DTYPE="${KV_DTYPE:-fp8}"
TP_SIZE="${TP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-2}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

# Resolve model path defaults per node if not explicitly set.
if [[ -z "$MODEL_PATH" ]]; then
  if [[ "$HEAD_IP" == "127.0.0.1" || "$HEAD_IP" == "localhost" ]]; then
    MODEL_PATH="/root/.cache/huggingface/redhat-ds-v4"
  else
    MODEL_PATH="/model"
  fi
fi

# DeepSeek-V4 has no MTP weights in the public checkpoint, so
# --speculative is OFF on both ranks.

echo "=== Atlas DeepSeek-V4-Flash EP=2 bring-up (RDMA-enabled) ==="
echo "Model path: $MODEL_PATH"
echo "Model name: $MODEL_NAME"
echo "Image:   $IMAGE"
echo "Head:    $HEAD_IP (rank 0, HTTP on $PORT)"
echo "Worker:  $WORKER_IP (rank 1)"
echo "GPU mem: $GPU_MEM_UTIL    Max seq: $MAX_SEQ_LEN"
echo "Topology: TP=$TP_SIZE EP=$EP_SIZE"
echo ""

# Clean old containers
echo "Cleaning up old containers..."
sudo docker rm -f atlas-ds-ep0 2>/dev/null || true
ssh "$WORKER_IP" "sudo docker rm -f atlas-ds-ep1 2>/dev/null || true"

RDMA_FLAGS="--device=/dev/infiniband --cap-add=IPC_LOCK --cap-add=SYS_NICE --ulimit memlock=-1 --security-opt seccomp=unconfined"

NCCL_ENV="\
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f0 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 \
  -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_TIMEOUT=22 \
  -e NCCL_IB_RETRY_CNT=7 \
  -e NCCL_NET_GDR_LEVEL=0 \
  -e NCCL_NET_GDR_C2C=0 \
  -e NCCL_DMABUF_ENABLE=0 \
  -e NCCL_NVLS_ENABLE=0 \
  -e NCCL_CUMEM_HOST_ENABLE=0 \
  -e NCCL_PROTO=Simple \
  -e NCCL_ALGO=Ring \
  -e NCCL_BUFFSIZE=33554432 \
  -e NCCL_MIN_NCHANNELS=1 \
  -e NCCL_MAX_NCHANNELS=2 \
  -e NCCL_DEBUG=WARN \
  -e NCCL_DEBUG_SUBSYS=INIT,NET"

# Volume mounts for model weights (host paths differ per node).
# DGX1 (head):   /home/azeez/.cache/huggingface/hub/<MODEL_DIR>
# DGX2 (worker): /raid/hf-cache/hub/<MODEL_DIR>
VOL_HEAD="-v ${MODEL_HOST_HEAD}:/model"
VOL_WORKER="-v ${MODEL_HOST_WORKER}:/model"

# Start rank 0 (head) — HTTP server + scheduler
echo "Starting rank 0 on $HEAD_IP..."
sudo docker run -d \
  --name atlas-ds-ep0 \
  --gpus all \
  --ipc=host \
  --network host \
  $RDMA_FLAGS \
  $NCCL_ENV \
  -e RUST_LOG=info \
  -e ATLAS_DIAG_V4_ALL_LAYERS=${ATLAS_DIAG:-0} \
  -e ATLAS_EP_GRAPHS=${ATLAS_EP_GRAPHS:-0} \
  $VOL_HEAD \
  "$IMAGE" serve \
    --model-from-path "$MODEL_PATH" \
    --model-name "$MODEL_NAME" \
    --rank 0 \
    --world-size 2 \
    --tp-size "$TP_SIZE" \
    --ep-size "$EP_SIZE" \
    --master-addr "$HEAD_IP" \
    --master-port "$MASTER_PORT" \
    --port "$PORT" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --max-batch-size 1 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --kv-cache-dtype "$KV_DTYPE" \
    --enable-prefix-caching \
    $EXTRA_FLAGS \
    --oom-guard-mb 512

# Start rank 1 (worker)
echo "Starting rank 1 on $WORKER_IP..."
ssh "$WORKER_IP" "sudo docker run -d \
  --name atlas-ds-ep1 \
  --gpus all \
  --ipc=host \
  --network host \
  $RDMA_FLAGS \
  $NCCL_ENV \
  -e RUST_LOG=info \
  -e ATLAS_DIAG_V4_ALL_LAYERS=${ATLAS_DIAG:-0} \
  -e ATLAS_EP_GRAPHS=${ATLAS_EP_GRAPHS:-0} \
  $VOL_WORKER \
  $IMAGE serve \
    --model-from-path $MODEL_PATH \
    --model-name $MODEL_NAME \
    --rank 1 \
    --world-size 2 \
    --tp-size $TP_SIZE \
    --ep-size $EP_SIZE \
    --master-addr $HEAD_IP \
    --master-port $MASTER_PORT \
    --port 0 \
    --max-seq-len $MAX_SEQ_LEN \
    --max-batch-size 1 \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --kv-cache-dtype $KV_DTYPE \
    --enable-prefix-caching \
    $EXTRA_FLAGS \
    --oom-guard-mb 512"

echo ""
echo "=== Both ranks starting ==="
echo "Monitor rank 0: sudo docker logs -f atlas-ds-ep0"
echo "Monitor rank 1: ssh $WORKER_IP 'sudo docker logs -f atlas-ds-ep1'"
echo "API endpoint:   http://$HEAD_IP:$PORT/v1/chat/completions"

echo ""
echo "=== Both ranks starting ==="
echo "Monitor rank 0: sudo docker logs -f atlas-ds-ep0"
echo "Monitor rank 1: ssh $WORKER_IP 'sudo docker logs -f atlas-ds-ep1'"
echo "API endpoint:   http://$HEAD_IP:$PORT/v1/chat/completions"
