#!/usr/bin/env bash
#
# Launch Atlas with Expert Parallelism (EP=2) across two GB10 nodes.
#
# Usage:
#   ./scripts/start-ep2.sh [MODEL]
#
# Default model: Sehyo/Qwen3.5-122B-A10B-NVFP4
#
# Prerequisites:
#   - Two GB10 nodes connected via RoCE (enp1s0f0np0), MTU 9000
#   - Passwordless SSH from head (HEAD_IP env) to worker (WORKER_IP env)
#   - atlas-122b:latest Docker image on both nodes (or atlas-gb10:latest)
#     Build: docker build -f docker/gb10/qwen3.5-122b-a10b/nvfp4/Dockerfile -t atlas-122b .
#   - Same image tag on BOTH nodes (mixing Atlas versions across ranks
#     causes NCCL to hang at ncclCommInitRank — see docs/EP2-TROUBLESHOOTING.md#4).
#   - Model weights cached on both nodes (~/.cache/huggingface)
#   - RDMA kernel support on host (IB device at /dev/infiniband)
#
# The head node (rank 0) runs the HTTP server + scheduler.
# The worker node (rank 1) mirrors model operations for EP all-reduce.
#
# Recommended GPU_MEM_UTIL for MiniMax / 122B-class models: 0.90.
# The default 0.70 is too tight for the MoE weight transpose (55-60 GB
# per rank) and causes a silent exit in `build_model`; see
# docs/EP2-TROUBLESHOOTING.md#5.
#
# For MiniMax M2.x checkpoints: do NOT pass --speculative. The loader's
# per-module MTP extraction is still a TODO; the Atlas pre-flight will
# bail with a clear error, but easier to drop the flag up-front.
# See docs/EP2-TROUBLESHOOTING.md#2.

set -euo pipefail

MODEL="${1:-Sehyo/Qwen3.5-122B-A10B-NVFP4}"
IMAGE="${IMAGE:-atlas-122b:latest}"
HEAD_IP="${HEAD_IP:-127.0.0.1}"
WORKER_IP="${WORKER_IP:-127.0.0.1}"
MASTER_PORT="29500"
PORT="8888"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.70}"
MTP_QUANT="${MTP_QUANT:-nvfp4}"
DEV_BINARY="${DEV_BINARY:-}"

echo "=== Atlas EP=2 Launch (RDMA-enabled) ==="
echo "Model:  $MODEL"
echo "Head:   $HEAD_IP (rank 0, HTTP on port $PORT)"
echo "Worker: $WORKER_IP (rank 1)"
echo "GPU mem utilization: $GPU_MEM_UTIL"
echo "MTP quantization:   $MTP_QUANT"
echo ""

# Stop any existing containers
echo "Cleaning up old containers..."
sudo docker rm -f atlas-ep0 2>/dev/null || true
ssh "$WORKER_IP" "sudo docker rm -f atlas-ep1 2>/dev/null || true"

# RDMA libraries are baked into the Docker image (libnccl2, libibverbs1,
# librdmacm1, ibverbs-providers, libnl). No host volume mounts needed.

# Optional dev binary mount (overrides container's /usr/local/bin/spark)
DEV_MOUNT=""
DEV_MOUNT_WORKER=""
if [ -n "$DEV_BINARY" ]; then
  DEV_MOUNT="-v $DEV_BINARY:/usr/local/bin/spark:ro"
  # For worker, copy binary first
  scp "$DEV_BINARY" "$WORKER_IP:/tmp/spark-dev" 2>/dev/null || true
  DEV_MOUNT_WORKER="-v /tmp/spark-dev:/usr/local/bin/spark:ro"
  echo "Using dev binary: $DEV_BINARY"
fi

# RDMA device and capability flags
RDMA_FLAGS="--device=/dev/infiniband --cap-add=IPC_LOCK --ulimit memlock=-1"

# NCCL environment for RoCE v2 transport over IPv4
# Key fixes (2026-04-06):
#   - Removed NCCL_IB_GID_INDEX=0 (selected RoCEv1 link-local, broke IPv4 routing)
#   - Added ROCE_VERSION_NUM=2 + ADDR_FAMILY=AF_INET (force RoCEv2/IPv4)
#   - Added GDR_C2C=0 + DMABUF_ENABLE=0 (nvidia_peermem broken on GB10 kernel)
#   - Added NVLS_ENABLE=0 (aarch64 Blackwell bug, NCCL #1769)
#   - Added IB_TIMEOUT=22 + IB_RETRY_CNT=7 (resilience for slow IB startup)
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
  -e NCCL_MIN_NCHANNELS=1 \
  -e NCCL_MAX_NCHANNELS=2 \
  -e NCCL_DEBUG=INFO \
  -e NCCL_DEBUG_SUBSYS=INIT,NET"

# Start rank 0 (head) — HTTP server + scheduler
echo "Starting rank 0 on $HEAD_IP..."
sudo docker run -d \
  --name atlas-ep0 \
  --gpus all \
  --ipc=host \
  --network host \
  $RDMA_FLAGS \
  $NCCL_ENV \
  -e RUST_LOG=info \
  $DEV_MOUNT \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  "$IMAGE" serve "$MODEL" \
    --rank 0 \
    --world-size 2 \
    --master-addr "$HEAD_IP" \
    --master-port "$MASTER_PORT" \
    --port "$PORT" \
    --max-batch-size 1 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --kv-cache-dtype nvfp4 \
    --enable-prefix-caching \
    --speculative \
    --mtp-quantization "$MTP_QUANT"

# Start rank 1 (worker) — EP worker loop only
echo "Starting rank 1 on $WORKER_IP..."
ssh "$WORKER_IP" "sudo docker run -d \
  --name atlas-ep1 \
  --gpus all \
  --ipc=host \
  --network host \
  $RDMA_FLAGS \
  $NCCL_ENV \
  -e RUST_LOG=info \
  $DEV_MOUNT_WORKER \
  -v \"\${HOME}/.cache/huggingface:/root/.cache/huggingface\" \
  $IMAGE serve $MODEL \
    --rank 1 \
    --world-size 2 \
    --master-addr $HEAD_IP \
    --master-port $MASTER_PORT \
    --port 0 \
    --max-batch-size 1 \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --kv-cache-dtype nvfp4 \
    --enable-prefix-caching \
    --speculative \
    --mtp-quantization "$MTP_QUANT""

echo ""
echo "=== Both ranks starting ==="
echo "Monitor rank 0: sudo docker logs -f atlas-ep0"
echo "Monitor rank 1: ssh $WORKER_IP 'sudo docker logs -f atlas-ep1'"
echo "API endpoint:   http://$HEAD_IP:$PORT/v1/chat/completions"
echo ""
echo "Wait for both ranks to complete NCCL init before sending requests."
echo "Look for 'NCCL initialized' and 'EP worker ready' in logs."
echo "Check for 'NET/IB' in NCCL_DEBUG output to confirm RDMA transport."
