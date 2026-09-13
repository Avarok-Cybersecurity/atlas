#!/usr/bin/env bash
# Dual-Spark TP=2 for the K3 0.40B twin. IPs from env — not in git.
#
#   HEAD_IP=... WORKER_IP=... MASTER_ADDR=... bash docs/k3/scripts/start-k3-tp2.sh
#
# Pin RoCE: NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_HCA=rocep1s0f1
# (enp1s0f0np0 is Down on this lab.)
set -euo pipefail

BIN="${BIN:-/home/pidtom/k3-lab/bin/spark-k3}"
MODEL="${MODEL:-/home/pidtom/k3-lab/refs/Kimi-K3-0.40B}"
HEAD_IP="${HEAD_IP:?set HEAD_IP}"
WORKER_IP="${WORKER_IP:?set WORKER_IP}"
MASTER_ADDR="${MASTER_ADDR:-$HEAD_IP}"
MASTER_PORT="${MASTER_PORT:-29500}"
PORT="${PORT:-8888}"

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_PROTO="${NCCL_PROTO:-Simple}"
export NCCL_ALGO="${NCCL_ALGO:-Ring}"

echo "TP=2 head=$HEAD_IP worker=$WORKER_IP master=$MASTER_ADDR iface=$NCCL_SOCKET_IFNAME"

ssh "$HEAD_IP" "pkill -x spark-k3 2>/dev/null || true"
ssh "$WORKER_IP" "pkill -x spark-k3 2>/dev/null || true"
sleep 2

# Rank 1 first so it waits for NCCL.
ssh "$WORKER_IP" "export CUDA_HOME=/usr/local/cuda-13.0
export NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME NCCL_IB_HCA=$NCCL_IB_HCA
export NCCL_NVLS_ENABLE=0 NCCL_PROTO=Simple NCCL_ALGO=Ring
nohup $BIN serve --model-from-path $MODEL --model-name kimi-k3-0.40b \
  --world-size 2 --tp-size 2 --rank 1 \
  --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
  --bind 127.0.0.1 --port 8889 \
  --gpu-memory-utilization 0.5 --max-num-seqs 2 --max-seq-len 2048 \
  --disable-thinking \
  > /tmp/k3-tp2-r1.log 2>&1 & echo worker_pid=\$!"

sleep 2

ssh "$HEAD_IP" "export CUDA_HOME=/usr/local/cuda-13.0
export NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME NCCL_IB_HCA=$NCCL_IB_HCA
export NCCL_NVLS_ENABLE=0 NCCL_PROTO=Simple NCCL_ALGO=Ring
nohup $BIN serve --model-from-path $MODEL --model-name kimi-k3-0.40b \
  --world-size 2 --tp-size 2 --rank 0 \
  --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
  --bind 0.0.0.0 --port $PORT \
  --gpu-memory-utilization 0.5 --max-num-seqs 2 --max-seq-len 2048 \
  --disable-thinking \
  > /tmp/k3-tp2-r0.log 2>&1 & echo head_pid=\$!"

echo "logs: head /tmp/k3-tp2-r0.log worker /tmp/k3-tp2-r1.log"
