#!/bin/bash
# GLM-5.3-Flash-NVFP4 on 2x GB10 — world=2, TP=2, EP=2 (OVERLAPPING groups).
#
# `serve_args` accepts either `world_size == tp_size * ep_size` (an orthogonal mesh) or
# `world_size == tp_size == ep_size` (overlapping groups on the SAME ranks). GLM is the
# second: both ranks are simultaneously a TP pair and an EP pair.
#
# 🔒 n3/n4 are OFF LIMITS — they run the frozen vLLM oracle (`vllm_glm53`), which is the
# only reference a first-forward bisect can compare against. This script never touches them.
#
# Topology decision (measured, do not re-derive):
#   TP=2 + EP=2   10,665 MB/rank/token   roofline 23.8 tok/s   88.713 GiB/rank resident
#   EP=2 only     18,952 MB/rank/token   roofline 13.4 tok/s   — 87 % replicated, rejected
#
# 🪤 `--device=/dev/infiniband --ulimit memlock=-1` is what makes RoCE work. Without it the
# image's libibverbs sees no uverbs nodes, NCCL logs `NET/IB : No device found.` and silently
# falls back to `NET/Socket` (measured 2026-08-28, first 2-node bring-up). Same flags as the
# known-good `start-ep2.sh` / `start-deepseek-ep2.sh` EP launchers.
#
# 🪤 VERIFY THE FABRIC BEFORE TRUSTING ANY NUMBER. grep rank 0's log for NET/IB dual-rail
# RoCE (`[0]rocep1s0f0 [1]roceP2p1s0f0`). `NET/Socket` means it silently fell back to
# TCP — STOP, do not benchmark, do not believe a tok/s figure taken that way.
set -euo pipefail

IMAGE="${IMAGE:-atlas-glm53:t3}"
MODEL_DIR="${MODEL_DIR:-/home/cluster/glm53-ckpt}"
NODES=(10.10.10.1 10.10.10.2)
MASTER=10.10.10.1
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
# Allocator budget, not a measurement. Weights alone are 88.713 GiB of ~121 GB (0.73);
# the rest is KV + activations + the mHC highway. Start conservative and raise only with
# a measured residency number.
GPU_UTIL="${GPU_UTIL:-0.90}"
# The fast loader's OOM pre-flight gate is `on-disk x multiplier + oom_guard <= free`.
# At 99.64 GB/rank the default 4 GB guard leaves the gate 1.7 GB short of a load that
# fits. 🪤 This shrinks the LOAD-TIME margin only; the OOM watchdog still runs.
OOM_GUARD_MB="${OOM_GUARD_MB:-1024}"
# Extra `-e K=V` flags, space separated. Used for profiling (ATLAS_GLM_PROFILE=1) and for
# NCCL A/Bs (NCCL_MAX_NCHANNELS=...). Empty by default so the serve path is unchanged.
EXTRA_ENV="${EXTRA_ENV:-}"

for RANK in 0 1; do
  IP=${NODES[$RANK]}
  PORT=$((8888 + RANK))
  NAME="atlas-glm53-r${RANK}"
  echo "=== rank $RANK on $IP (port $PORT) ==="
  ssh "cluster@$IP" "docker rm -f $NAME 2>/dev/null; \
    docker run -d --name $NAME \
      --network host --gpus all --ipc=host \
      --device=/dev/infiniband \
      --cap-add=IPC_LOCK --cap-add=SYS_NICE --ulimit memlock=-1 \
      --security-opt seccomp=unconfined \
      -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 \
      -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \
      -e NCCL_IB_DISABLE=0 \
      -e NCCL_DEBUG=INFO \
      -e CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0} \
      -e NCCL_NVLS_ENABLE=0 \
      -e RUST_LOG=info \
      $EXTRA_ENV \
      -v $MODEL_DIR:/model:ro \
      $IMAGE \
      serve --model-from-path /model \
      --port $PORT --rank $RANK --world-size 2 \
      --tp-size 2 --ep-size 2 \
      --master-addr $MASTER --master-port 29500 \
      --max-seq-len $MAX_SEQ_LEN --kv-cache-dtype fp8 \
      --gpu-memory-utilization $GPU_UTIL \
      --oom-guard-mb $OOM_GUARD_MB \
      --max-batch-size 1"
  [ "$RANK" -eq 0 ] && echo "waiting 10s for rank 0 to bind the master port..." && sleep 10
done

cat <<'EOF'
=== both ranks started ===

FABRIC CHECK (do this before anything else):
  ssh cluster@10.10.10.1 'docker logs atlas-glm53-r0 2>&1 | grep -E "NET/IB|NET/Socket"'
    NET/IB + both rails  -> good
    NET/Socket           -> STOP

PROGRESS:
  ssh cluster@10.10.10.1 'docker logs -f atlas-glm53-r0'

RESIDENCY (GB10 is unified memory — read the NODE, never docker stats):
  ssh cluster@10.10.10.1 'free -g | sed -n 2p'
EOF
