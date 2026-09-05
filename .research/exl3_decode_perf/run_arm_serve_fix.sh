#!/bin/bash
# EXL3 4.05bpw native serve from the worktree-built binary (shared-expert GEMV arm).
# $1 = port. Pass ATLAS_EXL3_SHARED_PREFILL_GEMM=1 in the environment for the control arm.
set -u
PORT=${1:-8890}
BIN=${SPARK_BIN:-/home/ms/.claude/jobs/5a7bd33d/tmp/exl3bench/spark-sharedgemv}
cd /home/ms/atlas/.claude/worktrees/exl3-research
export ATLAS_EXL3_NATIVE=1 ATLAS_EXL3_NATIVE_MOE=1 ATLAS_EXL3_NATIVE_DENSE=1
export ATLAS_PLE_MAX_TOKENS=9216 ATLAS_PLE_CACHE_SLOTS=4194304 ATLAS_QSA_MAX_TOKENS=32768
export ATLAS_INTHINK_TOOL_LEAK_OPENERS=0 ATLAS_NO_HW_PRECHECK=1
export RUST_LOG=${RUST_LOG:-info}
export LD_LIBRARY_PATH=/home/ms/nccl/build/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
exec "$BIN" serve \
  --model-from-path /tank/exl3-ckpt/qwen38-flash-next-4.05bpw \
  --model-name qwen3.8-flash-next --kernel-target qwen3.8-flash-next \
  --world-size 1 --bind 127.0.0.1 --port $PORT \
  --max-seq-len 32768 --max-num-seqs 1 --max-batch-size 1 \
  --gpu-memory-utilization 0.72 --kv-cache-dtype bf16 --ssm-cache-slots 16 \
  --request-timeout 1800 --fast-load-prefetch-shards \
  --default-chat-template-kwargs '{"reasoning_effort":"low"}'
