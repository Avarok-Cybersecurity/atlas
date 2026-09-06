#!/bin/bash
# The qwen3.8-flash-next-exl3 preset spelled out explicitly WITHOUT speculation (the preset's bool
# cannot be negated from the CLI). Everything else identical: 4 seqs, 128K, util 0.72, prefix cache,
# preserve_thinking, EXL3 native gates, caps. $1 = port.
set -u
PORT=${1:-8899}
BIN=${SPARK_BIN:-/home/ms/atlas/.claude/worktrees/exl3-research/target/release/spark}
cd /home/ms/atlas/.claude/worktrees/exl3-research
export ATLAS_EXL3_NATIVE=1 ATLAS_EXL3_NATIVE_MOE=1 ATLAS_EXL3_NATIVE_DENSE=1
export ATLAS_PLE_MAX_TOKENS=9216 ATLAS_PLE_CACHE_SLOTS=4194304 ATLAS_QSA_MAX_TOKENS=131072
export ATLAS_INTHINK_TOOL_LEAK_OPENERS=0 ATLAS_NO_HW_PRECHECK=1
export RUST_LOG=${RUST_LOG:-info}
export LD_LIBRARY_PATH=/home/ms/nccl/build/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
exec "$BIN" serve \
  --model-from-path /tank/exl3-ckpt/qwen38-flash-next-4.05bpw \
  --model-name qwen3.8-flash-next --kernel-target qwen3.8-flash-next \
  --world-size 1 --bind 127.0.0.1 --port "$PORT" \
  --max-seq-len 131072 --max-num-seqs 4 --max-batch-size 4 \
  --gpu-memory-utilization 0.72 --kv-cache-dtype bf16 --ssm-cache-slots 64 \
  --fast-load-prefetch-shards --enable-prefix-caching \
  --default-chat-template-kwargs '{"reasoning_effort":"low","preserve_thinking":true}'
