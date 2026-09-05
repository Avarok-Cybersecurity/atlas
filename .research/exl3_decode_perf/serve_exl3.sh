#!/bin/bash
# EXL3 4.05bpw native serve for decode profiling. $1 = port, $2 = PROF (0/1), rest = extra args.
set -u
PORT=${1:-8890}; PROF=${2:-0}; shift 2 || true
cd /home/ms/atlas
export ATLAS_EXL3_NATIVE=1 ATLAS_EXL3_NATIVE_MOE=1 ATLAS_EXL3_NATIVE_DENSE=1
export ATLAS_PLE_MAX_TOKENS=9216 ATLAS_PLE_CACHE_SLOTS=4194304 ATLAS_QSA_MAX_TOKENS=32768
export ATLAS_INTHINK_TOOL_LEAK_OPENERS=0 ATLAS_NO_HW_PRECHECK=1
export RUST_LOG=${RUST_LOG:-info}
[ "$PROF" = "1" ] && export ATLAS_QWEN4EXP_DECODE_PROF=1
exec ./target/release/spark serve \
  --model-from-path /tank/exl3-ckpt/qwen38-flash-next-4.05bpw \
  --model-name qwen3.8-flash-next --kernel-target qwen3.8-flash-next \
  --world-size 1 --bind 127.0.0.1 --port $PORT \
  --max-seq-len 32768 --max-num-seqs 1 --max-batch-size 1 \
  --gpu-memory-utilization 0.72 --kv-cache-dtype bf16 --ssm-cache-slots 16 \
  --request-timeout 1800 --fast-load-prefetch-shards \
  --default-chat-template-kwargs '{"reasoning_effort":"low"}' "$@"
