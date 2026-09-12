#!/bin/bash
# NVFP4 (nvidia/Qwen3.8-Flash-Next-NVFP4) serve for the decode A/B control arm.
# Same engine flags as serve_exl3.sh except the checkpoint (and no EXL3 gates).
# $1 = port, $2 = PROF (0/1), rest = extra args.
set -u
PORT=${1:-8891}; PROF=${2:-0}; shift 2 || true
cd /home/ms/atlas
export ATLAS_PLE_MAX_TOKENS=9216 ATLAS_PLE_CACHE_SLOTS=4194304 ATLAS_QSA_MAX_TOKENS=32768
export ATLAS_INTHINK_TOOL_LEAK_OPENERS=0 ATLAS_NO_HW_PRECHECK=1
export RUST_LOG=${RUST_LOG:-info}
[ "$PROF" = "1" ] && export ATLAS_QWEN4EXP_DECODE_PROF=1
exec ./target/release/spark serve \
  --model-from-path /tank/hf/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fab0aecb760cec45227f6656abcaafa11abca87a \
  --model-name qwen3.8-flash-next --kernel-target qwen3.8-flash-next \
  --world-size 1 --bind 127.0.0.1 --port $PORT \
  --max-seq-len 32768 --max-num-seqs 1 --max-batch-size 1 \
  --gpu-memory-utilization 0.72 --kv-cache-dtype bf16 --ssm-cache-slots 16 \
  --request-timeout 1800 --fast-load-prefetch-shards \
  --default-chat-template-kwargs '{"reasoning_effort":"low"}' "$@"
