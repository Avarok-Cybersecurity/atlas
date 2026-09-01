#!/bin/bash
# Single-node (dgx-00) boot: qwen4_exp 2.05bpw EXL3, routed experts served
# NATIVELY from packed trellis (ATLAS_EXL3_NATIVE_MOE=1).
set -uo pipefail
SNAP=/tank/exl3-ckpt/qwen38-flash-next-2.05bpw
CTX=8192
SEQS=1

export LD_LIBRARY_PATH="/home/ms/atlas-gdn-libs:/home/ms/nccl/build/lib:${LD_LIBRARY_PATH:-}"
export ATLAS_EXL3_NATIVE=1
export ATLAS_EXL3_NATIVE_MOE=1
export ATLAS_PLE_MAX_TOKENS=9000
export ATLAS_PLE_CACHE_SLOTS=4194304
export ATLAS_QSA_MAX_TOKENS=8192
export ATLAS_INTHINK_TOOL_LEAK_OPENERS=0
export RUST_LOG=info

exec /home/ms/atlas/.claude/worktrees/exl3-research/target/release/spark serve \
  --model-from-path "$SNAP" \
  --model-name qwen4exp-exl3 \
  --kernel-target qwen3.8-flash-next \
  --world-size 1 \
  --bind 0.0.0.0 --port 8890 \
  --max-seq-len "$CTX" \
  --max-num-seqs "$SEQS" --max-batch-size "$SEQS" \
  --gpu-memory-utilization 0.6 \
  --kv-cache-dtype bf16 \
  --ssm-cache-slots 16 \
  --request-timeout 1800 \
  --fast-load-prefetch-shards \
  --enable-prefix-caching \
  --default-chat-template-kwargs '{"reasoning_effort":"low"}' \
  --no-tui
