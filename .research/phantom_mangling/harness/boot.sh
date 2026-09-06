#!/bin/bash
# Boot the 4.05bpw EXL3 qwen4_exp serve. $1 = "on" | "off"  (prefix caching)
# EVERY other flag is identical between the two arms.
set -uo pipefail
ARM=$1
SNAP=/tank/exl3-ckpt/qwen38-flash-next-4.05bpw
CTX=32768
PREFILL=8192
SEQS=1

export LD_LIBRARY_PATH="/home/ms/atlas-gdn-libs:/home/ms/nccl/build/lib:${LD_LIBRARY_PATH:-}"
export ATLAS_EXL3_NATIVE=1
export ATLAS_EXL3_NATIVE_MOE=1
export ATLAS_EXL3_NATIVE_DENSE=1
export ATLAS_PLE_MAX_TOKENS=$((PREFILL + 1024))
export ATLAS_PLE_CACHE_SLOTS=$(( CTX * 16 * SEQS > 4194304 ? CTX * 16 * SEQS : 4194304 ))
export ATLAS_QSA_MAX_TOKENS=$CTX
export ATLAS_INTHINK_TOOL_LEAK_OPENERS=0
export ATLAS_NO_HW_PRECHECK=1
export RUST_LOG=info

PREFIX_FLAG=()
if [ "$ARM" = "on" ]; then PREFIX_FLAG=(--enable-prefix-caching); fi

exec /home/ms/atlas/.claude/worktrees/exl3-research/target/release/spark serve \
  --model-from-path "$SNAP" \
  --model-name qwen4exp-exl3 \
  --kernel-target qwen3.8-flash-next \
  --world-size 1 \
  --bind 0.0.0.0 --port 8890 \
  --max-seq-len "$CTX" \
  --max-num-seqs "$SEQS" --max-batch-size "$SEQS" \
  --gpu-memory-utilization 0.72 \
  --kv-cache-dtype bf16 \
  --ssm-cache-slots 16 \
  --request-timeout 1800 \
  --fast-load-prefetch-shards \
  "${PREFIX_FLAG[@]}" \
  --default-chat-template-kwargs '{"reasoning_effort":"low"}' \
  --no-tui
