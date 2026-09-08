#!/bin/bash
# Arm C: --enable-prefix-caching ON (so the GDN prefill recurrence kernel is the
# SAME token-sequential ladder as arm A -- gdn_replay.rs:63-70 forces it whenever
# prefix caching is active on a hybrid-SSM model), but EVERY Marconi SSM/PLE/QSA
# snapshot restore is declined by pushing the minimum snapshot depth above any
# reachable prompt length (mtp_carry.rs:91, default 256).
#
# A vs C therefore isolates the WARM RESTORE itself with the recurrence kernel
# held fixed; A vs B measures the shipped flag as a whole.
set -uo pipefail
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
export ATLAS_MARCONI_MIN_TOKENS=999999999
export RUST_LOG=info

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
  --enable-prefix-caching \
  --default-chat-template-kwargs '{"reasoning_effort":"low"}' \
  --no-tui
