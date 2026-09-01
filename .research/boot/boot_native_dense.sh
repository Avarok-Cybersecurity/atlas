#!/bin/bash
# Single-node boot: qwen4_exp 2.05bpw EXL3 with routed experts AND the routed
# dense projections (the whole GDN family — in_proj_qkv + in_proj_z as a
# shared-A pair into the fused [Q|K|V|Z] arena row, and out_proj — on 36
# layers, and the whole attention family — q/k/v/o_proj — on the 12
# full-attention layers) served natively from packed trellis
# (ATLAS_EXL3_NATIVE_MOE=1 + ATLAS_EXL3_NATIVE_DENSE=1; ATLAS_EXL3_NATIVE_GDN=0 /
# ATLAS_EXL3_NATIVE_ATTN=0 opt one family back out for A/B). Set LOG= to
# redirect; a MemAvailable watchdog kills the serve under 10 GB.
# Expected under the gate: the "EXL3 materialization done: ... N linears ->
# BF16 dense" count drops by 3 per GDN layer and 4 per attention layer
# (332 -> 332 - 108 - 48 = 176 of the MoE-only 332), one "EXL3 native GDN
# family installed" line per GDN layer and one "EXL3 native attention family
# installed" line per attention layer.
set -uo pipefail
SNAP=/tank/exl3-ckpt/qwen38-flash-next-2.05bpw
CTX=8192
SEQS=${SEQS:-1}
LOG=${LOG:-/home/ms/.claude/jobs/5a7bd33d/tmp/boot-native-dense.log}

export LD_LIBRARY_PATH="/home/ms/atlas-gdn-libs:/home/ms/nccl/build/lib:${LD_LIBRARY_PATH:-}"
export ATLAS_EXL3_NATIVE=1
export ATLAS_EXL3_NATIVE_MOE=1
export ATLAS_EXL3_NATIVE_DENSE=1
export ATLAS_PLE_MAX_TOKENS=9000
export ATLAS_PLE_CACHE_SLOTS=4194304
export ATLAS_QSA_MAX_TOKENS=8192
export ATLAS_INTHINK_TOOL_LEAK_OPENERS=0
export RUST_LOG=info

# MemAvailable watchdog: kill the serve if the box drops under 10 GB.
(
  while true; do
    avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    if [ "$avail_kb" -lt $((10 * 1024 * 1024)) ]; then
      echo "WATCHDOG: MemAvailable=${avail_kb}kB < 10GB — killing serve" >> "$LOG"
      pkill -f 'exl3-research/target/release/spark serve'
      exit 0
    fi
    pgrep -f 'exl3-research/target/release/spark serve' > /dev/null || exit 0
    sleep 2
  done
) &

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
  --ssm-cache-slots 48 \
  --request-timeout 1800 \
  --fast-load-prefetch-shards \
  --enable-prefix-caching \
  --default-chat-template-kwargs '{"reasoning_effort":"low"}' \
  --no-tui > "$LOG" 2>&1
