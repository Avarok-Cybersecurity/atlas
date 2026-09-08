#!/bin/bash
# ── Qwen3.8-Flash-Next EXL3 4.05bpw, single GB10 ─────────────────────────
# Byte-for-byte the shipped serve preset (kernels/gb10/qwen3.8-flash-next/
# MODEL.toml, [[serve_presets]] "qwen3.8-flash-next-exl3"), spelled out so an
# A/B can change exactly ONE variable and prove which one.
#
#   BIN=... CTX=32768 SEQS=4 ./run_exl3_single.sh
#
# Everything the caller may vary is an env default; nothing else moves.
set -uo pipefail
BIN="${BIN:-/home/ms/spark-exl3}"
SNAP="${SNAP:-/tank/exl3-ckpt/qwen38-flash-next-4.05bpw}"
PORT="${PORT:-8891}"
CTX="${CTX:-32768}"
SEQS="${SEQS:-4}"

export LD_LIBRARY_PATH="/home/ms/atlas-gdn-libs:/home/ms/nccl/build/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# [serve_presets.env], verbatim.
export ATLAS_EXL3_NATIVE=1 ATLAS_EXL3_NATIVE_MOE=1 ATLAS_EXL3_NATIVE_DENSE=1
export ATLAS_PLE_MAX_TOKENS=9216
export ATLAS_PLE_CACHE_SLOTS=4194304
export ATLAS_QSA_MAX_TOKENS="$CTX"
export ATLAS_INTHINK_TOOL_LEAK_OPENERS=0
export ATLAS_QWEN4EXP_MTP=1 ATLAS_QWEN4EXP_MTP_VERIFY=1 ATLAS_DFLASH_SPEC_THINK=1
export ATLAS_QWEN4EXP_MTP_HC_BATCHED=1
export ATLAS_VERIFY_EXL3_ROW_ROUTER=1 ATLAS_VERIFY_EXL3_STABLE_GRID=1
export ATLAS_NO_VERIFY_ROW_FFN=1 ATLAS_NO_THINKENDED_GPU_ARGMAX=1
export ATLAS_MTP_MAX_SEQS=4
# Caller-overridable so the row-cap A/B can move exactly this one knob; the
# default is the preset's committed value.
export ATLAS_EXL3_MOE_ROWS_PER_EXPERT="${MOE_ROWS:-1024}"
export ATLAS_EXL3_DENSE_RECONSTRUCT_ROWS=512
export ATLAS_MARCONI_MIN_TOKENS=64
export RUST_LOG="${RUST_LOG:-info}"

# ATLAS_AUX_COLLECT_BATCHED is deliberately NOT set here — it is the variable
# under test and the caller owns it. Unset/1 = batched gather, 0 = the legacy
# per-layer draining collect.
echo "host=$(hostname) ctx=$CTX seqs=$SEQS aux_batched=${ATLAS_AUX_COLLECT_BATCHED:-unset} extra=[${EXTRA_ARGS:-}] simhash=${ATLAS_SIMHASH_LOOP:-default} no_suppress=${ATLAS_LOOP_NO_SUPPRESS:-default} bin=$(sha256sum "$BIN" | cut -c1-16)"

# EXTRA_ARGS: appended verbatim to the serve command line. Empty by default, so
# the preset is unchanged unless an A/B explicitly asks for a flag (e.g.
# `--content-loop-watchdog false`). Deliberately unquoted at the call site so a
# multi-flag string splits into separate argv entries.
exec "$BIN" serve ${EXTRA_ARGS:-} \
  --model-from-path "$SNAP" \
  --model-name qwen3.8-flash-next \
  --kernel-target qwen3.8-flash-next \
  --bind 0.0.0.0 --port "$PORT" \
  --max-seq-len "$CTX" \
  --max-num-seqs "$SEQS" --max-batch-size "$SEQS" \
  --gpu-memory-utilization "${GPU_UTIL:-0.72}" \
  --kv-cache-dtype bf16 \
  --ssm-cache-slots 64 \
  --request-timeout 1800 \
  --fast-load-prefetch-shards \
  --enable-prefix-caching \
  --speculative --num-drafts 2 \
  --default-chat-template-kwargs '{"reasoning_effort":"low","preserve_thinking":true}' \
  --no-tui
