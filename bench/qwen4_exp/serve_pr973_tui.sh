#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# PR #973 (Qwen3.8-Flash-Next NVFP4, wide MTP verify fast-greedy) served with the
# TUI up.  The TUI enables itself on an interactive terminal, so run this
# DIRECTLY in your shell -- do not pipe it and do not run it under a harness, or
# it falls back to the plain log stream.
#
#   ./bench/qwen4_exp/serve_pr973_tui.sh                      # PR arm (fast greedy ON, the default)
#   ATLAS_MTP_THINK_FAST_GREEDY=0 ./bench/qwen4_exp/serve_pr973_tui.sh  # control arm (the #972 path)
#
# The env block and flags are serve_nvidia_mtp.sh's, with the PR's recorded
# deltas folded in (MTP_DRAFTS=2, --gpu-memory-utilization 0.76,
# ATLAS_QWEN4EXP_BF16_GDN=0) plus --enable-prefix-caching per the standing rule
# that every serve run uses a realistic config.  It is inlined rather than
# wrapping serve_nvidia_mtp.sh because clap rejects a repeated
# --gpu-memory-utilization, so 0.76 cannot simply be appended to that script.
#
# ONE Atlas instance at a time: --gpu-memory-utilization RESERVES its whole
# fraction of the box up front, so a second server fails its OOM pre-flight.
# Stop a running one by its exact PID first.
set -euo pipefail
cd "$(dirname "$0")/../.."

# libnccl comes from the out-of-tree NCCL build and is not on this box's default
# loader path; without this the binary dies with
#   "error while loading shared libraries: libnccl.so.2".
export LD_LIBRARY_PATH="/home/ms/nccl/build/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

SPARK_BIN="${ATLAS_SPARK_BIN:-$PWD/target/release/spark}"
MODEL_PATH="${QWEN4EXP_PATH:-/home/ms/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47}"
PORT="${PORT:-8892}"
MTP_DRAFTS="${MTP_DRAFTS:-2}"
GPU_UTIL="${GPU_UTIL:-0.76}"

export ATLAS_PLE_MAX_TOKENS=3072 ATLAS_PLE_CACHE_SLOTS=4194304 ATLAS_QSA_MAX_TOKENS=32768
export ATLAS_INTHINK_TOOL_LEAK_OPENERS=0
export ATLAS_NO_HW_PRECHECK=1
export ATLAS_QWEN4EXP_MTP=1 ATLAS_QWEN4EXP_MTP_VERIFY=1 ATLAS_DFLASH_SPEC_THINK=1
export ATLAS_QWEN4EXP_MTP_HC_BATCHED=1
export ATLAS_VERIFY_ROW_PROJ=1
export ATLAS_MTP_ACCEPT_DEBUG=1 ATLAS_NO_THINKENDED_GPU_ARGMAX=1
export ATLAS_QWEN4EXP_BF16_GDN="${ATLAS_QWEN4EXP_BF16_GDN:-0}"
export RUST_LOG="${RUST_LOG:-info}"

exec "$SPARK_BIN" serve \
  --model-from-path "$MODEL_PATH" \
  --model-name qwen4exp-nvfp4 --kernel-target qwen3.8-flash-next \
  --world-size 1 --bind 127.0.0.1 --port "$PORT" \
  --max-seq-len 32768 --max-num-seqs 1 --max-batch-size 1 \
  --gpu-memory-utilization "$GPU_UTIL" --kv-cache-dtype bf16 --ssm-cache-slots 4 \
  --max-prefill-tokens 2048 --vision-max-pixels 1048576 \
  --request-timeout 1800 --fast-load-prefetch-shards \
  --enable-prefix-caching \
  --speculative --num-drafts "$MTP_DRAFTS" --mtp-gate force \
  --default-chat-template-kwargs '{"reasoning_effort":"low"}' "$@"
