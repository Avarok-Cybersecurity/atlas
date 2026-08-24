#!/usr/bin/env bash
# Lean C=1 probe serve for the Apathy v2 block-16 drafter (32K x 2).
# The 128K x 8 profile does not fit this drafter at util 0.80: 4.26 GB
# weights + 8 capture layers (ctx acc 8x hidden/pos vs incoai's 5x) +
# gamma=16 doubling drafter-KV and the K=17 verify pools -> KV OOM.
set -uo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="/home/ms/atlas-gdn-libs:/home/ms/nccl/build/lib:${LD_LIBRARY_PATH:-}"
export ATLAS_GDN_FLASHINFER=1
export ATLAS_GDN_LIB=/home/ms/atlas-gdn-libs/libatlasgdn.so
export ATLAS_SIMHASH_LOOP=0
export ATLAS_LOOP_NO_SUPPRESS=1
export ATLAS_LOOP_SOFT_BIAS=1
export ATLAS_CONTENT_LOOP_MIN_REPEATS=12
# Drafter NVFP4 propose (24a96047): gate-validated 2026-08-24 (agentic
# 10/10+10/10, s/turn 6.83 vs fp8-twin 7.28). NVFP4=0 reverts.
export ATLAS_DFLASH_DRAFTER_NVFP4="${NVFP4:-1}"
export ATLAS_KV_OVERCOMMIT=1
export ATLAS_DFLASH_SPEC_THINK=1
export ATLAS_DFLASH_RESUME_GUARD=8
export RUST_LOG="${RUST_LOG:-info}"

# THINKING=1 REASONING=low|medium|xhigh mirrors serve_dflash_tui.sh: drops
# --disable-thinking and sets the template's reasoning_effort.
THINK_ARGS=(--disable-thinking)
if [ "${THINKING:-0}" = "1" ]; then
  THINK_ARGS=(--default-chat-template-kwargs "{\"reasoning_effort\":\"${REASONING:-medium}\"}")
fi

TARGET=$(ls -d /mnt/gx10-hf-hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/*/ | head -1)
# DRAFT_DIR overrides the drafter (e.g. incoai, for a same-profile control A/B).
DRAFT="${DRAFT_DIR:-/mnt/gx10-hf-hub/models--onewhosighs--Apathy-Qwen3.8-27B-DFlash-drafter-v2/snapshots/64f3e67ce7531279636964a253763482765789fa/}"

LAZY_ARGS=()
[ -n "${WY17_LAZY:-}" ] && LAZY_ARGS=(--gdn-wy17-lazy "$WY17_LAZY")

exec ./target/release/spark serve \
  --model-from-path "$TARGET" \
  --model-name unsloth/Qwen3.8-27B-NVFP4 \
  --draft-model "$DRAFT" --dflash \
  --bind 0.0.0.0 --port 8888 \
  --max-seq-len 32768 \
  --max-num-seqs 2 --max-batch-size 2 \
  --gpu-memory-utilization 0.80 \
  --request-timeout 900 \
  --max-prefill-tokens 8192 \
  --enable-prefix-caching true \
  --ssm-cache-slots 16 \
  --scheduling-policy slai \
  --tbt-deadline-ms 100 \
  --lm-head-dtype fp8 "${THINK_ARGS[@]}" --default-top-n-sigma 0 \
  "${LAZY_ARGS[@]}" \
  --no-tui
