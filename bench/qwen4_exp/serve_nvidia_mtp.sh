#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
cd "$(dirname "$0")/../.."
export ATLAS_PLE_MAX_TOKENS=3072 ATLAS_PLE_CACHE_SLOTS=4194304 ATLAS_QSA_MAX_TOKENS=32768
export ATLAS_INTHINK_TOOL_LEAK_OPENERS=0
export ATLAS_NO_HW_PRECHECK=1
export ATLAS_QWEN4EXP_MTP=1 ATLAS_QWEN4EXP_MTP_VERIFY=1 ATLAS_DFLASH_SPEC_THINK=1
export ATLAS_QWEN4EXP_MTP_HC_BATCHED=1
export ATLAS_VERIFY_ROW_PROJ=1
export ATLAS_MTP_ACCEPT_DEBUG=1 ATLAS_NO_THINKENDED_GPU_ARGMAX=1
exec "${ATLAS_SPARK_BIN:?set path to the built spark binary}" serve \
  --model-from-path "${QWEN4EXP_PATH:?set NVIDIA snapshot path}" \
  --model-name qwen4exp-nvfp4 --kernel-target qwen3.8-flash-next \
  --world-size 1 --bind 127.0.0.1 --port 8892 \
  --max-seq-len 32768 --max-num-seqs 1 --max-batch-size 1 \
  --gpu-memory-utilization 0.71 --kv-cache-dtype bf16 --ssm-cache-slots 4 --max-prefill-tokens 2048 --vision-max-pixels 1048576 \
  --request-timeout 1800 --fast-load-prefetch-shards \
  --speculative --num-drafts "${MTP_DRAFTS:?set MTP_DRAFTS}" --mtp-gate force \
  --default-chat-template-kwargs '{"reasoning_effort":"low"}' "$@"
