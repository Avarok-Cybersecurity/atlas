#!/usr/bin/env bash
# Nemotron-3.5-Lightning-30B-A3B-NVFP4 — the atlas-recipes#18 `defaults:`
# block, transcribed for the wip/nemo-integration branch (PR stack
# #544→#545→#566). Behavior flags (no_decode_graphs, thinking_in_tools,
# disable_tool_steering, no_tool_system_prompt) all live in the target's
# MODEL.toml — no env workarounds.
#
# Recipe rationale (see the PR): only 6/52 layers are attention, fp8 KV is
# ~3072 B/token, so 8×256K fits at util 0.60 with 4.3× margin — 0.60 is a
# genuine fit, not a squeeze. Reference numbers: decode 71.8-76.5 tok/s
# single-stream, prefill ~2.1K tok/s @61k, needle PASS at 61k/124k ×
# 10/50/90% depth. --speculative stays OFF (reference recipe pins it).
#
# THINKING: thinking_default = true is the Lightning policy — do NOT pass
# --disable-thinking; tool calling NEEDS thinking (PR #566: 14/15 with,
# loops without). Client max_tokens must exceed the thinking budget.
set -uo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="/home/ms/nccl/build/lib:${LD_LIBRARY_PATH:-}"
export RUST_LOG="${RUST_LOG:-info}"

MODEL=$(ls -d /mnt/gx10-hf-hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/snapshots/*/ | head -1)

exec ./target/release/spark serve \
  --model-from-path "$MODEL" \
  --model-name nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --bind 0.0.0.0 --port 8888 \
  --max-seq-len 262144 \
  --max-num-seqs 8 --max-batch-size 8 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.60 \
  --scheduling-policy slai \
  --request-timeout 900 \
  --no-tui
