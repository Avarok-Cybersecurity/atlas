#!/bin/bash
# HF cache triage helper — summarize on-disk HuggingFace model cache to guide
# what's safe to remove before pulling a large new checkpoint (e.g. the 230 GB
# MiniMaxAI/MiniMax-M2).
#
# Runs READ-ONLY. Never deletes anything. Prints:
#   - total free space on the cache filesystem
#   - per-model on-disk size (top 20)
#   - a ranked purge list (non-production first, then duplicates)
#
# Usage: scripts/hf-cache-triage.sh
#        scripts/hf-cache-triage.sh --target 230G
set -euo pipefail

CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub"
TARGET_GB="${1:-230}"
TARGET_GB="${TARGET_GB#--target }"
TARGET_GB="${TARGET_GB%G}"

if [[ ! -d "$CACHE_DIR" ]]; then
  echo "No HF cache at $CACHE_DIR"
  exit 1
fi

# Never-remove list: models in active test rotation per CLAUDE.md and
# feedback_sehyo_models.md. Script will refuse to include these in purge
# suggestions.
KEEP_MATCH=(
  'models--Sehyo--Qwen3.5-35B-A3B-NVFP4'       # production test 35B
  'models--Sehyo--Qwen3.5-122B-A10B-NVFP4'     # production test 122B
  'models--nvidia--Qwen3-Next-80B-A3B-Instruct-NVFP4'
  'models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4'
  'models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4'
  'models--nvidia--Gemma-4-31B-IT-NVFP4'
  'models--mistralai--Mistral-Small-4-119B-2603-NVFP4'
  'models--ig1--Qwen3-VL-30B-A3B-Instruct-NVFP4'
  'models--bg-digitalservices--Gemma-4-26B-A4B-it-NVFP4A16'
  'models--Qwen--Qwen3-Coder-Next-FP8'          # active FP8 bring-up
  'models--MiniMaxAI--MiniMax-M2'               # target pull (metadata cached)
  'models--MiniMaxAI--MiniMax-M2.7'             # target pull (metadata cached)
  'models--yujiepan--minimax-m2.7-tiny-random'  # dev harness
)

is_keep() {
  local name="$1"
  for k in "${KEEP_MATCH[@]}"; do
    [[ "$name" == *"$k"* ]] && return 0
  done
  return 1
}

echo "HF cache: $CACHE_DIR"
echo
df -BG "$CACHE_DIR" | awk 'NR==2 {printf "Filesystem free: %s / %s (%s used)\n\n", $4, $2, $5}'

echo "── Per-model size (top 20) ──"
du -sBG "$CACHE_DIR"/models--*/ 2>/dev/null \
  | sort -rh \
  | head -20 \
  | awk '{printf "%6s  %s\n", $1, $2}'

echo
# Compute how much more we need to free to reach TARGET_GB *total* free.
free_gb=$(df -BG "$CACHE_DIR" | awk 'NR==2 {sub("G","",$4); print $4}')
need_gb=$(( TARGET_GB > free_gb ? TARGET_GB - free_gb : 0 ))
if [[ $need_gb -eq 0 ]]; then
  echo "── Already ${free_gb} GB free (target ${TARGET_GB} GB) — no purge needed ──"
  exit 0
fi
echo "── Ranked purge candidates: need to free ${need_gb} GB more to hit ${TARGET_GB} GB total ──"
echo "   (skipping production-rotation models per feedback_sehyo_models.md)"
echo
# Purge order: Kbenkhaled duplicates first, then Qwen3.5-122B-FP8 (we have Sehyo NVFP4),
# then Qwen3.5-35B-FP8 (we have Sehyo NVFP4), then anything else sized.
PURGE_ORDER=(
  'models--Kbenkhaled--Qwen3.5-27B-NVFP4'
  'models--Kbenkhaled--Qwen3.5-35B-A3B-NVFP4'
  'models--chankhavu--Nemotron-Cascade-2-30B-A3B-NVFP4'
  'models--Qwen--Qwen-7B'
  'models--Qwen--Qwen3.5-122B-A10B'
  'models--Qwen--Qwen3.5-35B-A3B-FP8'
  'models--Qwen--Qwen3.5-122B-A10B-FP8'
)

cum=0
target_bytes=$((need_gb * 1024 * 1024 * 1024))
for candidate in "${PURGE_ORDER[@]}"; do
  path="$CACHE_DIR/$candidate"
  if [[ ! -d "$path" ]]; then continue; fi
  if is_keep "$candidate"; then
    printf "  SKIP (keep)    %-60s\n" "$candidate"
    continue
  fi
  size_bytes=$(du -sb "$path" 2>/dev/null | awk '{print $1}')
  size_gb=$(awk -v b="$size_bytes" 'BEGIN{printf "%.1f", b/1024/1024/1024}')
  cum=$((cum + size_bytes))
  cum_gb=$(awk -v b="$cum" 'BEGIN{printf "%.1f", b/1024/1024/1024}')
  printf "  rm -rf %-60s  # %5.1f GB (cum %5.1f GB)\n" "$path" "$size_gb" "$cum_gb"
  if [[ $cum -ge $target_bytes ]]; then
    echo "  # ^ reaches target; stopping"
    break
  fi
done

if [[ $cum -lt $target_bytes ]]; then
  missing=$(awk -v b="$((target_bytes - cum))" 'BEGIN{printf "%.1f", b/1024/1024/1024}')
  echo
  echo "# WARNING: purge plan frees only ${cum_gb} GB. Still ${missing} GB short."
  echo "# To reach target, you'd need to rm a production-rotation model."
fi

echo
echo "Nothing will run without your explicit rm. Review the list first."
