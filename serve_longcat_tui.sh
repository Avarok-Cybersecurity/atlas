#!/usr/bin/env bash
# Serve LongCat-Flash-Lite with Atlas and bring up the TUI dashboard (the TUI
# is automatic on an interactive terminal — do NOT pipe this, or it disables
# itself and you get the plain log stream).
#
# This is the n-gram-embedding model: 14 checkpoint layers -> 28 engine
# sublayers of MLA + shortcut MoE, with a fused input embedding that mixes the
# base token row with 12 hashed n-gram lookups.
#
# Validated at this config on 2026-08-25 against
# bench/ngram_ref/longcat_forward_golden.npz: the fused embedding matches the
# reference at cos 1.0000 and every one of the 14 checkpoint layers holds
# >= 0.9952 across all 28 sublayers.
#
#   ./serve_longcat_tui.sh                  # port 8888, TUI up
#   PORT=8899 ./serve_longcat_tui.sh        # somewhere else
#   MAX_SEQ_LEN=65536 ./serve_longcat_tui.sh
#
# Port defaults to 8888 because that is what bench/agentic/* expects
# (ATLAS_URL defaults to http://localhost:8888/v1/chat/completions), so the
# agentic harnesses point at this with no extra flags.
#
# ONE Atlas instance at a time: --gpu-memory-utilization RESERVES its whole
# fraction of the box up front, so a second server will fail its OOM
# pre-flight. Kill the running one by PID first.
#
# ── PRECISION LEVERS (all default OFF; measured 2026-08-26) ──
#
# LongCat ships plain BF16 with no NVFP4/FP8 calibration metadata, so Atlas
# runtime-quantizes everything to NVFP4 at load. That is lossy. Three env
# flags buy it back, measured against the reference logits in
# bench/ngram_ref/longcat_forward_golden.npz via bench/ngram_ref/logit_quality.py
# (full 131072-vocab KL, not a top-20 sample; deterministic on repeat):
#
#   arm                              KL vs ref  logit cos  top-5  tok/s
#   (default, all NVFP4)                0.0464   0.998250    3/5  24.31
#   FP8_EXPERTS                         0.0344   0.998588    4/5  22.69   <= best value
#   NVFP4_MLA=0                         0.0382   0.998567    3/5  20.18
#   NVFP4_MLA=0 + FP8_EXPERTS           0.0301   0.998799    4/5  ~19.2
#   NVFP4_MLA=0 + BF16_FFN              0.0240   0.999033    3/5  16.67   <= best quality
#
# ★ ATLAS_LONGCAT_FP8_EXPERTS=1 is the one to reach for first: -26% KL for
#   -6.7% decode, and it is the only arm that fixes the top-5 shortlist the
#   model card's `top_k: 4` actually samples from. It strictly dominates
#   ATLAS_NVFP4_MLA=0 — better quality AND faster.
#
# WHY the ordering is not intuitive: decode is weight-bandwidth bound (~185
# GB/s effective, calibrated from the measured pairs), so each lever costs in
# proportion to the extra weight bytes it reads PER TOKEN:
#
#   experts -> FP8       +0.74 GB/tok   sparse: top-12 of 256 fire per token
#   MLA -> BF16          +1.28 GB/tok   dense: read every token
#   dense FFN -> BF16    +2.21 GB/tok   dense: read every token
#
# The routed experts are 63.0 of the 70.2 GB resident and carry almost all of
# the quantization error, yet they are the CHEAPEST group to upgrade, because
# routing only ever touches 12/256 of them. The little dense FFN is the most
# expensive, because every token reads all of it.
#
# Off by default because none of this is free. Pick by what you are doing:
#   ATLAS_LONGCAT_FP8_EXPERTS=1 ./serve_longcat_tui.sh                    # recommended
#   ATLAS_NVFP4_MLA=0 ATLAS_LONGCAT_BF16_FFN=1 \
#     ATLAS_LONGCAT_FP8_EXPERTS=1 ./serve_longcat_tui.sh                  # max quality
set -euo pipefail
cd "$(dirname "$0")"

SNAP="${LONGCAT_PATH:-/tank/hf/hub/models--meituan-longcat--LongCat-Flash-Lite/snapshots/b62b68827ead0b7fef3ba98b57f18484acaaec06}"
if [[ ! -f "$SNAP/config.json" ]]; then
  echo "LongCat checkpoint not found at: $SNAP" >&2
  echo "Override with LONGCAT_PATH=/path/to/snapshot" >&2
  exit 1
fi

# NCCL lives outside the default loader path on this box.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/home/ms/nccl/build/lib"
# Serve at INFO — warn/error hides the load ledger and the n-gram cache line.
export RUST_LOG="${RUST_LOG:-info}"

# Resident rows per n-gram table. The 12 tables are 62.8 GB of BF16 on disk and
# are NEVER uploaded: they are served row-by-row off NVMe out of a pinned
# GPU-addressable arena. 65536 slots x 512 B x 12 tables = 403 MB, and that
# frees the rest for KV. Raise it if you see cache thrash on long contexts.
export ATLAS_NGRAM_CACHE_SLOTS="${ATLAS_NGRAM_CACHE_SLOTS:-65536}"

echo "LongCat-Flash-Lite  ->  port ${PORT:-8888}   (TUI: needs an interactive terminal)"
exec target/release/spark serve \
  --model-from-path "$SNAP" \
  --model-name "${MODEL_NAME:-longcat-full}" \
  --kernel-target longcat-flash-lite \
  --bind "${BIND:-127.0.0.1}" \
  --port "${PORT:-8888}" \
  --max-seq-len "${MAX_SEQ_LEN:-32768}" \
  --max-num-seqs "${MAX_NUM_SEQS:-16}" \
  --max-batch-size "${MAX_BATCH_SIZE:-16}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.80}" \
  --disable-thinking \
  "$@"
