#!/usr/bin/env bash
# Serve a model with Atlas on AMD GPUs (SCALE runtime). Verified coherent on
# gfx1151 / Strix Halo with Qwen/Qwen3.6-27B-FP8. See
# docs/porting/amd-strix-halo-scale.md.
set -euo pipefail
cd "$(dirname "$0")"
: "${SCALE_HOME:=$HOME/scale171/scale-1.7.1-Linux}"
MODEL="${1:-Qwen/Qwen3.6-27B-FP8}"
# gfx1151 runtime shims (each explained in docs §4):
export ATLAS_FORCE_GLOBAL_GDN=1   # GDN prefill -> global-mem kernel (RDNA3.5 64KB LDS cap)
export ATLAS_W4A16_VARIANT=v1     # BF16-MMA NVFP4 GEMM (SCALE device FP8 encode is broken on gfx1151)
export ATLAS_NO_FP8_PREDEQUANT=1  # skip NVFP4->FP8 predequant (same reason)
# SCALE libs FIRST so /opt/rocm cannot shadow the fixed libhsa-runtime64 (the
# gfx1151 queue-create fix lives in SCALE 1.7.1 bundled ROCm 7.2.3):
export LD_LIBRARY_PATH="$SCALE_HOME/targets/gfx1151/lib:$SCALE_HOME/lib"
export PATH="$SCALE_HOME/targets/gfx1151/bin:$PATH"
echo "serving $MODEL on $(/opt/rocm/bin/rocminfo 2>/dev/null | grep -m1 -o gfx[0-9]* || echo AMD)"
exec target/release/spark serve "$MODEL" \
  --port "${PORT:-8081}" --max-seq-len "${MAX_SEQ_LEN:-4096}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.70}" \
  --kv-cache-dtype bf16 --kv-high-precision-layers max --max-batch-size 4
