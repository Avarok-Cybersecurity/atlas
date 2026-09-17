#!/usr/bin/env bash
# Serve a model with Atlas on AMD GPUs (SCALE runtime).
#
# The hardware target comes from ATLAS_TARGET_HW and defaults to `strix`. The
# SCALE toolchain directory is read from the `arch` key of
# kernels/$ATLAS_TARGET_HW/HARDWARE.toml rather than hardcoded, so this script
# and the build agree about the arch by construction.
#
#   ./serve-amd.sh                                                   # strix / gfx1151
#   ATLAS_TARGET_HW=r9700 ./serve-amd.sh unsloth/Qwen3.8-27B-NVFP4   # r9700 / gfx1201
#
# Verified coherent on gfx1151 / Strix Halo with Qwen/Qwen3.6-27B-FP8. On
# gfx1201 / Radeon AI PRO R9700 the build is green and the two runtime knobs
# below are re-verified as necessary, but coherent generation has NOT been
# observed yet. See docs/porting/amd-strix-halo-scale.md and the r9700 section
# of docs/HARDWARE.md.
set -euo pipefail
cd "$(dirname "$0")"

: "${SCALE_HOME:=$HOME/scale171/scale-1.7.1-Linux}"
export ATLAS_TARGET_HW="${ATLAS_TARGET_HW:-strix}"

# Defaults that follow the hardware: the model served, and the fraction of the
# GPU pool the KV sizer may fill.
case "$ATLAS_TARGET_HW" in
  r9700)
    default_model="unsloth/Qwen3.8-27B-NVFP4"
    # 32 GB of dedicated GDDR6, but this is a discrete board that may also be
    # driving a desktop session; the compositor and its surfaces want VRAM the
    # KV sizer must not have already taken.
    default_gpu_util="0.75"
    ;;
  *)
    default_model="Qwen/Qwen3.6-27B-FP8"
    # Strix shares one LPDDR5X pool with the host, and CUDA-graph capture
    # allocates on top during warmup; above 0.70 the OOM watchdog fires
    # mid-warmup. A desktop session on the same silicon needs that headroom too.
    default_gpu_util="0.70"
    ;;
esac
MODEL="${1:-$default_model}"

hardware_toml="kernels/$ATLAS_TARGET_HW/HARDWARE.toml"
if [[ ! -f "$hardware_toml" ]]; then
  echo "serve-amd.sh: no such hardware target: $hardware_toml" >&2
  echo "  ATLAS_TARGET_HW must name a kernels/<hw>/ directory, e.g. strix or r9700." >&2
  exit 1
fi

# One grep + sed rather than a TOML tool: `arch` is a bare quoted scalar on its
# own line in every kernels/<hw>/HARDWARE.toml, and a bring-up box is not
# guaranteed to have a parser installed. The leading-whitespace anchor keeps
# this off the `# ... arch selects ...` comment lines above the key.
arch="$(grep -m1 -E '^[[:space:]]*arch[[:space:]]*=' "$hardware_toml" \
  | sed -E 's/^[^=]*=[[:space:]]*"([^"]+)".*/\1/')" || true
if [[ -z "$arch" ]]; then
  echo "serve-amd.sh: no [hardware].arch key in $hardware_toml" >&2
  exit 1
fi

scale_target="$SCALE_HOME/targets/$arch"
if [[ ! -d "$scale_target" ]]; then
  echo "serve-amd.sh: SCALE has no $arch target at $scale_target" >&2
  echo "  ($hardware_toml declares arch = \"$arch\".)" >&2
  echo "  Install the SCALE build that ships targets/$arch, or point SCALE_HOME at it." >&2
  exit 1
fi

# Runtime knobs. Only these two are exported: both have readers in this tree,
# and both are required on every SCALE target we have silicon for.
#
# ATLAS_W4A16_VARIANT=v1 (spark-model/src/layers/mod.rs) pins the BF16-MMA
# NVFP4 GEMM instead of the FP8 path. SCALE emits no e4m3 MMA codegen on
# either AMD arch: on gfx1201 it has no `fragment<accumulator, 16, 8, 32,
# float>` declaration at all and rejects inline cvt.rn.satfinite.e4m3x2.f32,
# while BF16 mma.sync m16n8k16 compiles, which is exactly what v1 uses.
export ATLAS_W4A16_VARIANT=v1
# ATLAS_NO_GDN_FP8_PREFILL=1 (spark-model/src/weight_loader/qwen35_dense.rs)
# keeps GDN/SSM prefill off the native-FP8 path. It is NOT set for strix: the
# gfx1151 config that produced coherent output never set it, and the native
# FP8 SSM prefill is on by default there (kernels/strix/HARDWARE.toml calls
# that an open question, not a settled shim). On r9700 it is set as the
# conservative first-serve default until the runtime bisect on gfx1201 says
# which way is correct; whichever wins gets recorded in kernels/r9700.
case "$ATLAS_TARGET_HW" in
  r9700) export ATLAS_NO_GDN_FP8_PREFILL=1 ;;
esac
# Removed here: ATLAS_FORCE_GLOBAL_GDN and ATLAS_NO_FP8_PREDEQUANT. Neither
# name has a reader anywhere in this tree as of this commit, so exporting them
# only suggested a control that does not exist.

# SCALE libs FIRST so /opt/rocm cannot shadow the fixed libhsa-runtime64 (the
# gfx1151 queue-create fix lives in SCALE 1.7.1's bundled ROCm 7.2.3):
export LD_LIBRARY_PATH="$scale_target/lib:$SCALE_HOME/lib"
export PATH="$scale_target/bin:$PATH"

echo "serving $MODEL on $(/opt/rocm/bin/rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9]*' || echo "$arch")"
exec target/release/spark serve "$MODEL" \
  --port "${PORT:-8081}" --max-seq-len "${MAX_SEQ_LEN:-4096}" \
  --gpu-memory-utilization "${GPU_UTIL:-$default_gpu_util}" \
  --kv-cache-dtype bf16 --kv-high-precision-layers max --max-batch-size 4
