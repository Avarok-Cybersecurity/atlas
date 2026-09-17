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
#
# On a SCALE build Atlas reads free VRAM from the amdgpu sysfs counters
# (/sys/class/drm/card*/device/mem_info_vram_{total,used}) rather than
# cuMemGetInfo, whose free figure is not truthful there. That is the default
# and needs no export here; ATLAS_MEMINFO_SOURCE=driver|sysfs|sysfs:<dir>
# overrides it for a bisect. See the r9700 section of docs/HARDWARE.md.
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
# ATLAS_NO_FP8_PREDEQUANT=1 (spark-model/src/layers/fp8_predequant.rs) stops the
# loader building NVFP4-to-FP8 prefill copies of the SSM out_proj, the attention
# q/k/v/o and the MoE gate + shared expert. The prefill dispatch PREFERS those
# copies over both NVFP4 arms, and the GEMM it then launches is
# w4a16_fp8_ldmab::fp8_fp8_gemm_ldmab, a module gfx1201 does not compile — which
# on 2026-09-17 made every Ornith-1.0-9B request on this board die at layer 0
# with `Module 'w4a16_fp8_ldmab' not loaded`. The Strix recipe carried this
# variable historically; it was dropped from this script when its reader was
# lost, and the reader is back.
#
# BELT, NOT THE FIX: the guard probes the kernels itself and skips the copies on
# any target that cannot launch them, so a serve without this export is correct
# too. It is exported here so the r9700 recipe says out loud which path it is
# on, and so an operator reading the serve log sees `ATLAS_NO_FP8_PREDEQUANT is
# set` rather than having to infer it from a kernel name.
case "$ATLAS_TARGET_HW" in
  r9700) export ATLAS_NO_FP8_PREDEQUANT=1 ;;
esac
# ATLAS_LOAD_TRANSPOSED_TWINS=auto (spark-model/src/weight_loader/qwen35_dense/
# transposed_twins.rs) lets the loader decide, once and before any layer
# allocates, whether this board can hold the transposed second copy of every
# quantised weight. THE TRADE, stated plainly: without the twins a 27B NVFP4
# model fits 32 GB, and FFN prefill falls off w4a16_gemm_t_m128 onto the plain
# w4a16_gemm, at ~7 TFLOP/s against ~51 on the Gemma-4-31B measurement, 7x
# slower prefill. Decode is untouched. The twins are 12.74 GiB on
# unsloth/Qwen3.8-27B-NVFP4 and 3.62 GiB on Ornith-1.0-9B, so `auto` will
# normally BUILD them for the small models and skip them for the 27B; the serve
# log says which way it went and why. `=1` forces them on (and forces the 27B
# back to not loading), `=0` forces them off unconditionally.
# See the r9700 section of docs/HARDWARE.md and docs/porting/r9700-residency.md.
case "$ATLAS_TARGET_HW" in
  r9700) export ATLAS_LOAD_TRANSPOSED_TWINS=auto ;;
esac
# Removed here: ATLAS_FORCE_GLOBAL_GDN. That name has no reader anywhere in this
# tree as of this commit, so exporting it only suggested a control that does not
# exist.

# SCALE libs FIRST so /opt/rocm cannot shadow the fixed libhsa-runtime64 (the
# gfx1151 queue-create fix lives in SCALE 1.7.1's bundled ROCm 7.2.3):
export LD_LIBRARY_PATH="$scale_target/lib:$SCALE_HOME/lib"
export PATH="$scale_target/bin:$PATH"

echo "serving $MODEL on $(/opt/rocm/bin/rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9]*' || echo "$arch")"
exec target/release/spark serve "$MODEL" \
  --port "${PORT:-8081}" --max-seq-len "${MAX_SEQ_LEN:-4096}" \
  --gpu-memory-utilization "${GPU_UTIL:-$default_gpu_util}" \
  --kv-cache-dtype bf16 --kv-high-precision-layers max --max-batch-size 4
