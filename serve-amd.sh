#!/usr/bin/env bash
# Serve a model with Avarok on AMD GPUs (SCALE runtime).
#
#   ./serve-amd.sh                                                   # strix / gfx1151
#   AVAROK_TARGET_HW=r9700 ./serve-amd.sh unsloth/Qwen3.8-27B-NVFP4   # r9700 / gfx1201
#
# Knobs, all with defaults:
#   $1                  model to serve. Default follows the hardware.
#   SCALE_HOME          SCALE install root. Default ~/scale171/scale-1.7.1-Linux.
#   AVAROK_TARGET_HW    kernels/<hw>/ directory to serve from. Default strix.
#   PORT                default 8081.
#   MAX_SEQ_LEN         default 4096.
#   GPU_UTIL            fraction of the GPU pool the KV sizer may fill.
#   MAX_BATCH           concurrent sequences.
#   OOM_GUARD_MB        headroom the fast loader's OOM pre-flight demands.
#
# The SCALE toolchain directory is read from the `arch` key of
# kernels/$AVAROK_TARGET_HW/HARDWARE.toml rather than hardcoded, so this script
# and the build agree about the arch by construction.
#
# Verified coherent on gfx1151 (Strix Halo) with Qwen/Qwen3.6-27B-FP8, and on
# gfx1201 (Radeon AI PRO R9700, SCALE 1.7.1 targets/gfx1201, ROCm 7.2.0) with
# unsloth/Qwen3.8-27B-NVFP4. See docs/porting/amd-strix-halo-scale.md and the
# r9700 section of docs/HARDWARE.md.
#
# Free VRAM on a SCALE build comes from the amdgpu sysfs counters
# (/sys/class/drm/card*/device/mem_info_vram_{total,used}) rather than
# cuMemGetInfo, whose free figure is not truthful there. That is the default
# and needs no export here; AVAROK_MEMINFO_SOURCE=driver|sysfs|sysfs:<dir>
# overrides it for a bisect.
set -euo pipefail
cd "$(dirname "$0")"

: "${SCALE_HOME:=$HOME/scale171/scale-1.7.1-Linux}"
export AVAROK_TARGET_HW="${AVAROK_TARGET_HW:-strix}"

# Defaults that follow the hardware: the model served, and the fraction of the
# GPU pool the KV sizer may fill.
case "$AVAROK_TARGET_HW" in
  r9700)
    default_model="unsloth/Qwen3.8-27B-NVFP4"
    # Measured on gfx1201 (Radeon AI PRO R9700, SCALE 1.7.1, ROCm 7.2.0),
    # 2026-09-17, on a board also driving a desktop session: the 27B loads to
    # 19.42 GB of weights (twins skipped, 9.35 GB released on consume) and
    # 23.0 GB pre-KV. At 0.75 or 0.80 the KV sizer refuses; at 0.90 with
    # --max-batch-size 1 it boots with a 3.7 GB KV budget (4128 tokens) and
    # 29.1 GB of VRAM in use, and answers coherently. A 9B (ornith-1.0-9b)
    # fits at 0.75 with batch 4.
    default_gpu_util="0.90"
    default_max_batch="1"
    # The fast loader's OOM pre-flight is on-disk bytes x 1.3 plus this
    # guard; the 4 GB default refuses a 21.8 GB checkpoint on a 32 GB board
    # before touching the card (measured: 28.35 GB projected peak + 4 GB
    # against 31.6 GB free). 1 GB is enough here; the load itself is bounded.
    default_oom_guard_mb="1024"
    ;;
  *)
    default_model="Qwen/Qwen3.6-27B-FP8"
    default_max_batch="4"
    default_oom_guard_mb="4096"
    # Strix shares one LPDDR5X pool with the host, and CUDA-graph capture
    # allocates on top during warmup; above 0.70 the OOM watchdog fires
    # mid-warmup. A desktop session on the same silicon needs that headroom too.
    default_gpu_util="0.70"
    ;;
esac
MODEL="${1:-$default_model}"

hardware_toml="kernels/$AVAROK_TARGET_HW/HARDWARE.toml"
if [[ ! -f "$hardware_toml" ]]; then
  echo "serve-amd.sh: no such hardware target: $hardware_toml" >&2
  echo "  AVAROK_TARGET_HW must name a kernels/<hw>/ directory, e.g. strix or r9700." >&2
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

# Runtime knobs. Each one exported below has a reader in this tree; see the
# r9700 section of docs/HARDWARE.md for the target facts behind them.
#
# AVAROK_W4A16_VARIANT=v1 (spark-model/src/layers/mod.rs) pins the BF16-MMA
# NVFP4 GEMM instead of the FP8 path. SCALE emits no e4m3 MMA codegen on
# either AMD arch: on gfx1201 it has no `fragment<accumulator, 16, 8, 32,
# float>` declaration at all and rejects inline cvt.rn.satfinite.e4m3x2.f32,
# while BF16 mma.sync m16n8k16 compiles, which is exactly what v1 uses.
export AVAROK_W4A16_VARIANT=v1
# AVAROK_NO_GDN_FP8_PREFILL=1 (spark-model/src/weight_loader/qwen35_dense.rs)
# keeps GDN/SSM prefill off the native-FP8 path. It is NOT set for strix: the
# gfx1151 config that produced coherent output never set it, and the native
# FP8 SSM prefill is on by default there (kernels/strix/HARDWARE.toml calls
# that an open question, not a settled shim). On r9700 it is set as the
# conservative first-serve default until the runtime bisect on gfx1201 says
# which way is correct; whichever wins gets recorded in kernels/r9700.
case "$AVAROK_TARGET_HW" in
  r9700) export AVAROK_NO_GDN_FP8_PREFILL=1 ;;
esac
# AVAROK_NO_FP8_PREDEQUANT=1 (spark-model/src/layers/fp8_predequant.rs) stops
# the loader building NVFP4-to-FP8 prefill copies of the SSM out_proj, the
# attention q/k/v/o and the MoE gate + shared expert. The prefill dispatch
# PREFERS those copies over both NVFP4 arms, and the GEMM it then launches is
# w4a16_fp8_ldmab::fp8_fp8_gemm_ldmab, a module gfx1201 does not compile, so
# every request died at layer 0 with `Module 'w4a16_fp8_ldmab' not loaded`.
# Belt, not the fix: the guard probes the kernels itself and skips the copies
# on any target that cannot launch them, so a serve without this export is
# correct too. It is exported so the serve log names the decision in words.
case "$AVAROK_TARGET_HW" in
  r9700) export AVAROK_NO_FP8_PREDEQUANT=1 ;;
esac
# AVAROK_LOAD_TRANSPOSED_TWINS=0 (spark-model/src/weight_loader/qwen35_dense/
# transposed_twins.rs) declines the transposed second copy of every quantised
# weight: 12.74 GiB on unsloth/Qwen3.8-27B-NVFP4, 3.62 GiB on Ornith-1.0-9B.
# On GB10 that copy is a large prefill win and declining it is a residency
# trade; on gfx1201 it is not a trade at all, because the twin arm
# w4a16_gemm_t_m128 measures ~1 TFLOP/s against ~4 for the plain w4a16_gemm it
# replaces. The 7-vs-51 TFLOP/s pair quoted for this lever elsewhere is a GB10
# measurement and was never measured on SCALE. Decode is untouched either way:
# it reads the packed original. `0` is also the loader's unset default under
# cfg(avarok_scale), so this export only makes the recipe explicit; `=1` builds
# them anyway and `=auto` restores the free-VRAM probe, which is how the A/B
# gets re-run. See docs/porting/r9700-residency.md.
case "$AVAROK_TARGET_HW" in
  r9700) export AVAROK_LOAD_TRANSPOSED_TWINS=0 ;;
esac
# Not exported: AVAROK_FORCE_GLOBAL_GDN. That name has no reader anywhere in
# this tree, so exporting it advertised a control that does not exist.

# SCALE libs FIRST so /opt/rocm cannot shadow the fixed libhsa-runtime64 (the
# gfx1151 queue-create fix lives in SCALE 1.7.1's bundled ROCm 7.2.3):
export LD_LIBRARY_PATH="$scale_target/lib:$SCALE_HOME/lib"
export PATH="$scale_target/bin:$PATH"

echo "serving $MODEL on $(/opt/rocm/bin/rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9]*' || echo "$arch")"
exec target/release/spark serve "$MODEL" \
  --port "${PORT:-8081}" --max-seq-len "${MAX_SEQ_LEN:-4096}" \
  --gpu-memory-utilization "${GPU_UTIL:-$default_gpu_util}" \
  --kv-cache-dtype bf16 --kv-high-precision-layers max \
  --max-batch-size "${MAX_BATCH:-$default_max_batch}" \
  --oom-guard-mb "${OOM_GUARD_MB:-$default_oom_guard_mb}"
