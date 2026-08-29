#!/usr/bin/env bash
# Serve a model with Atlas on AMD GPUs. Verified coherent on gfx1151 / Strix
# Halo with Qwen3.8-27B and Qwen3.6-27B. See docs/porting/amd-strix-halo-scale.md.
#
#   ./serve-amd.sh                                  # Qwen3.8-27B-NVFP4 (default)
#   ./serve-amd.sh nvidia/Qwen3.6-27B-NVFP4         # or any local snapshot path
#   PORT=9000 MAX_SEQ_LEN=32768 ./serve-amd.sh
#
# A binary built with ATLAS_TARGET_MODEL='*' (build-amd.sh's default) carries
# every strix kernel target, and resolution picks the right one from the
# checkpoint reference — so the same binary serves 3.6 and 3.8.
#
# Every flag and variable below is the one the measured configuration in
# ../40-bench/RESULTS.md and ../40-bench/BFCL.md actually ran with. If you
# change one, you are no longer running the configuration those numbers
# describe.
set -euo pipefail
cd "$(dirname "$0")"
MODEL="${1:-unsloth/Qwen3.8-27B-NVFP4}"
[ $# -gt 0 ] && shift            # anything left in "$@" is passed through to spark
HW="${ATLAS_TARGET_HW:-strix-hip}"

# ── gfx1151 runtime shims (each explained in docs §4) ────────────────────────
export ATLAS_W4A16_VARIANT=v1     # BF16-MMA NVFP4 GEMM (SCALE device FP8 encode is broken on gfx1151)
export ATLAS_W4A16_DP4A=1         # int8-DP4A decode GEMV
#
# NOT set, though every earlier Strix doc lists it: ATLAS_FORCE_GLOBAL_GDN=1.
# It has ZERO readers in crates/ on current main (only docs/ still mention it).
# It is unnecessary now because the strix kernel tree ships its own
# kernels/strix-hip/qwen3.6-27b/nvfp4/gated_delta_rule.cu as a model-specific
# override, already written for RDNA3.5's 64 KB LDS budget — the thing the
# lever used to force at dispatch time is now the only kernel there is.

# The GDN-projection prefill fast path (fp8_fp8_gemm_ldmab) is DEFAULT-ON on
# main and is NVIDIA-only: kernels/gb10/common/w4a16_fp8_ldmab.cu is built from
# `mma.sync.aligned.m16n8k32...e4m3.e4m3` + `ldmatrix.x4`, which have no RDNA3.5
# equivalent. The module is absent from the strix kernel set, and the lookup is
# a HARD runtime failure ("Module 'w4a16_fp8_ldmab' not loaded") on the first
# request, not a silent fallback. 0 takes the documented scalar path.
export ATLAS_FP8_LDMAB=0

# ── SSM / MTP levers, unchanged from the certified Qwen3.6 recipe ────────────
# Carried forward verbatim: these are what the 3.6 submission was measured
# under, and keeping them identical is what makes the 3.6-vs-3.8 comparison in
# ../40-bench/RESULTS.md apples-to-apples.
export ATLAS_SSM_TAIL_MIDCHUNK=1  # capture the mid-chunk SSM tail state
export ATLAS_MTP_GATE_REPROBE=64  # re-probe the MTP accept gate every 64 tokens
#
# Three more from the 3.6 recipe are deliberately dropped — all three are
# no-ops on current main, and two of them make the server print a warning:
#   ATLAS_SSM_TAIL_PROTECT=1   renamed 2026-08-05 to the opt-OUT
#                              ATLAS_DISABLE_SSM_TAIL_PROTECT; the lease is now
#                              on by default, so setting the old name does
#                              nothing and the behaviour is unchanged.
#   ATLAS_MTP_DRAFTER_PREFILL=1 } "OBSOLETE and IGNORED — MTP drafter prefill
#   ATLAS_MTP_CARRY_DRAFTER=1   } and cross-turn carry are ON by default"
#                              (spark_model::model::drafter_context). To turn
#                              them off you now set ATLAS_NO_MTP_DRAFTER_CONTEXT=1.

# ── memory: Strix Halo is a unified-memory part ──────────────────────────────
# The GPU allocates from the same RAM the OS uses. GTT reports 60 GB but the
# kernel will not hand over the last few — the measured allocatable ceiling is
# ~55 GB. These defaults are what actually serve a 27B here; raising
# GPU_UTIL or SSM_SLOTS pushes past the ceiling and fails with `cuMemAlloc_v2
# failed: status 2` AFTER the KV cache is already allocated.
GPU_UTIL="${GPU_UTIL:-0.86}"
SSM_SLOTS="${SSM_SLOTS:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-16384}"
# The kernel target now ships a vision_encoder, so the ~2.1 GB vision tower is
# loaded for every serve whether or not you send an image — it is part of the
# checkpoint. That does not fit alongside the default 8192-token prefill arena
# (3.63 GB); 2048 brings the arena to ~0.99 GB and buys back more than the
# tower costs. Measured on a 512-token single-stream run, this costs nothing:
# TTFT 817-1679 ms / decode 13.3-15.0 tok/s, the same range as the 8192 arena.
# Long prompts chunk more finely, so raise it if you have headroom and long
# inputs.
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-2048}"
# Atlas sizes its KV budget against the 60 GB the driver reports, not the ~55 GB
# the kernel will actually hand over. This tells it to hold 6 GB back.
export ATLAS_KV_EXTERNAL_RESERVE_GB="${ATLAS_KV_EXTERNAL_RESERVE_GB:-6}"

# Mixed-precision NVFP4 checkpoints (unsloth Qwen3.8-27B-NVFP4) keep 11.56 GB of
# tensors as FP8 inside an NVFP4 net. The loader requantises them and, with this
# flag set, RECLAIMS the FP8 sources (7.0 GB here) — without it the model does
# not fit at all on a 64 GB part.
#
# ⚠ TRADE-OFF: this also disables the GDN native-FP8 prefill precision policy,
# which keeps linear_attn qkvz/out_proj at >=FP8 rather than requantising them
# to NVFP4. It is a memory-for-accuracy trade, and it is REQUIRED for 3.8 here.
# A pure-NVFP4 checkpoint (nvidia/Qwen3.6-27B-NVFP4) has no FP8 tensors and does
# not need it — leave it unset for those.
# NOTE this flag is read by PRESENCE, not value
# (weight_map/nvfp4_detect.rs: `env::var_os(...).is_none()`), so exporting it as
# 0 would still turn it ON. Set ATLAS_NO_GDN_FP8_PREFILL=0 to genuinely disable
# it — this branch unsets the variable rather than passing a falsy value:
if [ "${ATLAS_NO_GDN_FP8_PREFILL:-1}" = "0" ]; then
  unset ATLAS_NO_GDN_FP8_PREFILL
else
  export ATLAS_NO_GDN_FP8_PREFILL=1
fi

if [ "$HW" = "strix-hip" ]; then
  # The HIP shims (libcuda/libcudart/libcublasLt) are built into atlas-kernels'
  # OUT_DIR; the loader needs them plus the AMD runtime at /opt/rocm/lib.
  SHIM=$(ls -dt target/release/build/atlas-kernels-*/out 2>/dev/null | head -1)
  export LD_LIBRARY_PATH="${SHIM:-}:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
else
  : "${SCALE_HOME:=$HOME/scale171/scale-1.7.1-Linux}"
  # SCALE libs FIRST so /opt/rocm cannot shadow the fixed libhsa-runtime64 (the
  # gfx1151 queue-create fix lives in SCALE 1.7.1's bundled ROCm 7.2.3):
  export LD_LIBRARY_PATH="$SCALE_HOME/targets/gfx1151/lib:$SCALE_HOME/lib"
  export PATH="$SCALE_HOME/targets/gfx1151/bin:$PATH"
fi

# The gfx1151 kernel set is 94 modules where gb10's is 167, so 92 dispatch sites
# resolve to a fallback and main's kernel audit (#388) refuses to serve without
# this flag. Those fallbacks are PRE-EXISTING — the audit landed on main after
# the Strix branch forked, so the certified 3.6 submission was produced under
# exactly the same ones. See ../30-verify/KERNEL_AUDIT.md before quoting any
# Strix perf number as final.
ALLOW_FALLBACKS="--dangerously-allow-unresolved-kernel-lookups"

# --model-name only matters when MODEL is a local snapshot path and you want the
# API to report the canonical repo id (as the benchmark configs expect).
NAME_ARG=(); [ -n "${MODEL_NAME:-}" ] && NAME_ARG=(--model-name "$MODEL_NAME")

echo "serving $MODEL on $(/opt/rocm/bin/rocminfo 2>/dev/null | grep -m1 -o gfx[0-9]* || echo AMD) via $HW"
exec target/release/spark serve "$MODEL" "${NAME_ARG[@]}" \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-8081}" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --kv-cache-dtype bf16 --max-batch-size "${MAX_BATCH:-1}" \
  --speculative --num-drafts "${NUM_DRAFTS:-2}" \
  --mtp-quantization bf16 --mtp-vocab 100000 \
  --disable-tool-grammar true --enable-prefix-caching \
  --ssm-cache-slots "$SSM_SLOTS" --ssm-checkpoint-interval 16 \
  $ALLOW_FALLBACKS \
  --disable-thinking \
  "$@"
