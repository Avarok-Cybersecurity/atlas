#!/usr/bin/env bash
# Strix Halo (gfx1151 / RDNA3.5) GOLDEN e2e serve config — 2026-07-19.
#
# Model:   nvidia/Qwen3.6-27B-NVFP4  (compressed-tensors mixed-precision NVFP4)
# Stack:   branch strix-accfix2 @ 2af4df6 (session-gate + midchunk tail capture +
#          #287 recency-only eviction + native-FP8-GDN loader + dense_gemv_bf16_batch2
#          port + origin/main merge). Build with ATLAS_TARGET_HW=strix-hip.
#
# Key env:
#   ATLAS_NO_GDN_FP8=0          keep GDN in_proj_qkv/out_proj E4M3 NATIVE (run w8a16 GEMM)
#                              — NO lossy FP8->BF16->NVFP4 double-quant. This is the
#                              accuracy fix that recovers BFCL non_live (~76 -> ~85).
#                              Loads ~37 GB pre-KV (fits 32k).
#   ATLAS_SSM_TAIL_MIDCHUNK=1   mid-chunk tail SSM capture -> warm-TTFT parity with
#                              llama.cpp (~2050 ms vs 2211 ms, 0 replay, byte-exact).
#   ATLAS_W4A16_DP4A=1          W4A8 integer-DP4A decode (v_dot4/sudot4) -> ~17 tok/s.
#   kv-cache-dtype bf16         fp8 KV garbles at ~10k depth on gfx1151; bf16 for deep
#                              multi-turn coherence. (fp8 KV is a separate memory-win
#                              variant for short-context / MLPerf-edge.)
#
# Binary: /workspace/hip-target-real/release/spark  (built from 2af4df6, ATLAS_TARGET_HW=strix-hip)
# Port:   8081
#
# Box-ops (60 GB unified): STOP any prior serve and WAIT for ~40 GB free before launching
# (overlapping restarts = 2x40 > 60 -> OOM-wedge). util 0.92 is required for the 39 GB nvidia
# ckpt; never run a heavy BFCL client concurrent with a calibrating/fresh serve.
export LD_LIBRARY_PATH=/home/azeez/hip-port/link:/opt/rocm/core-7.13/lib:/opt/rocm/lib
export ATLAS_W4A16_DP4A=1 ATLAS_FORCE_GLOBAL_GDN=1 ATLAS_W4A16_VARIANT=v1
export ATLAS_NO_GDN_FP8=0 ATLAS_SSM_TAIL_MIDCHUNK=1 ATLAS_SSM_TAIL_PROTECT=1
SNAP=/home/azeez/.cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/snapshots/0893e1606ff3d5f97a441f405d5fc541a6bdf404
exec /workspace/hip-target-real/release/spark serve "$SNAP" \
  --model-name nvidia/Qwen3.6-27B-NVFP4 --host 0.0.0.0 --port 8081 \
  --max-seq-len 32768 --gpu-memory-utilization 0.92 \
  --kv-cache-dtype bf16 --max-batch-size 1 \
  --speculative --num-drafts 1 --mtp-quantization bf16 --mtp-vocab 100000 \
  --disable-tool-grammar true --enable-prefix-caching \
  --ssm-cache-slots 32 --ssm-checkpoint-interval 16 --disable-thinking
