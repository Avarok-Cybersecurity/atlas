// SPDX-License-Identifier: AGPL-3.0-only
//
// EXL3 (QTIP trellis-quantized) native matmul: fused-rotation GEMM / mgemm /
// small-m GEMV over packed trellis codes, no weight reconstruction.
//
//     C = ( ((A .* suh) H128/sqrt128) @ W_hat ) H128/sqrt128 .* svh
//
// The device code is vendored VERBATIM from turboderp's ExLlamaV3 (MIT):
//   https://github.com/turboderp-org/exllamav3
//   Copyright (c) 2025 turboderp — MIT license; snapshots in
//   .research/exllamav3_ref/, adapted headers in exl3_vendor/ (each file
//   documents its deltas). This file only adds extern "C" __global__ wrappers
//   so the Rust host can select instances by name, plus BF16<->FP16/FP32
//   boundary converters (Atlas serves BF16; the EXL3 kernels are fp16-native).
//
// ── Format recap ───────────────────────────────────────────────────────────
//  * B (trellis): int16 [k/16, n/16, 16*K] — 16x16 tiles, K bits/weight,
//    procedural codebook cb: 0 = 3INST, 1 = MCG, 2 = MUL1 (qwen4_exp ships
//    MUL1; MCG kept for older checkpoints; 3INST not instantiated here).
//  * suh: f16 [k], svh: f16 [n] — expanded Hadamard sign vectors. MANDATORY,
//    as is A_had (the kernels dereference all three unconditionally).
//  * A: RAW (un-rotated) fp16 [m, k] row-major contiguous. A_had: fp16
//    scratch, >= m*k elems for gemm/gemv (may alias A), >= bszm*m*k for
//    mgemm (undersized = silent OOB). C: fp16 (_f16) or fp32 (_f32) [m, n];
//    needs no zeroing. locks: int32 device buffer of
//    (MAX_TILES_C + 2*MAX_BARRIERS + MOE_SCHED_INTS) = 1,050,690 ints
//    (4,202,760 bytes), zeroed ONCE at allocation — every protocol
//    self-resets, never re-zero between calls.
//
// ── Launch contracts (host side, next stage) ───────────────────────────────
// ALL matmul entries are COOPERATIVE launches (cuLaunchCooperativeKernel);
// grid.sync() deadlocks or faults under a plain launch.
//
// exl3_gemm_k{K}_cb{CB}_sh{S}_{f16|f32}   (K in 2,3,4,5,6,8; CB in 1,2)
//   shapes S: 1=(TK16,TN128) 2=(TK32,TN128) 3=(TK32,TN256) 4=(TK16,TN512)
//   block = (256*TK/16) threads: sh1->256, sh2->512, sh3->512, sh4->256
//   grid  = (num_sms, 1, 1), num_sms = max(min(k/TK * n/TN, SM_count), 1)
//   dynamic smem = 90*1024 B — the host MUST first raise
//   CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES to 92160 (once per
//   function per device) or the launch fails.
//   constraints: k % TK == 0, n % TN == 0; shape heuristic + compat rules in
//   exl3_vendor/exl3_kernel_map.cuh. Shape 1 is only ever picked by the
//   Blackwell heuristic for K in {2,4}, k <= 2048, single-matrix, so it is
//   instantiated only for those K.
//
// exl3_mgemm_k{K}_cb{CB}_sh{S}_{f16|f32}  (pointer-table multi-matrix / MoE;
//   S in 2,3,4 — the heuristic never picks shape 1 for multi)
//   args: EXL3_MGEMM_ARGS (19 slots; B_list/suh_list/svh_list are device
//   arrays of device pointers; <= 128 slots when index-filtering)
//   block as above; dynamic smem 90 KB (same attribute raise)
//   grid  = (num_sms, 1, concurrency), computed exactly as upstream:
//     tiles = max(k/TK * n/TN, 1); num_sms = tiles;
//     if (num_sms * bszm > total_sms) num_sms = max(total_sms / bszm, 1);
//     if (num_sms <= total_sms && tiles / num_sms > 48)
//         num_sms = min(total_sms, num_sms * 2);
//     concurrency = min(total_sms / num_sms, bszm);   // bszm = max(in, out)
//   HAZARD: the filtered path (min_index >= 0) uses module-scope __device__
//   globals (v_indices/v_weights/bszm_sync) — serialize filtered mgemm
//   launches per device.
//
// exl3_gemv_k{K}_cb{CB}_m{MMODE}_cfg{CFG}_{f16|f32}  (K in 2,3,4; CB in 1,2)
//   MMODE 0 = size_m == 1 fast path; MMODE 1 = 2 <= size_m <= 8.
//   CFG 0 "narrow": block 512, COLS 32; CFG 1 "wide": block 256, COLS 64.
//   grid  = (min(n/COLS, occupancy_blocks_per_sm * SM_count), 1, 1) — cap by
//   cuOccupancyMaxActiveBlocksPerMultiprocessor(func, block, 0), cached per
//   function; refuse the path if grid < 1. Dynamic smem = 0 (static only).
//   constraints: m <= 8, k % 128 == 0, n % 128 == 0. locks is unused (pass
//   the shared buffer anyway for signature parity).
//   Upstream cfg heuristic (Blackwell falls through, arch gate commented out
//   upstream): K==2 -> n<=8192 ? narrow : wide; else narrow if n/32 fits one
//   co-resident wave or (k<=2048 && n<=8192); K==3 else ineligible; K==4 wide
//   if (n>=8192 && k<=4096); else ineligible -> use the GEMM.
//
// Converters (plain launches, grid-stride):
//   exl3_bf16_to_f16(in, out, n)  — activation ingress before A. NOTE: values
//     with |v| > 65504 saturate to +-inf in fp16; safe for post-norm
//     activations, do not feed raw residuals/logits.
//   exl3_f16_to_bf16(in, out, n)  — C readback (f16 C).
//   exl3_f32_to_bf16(in, out, n)  — C readback (f32 C, preferred: upgrades
//     split-k partials and the MoE reduction to fp32).
//   block = 256, grid = min(ceil(n/256), 4096). Scalar loads: milestone-1
//   cost is one extra read+write of A (and of C) per matmul, ~2 elementwise
//   passes; fold into the Hadamard prologue later if it shows in profiles.

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "exl3_vendor/exl3_gemm_kernel.cuh"
#include "exl3_vendor/exl3_gemv_kernel.cuh"

// ── extern "C" wrapper instantiation ───────────────────────────────────────
// Symbol grammar encodes the full template selection:
//   K (bits/weight), cb (codebook), sh (tile shape) / m+cfg (gemv mode),
//   f16/f32 (C dtype — c_fp32 template flag; MMA accumulation is fp32 on
//   sm_121a either way).

#define EXL3_GEMM_WRAP(K, CB, S, BLKDIM, SUF, CFP32)                          \
    extern "C" __global__ void __launch_bounds__(BLKDIM)                      \
    exl3_gemm_k##K##_cb##CB##_sh##S##_##SUF(EXL3_GEMM_ARGS)                   \
    {                                                                         \
        exl3_gemm_kernel_body<K, CFP32, CB, EXL3_GEMM_SHAPE_##S>              \
            (A, B, C, size_m, size_k, size_n, locks, suh, A_had, svh);        \
    }

#define EXL3_MGEMM_WRAP(K, CB, S, BLKDIM, SUF, CFP32)                         \
    extern "C" __global__ void __launch_bounds__(BLKDIM)                      \
    exl3_mgemm_k##K##_cb##CB##_sh##S##_##SUF(EXL3_MGEMM_ARGS)                 \
    {                                                                         \
        exl3_mgemm_kernel_body<K, CFP32, CB, EXL3_GEMM_SHAPE_##S>             \
            (A, B_list, C, size_m, size_k, size_n, locks, suh_list, A_had,    \
             svh_list, B_indices, B_weights, bszm_in, bszm_out, min_index,    \
             max_index, num_tokens, size_n_list, C_list);                     \
    }

#define EXL3_GEMV_WRAP(K, CB, MM, CFG, BLKDIM, SUF, CFP32)                    \
    extern "C" __global__ void __launch_bounds__(BLKDIM)                      \
    exl3_gemv_k##K##_cb##CB##_m##MM##_cfg##CFG##_##SUF(EXL3_GEMM_ARGS)        \
    {                                                                         \
        exl3_gemv_kernel_body<K, CFP32, CB, MM, CFG, false>                   \
            (A, B, C, size_m, size_k, size_n, locks, suh, A_had, svh);        \
    }

#define EXL3_GEMM_WRAP_FP(K, CB, S, BLKDIM)                                   \
    EXL3_GEMM_WRAP(K, CB, S, BLKDIM, f16, false)                              \
    EXL3_GEMM_WRAP(K, CB, S, BLKDIM, f32, true)

#define EXL3_MGEMM_WRAP_FP(K, CB, S, BLKDIM)                                  \
    EXL3_MGEMM_WRAP(K, CB, S, BLKDIM, f16, false)                             \
    EXL3_MGEMM_WRAP(K, CB, S, BLKDIM, f32, true)

// Shapes 2/3/4 for every served K/cb (gemm + mgemm)
#define EXL3_GEMM_SET(K, CB)                                                  \
    EXL3_GEMM_WRAP_FP(K, CB, 2, 512)                                          \
    EXL3_GEMM_WRAP_FP(K, CB, 3, 512)                                          \
    EXL3_GEMM_WRAP_FP(K, CB, 4, 256)                                          \
    EXL3_MGEMM_WRAP_FP(K, CB, 2, 512)                                         \
    EXL3_MGEMM_WRAP_FP(K, CB, 3, 512)                                         \
    EXL3_MGEMM_WRAP_FP(K, CB, 4, 256)

EXL3_GEMM_SET(2, 1)
EXL3_GEMM_SET(2, 2)
EXL3_GEMM_SET(3, 1)
EXL3_GEMM_SET(3, 2)
EXL3_GEMM_SET(4, 1)
EXL3_GEMM_SET(4, 2)
EXL3_GEMM_SET(5, 1)
EXL3_GEMM_SET(5, 2)
EXL3_GEMM_SET(6, 1)
EXL3_GEMM_SET(6, 2)
EXL3_GEMM_SET(8, 1)
EXL3_GEMM_SET(8, 2)

// Shape 1: Blackwell heuristic picks it only for K in {2,4}, k <= 2048,
// single-matrix — gemm only
EXL3_GEMM_WRAP_FP(2, 1, 1, 256)
EXL3_GEMM_WRAP_FP(2, 2, 1, 256)
EXL3_GEMM_WRAP_FP(4, 1, 1, 256)
EXL3_GEMM_WRAP_FP(4, 2, 1, 256)

// GEMV: MMODE {0,1} x CFG {0,1} x C dtype, per (K, cb).
// SMEM_STAGE fixed false (lane-shuffle extraction — the upstream default;
// the smem-staging variant exists only for A/B evaluation).
#define EXL3_GEMV_SET(K, CB)                                                  \
    EXL3_GEMV_WRAP(K, CB, 0, 0, 512, f16, false)                              \
    EXL3_GEMV_WRAP(K, CB, 0, 0, 512, f32, true)                               \
    EXL3_GEMV_WRAP(K, CB, 0, 1, 256, f16, false)                              \
    EXL3_GEMV_WRAP(K, CB, 0, 1, 256, f32, true)                               \
    EXL3_GEMV_WRAP(K, CB, 1, 0, 512, f16, false)                              \
    EXL3_GEMV_WRAP(K, CB, 1, 0, 512, f32, true)                               \
    EXL3_GEMV_WRAP(K, CB, 1, 1, 256, f16, false)                              \
    EXL3_GEMV_WRAP(K, CB, 1, 1, 256, f32, true)

EXL3_GEMV_SET(2, 1)
EXL3_GEMV_SET(2, 2)
EXL3_GEMV_SET(3, 1)
EXL3_GEMV_SET(3, 2)
EXL3_GEMV_SET(4, 1)
EXL3_GEMV_SET(4, 2)

// ── dtype boundary converters ──────────────────────────────────────────────

extern "C" __global__ void __launch_bounds__(256) exl3_bf16_to_f16(
    const __nv_bfloat16* __restrict__ in, half* __restrict__ out, long long n)
{
    long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long) gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        out[i] = __float2half_rn(__bfloat162float(in[i]));
}

// NO __restrict__ here: the lm_head caller converts IN PLACE (in == out,
// each index read once then written once) — aliased __restrict__ pointers
// are formally UB even when the access pattern is safe.
extern "C" __global__ void __launch_bounds__(256) exl3_f16_to_bf16(
    const half* in, __nv_bfloat16* out, long long n)
{
    long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long) gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        out[i] = __float2bfloat16_rn(__half2float(in[i]));
}

extern "C" __global__ void __launch_bounds__(256) exl3_f32_to_bf16(
    const float* __restrict__ in, __nv_bfloat16* __restrict__ out, long long n)
{
    long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long) gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        out[i] = __float2bfloat16_rn(in[i]);
}
