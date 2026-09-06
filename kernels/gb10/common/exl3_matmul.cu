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
// exl3_gemm_k{K}_cb{CB}_sh{S}_f32_abf16 / _f32_abf16_obf16 — the same launch
//   geometry with A a RAW BF16 [m, k] (converted in the prologue); the _obf16
//   twin takes two more args (C_bf16 BF16 dst, ld_bf16 row stride in elems,
//   >= n; must not alias C) and stores BF16(C) from the epilogue. Both are
//   byte-identical to the converter-bracketed f32 kernel.
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
//   exl3_f16_to_bf16_2d / exl3_f32_to_bf16_2d(in, out, rows, cols, ld_in,
//     ld_out) — STRIDED C egress for the dense-linear arm: row r, col c of a
//     contiguous-or-pitched fp16/fp32 C (leading dim ld_in elems) lands at
//     out[r*ld_out + c], so a native projection can write one column block
//     of a wider BF16 arena row (GDN's [Q|K|V|Z] row). Elements past `cols`
//     in each destination row are NOT touched. NEVER in place: source and
//     destination strides differ, so the parallel read/write footprints
//     overlap across rows (a contiguous source row r+1 begins inside
//     destination row r). block = 256, grid = min(ceil(rows*cols/256), 4096).
//
// exl3_silu_mul_f16(gate, up, out, act_limit, n2)  — MoE decode-tier
//   activation between the gate/up and down mgemm calls, mirroring upstream
//   ext.silu_mul (act_mul_kernel_h<ACT_SILU>) EXACTLY: out = silu_h2(gate) *
//   up over half2 lanes; act_limit != 0 clamps up to [-limit, limit] and
//   silu(gate) to (-inf, limit] first (qwen4_exp declares none — pass 0.0f).
//   All three buffers are fp16 SLOT buffers (bszm, 1, inter) — the mgemm
//   gate/up tier writes f16 C like upstream, so no f32-input variant is
//   needed (add one only if a caller switches gate/up to c_fp32). n2 =
//   numel/2 half2 elements; numel must be even (inter=640 per slot — always).
//   In-place allowed (out == gate or out == up), hence no __restrict__.
//   block = 256, grid = min(ceil(n2/256), 4096), grid-stride, plain launch.
//
// MoE decode-tier staging (plain launches, grid-stride; the two device-side
// preludes of the 3x-mgemm routed-expert pipeline — no D2H on the hot path):
//   exl3_moe_stage_routing(indices, probs, b_indices, b_weights,
//                          local_start, num_local, s)
//     Maps Atlas's device routing state (u32 GLOBAL expert ids + f32 probs,
//     [s = T*top_k]) to the mgemm arguments: b_indices[i] = LOCAL table
//     index (gid - local_start) for an EP-local expert, -1 for a remote one
//     (the canonical `exl3_expert_slot_index` mapping — the -1 is what makes
//     the mgemm weighted reduction skip the slot; NEVER encode remote
//     experts as null table entries), and b_weights[i] = f16(probs[i])
//     (kernel B_weights is half, matching upstream's fp16 routing weights).
//   exl3_moe_replicate_a_bf16(in, out, top_k, hidden, total)
//     BF16 [T, hidden] token activations -> fp16 A [T*top_k, hidden] with
//     slot s = t*top_k + j holding a COPY of token t's row (the mgemm
//     bszm_in > 1 layout; bszm_in == 1 broadcast only covers T == 1, and one
//     uniform path is simpler at <= 8 tokens). total = T*top_k*hidden.
//     Same fp16 saturation note as exl3_bf16_to_f16.
//
// MoE prefill-tier staging (plain launches, grid-stride; contracts at the
// definitions): exl3_moe_stage_sorted maps Atlas's moe_sort_by_expert outputs
// onto the fused exl3_moe kernel's token_sorted/weight_sorted/expert_count
// forms (local-expert order + EP sentinel tail bucket);
// exl3_moe_gather_rows_h16 / exl3_moe_scatter_add_f32 are the
// overflow-expert (count > 128) f16 row gather and weighted fp32 scatter-add
// around the chunked exl3_gemm tier; exl3_moe_store_slots_f32 and
// exl3_moe_reduce_slots_f32 are the DETERMINISTIC epilogue's replacement for
// that scatter-add (per-sorted-slot store + fixed-order reduce, shared with
// the fused kernel's `output_slots` arm — see moe_prefill_det.rs).

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

// BF16-activation GEMM twins (`_abf16`): the input-Hadamard prologue converts
// BF16 -> f16 while rotating (Atlas adaptation `A_BF16`, bit-identical to the
// standalone `exl3_bf16_to_f16` launch it replaces). f32 C only — the dense
// decode arm (m <= 8, K outside the GEMV tier) is the consumer; every other
// gemm/mgemm site keeps the f16 ingress.
#define EXL3_GEMM_WRAP_ABF16(K, CB, S, BLKDIM)                                \
    extern "C" __global__ void __launch_bounds__(BLKDIM)                      \
    exl3_gemm_k##K##_cb##CB##_sh##S##_f32_abf16(EXL3_GEMM_ARGS)               \
    {                                                                         \
        exl3_gemm_kernel_body<K, true, CB, EXL3_GEMM_SHAPE_##S, true>         \
            (A, B, C, size_m, size_k, size_n, locks, suh, A_had, svh);        \
    }

#define EXL3_GEMM_SET_ABF16(K, CB)                                            \
    EXL3_GEMM_WRAP_ABF16(K, CB, 2, 512)                                       \
    EXL3_GEMM_WRAP_ABF16(K, CB, 3, 512)                                       \
    EXL3_GEMM_WRAP_ABF16(K, CB, 4, 256)

EXL3_GEMM_SET_ABF16(2, 1)
EXL3_GEMM_SET_ABF16(2, 2)
EXL3_GEMM_SET_ABF16(3, 1)
EXL3_GEMM_SET_ABF16(3, 2)
EXL3_GEMM_SET_ABF16(4, 1)
EXL3_GEMM_SET_ABF16(4, 2)
EXL3_GEMM_SET_ABF16(5, 1)
EXL3_GEMM_SET_ABF16(5, 2)
EXL3_GEMM_SET_ABF16(6, 1)
EXL3_GEMM_SET_ABF16(6, 2)
EXL3_GEMM_SET_ABF16(8, 1)
EXL3_GEMM_SET_ABF16(8, 2)
// Shape 1 exists only for K in {2,4} (gemm) — mirror it for the BF16 twin.
EXL3_GEMM_WRAP_ABF16(2, 1, 1, 256)
EXL3_GEMM_WRAP_ABF16(2, 2, 1, 256)
EXL3_GEMM_WRAP_ABF16(4, 1, 1, 256)
EXL3_GEMM_WRAP_ABF16(4, 2, 1, 256)

// BF16-in / BF16-out GEMM twins (`_abf16_obf16`): the `_abf16` prologue plus
// the `OUT_BF16` epilogue — the output-Hadamard store also writes
// `__float2bfloat16_rn(c)` into `C_bf16[row * ld_bf16 + col]` (the
// `exl3_f32_to_bf16[_2d]` arithmetic on the same values, so byte-identical to
// convert-after). Two extra trailing arguments beyond EXL3_GEMM_ARGS; the
// f32 C is still written (split-K scratch). Consumer: the dense decode arm's
// m <= 8, K outside 2..=4 path (`exl3_gemm_abf16_obf16`), which then skips its
// egress launch. Kill switch on the host: ATLAS_EXL3_NO_FUSED_EGRESS.
#define EXL3_GEMM_WRAP_ABF16_OBF16(K, CB, S, BLKDIM)                          \
    extern "C" __global__ void __launch_bounds__(BLKDIM)                      \
    exl3_gemm_k##K##_cb##CB##_sh##S##_f32_abf16_obf16                         \
        (EXL3_GEMM_ARGS, __nv_bfloat16* __restrict__ C_bf16, const int ld_bf16) \
    {                                                                         \
        exl3_gemm_kernel_body<K, true, CB, EXL3_GEMM_SHAPE_##S, true, true>   \
            (A, B, C, size_m, size_k, size_n, locks, suh, A_had, svh,         \
             C_bf16, ld_bf16);                                                \
    }

#define EXL3_GEMM_SET_ABF16_OBF16(K, CB)                                      \
    EXL3_GEMM_WRAP_ABF16_OBF16(K, CB, 2, 512)                                 \
    EXL3_GEMM_WRAP_ABF16_OBF16(K, CB, 3, 512)                                 \
    EXL3_GEMM_WRAP_ABF16_OBF16(K, CB, 4, 256)

EXL3_GEMM_SET_ABF16_OBF16(2, 1)
EXL3_GEMM_SET_ABF16_OBF16(2, 2)
EXL3_GEMM_SET_ABF16_OBF16(3, 1)
EXL3_GEMM_SET_ABF16_OBF16(3, 2)
EXL3_GEMM_SET_ABF16_OBF16(4, 1)
EXL3_GEMM_SET_ABF16_OBF16(4, 2)
EXL3_GEMM_SET_ABF16_OBF16(5, 1)
EXL3_GEMM_SET_ABF16_OBF16(5, 2)
EXL3_GEMM_SET_ABF16_OBF16(6, 1)
EXL3_GEMM_SET_ABF16_OBF16(6, 2)
EXL3_GEMM_SET_ABF16_OBF16(8, 1)
EXL3_GEMM_SET_ABF16_OBF16(8, 2)
EXL3_GEMM_WRAP_ABF16_OBF16(2, 1, 1, 256)
EXL3_GEMM_WRAP_ABF16_OBF16(2, 2, 1, 256)
EXL3_GEMM_WRAP_ABF16_OBF16(4, 1, 1, 256)
EXL3_GEMM_WRAP_ABF16_OBF16(4, 2, 1, 256)

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

// Strided (2-D) egress: contract in the header. __restrict__ is correct
// here because in-place use is forbidden (overlapping footprints).
extern "C" __global__ void __launch_bounds__(256) exl3_f16_to_bf16_2d(
    const half* __restrict__ in, __nv_bfloat16* __restrict__ out,
    long long rows, long long cols, long long ld_in, long long ld_out)
{
    long long total = rows * cols;
    long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long) gridDim.x * blockDim.x;
    for (; i < total; i += stride)
    {
        long long r = i / cols;
        long long c = i - r * cols;
        out[r * ld_out + c] = __float2bfloat16_rn(__half2float(in[r * ld_in + c]));
    }
}

extern "C" __global__ void __launch_bounds__(256) exl3_f32_to_bf16_2d(
    const float* __restrict__ in, __nv_bfloat16* __restrict__ out,
    long long rows, long long cols, long long ld_in, long long ld_out)
{
    long long total = rows * cols;
    long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long) gridDim.x * blockDim.x;
    for (; i < total; i += stride)
    {
        long long r = i / cols;
        long long c = i - r * cols;
        out[r * ld_out + c] = __float2bfloat16_rn(in[r * ld_in + c]);
    }
}

// ── MoE decode-tier activation (contract in the header) ────────────────────
// Numerics verbatim from upstream activation_kernels.cuh
// act_mul_kernel_h<ACT_SILU> (fetched from master 2026-09-01): silu computed
// in half precision (h2exp/h2rcp), clamp in half, product in half. NO
// __restrict__: upstream documents in-place use (z == x or z == y).

extern "C" __global__ void __launch_bounds__(256) exl3_silu_mul_f16(
    const half* gate, const half* up, half* out, float act_limit, long long n2)
{
    long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long) gridDim.x * blockDim.x;
    for (; i < n2; i += stride)
    {
        half2 x2 = ((const half2*) gate)[i];
        half2 y2 = ((const half2*) up)[i];

        // _silu(half2) upstream
        half2 one = __float2half2_rn(1.0f);
        half2 neg_x = __hneg2(x2);
        half2 e = h2exp(neg_x);
        half2 sum = __hadd2(one, e);
        half2 r = h2rcp(sum);
        x2 = __hmul2(x2, r);

        if (act_limit != 0.0f)
        {
            y2 = __hmax2(y2, __float2half2_rn(-act_limit));
            y2 = __hmin2(y2, __float2half2_rn(act_limit));
            x2 = __hmin2(x2, __float2half2_rn(act_limit));
        }

        ((half2*) out)[i] = __hmul2(x2, y2);
    }
}

// ── MoE decode-tier staging (contracts in the header) ──────────────────────

extern "C" __global__ void __launch_bounds__(256) exl3_moe_stage_routing(
    const unsigned int* __restrict__ indices,  // [s] GLOBAL expert ids (u32)
    const float* __restrict__ probs,           // [s] routing probabilities
    int64_t* __restrict__ b_indices,           // [s] LOCAL ids, -1 = remote
    half* __restrict__ b_weights,              // [s] f16 per-slot weights
    int local_start,
    int num_local,
    long long s)
{
    long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long) gridDim.x * blockDim.x;
    for (; i < s; i += stride)
    {
        long long gid = (long long) indices[i];
        long long local = gid - (long long) local_start;
        b_indices[i] = (local >= 0 && local < (long long) num_local) ? local : (int64_t) -1;
        b_weights[i] = __float2half_rn(probs[i]);
    }
}

extern "C" __global__ void __launch_bounds__(256) exl3_moe_replicate_a_bf16(
    const __nv_bfloat16* __restrict__ in,  // [T, hidden] BF16
    half* __restrict__ out,                // [T*top_k, hidden] fp16
    int top_k,
    long long hidden,
    long long total)                       // = T*top_k*hidden
{
    long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long) gridDim.x * blockDim.x;
    for (; i < total; i += stride)
    {
        long long slot = i / hidden;
        long long col = i - slot * hidden;
        long long token = slot / top_k;
        out[i] = __float2half_rn(__bfloat162float(in[token * hidden + col]));
    }
}

// Fused ingress: independent staging outputs, identical casts and flat slot
// order to the two kernels above. The activation launch geometry is retained;
// routing has its own grid-stride loop so small hidden widths remain valid.
extern "C" __global__ void __launch_bounds__(256) exl3_moe_stage_ingress(
    const __nv_bfloat16* __restrict__ in,
    const unsigned int* __restrict__ indices,
    const float* __restrict__ probs,
    int64_t* __restrict__ b_indices,
    half* __restrict__ b_weights,
    half* __restrict__ out,
    int local_start,
    int num_local,
    int top_k,
    long long hidden,
    long long s,
    long long total)
{
    long long first = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long) gridDim.x * blockDim.x;
    for (long long i = first; i < s; i += stride)
    {
        long long gid = (long long) indices[i];
        long long local = gid - (long long) local_start;
        b_indices[i] = (local >= 0 && local < (long long) num_local) ? local : (int64_t) -1;
        b_weights[i] = __float2half_rn(probs[i]);
    }
    for (long long i = first; i < total; i += stride)
    {
        long long slot = i / hidden;
        long long col = i - slot * hidden;
        long long token = slot / top_k;
        out[i] = __float2half_rn(__bfloat162float(in[token * hidden + col]));
    }
}

// ── MoE prefill-tier staging (contracts in the header) ─────────────────────
//
// exl3_moe_stage_sorted: map Atlas's sort outputs (moe_sort_by_expert:
// contiguous spans per GLOBAL expert ascending, expert_offsets [ne+1] i32
// prefix sums, token_to_perm [T*top_k] i32 flat-slot -> sorted position) onto
// the fused exl3_moe kernel's contract (token_sorted/weight_sorted i64/f16
// ordered by LOCAL expert with EP-remote slots parked in a sentinel tail
// bucket, expert_count i64 [num_local+1] bincount, count[num_local] = the
// sentinel). The EP-local range is one CONTIGUOUS run of global ids, so
// sorted positions [lo, hi) (lo = offsets[local_start], hi =
// offsets[local_start + num_local]) are exactly the local slots, already
// grouped by local expert ascending — a rotation of [0, hi) by lo puts them
// first and every remote slot lands in the tail:
//     p in [lo, hi)  ->  dst = p - lo          (local, bucket order kept)
//     p <  lo        ->  dst = p + (hi - lo)   (remote-before -> tail)
//     p >= hi        ->  dst = p               (remote-after already tail)
// Non-EP degenerates to the identity (lo = 0, hi = s, empty sentinel).
extern "C" __global__ void __launch_bounds__(256) exl3_moe_stage_sorted(
    const int* __restrict__ token_to_perm,   // [s] flat slot -> sorted pos
    const float* __restrict__ probs,         // [s] routing probs (flat order)
    const int* __restrict__ expert_offsets,  // [num_experts_global + 1]
    int64_t* __restrict__ token_sorted,      // [s] out: token idx per slot
    half* __restrict__ weight_sorted,        // [s] out: f16 weight per slot
    int64_t* __restrict__ expert_count,      // [num_local + 1] out
    int local_start,
    int num_local,
    int top_k,
    long long s)                             // = T * top_k
{
    const long long lo = (long long) expert_offsets[local_start];
    const long long hi = (long long) expert_offsets[local_start + num_local];
    long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long) gridDim.x * blockDim.x;
    for (long long j = i; j < s; j += stride)
    {
        long long p = (long long) token_to_perm[j];
        long long dst = (p >= lo && p < hi) ? (p - lo)
                      : (p < lo)            ? (p + (hi - lo))
                                            : p;
        token_sorted[dst] = j / top_k;
        weight_sorted[dst] = __float2half_rn(probs[j]);
    }
    for (long long j = i; j <= (long long) num_local; j += stride)
    {
        expert_count[j] = (j < (long long) num_local)
            ? (int64_t) (expert_offsets[local_start + j + 1]
                         - expert_offsets[local_start + j])
            : (int64_t) (s - (hi - lo));  // sentinel = every remote slot
    }
}

// Overflow-expert gather: one expert's sorted-span rows of the token-major
// BF16 activations into a contiguous [m, hidden] BF16 A for the reconstruct
// + dense-GEMM overflow path (token_count > max_tokens_per_expert experts the
// fused kernel skips). Caller offsets token_sorted to the span base.
extern "C" __global__ void __launch_bounds__(256) exl3_moe_gather_rows_h16(
    const uint16_t* __restrict__ in,           // [T, hidden], any 16-bit dtype
    const int64_t* __restrict__ token_sorted,  // span base, [m]
    uint16_t* __restrict__ out,                // [m, hidden]
    long long hidden,
    long long total)                           // = m * hidden
{
    long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long) gridDim.x * blockDim.x;
    for (; i < total; i += stride)
    {
        long long r = i / hidden;
        long long col = i - r * hidden;
        out[i] = in[token_sorted[r] * hidden + col];
    }
}

// Overflow-expert epilogue: weighted scatter-add of the expert's fp32 down
// GEMM output into the fp32 routed accumulator (the fused kernel's
// output_state), mirroring its atomicAdd epilogue numerically (f16 routing
// weight widened to f32, product in f32) AND atomically: real top-k routing
// gives a token at most one slot per expert, but nothing in this kernel's
// contract should hinge on that — atomicAdd also keeps degenerate
// duplicate-expert routings additive, exactly like the fused kernel's
// epilogue on the same buffer.
extern "C" __global__ void __launch_bounds__(256) exl3_moe_scatter_add_f32(
    const float* __restrict__ down,             // [m, hidden] fp32 GEMM C
    const int64_t* __restrict__ token_sorted,   // span base, [m]
    const half* __restrict__ weight_sorted,     // span base, [m]
    float* __restrict__ out,                    // [T, hidden] fp32 accumulator
    long long hidden,
    long long total)                            // = m * hidden
{
    long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long) gridDim.x * blockDim.x;
    for (; i < total; i += stride)
    {
        long long r = i / hidden;
        long long col = i - r * hidden;
        atomicAdd(out + token_sorted[r] * hidden + col,
                  __half2float(weight_sorted[r]) * down[i]);
    }
}

// Overflow-expert epilogue, DETERMINISTIC arm: the same weighted rows written
// to their OWN sorted slots instead of atomically accumulated into the shared
// per-token row. The caller offsets `out_slots` to the chunk's slot base, so
// this is a pure elementwise scale-and-store — the >128-rows-per-expert tier
// carries exactly the same defect as the fused kernel's atomicAdd epilogue
// (unordered fp32 accumulation into one row), and it is the tier that fires
// on LONG prefills, so fixing only the fused tier would leave long-context
// serving nondeterministic. Numerically identical to the atomic arm: f16
// routing weight widened to f32, product in f32.
extern "C" __global__ void __launch_bounds__(256) exl3_moe_store_slots_f32(
    const float* __restrict__ down,             // [m, hidden] fp32 GEMM C
    const half* __restrict__ weight_sorted,     // span base, [m]
    float* __restrict__ out_slots,              // slot base, [m, hidden]
    long long hidden,
    long long total)                            // = m * hidden
{
    long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long) gridDim.x * blockDim.x;
    for (; i < total; i += stride)
    {
        long long r = i / hidden;
        out_slots[i] = __half2float(weight_sorted[r]) * down[i];
    }
}

// Fixed-order reduction of the per-slot rows into the fp32 routed accumulator
// — the second half of the deterministic prefill epilogue, and the contract
// the DECODE tier already meets by construction (its mgemm reduces a token's
// expert slots in a fixed j = 0..stride-1 loop).
//
// One thread owns one (token, column) output element and adds that token's
// top_k slots in ASCENDING FLAT-SLOT ORDER k = 0..top_k-1 — the routing order
// the router emitted, identical every run — so the fp32 sum is bit-
// reproducible no matter how the expert groups were scheduled. The flat slot
// (token*top_k + k) is mapped to its LOCAL-sorted position by the SAME
// rotation `exl3_moe_stage_sorted` used to lay the slots out:
//     p = token_to_perm[flat];  lo = offsets[local_start];
//     hi = offsets[local_start + num_local];  nloc = hi - lo
//     p in [lo, hi) -> p - lo   |   p < lo -> p + nloc   |   p >= hi -> p
// and a slot whose mapped position is >= nloc is EP-REMOTE (the sentinel tail
// the fused kernel never processes), so it is skipped — a token whose experts
// are all remote yields an exact 0.0 row, the EP partial-sum convention.
//
// No zero-init of `slots` is required or performed: every local slot is
// written exactly once per call by the fused kernel or the overflow tier, and
// no remote slot is ever read. `out` is fully overwritten, so its memset is
// skipped on this arm too.
extern "C" __global__ void __launch_bounds__(256) exl3_moe_reduce_slots_f32(
    const float* __restrict__ slots,            // [s, hidden] per-slot rows
    const int* __restrict__ token_to_perm,      // [s] flat slot -> sorted pos
    const int* __restrict__ expert_offsets,     // [num_experts_global + 1]
    float* __restrict__ out,                    // [T, hidden] fp32 accumulator
    int local_start,
    int num_local,
    int top_k,
    long long hidden,
    long long total)                            // = T * hidden
{
    const long long lo = (long long) expert_offsets[local_start];
    const long long hi = (long long) expert_offsets[local_start + num_local];
    const long long nloc = hi - lo;
    long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long) gridDim.x * blockDim.x;
    for (; i < total; i += stride)
    {
        long long tok = i / hidden;
        long long col = i - tok * hidden;
        float acc = 0.0f;
        for (int k = 0; k < top_k; ++k)
        {
            long long p = (long long) token_to_perm[tok * top_k + k];
            long long dst = (p >= lo && p < hi) ? (p - lo)
                          : (p < lo)            ? (p + nloc)
                                                : p;
            if (dst < nloc) acc += slots[dst * hidden + col];
        }
        out[i] = acc;
    }
}
