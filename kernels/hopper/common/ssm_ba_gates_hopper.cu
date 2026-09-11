// SPDX-License-Identifier: AGPL-3.0-only

// Hopper twin of `ssm_preprocess.cu`'s `dense_gemm_ba_gates_prefill` (#928).
//
// WHAT THE PARENT DOES, and why it costs what it costs. The parent launches
//   Grid: (ceil(N/4), M_tokens, 1)  Block: (256, 1, 1)
// so one 256-thread CTA computes FOUR of the N BA outputs for ONE token, and
// EACH of those four outputs walks the token's whole K-element activation row:
// the CTA's 256 threads are 4 groups of 64 lanes, one group per output, and
// every group sweeps all of K. With the Qwen3.8-27B geometry (N = ssm_ba_size
// = 2*nv = 96, K = hidden = 5120) that is 24 CTAs per token and **96 reads of
// each token's activation row** — one per BA output — plus 96x the bf16->f32
// conversion of A's elements.
//
// (`h100-r13-attribution.md` SS A.4 states the amplification as 24x, counting
// CTAs. The per-CTA factor of 4 is the rest of it; the issued-traffic figure
// there is correspondingly 4x low. Neither changes that ranking's ORDER.)
//
// nsys, 1xH100 80GB HBM3, Qwen/Qwen3.8-27B-FP8, round 13 cell T1 (2026-09-11,
// `h100-r13-attribution.md` SS A.3/A.4): `dense_gemm_ba_gates_prefill` is
// 26 881.8 us = 5.85% of a 4593-token prefill (6 896.3 us = 3.13% of the
// 1168-token forward), 555.04 us per launch on the M=4576 chunk, with the
// observed grid (24, 4576). Compulsory traffic at that rate is 88 GB/s, 2.6%
// of HBM — this kernel has never been anywhere near a bandwidth bound.
//
// WHAT THIS TWIN DOES. One CTA per token. The same 256 threads, the same
// 4-outputs-by-64-lanes split, but a thread keeps its lane and sweeps the
// output groups in a register tile of BAH_GROUPS at a time, so one fetch and
// one conversion of the row serves BAH_GROUPS outputs instead of one: 12 reads
// of the row per token instead of 96. B, the BA weight, is the same traffic it
// always was — it is shared across tokens and served from L2.
//
// IT IS NOT A BANDWIDTH FIX, because there was no bandwidth problem. At
// `--fmad=false` (kernels/gb10/common/KERNEL.toml, inherited here) every
// multiply-accumulate is a separate `mul.f32` and `add.f32`, so the floor is
// INSTRUCTION ISSUE, and 3 instructions per element on the B side (one
// bf16->f32 convert, one multiply, one add) cannot be removed without changing
// the arithmetic. SASS, sm_90a, CUDA 13.0.88, `--fmad=false -O3`: the parent's
// inner loop is 175 instructions per 32 element-MACs (5.47/MAC) and this one's
// is 268 per 64 (4.19/MAC); counting each kernel's prologue and reduction tail
// as well, the launch issues ~1.8x fewer instructions for the same answer.
// That ratio, not a roofline, is what this file is worth.
// See `SSM-BA-GATES-ATTRIBUTION.md`.
//
// BIT-IDENTICAL, BY CONSTRUCTION, and that is the whole contract. The BA
// outputs feed `A_log`/`dt_bias` straight into the GDN recurrence as the
// per-head decay `gate` and write gate `beta`; a one-ulp difference there
// compounds over 48 GDN layers and every chunk of the scan. So this file
// reproduces the parent's reduction ORDER exactly:
//
//   1. per lane, `for (kv = lane; kv < K/8; kv += 64)` over uint4 vectors, and
//      inside each uint4 the eight bf16 in index order (lo, hi per 32-bit
//      word, words 0..3);
//   2. the 5-step `__shfl_down_sync` butterfly at offsets 16, 8, 4, 2, 1;
//   3. the cross-warp sum as `warp_even + warp_odd`, in that order;
//   4. the identical sigmoid / softplus / exp transforms and output indices.
//
// `__bfloat162float` is an EXACT widening (bf16 is a prefix of f32), so
// hoisting A's conversion out of the output loop cannot change a bit; the
// operands the mul sees are the same floats in the same order.
// `native_ssm_ba_gates_hopper_microtest` is the gate and it asserts byte
// equality, not a tolerance.

#include <cuda_bf16.h>
#include <math.h>

// The parent's block shape, restated as the contract it is: 256 threads split
// into BAH_OUTS outputs of BAH_LANES lanes each. Changing any of the three
// changes the reduction order and breaks the bit-identity contract above.
#define BAH_BLOCK 256
#define BAH_LANES 64
#define BAH_OUTS (BAH_BLOCK / BAH_LANES)
#define BAH_WARPS (BAH_BLOCK / 32)

// Output GROUPS (of BAH_OUTS outputs) a thread accumulates at once.
//
// The A-reuse / register-pressure knob, and MEASURED rather than picked: one
// A fetch serves this many outputs, so the whole cost model is the inner
// loop's instructions per element-MAC against the occupancy the register file
// allows. ptxas, sm_90a, CUDA 13.0.88, `-O3 --fmad=false`:
//
//   BAH_GROUPS |  4    |  8    | 12        | 16
//   registers  | 47    | 64    | 64 +spill | 80
//   CTAs/SM    |  5    |  4    |  -        |  3
//   inst/MAC   | 4.69  | 4.19  |  -        | 4.05
//
// 8 is the knee: 12 spills (16 B stack frame, which is a global-memory round
// trip in the hot loop), and 16 buys 3.3% of instructions for a quarter of the
// occupancy. 64 registers is exactly 1024 threads per SM.
#define BAH_GROUPS 8

// TODO(#928): the remaining instruction floor is the B-side convert+mul+add.
// Folding the N=96 projection into the SSM `in_proj_qkvz` cuBLASLt GEMM
// (N 16384 -> 16480) removes it entirely, but quantises BA to block-scaled FP8
// and so changes the numbers the GDN recurrence sees. That is a model-quality
// question, not a kernel one — it needs its own accuracy receipt against this
// bf16 path and must not be the default until it has one.

/// One CTA per token; all N BA outputs, the parent's arithmetic, in order.
///
/// Output layout is the parent's: `[gate(nv), beta(nv)]` per token, stride
/// `gate_stride`.
///
///   gate_out[token * gate_stride + vh]      = gate  (alpha -> exp transform)
///   gate_out[token * gate_stride + nv + vh] = beta  (sigmoid)
///
/// Grid: (M_tokens, 1, 1)  Block: (256, 1, 1)  Shared: 256 B, static.
///
/// Preconditions the LAUNCHER enforces (`ops::ssm_ba_gates_hopper`), because
/// violating one of them produces wrong numbers rather than a slow kernel:
/// `K % 8 == 0` (the uint4 sweep), `K_stride >= K`, and a token count large
/// enough to fill the device — at one CTA per token a 16-row decode step would
/// occupy 16 SMs of 132.
extern "C" __global__ __launch_bounds__(BAH_BLOCK) void dense_gemm_ba_gates_prefill_hopper(
    const __nv_bfloat16* __restrict__ A,  // [M, K_stride] activations
    const __nv_bfloat16* __restrict__ B,  // [N, K] BA weight (row-major)
    const float* __restrict__ A_log,      // [nv] learned A_log parameter
    const float* __restrict__ dt_bias,    // [nv] learned dt_bias parameter
    float* __restrict__ gate_out,         // [M, gate_stride] FP32 output
    unsigned int M,                       // num_tokens
    unsigned int N,                       // ssm_ba_size (2 * nv)
    unsigned int K,                       // hidden_size
    unsigned int K_stride,                // BF16 elements per token in A
    unsigned int gate_stride,             // FP32 elements per token in gate_out
    unsigned int nv,                      // num_v_heads
    unsigned int vheads_per_group
) {
    const unsigned int token = blockIdx.x;
    if (token >= M) return;

    // Same decomposition as the parent, so the lane -> k mapping is the same.
    const unsigned int local_out = threadIdx.x / BAH_LANES;  // 0..BAH_OUTS-1
    const unsigned int lane = threadIdx.x % BAH_LANES;       // 0..BAH_LANES-1
    // The parent writes `smem[local_out * 2 + (lane / 32)]`, which IS the
    // block-wide warp index — spelled that way here so the two cannot drift.
    const unsigned int warp = threadIdx.x / 32;

    const unsigned int K_VEC = K / 8;
    const uint4* __restrict__ A_vec = (const uint4*)(A + (unsigned long long)token * K_stride);

    // BAH_GROUPS groups x BAH_WARPS warp partials. Reused every tile, which is
    // why the tile loop syncs BEFORE writing as well as after.
    __shared__ float red[BAH_GROUPS * BAH_WARPS];

    const unsigned int n_groups = (N + BAH_OUTS - 1) / BAH_OUTS;

    for (unsigned int g0 = 0; g0 < n_groups; g0 += BAH_GROUPS) {
        float acc[BAH_GROUPS];
        const uint4* __restrict__ B_vec[BAH_GROUPS];
        #pragma unroll
        for (int g = 0; g < BAH_GROUPS; g++) {
            acc[g] = 0.0f;
            unsigned int n = (g0 + (unsigned int)g) * BAH_OUTS + local_out;
            // Inactive outputs (N not a multiple of BAH_GROUPS*BAH_OUTS) read
            // row 0 so the address is in bounds; their acc is never written.
            unsigned int row = (n < N) ? n : 0u;
            B_vec[g] = (const uint4*)(B + (unsigned long long)row * K);
        }

        // The parent's K sweep, with the A conversion hoisted out of the
        // output loop. Element index is kv*8 + (2*i) for the low half of word
        // i and kv*8 + (2*i+1) for the high half — the parent's order.
        for (unsigned int kv = lane; kv < K_VEC; kv += BAH_LANES) {
            uint4 a_data = A_vec[kv];
            const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
            float af[8];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                __nv_bfloat16 a_lo, a_hi;
                *(unsigned short*)&a_lo = (unsigned short)(a_raw[i] & 0xFFFF);
                *(unsigned short*)&a_hi = (unsigned short)(a_raw[i] >> 16);
                af[2 * i] = __bfloat162float(a_lo);
                af[2 * i + 1] = __bfloat162float(a_hi);
            }
            #pragma unroll
            for (int g = 0; g < BAH_GROUPS; g++) {
                uint4 b_data = B_vec[g][kv];
                const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    __nv_bfloat16 b_lo, b_hi;
                    *(unsigned short*)&b_lo = (unsigned short)(b_raw[i] & 0xFFFF);
                    *(unsigned short*)&b_hi = (unsigned short)(b_raw[i] >> 16);
                    acc[g] += af[2 * i] * __bfloat162float(b_lo);
                    acc[g] += af[2 * i + 1] * __bfloat162float(b_hi);
                }
            }
        }

        // Warp shuffle reduction — the parent's offsets, in the parent's order.
        #pragma unroll
        for (int g = 0; g < BAH_GROUPS; g++) {
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc[g] += __shfl_down_sync(0xFFFFFFFF, acc[g], offset);
            }
        }

        // The previous tile's partials must be fully consumed before this one
        // overwrites them; `red` is the only state carried across tiles.
        __syncthreads();
        if ((threadIdx.x % 32) == 0) {
            #pragma unroll
            for (int g = 0; g < BAH_GROUPS; g++) {
                red[g * BAH_WARPS + warp] = acc[g];
            }
        }
        __syncthreads();

        // Cross-warp sum + transforms. One thread per (group, output), and the
        // sum is `even + odd` exactly as the parent's
        // `smem[local_out*2] + smem[local_out*2 + 1]`.
        if (threadIdx.x < BAH_GROUPS * BAH_OUTS) {
            const unsigned int g = threadIdx.x / BAH_OUTS;
            const unsigned int lo = threadIdx.x % BAH_OUTS;
            const unsigned int n = (g0 + g) * BAH_OUTS + lo;
            if (n < N) {
                float result = red[g * BAH_WARPS + lo * 2] + red[g * BAH_WARPS + lo * 2 + 1];
                unsigned int group_dim_ba = 2 * vheads_per_group;
                unsigned int within_group = n % group_dim_ba;
                unsigned int group = n / group_dim_ba;

                float* gate_tok = gate_out + (unsigned long long)token * gate_stride;

                if (within_group < vheads_per_group) {
                    // Beta element: sigmoid(b_raw) -> stored at offset nv
                    unsigned int vh = group * vheads_per_group + within_group;
                    gate_tok[nv + vh] = 1.0f / (1.0f + __expf(-result));
                } else {
                    // Alpha (gate): exp(-exp(A_log) * softplus(alpha + dt_bias))
                    unsigned int vh = group * vheads_per_group + (within_group - vheads_per_group);
                    float a_log_val = A_log[vh];
                    float dt_b = dt_bias[vh];
                    float A_val = __expf(fminf(a_log_val, 20.0f));
                    float dt = __logf(1.0f + __expf(fminf(result + dt_b, 20.0f)));
                    gate_tok[vh] = __expf(-A_val * dt);
                }
            }
        }
    }
}
