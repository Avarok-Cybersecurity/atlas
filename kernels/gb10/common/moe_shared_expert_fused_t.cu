// SPDX-License-Identifier: AGPL-3.0-only
//
// Transposed-layout decode MoE GEMV — Phase 8a unified-layout MoE.
//
// Same semantics as moe_expert_silu_down_shared / moe_expert_gate_up_shared,
// but reads the weight tensor in transposed `[K/2, N]` layout (input-major)
// instead of the current `[N, K/2]` layout (output-major). Goal: a single
// weight layout shared between prefill and decode so we can free the
// untransposed copies (~59 GB on MiniMax M2.7-NVFP4 EP=2) and either
// run the persistent down transpose or use the headroom for KV cache.
//
// Coalescence strategy is inverted vs the original kernel:
//   - original: each thread owns ONE output (sub-group reduces over K)
//     → reads B[n_lane * (K/2) + k_iter*8] = 8 bytes contiguous per lane
//   - transposed: each thread STILL owns one output, but threads in a
//     warp have ADJACENT n's (lane = n within warp). Read B_t[k_half * N + n]:
//     32 bytes contiguous per warp = 1 cache line per K iter. No warp
//     reduction needed (each lane has its own accumulator + own output).
//
// Block: (128, 1, 1)  Grid: (ceil(N/128), top_k+1, 1)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Tuning note: 32-thread blocks (1 warp per block) outperform 128-thread blocks
// on GB10 for the transposed decode silu_down — more blocks → better SM
// occupancy, the s_act shared-mem precompute parallelism is unchanged because
// the K loop dominates (block_size irrelevant once each thread is 1 output).
#define BLOCK_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_T[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Transposed-layout fused gate+up decode kernel.
//
// Single-token GEMV: for each routed expert slot (top_k of them) plus the
// shared expert, compute gate_out = A @ W_gate^T and up_out = A @ W_up^T.
// blockIdx.z selects gate (0) or up (1). Same coalescence strategy as
// silu_down_t — each thread owns one output, lanes within warp adjacent
// in N.
extern "C" __global__ void moe_expert_gate_up_shared_t(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_t_ptrs,
    const unsigned long long* __restrict__ gate_scale_t_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_t_ptrs,
    const unsigned long long* __restrict__ up_scale_t_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    // Shared expert direct pointers (transposed).
    const unsigned char* __restrict__ sh_gate_t_packed,
    const unsigned char* __restrict__ sh_gate_t_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_t_packed,
    const unsigned char* __restrict__ sh_up_t_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (expert_slot == top_k);

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* C;

    if (is_shared) {
        if (proj == 0) {
            if (sh_gate_t_packed == 0) {
                const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
                if (n < N) sh_gate_out[n] = __float2bfloat16(0.0f);
                return;
            }
            B_packed = sh_gate_t_packed;
            B_scale = sh_gate_t_scale;
            s2 = sh_gate_s2;
            C = sh_gate_out;
        } else {
            if (sh_up_t_packed == 0) {
                const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
                if (n < N) sh_up_out[n] = __float2bfloat16(0.0f);
                return;
            }
            B_packed = sh_up_t_packed;
            B_scale = sh_up_t_scale;
            s2 = sh_up_s2;
            C = sh_up_out;
        }
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        if (proj == 0) {
            B_packed = (const unsigned char*)gate_packed_t_ptrs[expert_id];
            B_scale = (const unsigned char*)gate_scale_t_ptrs[expert_id];
            s2 = gate_scale2_vals[expert_id];
            C = gate_out;
        } else {
            B_packed = (const unsigned char*)up_packed_t_ptrs[expert_id];
            B_scale = (const unsigned char*)up_scale_t_ptrs[expert_id];
            s2 = up_scale2_vals[expert_id];
            C = up_out;
        }
        if (B_packed == 0) {
            const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
            if (n < N) C[(unsigned long long)expert_slot * N + n] = __float2bfloat16(0.0f);
            return;
        }
    }

    const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const bool valid = (n < N);

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_T[threadIdx.x];
    __syncthreads();

    if (!valid) return;

    // GEMV: C[n] = sum_k A[k] * W[n, k]. With transposed weight stored as
    // [K/2, N] packed: each byte at (k_half, n) holds two consecutive k
    // nibbles for output position n. Iterate by scale-group (16 K) to
    // cache the per-group scale.
    const unsigned int num_groups = K / GROUP_SIZE;
    float acc = 0.0f;
    for (unsigned int sg = 0; sg < num_groups; sg++) {
        unsigned char sb = B_scale[(unsigned long long)sg * N + n];
        __nv_fp8_e4m3 fp8_s; *(unsigned char*)&fp8_s = sb;
        float sc = (float)fp8_s * s2;
        const unsigned int kh_base = sg * 8;
        #pragma unroll
        for (unsigned int kh_off = 0; kh_off < 8; kh_off++) {
            unsigned int k_half = kh_base + kh_off;
            unsigned char byte = B_packed[(unsigned long long)k_half * N + n];
            float a_lo = __bfloat162float(A[k_half * 2]);
            float a_hi = __bfloat162float(A[k_half * 2 + 1]);
            float w_lo = s_lut[byte & 0xFu] * sc;
            float w_hi = s_lut[(byte >> 4) & 0xFu] * sc;
            acc += a_lo * w_lo + a_hi * w_hi;
        }
    }

    if (is_shared) {
        C[n] = __float2bfloat16(acc);
    } else {
        C[(unsigned long long)expert_slot * N + n] = __float2bfloat16(acc);
    }
}

// Transposed-layout silu_down decode kernel.
//
// Per-expert weight buffers `[K/2, N]` packed NVFP4 + `[K/16, N]` FP8 scales.
// Input gate_out / up_out: `[(top_k+1), K]` BF16 (per-slot). top_k slot is
// the shared-expert input. Output C: `[top_k, N]` BF16; shared-expert
// output goes to `sh_down_out: [N]`.
extern "C" __global__ void moe_expert_silu_down_shared_t(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_t_ptrs,   // [num_experts] device ptrs to [K/2 * N] bytes
    const unsigned long long* __restrict__ scale_t_ptrs,    // [num_experts] device ptrs to [K/16 * N] bytes
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    // Shared expert direct pointers (transposed layout)
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    const bool is_shared = (expert_slot == top_k);

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    const __nv_bfloat16* g_ptr;
    const __nv_bfloat16* u_ptr;
    if (is_shared) {
        if (sh_down_t_packed == 0) {
            // No shared expert — write zeros.
            const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
            if (n < N) sh_down_out[n] = __float2bfloat16(0.0f);
            return;
        }
        B_packed = sh_down_t_packed;
        B_scale = sh_down_t_scale;
        s2 = sh_down_s2;
        g_ptr = sh_gate_in;
        u_ptr = sh_up_in;
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        B_packed = (const unsigned char*)packed_t_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_t_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        g_ptr = gate_out + (unsigned long long)expert_slot * K;
        u_ptr = up_out + (unsigned long long)expert_slot * K;
        // EP remote expert: NULL pointer → write zero output and return.
        if (B_packed == 0) {
            const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
            if (n < N) C[expert_slot * N + n] = __float2bfloat16(0.0f);
            return;
        }
    }

    const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const bool valid = (n < N);

    // Phase 1: cooperatively precompute s_act[K] = SiLU(gate[K]) * up[K].
    extern __shared__ float s_act[];
    for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
        float gf = __bfloat162float(g_ptr[i]);
        float uf = __bfloat162float(u_ptr[i]);
        s_act[i] = (gf / (1.0f + __expf(-gf))) * uf;
    }

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_T[threadIdx.x];
    __syncthreads();

    if (!valid) return;

    // Phase 2: per-thread accumulate over K_half iterations. Each thread
    // owns ONE output position `n`; lanes in a warp have adjacent n's so
    // `B_packed[k_half * N + n]` reads are coalesced (1 byte per lane,
    // 32 bytes contiguous per warp per iter).
    const unsigned int K_half = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    float acc = 0.0f;

    // Iterate scale groups (8 K_half iters per group). Cache the
    // per-group scale in a register.
    for (unsigned int sg = 0; sg < num_groups; sg++) {
        unsigned char sb = B_scale[(unsigned long long)sg * N + n];
        __nv_fp8_e4m3 fp8_s; *(unsigned char*)&fp8_s = sb;
        float sc = (float)fp8_s * s2;
        const unsigned int kh_base = sg * 8;
        #pragma unroll
        for (unsigned int kh_off = 0; kh_off < 8; kh_off++) {
            unsigned int k_half = kh_base + kh_off;
            unsigned char byte = B_packed[(unsigned long long)k_half * N + n];
            unsigned int nibble_lo = byte & 0xFu;
            unsigned int nibble_hi = (byte >> 4) & 0xFu;
            float w_lo = s_lut[nibble_lo] * sc;
            float w_hi = s_lut[nibble_hi] * sc;
            float a_lo = s_act[k_half * 2];
            float a_hi = s_act[k_half * 2 + 1];
            acc += a_lo * w_lo + a_hi * w_hi;
        }
        if (kh_base + 8 > K_half) break;
    }

    // Output offset: routed → C[expert_slot * N + n]; shared → sh_down_out[n].
    if (is_shared) {
        sh_down_out[n] = __float2bfloat16(acc);
    } else {
        C[(unsigned long long)expert_slot * N + n] = __float2bfloat16(acc);
    }
}
