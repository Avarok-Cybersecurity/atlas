// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Fused MoE Expert+Shared GEMV — shared expert as extra blockIdx.y slot.
//
// Same as moe_expert_gemv_fused.cu gate_up_2x / silu_down_2x but with
// blockIdx.y == top_k serving the shared expert using direct weight pointers.
// The shared expert blocks run concurrently with routed expert blocks within
// the same kernel launch, eliminating 2 separate kernel launches per layer
// (96 graph nodes across 48 MoE layers).
//
// Grid: gate_up (ceil(N/8), top_k+1, 2),  silu_down (ceil(N/8), top_k+1, 1)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 128
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_SHARED[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// NVFP4 per-block FP8-E4M3 scale decode. SCALE/gfx1151 `(float)__nv_fp8_e4m3`
// is NON-STANDARD (same bug fixed in moe_sorted_prefill.cu / the decode GEMVs) —
// software scl_fp8 there; NVIDIA path is the verbatim cast.
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float atlas_dec_e4m3(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
#else
__device__ __forceinline__ float atlas_dec_e4m3(unsigned char b) {
    __nv_fp8_e4m3 f; *(unsigned char*)&f = b; return (float)f;
}
#endif

// ── Fused Gate+Up 2x with shared expert ──
//
// blockIdx.y < top_k: routed expert (pointer table lookup)
// blockIdx.y == top_k: shared expert (direct weight pointers)
// Grid: (ceil(N/8), top_k+1, 2)  Block: (128, 1, 1)
extern "C" __global__ void moe_expert_gate_up_shared(
    const __nv_bfloat16* __restrict__ A,
    // Routed expert tables
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    // Shared expert direct pointers
    const unsigned char* __restrict__ sh_gate_packed,
    const unsigned char* __restrict__ sh_gate_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_packed,
    const unsigned char* __restrict__ sh_up_scale,
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
        // NULL shared expert: model has no shared expert weights (e.g., Mistral).
        // Write zeros and return to prevent NULL pointer dereference.
        if (sh_gate_packed == 0) {
            __nv_bfloat16* out = (proj == 0) ? sh_gate_out : sh_up_out;
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE)
                out[n_base + i] = __float2bfloat16(0.0f);
            return;
        }
        if (proj == 0) {
            B_packed = sh_gate_packed; B_scale = sh_gate_scale;
            s2 = sh_gate_s2; C = sh_gate_out;
        } else {
            B_packed = sh_up_packed; B_scale = sh_up_scale;
            s2 = sh_up_s2; C = sh_up_out;
        }
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        if (proj == 0) {
            B_packed = (const unsigned char*)gate_packed_ptrs[expert_id];
            B_scale = (const unsigned char*)gate_scale_ptrs[expert_id];
            s2 = gate_scale2_vals[expert_id]; C = gate_out;
        } else {
            B_packed = (const unsigned char*)up_packed_ptrs[expert_id];
            B_scale = (const unsigned char*)up_scale_ptrs[expert_id];
            s2 = up_scale2_vals[expert_id]; C = up_out;
        }
        // EP: NULL pointer means remote expert — write zero output and return
        if (B_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
                C[expert_slot * N + n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
    }

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_SHARED[threadIdx.x];
    __syncthreads();

    float acc1 = 0.0f, acc2 = 0.0f;

    // 16 K-values per iteration: uint64 weight + 2×uint4 activation
    for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
        const unsigned int base_k = k16 * 16;
        uint4 a_lo = ((const uint4*)A)[k16 * 2];
        uint4 a_hi = ((const uint4*)A)[k16 * 2 + 1];
        const unsigned int a_raw[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                        a_hi.x, a_hi.y, a_hi.z, a_hi.w};

        unsigned long long packed8_1 = *(const unsigned long long*)(B_packed + (unsigned long long)n1 * half_K + k16 * 8);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
        float sc1 = atlas_dec_e4m3(sb1) * s2;

        unsigned long long packed8_2 = have_n2 ?
            *(const unsigned long long*)(B_packed + (unsigned long long)n2 * half_K + k16 * 8) : 0;
        unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
        float sc2 = have_n2 ? atlas_dec_e4m3(sb2) * s2 : 0.0f;

        #pragma unroll
        for (int b = 0; b < 8; b++) {
            unsigned char bv1 = (unsigned char)(packed8_1 >> (b * 8));
            float w1l = s_lut[bv1 & 0xF] * sc1, w1h = s_lut[bv1 >> 4] * sc1;
            unsigned char bv2 = (unsigned char)(packed8_2 >> (b * 8));
            float w2l = s_lut[bv2 & 0xF] * sc2, w2h = s_lut[bv2 >> 4] * sc2;
            __nv_bfloat16 al, ah;
            *(unsigned short*)&al = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&ah = (unsigned short)(a_raw[b] >> 16);
            float afl = __bfloat162float(al), afh = __bfloat162float(ah);
            acc1 += afl * w1l + afh * w1h;
            acc2 += afl * w2l + afh * w2h;
        }
    }

    // Output offset: shared expert writes at [0..N], routed at [slot*N..N]
    const unsigned long long base = is_shared ? 0 : (unsigned long long)expert_slot * N;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    if (lane == 0) C[base + n1] = __float2bfloat16(acc1);

    if (have_n2) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        if (lane == 0) C[base + n2] = __float2bfloat16(acc2);
    }
}

// ── Fused SiLU+Down 2x with shared expert ──
//
// Precomputes SiLU(gate)*up in shared memory once per block, eliminating
// redundant SiLU compute across all 4 thread groups and replacing global
// gate/up loads with fast shared memory reads in the GEMV inner loop.
//
// blockIdx.y < top_k: routed expert (pointer table + expert_gate_out/up_out)
// blockIdx.y == top_k: shared expert (direct pointers + sh_gate_in/up_in)
// Grid: (ceil(N/8), top_k+1, 1)  Block: (128, 1, 1)
extern "C" __global__ void moe_expert_silu_down_shared(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    // Shared expert
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_packed,
    const unsigned char* __restrict__ sh_down_scale,
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
        // NULL shared expert: write zeros and return
        if (sh_down_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE)
                sh_down_out[n_base + i] = __float2bfloat16(0.0f);
            return;
        }
        B_packed = sh_down_packed; B_scale = sh_down_scale; s2 = sh_down_s2;
        g_ptr = sh_gate_in; u_ptr = sh_up_in;
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        B_packed = (const unsigned char*)packed_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        g_ptr = gate_out + (unsigned long long)expert_slot * K;
        u_ptr = up_out + (unsigned long long)expert_slot * K;
        // EP: NULL pointer means remote expert — write zero output and return
        if (B_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
                C[expert_slot * N + n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
    }

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float s_lut[16];
    extern __shared__ float s_act[];

    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_SHARED[threadIdx.x];

    // Phase 1: Cooperatively precompute SiLU(gate)*up into shared memory.
    // ⚠ THIS CLAMP IS KNOWN-INCONSISTENT. Do not copy it into a sibling kernel
    // to "match"; that is how it spread in the first place. Left in place in
    // wave 55 because every way of resolving it moves numbers for a model no
    // box here can serve — recorded rather than guessed at.
    //
    // Three things are wrong with it, in increasing order of cost:
    //
    // 1. The old comment here claimed the shared expert "(DeepseekV4MLP) is NOT
    //    clamped", which is why the `!is_shared` guard below exists. The
    //    reference disagrees: `inference/model.py` in
    //    `deepseek-ai/DeepSeek-V4-Flash` constructs BOTH `self.experts` and
    //    `self.shared_experts` with `swiglu_limit=args.swiglu_limit`. So the
    //    guard makes the shared expert diverge from the reference.
    // 2. It is the ONLY kernel in this family that clamps at all.
    //    `_t`, `_bf16*`, `_fp8*`, `_batch2*`, `_batch3*`, `moe_prefill` and
    //    `moe_sorted_prefill` all compute a bare `silu(gf) * uf`. So for one
    //    model: single-token decode clamps, K=2/K=3 MTP verify does not, and
    //    concurrent decode at n>=4 (which lands in `moe_prefill`) does not
    //    either. Same weights, same token, three different activations
    //    depending only on how many sequences happen to be in flight.
    // 3. Like the copy that used to live in `common/moe_silu_mul.cu`, the value
    //    is DeepSeek-V4's `swiglu_limit` applied to every model that reaches
    //    this decode arm — Qwen3.5/3.6 MoE, Qwen3-Next, Qwen3-VL, MiniMax-M2,
    //    Step-3.7 — none of which declare a limit. (Gemma-4-26B escapes only
    //    because it ships clamp-free shadows of this file and uses GELU.)
    //
    // The `moe_silu_mul` half of this was fixed by moving the clamp into the
    // shadows of the two models whose configs declare a `swiglu_limit`, gated
    // on the 27B. The same move is NOT obviously right here: the MoE arm's gate
    // is a separate ~3.5 h bfcl-subset run on Qwen3.6-35B-A3B, and DeepSeek-V4
    // — the model the clamp is FOR, and the one a wrong answer hurts most — has
    // no checkpoint on any box to measure against. The real fix for all three
    // points is one change: thread `swiglu_limit` from `ModelConfig` into a
    // kernel argument, defaulting to "no clamp", and apply it uniformly to
    // routed and shared alike. That needs an ABI change across this whole
    // family, so it wants its own wave with both MoE gates budgeted.
    const float SWIGLU_LIMIT = 10.0f;
    for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
        float gf = __bfloat162float(g_ptr[i]);
        float uf = __bfloat162float(u_ptr[i]);
        if (!is_shared) {
            gf = fminf(gf, SWIGLU_LIMIT);
            uf = fminf(fmaxf(uf, -SWIGLU_LIMIT), SWIGLU_LIMIT);
        }
        s_act[i] = (gf / (1.0f + __expf(-gf))) * uf;
    }
    __syncthreads();

    // Phase 2: GEMV with 16 K-values per iteration
    float acc1 = 0.0f, acc2 = 0.0f;

    for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
        const unsigned int base_k = k16 * 16;

        unsigned long long packed8_1 = *(const unsigned long long*)(B_packed + (unsigned long long)n1 * half_K + k16 * 8);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
        float sc1 = atlas_dec_e4m3(sb1) * s2;

        unsigned long long packed8_2 = have_n2 ?
            *(const unsigned long long*)(B_packed + (unsigned long long)n2 * half_K + k16 * 8) : 0;
        unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
        float sc2 = have_n2 ? atlas_dec_e4m3(sb2) * s2 : 0.0f;

        #pragma unroll
        for (int b = 0; b < 8; b++) {
            float al = s_act[base_k + b * 2];
            float ah = s_act[base_k + b * 2 + 1];

            unsigned char bv1 = (unsigned char)(packed8_1 >> (b * 8));
            float w1l = s_lut[bv1 & 0xF] * sc1, w1h = s_lut[bv1 >> 4] * sc1;
            unsigned char bv2 = (unsigned char)(packed8_2 >> (b * 8));
            float w2l = s_lut[bv2 & 0xF] * sc2, w2h = s_lut[bv2 >> 4] * sc2;

            acc1 += al * w1l + ah * w1h;
            acc2 += al * w2l + ah * w2h;
        }
    }

    // Output: shared writes to sh_down_out, routed writes to C[slot*N]
    __nv_bfloat16* out = is_shared ? sh_down_out : (C + (unsigned long long)expert_slot * N);

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    if (lane == 0) out[n1] = __float2bfloat16(acc1);

    if (have_n2) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        if (lane == 0) out[n2] = __float2bfloat16(acc2);
    }
}
