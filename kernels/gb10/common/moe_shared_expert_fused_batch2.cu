// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Fused MoE Expert+Shared GEMV — K=2 multi-token batch variant.
//
// Processes 2 tokens through MoE in single kernel launches by expanding
// blockIdx.y to accommodate 2 sets of (top_k routed + 1 shared) experts.
// Weights are loaded once and applied to both tokens' inputs, halving
// weight bandwidth for the shared expert and gate projection.
//
// Token layout in blockIdx.y:
//   y ∈ [0, 2*top_k)         → routed experts (token = y/top_k, slot = y%top_k)
//   y ∈ [2*top_k, 2*top_k+2) → shared expert  (token = y - 2*top_k)
//
// Grid: gate_up_batch2  (ceil(N/8), 2*(top_k+1), 2)
//       silu_down_batch2 (ceil(N/8), 2*(top_k+1), 1)
//
// Block width is chosen by the HOST (`MoeLayer::forward_k2`) and read back
// here from `blockDim.x` — it is deliberately NOT a `#define`. It used to be
// one, and the host launched 256 threads whenever hidden_size >= 3072 while
// only three model shadows actually defined BLOCK_SIZE 256; every other
// large-hidden MoE model inherited this file's 128 and got a block twice as
// wide as the code it was compiled for. Threads 128..255 then recomputed the
// NEXT block's two columns and stored the same bits over them: correct output,
// twice the loads. Deriving the decomposition from `blockDim.x` is the one
// size the host and the device cannot disagree about.
//
// Supported widths: 128 (one warp per output pair) and 256 (two warps per
// output pair, joined through smem). Anything else is treated as 128, which
// is wasteful but never wrong — the same benign failure mode as before.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_NARROW 128
#define BLOCK_WIDE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_BATCH2[16] = {
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

// Reduce the per-thread partials of one output pair and store them. Shared by
// gate_up and silu_down, which used to carry a copy each.
//
// BLOCK=128 → 32 threads (one warp) per output pair, shuffle reduction only:
//             bit-for-bit the reduction this file has always done.
// BLOCK=256 → 64 threads (two warps) per output pair; the two warp partials
//             are joined through `smem_reduce`, matching what the 122B / M2 /
//             step3.7 shadows of this file did in their own copy.
template <unsigned int BLOCK>
__device__ void reduce_store_batch2(
    float acc1, float acc2,
    __nv_bfloat16* __restrict__ out,
    unsigned int n1, unsigned int n2,
    unsigned int local_out, unsigned int lane,
    bool active, bool have_n2
) {
    constexpr unsigned int THREADS_PER_OUT = BLOCK / N_PER_BLOCK;
    constexpr unsigned int WARPS_PER_OUT = THREADS_PER_OUT / WARP_SIZE;

    if constexpr (WARPS_PER_OUT == 1) {
        if (active) {
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
    } else {
        // Declared HERE, not beside THREADS_PER_OUT above: at BLOCK=128 the
        // shuffle branch is taken and a file-scope declaration is never
        // referenced, which nvcc reports as #177-D — a warning on the Linux
        // toolchain and a hard error under the Windows CUDA build's stricter
        // flags, where it broke `windows-x86_64-nvidia-cuda`. Scoping it to the
        // branch that uses it also retires the `(WARPS_PER_OUT > 1) ? … : 1`
        // guard the old declaration needed purely to avoid a zero-length array
        // in the case that never touched it.
        constexpr unsigned int SMEM_RED = N_PER_BLOCK * 2 * WARPS_PER_OUT;
        __shared__ float smem_reduce[SMEM_RED];
        const unsigned int warp_lane = lane % WARP_SIZE;
        const unsigned int warp_in_output = lane / WARP_SIZE;
        if (active) {
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
                acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
            }
            if (warp_lane == 0) {
                smem_reduce[local_out * 2 * WARPS_PER_OUT + warp_in_output * 2] = acc1;
                smem_reduce[local_out * 2 * WARPS_PER_OUT + warp_in_output * 2 + 1] = acc2;
            }
        }
        __syncthreads();

        if (active && lane == 0) {
            float final1 = 0.0f, final2 = 0.0f;
            #pragma unroll
            for (unsigned int w = 0; w < WARPS_PER_OUT; w++) {
                final1 += smem_reduce[local_out * 2 * WARPS_PER_OUT + w * 2];
                final2 += smem_reduce[local_out * 2 * WARPS_PER_OUT + w * 2 + 1];
            }
            out[n1] = __float2bfloat16(final1);
            if (have_n2) out[n2] = __float2bfloat16(final2);
        }
    }
}

// Per-output-pair gate/up GEMV, parameterised on the block width so that
// THREADS_PER_OUT stays a compile-time constant on both instantiations.
template <unsigned int BLOCK>
__device__ void gate_up_gemv_batch2(
    const __nv_bfloat16* __restrict__ A_token,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    float s2,
    __nv_bfloat16* __restrict__ C,
    unsigned int N, unsigned int K
) {
    constexpr unsigned int THREADS_PER_OUT = BLOCK / N_PER_BLOCK;
    static_assert(THREADS_PER_OUT % WARP_SIZE == 0, "output group must be whole warps");

    const unsigned int local_out = threadIdx.x / THREADS_PER_OUT;
    const unsigned int lane = threadIdx.x % THREADS_PER_OUT;

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    // Tail columns used to `return` here — in front of the __syncthreads()
    // below, which is undefined behaviour when only part of the block takes
    // it. Carry it as a predicate instead so every thread reaches the barrier.
    const bool active = (n1 < N);
    const bool have_n2 = (n2 < N);

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_BATCH2[threadIdx.x];
    __syncthreads();

    float acc1 = 0.0f, acc2 = 0.0f;

    if (active) {
        for (unsigned int k8 = lane; k8 < K8; k8 += THREADS_PER_OUT) {
            uint4 a_data = ((const uint4*)A_token)[k8];
            const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
            const unsigned int base_k = k8 * 8;

            unsigned int packed4_1 = *(const unsigned int*)(B_packed + (unsigned long long)n1 * half_K + k8 * 4);
            unsigned int sg = base_k / GROUP_SIZE;
            unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
            float sc1 = atlas_dec_e4m3(sb1) * s2;

            unsigned int packed4_2 = have_n2 ?
                *(const unsigned int*)(B_packed + (unsigned long long)n2 * half_K + k8 * 4) : 0;
            unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
            float sc2 = have_n2 ? atlas_dec_e4m3(sb2) * s2 : 0.0f;

            #pragma unroll
            for (int b = 0; b < 4; b++) {
                unsigned char bv1 = (packed4_1 >> (b * 8)) & 0xFF;
                float w1l = s_lut[bv1 & 0xF] * sc1, w1h = s_lut[bv1 >> 4] * sc1;
                unsigned char bv2 = (packed4_2 >> (b * 8)) & 0xFF;
                float w2l = s_lut[bv2 & 0xF] * sc2, w2h = s_lut[bv2 >> 4] * sc2;
                __nv_bfloat16 al, ah;
                *(unsigned short*)&al = (unsigned short)(a_raw[b] & 0xFFFF);
                *(unsigned short*)&ah = (unsigned short)(a_raw[b] >> 16);
                float afl = __bfloat162float(al), afh = __bfloat162float(ah);
                acc1 += afl * w1l + afh * w1h;
                acc2 += afl * w2l + afh * w2h;
            }
        }
    }

    reduce_store_batch2<BLOCK>(acc1, acc2, C, n1, n2, local_out, lane, active, have_n2);
}

// Per-output-pair SiLU+down GEMV. Phase 1 precomputes SiLU(gate)*up into
// dynamic shared memory (K floats, sized by the launcher), phase 2 is the
// down-projection GEMV over it.
template <unsigned int BLOCK>
__device__ void silu_down_gemv_batch2(
    const __nv_bfloat16* __restrict__ g_ptr,
    const __nv_bfloat16* __restrict__ u_ptr,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    float s2,
    __nv_bfloat16* __restrict__ out,
    unsigned int N, unsigned int K
) {
    constexpr unsigned int THREADS_PER_OUT = BLOCK / N_PER_BLOCK;
    static_assert(THREADS_PER_OUT % WARP_SIZE == 0, "output group must be whole warps");

    const unsigned int local_out = threadIdx.x / THREADS_PER_OUT;
    const unsigned int lane = threadIdx.x % THREADS_PER_OUT;

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    // Predicate, not a `return`: the tail columns must still fill their slice
    // of s_act and reach the __syncthreads() that publishes it.
    const bool active = (n1 < N);
    const bool have_n2 = (n2 < N);

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    // Dynamic: K floats, sized by the launcher (issue #85 -- the old
    // static s_act[1024] overflowed for Mistral-Small-4's
    // expert_hidden_dim=2048, illegal-addressing on the first batched
    // K=2 FFN; matches the extern pattern the _t variant already uses).
    extern __shared__ float s_act[];

    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_BATCH2[threadIdx.x];

    // Phase 1: Precompute SiLU(gate)*up into shared memory
    for (unsigned int i = threadIdx.x; i < K; i += BLOCK) {
        float gf = __bfloat162float(g_ptr[i]);
        float uf = __bfloat162float(u_ptr[i]);
        s_act[i] = (gf / (1.0f + __expf(-gf))) * uf;
    }
    __syncthreads();

    // Phase 2: GEMV reading precomputed activation from shared memory
    float acc1 = 0.0f, acc2 = 0.0f;

    if (active) {
        for (unsigned int k8 = lane; k8 < K8; k8 += THREADS_PER_OUT) {
            const unsigned int base_k = k8 * 8;

            unsigned int packed4_1 = *(const unsigned int*)(B_packed + (unsigned long long)n1 * half_K + k8 * 4);
            unsigned int sg = base_k / GROUP_SIZE;
            unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
            float sc1 = atlas_dec_e4m3(sb1) * s2;

            unsigned int packed4_2 = have_n2 ?
                *(const unsigned int*)(B_packed + (unsigned long long)n2 * half_K + k8 * 4) : 0;
            unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
            float sc2 = have_n2 ? atlas_dec_e4m3(sb2) * s2 : 0.0f;

            #pragma unroll
            for (int b = 0; b < 4; b++) {
                float al = s_act[base_k + b * 2];
                float ah = s_act[base_k + b * 2 + 1];

                unsigned char bv1 = (packed4_1 >> (b * 8)) & 0xFF;
                float w1l = s_lut[bv1 & 0xF] * sc1, w1h = s_lut[bv1 >> 4] * sc1;
                unsigned char bv2 = (packed4_2 >> (b * 8)) & 0xFF;
                float w2l = s_lut[bv2 & 0xF] * sc2, w2h = s_lut[bv2 >> 4] * sc2;

                acc1 += al * w1l + ah * w1h;
                acc2 += al * w2l + ah * w2h;
            }
        }
    }

    reduce_store_batch2<BLOCK>(acc1, acc2, out, n1, n2, local_out, lane, active, have_n2);
}

// ── Fused Gate+Up 2x with shared expert — K=2 batch variant ──
//
// Grid: (ceil(N/8), 2*(top_k+1), 2)  Block: (128 or 256, 1, 1)
// blockIdx.y: 0..2*top_k-1 = routed (token=y/top_k, slot=y%top_k)
//             2*top_k..2*top_k+1 = shared (token=y-2*top_k)
extern "C" __global__ void moe_expert_gate_up_shared_batch2(
    const __nv_bfloat16* __restrict__ A,       // [2, H] BF16 input (2 tokens)
    // Routed expert tables
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,      // [2*top_k, inter] BF16
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,        // [2*top_k, inter] BF16
    const unsigned int* __restrict__ expert_indices,  // [2*top_k] u32
    // Shared expert direct pointers
    const unsigned char* __restrict__ sh_gate_packed,
    const unsigned char* __restrict__ sh_gate_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,   // [2, inter] BF16
    const unsigned char* __restrict__ sh_up_packed,
    const unsigned char* __restrict__ sh_up_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,     // [2, inter] BF16
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int total_routed = 2 * top_k;
    const unsigned int y = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (y >= total_routed);

    // Determine token index and expert slot
    unsigned int token, expert_slot;
    if (is_shared) {
        token = y - total_routed;  // 0 or 1
        expert_slot = 0;           // unused for shared
    } else {
        token = y / top_k;         // 0 or 1
        expert_slot = y % top_k;   // 0..top_k-1
    }

    // Select input for this token
    const __nv_bfloat16* A_token = A + (unsigned long long)token * K;

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* C;

    if (is_shared) {
        // NULL shared-expert weights = "no in-kernel shared expert": zero the
        // output rows and exit, exactly as the _t variant does
        // (moe_shared_expert_fused_batch2_t.cu:85-100). A model whose shared
        // expert is a DIFFERENT precision from its routed experts (Laguna:
        // NVFP4 routed + BF16 shared) passes NULL here and computes the shared
        // half separately; without this guard that dereferences NULL and the
        // kernel faults the moment a 2-sequence batch forms.
        if (proj == 0) {
            if (sh_gate_packed == 0) {
                const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
                __nv_bfloat16* z = sh_gate_out + (unsigned long long)token * N;
                for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N;
                     i += blockDim.x) {
                    z[n_base + i] = __float2bfloat16(0.0f);
                }
                return;
            }
            B_packed = sh_gate_packed; B_scale = sh_gate_scale;
            s2 = sh_gate_s2; C = sh_gate_out + (unsigned long long)token * N;
        } else {
            if (sh_up_packed == 0) {
                const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
                __nv_bfloat16* z = sh_up_out + (unsigned long long)token * N;
                for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N;
                     i += blockDim.x) {
                    z[n_base + i] = __float2bfloat16(0.0f);
                }
                return;
            }
            B_packed = sh_up_packed; B_scale = sh_up_scale;
            s2 = sh_up_s2; C = sh_up_out + (unsigned long long)token * N;
        }
    } else {
        const unsigned int expert_id = expert_indices[token * top_k + expert_slot];
        const unsigned int flat_slot = token * top_k + expert_slot;
        if (proj == 0) {
            B_packed = (const unsigned char*)gate_packed_ptrs[expert_id];
            B_scale = (const unsigned char*)gate_scale_ptrs[expert_id];
            s2 = gate_scale2_vals[expert_id];
            C = gate_out + (unsigned long long)flat_slot * N;
        } else {
            B_packed = (const unsigned char*)up_packed_ptrs[expert_id];
            B_scale = (const unsigned char*)up_scale_ptrs[expert_id];
            s2 = up_scale2_vals[expert_id];
            C = up_out + (unsigned long long)flat_slot * N;
        }
        // EP: NULL pointer means remote expert — write zero output and return.
        // Block-uniform, so this return is not a partial-block exit.
        if (B_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += blockDim.x) {
                C[n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
    }

    if (blockDim.x == BLOCK_WIDE) {
        gate_up_gemv_batch2<BLOCK_WIDE>(A_token, B_packed, B_scale, s2, C, N, K);
    } else {
        gate_up_gemv_batch2<BLOCK_NARROW>(A_token, B_packed, B_scale, s2, C, N, K);
    }
}

// ── Fused SiLU+Down 2x with shared expert — K=2 batch variant ──
//
// Grid: (ceil(N/8), 2*(top_k+1), 1)  Block: (128 or 256, 1, 1)
extern "C" __global__ void moe_expert_silu_down_shared_batch2(
    const __nv_bfloat16* __restrict__ gate_out,  // [2*top_k, inter] BF16
    const __nv_bfloat16* __restrict__ up_out,    // [2*top_k, inter] BF16
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,               // [2*top_k, H] BF16
    const unsigned int* __restrict__ expert_indices,  // [2*top_k] u32
    // Shared expert
    const __nv_bfloat16* __restrict__ sh_gate_in,  // [2, inter] BF16
    const __nv_bfloat16* __restrict__ sh_up_in,    // [2, inter] BF16
    const unsigned char* __restrict__ sh_down_packed,
    const unsigned char* __restrict__ sh_down_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,       // [2, H] BF16
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int total_routed = 2 * top_k;
    const unsigned int y = blockIdx.y;
    const bool is_shared = (y >= total_routed);

    unsigned int token, expert_slot;
    if (is_shared) {
        token = y - total_routed;
        expert_slot = 0;
    } else {
        token = y / top_k;
        expert_slot = y % top_k;
    }

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;

    const __nv_bfloat16* g_ptr;
    const __nv_bfloat16* u_ptr;

    if (is_shared) {
        // NULL shared-expert down weight = no in-kernel shared expert (see the
        // gate_up kernel above). Zero this token's shared output rows and exit
        // rather than dereferencing NULL; the caller supplies the shared half
        // separately. Mirrors moe_shared_expert_fused_batch2_t.cu:188.
        if (sh_down_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            __nv_bfloat16* z = sh_down_out + (unsigned long long)token * N;
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N;
                 i += blockDim.x) {
                z[n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
        B_packed = sh_down_packed; B_scale = sh_down_scale; s2 = sh_down_s2;
        g_ptr = sh_gate_in + (unsigned long long)token * K;
        u_ptr = sh_up_in + (unsigned long long)token * K;
    } else {
        const unsigned int expert_id = expert_indices[token * top_k + expert_slot];
        const unsigned int flat_slot = token * top_k + expert_slot;
        B_packed = (const unsigned char*)packed_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        g_ptr = gate_out + (unsigned long long)flat_slot * K;
        u_ptr = up_out + (unsigned long long)flat_slot * K;
        // EP: NULL pointer means remote expert — write zero output and return.
        // Block-uniform, so this return is not a partial-block exit.
        if (B_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += blockDim.x) {
                C[(unsigned long long)(token * top_k + expert_slot) * N + n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
    }

    // Output: shared at sh_down_out[token*N], routed at C[flat_slot*N]
    __nv_bfloat16* out = is_shared
        ? (sh_down_out + (unsigned long long)token * N)
        : (C + (unsigned long long)(token * top_k + expert_slot) * N);

    if (blockDim.x == BLOCK_WIDE) {
        silu_down_gemv_batch2<BLOCK_WIDE>(g_ptr, u_ptr, B_packed, B_scale, s2, out, N, K);
    } else {
        silu_down_gemv_batch2<BLOCK_NARROW>(g_ptr, u_ptr, B_packed, B_scale, s2, out, N, K);
    }
}

// ── Weighted sum + sigmoid blend — K=2 batch variant ──
//
// Combines routed expert outputs with shared expert via sigmoid gate.
// blockIdx.y = token index (0 or 1).
//
// Grid: (ceil(hidden/256), 2, 1)  Block: (256, 1, 1)
extern "C" __global__ void moe_weighted_sum_blend_batch2(
    __nv_bfloat16* __restrict__ output,              // [2, hidden] BF16
    const __nv_bfloat16* __restrict__ expert_out,    // [2*top_k, hidden] BF16
    const float* __restrict__ expert_weights,         // [2*top_k] f32
    const __nv_bfloat16* __restrict__ shared_out,    // [2, hidden] BF16
    const __nv_bfloat16* __restrict__ input,         // [2, K] BF16 (MoE input)
    const __nv_bfloat16* __restrict__ gate_weight,   // [1, K] BF16 (shared gate)
    unsigned int hidden,
    unsigned int top_k,
    unsigned int K
) {
    const unsigned int token = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane = tid % WARP_SIZE;

    // Per-token input pointer
    const __nv_bfloat16* my_input = input + (unsigned long long)token * K;
    const float* my_weights = expert_weights + token * top_k;
    const __nv_bfloat16* my_expert_out = expert_out + (unsigned long long)token * top_k * hidden;
    const __nv_bfloat16* my_shared_out = shared_out + (unsigned long long)token * hidden;
    __nv_bfloat16* my_output = output + (unsigned long long)token * hidden;

    // ── Phase 1: Compute gate scalar (dot product + sigmoid) ──
    // NULL gate_weight = no gate modulation → sigmoid=1.0 (shared expert always
    // on). Matches the per-token moe_weighted_sum_blend; models with an
    // ungated shared expert (Laguna, Mistral) pass a NULL pointer here.
    __shared__ float s_warp_sums[8];
    __shared__ float sigmoid_val;

    if (gate_weight == 0) {
        if (tid == 0) sigmoid_val = 1.0f;
        __syncthreads();
    } else {

    float dot_acc = 0.0f;
    unsigned int K8 = K / 8;
    for (unsigned int k8 = tid; k8 < K8; k8 += 256) {
        uint4 a_data = ((const uint4*)my_input)[k8];
        uint4 w_data = ((const uint4*)gate_weight)[k8];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        const unsigned int w_raw[4] = {w_data.x, w_data.y, w_data.z, w_data.w};

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            __nv_bfloat16 a_lo, a_hi, w_lo, w_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[b] >> 16);
            *(unsigned short*)&w_lo = (unsigned short)(w_raw[b] & 0xFFFF);
            *(unsigned short*)&w_hi = (unsigned short)(w_raw[b] >> 16);
            dot_acc += __bfloat162float(a_lo) * __bfloat162float(w_lo);
            dot_acc += __bfloat162float(a_hi) * __bfloat162float(w_hi);
        }
    }

    // Warp shuffle reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        dot_acc += __shfl_down_sync(0xFFFFFFFF, dot_acc, offset);
    }
    if (lane == 0) {
        s_warp_sums[warp_id] = dot_acc;
    }
    __syncthreads();

    if (tid == 0) {
        float gate_scalar = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; w++) {
            gate_scalar += s_warp_sums[w];
        }
        sigmoid_val = 1.0f / (1.0f + __expf(-gate_scalar));
    }
    __syncthreads();

    }  // end else (gate_weight != 0)

    // ── Phase 2: Weighted sum + blend ──
    unsigned int j = blockIdx.x * blockDim.x + tid;
    if (j >= hidden) return;

    float acc = 0.0f;
    for (unsigned int e = 0; e < top_k; e++) {
        acc += my_weights[e] * __bfloat162float(my_expert_out[(unsigned long long)e * hidden + j]);
    }
    acc += sigmoid_val * __bfloat162float(my_shared_out[j]);
    my_output[j] = __float2bfloat16(acc);
}
