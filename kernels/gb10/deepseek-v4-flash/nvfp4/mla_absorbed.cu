// SPDX-License-Identifier: AGPL-3.0-only

// MLA Absorbed Attention Kernels — batched per-head GEMV for Q absorption and V extraction.
//
// Q absorption: Q_absorbed[n, Lkv] = Q_nope[n, P] @ W_UK_T[n, P, Lkv]
//   - 32 heads in parallel, each head does [1, P=64] @ [P, Lkv=256] → [1, Lkv=256]
//   - Grid: (ceil(Lkv/4), N_heads, 1)  Block: (256, 1, 1)
//
// V extraction: v_out[n, V] = attn_latent[n, Lkv] @ W_UV[n, V, Lkv]^T
//   - Actually: v_out[n, v] = sum_l(W_UV[n, v, l] * attn_latent[n, l])
//   - 32 heads in parallel, each head does [V=128, Lkv=256] @ [Lkv=256, 1] → [V=128, 1]
//   - Grid: (ceil(V/4), N_heads, 1)  Block: (256, 1, 1)
//
// Both kernels use the same structure: batched GEMV with per-head weight pointers.
// Input is at a fixed stride per head in the input buffer.
// Output is at a fixed stride per head in the output buffer.

#include <cuda_bf16.h>

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32

// Batched GEMV: output[head, n] = sum_k(weight[head, n, k] * input[head, k])
// for all heads in parallel.
//
// Grid: (ceil(N_out / (N_PER_BLOCK*2)), num_heads, 1)
// Block: (256, 1, 1)
//
// input:  [num_heads, K]        BF16, contiguous per head at stride input_stride
// weight: [num_heads, N_out, K] BF16, contiguous per head at stride N_out * K
// output: [num_heads, N_out]    BF16, contiguous per head at stride output_stride
extern "C" __global__ void mla_batched_gemv(
    const __nv_bfloat16* __restrict__ input,   // [num_heads * input_stride]
    const __nv_bfloat16* __restrict__ weight,  // [num_heads * N_out * K]
    __nv_bfloat16* __restrict__ output,         // [num_heads * output_stride]
    unsigned int N_out,                         // output dimension per head
    unsigned int K,                             // input dimension per head
    unsigned int input_stride,                  // elements between consecutive heads in input
    unsigned int output_stride                  // elements between consecutive heads in output
) {
    const unsigned int head = blockIdx.y;
    const unsigned int tid = threadIdx.x;

    // Each block computes N_PER_BLOCK * 2 output elements for one head
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = tid / threads_per_out;            // 0..3
    const unsigned int lane = tid % threads_per_out;                 // 0..63

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N_out) return;
    const bool have_n2 = (n2 < N_out);

    // Pointers for this head
    const __nv_bfloat16* A = input + (unsigned long long)head * input_stride;
    const __nv_bfloat16* B = weight + (unsigned long long)head * N_out * K;
    __nv_bfloat16* C = output + (unsigned long long)head * output_stride;

    const unsigned int K4 = K / 4;
    const unsigned long long* A64 = (const unsigned long long*)A;

    float acc1 = 0.0f, acc2 = 0.0f;

    for (unsigned int k4 = lane; k4 < K4; k4 += threads_per_out) {
        // Load 4 input values (vectorized 64-bit load)
        unsigned long long av = A64[k4];
        float a0, a1, a2, a3;
        unsigned int lo = (unsigned int)av;
        unsigned int hi = (unsigned int)(av >> 32);
        __nv_bfloat16 tmp;
        *(unsigned short*)&tmp = (unsigned short)(lo & 0xFFFF); a0 = __bfloat162float(tmp);
        *(unsigned short*)&tmp = (unsigned short)(lo >> 16);     a1 = __bfloat162float(tmp);
        *(unsigned short*)&tmp = (unsigned short)(hi & 0xFFFF); a2 = __bfloat162float(tmp);
        *(unsigned short*)&tmp = (unsigned short)(hi >> 16);     a3 = __bfloat162float(tmp);

        unsigned int base_k = k4 * 4;

        // Weight row n1
        float w10 = __bfloat162float(B[n1 * K + base_k]);
        float w11 = __bfloat162float(B[n1 * K + base_k + 1]);
        float w12 = __bfloat162float(B[n1 * K + base_k + 2]);
        float w13 = __bfloat162float(B[n1 * K + base_k + 3]);
        acc1 += a0 * w10 + a1 * w11 + a2 * w12 + a3 * w13;

        if (have_n2) {
            float w20 = __bfloat162float(B[n2 * K + base_k]);
            float w21 = __bfloat162float(B[n2 * K + base_k + 1]);
            float w22 = __bfloat162float(B[n2 * K + base_k + 2]);
            float w23 = __bfloat162float(B[n2 * K + base_k + 3]);
            acc2 += a0 * w20 + a1 * w21 + a2 * w22 + a3 * w23;
        }
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
        if (have_n2) acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
    }

    // Cross-warp reduction via shared memory
    __shared__ float s_partial[N_PER_BLOCK * 2][2]; // [out_idx][warp_idx within out]
    unsigned int warp_in_out = (tid % threads_per_out) / WARP_SIZE;
    unsigned int lane_in_warp = tid % WARP_SIZE;
    if (lane_in_warp == 0) {
        s_partial[local_out * 2][warp_in_out] = acc1;
        if (have_n2) s_partial[local_out * 2 + 1][warp_in_out] = acc2;
    }
    __syncthreads();

    // Final reduction: thread 0 of each output element
    unsigned int warps_per_out = threads_per_out / WARP_SIZE;
    if (lane_in_warp == 0 && warp_in_out == 0) {
        float sum1 = 0.0f;
        for (unsigned int w = 0; w < warps_per_out; w++) sum1 += s_partial[local_out * 2][w];
        C[n1] = __float2bfloat16(sum1);

        if (have_n2) {
            float sum2 = 0.0f;
            for (unsigned int w = 0; w < warps_per_out; w++) sum2 += s_partial[local_out * 2 + 1][w];
            C[n2] = __float2bfloat16(sum2);
        }
    }
}

// Assemble Q for absorbed MLA: copies Q_absorbed + Q_rope into contiguous [Lkv+R] per head.
// Also handles RoPE application to Q_rope and K_rope.
//
// Grid: (num_heads, 1, 1)  Block: (max(Lkv, R), 1, 1)
// NOT IMPLEMENTED YET — use D2D copies for now.

// Fused Q_rope extract + writeback: eliminates 64 D2D copies per layer.
// Extracts Q_rope from q_full[nq, hd] at offset nope per head,
// then writes to q_absorbed_buf at offset kv_lora per head (stride mla_cache_dim).
//
// Grid: (1, 1, 1)  Block: (256, 1, 1)
// Each thread handles ceil(nq * rope / 256) elements.
extern "C" __global__ void mla_q_rope_scatter(
    const __nv_bfloat16* __restrict__ q_full,      // [nq, hd]
    __nv_bfloat16* __restrict__ q_absorbed_buf,     // [nq, mla_cache_dim]
    __nv_bfloat16* __restrict__ q_rope_contiguous,  // [nq * rope] for RoPE kernel
    unsigned int nq,
    unsigned int hd,            // head_dim (512)
    unsigned int nope,          // nope head dim (448)
    unsigned int rope,          // rope head dim (64)
    unsigned int kv_lora,       // kv_lora_rank (512)
    unsigned int mla_cache_dim  // kv_lora + rope (576)
) {
    unsigned int total = nq * rope;
    for (unsigned int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        unsigned int head = idx / rope;
        unsigned int r = idx % rope;
        // Read from q_full[head * hd + nope + r]
        __nv_bfloat16 val = q_full[head * hd + nope + r];
        // Write to BOTH destinations in one pass (eliminates separate extract loop)
        q_absorbed_buf[head * mla_cache_dim + kv_lora + r] = val;
        q_rope_contiguous[head * rope + r] = val;
    }
}

// Scatter RoPE'd Q_rope back to strided q_absorbed_buf layout.
// After RoPE, q_rope_direct is [nq, rope] contiguous.
// Write to q_absorbed_buf[head * mla_cache_dim + kv_lora .. + kv_lora + rope].
extern "C" __global__ void mla_q_rope_writeback(
    const __nv_bfloat16* __restrict__ q_rope_direct,   // [nq * rope] contiguous
    __nv_bfloat16* __restrict__ q_absorbed_buf,         // [nq, mla_cache_dim]
    unsigned int nq,
    unsigned int rope,
    unsigned int kv_lora,
    unsigned int mla_cache_dim
) {
    unsigned int total = nq * rope;
    for (unsigned int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        unsigned int head = idx / rope;
        unsigned int r = idx % rope;
        q_absorbed_buf[head * mla_cache_dim + kv_lora + r] = q_rope_direct[head * rope + r];
    }
}

// ════════════════════════════════════════════════════════════════════════════
// BATCHED PREFILL VARIANTS — eliminate per-token per-head D2D copy loops
// ════════════════════════════════════════════════════════════════════════════

// Extract Q rope portions from expanded Q[N, nq, hd] into contiguous [N, nq, rope] for RoPE.
// Replaces: for t in 0..N { for h in 0..nq { copy_d2d(q_full[t,h,nope:], q_rope[t,h]) } }
// Grid: (ceil(total/256), 1, 1)  Block: (256, 1, 1)  where total = num_tokens * nq * rope
extern "C" __global__ void mla_q_rope_extract_batched(
    const __nv_bfloat16* __restrict__ q_full,     // [N, q_dim] where q_dim = nq * hd
    __nv_bfloat16* __restrict__ q_rope_out,        // [N, nq * rope] contiguous
    unsigned int num_tokens,
    unsigned int nq,
    unsigned int hd,
    unsigned int nope,
    unsigned int rope,
    unsigned int q_dim                              // nq * hd
) {
    unsigned int total = num_tokens * nq * rope;
    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
        unsigned int t = idx / (nq * rope);
        unsigned int rem = idx % (nq * rope);
        unsigned int head = rem / rope;
        unsigned int r = rem % rope;
        q_rope_out[t * nq * rope + head * rope + r] =
            q_full[t * q_dim + head * hd + nope + r];
    }
}

// Write back RoPE'd Q rope portions into expanded Q[N, nq, hd] at offset nope per head.
// Replaces: for t in 0..N { for h in 0..nq { copy_d2d(q_rope[t,h], q_full[t,h,nope:]) } }
// Grid: (ceil(total/256), 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void mla_q_rope_writeback_batched(
    const __nv_bfloat16* __restrict__ q_rope_in,  // [N, nq * rope] contiguous
    __nv_bfloat16* __restrict__ q_full,            // [N, q_dim]
    unsigned int num_tokens,
    unsigned int nq,
    unsigned int hd,
    unsigned int nope,
    unsigned int rope,
    unsigned int q_dim
) {
    unsigned int total = num_tokens * nq * rope;
    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
        unsigned int t = idx / (nq * rope);
        unsigned int rem = idx % (nq * rope);
        unsigned int head = rem / rope;
        unsigned int r = rem % rope;
        q_full[t * q_dim + head * hd + nope + r] =
            q_rope_in[t * nq * rope + head * rope + r];
    }
}

// Assemble K=[nope|rope] and extract V from kv_expanded for N tokens.
// K: concatenate k_nope (from kv_expanded per head) with k_rope (broadcast from single head).
// V: extract v_dim portion from kv_expanded per head.
// Replaces: for t in 0..N { for h in 0..nkv { 3 copy_d2d calls } }
// Grid: (num_tokens, 2, 1)  Block: (256, 1, 1)
//   blockIdx.y==0: assemble K [nkv * hd elements per token]
//   blockIdx.y==1: extract V [nkv * v_dim elements per token]
extern "C" __global__ void mla_kv_assemble_batched(
    const __nv_bfloat16* __restrict__ kv_expanded,  // [N, nkv * (nope + v_dim)]
    const __nv_bfloat16* __restrict__ k_rope_buf,   // [N, rope]
    __nv_bfloat16* __restrict__ k_out,               // [N, nkv * hd] where hd = nope + rope
    __nv_bfloat16* __restrict__ v_out,               // [N, nkv * v_dim]
    unsigned int nkv,
    unsigned int nope,
    unsigned int v_dim,
    unsigned int rope,
    unsigned int hd,                                 // nope + rope (= K head dim)
    unsigned int kv_expanded_stride                  // nkv * (nope + v_dim) per token
) {
    unsigned int t = blockIdx.x;  // token index

    if (blockIdx.y == 0) {
        // Assemble K: [nkv, hd] where hd = nope + rope
        unsigned int k_total = nkv * hd;
        for (unsigned int idx = threadIdx.x; idx < k_total; idx += blockDim.x) {
            unsigned int head = idx / hd;
            unsigned int dim = idx % hd;
            __nv_bfloat16 val;
            if (dim < nope) {
                // k_nope from kv_expanded[t, head, dim]
                val = kv_expanded[(unsigned long long)t * kv_expanded_stride + head * (nope + v_dim) + dim];
            } else {
                // k_rope broadcast from single-head k_rope_buf[t, dim - nope]
                val = k_rope_buf[(unsigned long long)t * rope + (dim - nope)];
            }
            k_out[(unsigned long long)t * nkv * hd + idx] = val;
        }
    } else {
        // Extract V: [nkv, v_dim]
        unsigned int v_total = nkv * v_dim;
        for (unsigned int idx = threadIdx.x; idx < v_total; idx += blockDim.x) {
            unsigned int head = idx / v_dim;
            unsigned int dim = idx % v_dim;
            // V is at offset nope within each head's (nope + v_dim) block
            v_out[(unsigned long long)t * nkv * v_dim + idx] =
                kv_expanded[(unsigned long long)t * kv_expanded_stride + head * (nope + v_dim) + nope + dim];
        }
    }
}

// Assemble compressed MLA cache entries for N tokens.
// K_cache = [kv_latent(kv_lora) | k_rope(rope)] per token
// V_cache = [kv_latent(kv_lora) | zeros(rope)] per token
// Replaces: for t in 0..N { 4 copy_d2d/memset calls }
// Grid: (num_tokens, 1, 1)  Block: (mla_cache_dim or 256, 1, 1)
extern "C" __global__ void mla_cache_assemble_batched(
    const __nv_bfloat16* __restrict__ kv_latent,    // [N, kv_lora]
    const __nv_bfloat16* __restrict__ k_rope,       // [N, rope]
    __nv_bfloat16* __restrict__ k_cache,             // [N, mla_cache_dim]
    __nv_bfloat16* __restrict__ v_cache,             // [N, mla_cache_dim]
    unsigned int kv_lora,
    unsigned int rope,
    unsigned int mla_cache_dim                       // kv_lora + rope
) {
    unsigned int t = blockIdx.x;
    unsigned long long k_off = (unsigned long long)t * mla_cache_dim;
    unsigned long long lat_off = (unsigned long long)t * kv_lora;
    unsigned long long rope_off = (unsigned long long)t * rope;

    for (unsigned int idx = threadIdx.x; idx < mla_cache_dim; idx += blockDim.x) {
        if (idx < kv_lora) {
            __nv_bfloat16 val = kv_latent[lat_off + idx];
            k_cache[k_off + idx] = val;
            v_cache[k_off + idx] = val;
        } else {
            unsigned int r = idx - kv_lora;
            k_cache[k_off + idx] = k_rope[rope_off + r];
            v_cache[k_off + idx] = __float2bfloat16(0.0f);
        }
    }
}

// Assemble Q_final from Q_absorbed and Q_rope: [absorbed(kv_lora)|rope(rope)] per head per token.
// Q_absorbed: [N, nq * kv_lora] contiguous
// Q_rope: [N, nq * rope] contiguous
// Q_final: [N, nq * mla_cache_dim] where mla_cache_dim = kv_lora + rope
// Grid: (ceil(total/256), 1, 1) where total = N * nq * mla_cache_dim
// Block: (256, 1, 1)
extern "C" __global__ void mla_q_final_assemble_batched(
    const __nv_bfloat16* __restrict__ q_absorbed,  // [N, nq * kv_lora]
    const __nv_bfloat16* __restrict__ q_rope,      // [N, nq * rope]
    __nv_bfloat16* __restrict__ q_final,           // [N, nq * mla_cache_dim]
    unsigned int num_tokens,
    unsigned int nq,
    unsigned int kv_lora,
    unsigned int rope,
    unsigned int mla_cache_dim   // kv_lora + rope
) {
    unsigned int total = num_tokens * nq * mla_cache_dim;
    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
        unsigned int t = idx / (nq * mla_cache_dim);
        unsigned int rem = idx % (nq * mla_cache_dim);
        unsigned int head = rem / mla_cache_dim;
        unsigned int d = rem % mla_cache_dim;
        if (d < kv_lora) {
            q_final[idx] = q_absorbed[t * nq * kv_lora + head * kv_lora + d];
        } else {
            q_final[idx] = q_rope[t * nq * rope + head * rope + (d - kv_lora)];
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// DECODE SINGLE-TOKEN VARIANTS (existing)
// ════════════════════════════════════════════════════════════════════════════

// Fused KV cache assembly: concatenate [kv_latent | k_rope] → K_cache and [kv_latent | zeros] → V_cache.
// Eliminates 4 D2D copies + 1 memset per decode step.
extern "C" __global__ void mla_cache_assemble(
    const __nv_bfloat16* __restrict__ kv_latent,  // [kv_lora]
    const __nv_bfloat16* __restrict__ k_rope,     // [rope]
    __nv_bfloat16* __restrict__ k_cache_entry,     // [mla_cache_dim]
    __nv_bfloat16* __restrict__ v_cache_entry,     // [mla_cache_dim]
    unsigned int kv_lora,
    unsigned int rope,
    unsigned int mla_cache_dim
) {
    unsigned int idx = threadIdx.x;
    // K = [latent | k_rope]
    if (idx < kv_lora) {
        k_cache_entry[idx] = kv_latent[idx];
        v_cache_entry[idx] = kv_latent[idx];
    } else if (idx < mla_cache_dim) {
        unsigned int r = idx - kv_lora;
        k_cache_entry[idx] = (r < rope) ? k_rope[r] : __float2bfloat16(0.0f);
        v_cache_entry[idx] = __float2bfloat16(0.0f); // V padding = zeros
    }
}
