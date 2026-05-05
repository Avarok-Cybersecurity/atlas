// SPDX-License-Identifier: AGPL-3.0-only

// Fused MLA Prefill: Q_absorption + Attention + V_extraction in one kernel.
//
// Eliminates 6 kernel launches and all intermediate buffer traffic per layer.
// Each CTA handles one (query_token, head) pair end-to-end:
//   1. Q_absorbed[256] = Q_nope[64] @ W_UK[256,64]^T
//   2. Q_final[320] = [Q_absorbed | Q_rope_rotated]
//   3. Online softmax attention over all KV tokens
//   4. V_out[128] = attn_latent[256] @ W_UV[128,256]^T
//
// Grid: (num_heads, num_q_tokens, 1)
// Block: (256, 1, 1)
//
// Memory: W_UK and W_UV read from global (L2 cached per head, 32KB + 64KB).
// No shared memory needed for weights — register-file dot products.

#include <cuda_bf16.h>
#include <float.h>

extern "C" __global__ void mla_fused_prefill(
    // Q inputs (from wq_b output)
    const __nv_bfloat16* __restrict__ q_full,       // [N, nq * hd]
    const __nv_bfloat16* __restrict__ q_rope,       // [N, nq * rope] (already RoPE'd)
    // KV inputs (from wkv_a + norm + wkv_a_rope + RoPE)
    const __nv_bfloat16* __restrict__ kv_latent,    // [N, kv_lora]
    const __nv_bfloat16* __restrict__ k_rope,       // [N, rope] (already RoPE'd)
    // Weights
    const __nv_bfloat16* __restrict__ w_uk,         // [nq * kv_lora, nope] row-major per head
    const __nv_bfloat16* __restrict__ w_uv,         // [nq * v_dim, kv_lora] row-major per head
    // Output
    __nv_bfloat16* __restrict__ v_out,              // [N, nq * v_dim]
    // KV cache write (optional — write compressed cache for decode)
    __nv_bfloat16* __restrict__ k_cache_out,        // [N, kv_lora + rope] or NULL
    __nv_bfloat16* __restrict__ v_cache_out,        // [N, kv_lora + rope] or NULL
    // Dimensions
    unsigned int seq_len,       // N (number of tokens)
    unsigned int nq,            // num Q heads (32)
    unsigned int nope,          // 64
    unsigned int rope_dim,      // 64
    unsigned int kv_lora,       // 256
    unsigned int v_dim,         // 128
    unsigned int hd,            // nope + rope = 128
    float inv_sqrt_d            // 1/sqrt(320)
) {
    const unsigned int head = blockIdx.x;
    const unsigned int q_pos = blockIdx.y;
    const unsigned int tid = threadIdx.x;  // 0..255

    if (head >= nq || q_pos >= seq_len) return;

    const unsigned int mla_cache_dim = kv_lora + rope_dim; // 320

    // ═══════════════════════════════════════════════════════════════
    // Step 1: Q absorption — Q_absorbed[256] = Q_nope[64] @ W_UK^T
    // ═══════════════════════════════════════════════════════════════
    // Each of 256 threads produces 1 output element of Q_absorbed
    // (for tid < 256, which covers all kv_lora=256 outputs)

    // Load Q_nope[64] into registers (shared across all threads via L1)
    const __nv_bfloat16* q_nope_ptr = q_full + (unsigned long long)q_pos * nq * hd + head * hd;
    // W_UK for this head: [kv_lora, nope] at offset head * kv_lora * nope
    const __nv_bfloat16* w_uk_head = w_uk + (unsigned long long)head * kv_lora * nope;

    float q_absorbed_val = 0.0f;
    if (tid < kv_lora) {
        // Dot product: W_UK[tid, :] · Q_nope[:]
        const __nv_bfloat16* w_row = w_uk_head + (unsigned long long)tid * nope;
        for (unsigned int k = 0; k < nope; k++) {
            q_absorbed_val += __bfloat162float(w_row[k]) * __bfloat162float(q_nope_ptr[k]);
        }
    }

    // Store Q_absorbed in shared memory for attention step
    __shared__ float smem_q[320];  // Q_final = [Q_absorbed(256) | Q_rope(64)]
    if (tid < kv_lora) {
        smem_q[tid] = q_absorbed_val;
    }

    // Load Q_rope into smem
    const __nv_bfloat16* q_rope_ptr = q_rope + (unsigned long long)q_pos * nq * rope_dim + head * rope_dim;
    if (tid < rope_dim) {
        smem_q[kv_lora + tid] = __bfloat162float(q_rope_ptr[tid]);
    }
    __syncthreads();

    // ═══════════════════════════════════════════════════════════════
    // Step 2: Write KV cache (if cache pointers provided)
    // ═══════════════════════════════════════════════════════════════
    // Only need to write once per token (not per head).
    // Use head==0 to write, other heads skip.
    if (head == 0 && k_cache_out != 0) {
        // K_cache = [kv_latent | k_rope], V_cache = [kv_latent | zeros]
        if (tid < kv_lora) {
            __nv_bfloat16 lat_val = kv_latent[q_pos * kv_lora + tid];
            k_cache_out[q_pos * mla_cache_dim + tid] = lat_val;
            v_cache_out[q_pos * mla_cache_dim + tid] = lat_val;
        } else if (tid < mla_cache_dim) {
            unsigned int r = tid - kv_lora;
            k_cache_out[q_pos * mla_cache_dim + tid] = (r < rope_dim) ?
                k_rope[q_pos * rope_dim + r] : __float2bfloat16(0.0f);
            v_cache_out[q_pos * mla_cache_dim + tid] = __float2bfloat16(0.0f);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Step 3: Online softmax attention over KV tokens
    // ═══════════════════════════════════════════════════════════════
    // Q_final is in smem_q[320]. For each KV token, compute dot product.
    // 256 threads collaborate to reduce 320 dims.
    // Each thread handles ceil(320/256) = 2 dims (with some idle).

    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;
    // Accumulate weighted KV latent (only first 256 dims for V extraction)
    float acc_latent[2] = {0.0f, 0.0f};  // each thread accumulates 1-2 latent dims
    // Thread tid handles latent dims: tid, tid+256 (if < kv_lora)
    // But we need to map threads to latent dims for V extraction accumulation.
    // Simple: tid < 256, each thread accumulates latent[tid] weighted by attention.

    unsigned int kv_end = min(q_pos + 1, seq_len); // causal
    for (unsigned int kv_pos = 0; kv_pos < kv_end; kv_pos++) {
        // Dot product: Q_final[320] · [kv_latent[256] | k_rope[64]]
        const __nv_bfloat16* kv_lat_row = kv_latent + (unsigned long long)kv_pos * kv_lora;
        const __nv_bfloat16* k_rope_row = k_rope + (unsigned long long)kv_pos * rope_dim;

        // Each thread computes partial dot product over ~2 dims
        float dot = 0.0f;
        // Latent portion: dims 0..255
        if (tid < kv_lora) {
            dot += smem_q[tid] * __bfloat162float(kv_lat_row[tid]);
        }
        // Rope portion: dims 256..319 (only first 64 threads)
        if (tid < rope_dim) {
            dot += smem_q[kv_lora + tid] * __bfloat162float(k_rope_row[tid]);
        }

        // Warp reduction (8 warps × 32 threads)
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);
        }
        // Lane 0 of each warp has partial sum. Reduce across warps via shared memory.
        __shared__ float smem_dot[8];  // 8 warps
        unsigned int warp_id = tid / 32;
        unsigned int lane_id = tid % 32;
        if (lane_id == 0) {
            smem_dot[warp_id] = dot;
        }
        __syncthreads();

        float score;
        if (tid == 0) {
            score = 0.0f;
            for (int w = 0; w < 8; w++) score += smem_dot[w];
            score *= inv_sqrt_d;
            smem_dot[0] = score;  // broadcast
        }
        __syncthreads();
        score = smem_dot[0];

        // Online softmax
        float m_new = fmaxf(m_prev, score);
        float alpha = expf(m_prev - m_new);
        float p = expf(score - m_new);
        float l_new = alpha * l_prev + p;

        // Update latent accumulator: acc_latent = alpha * acc_latent + p * kv_latent[kv_pos]
        if (tid < kv_lora) {
            acc_latent[0] = alpha * acc_latent[0] + p * __bfloat162float(kv_lat_row[tid]);
        }

        m_prev = m_new;
        l_prev = l_new;
        __syncthreads();
    }

    // Normalize by softmax denominator
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    if (tid < kv_lora) {
        acc_latent[0] *= inv_l;
    }

    // ═══════════════════════════════════════════════════════════════
    // Step 4: V extraction — V_out[128] = attn_latent[256] @ W_UV^T
    // ═══════════════════════════════════════════════════════════════
    // Store attn_latent to shared memory for all threads to read
    __shared__ float smem_latent[256];
    if (tid < kv_lora) {
        smem_latent[tid] = acc_latent[0];
    }
    __syncthreads();

    // W_UV for this head: [v_dim, kv_lora] at offset head * v_dim * kv_lora
    const __nv_bfloat16* w_uv_head = w_uv + (unsigned long long)head * v_dim * kv_lora;

    if (tid < v_dim) {
        // Dot product: W_UV[tid, :] · attn_latent[:]
        const __nv_bfloat16* w_row = w_uv_head + (unsigned long long)tid * kv_lora;
        float v_val = 0.0f;
        for (unsigned int l = 0; l < kv_lora; l++) {
            v_val += __bfloat162float(w_row[l]) * smem_latent[l];
        }
        v_out[(unsigned long long)q_pos * nq * v_dim + head * v_dim + tid] = __float2bfloat16(v_val);
    }
}
