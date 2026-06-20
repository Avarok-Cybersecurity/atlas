// SPDX-License-Identifier: AGPL-3.0-only

// Fused MLA Prefill: Q_absorption + Attention + V_extraction in one kernel.
//
// Eliminates 6 kernel launches and all intermediate buffer traffic per layer.
// Each CTA handles one (query_token, head) pair end-to-end:
//   1. Q_absorbed[512] = Q_nope[448] @ W_UK[512,448]^T
//   2. Q_final[576] = [Q_absorbed | Q_rope_rotated]
//   3. Online softmax attention over all KV tokens
//   4. V_out[512] = attn_latent[512] @ W_UV[512,512]^T
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
    unsigned int nq,            // num Q heads (64)
    unsigned int nope,          // 448
    unsigned int rope_dim,      // 64
    unsigned int kv_lora,       // 512
    unsigned int v_dim,         // 512
    unsigned int hd,            // nope + rope = 512
    unsigned int num_kv_heads,  // 1 (GQA ratio = nq / nkv)
    float inv_sqrt_d            // 1/sqrt(576)
) {
    const unsigned int head = blockIdx.x;
    const unsigned int q_pos = blockIdx.y;
    const unsigned int tid = threadIdx.x;  // 0..255

    if (head >= nq || q_pos >= seq_len) return;

    const unsigned int mla_cache_dim = kv_lora + rope_dim; // 576
    const unsigned int gqa_ratio = nq / max(num_kv_heads, 1u);
    const unsigned int kv_head = head / gqa_ratio;

    // ═══════════════════════════════════════════════════════════════
    // Step 1: Q absorption — Q_absorbed[512] = Q_nope[448] @ W_UK^T
    // ═══════════════════════════════════════════════════════════════
    // 256 threads cover all 512 outputs via strided loop (tid, tid+256).

    // Load Q_nope[448] into registers (shared across all threads via L1)
    const __nv_bfloat16* q_nope_ptr = q_full + (unsigned long long)q_pos * nq * hd + head * hd;
    // W_UK for this KV head: [kv_lora, nope] at offset kv_head * kv_lora * nope
    const __nv_bfloat16* w_uk_head = w_uk + (unsigned long long)kv_head * kv_lora * nope;

    __shared__ float smem_q[576];  // Q_final = [Q_absorbed(512) | Q_rope(64)]

    for (unsigned int idx = tid; idx < kv_lora; idx += blockDim.x) {
        // Dot product: W_UK[idx, :] · Q_nope[:]
        const __nv_bfloat16* w_row = w_uk_head + (unsigned long long)idx * nope;
        float q_absorbed_val = 0.0f;
        for (unsigned int k = 0; k < nope; k++) {
            q_absorbed_val += __bfloat162float(w_row[k]) * __bfloat162float(q_nope_ptr[k]);
        }
        smem_q[idx] = q_absorbed_val;
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
        // Latent portion: all 512 dims via strided loop
        for (unsigned int idx = tid; idx < kv_lora; idx += blockDim.x) {
            __nv_bfloat16 lat_val = kv_latent[q_pos * kv_lora + idx];
            k_cache_out[q_pos * mla_cache_dim + idx] = lat_val;
            v_cache_out[q_pos * mla_cache_dim + idx] = lat_val;
        }
        // Rope + zero padding portion: dims 512..575
        for (unsigned int idx = tid + kv_lora; idx < mla_cache_dim; idx += blockDim.x) {
            unsigned int r = idx - kv_lora;
            k_cache_out[q_pos * mla_cache_dim + idx] = (r < rope_dim) ?
                k_rope[q_pos * rope_dim + r] : __float2bfloat16(0.0f);
            v_cache_out[q_pos * mla_cache_dim + idx] = __float2bfloat16(0.0f);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Step 3: Online softmax attention over KV tokens
    // ═══════════════════════════════════════════════════════════════
    // Q_final is in smem_q[576]. For each KV token, compute dot product.
    // 256 threads collaborate to reduce 576 dims.
    // Latent portion (512) uses strided loop; rope (64) uses first 64 threads.

    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;
    // Accumulate weighted KV latent — each thread handles 2 dims (tid, tid+256)
    float acc_latent[2] = {0.0f, 0.0f};

    unsigned int kv_end = min(q_pos + 1, seq_len); // causal
    for (unsigned int kv_pos = 0; kv_pos < kv_end; kv_pos++) {
        // Dot product: Q_final[576] · [kv_latent[512] | k_rope[64]]
        const __nv_bfloat16* kv_lat_row = kv_latent + (unsigned long long)kv_pos * kv_lora;
        const __nv_bfloat16* k_rope_row = k_rope + (unsigned long long)kv_pos * rope_dim;

        // Each thread computes partial dot product over ~3 dims (576/256)
        float dot = 0.0f;
        // Latent portion: dims 0..511 via strided loop
        for (unsigned int idx = tid; idx < kv_lora; idx += blockDim.x) {
            dot += smem_q[idx] * __bfloat162float(kv_lat_row[idx]);
        }
        // Rope portion: dims 512..575 (only first 64 threads)
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
        for (unsigned int i = 0; i < 2; i++) {
            unsigned int idx = tid + i * blockDim.x;
            if (idx < kv_lora) {
                acc_latent[i] = alpha * acc_latent[i] + p * __bfloat162float(kv_lat_row[idx]);
            }
        }

        m_prev = m_new;
        l_prev = l_new;
        __syncthreads();
    }

    // Normalize by softmax denominator
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    for (unsigned int i = 0; i < 2; i++) {
        unsigned int idx = tid + i * blockDim.x;
        if (idx < kv_lora) {
            acc_latent[i] *= inv_l;
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Step 4: V extraction — V_out[512] = attn_latent[512] @ W_UV^T
    // ═══════════════════════════════════════════════════════════════
    // Store attn_latent to shared memory for all threads to read
    __shared__ float smem_latent[512];
    for (unsigned int i = 0; i < 2; i++) {
        unsigned int idx = tid + i * blockDim.x;
        if (idx < kv_lora) {
            smem_latent[idx] = acc_latent[i];
        }
    }
    __syncthreads();

    // W_UV for this KV head: [v_dim, kv_lora] at offset kv_head * v_dim * kv_lora
    const __nv_bfloat16* w_uv_head = w_uv + (unsigned long long)kv_head * v_dim * kv_lora;

    for (unsigned int idx = tid; idx < v_dim; idx += blockDim.x) {
        // Dot product: W_UV[idx, :] · attn_latent[:]
        const __nv_bfloat16* w_row = w_uv_head + (unsigned long long)idx * kv_lora;
        float v_val = 0.0f;
        for (unsigned int l = 0; l < kv_lora; l++) {
            v_val += __bfloat162float(w_row[l]) * smem_latent[l];
        }
        v_out[(unsigned long long)q_pos * nq * v_dim + head * v_dim + idx] = __float2bfloat16(v_val);
    }
}
