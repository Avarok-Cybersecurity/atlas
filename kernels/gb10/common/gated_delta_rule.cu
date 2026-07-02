// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Gated Delta Rule — Core SSM for Qwen3-Next linear attention layers.
//
// Implements the recurrent gated delta rule update for DECODE mode
// (single token per step). This is the critical path for autoregressive
// inference.
//
// State equation (per head):
//   h_t = exp(g_t) * h_{t-1} + k_t ⊗ v_t'
//   where v_t' = (v_t - h_{t-1}^T @ k_t) * beta_t
//   output_t = h_t^T @ q_t
//
// Dimensions (Qwen3-Next):
//   num_key_heads: 16    (K heads)
//   num_value_heads: 32  (V heads)
//   key_head_dim: 128
//   value_head_dim: 128
//   head_repeat: 2       (value_heads / key_heads)
//
// For decode: one token at a time, state persists between calls.
//
// State h: [batch, num_v_heads, k_dim, v_dim] FP32 (transposed for coalescing)
//          = [batch, 32, 128, 128]
//          Each head stores a 128x128 matrix.
//          v_dim is the FAST dimension — threads map to v_dim for coalesced access.
//
// This kernel processes all heads for one batch element.

#include <cuda_bf16.h>

// Thread block size for matvec operations
#define BLOCK_SIZE 128

// SSM state normalization: clamp per-head h_state norm to prevent
// catastrophic state explosion on long sequences (Stuffed Mamba, 2024).
// MAX_STATE_NORM: if ||h[head]||_F > this, scale h down. 0 = disabled.
//
// Tuning: 100.0 was too aggressive — at 6K+ tokens with large system prompts,
// state norms legitimately reach 200-500 and clipping destroys information
// needed for instruction following. 1000.0 allows natural state growth while
// still preventing numerical overflow (FP32 max ~ 3.4e38).
#ifndef SSM_STATE_NORM_ENABLED
#define SSM_STATE_NORM_ENABLED
#define SSM_STATE_MAX_NORM 1000.0f
#endif

// ============================================================
// DECODE: Recurrent gated delta rule (single token)
// ============================================================
// Grid: (num_v_heads, batch, 1)
// Block: (BLOCK_SIZE, 1, 1)
//
// Each block handles one (batch, head) pair.
// BLOCK_SIZE threads cooperate on the 128-dim matvec.
//
// State layout: H[k_dim, v_dim] — v_dim is contiguous (fast dimension).
// Thread tid maps to v_dim index tid. All threads in a warp access
// consecutive addresses → perfectly coalesced memory access.
extern "C" __global__ void gated_delta_rule_decode(
    // State (in/out): [batch, num_v_heads, k_dim, v_dim] FP32
    float* __restrict__ h_state,
    // Inputs (BF16, from projections):
    const __nv_bfloat16* __restrict__ query,   // [batch, num_k_heads, k_dim]
    const __nv_bfloat16* __restrict__ key,     // [batch, num_k_heads, k_dim]
    const __nv_bfloat16* __restrict__ value,   // [batch, num_v_heads, v_dim]
    // Scalars per head:
    const float* __restrict__ gate,            // [batch, num_v_heads] exp(g_t) decay
    const float* __restrict__ beta,            // [batch, num_v_heads] sigmoid(b_t)
    // Output:
    __nv_bfloat16* __restrict__ output,        // [batch, num_v_heads, v_dim]
    // Dimensions:
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim
) {
    const unsigned int vh = blockIdx.x;    // value head index
    const unsigned int b = blockIdx.y;     // batch index
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;

    // Map value head to key head (head_repeat = num_v_heads / num_k_heads)
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    // H is [k_dim, v_dim] with v_dim contiguous — tid indexes v_dim
    float* H = h_state + ((b * num_v_heads + vh) * k_dim * v_dim);
    const __nv_bfloat16* q_ptr = query + (b * num_k_heads + kh) * k_dim;
    const __nv_bfloat16* k_ptr = key + (b * num_k_heads + kh) * k_dim;
    const __nv_bfloat16* v_ptr = value + (b * num_v_heads + vh) * v_dim;

    // Gate decay clamped to (0, 1) to prevent H-state explosion (g > 1) or
    // sign inversion (g < 0). The reference impl uses exp(-softplus(alpha))
    // which naturally constrains to (0, 1), but FP32 rounding or extreme
    // activations could push outside this range.
    float g_raw = gate[b * num_v_heads + vh];
    const float g = fminf(fmaxf(g_raw, 1e-6f), 1.0f - 1e-6f);
    const float bt = beta[b * num_v_heads + vh];    // beta gating

    // Shared memory for key and query vectors (k_dim elements each)
    __shared__ float smem_k[128];
    __shared__ float smem_q[128];

    // Load key and query into shared memory (coalesced, enables broadcast)
    if (tid < k_dim) {
        smem_k[tid] = (float)k_ptr[tid];
        smem_q[tid] = (float)q_ptr[tid];
    }
    __syncthreads();

    // Each thread handles one v_dim element (column of transposed H).
    // H[j][tid] for all j in k_dim — reads are coalesced across threads.
    if (tid < v_dim) {
        float v_i = (float)v_ptr[tid];

        // Step 1: hk_dot = sum_j H[j][tid] * k[j] — coalesced reads
        float hk_dot = 0.0f;
        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H[(j + 0) * v_dim + tid];
            float h1 = H[(j + 1) * v_dim + tid];
            float h2 = H[(j + 2) * v_dim + tid];
            float h3 = H[(j + 3) * v_dim + tid];
            hk_dot += h0 * smem_k[j] + h1 * smem_k[j + 1]
                    + h2 * smem_k[j + 2] + h3 * smem_k[j + 3];
        }

        // Step 2: Gated residual value
        // HF applies decay BEFORE computing the correction:
        //   H_decayed = g * H; kv_mem = H_decayed^T @ k = g * (H^T @ k)
        //   delta = (v - kv_mem) * beta = (v - g * hk_dot) * beta
        float v_new_i = (v_i - g * hk_dot) * bt;

        // Steps 3+4 fused: State update + output dot product in single pass.
        // Coalesced reads/writes: all threads access H[j][0..v_dim-1] consecutively.
        float q_dot = 0.0f;
        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H[(j + 0) * v_dim + tid];
            float h1 = H[(j + 1) * v_dim + tid];
            float h2 = H[(j + 2) * v_dim + tid];
            float h3 = H[(j + 3) * v_dim + tid];
            h0 = g * h0 + smem_k[j]     * v_new_i;
            h1 = g * h1 + smem_k[j + 1] * v_new_i;
            h2 = g * h2 + smem_k[j + 2] * v_new_i;
            h3 = g * h3 + smem_k[j + 3] * v_new_i;
            H[(j + 0) * v_dim + tid] = h0;
            H[(j + 1) * v_dim + tid] = h1;
            H[(j + 2) * v_dim + tid] = h2;
            H[(j + 3) * v_dim + tid] = h3;
            q_dot += h0 * smem_q[j] + h1 * smem_q[j + 1]
                   + h2 * smem_q[j + 2] + h3 * smem_q[j + 3];
        }

        // ── SSM state normalization (Stuffed Mamba mitigation) ──
        // Clamp per-head h_state Frobenius norm to prevent state explosion.
        // Each thread holds the sum-of-squares for its v_dim column across all k_dim rows.
        #ifdef SSM_STATE_NORM_ENABLED
        {
            // local_sq already has sum over k_dim for this thread's v_dim slot
            float local_sq = 0.0f;
            for (unsigned int j = 0; j < k_dim; j++) {
                float hv = H[j * v_dim + tid];
                local_sq += hv * hv;
            }
            // Block-wide reduction to get full head norm
            // 128 threads = 4 warps
            unsigned int mask = __activemask();
            float warp_sum = local_sq;
            warp_sum += __shfl_down_sync(mask, warp_sum, 16);
            warp_sum += __shfl_down_sync(mask, warp_sum, 8);
            warp_sum += __shfl_down_sync(mask, warp_sum, 4);
            warp_sum += __shfl_down_sync(mask, warp_sum, 2);
            warp_sum += __shfl_down_sync(mask, warp_sum, 1);

            __shared__ float norm_sums[4];
            unsigned int warp_id = tid / 32;
            unsigned int lane_id = tid % 32;
            if (lane_id == 0) norm_sums[warp_id] = warp_sum;
            __syncthreads();

            float head_norm_sq;
            if (tid < 4) {
                float s = norm_sums[tid];
                s += __shfl_down_sync(0xf, s, 2);
                s += __shfl_down_sync(0xf, s, 1);
                norm_sums[0] = s;
            }
            __syncthreads();
            head_norm_sq = norm_sums[0];

            if (head_norm_sq > SSM_STATE_MAX_NORM * SSM_STATE_MAX_NORM) {
                float scale = SSM_STATE_MAX_NORM * rsqrtf(head_norm_sq);
                for (unsigned int j = 0; j < k_dim; j++) {
                    H[j * v_dim + tid] *= scale;
                }
            }
        }
        #endif

        // Scale output by 1/sqrt(k_dim) — matches HF reference line 535-536
        float inv_sqrt_d = rsqrtf((float)k_dim);
        output[(b * num_v_heads + vh) * v_dim + tid] = __float2bfloat16(q_dot * inv_sqrt_d);
    }
}

// ============================================================
// FP32 OUTPUT VARIANT: same recurrence but outputs FP32 instead of BF16.
// Eliminates the BF16 truncation bottleneck in the recurrent path,
// preventing cumulative precision drift at 15K+ tokens.
// ============================================================
extern "C" __global__ void gated_delta_rule_decode_f32(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ output,                   // FP32 output
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    if (tid >= v_dim) return;

    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    float* H = h_state + ((b * num_v_heads + vh) * k_dim * v_dim);
    const __nv_bfloat16* q_ptr = query + (b * num_k_heads + kh) * k_dim;
    const __nv_bfloat16* k_ptr = key + (b * num_k_heads + kh) * k_dim;
    const __nv_bfloat16* v_ptr = value + (b * num_v_heads + vh) * v_dim;

    float g_raw = gate[b * num_v_heads + vh];
    const float g = fminf(fmaxf(g_raw, 1e-6f), 1.0f - 1e-6f);
    const float bt = beta[b * num_v_heads + vh];

    __shared__ float smem_k[128];
    __shared__ float smem_q[128];

    if (tid < k_dim) {
        smem_k[tid] = (float)k_ptr[tid];
        smem_q[tid] = (float)q_ptr[tid];
    }
    __syncthreads();

    float v_i = (float)v_ptr[tid];
    float hk_dot = 0.0f;
    #pragma unroll 4
    for (unsigned int j = 0; j < k_dim; j += 4) {
        float h0 = H[(j + 0) * v_dim + tid];
        float h1 = H[(j + 1) * v_dim + tid];
        float h2 = H[(j + 2) * v_dim + tid];
        float h3 = H[(j + 3) * v_dim + tid];
        hk_dot += h0 * smem_k[j] + h1 * smem_k[j+1] + h2 * smem_k[j+2] + h3 * smem_k[j+3];
    }

    float v_new_i = (v_i - g * hk_dot) * bt;

    float q_dot = 0.0f;
    #pragma unroll 4
    for (unsigned int j = 0; j < k_dim; j += 4) {
        float h0 = H[(j + 0) * v_dim + tid];
        float h1 = H[(j + 1) * v_dim + tid];
        float h2 = H[(j + 2) * v_dim + tid];
        float h3 = H[(j + 3) * v_dim + tid];
        h0 = g * h0 + smem_k[j]     * v_new_i;
        h1 = g * h1 + smem_k[j + 1] * v_new_i;
        h2 = g * h2 + smem_k[j + 2] * v_new_i;
        h3 = g * h3 + smem_k[j + 3] * v_new_i;
        H[(j + 0) * v_dim + tid] = h0;
        H[(j + 1) * v_dim + tid] = h1;
        H[(j + 2) * v_dim + tid] = h2;
        H[(j + 3) * v_dim + tid] = h3;
        q_dot += h0 * smem_q[j] + h1 * smem_q[j+1] + h2 * smem_q[j+2] + h3 * smem_q[j+3];
    }

    #ifdef SSM_STATE_NORM_ENABLED
    {
        float local_sq = 0.0f;
        for (unsigned int j = 0; j < k_dim; j++) {
            float hv = H[j * v_dim + tid];
            local_sq += hv * hv;
        }
        for (int offset = 16; offset >= 1; offset >>= 1)
            local_sq += __shfl_down_sync(0xFFFFFFFF, local_sq, offset);
        __shared__ float norm_sums[4];
        if (tid % 32 == 0) norm_sums[tid / 32] = local_sq;
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for (int w = 0; w < 4; w++) total += norm_sums[w];
            norm_sums[0] = total;
        }
        __syncthreads();
        float head_norm_sq = norm_sums[0];
        if (head_norm_sq > SSM_STATE_MAX_NORM * SSM_STATE_MAX_NORM) {
            float scale = SSM_STATE_MAX_NORM * rsqrtf(head_norm_sq);
            for (unsigned int j = 0; j < k_dim; j++) {
                H[j * v_dim + tid] *= scale;
            }
        }
    }
    #endif

    float inv_sqrt_d = rsqrtf((float)k_dim);
    output[(b * num_v_heads + vh) * v_dim + tid] = q_dot * inv_sqrt_d;  // FP32 output!
}

// ============================================================
// CHUNK2: Fused 2-token GDN decode (speculative verification)
// ============================================================
// Processes exactly 2 tokens through GDN in a single kernel launch.
// Saves intermediate state H_1 for rollback on draft rejection.
//
// Memory traffic advantage over 2× sequential decode:
//   Sequential: H read × 4, H write × 4 (2 tokens × 2 passes each)
//   Chunk2:     H read × 2, H write × 1, H_inter write × 1, H_inter read × 1
//   The H_inter buffer stays in L2 cache (2 MB fits in GB10's 64 MB L2).
//
// Grid: (num_v_heads, batch, 1)
// Block: (BLOCK_SIZE, 1, 1)
extern "C" __global__ void gated_delta_rule_chunk2(
    // State (in/out): [batch, num_v_heads, k_dim, v_dim] FP32
    float* __restrict__ h_state,
    // Inputs for 2 tokens (BF16), accessed via stride params:
    const __nv_bfloat16* __restrict__ query,   // token t at: query + t * qk_stride + head * k_dim
    const __nv_bfloat16* __restrict__ key,     // token t at: key   + t * qk_stride + head * k_dim
    const __nv_bfloat16* __restrict__ value,   // token t at: value + t * v_stride  + head * v_dim
    // Scalars per token per head (FP32), accessed via gb_stride:
    const float* __restrict__ gate,            // token t at: gate + t * gb_stride + head
    const float* __restrict__ beta,            // token t at: beta + t * gb_stride + head
    // Output for 2 tokens:
    __nv_bfloat16* __restrict__ output,        // [batch, 2, num_v_heads, v_dim]
    // Intermediate H_1 for rollback:
    float* __restrict__ h_state_intermediate,  // [batch, num_v_heads, k_dim, v_dim] FP32
    // Dimensions:
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    // Strides (elements, not bytes) between token 0 and token 1:
    unsigned int qk_stride,     // BF16 elements between Q/K tokens
    unsigned int v_stride,      // BF16 elements between V tokens
    unsigned int gb_stride      // FP32 elements between gate/beta tokens
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    // State and intermediate pointers for this (batch, head)
    const unsigned int hv_size = k_dim * v_dim;
    float* H = h_state + ((b * num_v_heads + vh) * hv_size);
    float* H_inter = h_state_intermediate + ((b * num_v_heads + vh) * hv_size);

    // Input pointers: use caller-supplied strides for token dimension.
    // For batch b, token t: ptr + (b*2 + t) * stride + per_head_offset.

    // Token 0 inputs
    const __nv_bfloat16* q0 = query + (b * 2) * qk_stride + kh * k_dim;
    const __nv_bfloat16* k0 = key   + (b * 2) * qk_stride + kh * k_dim;
    const __nv_bfloat16* v0 = value + (b * 2) * v_stride  + vh * v_dim;
    const float g0 = gate[(b * 2) * gb_stride + vh];
    const float bt0 = beta[(b * 2) * gb_stride + vh];

    // Token 1 inputs
    const __nv_bfloat16* q1 = query + (b * 2 + 1) * qk_stride + kh * k_dim;
    const __nv_bfloat16* k1 = key   + (b * 2 + 1) * qk_stride + kh * k_dim;
    const __nv_bfloat16* v1 = value + (b * 2 + 1) * v_stride  + vh * v_dim;
    const float g1 = gate[(b * 2 + 1) * gb_stride + vh];
    const float bt1 = beta[(b * 2 + 1) * gb_stride + vh];

    // Load both token keys and queries into shared memory
    __shared__ float smem_k0[128];
    __shared__ float smem_q0[128];
    __shared__ float smem_k1[128];
    __shared__ float smem_q1[128];

    if (tid < k_dim) {
        smem_k0[tid] = (float)k0[tid];
        smem_q0[tid] = (float)q0[tid];
        smem_k1[tid] = (float)k1[tid];
        smem_q1[tid] = (float)q1[tid];
    }
    __syncthreads();

    if (tid < v_dim) {
        float vi0 = (float)v0[tid];
        float vi1 = (float)v1[tid];

        // ── Pass 1: Compute hk_dot_0 = H_0^T @ k_0 ──
        float hk0 = 0.0f;
        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H[(j + 0) * v_dim + tid];
            float h1 = H[(j + 1) * v_dim + tid];
            float h2 = H[(j + 2) * v_dim + tid];
            float h3 = H[(j + 3) * v_dim + tid];
            hk0 += h0 * smem_k0[j] + h1 * smem_k0[j + 1]
                 + h2 * smem_k0[j + 2] + h3 * smem_k0[j + 3];
        }

        // Token 0 gated residual
        float v_new_0 = (vi0 - g0 * hk0) * bt0;

        // ── Pass 2: H_0 → H_1, compute out_0, compute hk_dot_1 on H_1 ──
        float q0_dot = 0.0f;
        float hk1 = 0.0f;
        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H[(j + 0) * v_dim + tid];
            float h1 = H[(j + 1) * v_dim + tid];
            float h2 = H[(j + 2) * v_dim + tid];
            float h3 = H[(j + 3) * v_dim + tid];

            // Update: H_1 = g0 * H_0 + k0 ⊗ v_new_0
            h0 = g0 * h0 + smem_k0[j]     * v_new_0;
            h1 = g0 * h1 + smem_k0[j + 1] * v_new_0;
            h2 = g0 * h2 + smem_k0[j + 2] * v_new_0;
            h3 = g0 * h3 + smem_k0[j + 3] * v_new_0;

            // Save H_1 intermediate (for rollback on rejection)
            H_inter[(j + 0) * v_dim + tid] = h0;
            H_inter[(j + 1) * v_dim + tid] = h1;
            H_inter[(j + 2) * v_dim + tid] = h2;
            H_inter[(j + 3) * v_dim + tid] = h3;

            // Token 0 output: out_0 = H_1^T @ q_0
            q0_dot += h0 * smem_q0[j] + h1 * smem_q0[j + 1]
                    + h2 * smem_q0[j + 2] + h3 * smem_q0[j + 3];

            // Token 1: hk_dot_1 on H_1
            hk1 += h0 * smem_k1[j] + h1 * smem_k1[j + 1]
                 + h2 * smem_k1[j + 2] + h3 * smem_k1[j + 3];
        }

        // Token 1 gated residual
        float v_new_1 = (vi1 - g1 * hk1) * bt1;

        // ── Pass 3: H_1 → H_2, compute out_1 ──
        float q1_dot = 0.0f;
        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H_inter[(j + 0) * v_dim + tid];
            float h1 = H_inter[(j + 1) * v_dim + tid];
            float h2 = H_inter[(j + 2) * v_dim + tid];
            float h3 = H_inter[(j + 3) * v_dim + tid];

            // Update: H_2 = g1 * H_1 + k1 ⊗ v_new_1
            h0 = g1 * h0 + smem_k1[j]     * v_new_1;
            h1 = g1 * h1 + smem_k1[j + 1] * v_new_1;
            h2 = g1 * h2 + smem_k1[j + 2] * v_new_1;
            h3 = g1 * h3 + smem_k1[j + 3] * v_new_1;

            // Write final state H_2
            H[(j + 0) * v_dim + tid] = h0;
            H[(j + 1) * v_dim + tid] = h1;
            H[(j + 2) * v_dim + tid] = h2;
            H[(j + 3) * v_dim + tid] = h3;

            // Token 1 output: out_1 = H_2^T @ q_1
            q1_dot += h0 * smem_q1[j] + h1 * smem_q1[j + 1]
                    + h2 * smem_q1[j + 2] + h3 * smem_q1[j + 3];
        }

        // Scale outputs by 1/sqrt(k_dim)
        float inv_sqrt_d = rsqrtf((float)k_dim);
        unsigned int out_base0 = (b * 2 * num_v_heads + vh) * v_dim;
        unsigned int out_base1 = ((b * 2 + 1) * num_v_heads + vh) * v_dim;
        output[out_base0 + tid] = __float2bfloat16(q0_dot * inv_sqrt_d);
        output[out_base1 + tid] = __float2bfloat16(q1_dot * inv_sqrt_d);
    }
}

// ============================================================
// PREFILL: Chunked gated delta rule (sequential within chunk)
// ============================================================
// For prefill, we process the sequence sequentially per chunk.
// This is simpler than the FLA Triton kernel but correct.
//
// Supports strided access: Q/K/V/gate/beta may have different strides
// between tokens (e.g., Q/K at conv_dim stride, V at conv_dim stride,
// gate/beta at 2*num_v_heads stride). This eliminates the need to
// rearrange data into contiguous [seq_len, heads, dim] layout.
//
// Grid: (num_v_heads, batch, 1)
// Block: (BLOCK_SIZE, 1, 1)
//
// Each block handles one (batch, head) pair across all seq positions.
extern "C" __global__ void gated_delta_rule_prefill(
    // State (in/out): [batch, num_v_heads, k_dim, v_dim] FP32
    float* __restrict__ h_state,
    // Inputs (BF16):
    const __nv_bfloat16* __restrict__ query,   // Q for token t at: query + t * qk_stride + kh * k_dim
    const __nv_bfloat16* __restrict__ key,     // K for token t at: key   + t * qk_stride + kh * k_dim
    const __nv_bfloat16* __restrict__ value,   // V for token t at: value + t * v_stride  + vh * v_dim
    // Scalars per position per head:
    const float* __restrict__ gate,            // gate for token t at: gate + t * gb_stride + vh
    const float* __restrict__ beta,            // beta for token t at: beta + t * gb_stride + vh
    // Output:
    __nv_bfloat16* __restrict__ output,        // [batch, seq_len, num_v_heads, v_dim]
    // Dimensions:
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    // Strides (BF16 elements for Q/K/V, FP32 elements for gate/beta):
    unsigned int qk_stride,        // BF16 elements between consecutive tokens in Q/K
    unsigned int v_stride,         // BF16 elements between consecutive tokens in V
    unsigned int gb_stride         // FP32 elements between consecutive tokens in gate/beta
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    // H state in dynamic shared memory: [k_dim, v_dim] FP32
    // Allocated via launch parameter (k_dim * v_dim * 4 bytes = 64KB for 128×128)
    extern __shared__ float H_smem[];

    // Global H state pointer for this (batch, head)
    float* H_global = h_state + ((b * num_v_heads + vh) * k_dim * v_dim);

    // Cooperatively load H from global → shared memory
    // 128 threads, 16384 elements → 128 elements per thread
    for (unsigned int i = tid; i < k_dim * v_dim; i += blockDim.x) {
        H_smem[i] = H_global[i];
    }

    // Reuse tail of shared memory for k/q vectors (after H state)
    // H uses k_dim*v_dim floats; k and q each use k_dim floats
    float* smem_k = H_smem + k_dim * v_dim;
    float* smem_q = smem_k + k_dim;

    __syncthreads();

    float inv_sqrt_d = rsqrtf((float)k_dim);

    // Process each position sequentially — H stays in shared memory
    for (unsigned int t = 0; t < seq_len; t++) {
        const __nv_bfloat16* q_t = query + (unsigned long long)t * qk_stride + kh * k_dim;
        const __nv_bfloat16* k_t = key   + (unsigned long long)t * qk_stride + kh * k_dim;
        const __nv_bfloat16* v_t = value + (unsigned long long)t * v_stride  + vh * v_dim;

        float g_t = gate[(unsigned long long)t * gb_stride + vh];
        float bt = beta[(unsigned long long)t * gb_stride + vh];

        if (tid < k_dim) {
            smem_k[tid] = (float)k_t[tid];
            smem_q[tid] = (float)q_t[tid];
        }
        __syncthreads();

        if (tid < v_dim) {
            float v_i = (float)v_t[tid];

            float hk_dot = 0.0f;
            #pragma unroll 4
            for (unsigned int j = 0; j < k_dim; j += 4) {
                hk_dot += H_smem[(j + 0) * v_dim + tid] * smem_k[j]
                        + H_smem[(j + 1) * v_dim + tid] * smem_k[j + 1]
                        + H_smem[(j + 2) * v_dim + tid] * smem_k[j + 2]
                        + H_smem[(j + 3) * v_dim + tid] * smem_k[j + 3];
            }

            float v_new_i = (v_i - g_t * hk_dot) * bt;

            float q_dot = 0.0f;
            #pragma unroll 4
            for (unsigned int j = 0; j < k_dim; j += 4) {
                float h0 = g_t * H_smem[(j + 0) * v_dim + tid] + smem_k[j]     * v_new_i;
                float h1 = g_t * H_smem[(j + 1) * v_dim + tid] + smem_k[j + 1] * v_new_i;
                float h2 = g_t * H_smem[(j + 2) * v_dim + tid] + smem_k[j + 2] * v_new_i;
                float h3 = g_t * H_smem[(j + 3) * v_dim + tid] + smem_k[j + 3] * v_new_i;
                H_smem[(j + 0) * v_dim + tid] = h0;
                H_smem[(j + 1) * v_dim + tid] = h1;
                H_smem[(j + 2) * v_dim + tid] = h2;
                H_smem[(j + 3) * v_dim + tid] = h3;
                q_dot += h0 * smem_q[j] + h1 * smem_q[j + 1]
                       + h2 * smem_q[j + 2] + h3 * smem_q[j + 3];
            }

            output[((b * seq_len + t) * num_v_heads + vh) * v_dim + tid] =
                __float2bfloat16(q_dot * inv_sqrt_d);
        }
        __syncthreads();
    }

    // Cooperatively write H back from shared → global memory
    for (unsigned int i = tid; i < k_dim * v_dim; i += blockDim.x) {
        H_global[i] = H_smem[i];
    }
}
//   Both H_inter buffers (~2 MB each) stay in GB10's 64 MB L2 cache.
//
// Grid: (num_v_heads, batch, 1)
// Block: (BLOCK_SIZE, 1, 1)
extern "C" __global__ void gated_delta_rule_chunk3(
    // State (in/out): [batch, num_v_heads, k_dim, v_dim] FP32
    float* __restrict__ h_state,
    // Inputs for 3 tokens (BF16), accessed via stride params:
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    // Scalars per token per head (FP32):
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    // Output for 3 tokens:
    __nv_bfloat16* __restrict__ output,        // [batch, 3, num_v_heads, v_dim]
    // Two intermediate H states for rollback:
    float* __restrict__ h_state_inter0,        // H_1: [batch, num_v_heads, k_dim, v_dim]
    float* __restrict__ h_state_inter1,        // H_2: [batch, num_v_heads, k_dim, v_dim]
    // Dimensions:
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    // Strides (elements, not bytes) between consecutive tokens:
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    const unsigned int hv_size = k_dim * v_dim;
    float* H = h_state + ((b * num_v_heads + vh) * hv_size);
    float* Hi0 = h_state_inter0 + ((b * num_v_heads + vh) * hv_size);
    float* Hi1 = h_state_inter1 + ((b * num_v_heads + vh) * hv_size);

    // Token 0 inputs
    const __nv_bfloat16* q0 = query + (b * 3) * qk_stride + kh * k_dim;
    const __nv_bfloat16* k0 = key   + (b * 3) * qk_stride + kh * k_dim;
    const __nv_bfloat16* v0 = value + (b * 3) * v_stride  + vh * v_dim;
    const float g0 = gate[(b * 3) * gb_stride + vh];
    const float bt0 = beta[(b * 3) * gb_stride + vh];

    // Token 1 inputs
    const __nv_bfloat16* q1 = query + (b * 3 + 1) * qk_stride + kh * k_dim;
    const __nv_bfloat16* k1 = key   + (b * 3 + 1) * qk_stride + kh * k_dim;
    const __nv_bfloat16* v1 = value + (b * 3 + 1) * v_stride  + vh * v_dim;
    const float g1 = gate[(b * 3 + 1) * gb_stride + vh];
    const float bt1 = beta[(b * 3 + 1) * gb_stride + vh];

    // Token 2 inputs
    const __nv_bfloat16* q2 = query + (b * 3 + 2) * qk_stride + kh * k_dim;
    const __nv_bfloat16* k2 = key   + (b * 3 + 2) * qk_stride + kh * k_dim;
    const __nv_bfloat16* v2 = value + (b * 3 + 2) * v_stride  + vh * v_dim;
    const float g2 = gate[(b * 3 + 2) * gb_stride + vh];
    const float bt2 = beta[(b * 3 + 2) * gb_stride + vh];

    // Load all 3 token keys and queries into shared memory
    __shared__ float smem_k0[128];
    __shared__ float smem_q0[128];
    __shared__ float smem_k1[128];
    __shared__ float smem_q1[128];
    __shared__ float smem_k2[128];
    __shared__ float smem_q2[128];

    if (tid < k_dim) {
        smem_k0[tid] = (float)k0[tid]; smem_q0[tid] = (float)q0[tid];
        smem_k1[tid] = (float)k1[tid]; smem_q1[tid] = (float)q1[tid];
        smem_k2[tid] = (float)k2[tid]; smem_q2[tid] = (float)q2[tid];
    }
    __syncthreads();

    if (tid < v_dim) {
        float vi0 = (float)v0[tid];
        float vi1 = (float)v1[tid];
        float vi2 = (float)v2[tid];

        // ── Pass 1: Compute hk_dot_0 = H_0^T @ k_0 ──
        float hk0 = 0.0f;
        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H[(j + 0) * v_dim + tid];
            float h1 = H[(j + 1) * v_dim + tid];
            float h2 = H[(j + 2) * v_dim + tid];
            float h3 = H[(j + 3) * v_dim + tid];
            hk0 += h0 * smem_k0[j] + h1 * smem_k0[j + 1]
                 + h2 * smem_k0[j + 2] + h3 * smem_k0[j + 3];
        }
        float v_new_0 = (vi0 - g0 * hk0) * bt0;

        // ── Pass 2: H_0 → H_1, write Hi0, compute out_0, compute hk_dot_1 ──
        float q0_dot = 0.0f;
        float hk1 = 0.0f;
        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H[(j + 0) * v_dim + tid];
            float h1 = H[(j + 1) * v_dim + tid];
            float h2 = H[(j + 2) * v_dim + tid];
            float h3 = H[(j + 3) * v_dim + tid];
            h0 = g0 * h0 + smem_k0[j]     * v_new_0;
            h1 = g0 * h1 + smem_k0[j + 1] * v_new_0;
            h2 = g0 * h2 + smem_k0[j + 2] * v_new_0;
            h3 = g0 * h3 + smem_k0[j + 3] * v_new_0;
            Hi0[(j + 0) * v_dim + tid] = h0;
            Hi0[(j + 1) * v_dim + tid] = h1;
            Hi0[(j + 2) * v_dim + tid] = h2;
            Hi0[(j + 3) * v_dim + tid] = h3;
            q0_dot += h0 * smem_q0[j] + h1 * smem_q0[j + 1]
                    + h2 * smem_q0[j + 2] + h3 * smem_q0[j + 3];
            hk1 += h0 * smem_k1[j] + h1 * smem_k1[j + 1]
                 + h2 * smem_k1[j + 2] + h3 * smem_k1[j + 3];
        }
        float v_new_1 = (vi1 - g1 * hk1) * bt1;

        // ── Pass 3: H_1 → H_2, write Hi1, compute out_1, compute hk_dot_2 ──
        float q1_dot = 0.0f;
        float hk2 = 0.0f;
        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = Hi0[(j + 0) * v_dim + tid];
            float h1 = Hi0[(j + 1) * v_dim + tid];
            float h2 = Hi0[(j + 2) * v_dim + tid];
            float h3 = Hi0[(j + 3) * v_dim + tid];
            h0 = g1 * h0 + smem_k1[j]     * v_new_1;
            h1 = g1 * h1 + smem_k1[j + 1] * v_new_1;
            h2 = g1 * h2 + smem_k1[j + 2] * v_new_1;
            h3 = g1 * h3 + smem_k1[j + 3] * v_new_1;
            Hi1[(j + 0) * v_dim + tid] = h0;
            Hi1[(j + 1) * v_dim + tid] = h1;
            Hi1[(j + 2) * v_dim + tid] = h2;
            Hi1[(j + 3) * v_dim + tid] = h3;
            q1_dot += h0 * smem_q1[j] + h1 * smem_q1[j + 1]
                    + h2 * smem_q1[j + 2] + h3 * smem_q1[j + 3];
            hk2 += h0 * smem_k2[j] + h1 * smem_k2[j + 1]
                 + h2 * smem_k2[j + 2] + h3 * smem_k2[j + 3];
        }
        float v_new_2 = (vi2 - g2 * hk2) * bt2;

        // ── Pass 4: H_2 → H_3 (final), compute out_2 ──
        float q2_dot = 0.0f;
        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = Hi1[(j + 0) * v_dim + tid];
            float h1 = Hi1[(j + 1) * v_dim + tid];
            float h2 = Hi1[(j + 2) * v_dim + tid];
            float h3 = Hi1[(j + 3) * v_dim + tid];
            h0 = g2 * h0 + smem_k2[j]     * v_new_2;
            h1 = g2 * h1 + smem_k2[j + 1] * v_new_2;
            h2 = g2 * h2 + smem_k2[j + 2] * v_new_2;
            h3 = g2 * h3 + smem_k2[j + 3] * v_new_2;
            H[(j + 0) * v_dim + tid] = h0;
            H[(j + 1) * v_dim + tid] = h1;
            H[(j + 2) * v_dim + tid] = h2;
            H[(j + 3) * v_dim + tid] = h3;
            q2_dot += h0 * smem_q2[j] + h1 * smem_q2[j + 1]
                    + h2 * smem_q2[j + 2] + h3 * smem_q2[j + 3];
        }

        // Scale outputs by 1/sqrt(k_dim)
        float inv_sqrt_d = rsqrtf((float)k_dim);
        unsigned int out_base0 = (b * 3 * num_v_heads + vh) * v_dim;
        unsigned int out_base1 = ((b * 3 + 1) * num_v_heads + vh) * v_dim;
        unsigned int out_base2 = ((b * 3 + 2) * num_v_heads + vh) * v_dim;
        output[out_base0 + tid] = __float2bfloat16(q0_dot * inv_sqrt_d);
        output[out_base1 + tid] = __float2bfloat16(q1_dot * inv_sqrt_d);
        output[out_base2 + tid] = __float2bfloat16(q2_dot * inv_sqrt_d);
    }
}
