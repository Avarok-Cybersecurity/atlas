// SPDX-License-Identifier: AGPL-3.0-only

// Atlas WY32 Persistent GDN Prefill Kernel
//
// Extends the proven WY4 pattern to 32 tokens per iteration.
// H state stays in shared memory (64KB) for the entire sequence.
// K and Q stored as BF16 in shared memory to fit within SM121 limits.
//
// Algorithm (identical to WY4/WY8, scaled to C=32):
//   Pass 1: Read H → compute C hk_prev[t] = H^T @ k[t]
//   WY correction: compute corrected hk values using k-dot products
//   Pass 2: Read H → apply C state updates + C outputs
//
// Grid: (num_v_heads, batch, 1)  Block: (128, 1, 1)
// Shared memory: H[128×128]FP32 + k[C×128]BF16 + q[C×128]BF16 + kd[C×C]FP32 + misc
//              = 64KB + 8KB + 8KB + 4KB + 0.5KB ≈ 84.5KB

#include <cuda_bf16.h>

#define K_DIM 128
#define V_DIM 128
#define C     32    // tokens per WY iteration

__device__ __forceinline__ float wy_warp_reduce(float val) {
    for (int offset = 16; offset >= 1; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ __forceinline__ float wy_block_reduce(float val, float* smem_warp, unsigned int tid) {
    val = wy_warp_reduce(val);
    if (tid % 32 == 0) smem_warp[tid / 32] = val;
    __syncthreads();
    float result = 0.0f;
    if (tid == 0)
        result = smem_warp[0] + smem_warp[1] + smem_warp[2] + smem_warp[3];
    __syncthreads();
    return result;
}

extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_prefill_wy64(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
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
    const float inv_sqrt_d = rsqrtf((float)k_dim);

    // Shared memory layout (BF16 for K/Q to fit in 84.5KB):
    //   H_smem[K_DIM * V_DIM] FP32      = 64 KB
    //   smem_k[C * K_DIM] BF16           = 8 KB
    //   smem_q[C * K_DIM] BF16           = 8 KB
    //   smem_warp[4] FP32                = 16 B
    //   smem_kd[C * C] FP32              = 4 KB
    //   smem_g[C] FP32                   = 128 B
    //   smem_bt[C] FP32                  = 128 B
    extern __shared__ char smem_raw[];

    float* H_smem = (float*)smem_raw;
    __nv_bfloat16* smem_k = (__nv_bfloat16*)(smem_raw + K_DIM * V_DIM * 4);
    __nv_bfloat16* smem_q = smem_k + C * K_DIM;
    float* smem_warp = (float*)(smem_q + C * K_DIM);
    float* smem_kd = smem_warp + 4;
    float* smem_g = smem_kd + C * C;
    float* smem_bt = smem_g + C;

    // Load H from global → shared
    float* H_global = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);
    #pragma unroll 4
    for (unsigned int i = tid; i < K_DIM * V_DIM; i += V_DIM) {
        H_smem[i] = H_global[i];
    }
    __syncthreads();

    unsigned int wy_end = (seq_len / C) * C;

    // ═══════════════════════════════════════════════════════════════
    // Main WY32 loop
    // ═══════════════════════════════════════════════════════════════
    for (unsigned int chunk_start = 0; chunk_start < wy_end; chunk_start += C) {

        // Load C tokens' k, q as BF16, g/beta as FP32
        for (unsigned int idx = tid; idx < C * K_DIM; idx += V_DIM) {
            unsigned int tok = idx / K_DIM;
            unsigned int dim = idx % K_DIM;
            unsigned long long off = (unsigned long long)(chunk_start + tok) * qk_stride + kh * k_dim + dim;
            smem_k[tok * K_DIM + dim] = key[off];    // BF16 direct
            smem_q[tok * K_DIM + dim] = query[off];   // BF16 direct
        }
        if (tid < C) {
            smem_g[tid] = gate[(unsigned long long)(chunk_start + tid) * gb_stride + vh];
            smem_bt[tid] = beta[(unsigned long long)(chunk_start + tid) * gb_stride + vh];
        }
        __syncthreads();

        // Compute k-dot products: kd[i][j] = k_i^T @ k_j for i > j
        for (unsigned int idx = tid; idx < C * C; idx += V_DIM) {
            smem_kd[idx] = 0.0f;
        }
        __syncthreads();

        for (int i = 1; i < C; i++) {
            for (int j = 0; j < i; j++) {
                float partial = (tid < K_DIM) ?
                    (float)smem_k[i * K_DIM + tid] * (float)smem_k[j * K_DIM + tid] : 0.0f;
                float dot = wy_block_reduce(partial, smem_warp, tid);
                if (tid == 0) smem_kd[i * C + j] = dot;
                __syncthreads();
            }
        }

        // Pass 1: Read H once, compute all C hk_prev values
        float hk_prev[C];
        for (int t = 0; t < C; t++) hk_prev[t] = 0.0f;

        for (int j = 0; j < K_DIM; j++) {
            float h_j = H_smem[j * V_DIM + tid];
            for (int t = 0; t < C; t++) {
                hk_prev[t] += h_j * (float)smem_k[t * K_DIM + j];
            }
        }

        // WY correction + v_new computation
        float v_new_arr[C];
        for (int t = 0; t < C; t++) {
            float v_t = (float)value[(unsigned long long)(chunk_start + t) * v_stride + vh * v_dim + tid];

            // Cumulative gate product for H^T @ k[t] term: prod(g[0..t-1])
            float g_prod = 1.0f;
            for (int s = 0; s < t; s++) g_prod *= smem_g[s];

            float hk_corr = g_prod * hk_prev[t];

            // Corrections from prior tokens
            for (int s = 0; s < t; s++) {
                float g_prod_s = 1.0f;
                for (int m = s + 1; m < t; m++) g_prod_s *= smem_g[m];
                hk_corr += g_prod_s * smem_kd[t * C + s] * v_new_arr[s];
            }

            v_new_arr[t] = (v_t - smem_g[t] * hk_corr) * smem_bt[t];
        }

        // Pass 2: Apply all C state updates + compute outputs
        float o_out[C];
        for (int t = 0; t < C; t++) o_out[t] = 0.0f;

        for (int j = 0; j < K_DIM; j++) {
            float h_j = H_smem[j * V_DIM + tid];
            for (int t = 0; t < C; t++) {
                h_j = smem_g[t] * h_j + (float)smem_k[t * K_DIM + j] * v_new_arr[t];
                o_out[t] += h_j * (float)smem_q[t * K_DIM + j];
            }
            H_smem[j * V_DIM + tid] = h_j;
        }

        // Write outputs
        for (int t = 0; t < C; t++) {
            unsigned long long out_off = (unsigned long long)(chunk_start + t) * num_v_heads * v_dim + vh * v_dim + tid;
            output[out_off] = __float2bfloat16(o_out[t] * inv_sqrt_d);
        }
        __syncthreads();
    }

    // Handle remainder tokens (seq_len % C) with sequential processing
    for (unsigned int t = wy_end; t < seq_len; t++) {
        if (tid < K_DIM) {
            unsigned long long qk_off = (unsigned long long)t * qk_stride + kh * k_dim;
            smem_k[tid] = key[qk_off + tid];
            smem_q[tid] = query[qk_off + tid];
        }
        __syncthreads();

        float v_i = (float)value[(unsigned long long)t * v_stride + vh * v_dim + tid];
        float g_t = gate[(unsigned long long)t * gb_stride + vh];
        float bt_t = beta[(unsigned long long)t * gb_stride + vh];

        float hk = 0.0f;
        for (int j = 0; j < K_DIM; j += 4) {
            hk += H_smem[(j+0)*V_DIM+tid]*(float)smem_k[j]
                + H_smem[(j+1)*V_DIM+tid]*(float)smem_k[j+1]
                + H_smem[(j+2)*V_DIM+tid]*(float)smem_k[j+2]
                + H_smem[(j+3)*V_DIM+tid]*(float)smem_k[j+3];
        }
        float vn = (v_i - g_t * hk) * bt_t;

        float q_dot = 0.0f;
        for (int j = 0; j < K_DIM; j += 4) {
            float h0 = g_t*H_smem[(j+0)*V_DIM+tid] + (float)smem_k[j]*vn;
            float h1 = g_t*H_smem[(j+1)*V_DIM+tid] + (float)smem_k[j+1]*vn;
            float h2 = g_t*H_smem[(j+2)*V_DIM+tid] + (float)smem_k[j+2]*vn;
            float h3 = g_t*H_smem[(j+3)*V_DIM+tid] + (float)smem_k[j+3]*vn;
            H_smem[(j+0)*V_DIM+tid]=h0; H_smem[(j+1)*V_DIM+tid]=h1;
            H_smem[(j+2)*V_DIM+tid]=h2; H_smem[(j+3)*V_DIM+tid]=h3;
            q_dot += h0*(float)smem_q[j] + h1*(float)smem_q[j+1]
                   + h2*(float)smem_q[j+2] + h3*(float)smem_q[j+3];
        }
        unsigned long long out_off = (unsigned long long)t * num_v_heads * v_dim + vh * v_dim + tid;
        output[out_off] = __float2bfloat16(q_dot * inv_sqrt_d);
        __syncthreads();
    }

    // Write H from shared → global
    #pragma unroll 4
    for (unsigned int i = tid; i < K_DIM * V_DIM; i += V_DIM) {
        H_global[i] = H_smem[i];
    }
}

// ───────────────────────────────────────────────────────────────────────
// Q12 Phase 2b: same-chunk-len batched entry point.
//
// Differences from gated_delta_rule_prefill_wy64:
//   - h_state replaced with h_state_ptrs[]: each batch element owns its
//     own h_state allocation (per-stream SsmLayerState in Atlas).
//   - QKV / gate / beta / output read+written with per-batch offset
//     (b * seq_len * stride): inputs are stacked per stream contiguously.
//
// Constraint: all batched streams share the same seq_len. The scheduler-
// side gate (`can_batch_prefill_only` in phase_continue_prefills.rs) is
// responsible for enforcing this — streams with mismatched chunk_len
// fall through to per-stream sequential kernel launches.
//
// Validation: unvalidated — kernel-correctness check pending GPU run.
// ───────────────────────────────────────────────────────────────────────
extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_prefill_wy64_batched(
    float* const* __restrict__ h_state_ptrs,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
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
    const float inv_sqrt_d = rsqrtf((float)k_dim);

    // Per-batch offsets into stacked QKV / gate / beta / output buffers.
    const unsigned long long qk_batch_off = (unsigned long long)b * seq_len * qk_stride;
    const unsigned long long v_batch_off  = (unsigned long long)b * seq_len * v_stride;
    const unsigned long long gb_batch_off = (unsigned long long)b * seq_len * gb_stride;
    const unsigned long long out_batch_off = (unsigned long long)b * seq_len * num_v_heads * v_dim;

    extern __shared__ char smem_raw[];

    float* H_smem = (float*)smem_raw;
    __nv_bfloat16* smem_k = (__nv_bfloat16*)(smem_raw + K_DIM * V_DIM * 4);
    __nv_bfloat16* smem_q = smem_k + C * K_DIM;
    float* smem_warp = (float*)(smem_q + C * K_DIM);
    float* smem_kd = smem_warp + 4;
    float* smem_g = smem_kd + C * C;
    float* smem_bt = smem_g + C;

    // Per-stream H pointer dereferenced from the pointer array.
    float* H_global = h_state_ptrs[b] + ((unsigned long long)vh * K_DIM * V_DIM);
    #pragma unroll 4
    for (unsigned int i = tid; i < K_DIM * V_DIM; i += V_DIM) {
        H_smem[i] = H_global[i];
    }
    __syncthreads();

    unsigned int wy_end = (seq_len / C) * C;

    for (unsigned int chunk_start = 0; chunk_start < wy_end; chunk_start += C) {
        for (unsigned int idx = tid; idx < C * K_DIM; idx += V_DIM) {
            unsigned int tok = idx / K_DIM;
            unsigned int dim = idx % K_DIM;
            unsigned long long off = qk_batch_off + (unsigned long long)(chunk_start + tok) * qk_stride + kh * k_dim + dim;
            smem_k[tok * K_DIM + dim] = key[off];
            smem_q[tok * K_DIM + dim] = query[off];
        }
        if (tid < C) {
            smem_g[tid] = gate[gb_batch_off + (unsigned long long)(chunk_start + tid) * gb_stride + vh];
            smem_bt[tid] = beta[gb_batch_off + (unsigned long long)(chunk_start + tid) * gb_stride + vh];
        }
        __syncthreads();

        for (unsigned int idx = tid; idx < C * C; idx += V_DIM) {
            smem_kd[idx] = 0.0f;
        }
        __syncthreads();

        for (int i = 1; i < C; i++) {
            for (int j = 0; j < i; j++) {
                float partial = (tid < K_DIM) ?
                    (float)smem_k[i * K_DIM + tid] * (float)smem_k[j * K_DIM + tid] : 0.0f;
                float dot = wy_block_reduce(partial, smem_warp, tid);
                if (tid == 0) smem_kd[i * C + j] = dot;
                __syncthreads();
            }
        }

        float hk_prev[C];
        for (int t = 0; t < C; t++) hk_prev[t] = 0.0f;

        for (int j = 0; j < K_DIM; j++) {
            float h_j = H_smem[j * V_DIM + tid];
            for (int t = 0; t < C; t++) {
                hk_prev[t] += h_j * (float)smem_k[t * K_DIM + j];
            }
        }

        float v_new_arr[C];
        for (int t = 0; t < C; t++) {
            float v_t = (float)value[v_batch_off + (unsigned long long)(chunk_start + t) * v_stride + vh * v_dim + tid];

            float g_prod = 1.0f;
            for (int s = 0; s < t; s++) g_prod *= smem_g[s];

            float hk_corr = g_prod * hk_prev[t];

            for (int s = 0; s < t; s++) {
                float g_prod_s = 1.0f;
                for (int m = s + 1; m < t; m++) g_prod_s *= smem_g[m];
                hk_corr += g_prod_s * smem_kd[t * C + s] * v_new_arr[s];
            }

            v_new_arr[t] = (v_t - smem_g[t] * hk_corr) * smem_bt[t];
        }

        float o_out[C];
        for (int t = 0; t < C; t++) o_out[t] = 0.0f;

        for (int j = 0; j < K_DIM; j++) {
            float h_j = H_smem[j * V_DIM + tid];
            for (int t = 0; t < C; t++) {
                h_j = smem_g[t] * h_j + (float)smem_k[t * K_DIM + j] * v_new_arr[t];
                o_out[t] += h_j * (float)smem_q[t * K_DIM + j];
            }
            H_smem[j * V_DIM + tid] = h_j;
        }

        for (int t = 0; t < C; t++) {
            unsigned long long out_off = out_batch_off + (unsigned long long)(chunk_start + t) * num_v_heads * v_dim + vh * v_dim + tid;
            output[out_off] = __float2bfloat16(o_out[t] * inv_sqrt_d);
        }
        __syncthreads();
    }

    for (unsigned int t = wy_end; t < seq_len; t++) {
        if (tid < K_DIM) {
            unsigned long long qk_off = qk_batch_off + (unsigned long long)t * qk_stride + kh * k_dim;
            smem_k[tid] = key[qk_off + tid];
            smem_q[tid] = query[qk_off + tid];
        }
        __syncthreads();

        float v_i = (float)value[v_batch_off + (unsigned long long)t * v_stride + vh * v_dim + tid];
        float g_t = gate[gb_batch_off + (unsigned long long)t * gb_stride + vh];
        float bt_t = beta[gb_batch_off + (unsigned long long)t * gb_stride + vh];

        float hk = 0.0f;
        for (int j = 0; j < K_DIM; j += 4) {
            hk += H_smem[(j+0)*V_DIM+tid]*(float)smem_k[j]
                + H_smem[(j+1)*V_DIM+tid]*(float)smem_k[j+1]
                + H_smem[(j+2)*V_DIM+tid]*(float)smem_k[j+2]
                + H_smem[(j+3)*V_DIM+tid]*(float)smem_k[j+3];
        }
        float vn = (v_i - g_t * hk) * bt_t;

        float q_dot = 0.0f;
        for (int j = 0; j < K_DIM; j += 4) {
            float h0 = g_t*H_smem[(j+0)*V_DIM+tid] + (float)smem_k[j]*vn;
            float h1 = g_t*H_smem[(j+1)*V_DIM+tid] + (float)smem_k[j+1]*vn;
            float h2 = g_t*H_smem[(j+2)*V_DIM+tid] + (float)smem_k[j+2]*vn;
            float h3 = g_t*H_smem[(j+3)*V_DIM+tid] + (float)smem_k[j+3]*vn;
            H_smem[(j+0)*V_DIM+tid]=h0; H_smem[(j+1)*V_DIM+tid]=h1;
            H_smem[(j+2)*V_DIM+tid]=h2; H_smem[(j+3)*V_DIM+tid]=h3;
            q_dot += h0*(float)smem_q[j] + h1*(float)smem_q[j+1]
                   + h2*(float)smem_q[j+2] + h3*(float)smem_q[j+3];
        }
        unsigned long long out_off = out_batch_off + (unsigned long long)t * num_v_heads * v_dim + vh * v_dim + tid;
        output[out_off] = __float2bfloat16(q_dot * inv_sqrt_d);
        __syncthreads();
    }

    #pragma unroll 4
    for (unsigned int i = tid; i < K_DIM * V_DIM; i += V_DIM) {
        H_global[i] = H_smem[i];
    }
}
