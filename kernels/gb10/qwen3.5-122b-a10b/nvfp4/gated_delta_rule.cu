// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Register-Tiled Gated Delta Rule Prefill — 35B model shadow.
//
// Each thread holds its H column (128 floats) entirely in registers.
// Eliminates all shared memory latency for H access (0-cycle vs ~20-cycle).
//
// Optimizations over parent:
// - __launch_bounds__(128, 1): forces minBlocksPerSM=1, allowing compiler to
//   allocate up to 512 registers/thread (vs 42 with default occupancy target
//   of 12 blocks/SM on SM121). Without this, H_reg[128] spills to L1 cache
//   (28-cycle latency) causing ~8× slowdown vs ideal register access.
// - 4-way independent accumulators for hk_dot and q_dot reductions
//   (breaks serial FMA dependency chain: 512 cycles → ~140 cycles per pass)
// - Double-buffered smem for k/q (eliminates 1 syncthreads per token,
//   overlaps next token's L2 loads with current token's compute)
//
// Grid: (num_v_heads, batch, 1)   Block: (128, 1, 1)

#include <cuda_bf16.h>

#define K_DIM 128

extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_prefill(
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

    // Double-buffered k[128] + q[128] (512 floats = 2 KB)
    extern __shared__ float smem[];
    float* smem_k0 = smem;
    float* smem_q0 = smem + K_DIM;
    float* smem_k1 = smem + 2 * K_DIM;
    float* smem_q1 = smem + 3 * K_DIM;

    float* H_global = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * v_dim);

    // Load H column into registers — each thread owns one column of H[128×128]
    float H_reg[K_DIM];
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_reg[j] = H_global[j * v_dim + tid];
    }

    float inv_sqrt_d = rsqrtf((float)k_dim);

    if (seq_len == 0) goto store_h;

    // Load first token's k/q into buffer 0
    {
        unsigned long long qk_off = (unsigned long long)0 * qk_stride + kh * k_dim;
        smem_k0[tid] = (float)key[qk_off + tid];
        smem_q0[tid] = (float)query[qk_off + tid];
    }
    __syncthreads();

    // Process tokens with double-buffered k/q loads
    for (unsigned int t = 0; t < seq_len; t++) {
        // Select current and next buffers
        float* cur_k = (t & 1) ? smem_k1 : smem_k0;
        float* cur_q = (t & 1) ? smem_q1 : smem_q0;
        float* nxt_k = (t & 1) ? smem_k0 : smem_k1;
        float* nxt_q = (t & 1) ? smem_q0 : smem_q1;

        // Issue loads for NEXT token into other buffer (overlaps with compute)
        if (t + 1 < seq_len) {
            unsigned long long qk_off_nxt = (unsigned long long)(t + 1) * qk_stride + kh * k_dim;
            nxt_k[tid] = (float)key[qk_off_nxt + tid];
            nxt_q[tid] = (float)query[qk_off_nxt + tid];
        }

        float v_i = (float)value[(unsigned long long)t * v_stride + vh * v_dim + tid];
        float g_t = fminf(fmaxf(gate[(unsigned long long)t * gb_stride + vh], 1e-6f), 1.0f - 1e-6f);
        float bt_t = beta[(unsigned long long)t * gb_stride + vh];

        // Pass 1: hk_dot = H_reg^T · k
        // 4 independent accumulators break serial FMA dependency chain
        float hk0 = 0.0f, hk1 = 0.0f, hk2 = 0.0f, hk3 = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            hk0 += H_reg[j]     * cur_k[j];
            hk1 += H_reg[j + 1] * cur_k[j + 1];
            hk2 += H_reg[j + 2] * cur_k[j + 2];
            hk3 += H_reg[j + 3] * cur_k[j + 3];
        }
        float hk_dot = (hk0 + hk1) + (hk2 + hk3);

        float v_new = (v_i - g_t * hk_dot) * bt_t;

        // Pass 2: update H_reg, compute q_dot = H_new^T · q
        // 4 independent accumulators for q_dot
        float qd0 = 0.0f, qd1 = 0.0f, qd2 = 0.0f, qd3 = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            float h0 = g_t * H_reg[j]     + cur_k[j]     * v_new;
            float h1 = g_t * H_reg[j + 1] + cur_k[j + 1] * v_new;
            float h2 = g_t * H_reg[j + 2] + cur_k[j + 2] * v_new;
            float h3 = g_t * H_reg[j + 3] + cur_k[j + 3] * v_new;
            H_reg[j]     = h0;
            H_reg[j + 1] = h1;
            H_reg[j + 2] = h2;
            H_reg[j + 3] = h3;
            qd0 += h0 * cur_q[j];
            qd1 += h1 * cur_q[j + 1];
            qd2 += h2 * cur_q[j + 2];
            qd3 += h3 * cur_q[j + 3];
        }
        float q_dot = (qd0 + qd1) + (qd2 + qd3);

        output[((unsigned long long)(b * seq_len + t) * num_v_heads + vh) * v_dim + tid] =
            __float2bfloat16(q_dot * inv_sqrt_d);

        __syncthreads();  // Ensures next token's k/q are fully loaded
    }

store_h:
    // ── SSM state normalization (decode-only, Stuffed Mamba mitigation) ──
    if (seq_len <= 1) {
        float local_sq = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j++) {
            local_sq += H_reg[j] * H_reg[j];
        }
        unsigned int mask = __activemask();
        float ws = local_sq;
        ws += __shfl_down_sync(mask, ws, 16);
        ws += __shfl_down_sync(mask, ws, 8);
        ws += __shfl_down_sync(mask, ws, 4);
        ws += __shfl_down_sync(mask, ws, 2);
        ws += __shfl_down_sync(mask, ws, 1);
        __shared__ float ns[4];
        if (tid % 32 == 0) ns[tid / 32] = ws;
        __syncthreads();
        if (tid < 4) {
            float s = ns[tid];
            s += __shfl_down_sync(0xf, s, 2);
            s += __shfl_down_sync(0xf, s, 1);
            ns[0] = s;
        }
        __syncthreads();
        const float MAX_NORM = 50.0f;
        float norm_sq = ns[0];
        if (norm_sq > MAX_NORM * MAX_NORM) {
            float scale = MAX_NORM * rsqrtf(norm_sq);
            #pragma unroll
            for (int j = 0; j < K_DIM; j++) {
                H_reg[j] *= scale;
            }
        }
    }

    // Write H from registers → global
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_global[j * v_dim + tid] = H_reg[j];
    }
}

// ═══════════════════════════════════════════════════════════════════
// Split-v_dim prefill: 2 CTAs per v-head, 64 threads each.
//
// Identical math to gated_delta_rule_prefill, but splits v_dim across
// 2 independent CTAs per v-head. Doubles SM utilization (64 CTAs on
// 48 SMs vs 32 CTAs on 32 SMs) and allows cross-block latency hiding
// on SMs that host 2 independent blocks.
//
// Thread tid_local (0..63) handles v_dim column (split*64 + tid_local).
// Each thread still loads H_reg[K_DIM=128] — no register pressure change.
// Each thread loads 2 smem elements per k/q buffer (stride blockDim.x=64).
//
// Grid: (num_v_heads * 2, batch, 1)   Block: (64, 1, 1)
// ═══════════════════════════════════════════════════════════════════

extern "C" __global__ void __launch_bounds__(64, 1)
gated_delta_rule_prefill_split(
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
    // blockIdx.x = vh * 2 + split  (0..num_v_heads*2 - 1)
    const unsigned int vh    = blockIdx.x / 2;
    const unsigned int split = blockIdx.x % 2;
    const unsigned int b     = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid_local  = threadIdx.x;               // 0..63
    const unsigned int half       = blockDim.x;                 // 64
    const unsigned int tid        = split * half + tid_local;   // 0..127
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    // Double-buffered k[K_DIM] + q[K_DIM] in smem (same footprint as original).
    extern __shared__ float smem[];
    float* smem_k0 = smem;
    float* smem_q0 = smem + K_DIM;
    float* smem_k1 = smem + 2 * K_DIM;
    float* smem_q1 = smem + 3 * K_DIM;

    float* H_global = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * v_dim);

    // Load H column for tid into registers.
    float H_reg[K_DIM];
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_reg[j] = H_global[j * v_dim + tid];
    }

    float inv_sqrt_d = rsqrtf((float)k_dim);

    if (seq_len == 0) goto store_h_split;

    // Load first token's k/q into buffer 0.
    // Each thread loads 2 elements (indices tid_local and tid_local+half=tid_local+64).
    {
        unsigned long long qk_off = (unsigned long long)0 * qk_stride + kh * k_dim;
        smem_k0[tid_local]        = (float)key[qk_off + tid_local];
        smem_k0[tid_local + half] = (float)key[qk_off + tid_local + half];
        smem_q0[tid_local]        = (float)query[qk_off + tid_local];
        smem_q0[tid_local + half] = (float)query[qk_off + tid_local + half];
    }
    __syncthreads();

    for (unsigned int t = 0; t < seq_len; t++) {
        float* cur_k = (t & 1) ? smem_k1 : smem_k0;
        float* cur_q = (t & 1) ? smem_q1 : smem_q0;
        float* nxt_k = (t & 1) ? smem_k0 : smem_k1;
        float* nxt_q = (t & 1) ? smem_q0 : smem_q1;

        if (t + 1 < seq_len) {
            unsigned long long qk_off_nxt = (unsigned long long)(t + 1) * qk_stride + kh * k_dim;
            nxt_k[tid_local]        = (float)key[qk_off_nxt + tid_local];
            nxt_k[tid_local + half] = (float)key[qk_off_nxt + tid_local + half];
            nxt_q[tid_local]        = (float)query[qk_off_nxt + tid_local];
            nxt_q[tid_local + half] = (float)query[qk_off_nxt + tid_local + half];
        }

        float v_i  = (float)value[(unsigned long long)t * v_stride + vh * v_dim + tid];
        float g_t  = fminf(fmaxf(gate[(unsigned long long)t * gb_stride + vh], 1e-6f), 1.0f - 1e-6f);
        float bt_t = beta[(unsigned long long)t * gb_stride + vh];

        float hk0 = 0.0f, hk1 = 0.0f, hk2 = 0.0f, hk3 = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            hk0 += H_reg[j]     * cur_k[j];
            hk1 += H_reg[j + 1] * cur_k[j + 1];
            hk2 += H_reg[j + 2] * cur_k[j + 2];
            hk3 += H_reg[j + 3] * cur_k[j + 3];
        }
        float hk_dot = (hk0 + hk1) + (hk2 + hk3);

        float v_new = (v_i - g_t * hk_dot) * bt_t;

        float qd0 = 0.0f, qd1 = 0.0f, qd2 = 0.0f, qd3 = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            float h0 = g_t * H_reg[j]     + cur_k[j]     * v_new;
            float h1 = g_t * H_reg[j + 1] + cur_k[j + 1] * v_new;
            float h2 = g_t * H_reg[j + 2] + cur_k[j + 2] * v_new;
            float h3 = g_t * H_reg[j + 3] + cur_k[j + 3] * v_new;
            H_reg[j]     = h0;
            H_reg[j + 1] = h1;
            H_reg[j + 2] = h2;
            H_reg[j + 3] = h3;
            qd0 += h0 * cur_q[j];
            qd1 += h1 * cur_q[j + 1];
            qd2 += h2 * cur_q[j + 2];
            qd3 += h3 * cur_q[j + 3];
        }
        float q_dot = (qd0 + qd1) + (qd2 + qd3);

        output[((unsigned long long)(b * seq_len + t) * num_v_heads + vh) * v_dim + tid] =
            __float2bfloat16(q_dot * inv_sqrt_d);

        __syncthreads();
    }

store_h_split:
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_global[j * v_dim + tid] = H_reg[j];
    }
}

// ═══════════════════════════════════════════════════════════════════
// 4-way split prefill: 4 CTAs per v-head, 32 threads each (128 total).
//
// 128 CTAs on 48 SMs: ~2.67 blocks/SM average → SMs run 2-3 independent
// blocks, enabling cross-block latency hiding even with 1 warp per block.
// Each thread loads 4 smem elements per k/q buffer (stride 32).
//
// Grid: (num_v_heads * 4, batch, 1)   Block: (32, 1, 1)
// ═══════════════════════════════════════════════════════════════════

extern "C" __global__ void __launch_bounds__(32, 1)
gated_delta_rule_prefill_split4(
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
    // blockIdx.x = vh * 4 + split  (0..num_v_heads*4 - 1)
    const unsigned int vh    = blockIdx.x / 4;
    const unsigned int split = blockIdx.x % 4;
    const unsigned int b     = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid_local  = threadIdx.x;               // 0..31
    const unsigned int quarter    = blockDim.x;                 // 32
    const unsigned int tid        = split * quarter + tid_local; // 0..127
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    // Double-buffered k[K_DIM] + q[K_DIM] in smem (same footprint as original).
    extern __shared__ float smem[];
    float* smem_k0 = smem;
    float* smem_q0 = smem + K_DIM;
    float* smem_k1 = smem + 2 * K_DIM;
    float* smem_q1 = smem + 3 * K_DIM;

    float* H_global = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * v_dim);

    float H_reg[K_DIM];
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_reg[j] = H_global[j * v_dim + tid];
    }

    float inv_sqrt_d = rsqrtf((float)k_dim);

    if (seq_len == 0) goto store_h_split4;

    // Load first token's k/q into buffer 0 — each thread loads 4 elements.
    {
        unsigned long long qk_off = (unsigned long long)0 * qk_stride + kh * k_dim;
        smem_k0[tid_local]            = (float)key[qk_off + tid_local];
        smem_k0[tid_local + quarter]  = (float)key[qk_off + tid_local + quarter];
        smem_k0[tid_local + 2*quarter]= (float)key[qk_off + tid_local + 2*quarter];
        smem_k0[tid_local + 3*quarter]= (float)key[qk_off + tid_local + 3*quarter];
        smem_q0[tid_local]            = (float)query[qk_off + tid_local];
        smem_q0[tid_local + quarter]  = (float)query[qk_off + tid_local + quarter];
        smem_q0[tid_local + 2*quarter]= (float)query[qk_off + tid_local + 2*quarter];
        smem_q0[tid_local + 3*quarter]= (float)query[qk_off + tid_local + 3*quarter];
    }
    __syncthreads();

    for (unsigned int t = 0; t < seq_len; t++) {
        float* cur_k = (t & 1) ? smem_k1 : smem_k0;
        float* cur_q = (t & 1) ? smem_q1 : smem_q0;
        float* nxt_k = (t & 1) ? smem_k0 : smem_k1;
        float* nxt_q = (t & 1) ? smem_q0 : smem_q1;

        if (t + 1 < seq_len) {
            unsigned long long qk_off_nxt = (unsigned long long)(t + 1) * qk_stride + kh * k_dim;
            nxt_k[tid_local]            = (float)key[qk_off_nxt + tid_local];
            nxt_k[tid_local + quarter]  = (float)key[qk_off_nxt + tid_local + quarter];
            nxt_k[tid_local + 2*quarter]= (float)key[qk_off_nxt + tid_local + 2*quarter];
            nxt_k[tid_local + 3*quarter]= (float)key[qk_off_nxt + tid_local + 3*quarter];
            nxt_q[tid_local]            = (float)query[qk_off_nxt + tid_local];
            nxt_q[tid_local + quarter]  = (float)query[qk_off_nxt + tid_local + quarter];
            nxt_q[tid_local + 2*quarter]= (float)query[qk_off_nxt + tid_local + 2*quarter];
            nxt_q[tid_local + 3*quarter]= (float)query[qk_off_nxt + tid_local + 3*quarter];
        }

        float v_i  = (float)value[(unsigned long long)t * v_stride + vh * v_dim + tid];
        float g_t  = fminf(fmaxf(gate[(unsigned long long)t * gb_stride + vh], 1e-6f), 1.0f - 1e-6f);
        float bt_t = beta[(unsigned long long)t * gb_stride + vh];

        float hk0 = 0.0f, hk1 = 0.0f, hk2 = 0.0f, hk3 = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            hk0 += H_reg[j]     * cur_k[j];
            hk1 += H_reg[j + 1] * cur_k[j + 1];
            hk2 += H_reg[j + 2] * cur_k[j + 2];
            hk3 += H_reg[j + 3] * cur_k[j + 3];
        }
        float hk_dot = (hk0 + hk1) + (hk2 + hk3);

        float v_new = (v_i - g_t * hk_dot) * bt_t;

        float qd0 = 0.0f, qd1 = 0.0f, qd2 = 0.0f, qd3 = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            float h0 = g_t * H_reg[j]     + cur_k[j]     * v_new;
            float h1 = g_t * H_reg[j + 1] + cur_k[j + 1] * v_new;
            float h2 = g_t * H_reg[j + 2] + cur_k[j + 2] * v_new;
            float h3 = g_t * H_reg[j + 3] + cur_k[j + 3] * v_new;
            H_reg[j]     = h0;
            H_reg[j + 1] = h1;
            H_reg[j + 2] = h2;
            H_reg[j + 3] = h3;
            qd0 += h0 * cur_q[j];
            qd1 += h1 * cur_q[j + 1];
            qd2 += h2 * cur_q[j + 2];
            qd3 += h3 * cur_q[j + 3];
        }
        float q_dot = (qd0 + qd1) + (qd2 + qd3);

        output[((unsigned long long)(b * seq_len + t) * num_v_heads + vh) * v_dim + tid] =
            __float2bfloat16(q_dot * inv_sqrt_d);

        __syncthreads();
    }

store_h_split4:
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_global[j * v_dim + tid] = H_reg[j];
    }
}

// ───────────────────────────────────────────────────────────────────────
// Q12 Phase 2b: same-chunk-len batched split4. h_state per-stream via
// h_state_ptrs[b]; QKV/gate/beta/output stacked with `b * seq_len * stride`
// offset; otherwise byte-identical to gated_delta_rule_prefill_split4.
// Single-stream variant above unchanged.
// ───────────────────────────────────────────────────────────────────────
extern "C" __global__ void __launch_bounds__(32, 1)
gated_delta_rule_prefill_split4_batched(
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
    const unsigned int vh    = blockIdx.x / 4;
    const unsigned int split = blockIdx.x % 4;
    const unsigned int b     = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid_local  = threadIdx.x;
    const unsigned int quarter    = blockDim.x;
    const unsigned int tid        = split * quarter + tid_local;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    const unsigned long long qk_batch_off = (unsigned long long)b * seq_len * qk_stride;
    const unsigned long long v_batch_off  = (unsigned long long)b * seq_len * v_stride;
    const unsigned long long gb_batch_off = (unsigned long long)b * seq_len * gb_stride;
    const unsigned long long out_batch_off = (unsigned long long)b * seq_len * num_v_heads * v_dim;

    extern __shared__ float smem[];
    float* smem_k0 = smem;
    float* smem_q0 = smem + K_DIM;
    float* smem_k1 = smem + 2 * K_DIM;
    float* smem_q1 = smem + 3 * K_DIM;

    float* H_global = h_state_ptrs[b] + ((unsigned long long)vh * K_DIM * v_dim);

    float H_reg[K_DIM];
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_reg[j] = H_global[j * v_dim + tid];
    }

    float inv_sqrt_d = rsqrtf((float)k_dim);

    if (seq_len == 0) goto store_h_split4_batched;

    {
        unsigned long long qk_off = qk_batch_off + kh * k_dim;
        smem_k0[tid_local]            = (float)key[qk_off + tid_local];
        smem_k0[tid_local + quarter]  = (float)key[qk_off + tid_local + quarter];
        smem_k0[tid_local + 2*quarter]= (float)key[qk_off + tid_local + 2*quarter];
        smem_k0[tid_local + 3*quarter]= (float)key[qk_off + tid_local + 3*quarter];
        smem_q0[tid_local]            = (float)query[qk_off + tid_local];
        smem_q0[tid_local + quarter]  = (float)query[qk_off + tid_local + quarter];
        smem_q0[tid_local + 2*quarter]= (float)query[qk_off + tid_local + 2*quarter];
        smem_q0[tid_local + 3*quarter]= (float)query[qk_off + tid_local + 3*quarter];
    }
    __syncthreads();

    for (unsigned int t = 0; t < seq_len; t++) {
        float* cur_k = (t & 1) ? smem_k1 : smem_k0;
        float* cur_q = (t & 1) ? smem_q1 : smem_q0;
        float* nxt_k = (t & 1) ? smem_k0 : smem_k1;
        float* nxt_q = (t & 1) ? smem_q0 : smem_q1;

        if (t + 1 < seq_len) {
            unsigned long long qk_off_nxt = qk_batch_off + (unsigned long long)(t + 1) * qk_stride + kh * k_dim;
            nxt_k[tid_local]            = (float)key[qk_off_nxt + tid_local];
            nxt_k[tid_local + quarter]  = (float)key[qk_off_nxt + tid_local + quarter];
            nxt_k[tid_local + 2*quarter]= (float)key[qk_off_nxt + tid_local + 2*quarter];
            nxt_k[tid_local + 3*quarter]= (float)key[qk_off_nxt + tid_local + 3*quarter];
            nxt_q[tid_local]            = (float)query[qk_off_nxt + tid_local];
            nxt_q[tid_local + quarter]  = (float)query[qk_off_nxt + tid_local + quarter];
            nxt_q[tid_local + 2*quarter]= (float)query[qk_off_nxt + tid_local + 2*quarter];
            nxt_q[tid_local + 3*quarter]= (float)query[qk_off_nxt + tid_local + 3*quarter];
        }

        float v_i  = (float)value[v_batch_off + (unsigned long long)t * v_stride + vh * v_dim + tid];
        float g_t  = fminf(fmaxf(gate[gb_batch_off + (unsigned long long)t * gb_stride + vh], 1e-6f), 1.0f - 1e-6f);
        float bt_t = beta[gb_batch_off + (unsigned long long)t * gb_stride + vh];

        float hk0 = 0.0f, hk1 = 0.0f, hk2 = 0.0f, hk3 = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            hk0 += H_reg[j]     * cur_k[j];
            hk1 += H_reg[j + 1] * cur_k[j + 1];
            hk2 += H_reg[j + 2] * cur_k[j + 2];
            hk3 += H_reg[j + 3] * cur_k[j + 3];
        }
        float hk_dot = (hk0 + hk1) + (hk2 + hk3);

        float v_new = (v_i - g_t * hk_dot) * bt_t;

        float qd0 = 0.0f, qd1 = 0.0f, qd2 = 0.0f, qd3 = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            float h0 = g_t * H_reg[j]     + cur_k[j]     * v_new;
            float h1 = g_t * H_reg[j + 1] + cur_k[j + 1] * v_new;
            float h2 = g_t * H_reg[j + 2] + cur_k[j + 2] * v_new;
            float h3 = g_t * H_reg[j + 3] + cur_k[j + 3] * v_new;
            H_reg[j]     = h0;
            H_reg[j + 1] = h1;
            H_reg[j + 2] = h2;
            H_reg[j + 3] = h3;
            qd0 += h0 * cur_q[j];
            qd1 += h1 * cur_q[j + 1];
            qd2 += h2 * cur_q[j + 2];
            qd3 += h3 * cur_q[j + 3];
        }
        float q_dot = (qd0 + qd1) + (qd2 + qd3);

        output[out_batch_off + ((unsigned long long)t * num_v_heads + vh) * v_dim + tid] =
            __float2bfloat16(q_dot * inv_sqrt_d);

        __syncthreads();
    }

store_h_split4_batched:
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_global[j * v_dim + tid] = H_reg[j];
    }
}


// ═══════════════════════════════════════════════════════════════════
// Decode, Chunk2, Chunk3 kernels — identical to parent (no changes).
// ═══════════════════════════════════════════════════════════════════

#define BLOCK_SIZE 128

extern "C" __global__ void gated_delta_rule_decode(
    float* __restrict__ h_state,
    const float* __restrict__ query,       // FP32 — prevents recurrent precision drift
    const float* __restrict__ key,         // FP32
    const float* __restrict__ value,       // FP32
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
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
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    float* H = h_state + ((b * num_v_heads + vh) * k_dim * v_dim);
    const float* q_ptr = query + (b * num_k_heads + kh) * k_dim;
    const float* k_ptr = key + (b * num_k_heads + kh) * k_dim;
    const float* v_ptr = value + (b * num_v_heads + vh) * v_dim;
    const float g = fminf(fmaxf(gate[b * num_v_heads + vh], 1e-6f), 1.0f - 1e-6f);
    const float bt = beta[b * num_v_heads + vh];
    __shared__ float smem_k[128];
    __shared__ float smem_q[128];
    if (tid < k_dim) {
        smem_k[tid] = k_ptr[tid];
        smem_q[tid] = q_ptr[tid];
    }
    __syncthreads();
    if (tid < v_dim) {
        float v_i = v_ptr[tid];
        float hk_dot = 0.0f;
        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H[(j+0)*v_dim+tid]; float h1 = H[(j+1)*v_dim+tid];
            float h2 = H[(j+2)*v_dim+tid]; float h3 = H[(j+3)*v_dim+tid];
            hk_dot += h0*smem_k[j] + h1*smem_k[j+1] + h2*smem_k[j+2] + h3*smem_k[j+3];
        }
        float v_new_i = (v_i - g * hk_dot) * bt;
        float q_dot = 0.0f;
        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H[(j+0)*v_dim+tid]; float h1 = H[(j+1)*v_dim+tid];
            float h2 = H[(j+2)*v_dim+tid]; float h3 = H[(j+3)*v_dim+tid];
            h0 = g*h0 + smem_k[j]*v_new_i;     h1 = g*h1 + smem_k[j+1]*v_new_i;
            h2 = g*h2 + smem_k[j+2]*v_new_i;   h3 = g*h3 + smem_k[j+3]*v_new_i;
            H[(j+0)*v_dim+tid]=h0; H[(j+1)*v_dim+tid]=h1;
            H[(j+2)*v_dim+tid]=h2; H[(j+3)*v_dim+tid]=h3;
            q_dot += h0*smem_q[j] + h1*smem_q[j+1] + h2*smem_q[j+2] + h3*smem_q[j+3];
        }
        // ── SSM state normalization (Stuffed Mamba mitigation) ──
        {
            float local_sq = 0.0f;
            for (unsigned int j = 0; j < k_dim; j++)
                local_sq += H[j*v_dim+tid] * H[j*v_dim+tid];
            unsigned int mask = __activemask();
            float ws = local_sq;
            ws += __shfl_down_sync(mask, ws, 16);
            ws += __shfl_down_sync(mask, ws, 8);
            ws += __shfl_down_sync(mask, ws, 4);
            ws += __shfl_down_sync(mask, ws, 2);
            ws += __shfl_down_sync(mask, ws, 1);
            __shared__ float ns[4];
            if (tid % 32 == 0) ns[tid / 32] = ws;
            __syncthreads();
            if (tid < 4) { float s = ns[tid]; s += __shfl_down_sync(0xf, s, 2); s += __shfl_down_sync(0xf, s, 1); ns[0] = s; }
            __syncthreads();
            const float MAX_NORM = 50.0f;
            if (ns[0] > MAX_NORM * MAX_NORM) {
                float scale = MAX_NORM * rsqrtf(ns[0]);
                for (unsigned int j = 0; j < k_dim; j++)
                    H[j*v_dim+tid] *= scale;
            }
        }

        float inv_sqrt_d = rsqrtf((float)k_dim);
        output[(b*num_v_heads+vh)*v_dim+tid] = __float2bfloat16(q_dot*inv_sqrt_d);
    }
}

// FP32 output variant — eliminates BF16 truncation in recurrent path.
extern "C" __global__ void gated_delta_rule_decode_f32(
    float* __restrict__ h_state,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ output,
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
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    float* H = h_state + ((b * num_v_heads + vh) * k_dim * v_dim);
    const float* q_ptr = query + (b * num_k_heads + kh) * k_dim;
    const float* k_ptr = key + (b * num_k_heads + kh) * k_dim;
    const float* v_ptr = value + (b * num_v_heads + vh) * v_dim;
    const float g = fminf(fmaxf(gate[b * num_v_heads + vh], 1e-6f), 1.0f - 1e-6f);
    const float bt = beta[b * num_v_heads + vh];
    __shared__ float smem_k[128];
    __shared__ float smem_q[128];
    if (tid < k_dim) {
        smem_k[tid] = k_ptr[tid];
        smem_q[tid] = q_ptr[tid];
    }
    __syncthreads();
    if (tid < v_dim) {
        float v_i = v_ptr[tid];
        float hk_dot = 0.0f;
        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H[(j+0)*v_dim+tid]; float h1 = H[(j+1)*v_dim+tid];
            float h2 = H[(j+2)*v_dim+tid]; float h3 = H[(j+3)*v_dim+tid];
            hk_dot += h0*smem_k[j] + h1*smem_k[j+1] + h2*smem_k[j+2] + h3*smem_k[j+3];
        }
        float v_new_i = (v_i - g * hk_dot) * bt;
        float q_dot = 0.0f;
        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H[(j+0)*v_dim+tid]; float h1 = H[(j+1)*v_dim+tid];
            float h2 = H[(j+2)*v_dim+tid]; float h3 = H[(j+3)*v_dim+tid];
            h0 = g*h0 + smem_k[j]*v_new_i;     h1 = g*h1 + smem_k[j+1]*v_new_i;
            h2 = g*h2 + smem_k[j+2]*v_new_i;   h3 = g*h3 + smem_k[j+3]*v_new_i;
            H[(j+0)*v_dim+tid]=h0; H[(j+1)*v_dim+tid]=h1;
            H[(j+2)*v_dim+tid]=h2; H[(j+3)*v_dim+tid]=h3;
            q_dot += h0*smem_q[j] + h1*smem_q[j+1] + h2*smem_q[j+2] + h3*smem_q[j+3];
        }
        // SSM state normalization (same as BF16 variant)
        {
            float local_sq = 0.0f;
            for (unsigned int j = 0; j < k_dim; j++)
                local_sq += H[j*v_dim+tid] * H[j*v_dim+tid];
            unsigned int mask = __activemask();
            float ws = local_sq;
            ws += __shfl_down_sync(mask, ws, 16);
            ws += __shfl_down_sync(mask, ws, 8);
            ws += __shfl_down_sync(mask, ws, 4);
            ws += __shfl_down_sync(mask, ws, 2);
            ws += __shfl_down_sync(mask, ws, 1);
            __shared__ float ns[4];
            if (tid % 32 == 0) ns[tid / 32] = ws;
            __syncthreads();
            if (tid < 4) { float s = ns[tid]; s += __shfl_down_sync(0xf, s, 2); s += __shfl_down_sync(0xf, s, 1); ns[0] = s; }
            __syncthreads();
            const float MAX_NORM = 50.0f;
            if (ns[0] > MAX_NORM * MAX_NORM) {
                float scale = MAX_NORM * rsqrtf(ns[0]);
                for (unsigned int j = 0; j < k_dim; j++)
                    H[j*v_dim+tid] *= scale;
            }
        }
        float inv_sqrt_d = rsqrtf((float)k_dim);
        output[(b*num_v_heads+vh)*v_dim+tid] = q_dot * inv_sqrt_d;  // FP32 direct
    }
}

extern "C" __global__ void gated_delta_rule_chunk2(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ h_state_intermediate,
    unsigned int batch_size,
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
    const unsigned int hv_size = k_dim * v_dim;
    float* H = h_state + ((b*num_v_heads+vh)*hv_size);
    float* H_inter = h_state_intermediate + ((b*num_v_heads+vh)*hv_size);
    const __nv_bfloat16* q0=query+(b*2)*qk_stride+kh*k_dim;
    const __nv_bfloat16* k0=key+(b*2)*qk_stride+kh*k_dim;
    const __nv_bfloat16* v0=value+(b*2)*v_stride+vh*v_dim;
    const float g0=fminf(fmaxf(gate[(b*2)*gb_stride+vh], 1e-6f), 1.0f - 1e-6f), bt0=beta[(b*2)*gb_stride+vh];
    const __nv_bfloat16* q1=query+(b*2+1)*qk_stride+kh*k_dim;
    const __nv_bfloat16* k1=key+(b*2+1)*qk_stride+kh*k_dim;
    const __nv_bfloat16* v1=value+(b*2+1)*v_stride+vh*v_dim;
    const float g1=fminf(fmaxf(gate[(b*2+1)*gb_stride+vh], 1e-6f), 1.0f - 1e-6f), bt1=beta[(b*2+1)*gb_stride+vh];
    __shared__ float sk0[128],sq0[128],sk1[128],sq1[128];
    if (tid<k_dim) {
        sk0[tid]=(float)k0[tid]; sq0[tid]=(float)q0[tid];
        sk1[tid]=(float)k1[tid]; sq1[tid]=(float)q1[tid];
    }
    __syncthreads();
    if (tid<v_dim) {
        float vi0=(float)v0[tid], vi1=(float)v1[tid];
        float hk0=0.0f;
        #pragma unroll 4
        for (unsigned int j=0;j<k_dim;j+=4) {
            float h0=H[(j+0)*v_dim+tid],h1=H[(j+1)*v_dim+tid],h2=H[(j+2)*v_dim+tid],h3=H[(j+3)*v_dim+tid];
            hk0+=h0*sk0[j]+h1*sk0[j+1]+h2*sk0[j+2]+h3*sk0[j+3];
        }
        float v_new_0=(vi0-g0*hk0)*bt0;
        float q0_dot=0.0f,hk1=0.0f;
        #pragma unroll 4
        for (unsigned int j=0;j<k_dim;j+=4) {
            float h0=H[(j+0)*v_dim+tid],h1=H[(j+1)*v_dim+tid],h2=H[(j+2)*v_dim+tid],h3=H[(j+3)*v_dim+tid];
            h0=g0*h0+sk0[j]*v_new_0; h1=g0*h1+sk0[j+1]*v_new_0;
            h2=g0*h2+sk0[j+2]*v_new_0; h3=g0*h3+sk0[j+3]*v_new_0;
            H_inter[(j+0)*v_dim+tid]=h0; H_inter[(j+1)*v_dim+tid]=h1;
            H_inter[(j+2)*v_dim+tid]=h2; H_inter[(j+3)*v_dim+tid]=h3;
            q0_dot+=h0*sq0[j]+h1*sq0[j+1]+h2*sq0[j+2]+h3*sq0[j+3];
            hk1+=h0*sk1[j]+h1*sk1[j+1]+h2*sk1[j+2]+h3*sk1[j+3];
        }
        float v_new_1=(vi1-g1*hk1)*bt1;
        float q1_dot=0.0f;
        #pragma unroll 4
        for (unsigned int j=0;j<k_dim;j+=4) {
            float h0=H_inter[(j+0)*v_dim+tid],h1=H_inter[(j+1)*v_dim+tid],h2=H_inter[(j+2)*v_dim+tid],h3=H_inter[(j+3)*v_dim+tid];
            h0=g1*h0+sk1[j]*v_new_1; h1=g1*h1+sk1[j+1]*v_new_1;
            h2=g1*h2+sk1[j+2]*v_new_1; h3=g1*h3+sk1[j+3]*v_new_1;
            H[(j+0)*v_dim+tid]=h0; H[(j+1)*v_dim+tid]=h1;
            H[(j+2)*v_dim+tid]=h2; H[(j+3)*v_dim+tid]=h3;
            q1_dot+=h0*sq1[j]+h1*sq1[j+1]+h2*sq1[j+2]+h3*sq1[j+3];
        }
        float inv_sqrt_d=rsqrtf((float)k_dim);
        output[(b*2*num_v_heads+vh)*v_dim+tid]=__float2bfloat16(q0_dot*inv_sqrt_d);
        output[((b*2+1)*num_v_heads+vh)*v_dim+tid]=__float2bfloat16(q1_dot*inv_sqrt_d);
    }
}

extern "C" __global__ void gated_delta_rule_chunk3(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ h_state_inter0,
    float* __restrict__ h_state_inter1,
    unsigned int batch_size,
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
    const unsigned int hv_size = k_dim * v_dim;
    float* H = h_state + ((b*num_v_heads+vh)*hv_size);
    float* Hi0 = h_state_inter0 + ((b*num_v_heads+vh)*hv_size);
    float* Hi1 = h_state_inter1 + ((b*num_v_heads+vh)*hv_size);
    const __nv_bfloat16* q0=query+(b*3)*qk_stride+kh*k_dim;
    const __nv_bfloat16* k0=key+(b*3)*qk_stride+kh*k_dim;
    const __nv_bfloat16* v0=value+(b*3)*v_stride+vh*v_dim;
    const float g0=fminf(fmaxf(gate[(b*3)*gb_stride+vh], 1e-6f), 1.0f - 1e-6f), bt0=beta[(b*3)*gb_stride+vh];
    const __nv_bfloat16* q1=query+(b*3+1)*qk_stride+kh*k_dim;
    const __nv_bfloat16* k1=key+(b*3+1)*qk_stride+kh*k_dim;
    const __nv_bfloat16* v1=value+(b*3+1)*v_stride+vh*v_dim;
    const float g1=fminf(fmaxf(gate[(b*3+1)*gb_stride+vh], 1e-6f), 1.0f - 1e-6f), bt1=beta[(b*3+1)*gb_stride+vh];
    const __nv_bfloat16* q2=query+(b*3+2)*qk_stride+kh*k_dim;
    const __nv_bfloat16* k2=key+(b*3+2)*qk_stride+kh*k_dim;
    const __nv_bfloat16* v2=value+(b*3+2)*v_stride+vh*v_dim;
    const float g2=fminf(fmaxf(gate[(b*3+2)*gb_stride+vh], 1e-6f), 1.0f - 1e-6f), bt2=beta[(b*3+2)*gb_stride+vh];
    __shared__ float sk0[128],sq0[128],sk1[128],sq1[128],sk2[128],sq2[128];
    if (tid<k_dim) {
        sk0[tid]=(float)k0[tid]; sq0[tid]=(float)q0[tid];
        sk1[tid]=(float)k1[tid]; sq1[tid]=(float)q1[tid];
        sk2[tid]=(float)k2[tid]; sq2[tid]=(float)q2[tid];
    }
    __syncthreads();
    if (tid<v_dim) {
        float vi0=(float)v0[tid],vi1=(float)v1[tid],vi2=(float)v2[tid];
        float hk0=0.0f;
        #pragma unroll 4
        for (unsigned int j=0;j<k_dim;j+=4) {
            float h0=H[(j+0)*v_dim+tid],h1=H[(j+1)*v_dim+tid],h2=H[(j+2)*v_dim+tid],h3=H[(j+3)*v_dim+tid];
            hk0+=h0*sk0[j]+h1*sk0[j+1]+h2*sk0[j+2]+h3*sk0[j+3];
        }
        float v_new_0=(vi0-g0*hk0)*bt0;
        float q0_dot=0.0f,hk1=0.0f;
        #pragma unroll 4
        for (unsigned int j=0;j<k_dim;j+=4) {
            float h0=H[(j+0)*v_dim+tid],h1=H[(j+1)*v_dim+tid],h2=H[(j+2)*v_dim+tid],h3=H[(j+3)*v_dim+tid];
            h0=g0*h0+sk0[j]*v_new_0; h1=g0*h1+sk0[j+1]*v_new_0;
            h2=g0*h2+sk0[j+2]*v_new_0; h3=g0*h3+sk0[j+3]*v_new_0;
            Hi0[(j+0)*v_dim+tid]=h0; Hi0[(j+1)*v_dim+tid]=h1;
            Hi0[(j+2)*v_dim+tid]=h2; Hi0[(j+3)*v_dim+tid]=h3;
            q0_dot+=h0*sq0[j]+h1*sq0[j+1]+h2*sq0[j+2]+h3*sq0[j+3];
            hk1+=h0*sk1[j]+h1*sk1[j+1]+h2*sk1[j+2]+h3*sk1[j+3];
        }
        float v_new_1=(vi1-g1*hk1)*bt1;
        float q1_dot=0.0f,hk2=0.0f;
        #pragma unroll 4
        for (unsigned int j=0;j<k_dim;j+=4) {
            float h0=Hi0[(j+0)*v_dim+tid],h1=Hi0[(j+1)*v_dim+tid],h2=Hi0[(j+2)*v_dim+tid],h3=Hi0[(j+3)*v_dim+tid];
            h0=g1*h0+sk1[j]*v_new_1; h1=g1*h1+sk1[j+1]*v_new_1;
            h2=g1*h2+sk1[j+2]*v_new_1; h3=g1*h3+sk1[j+3]*v_new_1;
            Hi1[(j+0)*v_dim+tid]=h0; Hi1[(j+1)*v_dim+tid]=h1;
            Hi1[(j+2)*v_dim+tid]=h2; Hi1[(j+3)*v_dim+tid]=h3;
            q1_dot+=h0*sq1[j]+h1*sq1[j+1]+h2*sq1[j+2]+h3*sq1[j+3];
            hk2+=h0*sk2[j]+h1*sk2[j+1]+h2*sk2[j+2]+h3*sk2[j+3];
        }
        float v_new_2=(vi2-g2*hk2)*bt2;
        float q2_dot=0.0f;
        #pragma unroll 4
        for (unsigned int j=0;j<k_dim;j+=4) {
            float h0=Hi1[(j+0)*v_dim+tid],h1=Hi1[(j+1)*v_dim+tid],h2=Hi1[(j+2)*v_dim+tid],h3=Hi1[(j+3)*v_dim+tid];
            h0=g2*h0+sk2[j]*v_new_2; h1=g2*h1+sk2[j+1]*v_new_2;
            h2=g2*h2+sk2[j+2]*v_new_2; h3=g2*h3+sk2[j+3]*v_new_2;
            H[(j+0)*v_dim+tid]=h0; H[(j+1)*v_dim+tid]=h1;
            H[(j+2)*v_dim+tid]=h2; H[(j+3)*v_dim+tid]=h3;
            q2_dot+=h0*sq2[j]+h1*sq2[j+1]+h2*sq2[j+2]+h3*sq2[j+3];
        }
        float inv_sqrt_d=rsqrtf((float)k_dim);
        output[(b*3*num_v_heads+vh)*v_dim+tid]=__float2bfloat16(q0_dot*inv_sqrt_d);
        output[((b*3+1)*num_v_heads+vh)*v_dim+tid]=__float2bfloat16(q1_dot*inv_sqrt_d);
        output[((b*3+2)*num_v_heads+vh)*v_dim+tid]=__float2bfloat16(q2_dot*inv_sqrt_d);
    }
}

// ─── Prelude for the kernels ported from common/gated_delta_rule.cu ───
// Verbatim from common/. This fork defined none of them, and used none of
// them, so adding them cannot affect the kernels already in this file.
__device__ __forceinline__ void gdn_unpack_bf16x2(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

__device__ __forceinline__ unsigned int gdn_pack_bf16x2(float v0, float v1) {
    unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
    unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
    return lo | (hi << 16);
}

__device__ __forceinline__ float gdn_warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// MAX_STATE_NORM: if ||h[head]||_F > this, scale h down. 0 = disabled.
// 1000.0 allows natural state growth while still preventing FP32 overflow.
#ifndef SSM_STATE_NORM_ENABLED
#define SSM_STATE_NORM_ENABLED
#define SSM_STATE_MAX_NORM 1000.0f
#endif

// ─── Ported verbatim from kernels/gb10/common/gated_delta_rule.cu ───────
// Shadow-drift repair. This model-specific file whole-file-shadows common/,
// so every kernel below was silently DROPPED (try_kernel -> handle 0) even
// though qwen3_ssm/init.rs requests all of them unconditionally. Exact
// piecewise copy -- same signature, same h_state layout as this file's own
// decode kernels. Do not edit here without re-syncing common/.
// ───────────────────────────────────────────────────────────────────────

// ============================================================================
// FUSED conv1d_update_l2norm + recurrence + gated-RMS-norm decode kernel.
//
// Collapses the per-seq SSM decode critical-path chain `conv1d_l2norm -> gdn ->
// gated_norm` (3 dependent kernels) into ONE, shortening the decode dependency
// chain (decode is GPU-~93%-idle = chain-depth bound; the existing gdn+norm
// fusion already gave +18% c4). Race-free via a PER-K-HEAD grid: each block owns
// k-head kh AND its `head_repeat` v-heads, so it conv-updates its own q/k AND v
// conv_state exclusively (no cross-block conv_state share -> no race).
//
// Grid: [num_k_heads, batch].  Block: head_repeat * v_dim threads.
//   thread tid: which_v = tid / v_dim, vlocal = tid % v_dim, vh = kh*head_repeat+which_v.
// Requires (validated by the Rust dispatch gate): head_repeat*v_dim == blockDim,
//   2*k_dim <= blockDim (q+k conv covered one-per-thread), k_dim == v_dim.
// new_input/conv layout is [Q(num_k_heads*k_dim) | K(...) | V(num_v_heads*v_dim)].
extern "C" __global__ void gated_delta_rule_decode_f32_conv_norm(
    float* __restrict__ h_state,
    float* __restrict__ conv_state,            // [batch, conv_dim, d_conv]
    const __nv_bfloat16* __restrict__ new_input, // [batch, conv_dim] deint qkvz
    const __nv_bfloat16* __restrict__ conv_weight, // [conv_dim, d_conv]
    const float* __restrict__ conv_bias,       // [conv_dim] or null
    const float* __restrict__ gate,            // [batch, num_v_heads]
    const float* __restrict__ beta,            // [batch, num_v_heads]
    const __nv_bfloat16* __restrict__ z_gate,  // [batch, num_v_heads, v_dim]
    const __nv_bfloat16* __restrict__ norm_weight, // [v_dim]
    __nv_bfloat16* __restrict__ output,        // [batch, num_v_heads, v_dim]
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int conv_dim,
    unsigned int d_conv,
    float l2_eps,
    float eps
) {
    const unsigned int kh = blockIdx.x;
    const unsigned int b  = blockIdx.y;
    if (kh >= num_k_heads || b >= batch_size) return;

    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int tid     = threadIdx.x;
    const unsigned int which_v = tid / v_dim;       // 0..head_repeat-1
    const unsigned int vlocal  = tid % v_dim;       // 0..v_dim-1
    const unsigned int vh      = kh * head_repeat + which_v;
    const unsigned int key_dim_total = num_k_heads * k_dim;   // Q block width

    __shared__ float smem_q[128];
    __shared__ float smem_k[128];
    __shared__ float warp_sums[8];     // 8 warps for a 256-thread block

    // ---- Step 1: conv1d_update + SiLU for this block's q/k channel (one/thread) ----
    // tid [0,k_dim) -> q head kh ; tid [k_dim,2*k_dim) -> k head kh.
    float qk_silu = 0.0f;
    const bool is_q = (tid < k_dim);
    const bool is_k = (tid >= k_dim && tid < 2 * k_dim);
    if (is_q || is_k) {
        const unsigned int ch = is_q ? (kh * k_dim + tid)
                                     : (key_dim_total + kh * k_dim + (tid - k_dim));
        float* state = conv_state + ((unsigned long long)(b * conv_dim + ch)) * d_conv;
        for (unsigned int i = 0; i < d_conv - 1; i++) state[i] = state[i + 1];
        state[d_conv - 1] = (float)new_input[b * conv_dim + ch];
        const __nv_bfloat16* w = conv_weight + (unsigned long long)ch * d_conv;
        float acc = (conv_bias != nullptr) ? conv_bias[ch] : 0.0f;
        for (unsigned int k = 0; k < d_conv; k++) acc += state[k] * (float)w[k];
        qk_silu = acc / (1.0f + __expf(-acc));
    }
    // L2-norm q (warps 0-3) and k (warps 4-7) groups separately.
    {
        float sq = qk_silu * qk_silu;
        for (int off = 16; off >= 1; off >>= 1) sq += __shfl_down_sync(0xFFFFFFFF, sq, off);
        if ((tid & 31) == 0) warp_sums[tid >> 5] = sq;
        __syncthreads();
        const unsigned int grp = tid >> 7;   // 0 = q, 1 = k
        float total = warp_sums[grp * 4 + 0] + warp_sums[grp * 4 + 1]
                    + warp_sums[grp * 4 + 2] + warp_sums[grp * 4 + 3];
        float inv = rsqrtf(total + l2_eps);
        if (is_q) smem_q[tid] = qk_silu * inv;
        else if (is_k) smem_k[tid - k_dim] = qk_silu * inv;
    }

    // ---- Step 1b: conv1d_update + SiLU for this thread's V channel (no L2) ----
    float v_i = 0.0f;
    {
        const unsigned int vch = 2 * key_dim_total + vh * v_dim + vlocal;
        float* state = conv_state + ((unsigned long long)(b * conv_dim + vch)) * d_conv;
        for (unsigned int i = 0; i < d_conv - 1; i++) state[i] = state[i + 1];
        state[d_conv - 1] = (float)new_input[b * conv_dim + vch];
        const __nv_bfloat16* w = conv_weight + (unsigned long long)vch * d_conv;
        float acc = (conv_bias != nullptr) ? conv_bias[vch] : 0.0f;
        for (unsigned int k = 0; k < d_conv; k++) acc += state[k] * (float)w[k];
        v_i = acc / (1.0f + __expf(-acc));
    }
    __syncthreads();   // smem_q / smem_k complete before recurrence reads them

    // ---- Step 2: gated-delta recurrence for v-head vh, v_dim index vlocal ----
    float* H = h_state + ((unsigned long long)(b * num_v_heads + vh)) * k_dim * v_dim;
    const float g_raw = gate[b * num_v_heads + vh];
    const float g  = fminf(fmaxf(g_raw, 1e-6f), 1.0f - 1e-6f);
    const float bt = beta[b * num_v_heads + vh];

    float hk_dot = 0.0f;
    #pragma unroll 4
    for (unsigned int j = 0; j < k_dim; j += 4) {
        float h0 = H[(j + 0) * v_dim + vlocal];
        float h1 = H[(j + 1) * v_dim + vlocal];
        float h2 = H[(j + 2) * v_dim + vlocal];
        float h3 = H[(j + 3) * v_dim + vlocal];
        hk_dot += h0 * smem_k[j] + h1 * smem_k[j+1] + h2 * smem_k[j+2] + h3 * smem_k[j+3];
    }
    const float v_new = (v_i - g * hk_dot) * bt;

    float q_dot = 0.0f;
    #pragma unroll 4
    for (unsigned int j = 0; j < k_dim; j += 4) {
        float h0 = g * H[(j+0)*v_dim+vlocal] + smem_k[j]   * v_new;
        float h1 = g * H[(j+1)*v_dim+vlocal] + smem_k[j+1] * v_new;
        float h2 = g * H[(j+2)*v_dim+vlocal] + smem_k[j+2] * v_new;
        float h3 = g * H[(j+3)*v_dim+vlocal] + smem_k[j+3] * v_new;
        H[(j+0)*v_dim+vlocal] = h0;
        H[(j+1)*v_dim+vlocal] = h1;
        H[(j+2)*v_dim+vlocal] = h2;
        H[(j+3)*v_dim+vlocal] = h3;
        q_dot += h0 * smem_q[j] + h1 * smem_q[j+1] + h2 * smem_q[j+2] + h3 * smem_q[j+3];
    }

    // ---- State-norm clamp (per v-head; reduce within this which_v group) ----
    #ifdef SSM_STATE_NORM_ENABLED
    {
        float local_sq = 0.0f;
        for (unsigned int j = 0; j < k_dim; j++) {
            float hv = H[j * v_dim + vlocal];
            local_sq += hv * hv;
        }
        for (int off = 16; off >= 1; off >>= 1)
            local_sq += __shfl_down_sync(0xFFFFFFFF, local_sq, off);
        __syncthreads();   // reuse warp_sums
        if ((tid & 31) == 0) warp_sums[tid >> 5] = local_sq;
        __syncthreads();
        const unsigned int grp = tid >> 7;
        float head_norm_sq = warp_sums[grp*4+0] + warp_sums[grp*4+1]
                           + warp_sums[grp*4+2] + warp_sums[grp*4+3];
        if (head_norm_sq > SSM_STATE_MAX_NORM * SSM_STATE_MAX_NORM) {
            float scale = SSM_STATE_MAX_NORM * rsqrtf(head_norm_sq);
            for (unsigned int j = 0; j < k_dim; j++) H[j * v_dim + vlocal] *= scale;
        }
    }
    #endif

    // ---- Step 3: gated RMS norm (per v-head) -> output ----
    const float x = q_dot * rsqrtf((float)k_dim);
    float sum_sq = x * x;
    for (int off = 16; off >= 1; off >>= 1) sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, off);
    __syncthreads();   // reuse warp_sums
    if ((tid & 31) == 0) warp_sums[tid >> 5] = sum_sq;
    __syncthreads();
    {
        const unsigned int grp = tid >> 7;
        float total = warp_sums[grp*4+0] + warp_sums[grp*4+1]
                    + warp_sums[grp*4+2] + warp_sums[grp*4+3];
        const float rms = rsqrtf(total / (float)v_dim + eps);
        const float zg = (float)z_gate[(unsigned long long)(b * num_v_heads + vh) * v_dim + vlocal];
        const float wv = (float)norm_weight[vlocal];
        const float sg = zg / (1.0f + expf(-zg));
        output[(unsigned long long)(b * num_v_heads + vh) * v_dim + vlocal]
            = __float2bfloat16(x * rms * wv * sg);
    }
}

// FP32 GDN decode fused with gated RMS norm.
//
// Computes the same FP32 recurrent update as gated_delta_rule_decode_f32, then
// normalizes the per-v-head output and applies the BF16 Z gate + norm weight in
// the same CTA. Output matches gated_rms_norm_f32_input's mathematical path but
// skips the intermediate FP32 global write/read and one launch.
extern "C" __global__ void gated_delta_rule_decode_f32_norm(
    float* __restrict__ h_state,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    const __nv_bfloat16* __restrict__ z_gate,
    const __nv_bfloat16* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    float eps
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    if (tid >= v_dim) return;

    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    float* H = h_state + ((b * num_v_heads + vh) * k_dim * v_dim);
    const float* q_ptr = query + (b * num_k_heads + kh) * k_dim;
    const float* k_ptr = key + (b * num_k_heads + kh) * k_dim;
    const float* v_ptr = value + (b * num_v_heads + vh) * v_dim;

    float g_raw = gate[b * num_v_heads + vh];
    const float g = fminf(fmaxf(g_raw, 1e-6f), 1.0f - 1e-6f);
    const float bt = beta[b * num_v_heads + vh];

    __shared__ float smem_k[128];
    __shared__ float smem_q[128];

    if (tid < k_dim) {
        smem_k[tid] = k_ptr[tid];
        smem_q[tid] = q_ptr[tid];
    }
    __syncthreads();

    float v_i = v_ptr[tid];
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
#ifdef SSM_STATE_NORM_ENABLED
    float norm_acc = 0.0f;
#endif
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
#ifdef SSM_STATE_NORM_ENABLED
        // Frobenius accumulation from the registers we just stored. The clamp
        // below used to re-read all of H to get these -- a THIRD full pass over
        // 3 MB per seq per layer, immediately after writing it. One add at a
        // time in ascending j keeps the summation order identical to that loop,
        // so the result is bit-identical, not merely equivalent.
        norm_acc += h0 * h0;
        norm_acc += h1 * h1;
        norm_acc += h2 * h2;
        norm_acc += h3 * h3;
#endif
    }

    #ifdef SSM_STATE_NORM_ENABLED
    {
        float local_sq = norm_acc;
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

    const float inv_sqrt_d = rsqrtf((float)k_dim);
    const float x = q_dot * inv_sqrt_d;

    __shared__ float x_cache[128];
    x_cache[tid] = x;

    float sum_sq = x * x;
    sum_sq = gdn_warp_reduce_sum(sum_sq);
    __shared__ float rms_sums[4];
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    if (lane_id == 0) rms_sums[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? rms_sums[lane_id] : 0.0f;
        val = gdn_warp_reduce_sum(val);
        if (lane_id == 0) rms_sums[0] = val;
    }
    __syncthreads();

    const float rms = rsqrtf(rms_sums[0] / (float)v_dim + eps);

    const unsigned int quad_size = v_dim / 4;
    const unsigned long long* g64 = (const unsigned long long*)(z_gate + vh * v_dim);
    const unsigned long long* w64 = (const unsigned long long*)norm_weight;
    unsigned long long* out64 = (unsigned long long*)(output + vh * v_dim);
    for (unsigned int i = tid; i < quad_size; i += blockDim.x) {
        unsigned int base = i * 4;
        float f0 = x_cache[base];
        float f1 = x_cache[base + 1];
        float f2 = x_cache[base + 2];
        float f3 = x_cache[base + 3];

        unsigned long long wv = w64[i];
        float w0, w1, w2, w3;
        gdn_unpack_bf16x2((unsigned int)wv, w0, w1);
        gdn_unpack_bf16x2((unsigned int)(wv >> 32), w2, w3);

        unsigned long long gv = g64[i];
        float g0, g1, g2, g3;
        gdn_unpack_bf16x2((unsigned int)gv, g0, g1);
        gdn_unpack_bf16x2((unsigned int)(gv >> 32), g2, g3);

        float s0 = g0 / (1.0f + expf(-g0));
        float s1 = g1 / (1.0f + expf(-g1));
        float s2 = g2 / (1.0f + expf(-g2));
        float s3 = g3 / (1.0f + expf(-g3));

        unsigned int lo = gdn_pack_bf16x2(f0 * rms * w0 * s0, f1 * rms * w1 * s1);
        unsigned int hi = gdn_pack_bf16x2(f2 * rms * w2 * s2, f3 * rms * w3 * s3);
        out64[i] = ((unsigned long long)hi << 32) | (unsigned long long)lo;
    }
}

// FP32 strided decode variant for concurrent decode sequences.
//
// Unlike gated_delta_rule_decode_f32, Q/K/V are rows inside a wider
// per-sequence conv output buffer: [Q | K | V]. The stride arguments let the
// Rust multi-seq path batch independent sequence slots without repacking QKV.
extern "C" __global__ void gated_delta_rule_decode_f32_strided(
    float* __restrict__ h_state,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ output,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,
    unsigned int out_stride
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    if (tid >= v_dim) return;

    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    float* H = h_state + ((b * num_v_heads + vh) * k_dim * v_dim);
    const float* q_ptr = query + (unsigned long long)b * qk_stride + kh * k_dim;
    const float* k_ptr = key + (unsigned long long)b * qk_stride + kh * k_dim;
    const float* v_ptr = value + (unsigned long long)b * v_stride + vh * v_dim;

    float g_raw = gate[(unsigned long long)b * gb_stride + vh];
    const float g = fminf(fmaxf(g_raw, 1e-6f), 1.0f - 1e-6f);
    const float bt = beta[(unsigned long long)b * gb_stride + vh];

    __shared__ float smem_k[128];
    __shared__ float smem_q[128];

    if (tid < k_dim) {
        smem_k[tid] = k_ptr[tid];
        smem_q[tid] = q_ptr[tid];
    }
    __syncthreads();

    float v_i = v_ptr[tid];
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
#ifdef SSM_STATE_NORM_ENABLED
    float norm_acc = 0.0f;
#endif
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
#ifdef SSM_STATE_NORM_ENABLED
        // Frobenius accumulation from the registers we just stored. The clamp
        // below used to re-read all of H to get these -- a THIRD full pass over
        // 3 MB per seq per layer, immediately after writing it. One add at a
        // time in ascending j keeps the summation order identical to that loop,
        // so the result is bit-identical, not merely equivalent.
        norm_acc += h0 * h0;
        norm_acc += h1 * h1;
        norm_acc += h2 * h2;
        norm_acc += h3 * h3;
#endif
    }

    #ifdef SSM_STATE_NORM_ENABLED
    {
        float local_sq = norm_acc;
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
    output[(unsigned long long)b * out_stride + vh * v_dim + tid] = q_dot * inv_sqrt_d;
}

// FP32 strided decode variant fused with gated RMS norm.
//
// Used by the experimental concurrent SSM recurrent path. It keeps the same
// state-update math as gated_delta_rule_decode_f32_strided but writes the
// post-norm BF16 output directly, avoiding an FP32 global output row plus N
// separate gated_rms_norm launches.
extern "C" __global__ void gated_delta_rule_decode_f32_strided_norm(
    float* __restrict__ h_state,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    const __nv_bfloat16* __restrict__ z_gate,
    const __nv_bfloat16* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,
    unsigned int z_stride,
    unsigned int out_stride,
    float eps
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    if (tid >= v_dim) return;

    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    float* H = h_state + ((b * num_v_heads + vh) * k_dim * v_dim);
    const float* q_ptr = query + (unsigned long long)b * qk_stride + kh * k_dim;
    const float* k_ptr = key + (unsigned long long)b * qk_stride + kh * k_dim;
    const float* v_ptr = value + (unsigned long long)b * v_stride + vh * v_dim;

    float g_raw = gate[(unsigned long long)b * gb_stride + vh];
    const float g = fminf(fmaxf(g_raw, 1e-6f), 1.0f - 1e-6f);
    const float bt = beta[(unsigned long long)b * gb_stride + vh];

    __shared__ float smem_k[128];
    __shared__ float smem_q[128];

    if (tid < k_dim) {
        smem_k[tid] = k_ptr[tid];
        smem_q[tid] = q_ptr[tid];
    }
    __syncthreads();

    float v_i = v_ptr[tid];
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
#ifdef SSM_STATE_NORM_ENABLED
    float norm_acc = 0.0f;
#endif
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
#ifdef SSM_STATE_NORM_ENABLED
        // Frobenius accumulation from the registers we just stored. The clamp
        // below used to re-read all of H to get these -- a THIRD full pass over
        // 3 MB per seq per layer, immediately after writing it. One add at a
        // time in ascending j keeps the summation order identical to that loop,
        // so the result is bit-identical, not merely equivalent.
        norm_acc += h0 * h0;
        norm_acc += h1 * h1;
        norm_acc += h2 * h2;
        norm_acc += h3 * h3;
#endif
    }

    #ifdef SSM_STATE_NORM_ENABLED
    {
        float local_sq = norm_acc;
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

    const float inv_sqrt_d = rsqrtf((float)k_dim);
    const float x = q_dot * inv_sqrt_d;

    __shared__ float x_cache[128];
    x_cache[tid] = x;

    float sum_sq = x * x;
    sum_sq = gdn_warp_reduce_sum(sum_sq);
    __shared__ float rms_sums[4];
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    if (lane_id == 0) rms_sums[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? rms_sums[lane_id] : 0.0f;
        val = gdn_warp_reduce_sum(val);
        if (lane_id == 0) rms_sums[0] = val;
    }
    __syncthreads();

    const float rms = rsqrtf(rms_sums[0] / (float)v_dim + eps);

    const unsigned int quad_size = v_dim / 4;
    const unsigned long long* g64 = (const unsigned long long*)(
        z_gate + (unsigned long long)b * z_stride + vh * v_dim
    );
    const unsigned long long* w64 = (const unsigned long long*)norm_weight;
    unsigned long long* out64 = (unsigned long long*)(
        output + (unsigned long long)b * out_stride + vh * v_dim
    );
    for (unsigned int i = tid; i < quad_size; i += blockDim.x) {
        unsigned int base = i * 4;
        float f0 = x_cache[base];
        float f1 = x_cache[base + 1];
        float f2 = x_cache[base + 2];
        float f3 = x_cache[base + 3];

        unsigned long long wv = w64[i];
        float w0, w1, w2, w3;
        gdn_unpack_bf16x2((unsigned int)wv, w0, w1);
        gdn_unpack_bf16x2((unsigned int)(wv >> 32), w2, w3);

        unsigned long long gv = g64[i];
        float g0, g1, g2, g3;
        gdn_unpack_bf16x2((unsigned int)gv, g0, g1);
        gdn_unpack_bf16x2((unsigned int)(gv >> 32), g2, g3);

        float s0 = g0 / (1.0f + expf(-g0));
        float s1 = g1 / (1.0f + expf(-g1));
        float s2 = g2 / (1.0f + expf(-g2));
        float s3 = g3 / (1.0f + expf(-g3));

        unsigned int lo = gdn_pack_bf16x2(f0 * rms * w0 * s0, f1 * rms * w1 * s1);
        unsigned int hi = gdn_pack_bf16x2(f2 * rms * w2 * s2, f3 * rms * w3 * s3);
        out64[i] = ((unsigned long long)hi << 32) | (unsigned long long)lo;
    }
}
