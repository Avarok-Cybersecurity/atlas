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
    // ── SSM state normalization (Stuffed Mamba mitigation) ──
    // Only during decode (seq_len <= 1). During prefill the state legitimately
    // grows large to compress context; clamping would destroy information.
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
// Register-Tiled Decode — eliminates redundant H global reads.
//
// Original decode reads H[128×128] from global memory TWICE per head:
//   Pass 1: hk_dot = H^T · k  (128 global reads per thread)
//   Pass 2: H_new, q_dot      (128 global reads + 128 writes per thread)
//   Total: 256 reads + 128 writes = 384 transactions per thread
//
// Register-tiled version:
//   Load: H → H_reg[128]      (128 global reads)
//   Pass 1: hk_dot from H_reg (0 global reads — registers)
//   Pass 2: update + q_dot    (0 global reads — registers)
//   Store: H_reg → H          (128 global writes)
//   Total: 128 reads + 128 writes = 256 transactions per thread
//
// 33% less global memory traffic for GDN state per layer.
// At 30 SSM layers × 32 v_heads × 64KB/head = 60 MB saved per decode step.
//
// __launch_bounds__(128, 1) forces max register allocation (512 regs/thread)
// to keep H_reg[128] entirely in registers (no spill to L1).
//
// Grid: (num_v_heads, batch, 1)   Block: (128, 1, 1)
// ═══════════════════════════════════════════════════════════════════

extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_decode(
    float* __restrict__ h_state,
    const float* __restrict__ query,       // FP32 (was BF16) — prevents recurrent precision drift
    const float* __restrict__ key,         // FP32 (was BF16)
    const float* __restrict__ value,       // FP32 (was BF16)
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

    float* H_global = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * v_dim);

    // Load k, q into shared memory (broadcast across all 128 threads)
    __shared__ float smem_k[K_DIM];
    __shared__ float smem_q[K_DIM];
    const float* k_ptr = key + (b * num_k_heads + kh) * k_dim;
    const float* q_ptr = query + (b * num_k_heads + kh) * k_dim;
    smem_k[tid] = k_ptr[tid];  // Already FP32, no conversion needed
    smem_q[tid] = q_ptr[tid];
    __syncthreads();

    // Load H column into registers — ONE global read per element
    float H_reg[K_DIM];
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_reg[j] = H_global[j * v_dim + tid];
    }

    float v_i = value[(b * num_v_heads + vh) * v_dim + tid];  // Already FP32
    float g = fminf(fmaxf(gate[b * num_v_heads + vh], 1e-6f), 1.0f - 1e-6f);
    float bt = beta[b * num_v_heads + vh];

    // Pass 1: hk_dot = H_reg^T · k (from registers, zero global reads)
    float hk0 = 0.0f, hk1 = 0.0f, hk2 = 0.0f, hk3 = 0.0f;
    #pragma unroll
    for (int j = 0; j < K_DIM; j += 4) {
        hk0 += H_reg[j]     * smem_k[j];
        hk1 += H_reg[j + 1] * smem_k[j + 1];
        hk2 += H_reg[j + 2] * smem_k[j + 2];
        hk3 += H_reg[j + 3] * smem_k[j + 3];
    }
    float hk_dot = (hk0 + hk1) + (hk2 + hk3);

    float v_new = (v_i - g * hk_dot) * bt;

    // Pass 2: update H_reg + compute q_dot (from registers, zero global reads)
    float qd0 = 0.0f, qd1 = 0.0f, qd2 = 0.0f, qd3 = 0.0f;
    #pragma unroll
    for (int j = 0; j < K_DIM; j += 4) {
        float h0 = g * H_reg[j]     + smem_k[j]     * v_new;
        float h1 = g * H_reg[j + 1] + smem_k[j + 1] * v_new;
        float h2 = g * H_reg[j + 2] + smem_k[j + 2] * v_new;
        float h3 = g * H_reg[j + 3] + smem_k[j + 3] * v_new;
        H_reg[j]     = h0;
        H_reg[j + 1] = h1;
        H_reg[j + 2] = h2;
        H_reg[j + 3] = h3;
        qd0 += h0 * smem_q[j];
        qd1 += h1 * smem_q[j + 1];
        qd2 += h2 * smem_q[j + 2];
        qd3 += h3 * smem_q[j + 3];
    }
    float q_dot = (qd0 + qd1) + (qd2 + qd3);

    float inv_sqrt_d = rsqrtf((float)k_dim);
    output[(b * num_v_heads + vh) * v_dim + tid] = __float2bfloat16(q_dot * inv_sqrt_d);

    // Write H from registers → global — ONE global write per element
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_global[j * v_dim + tid] = H_reg[j];
    }
}

// ═══════════════════════════════════════════════════════════════════
// FP32 OUTPUT VARIANT — eliminates BF16 truncation in recurrent path.
// Prevents cumulative precision drift at 15K+ decode tokens.
// Identical to gated_delta_rule_decode except output is float*.
// ═══════════════════════════════════════════════════════════════════

extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_decode_f32(
    float* __restrict__ h_state,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ output,            // FP32 output (was BF16)
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

    float* H_global = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * v_dim);

    __shared__ float smem_k[K_DIM];
    __shared__ float smem_q[K_DIM];
    const float* k_ptr = key + (b * num_k_heads + kh) * k_dim;
    const float* q_ptr = query + (b * num_k_heads + kh) * k_dim;
    smem_k[tid] = k_ptr[tid];
    smem_q[tid] = q_ptr[tid];
    __syncthreads();

    float H_reg[K_DIM];
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_reg[j] = H_global[j * v_dim + tid];
    }

    float v_i = value[(b * num_v_heads + vh) * v_dim + tid];
    float g = fminf(fmaxf(gate[b * num_v_heads + vh], 1e-6f), 1.0f - 1e-6f);
    float bt = beta[b * num_v_heads + vh];

    float hk0 = 0.0f, hk1 = 0.0f, hk2 = 0.0f, hk3 = 0.0f;
    #pragma unroll
    for (int j = 0; j < K_DIM; j += 4) {
        hk0 += H_reg[j]     * smem_k[j];
        hk1 += H_reg[j + 1] * smem_k[j + 1];
        hk2 += H_reg[j + 2] * smem_k[j + 2];
        hk3 += H_reg[j + 3] * smem_k[j + 3];
    }
    float hk_dot = (hk0 + hk1) + (hk2 + hk3);

    float v_new = (v_i - g * hk_dot) * bt;

    float qd0 = 0.0f, qd1 = 0.0f, qd2 = 0.0f, qd3 = 0.0f;
    #pragma unroll
    for (int j = 0; j < K_DIM; j += 4) {
        float h0 = g * H_reg[j]     + smem_k[j]     * v_new;
        float h1 = g * H_reg[j + 1] + smem_k[j + 1] * v_new;
        float h2 = g * H_reg[j + 2] + smem_k[j + 2] * v_new;
        float h3 = g * H_reg[j + 3] + smem_k[j + 3] * v_new;
        H_reg[j]     = h0;
        H_reg[j + 1] = h1;
        H_reg[j + 2] = h2;
        H_reg[j + 3] = h3;
        qd0 += h0 * smem_q[j];
        qd1 += h1 * smem_q[j + 1];
        qd2 += h2 * smem_q[j + 2];
        qd3 += h3 * smem_q[j + 3];
    }
    float q_dot = (qd0 + qd1) + (qd2 + qd3);

    float inv_sqrt_d = rsqrtf((float)k_dim);
    output[(b * num_v_heads + vh) * v_dim + tid] = q_dot * inv_sqrt_d;  // FP32 direct

    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_global[j * v_dim + tid] = H_reg[j];
    }
}

// ═══════════════════════════════════════════════════════════════════
// Register-Tiled Chunk2 — MTP K=2 verify path.
//
// Original chunk2 uses h_state_intermediate as a global-memory
// ping-pong buffer between token 0 and token 1:
//   Token 0: read H (128), update → write H_inter (128)
//   Token 1: read H_inter (128), update → write H (128)
//   Total: 256 reads + 256 writes = 512 transactions per thread
//
// Register-tiled version:
//   Load H → H_reg[128]         (128 reads)
//   Token 0: hk_dot + update    (0 reads — registers)
//   Token 1: hk_dot + update    (0 reads — registers)
//   Store H_reg → H             (128 writes)
//   Total: 128 reads + 128 writes = 256 transactions per thread
//
// 50% less global memory traffic. h_state_intermediate is NOT USED
// (parameter kept for ABI compatibility with Rust launcher).
//
// Grid: (num_v_heads, batch, 1)   Block: (128, 1, 1)
// ═══════════════════════════════════════════════════════════════════

extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_chunk2(
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

    float* H_global = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * v_dim);

    // Load k/q for both tokens into shared memory
    __shared__ float sk0[K_DIM], sq0[K_DIM], sk1[K_DIM], sq1[K_DIM];
    {
        unsigned long long qk0 = (unsigned long long)(b * 2) * qk_stride + kh * k_dim;
        unsigned long long qk1 = (unsigned long long)(b * 2 + 1) * qk_stride + kh * k_dim;
        sk0[tid] = (float)key[qk0 + tid];   sq0[tid] = (float)query[qk0 + tid];
        sk1[tid] = (float)key[qk1 + tid];   sq1[tid] = (float)query[qk1 + tid];
    }
    __syncthreads();

    // Load H column into registers — ONE read
    float H_reg[K_DIM];
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_reg[j] = H_global[j * v_dim + tid];
    }

    float vi0 = (float)value[(unsigned long long)(b * 2) * v_stride + vh * v_dim + tid];
    float vi1 = (float)value[(unsigned long long)(b * 2 + 1) * v_stride + vh * v_dim + tid];
    float g0 = fminf(fmaxf(gate[(unsigned long long)(b * 2) * gb_stride + vh], 1e-6f), 1.0f - 1e-6f);
    float bt0 = beta[(unsigned long long)(b * 2) * gb_stride + vh];
    float g1 = fminf(fmaxf(gate[(unsigned long long)(b * 2 + 1) * gb_stride + vh], 1e-6f), 1.0f - 1e-6f);
    float bt1 = beta[(unsigned long long)(b * 2 + 1) * gb_stride + vh];

    // ── Token 0 ──
    // Pass 1: hk_dot from registers
    float hk_a = 0.0f, hk_b = 0.0f, hk_c = 0.0f, hk_d = 0.0f;
    #pragma unroll
    for (int j = 0; j < K_DIM; j += 4) {
        hk_a += H_reg[j]     * sk0[j];
        hk_b += H_reg[j + 1] * sk0[j + 1];
        hk_c += H_reg[j + 2] * sk0[j + 2];
        hk_d += H_reg[j + 3] * sk0[j + 3];
    }
    float v_new_0 = (vi0 - g0 * ((hk_a + hk_b) + (hk_c + hk_d))) * bt0;

    // Pass 2: update H_reg, compute q0_dot and hk1_dot simultaneously
    float qd0a = 0.0f, qd0b = 0.0f, qd0c = 0.0f, qd0d = 0.0f;
    float hk1a = 0.0f, hk1b = 0.0f, hk1c = 0.0f, hk1d = 0.0f;
    #pragma unroll
    for (int j = 0; j < K_DIM; j += 4) {
        float h0 = g0 * H_reg[j]     + sk0[j]     * v_new_0;
        float h1 = g0 * H_reg[j + 1] + sk0[j + 1] * v_new_0;
        float h2 = g0 * H_reg[j + 2] + sk0[j + 2] * v_new_0;
        float h3 = g0 * H_reg[j + 3] + sk0[j + 3] * v_new_0;
        H_reg[j]     = h0;
        H_reg[j + 1] = h1;
        H_reg[j + 2] = h2;
        H_reg[j + 3] = h3;
        qd0a += h0 * sq0[j];     qd0b += h1 * sq0[j + 1];
        qd0c += h2 * sq0[j + 2]; qd0d += h3 * sq0[j + 3];
        hk1a += h0 * sk1[j];     hk1b += h1 * sk1[j + 1];
        hk1c += h2 * sk1[j + 2]; hk1d += h3 * sk1[j + 3];
    }
    float q0_dot = (qd0a + qd0b) + (qd0c + qd0d);
    float v_new_1 = (vi1 - g1 * ((hk1a + hk1b) + (hk1c + hk1d))) * bt1;

    // ── Token 1 ──
    // Update H_reg, compute q1_dot
    float qd1a = 0.0f, qd1b = 0.0f, qd1c = 0.0f, qd1d = 0.0f;
    #pragma unroll
    for (int j = 0; j < K_DIM; j += 4) {
        float h0 = g1 * H_reg[j]     + sk1[j]     * v_new_1;
        float h1 = g1 * H_reg[j + 1] + sk1[j + 1] * v_new_1;
        float h2 = g1 * H_reg[j + 2] + sk1[j + 2] * v_new_1;
        float h3 = g1 * H_reg[j + 3] + sk1[j + 3] * v_new_1;
        H_reg[j]     = h0;
        H_reg[j + 1] = h1;
        H_reg[j + 2] = h2;
        H_reg[j + 3] = h3;
        qd1a += h0 * sq1[j];     qd1b += h1 * sq1[j + 1];
        qd1c += h2 * sq1[j + 2]; qd1d += h3 * sq1[j + 3];
    }
    float q1_dot = (qd1a + qd1b) + (qd1c + qd1d);

    float inv_sqrt_d = rsqrtf((float)k_dim);
    output[((unsigned long long)(b * 2) * num_v_heads + vh) * v_dim + tid] =
        __float2bfloat16(q0_dot * inv_sqrt_d);
    output[((unsigned long long)(b * 2 + 1) * num_v_heads + vh) * v_dim + tid] =
        __float2bfloat16(q1_dot * inv_sqrt_d);

    // Write H from registers → global — ONE write
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_global[j * v_dim + tid] = H_reg[j];
    }
}

// ═══════════════════════════════════════════════════════════════════
// Register-Tiled Chunk3 — MTP K=3 verify path.
//
// Original chunk3 uses TWO intermediate buffers in global memory:
//   Token 0: read H → write Hi0    (128 reads + 128 writes)
//   Token 1: read Hi0 → write Hi1  (128 reads + 128 writes)
//   Token 2: read Hi1 → write H    (128 reads + 128 writes)
//   Total: 384 reads + 384 writes = 768 transactions per thread
//
// Register-tiled: all 3 tokens from H_reg, no intermediate buffers.
//   Total: 128 reads + 128 writes = 256 transactions per thread
//   67% reduction in global memory traffic.
//
// Grid: (num_v_heads, batch, 1)   Block: (128, 1, 1)
// ═══════════════════════════════════════════════════════════════════

extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_chunk3(
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

    float* H_global = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * v_dim);

    // Load k/q for all 3 tokens
    __shared__ float sk0[K_DIM], sq0[K_DIM], sk1[K_DIM], sq1[K_DIM], sk2[K_DIM], sq2[K_DIM];
    {
        unsigned long long qk0 = (unsigned long long)(b * 3) * qk_stride + kh * k_dim;
        unsigned long long qk1 = (unsigned long long)(b * 3 + 1) * qk_stride + kh * k_dim;
        unsigned long long qk2 = (unsigned long long)(b * 3 + 2) * qk_stride + kh * k_dim;
        sk0[tid] = (float)key[qk0 + tid]; sq0[tid] = (float)query[qk0 + tid];
        sk1[tid] = (float)key[qk1 + tid]; sq1[tid] = (float)query[qk1 + tid];
        sk2[tid] = (float)key[qk2 + tid]; sq2[tid] = (float)query[qk2 + tid];
    }
    __syncthreads();

    // Load H column into registers — ONE read
    float H_reg[K_DIM];
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_reg[j] = H_global[j * v_dim + tid];
    }

    float vi0 = (float)value[(unsigned long long)(b * 3) * v_stride + vh * v_dim + tid];
    float vi1 = (float)value[(unsigned long long)(b * 3 + 1) * v_stride + vh * v_dim + tid];
    float vi2 = (float)value[(unsigned long long)(b * 3 + 2) * v_stride + vh * v_dim + tid];
    float g0 = fminf(fmaxf(gate[(unsigned long long)(b * 3) * gb_stride + vh], 1e-6f), 1.0f - 1e-6f);
    float bt0 = beta[(unsigned long long)(b * 3) * gb_stride + vh];
    float g1 = fminf(fmaxf(gate[(unsigned long long)(b * 3 + 1) * gb_stride + vh], 1e-6f), 1.0f - 1e-6f);
    float bt1 = beta[(unsigned long long)(b * 3 + 1) * gb_stride + vh];
    float g2 = fminf(fmaxf(gate[(unsigned long long)(b * 3 + 2) * gb_stride + vh], 1e-6f), 1.0f - 1e-6f);
    float bt2 = beta[(unsigned long long)(b * 3 + 2) * gb_stride + vh];

    // ── Token 0 ──
    float hk_a = 0.0f, hk_b = 0.0f, hk_c = 0.0f, hk_d = 0.0f;
    #pragma unroll
    for (int j = 0; j < K_DIM; j += 4) {
        hk_a += H_reg[j]     * sk0[j];
        hk_b += H_reg[j + 1] * sk0[j + 1];
        hk_c += H_reg[j + 2] * sk0[j + 2];
        hk_d += H_reg[j + 3] * sk0[j + 3];
    }
    float v_new_0 = (vi0 - g0 * ((hk_a + hk_b) + (hk_c + hk_d))) * bt0;

    // Update H_reg, compute q0_dot + hk1_dot
    float qd0a = 0.0f, qd0b = 0.0f, qd0c = 0.0f, qd0d = 0.0f;
    float hk1a = 0.0f, hk1b = 0.0f, hk1c = 0.0f, hk1d = 0.0f;
    #pragma unroll
    for (int j = 0; j < K_DIM; j += 4) {
        float h0 = g0 * H_reg[j]     + sk0[j]     * v_new_0;
        float h1 = g0 * H_reg[j + 1] + sk0[j + 1] * v_new_0;
        float h2 = g0 * H_reg[j + 2] + sk0[j + 2] * v_new_0;
        float h3 = g0 * H_reg[j + 3] + sk0[j + 3] * v_new_0;
        H_reg[j] = h0; H_reg[j+1] = h1; H_reg[j+2] = h2; H_reg[j+3] = h3;
        qd0a += h0 * sq0[j];     qd0b += h1 * sq0[j + 1];
        qd0c += h2 * sq0[j + 2]; qd0d += h3 * sq0[j + 3];
        hk1a += h0 * sk1[j];     hk1b += h1 * sk1[j + 1];
        hk1c += h2 * sk1[j + 2]; hk1d += h3 * sk1[j + 3];
    }
    float q0_dot = (qd0a + qd0b) + (qd0c + qd0d);
    float v_new_1 = (vi1 - g1 * ((hk1a + hk1b) + (hk1c + hk1d))) * bt1;

    // ── Token 1 ──
    // Update H_reg, compute q1_dot + hk2_dot
    float qd1a = 0.0f, qd1b = 0.0f, qd1c = 0.0f, qd1d = 0.0f;
    float hk2a = 0.0f, hk2b = 0.0f, hk2c = 0.0f, hk2d = 0.0f;
    #pragma unroll
    for (int j = 0; j < K_DIM; j += 4) {
        float h0 = g1 * H_reg[j]     + sk1[j]     * v_new_1;
        float h1 = g1 * H_reg[j + 1] + sk1[j + 1] * v_new_1;
        float h2 = g1 * H_reg[j + 2] + sk1[j + 2] * v_new_1;
        float h3 = g1 * H_reg[j + 3] + sk1[j + 3] * v_new_1;
        H_reg[j] = h0; H_reg[j+1] = h1; H_reg[j+2] = h2; H_reg[j+3] = h3;
        qd1a += h0 * sq1[j];     qd1b += h1 * sq1[j + 1];
        qd1c += h2 * sq1[j + 2]; qd1d += h3 * sq1[j + 3];
        hk2a += h0 * sk2[j];     hk2b += h1 * sk2[j + 1];
        hk2c += h2 * sk2[j + 2]; hk2d += h3 * sk2[j + 3];
    }
    float q1_dot = (qd1a + qd1b) + (qd1c + qd1d);
    float v_new_2 = (vi2 - g2 * ((hk2a + hk2b) + (hk2c + hk2d))) * bt2;

    // ── Token 2 ──
    // Update H_reg, compute q2_dot
    float qd2a = 0.0f, qd2b = 0.0f, qd2c = 0.0f, qd2d = 0.0f;
    #pragma unroll
    for (int j = 0; j < K_DIM; j += 4) {
        float h0 = g2 * H_reg[j]     + sk2[j]     * v_new_2;
        float h1 = g2 * H_reg[j + 1] + sk2[j + 1] * v_new_2;
        float h2 = g2 * H_reg[j + 2] + sk2[j + 2] * v_new_2;
        float h3 = g2 * H_reg[j + 3] + sk2[j + 3] * v_new_2;
        H_reg[j] = h0; H_reg[j+1] = h1; H_reg[j+2] = h2; H_reg[j+3] = h3;
        qd2a += h0 * sq2[j];     qd2b += h1 * sq2[j + 1];
        qd2c += h2 * sq2[j + 2]; qd2d += h3 * sq2[j + 3];
    }
    float q2_dot = (qd2a + qd2b) + (qd2c + qd2d);

    float inv_sqrt_d = rsqrtf((float)k_dim);
    output[((unsigned long long)(b * 3) * num_v_heads + vh) * v_dim + tid] =
        __float2bfloat16(q0_dot * inv_sqrt_d);
    output[((unsigned long long)(b * 3 + 1) * num_v_heads + vh) * v_dim + tid] =
        __float2bfloat16(q1_dot * inv_sqrt_d);
    output[((unsigned long long)(b * 3 + 2) * num_v_heads + vh) * v_dim + tid] =
        __float2bfloat16(q2_dot * inv_sqrt_d);

    // Write H from registers → global — ONE write
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_global[j * v_dim + tid] = H_reg[j];
    }
}

