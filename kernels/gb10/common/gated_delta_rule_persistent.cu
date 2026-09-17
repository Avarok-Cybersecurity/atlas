// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Persistent GDN Prefill — L2-Resident H_State Kernel.
//
// Eliminates LPDDR5X bandwidth bottleneck for h_state during prefill by
// keeping the 64 KB h_state matrix in shared memory for the ENTIRE sequence.
// A small persistent grid (num_v_heads CTAs, 1 per SM max) ensures each CTA
// monopolizes its SM's shared memory and L2 cache partition.
//
// Bandwidth model (per token per layer):
//   Before (global H per token): 2 × 64 KB × 32 heads = 4 MB LPDDR5X R+W
//   After  (H in smem):          Q+K+V+gate+beta reads only ≈ 1.5 KB/token
//
// H_state layout: [k_dim × v_dim] = [128 × 128] FP32 = 64 KB per head.
// GB10 shared memory: 228 KB per SM.  64 KB H + 2 KB double-buffered k/q = 66 KB.
//
// This kernel has IDENTICAL math to gated_delta_rule_prefill (register-tiled).
// The difference is architectural: shared memory H enables larger tile sizes
// and multiple heads per CTA, improving L2 cache residency across the full
// 2 MB h_state (32 heads × 64 KB).
//
// The persistent CTA processes ALL tokens for one head, then moves to the
// next head assigned to it. Between heads, the L2 cache retains Q/K/V data
// for heads sharing the same k_head group (head_repeat = num_v_heads / num_k_heads).
//
// Grid: (num_v_heads, batch, 1)  Block: (128, 1, 1)
// Dynamic shared memory: k_dim * v_dim * sizeof(float) + 4 * k_dim * sizeof(float)
//                       = 128*128*4 + 4*128*4 = 65536 + 2048 = 67584 bytes

#include <cuda_bf16.h>

#define K_DIM 128
#define V_DIM 128

// ═══════════════════════════════════════════════════════════════════
// Persistent prefill: H in shared memory, all tokens in tight loop.
//
// __launch_bounds__(128, 1) pins 1 CTA per SM → full 228 KB smem available.
// 128 threads each handle one v_dim column of H[K_DIM × V_DIM].
//
// Per-token work:
//   1. Load k[128], q[128] into double-buffered smem (from L2 or LPDDR5X)
//   2. Load v[1], gate[1], beta[1] per thread (scalar)
//   3. Pass 1: hk_dot = H_smem^T · k (128 FMA per thread, H in smem)
//   4. Compute: v_new = (v - g * hk_dot) * beta
//   5. Pass 2: Update H_smem, compute q_dot (128 FMA + 128 FMA per thread)
//   6. Write output[1] per thread (BF16, to global/L2)
//
// H_smem stays resident for all seq_len tokens — never written to global
// until the very end. For 128K tokens, this saves 128K × 2 × 64 KB = 16 GB
// of LPDDR5X traffic per head.
// ═══════════════════════════════════════════════════════════════════

extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_prefill_persistent(
    // State (in/out): [batch, num_v_heads, k_dim, v_dim] FP32
    float* __restrict__ h_state,
    // Inputs (BF16, strided):
    const __nv_bfloat16* __restrict__ query,   // token t at: query + t * qk_stride + kh * k_dim
    const __nv_bfloat16* __restrict__ key,     // token t at: key   + t * qk_stride + kh * k_dim
    const __nv_bfloat16* __restrict__ value,   // token t at: value + t * v_stride  + vh * v_dim
    // Scalars per position per head (FP32, strided):
    const float* __restrict__ gate,            // token t at: gate + t * gb_stride + vh
    const float* __restrict__ beta,            // token t at: beta + t * gb_stride + vh
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

    // ── Shared memory layout ──
    // [0 .. K_DIM*V_DIM-1]              : H_smem[K_DIM][V_DIM] = 64 KB
    // [K_DIM*V_DIM .. K_DIM*V_DIM+K_DIM-1]     : smem_k0[K_DIM] (buffer 0 key)
    // [K_DIM*V_DIM+K_DIM .. +2*K_DIM-1]        : smem_q0[K_DIM] (buffer 0 query)
    // [K_DIM*V_DIM+2*K_DIM .. +3*K_DIM-1]      : smem_k1[K_DIM] (buffer 1 key)
    // [K_DIM*V_DIM+3*K_DIM .. +4*K_DIM-1]      : smem_q1[K_DIM] (buffer 1 query)
    extern __shared__ float smem_base[];

    float* H_smem = smem_base;                             // [K_DIM * V_DIM]
    float* smem_k0 = smem_base + K_DIM * V_DIM;           // [K_DIM]
    float* smem_q0 = smem_k0 + K_DIM;                     // [K_DIM]
    float* smem_k1 = smem_q0 + K_DIM;                     // [K_DIM]
    float* smem_q1 = smem_k1 + K_DIM;                     // [K_DIM]

    // ── Load H from global → shared memory ──
    // 128 threads, K_DIM*V_DIM = 16384 elements → 128 iterations per thread
    float* H_global = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);

    #pragma unroll 4
    for (unsigned int i = tid; i < K_DIM * V_DIM; i += 128) {
        H_smem[i] = H_global[i];
    }

    float inv_sqrt_d = rsqrtf((float)k_dim);

    if (seq_len == 0) goto writeback;

    // ── Load first token's k/q into buffer 0 ──
    {
        unsigned long long qk_off = (unsigned long long)0 * qk_stride + kh * k_dim;
        smem_k0[tid] = (float)key[qk_off + tid];
        smem_q0[tid] = (float)query[qk_off + tid];
    }
    __syncthreads();

    // ═══════════════════════════════════════════════════════════════
    // Main token loop — H_smem stays resident for ALL tokens.
    //
    // Double-buffered k/q: while computing token t with cur_k/cur_q,
    // simultaneously load token t+1 into nxt_k/nxt_q.
    //
    // Memory traffic per token:
    //   Read:  k[128] + q[128] + v[1] + gate[1] + beta[1] from L2/LPDDR5X
    //   Write: output[1] to L2/LPDDR5X
    //   H_smem: 128 reads + 128 writes (shared memory, 0 global traffic)
    // ═══════════════════════════════════════════════════════════════

    for (unsigned int t = 0; t < seq_len; t++) {
        // Select current and next buffers (ping-pong)
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

        // Load per-thread scalars from global (small, L2-cached)
        float v_i  = (float)value[(unsigned long long)t * v_stride + vh * v_dim + tid];
        float g_t  = gate[(unsigned long long)t * gb_stride + vh];
        float bt_t = beta[(unsigned long long)t * gb_stride + vh];

        // ── Pass 1: hk_dot = H_smem^T · k ──
        // Each thread reads its column of H_smem[j][tid] for all j in [0, K_DIM).
        // 4 independent accumulators break serial FMA dependency chain.
        float hk_a = 0.0f, hk_b = 0.0f, hk_c = 0.0f, hk_d = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            hk_a += H_smem[(j + 0) * V_DIM + tid] * cur_k[j];
            hk_b += H_smem[(j + 1) * V_DIM + tid] * cur_k[j + 1];
            hk_c += H_smem[(j + 2) * V_DIM + tid] * cur_k[j + 2];
            hk_d += H_smem[(j + 3) * V_DIM + tid] * cur_k[j + 3];
        }
        float hk_dot = (hk_a + hk_b) + (hk_c + hk_d);

        // Gated residual: v_new = (v - g * H^T·k) * beta
        float v_new = (v_i - g_t * hk_dot) * bt_t;

        // ── Pass 2: Update H_smem, compute q_dot = H_new^T · q ──
        // Fused: read H, apply decay + outer product update, write H, accumulate output.
        float qd_a = 0.0f, qd_b = 0.0f, qd_c = 0.0f, qd_d = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            float h0 = g_t * H_smem[(j + 0) * V_DIM + tid] + cur_k[j]     * v_new;
            float h1 = g_t * H_smem[(j + 1) * V_DIM + tid] + cur_k[j + 1] * v_new;
            float h2 = g_t * H_smem[(j + 2) * V_DIM + tid] + cur_k[j + 2] * v_new;
            float h3 = g_t * H_smem[(j + 3) * V_DIM + tid] + cur_k[j + 3] * v_new;
            H_smem[(j + 0) * V_DIM + tid] = h0;
            H_smem[(j + 1) * V_DIM + tid] = h1;
            H_smem[(j + 2) * V_DIM + tid] = h2;
            H_smem[(j + 3) * V_DIM + tid] = h3;
            qd_a += h0 * cur_q[j];
            qd_b += h1 * cur_q[j + 1];
            qd_c += h2 * cur_q[j + 2];
            qd_d += h3 * cur_q[j + 3];
        }
        float q_dot = (qd_a + qd_b) + (qd_c + qd_d);

        // Write output for this token (BF16, goes to L2 → LPDDR5X)
        output[((unsigned long long)(b * seq_len + t) * num_v_heads + vh) * v_dim + tid] =
            __float2bfloat16(q_dot * inv_sqrt_d);

        __syncthreads();  // Ensures next token's k/q are fully loaded
    }

writeback:
    // ── Write H from shared → global memory (once, at end of sequence) ──
    #pragma unroll 4
    for (unsigned int i = tid; i < K_DIM * V_DIM; i += 128) {
        H_global[i] = H_smem[i];
    }
}

// ═══════════════════════════════════════════════════════════════════
// WY4-Persistent: Process 4 tokens per iteration with WY correction.
//
// Same H-in-shared-memory architecture as the single-token kernel, but
// uses the WY (Woodbury-Young) algebraic identity to compute all 4
// H^T @ k[t] dot products in a single pass over H, then applies all
// 4 state updates in a second pass. 2 passes per 4 tokens vs 2 passes
// per 1 token = 4× fewer sequential state multiplications.
//
// This prevents precision drift at long context (28K+) where O(L)
// sequential multiply-adds accumulate floating point error.
//
// Remainder tokens (seq_len % 4) use single-token sequential processing.
//
// Grid: (num_v_heads, batch, 1)  Block: (128, 1, 1)
// Dynamic shared memory: K_DIM*V_DIM*4 + 8*K_DIM*4 + 4*4 = 65536 + 4096 + 16 = 69648
// ═══════════════════════════════════════════════════════════════════

// Warp-level reduction for WY correction k-dot products
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
    return result;
}

extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_prefill_persistent_wy4(
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

    // Shared memory layout:
    // [0 .. K_DIM*V_DIM-1]         : H_smem[K_DIM][V_DIM] = 64 KB
    // [+0 .. +K_DIM-1]             : smem_k0[K_DIM]
    // [+K_DIM .. +2*K_DIM-1]       : smem_q0[K_DIM]
    // [+2*K_DIM .. +3*K_DIM-1]     : smem_k1[K_DIM]
    // [+3*K_DIM .. +4*K_DIM-1]     : smem_q1[K_DIM]
    // [+4*K_DIM .. +5*K_DIM-1]     : smem_k2[K_DIM]
    // [+5*K_DIM .. +6*K_DIM-1]     : smem_q2[K_DIM]
    // [+6*K_DIM .. +7*K_DIM-1]     : smem_k3[K_DIM]
    // [+7*K_DIM .. +8*K_DIM-1]     : smem_q3[K_DIM]
    // [+8*K_DIM .. +8*K_DIM+3]     : smem_warp[4] for reductions
    // [+8*K_DIM+4 .. +8*K_DIM+9]   : kd[6] for WY correction scalars
    extern __shared__ float smem_base[];

    float* H_smem = smem_base;
    float* smem_k0 = smem_base + K_DIM * V_DIM;
    float* smem_q0 = smem_k0 + K_DIM;
    float* smem_k1 = smem_q0 + K_DIM;
    float* smem_q1 = smem_k1 + K_DIM;
    float* smem_k2 = smem_q1 + K_DIM;
    float* smem_q2 = smem_k2 + K_DIM;
    float* smem_k3 = smem_q2 + K_DIM;
    float* smem_q3 = smem_k3 + K_DIM;
    float* smem_warp = smem_q3 + K_DIM;
    // WY correction scalars: kd10, kd20, kd21, kd30, kd31, kd32
    __shared__ float kd10, kd20, kd21, kd30, kd31, kd32;

    // Load H from global → shared memory
    float* H_global = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);
    #pragma unroll 4
    for (unsigned int i = tid; i < K_DIM * V_DIM; i += 128) {
        H_smem[i] = H_global[i];
    }

    float inv_sqrt_d = rsqrtf((float)k_dim);
    unsigned int wy4_end = (seq_len / 4) * 4;
    __syncthreads();

    // ═══════════════════════════════════════════════════════════════
    // WY4 main loop: process 4 tokens per iteration
    // ═══════════════════════════════════════════════════════════════
    for (unsigned int t = 0; t < wy4_end; t += 4) {
        // Load all 4 tokens' k/q into shared memory
        if (tid < K_DIM) {
            unsigned long long off0 = (unsigned long long)(t + 0) * qk_stride + kh * k_dim;
            unsigned long long off1 = (unsigned long long)(t + 1) * qk_stride + kh * k_dim;
            unsigned long long off2 = (unsigned long long)(t + 2) * qk_stride + kh * k_dim;
            unsigned long long off3 = (unsigned long long)(t + 3) * qk_stride + kh * k_dim;
            smem_k0[tid] = (float)key[off0 + tid];   smem_q0[tid] = (float)query[off0 + tid];
            smem_k1[tid] = (float)key[off1 + tid];   smem_q1[tid] = (float)query[off1 + tid];
            smem_k2[tid] = (float)key[off2 + tid];   smem_q2[tid] = (float)query[off2 + tid];
            smem_k3[tid] = (float)key[off3 + tid];   smem_q3[tid] = (float)query[off3 + tid];
        }
        __syncthreads();

        // Load per-thread scalars
        float vi0 = (float)value[(unsigned long long)(t + 0) * v_stride + vh * v_dim + tid];
        float vi1 = (float)value[(unsigned long long)(t + 1) * v_stride + vh * v_dim + tid];
        float vi2 = (float)value[(unsigned long long)(t + 2) * v_stride + vh * v_dim + tid];
        float vi3 = (float)value[(unsigned long long)(t + 3) * v_stride + vh * v_dim + tid];
        float g0 = gate[(unsigned long long)(t + 0) * gb_stride + vh];
        float g1 = gate[(unsigned long long)(t + 1) * gb_stride + vh];
        float g2 = gate[(unsigned long long)(t + 2) * gb_stride + vh];
        float g3 = gate[(unsigned long long)(t + 3) * gb_stride + vh];
        float bt0 = beta[(unsigned long long)(t + 0) * gb_stride + vh];
        float bt1 = beta[(unsigned long long)(t + 1) * gb_stride + vh];
        float bt2 = beta[(unsigned long long)(t + 2) * gb_stride + vh];
        float bt3 = beta[(unsigned long long)(t + 3) * gb_stride + vh];

        // ── Compute k-dot products for WY correction ──
        {
            float p10 = 0.0f, p20 = 0.0f, p21 = 0.0f, p30 = 0.0f, p31 = 0.0f, p32 = 0.0f;
            if (tid < K_DIM) {
                p10 = smem_k1[tid] * smem_k0[tid];
                p20 = smem_k2[tid] * smem_k0[tid];
                p21 = smem_k2[tid] * smem_k1[tid];
                p30 = smem_k3[tid] * smem_k0[tid];
                p31 = smem_k3[tid] * smem_k1[tid];
                p32 = smem_k3[tid] * smem_k2[tid];
            }
            // Block reduction for each k-dot product
            float r;
            r = wy_block_reduce(p10, smem_warp, tid); if (tid == 0) kd10 = r; __syncthreads();
            r = wy_block_reduce(p20, smem_warp, tid); if (tid == 0) kd20 = r; __syncthreads();
            r = wy_block_reduce(p21, smem_warp, tid); if (tid == 0) kd21 = r; __syncthreads();
            r = wy_block_reduce(p30, smem_warp, tid); if (tid == 0) kd30 = r; __syncthreads();
            r = wy_block_reduce(p31, smem_warp, tid); if (tid == 0) kd31 = r; __syncthreads();
            r = wy_block_reduce(p32, smem_warp, tid); if (tid == 0) kd32 = r; __syncthreads();
        }

        // ── Pass 1: Compute all 4 H^T @ k[i] (one pass over H_smem) ──
        float hk0 = 0.0f, hk1 = 0.0f, hk2 = 0.0f, hk3 = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            float h0 = H_smem[(j + 0) * V_DIM + tid];
            float h1 = H_smem[(j + 1) * V_DIM + tid];
            float h2 = H_smem[(j + 2) * V_DIM + tid];
            float h3 = H_smem[(j + 3) * V_DIM + tid];
            hk0 += h0*smem_k0[j] + h1*smem_k0[j+1] + h2*smem_k0[j+2] + h3*smem_k0[j+3];
            hk1 += h0*smem_k1[j] + h1*smem_k1[j+1] + h2*smem_k1[j+2] + h3*smem_k1[j+3];
            hk2 += h0*smem_k2[j] + h1*smem_k2[j+1] + h2*smem_k2[j+2] + h3*smem_k2[j+3];
            hk3 += h0*smem_k3[j] + h1*smem_k3[j+1] + h2*smem_k3[j+2] + h3*smem_k3[j+3];
        }

        // ── WY algebraic correction ──
        // v_new_0 uses original H
        float v_new_0 = (vi0 - g0 * hk0) * bt0;

        // hk1 correction: H_1^T @ k1 = g0 * H^T @ k1 + (k0^T @ k1) * v_new_0
        float hk1_corr = g0 * hk1 + kd10 * v_new_0;
        float v_new_1 = (vi1 - g1 * hk1_corr) * bt1;

        // hk2 correction: H_2^T @ k2 = g0*g1 * H^T @ k2 + g1*(k0^T@k2)*v_new_0 + (k1^T@k2)*v_new_1
        float hk2_corr = g0 * g1 * hk2 + g1 * kd20 * v_new_0 + kd21 * v_new_1;
        float v_new_2 = (vi2 - g2 * hk2_corr) * bt2;

        // hk3 correction: H_3^T @ k3 = g0*g1*g2 * H^T @ k3 + g1*g2*(k0^T@k3)*v_new_0
        //                              + g2*(k1^T@k3)*v_new_1 + (k2^T@k3)*v_new_2
        float hk3_corr = g0*g1*g2 * hk3 + g1*g2 * kd30 * v_new_0
                        + g2 * kd31 * v_new_1 + kd32 * v_new_2;
        float v_new_3 = (vi3 - g3 * hk3_corr) * bt3;

        // ── Pass 2: Apply all 4 state updates + compute outputs (one pass over H_smem) ──
        float qd0 = 0.0f, qd1 = 0.0f, qd2 = 0.0f, qd3 = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            float h0 = H_smem[(j + 0) * V_DIM + tid];
            float h1 = H_smem[(j + 1) * V_DIM + tid];
            float h2 = H_smem[(j + 2) * V_DIM + tid];
            float h3 = H_smem[(j + 3) * V_DIM + tid];

            // Token 0
            h0 = g0*h0 + smem_k0[j]*v_new_0;     h1 = g0*h1 + smem_k0[j+1]*v_new_0;
            h2 = g0*h2 + smem_k0[j+2]*v_new_0;   h3 = g0*h3 + smem_k0[j+3]*v_new_0;
            qd0 += h0*smem_q0[j] + h1*smem_q0[j+1] + h2*smem_q0[j+2] + h3*smem_q0[j+3];

            // Token 1
            h0 = g1*h0 + smem_k1[j]*v_new_1;     h1 = g1*h1 + smem_k1[j+1]*v_new_1;
            h2 = g1*h2 + smem_k1[j+2]*v_new_1;   h3 = g1*h3 + smem_k1[j+3]*v_new_1;
            qd1 += h0*smem_q1[j] + h1*smem_q1[j+1] + h2*smem_q1[j+2] + h3*smem_q1[j+3];

            // Token 2
            h0 = g2*h0 + smem_k2[j]*v_new_2;     h1 = g2*h1 + smem_k2[j+1]*v_new_2;
            h2 = g2*h2 + smem_k2[j+2]*v_new_2;   h3 = g2*h3 + smem_k2[j+3]*v_new_2;
            qd2 += h0*smem_q2[j] + h1*smem_q2[j+1] + h2*smem_q2[j+2] + h3*smem_q2[j+3];

            // Token 3
            h0 = g3*h0 + smem_k3[j]*v_new_3;     h1 = g3*h1 + smem_k3[j+1]*v_new_3;
            h2 = g3*h2 + smem_k3[j+2]*v_new_3;   h3 = g3*h3 + smem_k3[j+3]*v_new_3;
            qd3 += h0*smem_q3[j] + h1*smem_q3[j+1] + h2*smem_q3[j+2] + h3*smem_q3[j+3];

            H_smem[(j + 0) * V_DIM + tid] = h0;
            H_smem[(j + 1) * V_DIM + tid] = h1;
            H_smem[(j + 2) * V_DIM + tid] = h2;
            H_smem[(j + 3) * V_DIM + tid] = h3;
        }

        // Write 4 outputs
        unsigned long long out_base = ((unsigned long long)(b * seq_len) * num_v_heads + vh) * v_dim;
        output[out_base + (unsigned long long)(t + 0) * num_v_heads * v_dim + tid] = __float2bfloat16(qd0 * inv_sqrt_d);
        output[out_base + (unsigned long long)(t + 1) * num_v_heads * v_dim + tid] = __float2bfloat16(qd1 * inv_sqrt_d);
        output[out_base + (unsigned long long)(t + 2) * num_v_heads * v_dim + tid] = __float2bfloat16(qd2 * inv_sqrt_d);
        output[out_base + (unsigned long long)(t + 3) * num_v_heads * v_dim + tid] = __float2bfloat16(qd3 * inv_sqrt_d);

        __syncthreads();
    }

    // ═══════════════════════════════════════════════════════════════
    // Handle remainder tokens (seq_len % 4) with single-token processing
    // ═══════════════════════════════════════════════════════════════
    for (unsigned int t = wy4_end; t < seq_len; t++) {
        if (tid < K_DIM) {
            unsigned long long qk_off = (unsigned long long)t * qk_stride + kh * k_dim;
            smem_k0[tid] = (float)key[qk_off + tid];
            smem_q0[tid] = (float)query[qk_off + tid];
        }
        __syncthreads();

        float v_i  = (float)value[(unsigned long long)t * v_stride + vh * v_dim + tid];
        float g_t  = gate[(unsigned long long)t * gb_stride + vh];
        float bt_t = beta[(unsigned long long)t * gb_stride + vh];

        float hk_a = 0.0f, hk_b = 0.0f, hk_c = 0.0f, hk_d = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            hk_a += H_smem[(j+0)*V_DIM+tid]*smem_k0[j];   hk_b += H_smem[(j+1)*V_DIM+tid]*smem_k0[j+1];
            hk_c += H_smem[(j+2)*V_DIM+tid]*smem_k0[j+2]; hk_d += H_smem[(j+3)*V_DIM+tid]*smem_k0[j+3];
        }
        float hk_dot = (hk_a + hk_b) + (hk_c + hk_d);
        float v_new = (v_i - g_t * hk_dot) * bt_t;

        float qd_a = 0.0f, qd_b = 0.0f, qd_c = 0.0f, qd_d = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            float h0 = g_t*H_smem[(j+0)*V_DIM+tid] + smem_k0[j]*v_new;
            float h1 = g_t*H_smem[(j+1)*V_DIM+tid] + smem_k0[j+1]*v_new;
            float h2 = g_t*H_smem[(j+2)*V_DIM+tid] + smem_k0[j+2]*v_new;
            float h3 = g_t*H_smem[(j+3)*V_DIM+tid] + smem_k0[j+3]*v_new;
            H_smem[(j+0)*V_DIM+tid]=h0; H_smem[(j+1)*V_DIM+tid]=h1;
            H_smem[(j+2)*V_DIM+tid]=h2; H_smem[(j+3)*V_DIM+tid]=h3;
            qd_a += h0*smem_q0[j]; qd_b += h1*smem_q0[j+1];
            qd_c += h2*smem_q0[j+2]; qd_d += h3*smem_q0[j+3];
        }
        float q_dot = (qd_a + qd_b) + (qd_c + qd_d);
        unsigned long long out_base = ((unsigned long long)(b * seq_len) * num_v_heads + vh) * v_dim;
        output[out_base + (unsigned long long)t * num_v_heads * v_dim + tid] = __float2bfloat16(q_dot * inv_sqrt_d);
        __syncthreads();
    }

    // Write H from shared → global memory
    #pragma unroll 4
    for (unsigned int i = tid; i < K_DIM * V_DIM; i += 128) {
        H_global[i] = H_smem[i];
    }
}

// ═══════════════════════════════════════════════════════════════════
// Persistent multi-head prefill: One CTA processes MULTIPLE v_heads.
//
// The key L2 cache advantage: when processing heads that share the same
// k_head (head_repeat=2 for Qwen3-Next), Q and K data is identical.
// By processing both v_heads on the same SM, the Q/K reads for the second
// head hit L2 cache (previously loaded by first head).
//
// For 32 v_heads on 16 SMs: each CTA handles 2 heads.
// H is loaded/stored from shared memory per head, but Q/K remain L2-hot.
//
// Grid: (PERSISTENT_GRID_X, batch, 1) where PERSISTENT_GRID_X = min(num_v_heads, 16)
// Block: (128, 1, 1)
// Dynamic shared memory: same as single-head variant (67584 bytes)
// ═══════════════════════════════════════════════════════════════════

extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_prefill_persistent_multihead(
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
    unsigned int gb_stride,
    // Extra param: number of CTAs in grid dim X (for stride loop)
    unsigned int grid_x
) {
    const unsigned int cta_id = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;

    // ── Shared memory layout (same as single-head) ──
    extern __shared__ float smem_base[];
    float* H_smem = smem_base;
    float* smem_k0 = smem_base + K_DIM * V_DIM;
    float* smem_q0 = smem_k0 + K_DIM;
    float* smem_k1 = smem_q0 + K_DIM;
    float* smem_q1 = smem_k1 + K_DIM;

    float inv_sqrt_d = rsqrtf((float)k_dim);

    // ── Persistent loop over v_heads assigned to this CTA ──
    // Stride by grid_x: CTA 0 handles heads 0, grid_x, 2*grid_x, ...
    // CTA 1 handles heads 1, 1+grid_x, 1+2*grid_x, ...
    for (unsigned int vh = cta_id; vh < num_v_heads; vh += grid_x) {
        const unsigned int kh = vh / head_repeat;

        // ── Load H for this head from global → shared memory ──
        float* H_global = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);

        #pragma unroll 4
        for (unsigned int i = tid; i < K_DIM * V_DIM; i += 128) {
            H_smem[i] = H_global[i];
        }

        if (seq_len == 0) {
            // Nothing to process, H is unchanged, no writeback needed
            continue;
        }

        // Load first token's k/q into buffer 0
        {
            unsigned long long qk_off = (unsigned long long)0 * qk_stride + kh * k_dim;
            smem_k0[tid] = (float)key[qk_off + tid];
            smem_q0[tid] = (float)query[qk_off + tid];
        }
        __syncthreads();

        // ── Token loop: all seq_len tokens with H in shared memory ──
        for (unsigned int t = 0; t < seq_len; t++) {
            float* cur_k = (t & 1) ? smem_k1 : smem_k0;
            float* cur_q = (t & 1) ? smem_q1 : smem_q0;
            float* nxt_k = (t & 1) ? smem_k0 : smem_k1;
            float* nxt_q = (t & 1) ? smem_q0 : smem_q1;

            // Prefetch next token's k/q into alternate buffer
            if (t + 1 < seq_len) {
                unsigned long long qk_off_nxt = (unsigned long long)(t + 1) * qk_stride + kh * k_dim;
                nxt_k[tid] = (float)key[qk_off_nxt + tid];
                nxt_q[tid] = (float)query[qk_off_nxt + tid];
            }

            float v_i  = (float)value[(unsigned long long)t * v_stride + vh * v_dim + tid];
            float g_t  = gate[(unsigned long long)t * gb_stride + vh];
            float bt_t = beta[(unsigned long long)t * gb_stride + vh];

            // Pass 1: hk_dot = H_smem^T · k
            float hk_a = 0.0f, hk_b = 0.0f, hk_c = 0.0f, hk_d = 0.0f;
            #pragma unroll
            for (int j = 0; j < K_DIM; j += 4) {
                hk_a += H_smem[(j + 0) * V_DIM + tid] * cur_k[j];
                hk_b += H_smem[(j + 1) * V_DIM + tid] * cur_k[j + 1];
                hk_c += H_smem[(j + 2) * V_DIM + tid] * cur_k[j + 2];
                hk_d += H_smem[(j + 3) * V_DIM + tid] * cur_k[j + 3];
            }
            float hk_dot = (hk_a + hk_b) + (hk_c + hk_d);

            float v_new = (v_i - g_t * hk_dot) * bt_t;

            // Pass 2: Update H_smem, compute q_dot
            float qd_a = 0.0f, qd_b = 0.0f, qd_c = 0.0f, qd_d = 0.0f;
            #pragma unroll
            for (int j = 0; j < K_DIM; j += 4) {
                float h0 = g_t * H_smem[(j + 0) * V_DIM + tid] + cur_k[j]     * v_new;
                float h1 = g_t * H_smem[(j + 1) * V_DIM + tid] + cur_k[j + 1] * v_new;
                float h2 = g_t * H_smem[(j + 2) * V_DIM + tid] + cur_k[j + 2] * v_new;
                float h3 = g_t * H_smem[(j + 3) * V_DIM + tid] + cur_k[j + 3] * v_new;
                H_smem[(j + 0) * V_DIM + tid] = h0;
                H_smem[(j + 1) * V_DIM + tid] = h1;
                H_smem[(j + 2) * V_DIM + tid] = h2;
                H_smem[(j + 3) * V_DIM + tid] = h3;
                qd_a += h0 * cur_q[j];
                qd_b += h1 * cur_q[j + 1];
                qd_c += h2 * cur_q[j + 2];
                qd_d += h3 * cur_q[j + 3];
            }
            float q_dot = (qd_a + qd_b) + (qd_c + qd_d);

            output[((unsigned long long)(b * seq_len + t) * num_v_heads + vh) * v_dim + tid] =
                __float2bfloat16(q_dot * inv_sqrt_d);

            __syncthreads();
        }

        // ── Write H from shared → global for this head ──
        #pragma unroll 4
        for (unsigned int i = tid; i < K_DIM * V_DIM; i += 128) {
            H_global[i] = H_smem[i];
        }

        // Sync before loading next head's H into same shared memory
        __syncthreads();
    }
}

// ═══════════════════════════════════════════════════════════════════
// Persistent prefill with register-tiled H (hybrid approach).
//
// Combines the best of both worlds:
//   - H_state in REGISTERS (0-cycle access, same as gated_delta_rule_prefill)
//   - Persistent CTA scheduling (fewer CTAs → less L2 thrashing)
//   - Multi-head loop per CTA (L2 reuse of Q/K across head_repeat pairs)
//
// Each CTA processes multiple v_heads sequentially. For each head:
//   1. Load H[128×128] → H_reg[128] per thread (128 global reads)
//   2. Process all seq_len tokens (H in registers, k/q in double-buffered smem)
//   3. Store H_reg → H global (128 global writes)
//
// Between heads sharing the same k_head, the Q/K data for the next head
// is likely still in L2 cache from the previous head's accesses.
//
// Grid: (grid_x, batch, 1)  where grid_x = min(num_v_heads, NUM_SMS)
// Block: (128, 1, 1)
// Dynamic shared memory: 4 * K_DIM * sizeof(float) = 2048 bytes
// ═══════════════════════════════════════════════════════════════════

extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_prefill_persistent_regtile(
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
    unsigned int gb_stride,
    unsigned int grid_x
) {
    const unsigned int cta_id = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;

    // Double-buffered k[128] + q[128] in shared memory (2 KB)
    extern __shared__ float smem[];
    float* smem_k0 = smem;
    float* smem_q0 = smem + K_DIM;
    float* smem_k1 = smem + 2 * K_DIM;
    float* smem_q1 = smem + 3 * K_DIM;

    float inv_sqrt_d = rsqrtf((float)k_dim);

    // ── Persistent loop over v_heads assigned to this CTA ──
    for (unsigned int vh = cta_id; vh < num_v_heads; vh += grid_x) {
        const unsigned int kh = vh / head_repeat;

        float* H_global = h_state +
            ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);

        // Load H column into registers — each thread owns one v_dim column
        float H_reg[K_DIM];
        #pragma unroll
        for (int j = 0; j < K_DIM; j++) {
            H_reg[j] = H_global[j * V_DIM + tid];
        }

        if (seq_len == 0) {
            // H unchanged — no writeback needed, skip to next head
            continue;
        }

        // Load first token's k/q into buffer 0
        {
            unsigned long long qk_off = (unsigned long long)0 * qk_stride + kh * k_dim;
            smem_k0[tid] = (float)key[qk_off + tid];
            smem_q0[tid] = (float)query[qk_off + tid];
        }
        __syncthreads();

        // ── Token loop with H in registers ──
        for (unsigned int t = 0; t < seq_len; t++) {
            float* cur_k = (t & 1) ? smem_k1 : smem_k0;
            float* cur_q = (t & 1) ? smem_q1 : smem_q0;
            float* nxt_k = (t & 1) ? smem_k0 : smem_k1;
            float* nxt_q = (t & 1) ? smem_q0 : smem_q1;

            // Prefetch next token's k/q
            if (t + 1 < seq_len) {
                unsigned long long qk_off_nxt =
                    (unsigned long long)(t + 1) * qk_stride + kh * k_dim;
                nxt_k[tid] = (float)key[qk_off_nxt + tid];
                nxt_q[tid] = (float)query[qk_off_nxt + tid];
            }

            float v_i  = (float)value[(unsigned long long)t * v_stride + vh * v_dim + tid];
            float g_t  = gate[(unsigned long long)t * gb_stride + vh];
            float bt_t = beta[(unsigned long long)t * gb_stride + vh];

            // Pass 1: hk_dot = H_reg^T · k
            float hk_a = 0.0f, hk_b = 0.0f, hk_c = 0.0f, hk_d = 0.0f;
            #pragma unroll
            for (int j = 0; j < K_DIM; j += 4) {
                hk_a += H_reg[j]     * cur_k[j];
                hk_b += H_reg[j + 1] * cur_k[j + 1];
                hk_c += H_reg[j + 2] * cur_k[j + 2];
                hk_d += H_reg[j + 3] * cur_k[j + 3];
            }
            float hk_dot = (hk_a + hk_b) + (hk_c + hk_d);

            float v_new = (v_i - g_t * hk_dot) * bt_t;

            // Pass 2: Update H_reg, compute q_dot
            float qd_a = 0.0f, qd_b = 0.0f, qd_c = 0.0f, qd_d = 0.0f;
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
                qd_a += h0 * cur_q[j];
                qd_b += h1 * cur_q[j + 1];
                qd_c += h2 * cur_q[j + 2];
                qd_d += h3 * cur_q[j + 3];
            }
            float q_dot = (qd_a + qd_b) + (qd_c + qd_d);

            output[((unsigned long long)(b * seq_len + t) * num_v_heads + vh) * v_dim + tid] =
                __float2bfloat16(q_dot * inv_sqrt_d);

            __syncthreads();
        }

        // Store H from registers → global
        #pragma unroll
        for (int j = 0; j < K_DIM; j++) {
            H_global[j * V_DIM + tid] = H_reg[j];
        }

        // Sync before next head iteration (ensures all threads done writing)
        __syncthreads();
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Q12 Phase 2b: same-chunk-len batched variants for persistent + persistent_wy4.
//
// Same constraints as gated_delta_rule_prefill_wy64_batched:
//   - All batched streams share the same seq_len.
//   - Each stream owns its own h_state allocation; h_state_ptrs[b] selects.
//   - QKV/gate/beta/output reads add per-batch offset (b * seq_len * stride).
//
// Single-stream variants above unchanged — purely additive.
// ═══════════════════════════════════════════════════════════════════════

extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_prefill_persistent_batched(
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

    const unsigned long long qk_batch_off = (unsigned long long)b * seq_len * qk_stride;
    const unsigned long long v_batch_off  = (unsigned long long)b * seq_len * v_stride;
    const unsigned long long gb_batch_off = (unsigned long long)b * seq_len * gb_stride;
    const unsigned long long out_batch_off = (unsigned long long)b * seq_len * num_v_heads * v_dim;

    extern __shared__ float smem_base[];

    float* H_smem = smem_base;
    float* smem_k0 = smem_base + K_DIM * V_DIM;
    float* smem_q0 = smem_k0 + K_DIM;
    float* smem_k1 = smem_q0 + K_DIM;
    float* smem_q1 = smem_k1 + K_DIM;

    float* H_global = h_state_ptrs[b] + ((unsigned long long)vh * K_DIM * V_DIM);

    #pragma unroll 4
    for (unsigned int i = tid; i < K_DIM * V_DIM; i += 128) {
        H_smem[i] = H_global[i];
    }

    float inv_sqrt_d = rsqrtf((float)k_dim);

    if (seq_len == 0) goto writeback_persistent_batched;

    {
        unsigned long long qk_off = qk_batch_off + kh * k_dim;
        smem_k0[tid] = (float)key[qk_off + tid];
        smem_q0[tid] = (float)query[qk_off + tid];
    }
    __syncthreads();

    for (unsigned int t = 0; t < seq_len; t++) {
        float* cur_k = (t & 1) ? smem_k1 : smem_k0;
        float* cur_q = (t & 1) ? smem_q1 : smem_q0;
        float* nxt_k = (t & 1) ? smem_k0 : smem_k1;
        float* nxt_q = (t & 1) ? smem_q0 : smem_q1;

        if (t + 1 < seq_len) {
            unsigned long long qk_off_nxt = qk_batch_off + (unsigned long long)(t + 1) * qk_stride + kh * k_dim;
            nxt_k[tid] = (float)key[qk_off_nxt + tid];
            nxt_q[tid] = (float)query[qk_off_nxt + tid];
        }

        float v_i  = (float)value[v_batch_off + (unsigned long long)t * v_stride + vh * v_dim + tid];
        float g_t  = gate[gb_batch_off + (unsigned long long)t * gb_stride + vh];
        float bt_t = beta[gb_batch_off + (unsigned long long)t * gb_stride + vh];

        float hk_a = 0.0f, hk_b = 0.0f, hk_c = 0.0f, hk_d = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            hk_a += H_smem[(j + 0) * V_DIM + tid] * cur_k[j];
            hk_b += H_smem[(j + 1) * V_DIM + tid] * cur_k[j + 1];
            hk_c += H_smem[(j + 2) * V_DIM + tid] * cur_k[j + 2];
            hk_d += H_smem[(j + 3) * V_DIM + tid] * cur_k[j + 3];
        }
        float hk_dot = (hk_a + hk_b) + (hk_c + hk_d);
        float v_new = (v_i - g_t * hk_dot) * bt_t;

        float qd_a = 0.0f, qd_b = 0.0f, qd_c = 0.0f, qd_d = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            float h0 = g_t * H_smem[(j + 0) * V_DIM + tid] + cur_k[j]     * v_new;
            float h1 = g_t * H_smem[(j + 1) * V_DIM + tid] + cur_k[j + 1] * v_new;
            float h2 = g_t * H_smem[(j + 2) * V_DIM + tid] + cur_k[j + 2] * v_new;
            float h3 = g_t * H_smem[(j + 3) * V_DIM + tid] + cur_k[j + 3] * v_new;
            H_smem[(j + 0) * V_DIM + tid] = h0;
            H_smem[(j + 1) * V_DIM + tid] = h1;
            H_smem[(j + 2) * V_DIM + tid] = h2;
            H_smem[(j + 3) * V_DIM + tid] = h3;
            qd_a += h0 * cur_q[j];
            qd_b += h1 * cur_q[j + 1];
            qd_c += h2 * cur_q[j + 2];
            qd_d += h3 * cur_q[j + 3];
        }
        float q_dot = (qd_a + qd_b) + (qd_c + qd_d);

        output[out_batch_off + ((unsigned long long)t * num_v_heads + vh) * v_dim + tid] =
            __float2bfloat16(q_dot * inv_sqrt_d);

        __syncthreads();
    }

writeback_persistent_batched:
    #pragma unroll 4
    for (unsigned int i = tid; i < K_DIM * V_DIM; i += 128) {
        H_global[i] = H_smem[i];
    }
}

extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_prefill_persistent_wy4_batched(
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

    const unsigned long long qk_batch_off = (unsigned long long)b * seq_len * qk_stride;
    const unsigned long long v_batch_off  = (unsigned long long)b * seq_len * v_stride;
    const unsigned long long gb_batch_off = (unsigned long long)b * seq_len * gb_stride;
    const unsigned long long out_batch_off = (unsigned long long)b * seq_len * num_v_heads * v_dim;

    extern __shared__ float smem_base[];

    float* H_smem = smem_base;
    float* smem_k0 = smem_base + K_DIM * V_DIM;
    float* smem_q0 = smem_k0 + K_DIM;
    float* smem_k1 = smem_q0 + K_DIM;
    float* smem_q1 = smem_k1 + K_DIM;
    float* smem_k2 = smem_q1 + K_DIM;
    float* smem_q2 = smem_k2 + K_DIM;
    float* smem_k3 = smem_q2 + K_DIM;
    float* smem_q3 = smem_k3 + K_DIM;
    float* smem_warp = smem_q3 + K_DIM;
    __shared__ float kd10b, kd20b, kd21b, kd30b, kd31b, kd32b;

    float* H_global = h_state_ptrs[b] + ((unsigned long long)vh * K_DIM * V_DIM);
    #pragma unroll 4
    for (unsigned int i = tid; i < K_DIM * V_DIM; i += 128) {
        H_smem[i] = H_global[i];
    }

    float inv_sqrt_d = rsqrtf((float)k_dim);
    unsigned int wy4_end = (seq_len / 4) * 4;
    __syncthreads();

    for (unsigned int t = 0; t < wy4_end; t += 4) {
        if (tid < K_DIM) {
            unsigned long long off0 = qk_batch_off + (unsigned long long)(t + 0) * qk_stride + kh * k_dim;
            unsigned long long off1 = qk_batch_off + (unsigned long long)(t + 1) * qk_stride + kh * k_dim;
            unsigned long long off2 = qk_batch_off + (unsigned long long)(t + 2) * qk_stride + kh * k_dim;
            unsigned long long off3 = qk_batch_off + (unsigned long long)(t + 3) * qk_stride + kh * k_dim;
            smem_k0[tid] = (float)key[off0 + tid];   smem_q0[tid] = (float)query[off0 + tid];
            smem_k1[tid] = (float)key[off1 + tid];   smem_q1[tid] = (float)query[off1 + tid];
            smem_k2[tid] = (float)key[off2 + tid];   smem_q2[tid] = (float)query[off2 + tid];
            smem_k3[tid] = (float)key[off3 + tid];   smem_q3[tid] = (float)query[off3 + tid];
        }
        __syncthreads();

        float vi0 = (float)value[v_batch_off + (unsigned long long)(t + 0) * v_stride + vh * v_dim + tid];
        float vi1 = (float)value[v_batch_off + (unsigned long long)(t + 1) * v_stride + vh * v_dim + tid];
        float vi2 = (float)value[v_batch_off + (unsigned long long)(t + 2) * v_stride + vh * v_dim + tid];
        float vi3 = (float)value[v_batch_off + (unsigned long long)(t + 3) * v_stride + vh * v_dim + tid];
        float g0 = gate[gb_batch_off + (unsigned long long)(t + 0) * gb_stride + vh];
        float g1 = gate[gb_batch_off + (unsigned long long)(t + 1) * gb_stride + vh];
        float g2 = gate[gb_batch_off + (unsigned long long)(t + 2) * gb_stride + vh];
        float g3 = gate[gb_batch_off + (unsigned long long)(t + 3) * gb_stride + vh];
        float bt0 = beta[gb_batch_off + (unsigned long long)(t + 0) * gb_stride + vh];
        float bt1 = beta[gb_batch_off + (unsigned long long)(t + 1) * gb_stride + vh];
        float bt2 = beta[gb_batch_off + (unsigned long long)(t + 2) * gb_stride + vh];
        float bt3 = beta[gb_batch_off + (unsigned long long)(t + 3) * gb_stride + vh];

        {
            float p10 = 0.0f, p20 = 0.0f, p21 = 0.0f, p30 = 0.0f, p31 = 0.0f, p32 = 0.0f;
            if (tid < K_DIM) {
                p10 = smem_k1[tid] * smem_k0[tid];
                p20 = smem_k2[tid] * smem_k0[tid];
                p21 = smem_k2[tid] * smem_k1[tid];
                p30 = smem_k3[tid] * smem_k0[tid];
                p31 = smem_k3[tid] * smem_k1[tid];
                p32 = smem_k3[tid] * smem_k2[tid];
            }
            float r;
            r = wy_block_reduce(p10, smem_warp, tid); if (tid == 0) kd10b = r; __syncthreads();
            r = wy_block_reduce(p20, smem_warp, tid); if (tid == 0) kd20b = r; __syncthreads();
            r = wy_block_reduce(p21, smem_warp, tid); if (tid == 0) kd21b = r; __syncthreads();
            r = wy_block_reduce(p30, smem_warp, tid); if (tid == 0) kd30b = r; __syncthreads();
            r = wy_block_reduce(p31, smem_warp, tid); if (tid == 0) kd31b = r; __syncthreads();
            r = wy_block_reduce(p32, smem_warp, tid); if (tid == 0) kd32b = r; __syncthreads();
        }

        float hk0 = 0.0f, hk1 = 0.0f, hk2 = 0.0f, hk3 = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            float h0 = H_smem[(j + 0) * V_DIM + tid];
            float h1 = H_smem[(j + 1) * V_DIM + tid];
            float h2 = H_smem[(j + 2) * V_DIM + tid];
            float h3 = H_smem[(j + 3) * V_DIM + tid];
            hk0 += h0*smem_k0[j] + h1*smem_k0[j+1] + h2*smem_k0[j+2] + h3*smem_k0[j+3];
            hk1 += h0*smem_k1[j] + h1*smem_k1[j+1] + h2*smem_k1[j+2] + h3*smem_k1[j+3];
            hk2 += h0*smem_k2[j] + h1*smem_k2[j+1] + h2*smem_k2[j+2] + h3*smem_k2[j+3];
            hk3 += h0*smem_k3[j] + h1*smem_k3[j+1] + h2*smem_k3[j+2] + h3*smem_k3[j+3];
        }

        float v_new_0 = (vi0 - g0 * hk0) * bt0;
        float hk1_corr = g0 * hk1 + kd10b * v_new_0;
        float v_new_1 = (vi1 - g1 * hk1_corr) * bt1;
        float hk2_corr = g0 * g1 * hk2 + g1 * kd20b * v_new_0 + kd21b * v_new_1;
        float v_new_2 = (vi2 - g2 * hk2_corr) * bt2;
        float hk3_corr = g0*g1*g2 * hk3 + g1*g2 * kd30b * v_new_0
                        + g2 * kd31b * v_new_1 + kd32b * v_new_2;
        float v_new_3 = (vi3 - g3 * hk3_corr) * bt3;

        float qd0 = 0.0f, qd1 = 0.0f, qd2 = 0.0f, qd3 = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            float h0 = H_smem[(j + 0) * V_DIM + tid];
            float h1 = H_smem[(j + 1) * V_DIM + tid];
            float h2 = H_smem[(j + 2) * V_DIM + tid];
            float h3 = H_smem[(j + 3) * V_DIM + tid];

            h0 = g0*h0 + smem_k0[j]*v_new_0;     h1 = g0*h1 + smem_k0[j+1]*v_new_0;
            h2 = g0*h2 + smem_k0[j+2]*v_new_0;   h3 = g0*h3 + smem_k0[j+3]*v_new_0;
            qd0 += h0*smem_q0[j] + h1*smem_q0[j+1] + h2*smem_q0[j+2] + h3*smem_q0[j+3];

            h0 = g1*h0 + smem_k1[j]*v_new_1;     h1 = g1*h1 + smem_k1[j+1]*v_new_1;
            h2 = g1*h2 + smem_k1[j+2]*v_new_1;   h3 = g1*h3 + smem_k1[j+3]*v_new_1;
            qd1 += h0*smem_q1[j] + h1*smem_q1[j+1] + h2*smem_q1[j+2] + h3*smem_q1[j+3];

            h0 = g2*h0 + smem_k2[j]*v_new_2;     h1 = g2*h1 + smem_k2[j+1]*v_new_2;
            h2 = g2*h2 + smem_k2[j+2]*v_new_2;   h3 = g2*h3 + smem_k2[j+3]*v_new_2;
            qd2 += h0*smem_q2[j] + h1*smem_q2[j+1] + h2*smem_q2[j+2] + h3*smem_q2[j+3];

            h0 = g3*h0 + smem_k3[j]*v_new_3;     h1 = g3*h1 + smem_k3[j+1]*v_new_3;
            h2 = g3*h2 + smem_k3[j+2]*v_new_3;   h3 = g3*h3 + smem_k3[j+3]*v_new_3;
            qd3 += h0*smem_q3[j] + h1*smem_q3[j+1] + h2*smem_q3[j+2] + h3*smem_q3[j+3];

            H_smem[(j + 0) * V_DIM + tid] = h0;
            H_smem[(j + 1) * V_DIM + tid] = h1;
            H_smem[(j + 2) * V_DIM + tid] = h2;
            H_smem[(j + 3) * V_DIM + tid] = h3;
        }

        unsigned long long out_base = out_batch_off + (unsigned long long)vh * v_dim;
        output[out_base + (unsigned long long)(t + 0) * num_v_heads * v_dim + tid] = __float2bfloat16(qd0 * inv_sqrt_d);
        output[out_base + (unsigned long long)(t + 1) * num_v_heads * v_dim + tid] = __float2bfloat16(qd1 * inv_sqrt_d);
        output[out_base + (unsigned long long)(t + 2) * num_v_heads * v_dim + tid] = __float2bfloat16(qd2 * inv_sqrt_d);
        output[out_base + (unsigned long long)(t + 3) * num_v_heads * v_dim + tid] = __float2bfloat16(qd3 * inv_sqrt_d);

        __syncthreads();
    }

    for (unsigned int t = wy4_end; t < seq_len; t++) {
        if (tid < K_DIM) {
            unsigned long long qk_off = qk_batch_off + (unsigned long long)t * qk_stride + kh * k_dim;
            smem_k0[tid] = (float)key[qk_off + tid];
            smem_q0[tid] = (float)query[qk_off + tid];
        }
        __syncthreads();

        float v_i  = (float)value[v_batch_off + (unsigned long long)t * v_stride + vh * v_dim + tid];
        float g_t  = gate[gb_batch_off + (unsigned long long)t * gb_stride + vh];
        float bt_t = beta[gb_batch_off + (unsigned long long)t * gb_stride + vh];

        float hk_a = 0.0f, hk_b = 0.0f, hk_c = 0.0f, hk_d = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            hk_a += H_smem[(j+0)*V_DIM+tid]*smem_k0[j];   hk_b += H_smem[(j+1)*V_DIM+tid]*smem_k0[j+1];
            hk_c += H_smem[(j+2)*V_DIM+tid]*smem_k0[j+2]; hk_d += H_smem[(j+3)*V_DIM+tid]*smem_k0[j+3];
        }
        float hk_dot = (hk_a + hk_b) + (hk_c + hk_d);
        float v_new = (v_i - g_t * hk_dot) * bt_t;

        float qd_a = 0.0f, qd_b = 0.0f, qd_c = 0.0f, qd_d = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            float h0 = g_t*H_smem[(j+0)*V_DIM+tid] + smem_k0[j]*v_new;
            float h1 = g_t*H_smem[(j+1)*V_DIM+tid] + smem_k0[j+1]*v_new;
            float h2 = g_t*H_smem[(j+2)*V_DIM+tid] + smem_k0[j+2]*v_new;
            float h3 = g_t*H_smem[(j+3)*V_DIM+tid] + smem_k0[j+3]*v_new;
            H_smem[(j+0)*V_DIM+tid]=h0; H_smem[(j+1)*V_DIM+tid]=h1;
            H_smem[(j+2)*V_DIM+tid]=h2; H_smem[(j+3)*V_DIM+tid]=h3;
            qd_a += h0*smem_q0[j]; qd_b += h1*smem_q0[j+1];
            qd_c += h2*smem_q0[j+2]; qd_d += h3*smem_q0[j+3];
        }
        float q_dot = (qd_a + qd_b) + (qd_c + qd_d);
        unsigned long long out_base = out_batch_off + (unsigned long long)vh * v_dim;
        output[out_base + (unsigned long long)t * num_v_heads * v_dim + tid] = __float2bfloat16(q_dot * inv_sqrt_d);
        __syncthreads();
    }

    #pragma unroll 4
    for (unsigned int i = tid; i < K_DIM * V_DIM; i += 128) {
        H_global[i] = H_smem[i];
    }
}
