// SPDX-License-Identifier: AGPL-3.0-only

// Atlas SSM Preprocessing Kernels for Qwen3-Next linear attention.
//
// 1. deinterleave_qkvz: Scatter interleaved QKVZ projection output
//    from [num_groups × group_dim] to sequential [Q | K | V | Z].
//
// 2. compute_gdn_gates: Deinterleave BA projection and compute GDN
//    gate/beta values from A_log, dt_bias parameters.
//
// Qwen3-Next dimensions (80B):
//   num_k_heads=16, head_k_dim=128, num_v_heads=32, head_v_dim=128
//   num_groups = num_k_heads = 16
//   vheads_per_group = num_v_heads / num_k_heads = 2
//   group_dim = 2*128 + 2*2*128 = 768
//   total QKVZ = 16*768 = 12288

#include <cuda_bf16.h>

// ============================================================
// Deinterleave QKVZ projection output
// ============================================================
// Input layout (interleaved by GQA group):
//   Group 0: [Q0_128 | K0_128 | V0_V1_256 | Z0_Z1_256] = 768
//   Group 1: [Q1_128 | K1_128 | V2_V3_256 | Z2_Z3_256] = 768
//   ...
//   Group 15: [Q15_128 | K15_128 | V30_V31_256 | Z30_Z31_256] = 768
//
// Output layout (sequential):
//   [Q_2048 | K_2048 | V_4096 | Z_4096] = 12288
//
// Grid: (num_tokens, ceil(total/256), 1)  Block: (256, 1, 1)
// blockIdx.x = token index, blockIdx.y = element tile
extern "C" __global__ void deinterleave_qkvz(
    const __nv_bfloat16* __restrict__ interleaved,  // [num_tokens, num_groups * group_dim]
    __nv_bfloat16* __restrict__ output,              // [num_tokens, Q | K | V | Z] sequential
    unsigned int num_groups,        // 16
    unsigned int head_k_dim,        // 128
    unsigned int vheads_per_group,  // 2
    unsigned int head_v_dim         // 128
) {
    unsigned int token = blockIdx.x;
    unsigned int tid = blockIdx.y * blockDim.x + threadIdx.x;

    unsigned int v_group_size = vheads_per_group * head_v_dim;
    unsigned int group_dim = 2 * head_k_dim + 2 * v_group_size;
    unsigned int total = num_groups * group_dim;
    if (tid >= total) return;

    // Per-token offsets
    const __nv_bfloat16* in_tok = interleaved + (unsigned long long)token * total;
    __nv_bfloat16* out_tok = output + (unsigned long long)token * total;

    unsigned int g = tid / group_dim;
    unsigned int idx = tid % group_dim;

    // Cumulative offsets in output
    unsigned int q_total = num_groups * head_k_dim;
    unsigned int k_total = num_groups * head_k_dim;
    unsigned int v_total = num_groups * v_group_size;

    unsigned int out_idx;
    if (idx < head_k_dim) {
        out_idx = g * head_k_dim + idx;
    } else if (idx < 2 * head_k_dim) {
        out_idx = q_total + g * head_k_dim + (idx - head_k_dim);
    } else if (idx < 2 * head_k_dim + v_group_size) {
        out_idx = q_total + k_total + g * v_group_size + (idx - 2 * head_k_dim);
    } else {
        out_idx = q_total + k_total + v_total + g * v_group_size
                + (idx - 2 * head_k_dim - v_group_size);
    }

    out_tok[out_idx] = in_tok[tid];
}

// ============================================================
// Deinterleave Q/Gate from per-head interleaved to contiguous layout
// ============================================================
// HF q_proj output: [Q_h0(hd), G_h0(hd), Q_h1(hd), G_h1(hd), ...]
// Required layout:  [Q_h0(hd), Q_h1(hd), ..., G_h0(hd), G_h1(hd), ...]
//
// In-place via shared memory. Total data = num_heads * head_dim * 2.
// For Qwen3-Next: 16 * 256 * 2 = 8192 BF16 = 16KB shared memory.
//
// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1)
// Dynamic shared memory: num_heads * head_dim * 2 * sizeof(bf16) bytes
// blockIdx.x = token index
extern "C" __global__ void deinterleave_qg(
    __nv_bfloat16* __restrict__ data,  // [num_tokens, num_heads * head_dim * 2] in-place
    unsigned int num_heads,             // 16
    unsigned int head_dim,              // 256
    unsigned int stride                 // elements between tokens (num_heads * head_dim * 2)
) {
    extern __shared__ __nv_bfloat16 smem[];

    unsigned int total = num_heads * head_dim * 2;
    unsigned int tid = threadIdx.x;
    __nv_bfloat16* tok_data = data + (unsigned long long)blockIdx.x * stride;

    // Load all data to shared memory (coalesced)
    for (unsigned int i = tid; i < total; i += blockDim.x) {
        smem[i] = tok_data[i];
    }
    __syncthreads();

    // Write back in deinterleaved order
    unsigned int group_dim = 2 * head_dim;
    unsigned int q_total = num_heads * head_dim;

    for (unsigned int i = tid; i < total; i += blockDim.x) {
        unsigned int src;
        if (i < q_total) {
            unsigned int h = i / head_dim;
            unsigned int d = i % head_dim;
            src = h * group_dim + d;
        } else {
            unsigned int gi = i - q_total;
            unsigned int h = gi / head_dim;
            unsigned int d = gi % head_dim;
            src = h * group_dim + head_dim + d;
        }
        tok_data[i] = smem[src];
    }
}

// ============================================================
// Deinterleave Q/Gate with split output (eliminates per-token copy loop)
// ============================================================
// Same deinterleave as deinterleave_qg, but writes Q to a separate contiguous
// output buffer instead of in-place. Gate is written back to data in-place.
// This eliminates the per-token D2D copy loop that previously extracted Q.
//
// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1)
// Dynamic shared memory: num_heads * head_dim * 2 * sizeof(bf16) bytes
extern "C" __global__ void deinterleave_qg_split(
    __nv_bfloat16* __restrict__ data,      // [num_tokens, stride] in-place (gate written back)
    __nv_bfloat16* __restrict__ q_out,     // [num_tokens, num_heads * head_dim] contiguous Q
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int stride                     // elements between tokens in data
) {
    extern __shared__ __nv_bfloat16 smem[];

    unsigned int total = num_heads * head_dim * 2;
    unsigned int tid = threadIdx.x;
    unsigned int q_total = num_heads * head_dim;
    __nv_bfloat16* tok_data = data + (unsigned long long)blockIdx.x * stride;
    __nv_bfloat16* tok_q = q_out + (unsigned long long)blockIdx.x * q_total;

    // Load all data to shared memory (coalesced from data)
    for (unsigned int i = tid; i < total; i += blockDim.x) {
        smem[i] = tok_data[i];
    }
    __syncthreads();

    unsigned int group_dim = 2 * head_dim;

    // Write Q to separate contiguous buffer, Gate back to data in-place
    for (unsigned int i = tid; i < total; i += blockDim.x) {
        unsigned int src;
        if (i < q_total) {
            unsigned int h = i / head_dim;
            unsigned int d = i % head_dim;
            src = h * group_dim + d;
            tok_q[i] = smem[src];
        } else {
            unsigned int gi = i - q_total;
            unsigned int h = gi / head_dim;
            unsigned int d = gi % head_dim;
            src = h * group_dim + head_dim + d;
            tok_data[i] = smem[src];
        }
    }
}

// ============================================================
// Fused deinterleave Q/Gate + per-head Q RMS norm
// ============================================================
// Eliminates one global memory round-trip for Q (write then read-back
// for norm). Gate is deinterleaved and written to data in-place,
// Q is deinterleaved → normalized → written to q_out in a single pass.
//
// Warp mapping: one warp per Q head (256 threads / 32 = 8 warps).
// For models with more heads than warps, each warp loops over heads.
//
// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1)
// Dynamic shared memory: num_heads * head_dim * 2 * sizeof(bf16) bytes
extern "C" __global__ void deinterleave_qg_split_qnorm(
    __nv_bfloat16* __restrict__ data,              // [num_tokens, stride] in-place (gate written back)
    __nv_bfloat16* __restrict__ q_out,             // [num_tokens, num_heads * head_dim] normalized Q output
    const __nv_bfloat16* __restrict__ q_norm_weight, // [num_heads * head_dim] RMS norm weights
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int stride,                            // elements between tokens in data
    float eps
) {
    extern __shared__ __nv_bfloat16 smem[];

    unsigned int total = num_heads * head_dim * 2;  // Q + Gate interleaved
    unsigned int tid = threadIdx.x;
    unsigned int q_total = num_heads * head_dim;
    unsigned int group_dim = 2 * head_dim;
    __nv_bfloat16* tok_data = data + (unsigned long long)blockIdx.x * stride;
    __nv_bfloat16* tok_q = q_out + (unsigned long long)blockIdx.x * q_total;

    // Step 1: Load all interleaved data to shared memory (coalesced)
    for (unsigned int i = tid; i < total; i += blockDim.x) {
        smem[i] = tok_data[i];
    }
    __syncthreads();

    // Step 2: Write Gate to data at offset q_total (same layout as deinterleave_qg_split)
    for (unsigned int i = tid; i < q_total; i += blockDim.x) {
        unsigned int h = i / head_dim;
        unsigned int d = i % head_dim;
        unsigned int src = h * group_dim + head_dim + d;
        tok_data[q_total + i] = smem[src];
    }

    // Step 3: Per-head Q RMS norm using warp-level reduction
    // One warp per head: 32 threads handle head_dim elements.
    // For head_dim=256: 256/32 = 8 elements per thread.
    unsigned int warp_id = tid / 32;
    unsigned int lane = tid % 32;
    unsigned int num_warps = blockDim.x / 32;
    unsigned int elems_per_thread = head_dim / 32;  // assumes head_dim % 32 == 0

    for (unsigned int head = warp_id; head < num_heads; head += num_warps) {
        // Read Q from shared memory and compute sum of squares.
        // Use stride-32 element assignment to avoid shared memory bank conflicts:
        // lane L handles dims: L, L+32, L+64, ..., L+(elems_per_thread-1)*32
        float sum_sq = 0.0f;
        for (unsigned int e = 0; e < elems_per_thread; e++) {
            unsigned int dim = lane + e * 32;
            unsigned int src_idx = head * group_dim + dim;  // Q at even position in interleaved
            float val = __bfloat162float(smem[src_idx]);
            sum_sq += val * val;
        }

        // Warp-level reduction (no shared memory needed)
        sum_sq = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 16) + sum_sq;
        sum_sq = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 8) + sum_sq;
        sum_sq = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 4) + sum_sq;
        sum_sq = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 2) + sum_sq;
        sum_sq = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 1) + sum_sq;

        // Compute normalization factor: rsqrt(mean(x^2) + eps)
        float rms = rsqrtf(sum_sq / (float)head_dim + eps);

        // Apply norm with offset-from-1 weight: out = x * rms * (1 + weight)
        // Note: q_norm_weight is [head_dim], shared across all heads (same weight per head).
        for (unsigned int e = 0; e < elems_per_thread; e++) {
            unsigned int dim = lane + e * 32;
            unsigned int src_idx = head * group_dim + dim;
            unsigned int out_idx = head * head_dim + dim;
            float val = __bfloat162float(smem[src_idx]);
            float w = __bfloat162float(q_norm_weight[dim]);  // weight is [head_dim], not [num_heads * head_dim]
            tok_q[out_idx] = __float2bfloat16(val * rms * (1.0f + w));
        }
    }
}

// ============================================================
// Fused BA projection + GDN gates (eliminates intermediate buffer)
// ============================================================
// Dense BF16 GEMV (N=64, K=2048) with inline gate/beta transforms.
// Replaces separate dense_gemv + compute_gdn_gates kernels.
//
// BA output layout (interleaved, 16 groups of 4):
//   Group g: [B_{g*vpg+0}, B_{g*vpg+1}, A_{g*vpg+0}, A_{g*vpg+1}]
//   N = num_groups * (2 * vpg) = 16 * 4 = 64
//
// After GEMV reduction, element n maps to:
//   within_group = n % (2*vpg)
//   If within_group < vpg: beta element → sigmoid(result)
//   Else: alpha element → exp(-exp(A_log) * softplus(result + dt_bias))
//
// Grid: (ceil(N / 4), 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void dense_gemv_ba_gates(
    const __nv_bfloat16* __restrict__ A,        // [1, K] input
    const __nv_bfloat16* __restrict__ B,        // [N, K] BA weight
    const float* __restrict__ A_log,    // [num_v_heads]
    const float* __restrict__ dt_bias,  // [num_v_heads]
    float* __restrict__ gate_out,                // [num_v_heads] FP32 decay
    float* __restrict__ beta_out,                // [num_v_heads] FP32 sigmoid
    unsigned int N,                              // 64
    unsigned int K,                              // 2048
    unsigned int vheads_per_group                // 2
) {
    const unsigned int threads_per_out = 256 / 4;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * 4 + local_out;
    if (n >= N) return;

    // Vectorized K-reduction: 8 BF16 per uint4 load
    float acc = 0.0f;
    const unsigned int K_VEC = K / 8;
    const uint4* A_vec = (const uint4*)A;
    const uint4* B_vec = (const uint4*)(B + (unsigned long long)n * K);

    for (unsigned int kv = lane; kv < K_VEC; kv += threads_per_out) {
        uint4 a_data = A_vec[kv];
        uint4 b_data = B_vec[kv];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            __nv_bfloat16 a_lo, a_hi, b_lo, b_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[i] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[i] >> 16);
            *(unsigned short*)&b_lo = (unsigned short)(b_raw[i] & 0xFFFF);
            *(unsigned short*)&b_hi = (unsigned short)(b_raw[i] >> 16);
            acc += __bfloat162float(a_lo) * __bfloat162float(b_lo);
            acc += __bfloat162float(a_hi) * __bfloat162float(b_hi);
        }
    }

    // Warp shuffle reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    // Cross-warp smem reduction (threads_per_out=64 = 2 warps)
    __shared__ float smem[4 * 2];
    const unsigned int warp_lane = threadIdx.x % 32;
    if (warp_lane == 0) {
        smem[local_out * 2 + (lane / 32)] = acc;
    }
    __syncthreads();

    // Apply gate/beta transforms and write directly to output
    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        unsigned int group_dim_ba = 2 * vheads_per_group;
        unsigned int within_group = n % group_dim_ba;
        unsigned int group = n / group_dim_ba;

        if (within_group < vheads_per_group) {
            // Beta element: sigmoid(b_raw)
            unsigned int vh = group * vheads_per_group + within_group;
            beta_out[vh] = 1.0f / (1.0f + __expf(-result));
        } else {
            // Alpha element: exp(-exp(A_log) * softplus(alpha + dt_bias))
            unsigned int vh = group * vheads_per_group + (within_group - vheads_per_group);
            float a_log_val = A_log[vh];
            float dt_b = dt_bias[vh];
            float A_val = __expf(fminf(a_log_val, 20.0f));
            float dt = __logf(1.0f + __expf(fminf(result + dt_b, 20.0f)));
            gate_out[vh] = __expf(-A_val * dt);
        }
    }
}

// ============================================================
// Fused BA GEMM + GDN gates for prefill (token-parallel)
// ============================================================
// Replaces dense_gemm_bf16([M,K]×[N,K]) + compute_gdn_gates for prefill.
// Directly computes BA dot products for each token and immediately applies
// the sigmoid/exp gate transforms, skipping the intermediate ba_out buffer.
//
// The naive dense_gemm_bf16 uses 16×16 scalar tiles with no vectorization.
// This kernel uses uint4 (8×BF16) vectorized loads and warp shuffle reduction,
// matching the decode-path dense_gemv_ba_gates but with token parallelism.
//
// Output layout: [gate(nv), beta(nv)] per token, interleaved, stride gate_stride:
//   gate_out[token * gate_stride + vh]      = gate (alpha→exp transform)
//   gate_out[token * gate_stride + nv + vh] = beta (sigmoid)
//
// Grid: (ceil(N/4), M_tokens, 1)  Block: (256, 1, 1)
// blockIdx.x = which group of 4 BA outputs (0..N/4-1)
// blockIdx.y = token index (0..M-1)
extern "C" __global__ void dense_gemm_ba_gates_prefill(
    const __nv_bfloat16* __restrict__ A,         // [M, K_stride] activations
    const __nv_bfloat16* __restrict__ B,         // [N, K] BA weight (row-major)
    const float* __restrict__ A_log,     // [nv] learned A_log parameter
    const float* __restrict__ dt_bias,   // [nv] learned dt_bias parameter
    float* __restrict__ gate_out,                 // [M, gate_stride] FP32 output
    unsigned int M,               // num_tokens
    unsigned int N,               // 64 = ssm_ba_size
    unsigned int K,               // 2048 = hidden_size
    unsigned int K_stride,        // BF16 elements per token in A (= K for dense)
    unsigned int gate_stride,     // FP32 elements per token in gate_out (= 2*nv)
    unsigned int nv,              // num_v_heads (32)
    unsigned int vheads_per_group // 2
) {
    const unsigned int threads_per_out = 256 / 4;  // 64 (2 warps per output)
    const unsigned int local_out = threadIdx.x / threads_per_out;  // 0..3
    const unsigned int lane = threadIdx.x % threads_per_out;        // 0..63

    const unsigned int token = blockIdx.y;
    const unsigned int n = blockIdx.x * 4 + local_out;
    if (n >= N || token >= M) return;

    // Vectorized K-reduction: 8 BF16 per uint4 load
    float acc = 0.0f;
    const unsigned int K_VEC = K / 8;
    const uint4* A_vec = (const uint4*)(A + (unsigned long long)token * K_stride);
    const uint4* B_vec = (const uint4*)(B + (unsigned long long)n * K);

    for (unsigned int kv = lane; kv < K_VEC; kv += threads_per_out) {
        uint4 a_data = A_vec[kv];
        uint4 b_data = B_vec[kv];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            __nv_bfloat16 a_lo, a_hi, b_lo, b_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[i] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[i] >> 16);
            *(unsigned short*)&b_lo = (unsigned short)(b_raw[i] & 0xFFFF);
            *(unsigned short*)&b_hi = (unsigned short)(b_raw[i] >> 16);
            acc += __bfloat162float(a_lo) * __bfloat162float(b_lo);
            acc += __bfloat162float(a_hi) * __bfloat162float(b_hi);
        }
    }

    // Warp shuffle reduction (within 32-thread warp)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    // Cross-warp smem reduction (threads_per_out=64 = 2 warps per output)
    __shared__ float smem[4 * 2];
    const unsigned int warp_lane = threadIdx.x % 32;
    if (warp_lane == 0) {
        smem[local_out * 2 + (lane / 32)] = acc;
    }
    __syncthreads();

    // Apply gate/beta transforms and write to interleaved output
    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        unsigned int group_dim_ba = 2 * vheads_per_group;
        unsigned int within_group = n % group_dim_ba;
        unsigned int group = n / group_dim_ba;

        float* gate_tok = gate_out + (unsigned long long)token * gate_stride;

        if (within_group < vheads_per_group) {
            // Beta element: sigmoid(b_raw) → stored at offset nv
            unsigned int vh = group * vheads_per_group + within_group;
            gate_tok[nv + vh] = 1.0f / (1.0f + __expf(-result));
        } else {
            // Alpha (gate) element: exp(-exp(A_log) * softplus(alpha + dt_bias))
            unsigned int vh = group * vheads_per_group + (within_group - vheads_per_group);
            float a_log_val = A_log[vh];
            float dt_b = dt_bias[vh];
            float A_val = __expf(fminf(a_log_val, 20.0f));
            float dt = __logf(1.0f + __expf(fminf(result + dt_b, 20.0f)));
            gate_tok[vh] = __expf(-A_val * dt);
        }
    }
}

// ============================================================
// Compute GDN gates from interleaved BA + learned parameters
// ============================================================
// Reads BA in interleaved format (16 groups of [B_2, A_2]),
// deinterleaves to per-head beta and alpha, then computes:
//   gate = exp(-exp(A_log) * softplus(alpha + dt_bias))
//   beta = sigmoid(beta_raw)
//
// Grid: (num_tokens, 1, 1)  Block: (num_v_heads, 1, 1)
// blockIdx.x = token index. (num_v_heads=32 fits in a single warp)
extern "C" __global__ void compute_gdn_gates(
    const __nv_bfloat16* __restrict__ ba_interleaved, // [num_tokens, num_groups * group_dim_ba]
    const float* __restrict__ A_log,          // [num_v_heads]
    const float* __restrict__ dt_bias,        // [num_v_heads]
    float* __restrict__ gate_out,                      // [num_tokens, num_v_heads] decay factor
    float* __restrict__ beta_out,                      // [num_tokens, num_v_heads] write gate
    unsigned int num_v_heads,      // 32
    unsigned int num_groups,       // 16
    unsigned int vheads_per_group, // 2
    unsigned int ba_stride         // elements between tokens in ba_interleaved
) {
    unsigned int token = blockIdx.x;
    unsigned int vh = threadIdx.x;
    if (vh >= num_v_heads) return;

    unsigned int group = vh / vheads_per_group;
    unsigned int local_idx = vh % vheads_per_group;
    unsigned int group_dim_ba = 2 * vheads_per_group;

    // Per-token offsets. Gate/beta are interleaved: [gate(nv), beta(nv)] per token,
    // so output stride is 2*num_v_heads floats per token.
    unsigned int out_stride = 2 * num_v_heads;
    const __nv_bfloat16* ba_tok = ba_interleaved + (unsigned long long)token * ba_stride;
    float* gate_tok = gate_out + (unsigned long long)token * out_stride;
    float* beta_tok = beta_out + (unsigned long long)token * out_stride;

    // BA layout per group: [B_0, B_1, A_0, A_1]
    float b_raw = (float)ba_tok[group * group_dim_ba + local_idx];
    float a_raw = (float)ba_tok[group * group_dim_ba + vheads_per_group + local_idx];

    float a_log_val = A_log[vh];
    float dt_b = dt_bias[vh];

    // Gate: g = -exp(A_log) * softplus(a + dt_bias), then exp(g)
    float A_val = __expf(fminf(a_log_val, 20.0f));
    float dt = __logf(1.0f + __expf(fminf(a_raw + dt_b, 20.0f)));  // softplus
    float g = -A_val * dt;
    gate_tok[vh] = __expf(g);  // multiplicative decay in (0, 1)

    // Beta: sigmoid(b)
    beta_tok[vh] = 1.0f / (1.0f + __expf(-b_raw));
}
