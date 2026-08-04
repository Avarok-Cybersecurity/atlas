// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Spark — Qwen3-VL Vision Encoder CUDA Kernels
//
// All ops use BF16 storage for weights; computations use f32 accumulators.
// Kernels run once per prefill (P ≤ 400 patches), so simplicity > performance.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>

// ── Helpers ──────────────────────────────────────────────────────────────────

__device__ inline float bf16_to_f32(__nv_bfloat16 v) {
    return __bfloat162float(v);
}
__device__ inline __nv_bfloat16 f32_to_bf16(float v) {
    return __float2bfloat16(v);
}

// ── GEMM: C[M,N] = A[M,K] @ B[N,K]^T + bias[N]  (BF16 A, BF16 B, BF16 C) ──
// Grid: (ceil(N/32), ceil(M/32), 1)  Block: (32, 32, 1)
extern "C" __global__
void vision_gemm_bias(
    const __nv_bfloat16* __restrict__ A,   // [M, K]
    const __nv_bfloat16* __restrict__ B,   // [N, K]  (transposed)
    const __nv_bfloat16* __restrict__ bias,// [N]
    __nv_bfloat16* __restrict__ C,         // [M, N]
    unsigned int M, unsigned int N, unsigned int K
) {
    unsigned int row = blockIdx.y * 32 + threadIdx.y;
    unsigned int col = blockIdx.x * 32 + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (unsigned int k = 0; k < K; ++k) {
        acc += bf16_to_f32(A[row * K + k]) * bf16_to_f32(B[col * K + k]);
    }
    acc += bf16_to_f32(bias[col]);
    C[row * N + col] = f32_to_bf16(acc);
}

// ── GEMM: C[M,N] = A[M,K] @ B[K,N] + bias[N]  (B is NOT transposed) ──
// Grid: (ceil(N/32), ceil(M/32), 1)  Block: (32, 32, 1)
extern "C" __global__
void vision_gemm_bias_nn(
    const __nv_bfloat16* __restrict__ A,   // [M, K]
    const __nv_bfloat16* __restrict__ B,   // [K, N]
    const __nv_bfloat16* __restrict__ bias,// [N]
    __nv_bfloat16* __restrict__ C,         // [M, N]
    unsigned int M, unsigned int N, unsigned int K
) {
    unsigned int row = blockIdx.y * 32 + threadIdx.y;
    unsigned int col = blockIdx.x * 32 + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (unsigned int k = 0; k < K; ++k) {
        acc += bf16_to_f32(A[row * K + k]) * bf16_to_f32(B[k * N + col]);
    }
    acc += bf16_to_f32(bias[col]);
    C[row * N + col] = f32_to_bf16(acc);
}

// ── Row-broadcast bias add: C[m,n] += bias[n] (in-place) ──
// Used to fuse the bias for the tensor-core GEMM path (dense_gemm_bf16_pipelined,
// ~40× the naive vision_gemm_bias) which has no built-in bias epilogue. The ViT
// GEMMs (QKV/proj/fc1/fc2 × 27 blocks + merger + patch_embed) dominate image
// prefill (~5s/image on the scalar kernel); pipelined-GEMM + this add is the fix.
extern "C" __global__ void vision_add_bias(
    __nv_bfloat16* __restrict__ C,         // [M, N] in-place
    const __nv_bfloat16* __restrict__ bias,// [N]
    unsigned int M, unsigned int N
) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (unsigned long long)M * N) return;
    unsigned int col = (unsigned int)(idx % N);
    C[idx] = f32_to_bf16(bf16_to_f32(C[idx]) + bf16_to_f32(bias[col]));
}

// ── LayerNorm: x = (x - mean) / sqrt(var + eps) * w + b ──
// One block per row.  Block: (min(D, 1024), 1, 1)
extern "C" __global__
void vision_layer_norm(
    __nv_bfloat16* __restrict__ x,         // [N, D] in-place
    const __nv_bfloat16* __restrict__ w,   // [D]
    const __nv_bfloat16* __restrict__ b,   // [D]
    unsigned int N, unsigned int D,
    float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= N) return;

    __nv_bfloat16* row_ptr = x + row * D;

    // Compute mean.
    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < D; i += blockDim.x) {
        sum += bf16_to_f32(row_ptr[i]);
    }
    // Warp reduce.
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ float smem_sum[32];
    if (threadIdx.x % 32 == 0) smem_sum[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x < (blockDim.x + 31) / 32) sum = smem_sum[threadIdx.x];
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ float mean_val;
    if (threadIdx.x == 0) mean_val = sum / D;
    __syncthreads();

    // Compute variance.
    float var = 0.0f;
    for (unsigned int i = threadIdx.x; i < D; i += blockDim.x) {
        float diff = bf16_to_f32(row_ptr[i]) - mean_val;
        var += diff * diff;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        var += __shfl_down_sync(0xffffffff, var, offset);
    __shared__ float smem_var[32];
    if (threadIdx.x % 32 == 0) smem_var[threadIdx.x / 32] = var;
    __syncthreads();
    if (threadIdx.x < (blockDim.x + 31) / 32) var = smem_var[threadIdx.x];
    for (int offset = 16; offset > 0; offset >>= 1)
        var += __shfl_down_sync(0xffffffff, var, offset);
    __shared__ float inv_std;
    if (threadIdx.x == 0) inv_std = rsqrtf(var / D + eps);
    __syncthreads();

    // Normalize and scale.
    for (unsigned int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = (bf16_to_f32(row_ptr[i]) - mean_val) * inv_std;
        val = val * bf16_to_f32(w[i]) + bf16_to_f32(b[i]);
        row_ptr[i] = f32_to_bf16(val);
    }
}

// ── Add residual in-place: dst[i] += src[i] ──
// Grid: (ceil(n/256), 1, 1)  Block: (256, 1, 1)
extern "C" __global__
void vision_add_inplace(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    unsigned int n
) {
    unsigned int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) dst[i] = f32_to_bf16(bf16_to_f32(dst[i]) + bf16_to_f32(src[i]));
}

// ── Add positional embeddings: x[i*D .. (i+1)*D] += pos[i*D .. (i+1)*D] ──
// Identical to add_inplace but with a separate pos_embed source.
// Reuse vision_add_inplace for this.

// ── GELU activation (in-place): x[i] = x[i] * Φ(x[i]*√2) ──
// Grid: (ceil(n/256), 1, 1)  Block: (256, 1, 1)
#define SQRT2F 1.41421356237f
extern "C" __global__
void vision_gelu(
    __nv_bfloat16* __restrict__ x,
    unsigned int n
) {
    unsigned int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    float v = bf16_to_f32(x[i]);
    // tanh-approximation GELU (matches PyTorch's "gelu_pytorch_tanh", which
    // the Qwen3-VL vision config declares via `hidden_act: "gelu_pytorch_tanh"`).
    // Exact GELU via erf differs by ~1e-4 per value; that error compounds
    // across all 27 ViT blocks and was the tail-end of what was keeping
    // specific image recognition from landing.
    //   GELU_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    const float SQRT_2_OVER_PI = 0.7978845608028654f; // sqrt(2/π)
    const float COEFF = 0.044715f;
    float inner = SQRT_2_OVER_PI * (v + COEFF * v * v * v);
    x[i] = f32_to_bf16(0.5f * v * (1.0f + tanhf(inner)));
}

// ── Scaled dot-product attention + 2D rotary pos emb on Q/K ──
// seq tokens, num_heads heads, head_dim D.
// QKV layout: [seq, 3*H*D] fused.
// rope_cos, rope_sin: [seq, D] precomputed BF16 tables. For each query
// and key token `t` at head dim `d`, the rotary transform is:
//   rotated[d]      = x[d]       * cos[t,d]       - x[d + D/2] * sin[t,d]       (for d < D/2)
//   rotated[d]      = x[d]       * cos[t,d]       + x[d - D/2] * sin[t,d]       (for d ≥ D/2)
// The rope buffers are prebuilt to already reflect the sign pattern of
// the HF `rotate_half` helper, so the kernel only needs to read the
// partner dim (d ± D/2) with the appropriate sign.
// Grid: (seq, num_heads, 1)  Block: (32, 1, 1)
extern "C" __global__
void vision_attention_rope(
    const __nv_bfloat16* __restrict__ QKV, // [seq, 3*H*D]
    __nv_bfloat16* __restrict__ O,          // [seq, H*D]
    const __nv_bfloat16* __restrict__ rope_cos, // [seq, D]
    const __nv_bfloat16* __restrict__ rope_sin, // [seq, D]
    unsigned int seq, unsigned int H, unsigned int D
) {
    unsigned int qi = blockIdx.x;  // query token index
    unsigned int h  = blockIdx.y;  // head index
    if (qi >= seq || h >= H) return;

    unsigned int stride_qkv = 3 * H * D;
    float scale = rsqrtf((float)D);
    unsigned int half_D = D / 2;

    // Pointers into QKV for this head.
    const __nv_bfloat16* Q_row = QKV + qi * stride_qkv + h * D;          // Q

    // Shared memory: first seq floats for scores, then 2*D floats for
    // this query's pre-rotated Q (cached across all seq kj iterations).
    extern __shared__ float smem[]; // [seq + 2*D]
    float* scores  = smem;
    float* q_rope  = smem + seq;              // [D]
    // (no separate q_rot buffer: we can recompute rotation from q_rope
    // on the fly for K below, but Q needs its own since it's fixed).

    // 1. Rotate Q once per query (all key iterations share it). Each
    //    thread handles a strided slice of D; use all 32 lanes.
    for (unsigned int d = threadIdx.x; d < D; d += blockDim.x) {
        float q_val   = bf16_to_f32(Q_row[d]);
        float q_part  = (d < half_D) ? bf16_to_f32(Q_row[d + half_D])
                                     : bf16_to_f32(Q_row[d - half_D]);
        float q_rot   = (d < half_D) ? -q_part : q_part;
        float q_cos_v = bf16_to_f32(rope_cos[qi * D + d]);
        float q_sin_v = bf16_to_f32(rope_sin[qi * D + d]);
        q_rope[d] = q_val * q_cos_v + q_rot * q_sin_v;
    }
    __syncthreads();

    // 2. Attention scores: rotate K on the fly, dot with cached Q.
    for (unsigned int kj = 0; kj < seq; ++kj) {
        const __nv_bfloat16* K_row = QKV + kj * stride_qkv + H * D + h * D;
        float dot = 0.0f;
        for (unsigned int d = threadIdx.x; d < D; d += blockDim.x) {
            float k_val   = bf16_to_f32(K_row[d]);
            float k_part  = (d < half_D) ? bf16_to_f32(K_row[d + half_D])
                                         : bf16_to_f32(K_row[d - half_D]);
            float k_rot   = (d < half_D) ? -k_part : k_part;
            float k_cos_v = bf16_to_f32(rope_cos[kj * D + d]);
            float k_sin_v = bf16_to_f32(rope_sin[kj * D + d]);
            float k_r     = k_val * k_cos_v + k_rot * k_sin_v;
            dot += q_rope[d] * k_r;
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        if (threadIdx.x == 0) scores[kj] = dot * scale;
        __syncthreads();
    }

    // Softmax over scores.
    if (threadIdx.x == 0) {
        float max_s = -1e30f;
        for (unsigned int j = 0; j < seq; ++j) max_s = fmaxf(max_s, scores[j]);
        float sum_exp = 0.0f;
        for (unsigned int j = 0; j < seq; ++j) {
            scores[j] = expf(scores[j] - max_s);
            sum_exp += scores[j];
        }
        float inv_sum = 1.0f / sum_exp;
        for (unsigned int j = 0; j < seq; ++j) scores[j] *= inv_sum;
    }
    __syncthreads();

    // Weighted sum of values.
    for (unsigned int d = threadIdx.x; d < D; d += blockDim.x) {
        float out = 0.0f;
        for (unsigned int vj = 0; vj < seq; ++vj) {
            const __nv_bfloat16* V_row = QKV + vj * stride_qkv + 2 * H * D + h * D;
            out += scores[vj] * bf16_to_f32(V_row[d]);
        }
        O[qi * H * D + h * D + d] = f32_to_bf16(out);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GEMM-based ViT SDPA (fast replacement for vision_attention_rope).
// Pipeline per image: vit_rope_deinterleave → [per head: dense_gemm_bf16_f32out
// QKᵀ → vit_softmax_rows → dense_gemm_bf16_pipelined P·V → vit_scatter_head].
// Semantics IDENTICAL to vision_attention_rope: interleaved QKV strides,
// rotate-half 2D RoPE on Q,K only, scale=rsqrt(D), non-causal max-subtracted
// softmax. The two GEMMs run on BF16 tensor cores; rope/softmax/scatter are
// memory-bound element passes.
// ─────────────────────────────────────────────────────────────────────────────

// (A) Deinterleave QKV[seq,3*H*D] → head-contiguous rotated Qr,Kr [H,seq,D] and
//     TRANSPOSED Vt [H,D,seq]. V is pre-transposed so GEMM2 (which computes
//     A·Bᵀ) yields P·V. One thread per (token,head,d) element.
//     grid = ( ceil(seq*D/256), H, 1 ), block = (256,1,1)
extern "C" __global__
void vit_rope_deinterleave(
    const __nv_bfloat16* __restrict__ QKV,      // [seq, 3*H*D] interleaved
    __nv_bfloat16* __restrict__ Qr,             // [H, seq, D]
    __nv_bfloat16* __restrict__ Kr,             // [H, seq, D]
    __nv_bfloat16* __restrict__ Vt,             // [H, D, seq]  (transposed!)
    const __nv_bfloat16* __restrict__ rope_cos, // [seq, D]
    const __nv_bfloat16* __restrict__ rope_sin, // [seq, D]
    unsigned int seq, unsigned int H, unsigned int D)
{
    unsigned int h   = blockIdx.y;
    unsigned int lin = blockIdx.x * blockDim.x + threadIdx.x; // tok*D + d
    if (h >= H || lin >= seq * D) return;
    unsigned int tok = lin / D;
    unsigned int d   = lin % D;
    unsigned int half_D = D / 2;

    unsigned int stride_qkv = 3u * H * D;
    const __nv_bfloat16* Q_row = QKV + (size_t)tok * stride_qkv + 0u * H * D + h * D;
    const __nv_bfloat16* K_row = QKV + (size_t)tok * stride_qkv + 1u * H * D + h * D;
    const __nv_bfloat16* V_row = QKV + (size_t)tok * stride_qkv + 2u * H * D + h * D;

    float cos_v = bf16_to_f32(rope_cos[(size_t)tok * D + d]);
    float sin_v = bf16_to_f32(rope_sin[(size_t)tok * D + d]);

    // Q rotate-half (matches the legacy kernel's formula exactly)
    float qv    = bf16_to_f32(Q_row[d]);
    float qpart = (d < half_D) ? bf16_to_f32(Q_row[d + half_D])
                               : bf16_to_f32(Q_row[d - half_D]);
    float qrot  = (d < half_D) ? -qpart : qpart;
    float q_r   = qv * cos_v + qrot * sin_v;

    // K rotate-half
    float kv    = bf16_to_f32(K_row[d]);
    float kpart = (d < half_D) ? bf16_to_f32(K_row[d + half_D])
                               : bf16_to_f32(K_row[d - half_D]);
    float krot  = (d < half_D) ? -kpart : kpart;
    float k_r   = kv * cos_v + krot * sin_v;

    // V un-rotated
    float v_v   = bf16_to_f32(V_row[d]);

    size_t hc = (size_t)h * seq * D + (size_t)tok * D + d;   // Qr/Kr [H,seq,D]
    Qr[hc] = f32_to_bf16(q_r);
    Kr[hc] = f32_to_bf16(k_r);
    // Vt [H,D,seq]: index h*D*seq + d*seq + tok  (d,tok swapped = transpose)
    Vt[(size_t)h * D * seq + (size_t)d * seq + (size_t)tok] = f32_to_bf16(v_v);
}

// (B) Row softmax over raw scores S[seq,seq] (f32) → probs P[seq,seq] (bf16).
//     scale = rsqrtf(D) folded in here (GEMM1 produced raw Q·Kᵀ). Parallel
//     block-per-row, 3-pass (max / sumexp / normalize). Replaces the legacy
//     single-thread softmax.
//     grid = (seq,1,1), block = (256,1,1)
extern "C" __global__
void vit_softmax_rows(
    const float* __restrict__ S,        // [seq, seq] f32 raw scores
    __nv_bfloat16* __restrict__ P,      // [seq, seq] bf16 probs
    unsigned int seq, unsigned int D)
{
    unsigned int row = blockIdx.x;
    if (row >= seq) return;
    const float* srow = S + (size_t)row * seq;
    __nv_bfloat16* prow = P + (size_t)row * seq;
    float scale = rsqrtf((float)D);

    __shared__ float red[256 / 32];     // one slot per warp
    unsigned int tid  = threadIdx.x;
    unsigned int lane = tid & 31u, warp = tid >> 5;

    // pass 1: row max
    float m = -1e30f;
    for (unsigned int j = tid; j < seq; j += blockDim.x)
        m = fmaxf(m, srow[j] * scale);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffff, m, o));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    if (tid == 0) {
        float mm = -1e30f;
        for (unsigned int w = 0; w < blockDim.x / 32; ++w) mm = fmaxf(mm, red[w]);
        red[0] = mm;
    }
    __syncthreads();
    float row_max = red[0];

    // pass 2: sum exp
    float s = 0.0f;
    for (unsigned int j = tid; j < seq; j += blockDim.x)
        s += expf(srow[j] * scale - row_max);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    if (lane == 0) red[warp] = s;
    __syncthreads();
    if (tid == 0) {
        float ss = 0.0f;
        for (unsigned int w = 0; w < blockDim.x / 32; ++w) ss += red[w];
        red[0] = (ss > 0.0f) ? (1.0f / ss) : 0.0f;
    }
    __syncthreads();
    float inv_sum = red[0];

    // pass 3: normalize + store bf16
    for (unsigned int j = tid; j < seq; j += blockDim.x)
        prow[j] = f32_to_bf16(expf(srow[j] * scale - row_max) * inv_sum);
}

// (C) Scatter contiguous Oh[seq,D] → interleaved O[seq, dst_stride] head slot.
//     grid = ( ceil(seq*D/256), 1, 1 ), block = (256,1,1)
extern "C" __global__
void vit_scatter_head(
    const __nv_bfloat16* __restrict__ Oh,  // [seq, D] contiguous
    __nv_bfloat16* __restrict__ O,          // [seq, dst_stride], head-slot base
    unsigned int seq, unsigned int D, unsigned int dst_stride)
{
    unsigned int lin = blockIdx.x * blockDim.x + threadIdx.x;
    if (lin >= seq * D) return;
    unsigned int tok = lin / D, d = lin % D;
    O[(size_t)tok * dst_stride + d] = Oh[(size_t)tok * D + d];
}

// ── Spatial merge: reshape [P, D] → [P/S², S²*D] in-place ──
// grid_h × grid_w = P patches; merge_size=2 → groups of 2×2.
// Output shape: [P/(merge_size²), merge_size²*D].
// Grid: (P/(merge_size²), 1, 1)  Block: (merge_size²*D but capped at 1024, 1, 1)
extern "C" __global__
void vision_spatial_merge(
    const __nv_bfloat16* __restrict__ src, // [P, D]
    __nv_bfloat16* __restrict__ dst,        // [P/m², m²*D]
    unsigned int grid_h, unsigned int grid_w,
    unsigned int D, unsigned int merge_size
) {
    unsigned int out_idx = blockIdx.x; // output token index
    unsigned int m  = merge_size;
    unsigned int m2 = m * m;
    unsigned int out_gh = grid_h / m;
    unsigned int out_gw = grid_w / m;
    if (out_idx >= out_gh * out_gw) return;

    unsigned int oh = out_idx / out_gw;
    unsigned int ow = out_idx % out_gw;

    // Gather m×m source patches and concatenate their D features.
    for (unsigned int pi = threadIdx.x; pi < m2 * D; pi += blockDim.x) {
        unsigned int p_local = pi / D;  // which patch in the m×m group
        unsigned int d       = pi % D;  // feature index
        unsigned int ph = oh * m + p_local / m;
        unsigned int pw = ow * m + p_local % m;
        unsigned int src_idx = (ph * grid_w + pw) * D + d;
        dst[out_idx * (m2 * D) + pi] = src[src_idx];
    }
}

// ── Copy f32 → bf16 (for uploading CPU-computed embeddings to GPU) ──
// Grid: (ceil(n/256), 1, 1)  Block: (256, 1, 1)
extern "C" __global__
void vision_f32_to_bf16(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    unsigned int n
) {
    unsigned int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) dst[i] = f32_to_bf16(src[i]);
}

// ── Copy bf16 → bf16 (for pos_embed slice) ──
// Grid: (ceil(n/256), 1, 1)  Block: (256, 1, 1)
extern "C" __global__
void vision_bf16_copy(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    unsigned int n
) {
    unsigned int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
