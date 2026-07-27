// SPDX-License-Identifier: AGPL-3.0-only

// Atlas RMS Normalization kernel for Qwen3-VL (SM121).
//
// Qwen3-VL uses STANDARD RMS normalization (NOT offset-from-1):
//   RMSNorm(x) = x * weight / sqrt(mean(x^2) + eps)
//
// This differs from Qwen3-Next which uses (1 + weight) offset scaling.
//
// Input/output: BF16, computation in FP32.
// Vectorized: 2 BF16 elements per 32-bit load/store.

#include <cuda_bf16.h>

// Unpack a 32-bit word containing 2 packed BF16 values into 2 floats.
__device__ __forceinline__ void unpack_bf16x2(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

// Pack 2 floats into a 32-bit word of 2 BF16 values.
__device__ __forceinline__ unsigned int pack_bf16x2(float v0, float v1) {
    unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
    unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
    return lo | (hi << 16);
}

// Warp-level reduction using shuffle
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// RMS Normalization: out = x * weight / sqrt(mean(x^2) + eps)
//
// Standard formulation (no offset). Used by Qwen3-VL.
//
// Grid: (num_tokens, 1, 1)
// Block: (min(hidden_size, 1024), 1, 1)
extern "C" __global__ void rms_norm(
    const __nv_bfloat16* __restrict__ input,   // [num_tokens, hidden_size]
    const __nv_bfloat16* __restrict__ weight,  // [hidden_size]
    __nv_bfloat16* __restrict__ output,         // [num_tokens, hidden_size]
    unsigned int hidden_size,
    float eps
) {
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;

    const __nv_bfloat16* x = input + token * hidden_size;
    __nv_bfloat16* out = output + token * hidden_size;

    // Step 1: Compute sum of squares — vectorized 2-wide BF16 loads
    float sum_sq = 0.0f;
    const unsigned int half_size = hidden_size / 2;
    const unsigned int* x32 = (const unsigned int*)x;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float v0, v1;
        unpack_bf16x2(x32[i], v0, v1);
        sum_sq += v0 * v0 + v1 * v1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = __bfloat162float(x[hidden_size - 1]);
        sum_sq += val * val;
    }

    // Step 2: Block-level reduction
    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;

    if (lane_id == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();

    // Step 3: Compute normalization factor
    float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);

    // Step 4: Apply normalization and weight — standard (no 1+offset)
    const unsigned int* w32 = (const unsigned int*)weight;
    unsigned int* out32 = (unsigned int*)out;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float xv0, xv1, wv0, wv1;
        unpack_bf16x2(x32[i], xv0, xv1);
        unpack_bf16x2(w32[i], wv0, wv1);
        out32[i] = pack_bf16x2(xv0 * rms * wv0, xv1 * rms * wv1);
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = __bfloat162float(x[hidden_size - 1]);
        float w = __bfloat162float(weight[hidden_size - 1]);
        out[hidden_size - 1] = __float2bfloat16(val * rms * w);
    }
}

// Fused RMS Norm + Residual Save: normed = w * norm(input), residual = input.
//
// Standard formulation (no offset). Used by Qwen3-VL.
//
// Grid: (num_tokens, 1, 1)
// Block: (min(hidden_size, 1024), 1, 1)
extern "C" __global__ void rms_norm_residual(
    const __nv_bfloat16* __restrict__ input,     // [num_tokens, hidden_size]
    const __nv_bfloat16* __restrict__ weight,    // [hidden_size]
    __nv_bfloat16* __restrict__ output,           // [num_tokens, hidden_size] (normed)
    __nv_bfloat16* __restrict__ residual,         // [num_tokens, hidden_size] (raw copy of input)
    unsigned int hidden_size,
    float eps
) {
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;

    const __nv_bfloat16* x = input + token * hidden_size;
    __nv_bfloat16* out = output + token * hidden_size;
    __nv_bfloat16* res = residual + token * hidden_size;

    float sum_sq = 0.0f;
    const unsigned int half_size = hidden_size / 2;
    const unsigned int* x32 = (const unsigned int*)x;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float v0, v1;
        unpack_bf16x2(x32[i], v0, v1);
        sum_sq += v0 * v0 + v1 * v1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = __bfloat162float(x[hidden_size - 1]);
        sum_sq += val * val;
    }

    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;

    if (lane_id == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);

    // Apply normalization + weight (standard, no offset), copy raw input to residual
    const unsigned int* w32 = (const unsigned int*)weight;
    unsigned int* out32 = (unsigned int*)out;
    unsigned int* res32 = (unsigned int*)res;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        unsigned int x_packed = x32[i];
        float xv0, xv1, wv0, wv1;
        unpack_bf16x2(x_packed, xv0, xv1);
        unpack_bf16x2(w32[i], wv0, wv1);
        out32[i] = pack_bf16x2(xv0 * rms * wv0, xv1 * rms * wv1);
        res32[i] = x_packed;
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = __bfloat162float(x[hidden_size - 1]);
        float w = __bfloat162float(weight[hidden_size - 1]);
        out[hidden_size - 1] = __float2bfloat16(val * rms * w);
        res[hidden_size - 1] = x[hidden_size - 1];
    }
}

// Fused Residual Add + RMS Norm + Residual Save.
//
// hidden[i] += src[i]; normed = rms_norm(hidden); residual = hidden.
// Standard formulation (no offset). Used by Qwen3-VL.
//
// Grid: (num_tokens, 1, 1)
// Block: (min(hidden_size, 1024), 1, 1)
extern "C" __global__ void residual_add_rms_norm(
    __nv_bfloat16* __restrict__ hidden,      // [num_tokens, hidden_size] in/out (hidden += src)
    const __nv_bfloat16* __restrict__ src,    // [num_tokens, hidden_size] added to hidden
    const __nv_bfloat16* __restrict__ weight, // [hidden_size]
    __nv_bfloat16* __restrict__ output,       // [num_tokens, hidden_size] (normed)
    __nv_bfloat16* __restrict__ residual,     // [num_tokens, hidden_size] (raw copy of updated hidden)
    unsigned int hidden_size,
    float eps
) {
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;

    __nv_bfloat16* h = hidden + token * hidden_size;
    const __nv_bfloat16* s = src + token * hidden_size;
    __nv_bfloat16* out = output + token * hidden_size;
    __nv_bfloat16* res = residual + token * hidden_size;

    // Pass 1: Add src to hidden, compute sum of squares
    float sum_sq = 0.0f;
    const unsigned int half_size = hidden_size / 2;
    unsigned int* h32 = (unsigned int*)h;
    const unsigned int* s32 = (const unsigned int*)s;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float hv0, hv1, sv0, sv1;
        unpack_bf16x2(h32[i], hv0, hv1);
        unpack_bf16x2(s32[i], sv0, sv1);
        float new0 = hv0 + sv0;
        float new1 = hv1 + sv1;
        h32[i] = pack_bf16x2(new0, new1);
        sum_sq += new0 * new0 + new1 * new1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float hv = __bfloat162float(h[hidden_size - 1]);
        float sv = __bfloat162float(s[hidden_size - 1]);
        float nv = hv + sv;
        h[hidden_size - 1] = __float2bfloat16(nv);
        sum_sq += nv * nv;
    }

    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;

    if (lane_id == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);

    // Pass 2: Apply normalization + weight (standard, no offset), copy to residual
    const unsigned int* w32 = (const unsigned int*)weight;
    unsigned int* out32 = (unsigned int*)out;
    unsigned int* res32 = (unsigned int*)res;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        unsigned int h_packed = h32[i];
        float xv0, xv1, wv0, wv1;
        unpack_bf16x2(h_packed, xv0, xv1);
        unpack_bf16x2(w32[i], wv0, wv1);
        out32[i] = pack_bf16x2(xv0 * rms * wv0, xv1 * rms * wv1);
        res32[i] = h_packed;
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = __bfloat162float(h[hidden_size - 1]);
        float w = __bfloat162float(weight[hidden_size - 1]);
        out[hidden_size - 1] = __float2bfloat16(val * rms * w);
        res[hidden_size - 1] = h[hidden_size - 1];
    }
}

// Fused RMS Norm + Gated variant (for Mamba layers).
// out = rms_norm(x) * SiLU(gate)   where SiLU(x) = x * sigmoid(x)
//
// Uses STANDARD weight (no 1+ offset).
// Gated RMS norm with norm_before_gate=False and per-group normalization.
// (Nemotron-H convention from mamba_ssm: gate first, then normalize per-group.)
//
// Algorithm:
//   1. temp = input * silu(gate)
//   2. For each group of group_size elements, compute RMS of temp
//   3. output = temp * weight / rms_per_group
//
// Grid: (num_tokens, 1, 1)
// Block: (min(hidden_size, 1024), 1, 1)
//
// Handles hidden_size > blockDim (e.g., d_inner=8192 for Super 120B):
// within one loop iteration, all warp threads process consecutive quads in the
// same group (group_quads >= 32), so per-iteration warp_reduce_sum is correct
// per-group. Cross-iteration accumulation uses atomicAdd to shared group buckets.
extern "C" __global__ void gated_rms_norm(
    const __nv_bfloat16* __restrict__ input,   // [num_tokens, hidden_size]
    const __nv_bfloat16* __restrict__ gate,    // [num_tokens, gate_stride]
    const __nv_bfloat16* __restrict__ weight,  // [hidden_size]
    __nv_bfloat16* __restrict__ output,         // [num_tokens, hidden_size]
    unsigned int hidden_size,
    float eps,
    unsigned int gate_stride,
    unsigned int group_size                     // 1024 for Super 120B, 512 for Nano 30B
) {
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;

    const __nv_bfloat16* x = input + token * hidden_size;
    const __nv_bfloat16* g = gate + (unsigned long long)token * gate_stride;
    __nv_bfloat16* out = output + token * hidden_size;

    const unsigned int quad_size = hidden_size / 4;
    const unsigned int group_quads = group_size / 4;
    const unsigned int num_groups = hidden_size / group_size;

    const unsigned long long* x64 = (const unsigned long long*)x;
    const unsigned long long* g64 = (const unsigned long long*)g;
    const unsigned long long* w64 = (const unsigned long long*)weight;
    unsigned long long* out64 = (unsigned long long*)out;

    unsigned int lane_id = tid % 32;

    // ── Per-group sum accumulators (shared memory) ──
    __shared__ float group_sums[8];
    __shared__ float group_rms[8];
    if (tid < num_groups) group_sums[tid] = 0.0f;
    __syncthreads();

    // ── Pass 1: Compute temp = input * silu(gate), accumulate per-group sum_sq ──
    float temp_cache[16];  // cache for temp values (max 4 quads per thread)
    unsigned int n_cached = 0;

    for (unsigned int i = tid; i < quad_size; i += blockDim.x) {
        unsigned long long xv = x64[i];
        float f0, f1, f2, f3;
        unpack_bf16x2((unsigned int)xv, f0, f1);
        unpack_bf16x2((unsigned int)(xv >> 32), f2, f3);

        unsigned long long gv = g64[i];
        float g0, g1, g2, g3;
        unpack_bf16x2((unsigned int)gv, g0, g1);
        unpack_bf16x2((unsigned int)(gv >> 32), g2, g3);

        // silu(z) = z * sigmoid(z)
        float s0 = g0 / (1.0f + __expf(-g0));
        float s1 = g1 / (1.0f + __expf(-g1));
        float s2 = g2 / (1.0f + __expf(-g2));
        float s3 = g3 / (1.0f + __expf(-g3));

        // temp = input * silu(gate)
        float t0 = f0 * s0, t1 = f1 * s1, t2 = f2 * s2, t3 = f3 * s3;
        temp_cache[n_cached]     = t0;
        temp_cache[n_cached + 1] = t1;
        temp_cache[n_cached + 2] = t2;
        temp_cache[n_cached + 3] = t3;
        n_cached += 4;

        // Per-group accumulation: warp-reduce, then atomicAdd to group bucket.
        // Within one iteration all warp threads process consecutive quads in the
        // same group (group_quads >= 32), so warp_reduce_sum is per-group correct.
        float sq = t0*t0 + t1*t1 + t2*t2 + t3*t3;
        float warp_sq = warp_reduce_sum(sq);
        if (lane_id == 0) {
            unsigned int grp = i / group_quads;
            atomicAdd(&group_sums[grp], warp_sq);
        }
    }
    __syncthreads();

    // Compute per-group RMS
    if (tid < num_groups) {
        group_rms[tid] = rsqrtf(group_sums[tid] / (float)group_size + eps);
    }
    __syncthreads();

    // ── Pass 2: Apply per-group rms and weight ──
    unsigned int ci = 0;
    for (unsigned int i = tid; i < quad_size; i += blockDim.x) {
        float rms = group_rms[i / group_quads];

        float t0 = temp_cache[ci];
        float t1 = temp_cache[ci + 1];
        float t2 = temp_cache[ci + 2];
        float t3 = temp_cache[ci + 3];
        ci += 4;

        unsigned long long wv = w64[i];
        float w0, w1, w2, w3;
        unpack_bf16x2((unsigned int)wv, w0, w1);
        unpack_bf16x2((unsigned int)(wv >> 32), w2, w3);

        unsigned int lo = pack_bf16x2(t0 * rms * w0, t1 * rms * w1);
        unsigned int hi = pack_bf16x2(t2 * rms * w2, t3 * rms * w3);
        out64[i] = ((unsigned long long)hi << 32) | (unsigned long long)lo;
    }
}

// L2 Normalization (in-place): x[i] = x[i] / sqrt(sum(x^2) + eps)
//
// Used for Q/K normalization.
//
// Grid: (num_heads, num_tokens, 1)
// Block: (min(head_dim, 1024), 1, 1)
extern "C" __global__ void l2_norm_bf16(
    __nv_bfloat16* __restrict__ data,
    unsigned int head_dim,
    float eps,
    unsigned int stride
) {
    unsigned int head = blockIdx.x;
    unsigned int token = blockIdx.y;
    unsigned int tid = threadIdx.x;

    __nv_bfloat16* x = data + (unsigned long long)token * stride + head * head_dim;

    float sum_sq = 0.0f;
    const unsigned int half_size = head_dim / 2;
    const unsigned int* x32 = (const unsigned int*)x;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float v0, v1;
        unpack_bf16x2(x32[i], v0, v1);
        sum_sq += v0 * v0 + v1 * v1;
    }
    if ((head_dim & 1) && tid == 0) {
        float val = __bfloat162float(x[head_dim - 1]);
        sum_sq += val * val;
    }

    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;

    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) warp_sums[0] = val;
    }
    __syncthreads();

    float inv_norm = rsqrtf(warp_sums[0] + eps);

    unsigned int* out32 = (unsigned int*)x;
    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float v0, v1;
        unpack_bf16x2(x32[i], v0, v1);
        out32[i] = pack_bf16x2(v0 * inv_norm, v1 * inv_norm);
    }
    if ((head_dim & 1) && tid == 0) {
        float val = __bfloat162float(x[head_dim - 1]);
        x[head_dim - 1] = __float2bfloat16(val * inv_norm);
    }
}

