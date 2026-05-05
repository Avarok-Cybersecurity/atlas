// SPDX-License-Identifier: AGPL-3.0-only

// Atlas RMS Normalization kernel for Gemma-4 (SM121).
//
// Gemma-4 uses STANDARD RMS normalization:
//   RMSNorm(x) = x * weight / sqrt(mean(x^2) + eps)
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
// Standard formulation (no offset). Used by Gemma-4.
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

    // Step 1: Compute sum of squares -- vectorized 2-wide BF16 loads
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

    // Step 4: Apply normalization and weight -- standard (no 1+offset)
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
// Standard formulation (no offset). Used by Gemma-4.
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
// Standard formulation (no offset). Used by Gemma-4.
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

// ═══════════════════════════════════════════════════════════════════
// FP32 residual variants — absolute formula `out = x * rms * w`
// (shared with gemma-4-31b; see that file's equivalent block for notes).
// ═══════════════════════════════════════════════════════════════════

extern "C" __global__ void rms_norm_residual_f32_abs(
    const float* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ residual,
    unsigned int hidden_size,
    float eps
) {
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;

    const float* x = input + token * hidden_size;
    __nv_bfloat16* out = output + token * hidden_size;
    float* res = residual + token * hidden_size;

    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < hidden_size; i += blockDim.x) {
        float v = x[i];
        sum_sq += v * v;
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

    float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);

    const unsigned int half_size = hidden_size / 2;
    const unsigned int* w32 = (const unsigned int*)weight;
    unsigned int* out32 = (unsigned int*)out;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        unsigned int base = i * 2;
        float xv0 = x[base];
        float xv1 = x[base + 1];
        float wv0, wv1;
        unpack_bf16x2(w32[i], wv0, wv1);
        out32[i] = pack_bf16x2(xv0 * rms * wv0, xv1 * rms * wv1);
        res[base]     = xv0;
        res[base + 1] = xv1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = x[hidden_size - 1];
        float w = __bfloat162float(weight[hidden_size - 1]);
        out[hidden_size - 1] = __float2bfloat16(val * rms * w);
        res[hidden_size - 1] = val;
    }
}

extern "C" __global__ void residual_add_rms_norm_f32_abs(
    float* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ src,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ residual,
    unsigned int hidden_size,
    float eps
) {
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;

    float* h = hidden + token * hidden_size;
    const __nv_bfloat16* s = src + token * hidden_size;
    __nv_bfloat16* out = output + token * hidden_size;
    float* res = residual + token * hidden_size;

    float sum_sq = 0.0f;
    const unsigned int half_size = hidden_size / 2;
    const unsigned int* s32 = (const unsigned int*)s;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        unsigned int base = i * 2;
        float sv0, sv1;
        unpack_bf16x2(s32[i], sv0, sv1);
        float new0 = h[base]     + sv0;
        float new1 = h[base + 1] + sv1;
        h[base]     = new0;
        h[base + 1] = new1;
        sum_sq += new0 * new0 + new1 * new1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float hv = h[hidden_size - 1];
        float sv = __bfloat162float(s[hidden_size - 1]);
        float nv = hv + sv;
        h[hidden_size - 1] = nv;
        sum_sq += nv * nv;
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

    float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);

    const unsigned int* w32 = (const unsigned int*)weight;
    unsigned int* out32 = (unsigned int*)out;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        unsigned int base = i * 2;
        float xv0 = h[base];
        float xv1 = h[base + 1];
        float wv0, wv1;
        unpack_bf16x2(w32[i], wv0, wv1);
        out32[i] = pack_bf16x2(xv0 * rms * wv0, xv1 * rms * wv1);
        res[base]     = xv0;
        res[base + 1] = xv1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = h[hidden_size - 1];
        float w = __bfloat162float(weight[hidden_size - 1]);
        out[hidden_size - 1] = __float2bfloat16(val * rms * w);
        res[hidden_size - 1] = val;
    }
}

// See gemma-4-31b variant for docs — absolute-formula FP32-input rms_norm
// used by the final-norm call before LM head.
extern "C" __global__ void rms_norm_f32(
    const float* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int hidden_size,
    float eps
) {
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;

    const float* x = input + token * hidden_size;
    __nv_bfloat16* out = output + token * hidden_size;

    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < hidden_size; i += blockDim.x) {
        float v = x[i];
        sum_sq += v * v;
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

    float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);

    const unsigned int half_size = hidden_size / 2;
    const unsigned int* w32 = (const unsigned int*)weight;
    unsigned int* out32 = (unsigned int*)out;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        unsigned int base = i * 2;
        float xv0 = x[base];
        float xv1 = x[base + 1];
        float wv0, wv1;
        unpack_bf16x2(w32[i], wv0, wv1);
        out32[i] = pack_bf16x2(xv0 * rms * wv0, xv1 * rms * wv1);
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = x[hidden_size - 1];
        float w = __bfloat162float(weight[hidden_size - 1]);
        out[hidden_size - 1] = __float2bfloat16(val * rms * w);
    }
}

extern "C" __global__ void f32_residual_add(
    float* __restrict__ residual,
    const __nv_bfloat16* __restrict__ src,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        residual[i] += __bfloat162float(src[i]);
    }
}
