// SPDX-License-Identifier: AGPL-3.0-only

// Argmax over BF16 logits — single-block tree reduction.
//
// Finds the index of the maximum BF16 value in an array of `n` elements.
// Writes a single u32 token ID to `out`.
//
// Grid: (1, 1, 1)  Block: (1024, 1, 1)
// For vocab_size ≤ ~200K, a single block with 1024 threads is sufficient
// (each thread handles ceil(n/1024) elements).

#include <cuda_bf16.h>

extern "C" __global__ void argmax_bf16(
    const __nv_bfloat16* __restrict__ logits,
    unsigned int* __restrict__ out,
    unsigned int n
) {
    __shared__ float s_val[1024];
    __shared__ unsigned int s_idx[1024];

    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;

    // Phase 1: each thread finds its local max
    float local_max = -1e30f;
    unsigned int local_idx = 0;

    for (unsigned int i = tid; i < n; i += stride) {
        float v = __bfloat162float(logits[i]);
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }

    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    // Phase 2: tree reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_val[tid + s] > s_val[tid]) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }

    // Phase 3: thread 0 writes result
    if (tid == 0) {
        out[0] = s_idx[0];
    }
}

// BATCHED argmax over BF16 logits — ONE BLOCK PER ROW.
//
// `argmax_batch` used to launch this kernel n times on the same stream, and the
// single-row kernel is a ONE-CTA reduction (grid [1,1,1]), so it uses 1 of 48 SMs
// and measured 100.6 us to reduce 248320 bf16 = 497 KB (~5 GB/s). At n=16 that is
// 16 serial launches = 1.6 ms per decode step.
//
// Each block here runs the IDENTICAL per-row body: a strided scan keeping the first
// strict max, then the same tree reduction preferring the lower tid. Ties therefore
// resolve to the same index as n sequential calls — byte-identical by construction.
extern "C" __global__ void argmax_bf16_batch(
    const __nv_bfloat16* __restrict__ logits,
    unsigned int* __restrict__ out,
    unsigned int n,
    unsigned int row_stride
) {
    __shared__ float s_val[1024];
    __shared__ unsigned int s_idx[1024];

    const unsigned int row = blockIdx.x;
    const __nv_bfloat16* __restrict__ row_logits =
        logits + (unsigned long long)row * (unsigned long long)row_stride;

    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;

    float local_max = -1e30f;
    unsigned int local_idx = 0;
    for (unsigned int i = tid; i < n; i += stride) {
        float v = __bfloat162float(row_logits[i]);
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }

    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_val[tid + s] > s_val[tid]) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[row] = s_idx[0];
    }
}

// BATCHED argmax + top-1 LOG-PROBABILITY over BF16 logits — one block per row.
//
// Identical index semantics to `argmax_bf16_batch` (same strided first-strict-max
// scan, same lower-tid-preferring tree reduction), plus `out_logprob[row] =
// log softmax(logits)[argmax]` computed by ONLINE softmax in the SAME pass — no
// second read of the row, so the kernel costs what the plain batched argmax costs.
//
// Consumer: D-Cut adaptive verification-depth pruning (ATLAS_MTP_DCUT). The
// drafter's per-position top-1 log-probability is the confidence whose prefix
// SUM (= log of the prefix PRODUCT of survival probabilities) ranks candidate
// verify positions across the batch. A separate kernel (rather than an extra
// argument on `argmax_bf16_batch`) keeps every existing caller byte-identical
// and makes an unresolved handle a silent-0 the caller can gate on.
//
// log p_max = m - logsumexp = m - (m + log Σ exp(v_i - m)) = -log Σ exp(v_i - m).
extern "C" __global__ void argmax_bf16_batch_lp(
    const __nv_bfloat16* __restrict__ logits,
    unsigned int* __restrict__ out,
    float* __restrict__ out_logprob,
    unsigned int n,
    unsigned int row_stride
) {
    __shared__ float s_val[1024];
    __shared__ unsigned int s_idx[1024];
    __shared__ float s_sum[1024];

    const unsigned int row = blockIdx.x;
    const __nv_bfloat16* __restrict__ row_logits =
        logits + (unsigned long long)row * (unsigned long long)row_stride;

    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;

    float local_max = -1e30f;
    unsigned int local_idx = 0;
    float local_sum = 0.0f;
    for (unsigned int i = tid; i < n; i += stride) {
        float v = __bfloat162float(row_logits[i]);
        if (v > local_max) {
            // Rescale the running sum to the new max, then add this element.
            local_sum = local_sum * __expf(local_max - v) + 1.0f;
            local_max = v;
            local_idx = i;
        } else {
            local_sum += __expf(v - local_max);
        }
    }

    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    s_sum[tid] = local_sum;
    __syncthreads();

    // Tree reduction over (max, argmax, sum) triples. Index/max half is the
    // byte-identical body of `argmax_bf16_batch`; the sum half is the standard
    // online-softmax merge. Threads that scanned nothing carry max=-1e30 and
    // sum=0, which merge as a no-op.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float mine = s_val[tid];
            const float other = s_val[tid + s];
            if (other > mine) {
                s_sum[tid] = s_sum[tid] * __expf(mine - other) + s_sum[tid + s];
                s_val[tid] = other;
                s_idx[tid] = s_idx[tid + s];
            } else {
                s_sum[tid] = s_sum[tid] + s_sum[tid + s] * __expf(other - mine);
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[row] = s_idx[0];
        const float total = s_sum[0];
        // total >= 1 for any non-empty row (the max element contributes 1).
        out_logprob[row] = (total > 0.0f) ? -__logf(total) : 0.0f;
    }
}

// Argmax over FP32 logits — used when LM head outputs FP32 for sampling quality.
extern "C" __global__ void argmax_fp32(
    const float* __restrict__ logits,
    unsigned int* __restrict__ out,
    unsigned int n
) {
    __shared__ float s_val[1024];
    __shared__ unsigned int s_idx[1024];

    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;

    float local_max = -1e30f;
    unsigned int local_idx = 0;
    for (unsigned int i = tid; i < n; i += stride) {
        float v = logits[i];
        if (v > local_max) { local_max = v; local_idx = i; }
    }
    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && s_val[tid + s] > s_val[tid]) {
            s_val[tid] = s_val[tid + s];
            s_idx[tid] = s_idx[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) out[0] = s_idx[0];
}
