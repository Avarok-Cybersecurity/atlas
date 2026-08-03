// SPDX-License-Identifier: AGPL-3.0-only

// GPU-only DeepSeek-V4 MTP multi-stream combiner.
//
// The legacy proposer copied the four FP32 mHC streams to the host, converted
// and projected each stream separately, then copied the results back. These
// kernels preserve that path's BF16 boundaries while keeping the whole
// combiner on the device.

#include <cuda_bf16.h>

#define MTP_BLOCK_SIZE 256
#define MTP_N_PER_BLOCK 4
#define MTP_WARP_SIZE 32
#define MTP_VEC_SIZE 8
#define MTP_MAX_STREAMS 4

__device__ __forceinline__ __nv_bfloat16 mtp_legacy_f32_to_bf16(float value) {
    // Match the legacy Rust conversion exactly:
    // value.to_bits().wrapping_add(0x8000) >> 16.
    unsigned int bits = __float_as_uint(value);
    unsigned short rounded = (unsigned short)((bits + 0x8000u) >> 16);
    __nv_bfloat16 out;
    *(unsigned short*)&out = rounded;
    return out;
}

extern "C" __global__ void mtp_hc_f32_to_bf16_legacy(
    const float* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    unsigned int total_elements
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_elements) {
        output[i] = mtp_legacy_f32_to_bf16(input[i]);
    }
}

// Compute all MTP h_proj stream rows in one weight pass:
//
//   streams_out[s, n] = bf16(bf16(dot(normed[s], weight[n])) + e_branch[n])
//
// The final BF16 value is promoted to FP32 for the mHC highway. The reduction
// geometry and accumulation order match dense_gemv_bf16, so this retains the
// legacy numerical boundaries while reading the 32 MiB h_proj weight once
// instead of once per stream.
//
// Grid: (ceil(N / 4), 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void mtp_hproj_gemv_batch4(
    const __nv_bfloat16* __restrict__ A,         // [hc_mult, K]
    const __nv_bfloat16* __restrict__ B,         // [N, K]
    const __nv_bfloat16* __restrict__ e_branch,  // [N]
    float* __restrict__ streams_out,              // [hc_mult, N]
    unsigned int hc_mult,
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = MTP_BLOCK_SIZE / MTP_N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n = blockIdx.x * MTP_N_PER_BLOCK + local_out;
    const bool active = n < N;

    float acc[MTP_MAX_STREAMS] = {0.f, 0.f, 0.f, 0.f};
    const unsigned int k_vec = K / MTP_VEC_SIZE;
    if (active) {
        const uint4* b_vec = (const uint4*)(B + (unsigned long long)n * K);
        for (unsigned int kv = lane; kv < k_vec; kv += threads_per_out) {
            uint4 b_data = b_vec[kv];
            const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};

            #pragma unroll
            for (unsigned int s = 0; s < MTP_MAX_STREAMS; ++s) {
                if (s >= hc_mult) break;
                const uint4* a_vec = (const uint4*)(A + (unsigned long long)s * K);
                uint4 a_data = a_vec[kv];
                const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};

                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    __nv_bfloat16 a_lo, a_hi, b_lo, b_hi;
                    *(unsigned short*)&a_lo = (unsigned short)(a_raw[i] & 0xFFFF);
                    *(unsigned short*)&a_hi = (unsigned short)(a_raw[i] >> 16);
                    *(unsigned short*)&b_lo = (unsigned short)(b_raw[i] & 0xFFFF);
                    *(unsigned short*)&b_hi = (unsigned short)(b_raw[i] >> 16);
                    acc[s] += __bfloat162float(a_lo) * __bfloat162float(b_lo);
                    acc[s] += __bfloat162float(a_hi) * __bfloat162float(b_hi);
                }
            }
        }

        const unsigned int tail_start = k_vec * MTP_VEC_SIZE;
        const __nv_bfloat16* b_row = B + (unsigned long long)n * K;
        for (unsigned int k = tail_start + lane; k < K; k += threads_per_out) {
            const float b = __bfloat162float(b_row[k]);
            #pragma unroll
            for (unsigned int s = 0; s < MTP_MAX_STREAMS; ++s) {
                if (s >= hc_mult) break;
                acc[s] += __bfloat162float(A[(unsigned long long)s * K + k]) * b;
            }
        }
    }

    #pragma unroll
    for (int offset = MTP_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        #pragma unroll
        for (unsigned int s = 0; s < MTP_MAX_STREAMS; ++s) {
            if (s >= hc_mult) break;
            acc[s] += __shfl_down_sync(0xFFFFFFFF, acc[s], offset);
        }
    }

    __shared__ float partial[MTP_MAX_STREAMS][MTP_N_PER_BLOCK][2];
    const unsigned int warp_lane = threadIdx.x % MTP_WARP_SIZE;
    if (warp_lane == 0) {
        #pragma unroll
        for (unsigned int s = 0; s < MTP_MAX_STREAMS; ++s) {
            if (s >= hc_mult) break;
            partial[s][local_out][lane / MTP_WARP_SIZE] = acc[s];
        }
    }
    __syncthreads();

    if (lane == 0 && active) {
        const float e = __bfloat162float(e_branch[n]);
        #pragma unroll
        for (unsigned int s = 0; s < MTP_MAX_STREAMS; ++s) {
            if (s >= hc_mult) break;
            float projected = partial[s][local_out][0] + partial[s][local_out][1];
            __nv_bfloat16 projected_bf16 = __float2bfloat16(projected);
            __nv_bfloat16 combined = __float2bfloat16(__bfloat162float(projected_bf16) + e);
            streams_out[(unsigned long long)s * N + n] = __bfloat162float(combined);
        }
    }
}

// Batched combiner epilogue:
//
//   streams_out[t, s, d] =
//       f32(bf16(h_branch[t, s, d] + e_branch[t, d]))
//
// Both projection outputs are already BF16. The explicit final BF16 rounding
// matches the decode combiner. V4 masks the embedding branch at target
// position zero.
extern "C" __global__ void mtp_hproj_broadcast_add_batched(
    const __nv_bfloat16* __restrict__ h_branch,  // [T, hc_mult, H]
    const __nv_bfloat16* __restrict__ e_branch,  // [T, H]
    float* __restrict__ streams_out,             // [T, hc_mult, H]
    unsigned int num_tokens,
    unsigned int hc_mult,
    unsigned int hidden_size,
    unsigned int first_position
) {
    const unsigned long long total =
        (unsigned long long)num_tokens * hc_mult * hidden_size;
    const unsigned long long i =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }

    const unsigned int d = i % hidden_size;
    const unsigned int token =
        (unsigned int)(i / ((unsigned long long)hc_mult * hidden_size));
    const float h = __bfloat162float(h_branch[i]);
    const float e = first_position + token == 0
        ? 0.0f
        : __bfloat162float(e_branch[(unsigned long long)token * hidden_size + d]);
    streams_out[i] = __bfloat162float(__float2bfloat16(h + e));
}
