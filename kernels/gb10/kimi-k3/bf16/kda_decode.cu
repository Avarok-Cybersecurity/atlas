// SPDX-License-Identifier: AGPL-3.0-only
//
// Kimi K3 one-token KDA decode. Unique stem `kda_decode` — not a shadow of
// GLM recurrent KDA, GDN decode, or Mamba-2 SSM kernels in common/.
//
// Matches avarok-core `kda_decode_token`:
//   conv: Avarok-width [C, K] shift-left, write x into last slot, SiLU(dot(w,s))
//   recurrent: L2(q), L2(k), V raw; S *= exp(gate) on KEY; beta is a logit
//              (sigmoid here); delta = (v - S^T k) * sigmoid(beta);
//              S += k ⊗ delta; o = S^T q / sqrt(D)
//
// Twin: H=8 D=32 K=4. Production: H=96 D=128 K=4. Runtime H/D/K, not hardcoded.

#include <math.h>

__device__ __forceinline__ float k3_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// One thread per conv channel. State is [C, K] row-major, slot 0 aged out.
extern "C" __global__ void k3_kda_conv_update_f32(
    const float* __restrict__ x,      // [C]
    const float* __restrict__ w,      // [C, K]
    float* __restrict__ state,        // [C, K] rmw
    float* __restrict__ y,            // [C]
    unsigned int C,
    unsigned int K
) {
    const unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C || K == 0u) {
        return;
    }
    const unsigned int row = c * K;
    for (unsigned int k = 0; k + 1u < K; ++k) {
        state[row + k] = state[row + k + 1u];
    }
    state[row + (K - 1u)] = x[c];
    float acc = 0.0f;
    for (unsigned int k = 0; k < K; ++k) {
        acc += w[row + k] * state[row + k];
    }
    y[c] = acc * k3_sigmoid(acc);
}

__device__ void k3_l2_row(const float* in, float* out, unsigned int D, float eps) {
    float ss = eps;
    for (unsigned int i = 0; i < D; ++i) {
        const float v = in[i];
        ss += v * v;
    }
    // 1/sqrt, not rsqrtf: must match Rust `1.0 / (sum + eps).sqrt()`.
    const float inv = 1.0f / sqrtf(ss);
    for (unsigned int i = 0; i < D; ++i) {
        out[i] = in[i] * inv;
    }
}

// One block per head. Threads stride the V axis; K is sequential (CPU order).
// qkv is post-conv [q | k | v]. gate is log-decay. beta is a raw logit.
extern "C" __global__ void k3_kda_recurrent_step_f32(
    const float* __restrict__ qkv,    // [3 * H * D]
    const float* __restrict__ gate,   // [H * D] log-decay
    const float* __restrict__ beta,   // [H] logit
    float* __restrict__ state,        // [H, D, D] K-major rmw
    float* __restrict__ out,          // [H * D]
    unsigned int H,
    unsigned int D,
    float l2_eps
) {
    const unsigned int h = blockIdx.x;
    if (h >= H || D == 0u) {
        return;
    }
    extern __shared__ float sh[];
    float* sh_q = sh;
    float* sh_k = sh + D;
    float* sh_decay = sh + 2u * D;

    const unsigned int qkv_dim = H * D;
    const unsigned int base = h * D;
    if (threadIdx.x == 0u) {
        k3_l2_row(qkv + base, sh_q, D, l2_eps);
        k3_l2_row(qkv + qkv_dim + base, sh_k, D, l2_eps);
        for (unsigned int i = 0; i < D; ++i) {
            sh_decay[i] = expf(gate[base + i]);
        }
    }
    __syncthreads();

    const float b = k3_sigmoid(beta[h]);
    const float scale = 1.0f / sqrtf((float)D);
    const float* v = qkv + 2u * qkv_dim + base;
    float* S = state + (size_t)h * (size_t)D * (size_t)D;

    for (unsigned int vi = threadIdx.x; vi < D; vi += blockDim.x) {
        float kv = 0.0f;
        for (unsigned int kk = 0; kk < D; ++kk) {
            const size_t idx = (size_t)kk * D + vi;
            const float s = S[idx] * sh_decay[kk];
            S[idx] = s;
            kv += s * sh_k[kk];
        }
        const float delta = (v[vi] - kv) * b;
        float o = 0.0f;
        for (unsigned int kk = 0; kk < D; ++kk) {
            const size_t idx = (size_t)kk * D + vi;
            const float s = S[idx] + sh_k[kk] * delta;
            S[idx] = s;
            o += s * sh_q[kk] * scale;
        }
        out[base + vi] = o;
    }
}
