// SPDX-License-Identifier: AGPL-3.0-only
//
// Kimi K3 one-token gated NoPE MLA decode. Unique stem `mla_decode`.
// Not a GDN mixer, not a paged-latent paste, not Qwen full-attn.
//
// Matches atlas-core `mla_decode_token`:
//   maybe_rope: rope slots exist (nope|rope packed) but NoPE does not rotate
//   sdpa: one query [H, dq] vs cached K/V length T, scale 1/sqrt(dq)
//   output gate: sigmoid(g) ⊙ attn when enabled
//
// Twin: H=8 nope=64 rope=32 dv=64. Prod: H=96 nope=128 rope=64 dv=128.
// Runtime H/dq/dv/T, not hardcoded.

#include <math.h>

__device__ __forceinline__ float k3_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ void k3_mla_rope_one(
    float* x,
    unsigned int nope,
    unsigned int rope,
    unsigned int pos,
    float theta
) {
    float* r = x + nope;
    for (unsigned int i = 0; i < rope / 2u; ++i) {
        const float freq = (float)pos / powf(theta, 2.0f * (float)i / (float)rope);
        float s, c;
        sincosf(freq, &s, &c);
        const float a = r[i];
        const float b = r[i + rope / 2u];
        r[i] = a * c - b * s;
        r[i + rope / 2u] = a * s + b * c;
    }
}

// One thread per head. use_nope=1 leaves q/k unchanged (slots still packed).
extern "C" __global__ void k3_mla_maybe_rope_f32(
    float* __restrict__ q,            // [H, nope+rope]
    float* __restrict__ k,            // [H, nope+rope]
    unsigned int H,
    unsigned int nope,
    unsigned int rope,
    unsigned int pos,
    float theta,
    unsigned int use_nope
) {
    if (use_nope != 0u || rope == 0u) {
        return;
    }
    const unsigned int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) {
        return;
    }
    const unsigned int dq = nope + rope;
    k3_mla_rope_one(q + h * dq, nope, rope, pos, theta);
    k3_mla_rope_one(k + h * dq, nope, rope, pos, theta);
}

// One block per head. Thread 0 matches CPU sdpa_one + apply_output_gate order.
extern "C" __global__ void k3_mla_sdpa_gate_f32(
    const float* __restrict__ q,      // [H, dq]
    const float* __restrict__ k,      // [T, H, dq]
    const float* __restrict__ v,      // [T, H, dv]
    const float* __restrict__ g,      // [H, dv]
    float* __restrict__ out,          // [H, dv]
    unsigned int T,
    unsigned int H,
    unsigned int dq,
    unsigned int dv,
    unsigned int use_gate
) {
    const unsigned int h = blockIdx.x;
    if (h >= H || threadIdx.x != 0u || dq == 0u) {
        return;
    }
    const float scale = 1.0f / sqrtf((float)dq);
    const float* qrow = q + h * dq;
    float m = -INFINITY;
    for (unsigned int kj = 0; kj < T; ++kj) {
        const float* krow = k + (kj * H + h) * dq;
        float s = 0.0f;
        for (unsigned int d = 0; d < dq; ++d) {
            s += qrow[d] * krow[d];
        }
        s *= scale;
        if (s > m) {
            m = s;
        }
    }
    float z = 0.0f;
    for (unsigned int kj = 0; kj < T; ++kj) {
        const float* krow = k + (kj * H + h) * dq;
        float s = 0.0f;
        for (unsigned int d = 0; d < dq; ++d) {
            s += qrow[d] * krow[d];
        }
        z += expf(s * scale - m);
    }
    for (unsigned int d = 0; d < dv; ++d) {
        float o = 0.0f;
        for (unsigned int kj = 0; kj < T; ++kj) {
            const float* krow = k + (kj * H + h) * dq;
            float s = 0.0f;
            for (unsigned int i = 0; i < dq; ++i) {
                s += qrow[i] * krow[i];
            }
            const float a = expf(s * scale - m) / z;
            o += a * v[(kj * H + h) * dv + d];
        }
        if (use_gate != 0u) {
            o *= k3_sigmoid(g[h * dv + d]);
        }
        out[h * dv + d] = o;
    }
}
