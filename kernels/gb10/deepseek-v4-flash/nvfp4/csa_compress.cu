// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4 sparse-attention compressor: window softmax-gated KV compression.
//
// Reference: modeling_deepseek_v4.py DeepseekV4CSACompressor / HCACompressor.
// Produces one compressed KV entry per window of `ratio` source tokens:
//   C[w,d] = sum_s softmax_s(gate[w,s,d] + ape[r,*]) * kv[w,s,d]    (per-dim softmax)
//
// CSA (ratio 4): proj_dim = 2*head_dim. Two interleaved series Ca/Cb with a
//   2*ratio-wide overlap window (stride ratio): slots [0,ratio) = previous
//   window's Ca (kv[..,:head_dim]); slots [ratio,2*ratio) = current window's Cb
//   (kv[..,head_dim:]). Window 0's Ca half is masked (gate -inf, weight 0).
// HCA (ratio 128): proj_dim = head_dim. Single non-overlapping window of `ratio`.
//
// Output `out` is the RAW compressed KV [n_win, head_dim] (kv_norm + rope applied
// by the caller). Grid: (n_win,1,1)  Block: (256,1,1).

#include <cuda_bf16.h>

extern "C" __global__ void csa_compress(
    const __nv_bfloat16* __restrict__ kv,   // [S, proj_dim]
    const __nv_bfloat16* __restrict__ gate, // [S, proj_dim]
    const float* __restrict__ ape,          // [ratio, proj_dim] FP32: checkpoint-native
                                            // dtype (reading as bf16 read half of the
                                            // wrong element and corrupted the window
                                            // softmax gate on every compressed layer)
    __nv_bfloat16* __restrict__ out,        // [n_win, head_dim]
    const unsigned int seq_len,
    const unsigned int ratio,
    const unsigned int head_dim,
    const unsigned int proj_dim,
    const unsigned int is_csa
) {
    const unsigned int w = blockIdx.x;
    const unsigned int n_win = seq_len / ratio;
    if (w >= n_win) return;

    for (unsigned int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        // Online softmax over the slots, per output dim d.
        float m = -1e30f, l = 0.0f, acc = 0.0f;

        if (is_csa) {
            // Ca slots: previous window's first head_dim of the proj. Skipped
            // (gate -inf, weight 0) for window 0 — no cross-call cache here.
            if (w > 0) {
                for (unsigned int r = 0; r < ratio; ++r) {
                    const unsigned int tok = (w - 1) * ratio + r;
                    const float g = __bfloat162float(gate[(size_t)tok * proj_dim + d])
                                  + ape[(size_t)r * proj_dim + d];
                    const float v = __bfloat162float(kv[(size_t)tok * proj_dim + d]);
                    const float mn = fmaxf(m, g);
                    const float eo = __expf(m - mn);
                    const float en = __expf(g - mn);
                    l = l * eo + en;
                    acc = acc * eo + en * v;
                    m = mn;
                }
            }
            // Cb slots: current window's second head_dim of the proj.
            for (unsigned int r = 0; r < ratio; ++r) {
                const unsigned int tok = w * ratio + r;
                const unsigned int c = head_dim + d;
                const float g = __bfloat162float(gate[(size_t)tok * proj_dim + c])
                              + ape[(size_t)r * proj_dim + c];
                const float v = __bfloat162float(kv[(size_t)tok * proj_dim + c]);
                const float mn = fmaxf(m, g);
                const float eo = __expf(m - mn);
                const float en = __expf(g - mn);
                l = l * eo + en;
                acc = acc * eo + en * v;
                m = mn;
            }
        } else {
            // HCA: single window of `ratio` tokens, proj_dim == head_dim.
            for (unsigned int r = 0; r < ratio; ++r) {
                const unsigned int tok = w * ratio + r;
                const float g = __bfloat162float(gate[(size_t)tok * proj_dim + d])
                              + ape[(size_t)r * proj_dim + d];
                const float v = __bfloat162float(kv[(size_t)tok * proj_dim + d]);
                const float mn = fmaxf(m, g);
                const float eo = __expf(m - mn);
                const float en = __expf(g - mn);
                l = l * eo + en;
                acc = acc * eo + en * v;
                m = mn;
            }
        }

        out[(size_t)w * head_dim + d] = __float2bfloat16(l > 0.0f ? acc / l : 0.0f);
    }
}
