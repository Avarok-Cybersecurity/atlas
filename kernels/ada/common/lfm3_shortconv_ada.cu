// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from HuggingFace transformers (models/lfm3, Apache-2.0, Copyright 2026 The
// HuggingFace Team) and vLLM (vllm/model_executor/models/lfm3.py, Apache-2.0, Copyright contributors
// to the vLLM project). Changes: ported from Python/Triton to Rust/CUDA for Atlas. The math is
// `Lfm3ShortConv.forward` (B, C, x = chunk(in_proj(h), 3); y = C * conv7(B * x)) and the decode and
// packed-sequence prefill of `lfm3_conv.py`; here the gating is fused around the convolution, the
// input is the token-major in_proj output, and the state is a per-sequence BF16 `[dim, 6]` window.

// Ada (sm_89) gated short convolution of Liquid AI LFM3 (port lane L38), three entries:
//
//   lfm3_shortconv_decode_bf16        one token per sequence, up to 16 sequences per launch
//   lfm3_shortconv_prefill_bf16       y for a token-major chunk, packed sequences, 6-token halo
//   lfm3_shortconv_prefill_state_bf16 the chunk's final state, in place (runs after the y kernel)
//
// ── MATH (per channel c, per token t of one sequence) ─────────────────────
//
//   BCx[t] = [ B(dim) | C(dim) | x(dim) ]          the in_proj row, BF16
//   u[t]   = bf16( B[t, c] * x[t, c] )               torch's bf16 elementwise product
//   v[t]   = bf16( sum_{k=0..6} w[c, k] * u[t-6+k] ) FP32 accumulate, k ascending, from 0.0f
//   y[t]   = bf16( C[t, c] * v[t] )
//
// `u` at a position before the chunk comes from the state: state[c, j] holds u at relative
// position j - 6 (j = 0 is the oldest of the six). A fresh sequence's state is zero, which is the
// reference's left zero padding. w[c, 0] multiplies the oldest input (PyTorch conv1d correlation).
//
// ── NUMERICS ───────────────────────────────────────────────────────────────
//
// Every product and sum is an explicit `__fmul_rn` / `__fadd_rn`: the reference Triton kernels
// disable FMA contraction (`enable_fp_fusion=False`), and common/KERNEL.toml already passes
// `--fmad=false`, so the two agree twice over. The CPU reference
// (`crates/spark-model/src/port/lfm3_conv/reference.rs`) performs the same IEEE f32 operations in
// the same order and the same round-to-nearest-even BF16 narrowing, so the GPU output is
// bit-identical to it (the microtest gate is exact equality).
//
// ── LAYOUT ─────────────────────────────────────────────────────────────────
//
//   weight  [dim, 7]  BF16, the checkpoint's conv.conv.weight [dim, 1, 7] squeezed
//   state   [dim, 6]  BF16 per sequence per conv layer, channel-major
//   BCx     [rows, bcx_stride] BF16, bcx_stride >= 3 * dim (the in_proj output, no transpose)
//   y       [rows, y_stride]   BF16
//
// Threads map to channels (128 per block), so every BCx and y access of a warp is one coalesced
// 64-byte segment per token. The state window is 12 bytes per channel and stays in registers.

#include <cuda_bf16.h>

#define LFM3_CONV_WIDTH 7
#define LFM3_STATE_LEN 6
#define LFM3_THREADS 128
#define LFM3_TILE_T 64
#define LFM3_DECODE_SEQS 16

__device__ __forceinline__ float lfm3_bf(const __nv_bfloat16 v) { return __bfloat162float(v); }

// Round an FP32 value to BF16 (round to nearest even) and widen it back: the value a BF16 tensor
// holds after torch's elementwise op.
__device__ __forceinline__ float lfm3_round(const float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// u = bf16(B * x) for one channel of one BCx row.
__device__ __forceinline__ float lfm3_u(const __nv_bfloat16* __restrict__ row, const unsigned int c,
                                        const unsigned int dim) {
    return lfm3_round(__fmul_rn(lfm3_bf(row[c]), lfm3_bf(row[2u * dim + c])));
}

// v = bf16(sum_k w[k] * window[k]), window = six older inputs then the current one.
__device__ __forceinline__ float lfm3_conv(const float (&w)[LFM3_CONV_WIDTH], const float (&win)[LFM3_STATE_LEN],
                                           const float u) {
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < LFM3_STATE_LEN; ++k) {
        acc = __fadd_rn(acc, __fmul_rn(w[k], win[k]));
    }
    acc = __fadd_rn(acc, __fmul_rn(w[LFM3_STATE_LEN], u));
    return lfm3_round(acc);
}

__device__ __forceinline__ void lfm3_load_weight(const __nv_bfloat16* __restrict__ weight, const unsigned int c,
                                                 float (&w)[LFM3_CONV_WIDTH]) {
#pragma unroll
    for (int k = 0; k < LFM3_CONV_WIDTH; ++k) {
        w[k] = lfm3_bf(weight[(unsigned long long)c * LFM3_CONV_WIDTH + k]);
    }
}

// The sequence that owns global token `t`: the largest s with cu[s] <= t (zero-length sequences
// are skipped because cu[s+1] == cu[s]). `cu` has num_seqs + 1 entries.
__device__ __forceinline__ unsigned int lfm3_find_seq(const int* __restrict__ cu, const unsigned int num_seqs,
                                                      const unsigned int t) {
    unsigned int lo = 0, hi = num_seqs;  // invariant: cu[lo] <= t < cu[hi]
    while (hi - lo > 1u) {
        const unsigned int mid = (lo + hi) >> 1;
        if ((unsigned int)cu[mid] <= t) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

__device__ __forceinline__ __nv_bfloat16* lfm3_state_of(unsigned long long const* __restrict__ table,
                                                        __nv_bfloat16* state0, const unsigned int s) {
    return table != nullptr ? reinterpret_cast<__nv_bfloat16*>(table[s]) : state0;
}

// ─────────────────────────────────────────────────────────────────────────────
// DECODE: one token per sequence. Row r of BCx and y belongs to sequence r of this launch, whose
// state pointer is s<r>. The window shifts in place: s' = [s1..s5, u].
//
// Grid (ceil(dim / 128), num_seqs, 1), block 128. num_seqs <= 16; the host launches again for
// more rows. The pointers are kernel parameters, so a captured graph bakes them exactly as it
// bakes every other argument (the layer reports graph_stale_on_new_sequence, so a slot's graphs
// are dropped when its sequence's state is freed).
// ─────────────────────────────────────────────────────────────────────────────
extern "C" __global__ void __launch_bounds__(LFM3_THREADS)
    lfm3_shortconv_decode_bf16(const __nv_bfloat16* __restrict__ bcx, const __nv_bfloat16* __restrict__ weight,
                               __nv_bfloat16* __restrict__ y, const unsigned int dim, const unsigned int bcx_stride,
                               const unsigned int y_stride, const unsigned int num_seqs, __nv_bfloat16* s0,
                               __nv_bfloat16* s1, __nv_bfloat16* s2, __nv_bfloat16* s3, __nv_bfloat16* s4,
                               __nv_bfloat16* s5, __nv_bfloat16* s6, __nv_bfloat16* s7, __nv_bfloat16* s8,
                               __nv_bfloat16* s9, __nv_bfloat16* s10, __nv_bfloat16* s11, __nv_bfloat16* s12,
                               __nv_bfloat16* s13, __nv_bfloat16* s14, __nv_bfloat16* s15) {
    const unsigned int c = blockIdx.x * LFM3_THREADS + threadIdx.x;
    const unsigned int r = blockIdx.y;
    if (c >= dim || r >= num_seqs || r >= LFM3_DECODE_SEQS) {
        return;
    }
    __nv_bfloat16* state;
    switch (r) {
        case 0: state = s0; break;
        case 1: state = s1; break;
        case 2: state = s2; break;
        case 3: state = s3; break;
        case 4: state = s4; break;
        case 5: state = s5; break;
        case 6: state = s6; break;
        case 7: state = s7; break;
        case 8: state = s8; break;
        case 9: state = s9; break;
        case 10: state = s10; break;
        case 11: state = s11; break;
        case 12: state = s12; break;
        case 13: state = s13; break;
        case 14: state = s14; break;
        default: state = s15; break;
    }
    if (state == nullptr) {
        return;
    }
    const __nv_bfloat16* row = bcx + (unsigned long long)r * bcx_stride;
    float w[LFM3_CONV_WIDTH];
    lfm3_load_weight(weight, c, w);
    __nv_bfloat16* sc = state + (unsigned long long)c * LFM3_STATE_LEN;
    float win[LFM3_STATE_LEN];
#pragma unroll
    for (int j = 0; j < LFM3_STATE_LEN; ++j) {
        win[j] = lfm3_bf(sc[j]);
    }
    const float u = lfm3_u(row, c, dim);
    const float v = lfm3_conv(w, win, u);
    const float cg = lfm3_bf(row[dim + c]);
    y[(unsigned long long)r * y_stride + c] = __float2bfloat16_rn(__fmul_rn(cg, v));
#pragma unroll
    for (int j = 0; j < LFM3_STATE_LEN - 1; ++j) {
        sc[j] = __float2bfloat16_rn(win[j + 1]);
    }
    sc[LFM3_STATE_LEN - 1] = __float2bfloat16_rn(u);
}

// ─────────────────────────────────────────────────────────────────────────────
// PREFILL y: `total_tokens` token-major rows, packed sequences. Sequence s owns global rows
// [cu[s], cu[s+1]); its state (read only here) is `state_table[s]`. With `cu == nullptr` the whole
// chunk is one sequence whose state is `state0` (the single-sequence form needs no device
// metadata). Each thread walks LFM3_TILE_T tokens of one channel with a six-value register window,
// preloaded from the tile's halo: the six previous tokens of the SAME sequence, or the state where
// the tile starts fewer than six tokens into its sequence. No value ever crosses a sequence
// boundary, so packed sequences cannot leak into each other.
//
// Grid (ceil(dim / 128), ceil(total_tokens / 64), 1), block 128.
// The final state is written by lfm3_shortconv_prefill_state_bf16, launched after this kernel on
// the same stream: writing it here would race with the tiles that still read the initial state.
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ void lfm3_fill_window(const __nv_bfloat16* __restrict__ bcx, const unsigned int bcx_stride,
                                                 const __nv_bfloat16* __restrict__ state, const unsigned int start,
                                                 const unsigned int local, const unsigned int c,
                                                 const unsigned int dim, float (&win)[LFM3_STATE_LEN]) {
#pragma unroll
    for (int j = 0; j < LFM3_STATE_LEN; ++j) {
        const int li = (int)local - LFM3_STATE_LEN + j;  // relative position of window slot j
        if (li >= 0) {
            win[j] = lfm3_u(bcx + (unsigned long long)(start + (unsigned int)li) * bcx_stride, c, dim);
        } else {
            win[j] = lfm3_bf(state[(unsigned long long)c * LFM3_STATE_LEN + (unsigned int)(li + LFM3_STATE_LEN)]);
        }
    }
}

extern "C" __global__ void __launch_bounds__(LFM3_THREADS)
    lfm3_shortconv_prefill_bf16(const __nv_bfloat16* __restrict__ bcx, const __nv_bfloat16* __restrict__ weight,
                                __nv_bfloat16* __restrict__ y, const int* __restrict__ cu,
                                unsigned long long const* __restrict__ state_table, __nv_bfloat16* state0,
                                const unsigned int total_tokens, const unsigned int num_seqs, const unsigned int dim,
                                const unsigned int bcx_stride, const unsigned int y_stride) {
    const unsigned int c = blockIdx.x * LFM3_THREADS + threadIdx.x;
    const unsigned int t0 = blockIdx.y * LFM3_TILE_T;
    if (c >= dim || t0 >= total_tokens) {
        return;
    }
    const unsigned int t1 = min(t0 + LFM3_TILE_T, total_tokens);
    float w[LFM3_CONV_WIDTH];
    lfm3_load_weight(weight, c, w);

    unsigned int s = 0, start = 0, end = total_tokens;
    if (cu != nullptr) {
        s = lfm3_find_seq(cu, num_seqs, t0);
        start = (unsigned int)cu[s];
        end = (unsigned int)cu[s + 1];
    }
    const __nv_bfloat16* state = lfm3_state_of(state_table, state0, s);
    float win[LFM3_STATE_LEN];
    lfm3_fill_window(bcx, bcx_stride, state, start, t0 - start, c, dim, win);

    for (unsigned int t = t0; t < t1; ++t) {
        if (t >= end) {
            // Next non-empty sequence: its window is its own initial state.
            do {
                ++s;
                start = (unsigned int)cu[s];
                end = (unsigned int)cu[s + 1];
            } while (t >= end);
            state = lfm3_state_of(state_table, state0, s);
            lfm3_fill_window(bcx, bcx_stride, state, start, 0u, c, dim, win);
        }
        const __nv_bfloat16* row = bcx + (unsigned long long)t * bcx_stride;
        const float u = lfm3_u(row, c, dim);
        const float v = lfm3_conv(w, win, u);
        y[(unsigned long long)t * y_stride + c] = __float2bfloat16_rn(__fmul_rn(lfm3_bf(row[dim + c]), v));
#pragma unroll
        for (int j = 0; j < LFM3_STATE_LEN - 1; ++j) {
            win[j] = win[j + 1];
        }
        win[LFM3_STATE_LEN - 1] = u;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PREFILL final state, in place. For sequence s with T = cu[s+1] - cu[s] chunk tokens, the extended
// input is E = [state(6), u(0..T)] and the new state is E[T .. T+6): the last six u values when
// T >= 6, otherwise the old window shifted left by T with the chunk's u appended (T < 6, including
// T = 0, which leaves the state unchanged). Each thread reads its six old values before writing,
// and no other thread touches that channel of that sequence, so in place is safe.
//
// Grid (ceil(dim / 128), num_seqs, 1), block 128. Same `cu == nullptr` single-sequence form.
// ─────────────────────────────────────────────────────────────────────────────
extern "C" __global__ void __launch_bounds__(LFM3_THREADS)
    lfm3_shortconv_prefill_state_bf16(const __nv_bfloat16* __restrict__ bcx, const int* __restrict__ cu,
                                      unsigned long long const* __restrict__ state_table, __nv_bfloat16* state0,
                                      const unsigned int total_tokens, const unsigned int num_seqs,
                                      const unsigned int dim, const unsigned int bcx_stride) {
    const unsigned int c = blockIdx.x * LFM3_THREADS + threadIdx.x;
    const unsigned int s = blockIdx.y;
    if (c >= dim || s >= num_seqs) {
        return;
    }
    unsigned int start = 0, end = total_tokens;
    if (cu != nullptr) {
        start = (unsigned int)cu[s];
        end = (unsigned int)cu[s + 1];
    }
    const unsigned int len = end - start;
    __nv_bfloat16* state = lfm3_state_of(state_table, state0, s);
    if (state == nullptr) {
        return;
    }
    __nv_bfloat16* sc = state + (unsigned long long)c * LFM3_STATE_LEN;
    float old_win[LFM3_STATE_LEN];
#pragma unroll
    for (int j = 0; j < LFM3_STATE_LEN; ++j) {
        old_win[j] = lfm3_bf(sc[j]);
    }
    float next[LFM3_STATE_LEN];
#pragma unroll
    for (int j = 0; j < LFM3_STATE_LEN; ++j) {
        const unsigned int e = len + (unsigned int)j;  // index into E
        if (e < LFM3_STATE_LEN) {
            next[j] = old_win[e];
        } else {
            next[j] = lfm3_u(bcx + (unsigned long long)(start + e - LFM3_STATE_LEN) * bcx_stride, c, dim);
        }
    }
#pragma unroll
    for (int j = 0; j < LFM3_STATE_LEN; ++j) {
        sc[j] = __float2bfloat16_rn(next[j]);
    }
}
