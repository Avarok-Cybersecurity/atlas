// SPDX-License-Identifier: AGPL-3.0-only

// DFlash2 grouped dynamic causal convolution (z-lab `GroupedDynamicCausalConv`).
//
// The DFlash2 drafter wraps each attention and MLP sublayer in a pair of
// causal 2-tap convolutions whose kernels are the sum of a learned static
// per-channel base and a per-position, per-group dynamic part predicted
// from the sublayer's normed input by `kernel_projection`:
//
//   out[t, c] = sum_{o=0}^{ks-1} (base[o, c] + dyn[t, o, c/G]) * x[t - o, c]
//
// with x[<0] = 0 (causal zero pad WITHIN the block: the drafter's forward
// only ever sees the gamma block, so row 0 has no predecessor and there is
// no conv state carried across steps — matches the reference exactly).
//
// Layouts (all BF16, row-major):
//   x    [n, hidden]                       the block rows
//   dyn  [n, 2 * ks * groups]              full `kernel_projection` output;
//                                          stage s tap o group g lives at
//                                          (s * ks + o) * groups + g
//   base [ks, hidden]                      caller pre-offsets to the stage:
//                                          stage s of the checkpoint's
//                                          [2, ks, hidden] tensor
//   out  [n, hidden]                       must not alias x
//
// `stage` selects the dyn half (0 = prepare/pre-sublayer, 1 = finish/
// post-sublayer). Accumulation in f32. n = gamma <= 16, hidden = 5120 for
// Qwen3.8-27B-DFlash2 — a trivially small launch; clarity over cleverness.

#include <cuda_bf16.h>

extern "C" __global__ void dflash2_grouped_dynamic_causal_conv_bf16(
    const __nv_bfloat16* __restrict__ x,     // [n, hidden]
    const __nv_bfloat16* __restrict__ dyn,   // [n, 2 * ks * groups]
    const __nv_bfloat16* __restrict__ base,  // [ks, hidden] (stage-offset)
    __nv_bfloat16* __restrict__ out,         // [n, hidden]
    unsigned int n,
    unsigned int hidden,
    unsigned int ks,          // conv_kernel_size (2)
    unsigned int group_size,  // channels per dynamic-kernel scalar (16)
    unsigned int stage,       // 0 = prepare, 1 = finish
    // Rows per independent block. The causal window RESETS at every
    // multiple of this, so a cross-sequence batch ([n_seq * gamma, hidden]
    // stacked seq-major) cannot convolve sequence i's first row with
    // sequence i-1's tail. Pass `n` (or 0) for a single block — then
    // t_local == t and the arithmetic is bit-identical to the pre-batch
    // kernel. This is the one silent-corruption hazard in stacking DFlash
    // draft blocks: every other op here is row-independent.
    unsigned int block_len
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * hidden) {
        return;
    }
    unsigned int t = i / hidden;
    unsigned int c = i % hidden;
    unsigned int groups = hidden / group_size;
    unsigned int g = c / group_size;
    unsigned int bl = (block_len == 0u) ? n : block_len;
    unsigned int t_local = t % bl;

    float acc = 0.0f;
    for (unsigned int o = 0; o < ks; ++o) {
        if (o > t_local) {
            break;  // causal zero pad: x[t - o] is outside this block
        }
        float b = __bfloat162float(base[o * hidden + c]);
        float d = __bfloat162float(
            dyn[t * (2u * ks * groups) + (stage * ks + o) * groups + g]);
        float v = __bfloat162float(x[(t - o) * hidden + c]);
        acc += (b + d) * v;
    }
    out[i] = __float2bfloat16(acc);
}
