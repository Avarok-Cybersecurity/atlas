// SPDX-License-Identifier: AGPL-3.0-only

// SiLU(gate) * up over ROW-STRIDED operands — the consumer of the fused
// dense-FFN gate+up projection (#927).
//
// HOPPER-OWNED (`[kernels] overrides`, an ADDITION — a new stem bringing an
// entry point gb10 does not declare), and not because the code is sm_90a: it
// is ordinary bf16 elementwise work. It lives here because the ONLY thing that
// dispatches it is the fused gate+up arm, which `kernels/hopper/HARDWARE.toml`
// alone declares (`[defaults] ffn_gateup_fused = true`) on an H100 receipt.
// gb10 and b200 declare that row FALSE, so putting the source in
// `kernels/gb10/common` would compile a kernel for two targets that can never
// launch it — and would make a new cross-hardware symlink in each of their
// mirrors, which `scripts/check_cross_hardware.py` rule S1 refuses outright.
// Same shape as `gdn_decode_hopper.cu`. If a GB10 receipt ever arrives, the
// move is this file into `kernels/gb10/common` plus the two mirror links, in
// the commit that carries the measurement.
//
// `moe_silu_mul` (kernels/gb10/common/moe_silu_mul.cu) indexes gate, up and
// output with ONE flat element index, which requires all three to be
// contiguous `[m, cols]`. The fused gate+up GEMM issues a single cuBLASLt call
// at `N = 2 * intermediate`, so its output is `[m, 2*inter]` with gate in
// columns `[0, inter)` and up in `[inter, 2*inter)` — the two halves of a row
// are `inter` elements apart, not `m * inter`. That is a stride, not a
// different arithmetic: this kernel is `moe_silu_mul`'s expression verbatim,
// reached through `row * stride + col`.
//
// LAYOUT AND WHY IT IS THIS ONE. The alternative was interleaving gate and up
// by tile so the consumer could stay flat. Splitting on N keeps three things
// simple that the interleave would complicate: the loader's fused weight is a
// straight device-to-device append of the two `[inter, K]` E4M3 blocks, its
// `[N/128, K/128]` FP32 block-scale grid is the same append one row-block
// wider, and each half stays addressable as an un-fused `Fp8Weight` VIEW so
// the non-fused dispatch rungs need no change at all. Coalescing survives it:
// a row half is `inter` contiguous BF16 (34 816 B at Qwen3.8-27B), so every
// warp's 128-byte segments are whole — only the jump BETWEEN rows differs.
//
// The grid is `[ceil(cols/256), rows, 1]` rather than a flat 1-D sweep so no
// thread pays an integer division to recover its row; `rows` at the decode
// widths this serves is 5..=16.

#include <cuda_bf16.h>

extern "C" __global__ void silu_mul_strided(
    const __nv_bfloat16* __restrict__ gate,    // [rows, in_stride], cols used
    const __nv_bfloat16* __restrict__ up,      // [rows, in_stride], cols used
    __nv_bfloat16* __restrict__ output,        // [rows, out_stride], cols used
    unsigned int rows,
    unsigned int cols,
    unsigned int in_stride,
    unsigned int out_stride
) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int r = blockIdx.y;
    if (c >= cols || r >= rows) return;

    const size_t in_base = (size_t)r * (size_t)in_stride + (size_t)c;
    float g = __bfloat162float(gate[in_base]);
    float u = __bfloat162float(up[in_base]);
    float sigmoid_g = 1.0f / (1.0f + __expf(-g));
    float result = g * sigmoid_g * u;
    output[(size_t)r * (size_t)out_stride + (size_t)c] = __float2bfloat16(result);
}
