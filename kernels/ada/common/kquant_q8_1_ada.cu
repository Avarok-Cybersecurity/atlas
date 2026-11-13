// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from llama.cpp (ggml-org/llama.cpp), MIT License, Copyright (c) 2023-2026 The ggml
// authors: ggml/src/ggml-cuda/quantize.cu (quantize_q8_1, quantize_mmq_q8_1), through the vendored copy in
// kernels/gb10/qwen3.6-27b/nvfp4/q4k_vendor/quantize_impl.cuh. Changes: bf16 input, one warp per 32-value
// block for the plain rows, Ada (sm_89) entry points of their own, by Atlas contributors.
//
// Responsibility: the q8_1 ACTIVATION quantizers of the keep-packed GGUF 4-bit route on Ada (port lane L19,
// row `gguf_native_iq4`). The weight GEMVs and GEMMs in kquant_mmvq_q4_ada.cu and iq4_xs_mmq_ada.cu read
// the activations in these layouts.
//
// WHY A FILE OF ITS OWN. The gb10 rows quantizer `kquant_q8_1_rows_bf16` lives in the DeepSeek model dir
// (kernels/gb10/deepseek-v4-flash/nvfp4/kquant_moe.cu), which neither Ada holo target nor the Ada 27B target
// compiles, and the MMQ quantizers live in gb10's qwen3.6-27b dir, also not compiled here. The math below is
// the same; only the entry names are new, so nothing that resolves the gb10 names can pick these up.
//
// Entry points (module `kquant_q8_1_ada`):
//   * `atlas_q8_1_rows_bf16_ada(x, vy, K, M)`: bf16 [M, K] -> block_q8_1 [M, K/32], the decode (M <= 8)
//     form the mmvq dots read. One warp per 32-value block: d = amax / 127, q = round(x / d),
//     ds = (d, sum x). K % 32 == 0. Grid ceil(M*K/32*32 / 128), block 128.
//   * `atlas_q8_1_mmq_d4_bf16_ada` / `atlas_q8_1_mmq_ds4_bf16_ada(x, vy, ne00, s01, ne0, ne1)`: bf16 [ne1, ne00]
//     -> block_q8_1_mmq in the D4 layout (IQ4_XS, Q6_K) or the DS4 layout (Q4_K), the prefill form the
//     MMQ tiles read (`mmq_get_q8_1_ds_layout`). Grid (ne1, ceil(ne0 / 512), 1), block 128.
//
// Every row is quantized independently of every other row, so a row's bytes never depend on M.
// `--fmad=false` is inherited from common/KERNEL.toml.

#include <cuda_bf16.h>
// Spelled from this directory: the vendored ggml headers live in gb10's 27B tree and are included, never
// edited (master plan 3.1, "include-only files"). nvcc resolves a quoted include against the directory
// of the path it is handed, which for this real (non-symlink) file is kernels/ada/common.
#include "../../gb10/qwen3.6-27b/nvfp4/q4k_vendor/mmq.cuh"
#include "../../gb10/qwen3.6-27b/nvfp4/q4k_vendor/quantize_impl.cuh"

// bf16 [M, K] -> block_q8_1 [M, K/32]. The same expression, reductions and rounding as gb10's
// `kquant_q8_1_rows_bf16`, so a byte comparison between the two is a valid oracle where both exist.
extern "C" __global__ void __launch_bounds__(128) atlas_q8_1_rows_bf16_ada(
        const __nv_bfloat16* __restrict__ x, void* __restrict__ vy,
        unsigned int K, unsigned int M) {
    const unsigned int nblk = K / QK8_1;
    const unsigned int gw   = (blockIdx.x * blockDim.x + threadIdx.x) / 32u;
    const unsigned int lane = threadIdx.x % 32u;
    if (gw >= M * nblk) return;
    const unsigned int m  = gw / nblk;
    const unsigned int ib = gw % nblk;
    const float xi = __bfloat162float(x[(size_t)m * K + (size_t)ib * QK8_1 + lane]);
    float amax = fabsf(xi);
    float sum  = xi;
    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : (int8_t)roundf(xi / d);
    block_q8_1* y = (block_q8_1*)vy;
    y[gw].qs[lane] = q;
    if (lane == 0) y[gw].ds = make_half2(d, sum);
}

// MMQ activation quantizers, bf16 in, in the layout each weight type's MMQ dot expects:
// IQ4_XS and Q6_K read D4 (scale only), Q4_K reads DS4 (scale and partial sum).
extern "C" __global__ void __launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE_MMQ) atlas_q8_1_mmq_d4_bf16_ada(
        const __nv_bfloat16* x, void* vy, long ne00, long s01, long ne0, int ne1) {
    quantize_mmq_q8_1_worker<MMQ_Q8_1_DS_LAYOUT_D4, __nv_bfloat16>(x, nullptr, vy, ne00, s01, 0, 0, ne0, ne1, 1);
}
extern "C" __global__ void __launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE_MMQ) atlas_q8_1_mmq_ds4_bf16_ada(
        const __nv_bfloat16* x, void* vy, long ne00, long s01, long ne0, int ne1) {
    quantize_mmq_q8_1_worker<MMQ_Q8_1_DS_LAYOUT_DS4, __nv_bfloat16>(x, nullptr, vy, ne00, s01, 0, 0, ne0, ne1, 1);
}
