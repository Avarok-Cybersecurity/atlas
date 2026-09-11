// SPDX-License-Identifier: AGPL-3.0-only
//
// VEC128 activation-scale layout adapter for the cuBLASLt block-scaled FP8 GEMM.
//
// WHY. `per_token_group_quant_fp8` writes `a_scale[m * (K/128) + kg]`, i.e.
// row-major `[M, K/128]` with the K-group index contiguous — the order
// `fp8_gemm_t_blockscaled` (the in-tree kernel) indexes. cuBLASLt's
// CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F wants the OTHER order. cuBLAS
// "Scaling factors layouts" (§3.1.4.5.1, cuBLAS 13.4) states the VEC128
// factors are "N-major for B with shape N x L" (and M-major for A), L =
// ceil(K/128) — the MN dimension is the contiguous one, not K. Feeding the
// row-major tensor is a permutation of the right values, which is why the
// H100 measurement (2026-09-11, native_fp8_ffn_w8a8_microtest, tip 5f78270dc)
// saw plausible-but-wrong output rather than garbage: cuBLASLt vs the kernel
// at 1140 TFLOP/s but rel_rms 1.1e-2 at M=64 and 7.7e-2-8.8e-2 at M=1193,
// ~33000 BF16 ULP, worsening with M because a bigger M mixes more scales.
//
// So the cuBLASLt arm transposes the quantizer's output once per GEMM into
// `[K/128, M_pad]`, which is what this kernel does. The quantizer's own output
// is left untouched — the in-tree kernel still reads it directly.
//
// Rows `M..M_pad` (the ceil16(M) pad cuBLASLt needs, because the docs require
// the gemm's M and N to be multiples of 4) are written as 0.0f: the phantom
// tokens' FP8 bytes are zeroed by the caller, and a zero scale keeps their
// contribution defined rather than reading whatever the scratch held.
//
// COST. M_pad x L floats — 1193x136x4 = 0.65 MB for the widest Qwen3.8-27B FFN
// shape, against ~89 MB of FP8 weight the same GEMM streams. A plain
// strided-read/coalesced-write pass is well under the noise; a tiled
// shared-memory transpose would optimize ~0.1% of the call.
//
// Grid: (ceil(M_pad/256), K/128, 1)  Block: (256, 1, 1)
// Index SSOT (Rust mirror + CPU unit test): `spark_runtime::cublaslt::scale_layout`.

extern "C" __global__ void fp8_act_scale_to_kmajor(
    const float* __restrict__ src,   // [M, L] FP32, row-major (quantizer output)
    float* __restrict__ dst,         // [L, M_pad] FP32 — cuBLASLt VEC128 B-scale
    unsigned int M,                  // real token rows
    unsigned int M_pad,              // padded token rows handed to cuBLASLt
    unsigned int L                   // K / 128 scale groups
) {
    const unsigned int m = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int l = blockIdx.y;
    if (m >= M_pad || l >= L) return;
    dst[(size_t)l * M_pad + m] = (m < M) ? src[(size_t)m * L + l] : 0.0f;
}
