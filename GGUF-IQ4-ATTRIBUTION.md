# Keep-packed GGUF IQ4_XS / Q4_K / Q6_K on Ada (lane L19)

Atlas's Ada (sm_89) keep-packed GGUF 4-bit route serves the file's raw IQ4_XS, Q4_K and Q6_K
super-blocks without expanding them to BF16. Its kernels are thin entry points over the
llama.cpp CUDA quant engine Atlas already vendors, and its CPU decoders and launch helpers are
translations of llama.cpp code. This file records where every derived line came from, what
changed, and the license terms. The route is behind the `gguf_native_iq4` row, off on every
target.

## Upstream

- llama.cpp / ggml, https://github.com/ggml-org/llama.cpp. Repository license: MIT,
  "Copyright (c) 2023-2026 The ggml authors". Atlas carries the text in
  `LICENSES/llama.cpp-MIT.txt` and a section in `NOTICE`.
- The engine is the copy Atlas vendors in `kernels/gb10/qwen3.6-27b/nvfp4/q4k_vendor/`
  (`ggml-common.h`, `common.cuh`, `vecdotq.cuh`, `mma.cuh`, `mmq.cuh`, `quantize.cuh`,
  `quantize_impl.cuh` and the headers they include). Those files are included by relative
  path, never edited by this lane, and no file is added inside that directory: the notice
  lives at the repository root. Before this lane the
  vendored headers carried no MIT notice anywhere in the tree; the `NOTICE` section now covers
  them and the pre-existing kernels that include them.
- No upstream file was fetched or opened for this lane beyond the vendored copy: the CPU
  decoders follow the arithmetic of `dequantize_row_iq4_xs`, `dequantize_row_iq4_nl` and
  `dequantize_row_q5_K` (`ggml/src/ggml-quants.c`) as the vendored `ggml-common.h` block
  layouts and the existing Atlas Q5_K decoder (`dequant_cpu/blocks.rs`) define it.

## Files

| Atlas file | Derived from | Kind |
|---|---|---|
| `kernels/ada/common/kquant_q8_1_ada.cu` | `ggml-cuda/quantize.cu` (`quantize_q8_1`, `quantize_mmq_q8_1`) through the vendored `quantize_impl.cuh` worker | entry points over vendored code; the rows quantizer is the arithmetic of Atlas's `kquant_q8_1_rows_bf16` |
| `kernels/ada/common/kquant_mmvq_q4_ada.cu` | `ggml-cuda/mmvq.cu` (`mul_mat_vec_q`: lane-to-block split, `blocks_per_iter`, warp reduction) over the vendored `vec_dot_{iq4_xs,q4_K,q6_K}_q8_1` | translation, structure after Atlas's `kquant_moe.cu` |
| `kernels/ada/common/iq4_xs_mmq_ada.cu` | `ggml-cuda/mmq.cuh` (`mul_mat_q` conventional tiling around `mul_mat_q_process_tile`) | copy of Atlas's `q4k_mmq.cu` with the type switched |
| `kernels/ada/common/dequant_iq4xs_q5k_bf16_ada.cu` | `ggml-common.h` (`block_iq4_xs`, `block_iq4_nl`, `block_q5_K`, `kvalues_iq4nl`), `ggml-quants.c` (`dequantize_row_iq4_xs`, `_iq4_nl`, `_q5_K`), `ggml-cuda/convert.cu` (`dequantize_block_iq4_xs`) | translation, the style of Atlas's `dequant_gguf_bf16.cu` |
| `crates/spark-runtime/src/weights/gguf/dequant_cpu/iq4.rs` | `ggml-common.h`, `ggml-quants.c` (`dequantize_row_iq4_xs`, `_iq4_nl`) | translation to Rust |
| `crates/spark-runtime/src/weights/gguf/dequant_cpu/iq4_tests.rs` | the same loops, written a second way as the test oracle | translation to Rust |
| `crates/spark-model/src/port/gguf_q4/launch.rs` | `ggml-cuda/mmvq.cu`, `mmq.cuh`, `quantize.cu` (launch geometry, activation layouts and byte sizes) | translation to Rust launch helpers |
| `crates/spark-model/src/port/gguf_q4/reference.rs` | `ggml-cuda/quantize.cu` (the q8_1 block, the MMQ D4 / DS4 layouts), `ggml-quants.c` (`get_scale_min_k4`) | host mirror in Rust |

Every derived file starts with `// SPDX-License-Identifier: AGPL-3.0-only` and, under it, a
translation header: `// Portions translated from llama.cpp (ggml-org/llama.cpp), MIT License, Copyright
(c) 2023-2026 The ggml authors: <files>. Changes: <list>.` The other files of the route
(`weights/gguf/keep_packed_iq4.rs`, `names/qwen35_mtp.rs`, `port/gguf_q4/{mod.rs, w4linear.rs,
tests.rs}`, the row resolver and the microtest) are Atlas-native.

## Changes for Atlas (by Atlas contributors)

1. **Ada entry points.** The decode GEMVs take bf16 output and up to eight activation rows
   sharing every weight read, in two shapes (one block of four warps per output row with a
   fixed cross-warp order, llama.cpp's own shape for small batches; and one warp per row, the
   microtest's A/B arm). Row r's bits never depend on the row count, which the `W4Linear`
   contract requires for exact speculative verification.
2. **q8_1 activations of their own.** gb10's `kquant_q8_1_rows_bf16` lives in the DeepSeek model
   directory, which no Ada target compiles, so `kquant_q8_1_ada.cu` carries the same arithmetic
   under Ada names, plus the MMQ D4 (IQ4_XS, Q6_K) and DS4 (Q4_K) layouts with bf16 input.
3. **MMQ prefill for three types.** `atlas_{iq4_xs,q4_k,q6_k}_mmq128_{nc,wc}`: 128 x 128 tiles,
   eight warps, 57,856 bytes of dynamic shared memory for each type (asserted at compile time
   against the host constant).
4. **Dequant.** IQ4_XS, IQ4_NL and Q5_K to bf16, one CUDA block per GGUF block, bit-exact with
   the CPU decoder (same products, no fused multiply-add, round-to-nearest-even store); Q4_K and
   Q6_K reuse gb10's `dequant_gguf_bf16`.
5. **Rust.** The IQ4 decoders work one block at a time over a byte slice with little-endian
   field reads; the launch helpers wrap Atlas's `KernelLaunch`; the host mirrors exist only for
   the oracle.

## Oracle

- `dequant_cpu/iq4_tests.rs`: the decoders against a second implementation that walks blocks the
  way ggml's C loops do, bit for bit on random blocks (sub-normal scales included), closed-form
  blocks, and the loader entry points.
- `port/gguf_q4/tests.rs`: the byte arithmetic, every `W4Linear` entry point on the mock backend,
  and the float64 GEMV oracle with planted controls.
- `crates/spark-model/examples/iq4xs_gemv_microtest.rs` (GPU, deferred): dequant bitwise, the
  quantizer mirrors, the GEMV and MMQ within the float64 bound, row invariance, and the R2 / R3
  bandwidth bars of `docs/porting/qwen3.8-27b-ada.md`.
