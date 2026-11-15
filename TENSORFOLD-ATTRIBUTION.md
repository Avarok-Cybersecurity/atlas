# TensorFold affine 4-bit kernels for Ada (lanes L17a, L17b, L17c)

Atlas's Ada (sm_89) affine 4-bit route serves MLX-affine weights, quantized from a BF16 checkpoint at
load, on TensorFold's lane MMA GEMV and prompt GEMMs. The CUDA kernels are a code copy, adapted; the
Zig host code (the MLX quantizer, the tiled layout, the split rule and the launchers) is translated to
Rust. This file records where every copied or translated line came from, what changed, and the
license terms. Every route is off by default on every target: the rows that turn it on
(`dense_affine4_from_bf16`, `gdn_affine4_from_bf16`, `attn_affine4_from_bf16`,
`affine4_prefill_int8`) are lane L23a's, and the kernels exist only in the Ada build.

## Upstream

- TensorFold, https://github.com/ashhart/TensorFold, commit `3885dce` ("changelog: #3002 under
  Unreleased", 2026-11-12; a shallow clone, so per-file history was not available).
- Repository license: Apache License 2.0 (`LICENSE`) from TensorFold 0.6.0. Releases up to 0.5.0 were
  MIT, and code written before 0.6.0 keeps its MIT notice. The clone cannot tell which files predate
  0.6.0, so Atlas carries both notices for every copied or translated file (both are permissive and
  compatible with AGPL-3.0-only).
- The copied CUDA files carry no per-file license header; the root license applies. `qmm.cu`,
  `qmm_group.cu` and `qmm_prefill.cu` describe themselves as device code of TensorFold's own Python
  engine kernels. None of the separately licensed parts of TensorFold (`LICENSES/Kimi-K3*.txt`,
  `LFM-Open-1.0.txt`, `NVIDIA-OpenMDW-1.1.txt`, the CC BY-NC-ND weights in `THIRD_PARTY_NOTICES.md`,
  the ggml tables, the b12x-derived file, `web/`, `app/`) is used.
- The weight format is MLX's (https://github.com/ml-explore/mlx, Apple, MIT): credited in
  `CITATIONS.md`; no MLX source is copied.

### TensorFold's NOTICE (verbatim)

```
TensorFold
Copyright 2026 TensorFold contributors
https://github.com/ashhart/TensorFold

TensorFold is licensed under the Apache License, Version 2.0 (LICENSE) from 0.6.0.
Releases up to 0.5.0 were MIT-licensed, and code written before 0.6.0 keeps its
MIT notice in LICENSES/MIT.txt. Third-party code and its licenses are listed in
THIRD_PARTY_NOTICES.md.
```

### License texts

- Apache License 2.0: `LICENSES/Apache-2.0.txt`.
- The pre-0.6.0 MIT notice ("Copyright (c) 2026 TensorFold contributors"): `LICENSES/TensorFold-pre0.6-MIT.txt`,
  a byte copy of TensorFold's `LICENSES/MIT.txt`.
- The NOTICE section for TensorFold is in Atlas's `NOTICE`.

## Files

Kind: **copy** (CUDA copied and edited), **translation** (Zig or CUDA logic rewritten in Rust),
**design** (method only, no code). Every copied file's first line is
`// SPDX-License-Identifier: AGPL-3.0-only AND Apache-2.0`; every file names its upstream paths, the
copyright line, both licenses and its own change list right under the SPDX line.

### Lane L17a: the decode core

| Atlas file | Upstream source (commit `3885dce`) | Kind |
|---|---|---|
| `kernels/ada/common/affine_frag_ada.cuh` | `zig/kernels/cuda/common/qmm_frag.cuh:1-174`, `common/cp_async.cuh:1-67` | copy, pruned |
| `kernels/ada/common/affine4_lane_ada.cu` | `zig/kernels/cuda/lane_gemv.cu:1-346` (`gemv_items`, `gemv_store`, `gemv_kernel`, `gemv_split`, `tf_lane_gemv_split`, `_seg`, `_many`); the group-32 decode role of `qmm.cu:227-231` | copy, edited |
| `kernels/ada/common/affine_pack_ada.cu` | `zig/kernels/cuda/affine4_pack.cu:1-36` (`tf_pack_dense`, `tf_transpose_pad16`), `qwen35.cu:12-23, 184-190` (`group_sum`, `tf_q35_sums`); a GPU quantizer translated from `zig/src/core/affine4_host.zig:7-12, 40-66` (`f32of`, `bf16of`, `quantizeMlx`) | copy, edited; translation |
| `crates/spark-model/src/port/affine/weight.rs` | `zig/src/cuda/qlinear.zig:28-32` (`Affine4.layout`), `271-275` (lane shared-memory sizes), `308-314` (`splitK`) | translation |
| `crates/spark-model/src/port/affine/kernels.rs` | `zig/src/cuda/affine4_kernels.zig:107-133, 222-250, 291-318` (`pack`, `decode`, `gemvSeg`, `gemvSplit`, `gemv`) | translation |
| `crates/spark-model/src/port/affine/params.rs` | `zig/src/cuda/qlinear.zig:250-268` (`GemvPart`, `GemvManyPart`, `Seg`, `Segs`) | translation |
| `crates/spark-model/src/port/affine/quantize.rs` | `zig/src/cuda/affine4_kernels.zig:107-125` (`pack`) | translation |
| `crates/spark-model/src/port/affine/w4linear.rs` | `zig/src/cuda/affine4_kernels.zig:127-144` (`decode`, `decodeMany`) | translation |
| `crates/spark-model/src/port/affine/reference.rs` | `zig/src/core/affine4_host.zig:7-66`, `zig/kernels/cuda/affine4_pack.cu:5-36` | translation |
| `crates/spark-model/src/port/affine/reference_gemv.rs` | `docs/kernel-contract.md` gate 3 (the float64 bound) | design |
| `crates/spark-model/src/port/affine/tests.rs` | `zig/src/core/affine4_host.zig:68-83` (the quantizer and bf16 tests) | translation |
| `crates/spark-model/examples/affine_ada_microtest.rs` and `affine_ada_microtest/timing.rs` | `docs/kernel-contract.md:1-10` (planted controls, byte repeat), `docs/recipes/qwen3.8-27b.md:84-86` (distinct weight copies), `zig/src/cuda/qmmf.md` (rotated timing) | design |

### Lane L17b: the prompt GEMMs (as each file's header states)

| Atlas file | Upstream source | Kind |
|---|---|---|
| `kernels/ada/common/affine_prefill_ada.cu` | `zig/kernels/cuda/qmm_prefill.cu:1-301`, `w4a8_prefill.cu:1-172`, `w4a8_store.cuh:1-20`, the primitives of `common/qmm_frag.cuh` and `common/cp_async.cuh` | copy, edited |
| `kernels/ada/common/affine_swiglu_sums_ada.cu` | `zig/kernels/cuda/w4a8_quantize.cuh:1-51`, `w4a8_normalize.h:1-14`, with a new SwiGLU and group-sums fusion | copy, edited |
| `crates/spark-model/src/port/affine/prefill.rs`, `prefill_launch.rs`, `prefill_oracle.rs` | `zig/src/cuda/w4a8.zig:1-55`, `qlinear.zig:249-276`, the quantizer and epilogue of `w4a8_quantize.cuh`, `w4a8_prefill.cu:22-28, 125-147` | translation |

### Lane L17c: widths 2, 3, 5, 6 and 8 (as each file's header states)

| Atlas file | Upstream source | Kind |
|---|---|---|
| `kernels/ada/common/affine_n_ada.cu` | `zig/kernels/cuda/affine8.cu:1-385`, `affine4_pack.cu:29-36`, `qwen35.cu:12-23, 184-190`; a quantizer after `affine4_host.zig:41-66` | copy, edited |
| `kernels/ada/common/affine_codes_ada.cuh` | `zig/kernels/cuda/affine_codes.cuh:1-45`, primitives of `common/qmm_frag.cuh`, `common/cp_async.cuh` | copy, edited |
| `crates/spark-model/src/port/affine/widths.rs`, `widths/{kernels,layout,linear,reference,reference_gemv,tests_mock}.rs`, `widths_tests.rs` | `zig/src/cuda/qlinear.zig:35-47, 283-288, 308-314, 326-452`, `affine_codes.cuh`, `affine8.cu:319-385`, `affine4_host.zig:41-66` | translation |

## Changes for Atlas (lane L17a, by Atlas contributors)

1. **Entry points and names.** Namespaces `qmm_frag`, `tf_common` and `tf_lane_gemv` became
   `avarok_affine` and `avarok_affine_lane`; every kernel is an `extern "C"` wrapper named
   `affine4_lane_{gemv,split,seg,many}_g{32,64,128}`, `affine4_quantize_bf16_g{32,64,128}`,
   `affine4_pack_tiles`, `affine_transpose_pad16`, `affine_group_sums`, `affine4_dequant_bf16`.
2. **sm_89 only.** The PDL `grid_wait`/`grid_launch` calls and the sm_90 `sub.rn.bf16x2` branch of the
   code-pair decode are dropped; the `fma.rn.bf16x2` form gives the same bits (`t * 1 - 128` is exact for
   `t = 128 + q`). The fp_pairs include (e2m1) is dropped.
3. **Group sizes.** The lane body is instantiated at groups 32, 64 and 128 (upstream instantiates it at
   64; its group-32 decode tile is `qmm.cu`'s `qmm_kernel`, which is not copied: the same lane body at
   GS = 32 serves that role). The pack is generalized to GS/32 words a lane a group; at GS = 64 its
   index map and every output bit are upstream's (a CPU test pins this). `split_k` counts groups of the
   weight's own size; at 64 it is upstream's function.
4. **One body for the persistent GEMV.** `gemv_kernel` and `many_body` share one body taking
   `(block, blocks)`; the arithmetic is unchanged.
5. **Per-weight tickets.** A `Seg` carries its projection's own ticket array instead of an offset into
   one shared array, so every weight owns its split-K tickets (zeroed once at load; the kernel returns
   each to zero).
6. **The cluster decode tile is not ported.** On a shape with one K slice, decode runs the lane body
   (`_gemv`) where upstream runs `qmm_group`'s cluster kernel; both keep the group and slice order.
7. **The quantizer runs on the GPU.** `quantizeMlx` became a kernel, one warp a group: warp min and max,
   the same f32 operations through `__f*_rn`, round half away from zero, bf16 round to nearest even;
   the first bad group is reported through `atomicMin` instead of a return value; a `-0.0` edge is
   canonicalized to `+0.0` (no effect on any code, it makes the result independent of which zero the
   reduction keeps); an optional clip search (shrink the group's range about its midpoint by
   `1 - 0.05 i`, `i = 1..clip`, keep the least squared error of the stored dequantization) is an Atlas
   addition, off at `clip = 0`.
8. **New kernels.** `affine4_dequant_bf16` (tiled codes to row-major BF16, the fallback of a declined
   prompt GEMM and an oracle) is new.
9. **Host side.** The Zig launchers are Rust over `GpuBackend::launch_typed`, with handles resolved
   through `try_target_kernel` (no lookup on a target without the modules). The `W4Linear`
   implementation (`w4linear.rs`) carves the group sums and slice partials from the caller's scratch
   workspace and fuses a gate/up pair into one `_seg` launch.

## Changes for Atlas (lane L17c, by Atlas contributors)

1. **Names and ABI.** Namespace `tf_affine8` became `avarok_affine_n`; entries are `extern "C"`
   `affine{2,3,5,6,8}_gemv_ada`, `affine{2,3,5,6,8}_split_ada`, `affine_n_pack_ada`, `affine8_pack_ada`,
   `affine_n_transpose_pad16_ada`, `affine_n_sums_ada`, `affine_n_quantize_ada`, `affine_n_dequant_bf16_ada`.
   The GEMV entries take scalar and pointer arguments instead of a by-value `__grid_constant__` Part.
2. **sm_89 only.** The PDL `grid_wait`/`grid_launch` pair is dropped. The primitives the tiles use
   (cp.async, ldmatrix, bf16 mma.sync, swizzle) are copied into `affine_codes_ada.cuh` under the same
   Atlas namespace, so the file does not depend on `affine_frag_ada.cuh`.
3. **Not carried.** `tf_affine8_pack4` (2 to 4-bit codes onto the 4-bit tiles) and `tf_affine8_widen`
   have no Atlas caller and are not copied; the narrow-width prompt GEMM (`affine_prefill.cu`'s other
   widths) is not ported, so a 2/3/5/6/8-bit weight declines `prefill` and the caller's BF16 fallback runs.
4. **The quantizer runs on the GPU** for 2^bits - 1 levels, one warp a 64-group, with the 4-bit
   quantizer's `-0.0` canonicalization and clip search (midpoint shrink by `1 - 0.05 i`), so
   `--w4-clip-search` means one thing at every width.
5. **New kernel.** `affine_n_dequant_bf16_ada` (tiles to row-major BF16; the decline fallback and an
   oracle input).
6. **Host side.** Zig to Rust over `GpuBackend`/`KernelLaunch`, handles through `try_target_kernel`
   (no lookup on a target without the module); split-K scratch from the caller's `W4Scratch`, with the
   bit-identical unsplit form when it is missing.

## Verification

- CPU (`cargo test -p spark-model --lib port::affine`): the quantizer mirror (edge groups, the ported
  upstream tests, the clip search), the pack and its inverse (upstream's index map at group 64), the
  scale transpose, the dequantizer, the group sums, `split_k` and the shared-memory sizes against
  upstream's constants, the float64 GEMV bound against an f32 emulation of the lane chain, the mock
  launches and descriptors; a planted control for each gate.
- GPU (`examples/affine_ada_microtest.rs`, deferred to the card): every kernel byte for byte against
  the host mirror, the GEMV within the float64 bound at M = 1, 2, 4, 5, 8, 16, row invariance across M,
  the five-run repeat, pair and many forms against separate calls, each with a planted control (the
  decode gates run at group 64 on the 15 plan shapes and at groups 32 and 128 on three of them); the
  DRAM-bound GB/s rows against `fp8_gemv_ada`.
