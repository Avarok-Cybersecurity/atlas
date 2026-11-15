# Marlin W4A16 kernels for Ada (lane L18a)

Atlas's Ada (sm_89) Marlin kernels are a code copy of the Marlin mixed-precision GEMM as
vLLM ships it. This file records where every copied line came from, what changed, and the
license terms. The host side (configuration choice, workspace, GPTQ/AWQ repack, the
`W4Linear` implementation and the `marlin_w4a16` row) is lane L18b's, under
`crates/spark-model/src/port/marlin/`, and is a translation, not a copy.

## Upstream

- vLLM, https://github.com/vllm-project/vllm, commit `8a36f08` ("[Bugfix][KV Offload] Enable
  partial-tail offload for hybrid models under DCP (#58619)", 2026-11-12). Repository license:
  Apache License 2.0 (`LICENSE`). vLLM ships no `NOTICE` file.
- Copyright lines carried by the copied files:
  - `csrc/quantization/marlin/marlin_template.h` and `marlin.cu`: "Modified by Neural Magic",
    "Copyright (C) Marlin.2024 Elias Frantar", Apache-2.0, "Adapted from
    https://github.com/IST-DASLab/marlin".
  - `dequant.h`, `marlin_mma.h`, `marlin_dtypes.cuh`, `marlin.cuh`, `kernel.h`,
    `generate_kernels.py` and `csrc/core/scalar_type.hpp`: no per-file header
    (`generate_kernels.py` carries `SPDX-FileCopyrightText: Copyright contributors to the vLLM
    project`); the repository's Apache-2.0 applies. Atlas records them as "Copyright
    contributors to the vLLM project".
- The Marlin algorithm: Elias Frantar, Roberto L. Castro, Jiale Chen, Torsten Hoefler and Dan
  Alistarh, "MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language
  Models", arXiv:2408.11743; original kernels at https://github.com/IST-DASLab/marlin
  (Apache-2.0).

## Files

| Atlas file (`kernels/ada/common/`) | Upstream source (commit `8a36f08`) | Kind |
|---|---|---|
| `marlin_template_ada.cuh` | `csrc/quantization/marlin/marlin_template.h:1-1567`, `kernel.h:9-16` | copy, edited |
| `marlin_dequant_ada.cuh` | `csrc/quantization/marlin/dequant.h:1-216`, `473-474` | copy, pruned |
| `marlin_mma_ada.cuh` | `csrc/quantization/marlin/marlin_mma.h:1-148` | copy, edited |
| `marlin_dtypes_ada.cuh` | `csrc/quantization/marlin/marlin_dtypes.cuh:1-109` | copy, edited |
| `marlin_common_ada.cuh` | `csrc/quantization/marlin/marlin.cuh:1-152` | copy, edited |
| `marlin_scalar_type_ada.cuh` | `csrc/core/scalar_type.hpp:1-363` | copy, torch-free rewrite of two members |
| `marlin_ada.cu` | `generate_kernels.py:36-180` (instantiation rule), `kernel.h:9-16` (parameters), `marlin.cu:96-147` (shared-memory budget) | hand-written instantiation unit after the generator |

Each file starts with `// SPDX-License-Identifier: AGPL-3.0-only AND Apache-2.0`, the upstream
path and commit, the copyright lines above, and its own change list. The CPU test
`crates/avarok-kernels/tests/marlin_ada_entries.rs` checks those headers and the export table.

## Changes for Atlas (by Atlas contributors)

1. **Entry points.** Upstream `Marlin<...>` is a `__global__` template, instantiated by
   generated `sm80_kernel_*.cu` files and selected through a generated `kernel_selector.h`.
   Atlas binds kernels by PTX entry name, so `Marlin` became a `__device__ __forceinline__`
   template and `marlin_ada.cu` exports each instantiation through a one-line
   `extern "C" __global__` wrapper. The wrapper ABI is kernel.h's parameter list with the three
   `bool` flags as `int`. The 30 names and the full ABI are in the header of `marlin_ada.cu`.
2. **Pruned instantiation set.** BF16 activations, scales and output; weight types `kU4B8`
   (GPTQ int4) and `kU4` (AWQ int4 with int4 zero points); group size 128 (`group_blocks` 8);
   4 stages; `thread_m_blocks` 0.5 (m8), 1, 2, 3 and 4; the thread configurations upstream's
   generator keeps for each (three per batch size). Upstream builds also FP16, int8, FP8,
   NVFP4, MXFP4, MXFP8, per-channel and group 32/64 variants; none is built here.
   `thread_m_blocks` 3 is kept although the lane brief listed {0.5, 1, 2, 4}, because
   `marlin.cu:307` asks for it at m = 33 to 48.
3. **No relaxed constexpr.** The Atlas kernel build does not pass `--expt-relaxed-constexpr`,
   which vLLM's CMake does. Therefore:
   - every constexpr member of `ScalarType` is `__host__ __device__`;
   - `ScalarType::id()` and `from_id()` are explicit shifts and masks over the same fields in
     the same order and widths (the std::pair / std::tuple / std::apply fold is host-only),
     with `static_assert`s pinning the ids of `kU4`, `kU4B8` and `kBFloat16`;
   - scalar `k*Id` constants are added, and every type test in device code compares ids
     (`b_type_id == vllm::kU4Id`) instead of ScalarType objects, because device code may not
     reference a namespace-scope class object;
   - `div_ceil` is `__host__ __device__`.
4. **Torch-free scalar type.** `<torch/headeronly/util/Exception.h>` and `STD_TORCH_CHECK` are
   replaced by `assert`. The host-only `max()`, `min()` and `str()` members (std::variant,
   std::string) are dropped; no kernel calls them.
5. **Strict build hygiene.** Atlas compiles every kernel with `--Werror all-warnings`.
   - Locals the pruned set leaves unreferenced carry `[[maybe_unused]]`.
   - `slice_col`, `zp_gl_rd` and `zp_sh_rd` are zero-initialized; every path that reads them
     assigns them first, so no value read changes.
   - `mma_trans` drops a function-scope `float* c` that every branch shadowed.
   - Two FP16 dequant constants above INT_MAX are written as casts of unsigned literals,
     with the same bits.
   - `MarlinScalarType`'s template parameter is `vllm::ScalarTypeId` (int64_t) instead of
     `long`, which is 32 bits on Windows.
6. **sm_89 only.** The pre-sm_75 stub, the Turing (sm_75) m16n8k8 and FP16-accumulate
   paths and the pre-sm_80 cp.async fallbacks are removed. FP32 accumulation is what upstream
   uses on every architecture from sm_80 on.
7. **Compile-time shared-memory check.** `marlin_ada.cu` restates `get_kernel_cache_size`
   (`marlin.cu:114-147`) as a constexpr function and asserts that every configuration fits the
   sm_89 per-block opt-in (101,376 bytes) less vLLM's 512-byte margin. The largest one, m64
   with (64, 256, 256) and zero points, needs 69,120 bytes.
8. Includes point at the Atlas file names, include guards are `#pragma once`, and the unused
   `<iostream>` and `<cuda.h>` includes are gone.

Numerics are unchanged: the dequant bit tricks (`lop3`, `__hsub2`, `__hfma2`, `__hmul2`) are
explicit instructions, so the Ada tree's `--fmad=false` does not alter them. The kernel bodies
compute what vLLM's do. The numerics gate is L18b's GPU microtest
(`crates/spark-model/examples/marlin_ada_microtest.rs`).

## License

Apache License 2.0: `LICENSES/Apache-2.0.txt`. The copyright and attribution notices above
are reproduced in `NOTICE` (section "vLLM Marlin W4A16 kernels"). Atlas as a whole is
AGPL-3.0-only (`LICENSE`).
