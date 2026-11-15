# Marlin W4A16 host for Ada (lane L18b)

Atlas's Marlin host (configuration choice, workspace, GPTQ/AWQ repack, the opt-in round-to-nearest
quantizer, the `W4Linear` implementation and the `marlin_w4a16` row) is a translation into Rust of
vLLM's Marlin host code. No upstream code is copied: C++, CUDA and Python were restated in Rust,
and the file headers name the upstream lines each Rust function follows. The kernels themselves are
lane L18a's code copy, recorded in `VLLM-MARLIN-ATTRIBUTION.md`.

## Upstream

- vLLM, https://github.com/vllm-project/vllm, commit `8a36f08`, Apache License 2.0 (`LICENSE`).
  vLLM ships no `NOTICE` file.
- Copyright lines: "Copyright contributors to the vLLM project"; the Marlin files
  (`marlin.cu`, the repack kernels) also carry "Copyright (C) Marlin.2024 Elias Frantar",
  Apache-2.0, "Adapted from https://github.com/IST-DASLab/marlin".
- Paper: Elias Frantar, Roberto L. Castro, Jiale Chen, Torsten Hoefler and Dan Alistarh, "MARLIN:
  Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models", arXiv:2408.11743.

## Files

| Atlas file (`crates/spark-model/`) | Upstream source (commit `8a36f08`) | Kind |
|---|---|---|
| `src/port/marlin/config.rs` | `csrc/quantization/marlin/marlin.cu:69-238` (thread configs, `get_scales_cache_size`, `get_kernel_cache_size`, `is_valid_config`, `determine_exec_config`), `:240-403` (the M split of `marlin_mm`), `marlin.cuh:21-31` (constants) | translation |
| `src/port/marlin/kernels.rs` | `csrc/quantization/marlin/generate_kernels.py:53-161` (instance set), `kernel.h:10-18` (parameter list) | translation |
| `src/port/marlin/repack.rs` | `csrc/quantization/marlin/gptq_marlin_repack.cu:14-184`, `awq_marlin_repack.cu:14-140`, `vllm/model_executor/layers/quantization/utils/marlin_utils.py:509-575`, `quant_utils.py:858-910` | translation; inverses new |
| `src/port/marlin/checkpoint.rs` | `quant_utils.py:834-941` (`pack_rows`, `pack_cols`, `unpack_cols`, `awq_pack`), `marlin_utils.py:577-601`, `csrc/quantization/marlin/marlin_template.h:320-321` (the act-order limit) | translation |
| `src/port/marlin/quantize.rs` | `quant_utils.py:716-812` (`quantize_weights`, the `uint4b8` grid) | translation, changed (below) |
| `src/port/marlin/reference.rs` | `csrc/quantization/marlin/dequant.h`, `marlin_template.h:102-145` (the dequant arithmetic) | CPU model of the kernel arithmetic |
| `src/port/marlin/weight.rs` | `marlin.cu:528-541` (the FP32 reduce buffer), `marlin_utils.py:407-415` (zeroed locks) | translation |
| `src/port/marlin/w4linear.rs` | `marlin.cu:240-403`, `marlin_utils.py:692-740` (`apply_gptq_marlin_linear`) | translation |
| `src/port/marlin/repack_tests.rs` | `vllm/model_executor/layers/quantization/utils/marlin_utils_test.py:27-89` (the permutation formulation used as an independent oracle) | translation |
| `src/port/marlin/{mod.rs, route.rs, tests.rs, quantize_tests.rs, w4linear_tests.rs, route_tests.rs}`, `src/layers/ops/port_rows/marlin_w4a16.rs` | none | Atlas-native |
| `examples/marlin_ada_microtest.rs`, `examples/marlin_ada_microtest/` | gate shapes after `tests/kernels/quantization/test_marlin_gemm.py` | Atlas-native (no code copied) |

## Changes (by Atlas contributors)

- C++ and Python to Rust; vLLM's `-1` sentinels become `Option`; the kernel selector
  (`get_marlin_kernel` returning `MarlinDefault`) becomes an availability callback over the pruned
  instance table of `marlin_ada.cu`.
- The CUDA repack kernels become one CPU loop over the same (k tile, n tile, warp, lane) indices;
  exact inverses (unrepack, unpermute, unpack) were added for the round-trip oracle and
  `dequant_bf16`.
- A row-invariant decode policy (one 128-thread configuration per weight, never the 8-row tile),
  because Atlas's `W4Linear` contract requires a row's bits not to depend on the row count; vLLM's
  per-M selector is kept for prompts.
- The RTN quantizer rounds the scale to BF16 before choosing codes, offers a per-group clip search,
  gives all-zero groups scale 0 and refuses non-finite inputs.
- Float zero points, the atomic-add reduce, bias, act-order and the Turing 2-stage path are not
  served (not instantiated); GPTQ `desc_act` checkpoints are refused with a reason.

## License

Apache License 2.0, full text in `LICENSES/Apache-2.0.txt`; the NOTICE entry is
"vLLM Marlin W4A16 host logic (translated to Rust)".
