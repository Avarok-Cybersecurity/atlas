# EXL3 trellis weights: reader and kernels (port sprint lanes L20a, L20b)

Atlas can read ExLlamaV3's EXL3 trellis-quantized checkpoints and, on Ada (sm_89), serve their linear
layers on dedicated decode kernels. The format and its arithmetic are ExLlamaV3's; Atlas's code is
translated from TensorFold's independent Zig and CUDA implementation of that format. This file records
where every translated line came from, what changed, and the license terms. Nothing here runs on any
target unless lane L20b's row `weights_exl3` is on, and it is false on every target.

## Upstream

- **ExLlamaV3**, https://github.com/turboderp-org/exllamav3: the EXL3 format (trellis layout and
  bitstream, the `3inst`, `mcg` and `mul1` codebooks, half-integer bit widths, the tensor-core fragment
  order, the Hadamard-rotated inputs and outputs with the `suh` / `svh` scales). MIT License,
  "Copyright (c) 2025 Turboderp" (the holder line as TensorFold's `THIRD_PARTY_NOTICES.md` gives it; the
  ExLlamaV3 repository itself was not cloned for this sprint). No ExLlamaV3 source is copied.
- **TensorFold**, https://github.com/ashhart/TensorFold, commit `3885dce` (2026-11-12). Apache License
  2.0 from TensorFold 0.6.0; code written before 0.6.0 also MIT (see `TENSORFOLD-ATTRIBUTION.md`, lane
  L17a, for the dual-notice reasoning, which applies here unchanged). TensorFold's
  `THIRD_PARTY_NOTICES.md` ("CUDA" section) states that its EXL3 decoders and kernels are separate
  implementations of the format, checked bit for bit against ExLlamaV3's dequantization, under the
  Turboderp attribution. TensorFold's separately licensed parts (the kindlingai CC BY 4.0 EXLR8 format,
  the Kimi, LFM and NVIDIA licenses, the CC BY-NC-ND weights) are not used: EXLR8 is not EXL3 and is not
  read by Atlas.
- **QTIP** (Tseng et al., trellis quantization with incoherence processing) is the method family EXL3
  belongs to; cited in `CITATIONS.md`, no code.

### License texts

- ExLlamaV3 MIT text with the Turboderp copyright line: `LICENSES/ExLlamaV3-MIT.txt` (lane L20a).
- Apache License 2.0: `LICENSES/Apache-2.0.txt`.
- TensorFold's pre-0.6.0 MIT notice: `LICENSES/TensorFold-pre0.6-MIT.txt` (lane L17a).
- TensorFold's NOTICE text and the ExLlamaV3 entry are in Atlas's `NOTICE`.

## Files

Kind: **translation** (Zig logic rewritten in Rust), **copy** (CUDA copied and edited), **design**
(method only, no code). Every file names its upstream path and line range, both copyright lines and its
own change list right under its SPDX line.

### Lane L20a: the reader (`crates/spark-runtime/src/weights/exl3/`)

| Atlas file | Upstream source (TensorFold commit `3885dce`) | Kind |
|---|---|---|
| `mod.rs` | the module split of `zig/src/core/exl3_*.zig` | translation (structure) |
| `fp16.rs` | none (Atlas-native exact fp16 conversion the format's "round once" needs) | Atlas |
| `format.rs` | `zig/src/core/exl3_format.zig:1-134` (codebook constants, `Width`, `codebookValue`, `streamEnds`, `streamBit`, `tileStates`, `tilePlacement`, `unpack`, `unpackSigns`) | translation |
| `dequant.rs` | `zig/src/core/exl3_format.zig:136-187` (`rotate`, `rotateBlock`, `dequantize`) | translation |
| `layout.rs` | `zig/src/core/exl3_layout.zig:17-192` (`Group`, `Split`, `Span`, `Part`, `Layout`), `zig/src/core/exl3_prepare.zig:5-21` (`strips`) | translation |
| `load.rs` | `zig/src/core/exl3_load.zig:1-177` (markers, `Group`, `Summary`, `Scan`, `admit`, `scale`, `parts`) | translation |
| `config.rs` | `docs/recipes/exl3.md` ("What a family has to say", "Traps") | design |
| `shards.rs` | none (Atlas-native header-only safetensors reader; replaces TensorFold's `checkpoint.zig` and `safetensors.zig`, which were not read) | Atlas |
| `tensor.rs` | `zig/src/core/exl3_load.zig:15-24, 52-59` (the group and its parts), `exl3_layout.zig:69-93` (a rank's part) | translation |
| `tests.rs`, `fixture.rs` | none (tests and fixtures written from `docs/recipes/exl3.md`'s format paragraph) | Atlas |
| `layout_tests.rs` | `zig/src/core/exl3_layout.zig:196-250` (tests) | translation |
| `load_tests.rs` | `zig/src/core/exl3_load.zig:179-350` (tests) | translation |
| `oracle_tests.rs` | `zig/src/core/exl3_format.zig:189-338` (tests and the fixture's byte layout) | translation |

Changes made for Atlas, across the L20a files:

- Zig to Rust; errors are Rust `Result`s with reasons (`Refusal`, `LayoutError`, `anyhow`) instead of
  Zig error sets and debug asserts; every size and offset computed with checked arithmetic.
- The codebooks round from float64 to fp16 through `fp16.rs` (one round-to-nearest-even step from the
  full 53-bit significand). The `half` crate's `f16::from_f64` was not used: it rounds through `f32` on
  x86 with F16C and truncates the low mantissa word in its fallback, either of which can double-round.
- The 128-point Hadamard rotation is a fast Walsh-Hadamard butterfly (the same Sylvester matrix as
  TensorFold's naive product, a different FP64 summation order; `tests.rs` compares the two to 1e-12).
- Added: block-local dequantization (`dequantize_block`, a 128x128 block of `W` from its own tiles and
  scale slices), the factorized FP64 matvec (`y = ((((x * suh) H) W_q) H) * svh + bias`), the owned
  host tensor `Exl3Tensor` with its byte views and materialized tensor-parallel parts, the refusal
  reasons, the plain-tensor list, `config.json` parsing, and shard discovery through
  `model.safetensors.index.json`.
- Marker tensors: TensorFold reads a 4-byte scalar holding the multiplier; Atlas accepts it as I32 or
  U32 of rank 0 or 1 (TensorFold checks only the byte length).
- The oracle test reads TensorFold's `zig/tests/fixtures/exl3_oracle.bin` from outside the repository
  (`ATLAS_EXL3_ORACLE`, else `$HOME/research/TensorFold/...`) and skips when it is absent. The fixture is
  not vendored: it holds a block of a third-party checkpoint (`UnstableLlama/Qwen3.5-2B-exl3-8.00bpw`).

### Lane L20b: the kernels and `W4Linear` (owned and filled in by lane L20b)

Lane L20b's brief names these files; their exact upstream line ranges and change lists are recorded by
that lane (its `REPORT.md` and file headers) and added here through the coordinator:

| Atlas file | Upstream source (TensorFold commit `3885dce`) | Kind |
|---|---|---|
| `kernels/ada/common/exl3_decode_ada.cuh` | `zig/kernels/cuda/exl3_decode.cuh:1-157` (header: "After ExLlamaV3, MIT, Copyright (c) 2025 Turboderp; DeepSeek integration by @jayleaton"), the mma f16 wrapper of `common/qmm_frag.cuh:156-161` and `ld_nc32` of `common/cp_async.cuh:59-63` | copy |
| `kernels/ada/common/exl3_linear_ada.cu` | `zig/kernels/cuda/exl3_linear.cu:1-477` (`finish` 71-81, `load_step` / `step_lane_words` 83-101, `linear_body` 143-280, the instantiation macros 430-446) | copy |
| `kernels/ada/common/exl3_hadamard_ada.cu` | `zig/kernels/cuda/exl3_linear.cu:107-127` (`rot_in_body`, `tf_exl3_rot_in`); the BF16 dequant kernel is new, a CUDA form of `zig/src/core/exl3_format.zig:174-187` (`dequantize`) | copy (rotation), translation (dequant) |
| `crates/spark-model/src/port/exl3/format.rs` | `zig/src/core/exl3_load.zig:15-23, 70-90, 120-165` (admission), `zig/src/core/exl3_format.zig:17-28, 125-134` (`Width`, `unpackSigns`) | translation |
| `crates/spark-model/src/port/exl3/plan.rs` | `zig/src/cuda/exl3.zig:73-112, 202-217, 466-474` (`plan`, `strides`, `sharedBytes`, `zBytes`, `counterBytes`, the plan's tests) | translation |
| `crates/spark-model/src/port/exl3/kernels.rs` | `zig/src/cuda/exl3.zig:50-55, 219-284` (entry names, `matmul`'s two launches) | translation |
| `crates/spark-model/src/port/exl3/w4linear.rs` | `zig/src/cuda/exl3.zig:114-148, 219-284` (`Weight`, `planOf`, `matmul`) | translation |
| `crates/spark-model/src/port/exl3/reference.rs` | `zig/src/core/exl3_format.zig:5-123` (codebooks, `streamEnds`, `streamBit`, `tileStates`, `tilePlacement`, `unpack`) and `zig/kernels/cuda/exl3_decode.cuh:14-86, 129-131` (the lane extraction, mirrored on the host) | translation |
| `crates/spark-model/src/port/exl3/reference_math.rs` | `zig/src/core/exl3_format.zig:136-187` (`rotate`, `dequantize`), `zig/kernels/cuda/exl3_decode.cuh:138-155` (`fwht128`); the bit-exact fp32 kernel mirrors and the GEMV bound are new | translation |
| `crates/spark-model/src/port/exl3/tests.rs`, `tests_fixture.rs` | test design after `zig/src/core/exl3_projection_decode.zig:64-105`, `exl3_layout.zig:196-206`; the fixture walk after `exl3_format.zig:189-338` | idea (no code copied) |
| `crates/spark-model/examples/exl3_microtest.rs` (+ `exl3_microtest/`) | gate design after `docs/kernel-contract.md:1-10`, `zig/src/core/exl3_projection_gpu_test.zig`; group naming after `zig/src/core/exl3_load.zig:7-13, 120-165` | idea (no code copied) |
| `crates/spark-model/src/port/exl3/tensor.rs`, `tests_math.rs`, `tests_mock.rs`, `crates/spark-model/src/layers/ops/port_rows/weights_exl3.rs` | none | Atlas |

Changes for Atlas in the L20b files (each file header has its own list): namespace `tf_exl3` renamed
`avarok_exl3` and entry points renamed `exl3_ada_linear_<K2>_<CB>_<WK>`, `exl3_ada_rot_in_bf16`,
`exl3_ada_dequant_bf16_<K2>_<CB>`; plain kernel parameters instead of the by-value `LinearArgs` /
`RotInArgs` structs; bf16 input and output only and no bias (the host refuses biased groups, because the
`W4Linear` contract has no bias term); the PDL `grid_wait` / `grid_launch` calls dropped (sm_90 only);
only 4 and 8 warps instantiated (the warps the plan can pick); the mid-M, `linear_wc` and projection-group
kernels not ported (17-128 rows run `linear_body`, the same bits); the plan's block target is 4 x the SM
count instead of GB10's 192; a new shared-memory BF16 dequant kernel (the `W4Linear` prefill fallback);
the f64-to-fp16 rounding of the host reference is a one-step round to nearest even (not the `half` crate's
`from_f64`).

## Verification status

The reader and lane L20b's reference decoder both come from TensorFold, so a CPU oracle written in this
sprint is not independent of the translation. Two checks are: `oracle_tests.rs` runs the reader against
TensorFold's numpy-generated oracle (162 tiles of states over every codebook and width, codebook spot
values, and one real 8.00 bpw block bit for bit) when the fixture is present, and `tests.rs` checks every
codebook state against an independent f32 evaluation of the recipe's constants. The real-checkpoint scan
and decode (`turboderp/Qwen3.8-27B-exl3`) waits on the download (plan section 6, D6).
