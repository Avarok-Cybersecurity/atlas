# ADR-0004: NVFP4 + FP8 as the primary quant formats

**Status:** Accepted
**Date:** 2026-04-17

## Context

Model weights at production sizes (35B–229B params) won't fit in 119 GB of
GB10 unified memory at BF16; we have to quantize. The format choices in
late-2025 were:

- **AWQ / GPTQ / W4A16 (Marlin)**: 4-bit weights, 16-bit activations,
  group-quantized. Mature, lots of pretrained checkpoints, pure-software
  dequant. Good for SM<sub>89</sub>; less ideal where native FP4 hardware
  exists.
- **FP8 (E4M3 / E5M2, block-scaled)**: 8-bit weights *and* activations
  ("native" FP8 path). Tensor-core supported on Hopper and Blackwell.
  Numerically wider than FP4; no reorder needed.
- **NVFP4 (E2M1 + FP8 group scales)**: NVIDIA's 4-bit format with
  group-of-16 FP8 scales. Tensor-core supported on Blackwell (sm_100,
  sm_120). Half the weight footprint of FP8; numerically tight enough
  for production use when calibrated.
- **MX4 / MX6**: open MX-format spec. Less hardware support today.

Atlas's target hardware is GB10 (Blackwell, sm_121). Critically:
- sm_121 has *cooperative-only* CUTLASS NVFP4 MoE GEMM (no Pingpong
  scheduler tile shapes — see `project_fp4_mma_gb10`), so we are not
  at the bleeding edge of NVFP4 perf. But the format itself is fine.
- sm_121 lacks the `cvt.rn.satfinite.e2m1x2.f32` PTX instruction that
  sm_120 has, requiring a software E2M1 conversion fallback (this is why
  vLLM v21 is fast on GB10 — same trick).
- Several model checkpoints we care about ship as NVFP4 (Sehyo's
  Qwen3.5-35B-A3B-NVFP4, MiniMax-M2.7-NVFP4, etc.). We don't get to
  pick the format; the customer-facing checkpoint dictates.

## Decision

Atlas commits to **NVFP4** and **FP8 block-scaled** as the primary quant
formats, with **BF16** as a raw-precision fallback for prototypes and
sanity checks. Everything else (W4A16 Marlin, AWQ, GPTQ, MX4) is
explicitly out of scope for the first wave; future ADRs may revisit.

Implementation:

- `crates/atlas-core/src/config.rs` recognizes the format from the
  model's `quantization_config` block in `config.json`.
- `crates/spark-model/src/weight_map/` per-format loaders produce a
  typed `QuantizedWeight` enum.
- Per-format CUDA kernels live under `kernels/gb10/<model>/<quant>/*.cu`
  (NVFP4) and `kernels/gb10/<model>/fp8/*.cu`. Shared kernels that work
  across models live at `kernels/gb10/<quant>/*.cu`.
- A `QuantFormat` trait (introduced in `project_modelopt_nvfp4_fix`)
  config-first dispatches between modelopt-NVFP4, compressed-tensors-NVFP4,
  and FP8 paths. Heuristic dispatch is kept only as a fallback.

## Consequences

**Better:**
- Two formats covers ~all currently shipping checkpoints we want to run.
- Format-specific kernels can be tuned aggressively without a generic
  abstraction tax.
- Adding a *third* format (e.g. MX4) is mechanical: new kernel directory,
  new `QuantFormat` impl, new `weight_map` loader. The pattern is set.

**Worse:**
- We have to ship two parallel kernel trees per model. NVFP4 work doesn't
  automatically benefit FP8 and vice versa.
- The matrix grows quadratically as new model families land — every
  family needs both an NVFP4 kernel pack and an FP8 kernel pack. The
  500-LoC discipline (ADR-0005) keeps each individual file manageable
  but doesn't reduce the count.

**New problems we created:**
- We are coupled to NVIDIA's NVFP4 PTX codegen. Hardware that lacks
  proper E2M1 conversion (today: GB10) needs software fallbacks, and we
  pay the cost of detecting and dispatching. This is a known multi-month
  drag (see `project_fp4_mma_gb10`).
- Calibration is on the publisher of the checkpoint, not us. A poorly
  calibrated NVFP4 checkpoint produces garbage; we can detect this only
  at the test-harness level.
