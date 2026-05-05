# ADR-0002: Pure-Rust runtime, no PyTorch in the hot path

**Status:** Accepted
**Date:** 2026-04-17

## Context

The dominant LLM-inference stacks (vLLM, TensorRT-LLM, SGLang) are
Python-orchestrated with C++/CUDA kernels. They inherit PyTorch's allocator,
its dispatcher overhead, GIL contention, and a full PyTorch+CUDA runtime
image (>5 GB).

Atlas was built for a specific operator: NVIDIA DGX Spark GB10 (sm_121),
unified-memory Grace+Blackwell, 119 GB pooled. On that hardware we wanted:

- Tight control of allocator behavior (paged KV cache, pinned bounce
  buffers, NVMe swap); PyTorch's caching allocator fights us.
- Sub-millisecond scheduler-tick latency. PyTorch dispatcher + Python loop
  introduces tens of microseconds per op; over hundreds of ops per token
  this is a real fraction of decode latency.
- A small, auditable image — the production binary should be one static
  Rust executable plus PTX bytes plus libcuda.

Two candidate shapes:

1. **Rust scheduler + Python model code** (à la vLLM with a Rust front-end).
   Easier model porting (HuggingFace `transformers` exists), but you keep
   the entire PyTorch stack in the hot path.
2. **Pure Rust + cudarc, with model layers reimplemented as Rust+CUDA.**
   Higher up-front cost per model family, no Python in the inference loop,
   single static binary.

## Decision

Atlas is **pure Rust + cudarc** end-to-end. There is no PyTorch in the hot
path. Each supported model family has its own `TransformerLayer` impl in
`crates/spark-model/src/layers/<arch>/`, its own `WeightLoader` for
safetensors → typed-struct mapping, and its own per-quant CUDA kernels under
`kernels/<hw>/<model>/<quant>/*.cu`.

The shared "vocabulary" lives in `atlas-core` (`ModelConfig`, `LayerType`,
`KernelTarget`). The build script `atlas-kernels/build.rs` compiles `*.cu`
files to PTX via `nvcc`, embeds the PTX in the binary at compile time, and
loads it through cudarc's driver API at startup.

Python is allowed in:

- Test harnesses (`tests/run_all_models.py`, `tests/single_gpu_suite.py`).
- Reference implementations for byte-exact correctness checks.
- Examples and docs.

Python is **not** allowed in `crates/`.

## Consequences

**Better:**
- One static binary (~120 MB) versus a multi-GB PyTorch container.
- Allocator + scheduler are entirely under our control; KV-cache paging,
  NVMe swap, and pinned-host bounce buffers compose cleanly without
  fighting an upstream caching allocator.
- No GIL, no Python-side scheduling jitter. The scheduler's per-tick
  budget is dominated by GPU work, not host overhead.
- License story is simpler — no PyTorch (BSD-3) intermixed with our AGPL
  code in the same process.

**Worse:**
- Adding a new model family means writing a Rust `TransformerLayer`,
  a `WeightLoader`, and per-quant kernels. There is no `from_pretrained`
  shortcut. Time-to-first-bring-up is days, not hours.
- We don't get free upgrades from upstream PyTorch perf work (FlashAttn-3,
  new CUDA 13 kernels, etc.); we have to port them ourselves.
- Smaller talent pool: Rust+CUDA contributors are rarer than Python+CUDA.

**New problems we created:**
- A "reference impl" treadmill: every supported model needs an unquantized
  Python reference for byte-exact-diff debugging. We lean on
  HuggingFace `transformers` for these in the test harness.
- We have to track CUDA / cudarc / nvcc compatibility ourselves; no
  PyTorch wheel matrix to lean on.
