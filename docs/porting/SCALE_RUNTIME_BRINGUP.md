# Atlas → Strix Halo (SCALE/gfx1151) — Runtime Bring-Up Status

**Date:** 2026-05-21  **Branch:** `port/amd-strix-halo`

## Summary

On native Ubuntu (the WSL `/dev/kfd` blocker is gone), `spark-server`
**builds, links, starts, loads the qwen3.6-27b-FP8 config, selects the
88-kernel gfx1151 target, and initialises cudarc.** It then fails at
`cuModuleLoadData` — the documented §5 "offload-bundle fork."

## Fixes landed this session (all in this commit)

1. **NCCL feature-split** — `nccl` is now a Cargo feature separate from
   `cuda` (default-on; NVIDIA byte-identical). SCALE builds use
   `--no-default-features --features cuda`. `init_nccl_comm` gains a
   cuda-without-nccl fail-fast variant. (spark-comm, spark-server)
2. **SCALE `cuGraphInstantiate`** — SCALE's libcuda exports the 3-arg
   `cuGraphInstantiate` (no `WithFlags` alias). `atlas_scale` cfg
   (emitted by spark-runtime/build.rs from `ATLAS_TARGET_HW=strix`)
   selects the right symbol. NVIDIA path unchanged.
3. **Binary-kernel registry** — `atlas-kernels` codegen previously
   stubbed `all_ptx_sets()` empty for any non-text-PTX backend (it
   conflated "binary kernels" with "Metal"). SCALE is binary-kernels +
   CUDA-API; `ComputeTarget::uses_cuda_module_api()` now drives a real
   registry for it. `TargetPtxSet.modules` is `&[u8]`; the runtime
   detects PTX-vs-ELF per blob.
4. **cudarc tolerant loader** — vendored cudarc 0.19.2
   (`vendor/cudarc`, `[patch.crates-io]`). cudarc eagerly resolves the
   whole CUDA driver API and hard-panics on the first missing symbol;
   SCALE implements a subset. All 483 `.expect("Expected symbol…")`
   now fall back to a panic-stub — symbols Atlas never calls are fine;
   NVIDIA (all present) is unaffected.
5. **SCALE device-link** — `ScaleTarget::compile` now device-links the
   relocatable (`--cuda-device-only -c`) into a loadable `ELF DYN`
   code object via `ld.lld -shared`.

## §5 blocker — `cuModuleLoadData` format

`cuModuleLoadData` returns `CUDA_ERROR_INVALID_VALUE` for **both**:
- the relocatable AMD-GPU ELF (`Type: REL`), and
- the `ld.lld -shared` linked code object (`Type: DYN`, 0 undefined syms).

Evidence:
- A normal SCALE host+device compile (`nvcc -c tiny.cu`) embeds **no**
  device code in the host `.o` (pure x86-64; only an `.init_array`
  + `.linker-options` registration hook). SCALE does not use the
  NVIDIA fatbin-in-host-object model.
- libredscale exports `cuModuleLoadData`, `cuModuleLoadDataEx`,
  `cuModuleLoadFatBinary` — but ships no docs and no format-hint
  strings.
- SCALE nvcc has `--gpu-bundle-output` (clang-offload-bundler).

**Hypothesis (needs Spectral confirmation):** SCALE's `cuModuleLoadData`
expects a clang-offload-bundler-wrapped blob (`__CLANG_OFFLOAD_BUNDLE__`
magic, gfx1151 target entry) and/or a specific AMD code-object version —
not a bare HSA code object. Next experiments: (a) wrap the linked code
object with `clang-offload-bundler`; (b) pin `--offload-arch` /
code-object-version; (c) ask Spectral what `cuModuleLoadData` consumes.

## Reproduce

Build: `bash /workspace/atlas/run-build.sh` (env: SCALE_HOME,
ATLAS_TARGET_HW=strix, ATLAS_TARGET_MODEL=qwen3.6-27b,
ATLAS_TARGET_QUANT=fp8, CUDARC_CUDA_VERSION=12080, CUDA_HOME +
LIBRARY_PATH → SCALE gfx1151 lib).
Serve: `bash /workspace/atlas/run-serve.sh`.
