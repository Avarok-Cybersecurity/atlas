# GB10 prefill static shared-memory failures — 2026-09-05

**The gate found a real static shared-memory assembly defect in the emitted
PTX. The proposed runtime dynamic-memory opt-in does not explain it.**
CUDA 13.0.88 rejects the affected entries for `sm_121f`; retaining these
failures is necessary. This establishes the assembly part of hypothesis (b),
but its suggestion that the runtime silently never uses them does not follow:
the source explicitly resolves and dispatches many of them. Driver JIT
acceptance and the identity of the running production binary remain unmeasured.

Scope: upstream `Avarok-Cybersecurity/atlas` commit
`567b5ebe7784ac3657a4ae97940f6783c8414393`, target
`gb10/qwen3.6-35b-a3b/nvfp4`. This is independent of PR #895's Hopper/B200
targets. That PR and the production container were not modified.

## Measured result

The original gate ledger was generated on `spark1` at
`2026-09-05T03:29:23Z`: 151/173 source files passed, with 22 failures after
successful PTX emission. The unchanged gate reproduced exactly those counts
against the upstream source above. Its positive and negative self-tests held.
All compilation ran on Spark 1 (`pidtom@192.168.50.125`) using
`/usr/local/cuda/bin/nvcc`, CUDA **13.0.88**, `nice -n 10`, four jobs and
`CUDA_VISIBLE_DEVICES=`. No CUDA context, GPU query, kernel launch or
production-container operation was performed.

The [machine-readable receipt](receipts/gb10-prefill-shared-memory-2026-09-05.json)
records source and gate hashes, commands, return codes, and **full assembler
output** for all 22 failed files. Each file was compiled again with the same
flags to retain its PTX and all diagnostics. The gate's `head()` function
keeps only the first error per file, which hid 20 additional BR=32 failures.
The full result is **42 rejected entries**, including all 22 `_64` entries.

## Why the opt-in cannot repair these allocations

The common paged implementation expands from
[`prefill_paged_compute.cuh`](../kernels/gb10/common/prefill_paged_compute.cuh).
Its BR=64 declarations are fixed-size `__shared__` arrays, as are the
asymmetric implementation in
[`prefill_paged_compute_asym.cuh`](../kernels/gb10/common/prefill_paged_compute_asym.cuh),
the contiguous [`inferspark_prefill.cu`](../kernels/gb10/common/inferspark_prefill.cu)
and [`inferspark_prefill_fp8kv.cu`](../kernels/gb10/common/inferspark_prefill_fp8kv.cu).
There is no `extern __shared__` workspace in these paths.

For HDIM=256, BC=32, PAD_KV=8 and PAD_P=8, the BR=64 BF16 layout is:

| Array | Shape and element size | Bytes |
|---|---|---:|
| Q | 64 × 264 × 2 | 33,792 |
| K, double buffered | 2 × 32 × 264 × 2 | 33,792 |
| V | 32 × 264 × 2 | 16,896 |
| P | 64 × 40 × 2 | 5,120 |
| m/l | 64 × 2 × 4 | 512 |
| **Total** | | **90,112 (`0x16000`)** |

The emitted PTX retains these as sized `.shared .b8` declarations. BR=32
uses 70,400 bytes (`0x11300`). Variant lookup tables add 16, 32, 64 or
96 bytes; indirect metadata adds 12 bytes. The non-batched FP8 paged variant
uses byte-sized K/V and PAD_KV=16, reducing BR=64 to 66,560 bytes (`0x10400`).

NVIDIA's [CUDA 13.0 Blackwell tuning guide](https://docs.nvidia.com/cuda/archive/13.0.0/blackwell-tuning-guide/index.html#unified-shared-memory-l1-texture-cache)
retains a 48 KiB limit for static allocations. Larger allocations must use
dynamic shared memory with explicit opt-in; see the
[CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#configuring-l1-shared-memory-balance).
Increasing the dynamic limit does not convert fixed arrays into a dynamic
workspace or legalize oversized static allocations.

CPU-only controls confirmed the distinction with the same compiler/assembler:

| Fixture | `nvcc --ptx` | `ptxas -arch=sm_121f` |
|---|---|---|
| Static 49,152 bytes | pass | pass |
| Static 49,156 bytes | pass | reject, 49,152 max |
| Static 90,112 bytes | pass | reject, 49,152 max |
| `extern __shared__` with a 90,112-byte intended workspace | pass | pass |

The dynamic fixture's assembly success does not validate its launch or
runtime opt-in. It only demonstrates why dynamic allocation is different.

## Rejected entry inventory

Every row names a source stem under `kernels/gb10/common/`; its BR=64 entry
is `<stem>_64`, and its BR=32 entry is the unsuffixed `<stem>`. All listed
numeric values are rejected static bytes; the limit is 49,152 bytes.

| Source stem | BR=64 rejected bytes | BR=32 rejected bytes |
|---|---:|---:|
| `inferspark_prefill` | 90,112 | 70,400 |
| `inferspark_prefill_fp8kv` | 90,112 | not emitted |
| `inferspark_prefill_paged` | 90,112 | 70,400 |
| `inferspark_prefill_paged_batched` | 90,112 | 70,400 |
| `inferspark_prefill_paged_bf16k_turbo2v` | 90,128 | 70,416 |
| `inferspark_prefill_paged_bf16k_turbo3v` | 90,144 | 70,432 |
| `inferspark_prefill_paged_bf16k_turbo4v` | 90,176 | 70,464 |
| `inferspark_prefill_paged_fp8` | 66,560 | passes: 46,336 bytes |
| `inferspark_prefill_paged_fp8_batched` | 90,112 | 70,400 |
| `inferspark_prefill_paged_fp8k_turbo2v` | 90,128 | 70,416 |
| `inferspark_prefill_paged_fp8k_turbo3v` | 90,144 | 70,432 |
| `inferspark_prefill_paged_fp8k_turbo4v` | 90,176 | 70,464 |
| `inferspark_prefill_paged_indirect` | 90,124 | 70,412 |
| `inferspark_prefill_paged_nvfp4` | 90,176 | 70,464 |
| `inferspark_prefill_paged_nvfp4_batched` | 90,176 | 70,464 |
| `inferspark_prefill_paged_turbo2` | 90,128 | 70,416 |
| `inferspark_prefill_paged_turbo3` | 90,144 | 70,432 |
| `inferspark_prefill_paged_turbo3k_turbo8v` | 90,144 | 70,432 |
| `inferspark_prefill_paged_turbo4` | 90,176 | 70,464 |
| `inferspark_prefill_paged_turbo4k_turbo3v` | 90,208 | 70,496 |
| `inferspark_prefill_paged_turbo4k_turbo8v` | 90,176 | 70,464 |
| `inferspark_prefill_paged_turbo8` | 90,112 | 70,400 |

The FP8 BR=32 entry assembles when selected alone with `ptxas --entry
inferspark_prefill_paged_fp8`; selecting its `_64` sibling still fails.
This was an isolation experiment. The gate still assembles whole modules.

## Shipped build and runtime trace

1. [`build.rs`](../crates/atlas-kernels/build.rs) delegates compilation to
   `ComputeTarget::compile`.
   [`NvidiaTarget::compile`](../crates/atlas-kernels/build_target.rs) invokes
   `nvcc --ptx -arch={arch} -O3`, merged flags and strict warnings. It does
   **not** invoke `ptxas`. [`build_codegen.rs`](../crates/atlas-kernels/build_codegen.rs)
   embeds the resulting complete blobs with `include_bytes!`.
2. Server [`preflight.rs`](../crates/spark-server/src/main_modules/serve_phases/preflight.rs)
   supplies `ptx_set.modules` to
   [`AtlasCudaBackend::new`](../crates/spark-runtime/src/cuda_backend.rs).
   [`AtlasRegistry::init`](../crates/atlas-core/src/registry.rs) loads every
   supplied blob via cudarc and raw `cuModuleLoadData`, propagating load
   errors. No entry pruning or failed-module fallback was found.
3. Searching `spark-runtime` alone misses the opt-in: it lives one layer
   below, in `AtlasRegistry::launch_on_stream` in `atlas-core`. The call to
   `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, ...)`
   happens only when `cfg.shared_mem_bytes > 48 * 1024`.
   [`KernelLaunch::new`](../crates/spark-runtime/src/kernel_args.rs) initializes
   that request to zero. The affected prefill wrappers in
   [`prefill_attn_main_a.rs`](../crates/spark-model/src/layers/ops/prefill_attn_main_a.rs),
   [`prefill_attn_main_b.rs`](../crates/spark-model/src/layers/ops/prefill_attn_main_b.rs)
   and their companion files do not call `.shared_mem(...)`.
4. The unrelated paged HDIM=512 implementation already uses
   `extern __shared__` in
   [`prefill_paged_compute_512.cuh`](../kernels/gb10/common/prefill_paged_compute_512.cuh)
   and explicitly requests dynamic memory in `prefill_attn_main_b.rs`.
   Its opt-in cannot apply to the fixed-array entries above.

### These are not all silently unused symbols

[`Qwen3AttentionLayer::new`](../crates/spark-model/src/layers/qwen3_attention/init.rs)
requires contiguous `_64` and BF16/FP8/NVFP4 paged `_64` handles through
`gpu.kernel(...)?`. Ordinary
[`paged dispatch`](../crates/spark-model/src/layers/qwen3_attention/prefill/paged_attn.rs)
selects BR=64 for chunks of at least 256 tokens. Turbo3/4/8 and asymmetric
formats also select BR=64, including short chunks; their missing-handle
checks return errors. Optional lookups are not proof of an alternate working
implementation. The model's declared head dimension is 256.

There are narrower unused-symbol findings: the Turbo2 field named `_64_k`
actually resolves its unsuffixed BR=32 entry and forces BR=32 at launch;
DFlash [`from_weights.rs`](../crates/spark-model/src/layers/dflash_head/from_weights.rs)
requests the unsuffixed indirect entry; no production lookup was found for
contiguous `inferspark_prefill_fp8kv_64`. Their modules still contain the
rejected entries, and Turbo2/indirect BR=32 also fail assembly.

Consequently, **offline assembly failure is measured; a claim that the
current production container never executes these entries is not**. Its
exact embedded artifacts, build overrides/cache provenance and driver-JIT
behavior were not inspected. The documented static limit and source trace
give no supported dynamic opt-in explanation for this discrepancy. A later
device investigation must use an idle, authorized GPU and the exact artifact;
this report does not claim a measured `cuModuleLoadData` failure.

## Disposition and reproduction

Keep all failures and the nonzero gate exit. Do not add an allowlist,
suppress the shared-memory errors, select only passing entries, or label
GB10 a green control. `--strict` matches the build's warning policy; the
gate's additional assembler stage still checks a condition the PTX-only
build does not. No assembler opt-in corresponding to the runtime attribute
is appropriate for these static declarations.

A kernel repair would require a correctly aligned dynamic workspace and
matching launch byte counts/opt-in, or a validated tile/storage redesign.
Changing only BR=64 leaves 20 rejected BR=32 siblings. That numerical/runtime
change requires device correctness testing and is not implemented here.

The gate is not present at the investigated upstream commit. The reproduction
used an unchanged copy from the original GB10 control directory, identified
by SHA-256 in the receipt. This report adds no dependency on or modification
to PR #895. With that gate available, run the original command in an isolated
source checkout on Spark 1:

```bash
CUDA_VISIBLE_DEVICES= nice -n 10 bash scripts/hopper_ptx_gate.sh \
  --hw gb10 --model qwen3.6-35b-a3b --jobs 4 --strict \
  --nvcc /usr/local/cuda/bin/nvcc --out /tmp/gb10-prefill-gate.json
```

To preserve all errors for a representative module, run on Spark 1 from the
source checkout (the final command is expected to return nonzero):

```bash
CUDA_VISIBLE_DEVICES= nice -n 10 /usr/local/cuda/bin/nvcc \
  --ptx -arch=sm_121f -O3 --fmad=false -DTQ_PLUS_SIGNS \
  --Werror all-warnings \
  -Ikernels/gb10/qwen3.6-35b-a3b/nvfp4 -Ikernels/gb10/common \
  kernels/gb10/common/inferspark_prefill_paged.cu -o /tmp/paged.ptx
CUDA_VISIBLE_DEVICES= nice -n 10 /usr/local/cuda/bin/ptxas \
  -arch=sm_121f -v /tmp/paged.ptx -o /dev/null
```

That reports both `inferspark_prefill_paged_64` at `0x16000` and
`inferspark_prefill_paged` at `0x11300`, against `0xc000` maximum.
