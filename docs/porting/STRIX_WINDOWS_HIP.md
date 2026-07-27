# Strix Halo on Windows (native HIP) — build status and test guide

Windows AMD is the **native-HIP** path (`ATLAS_TARGET_HW=strix-hip`, hipcc, gfx1151).
SCALE — Atlas's other AMD toolchain — is Linux-only, so HIP is the only conceivable
Windows AMD target.

> **Status: COMPILE-VERIFIED, RUNTIME UNVERIFIED.**
> Every claim below about *building* is checked by CI on a hosted `windows-2022`
> runner. Nothing below about *running* has been executed on Windows — there is no
> Windows AMD machine in the fleet. Treat the serve section as a starting point to
> be corrected by the first person who runs it, not as a reproduction recipe.
> The Linux numbers in [`STRIX_NVFP4_MLPERF.md`](STRIX_NVFP4_MLPERF.md) do **not**
> transfer; see "Why the Linux numbers won't reproduce" below.

## What CI proves

The `windows-x86_64-amd-hip` row in `.github/release-matrix.json` builds the same
kernel triple the Linux golden run uses — `strix-hip` / `qwen3.6-27b` / `nvfp4` —
with the Windows HIP SDK (installer `25.Q3`, ROCm 6.4-era, LLVM 19), and packages a
zip containing `spark.exe`, `cuda.dll`, `nvcuda.dll` and `amdhip64_6.dll`.

Measured on run `30241443396` (2026-07-27), the first Windows build carrying the
full Strix kernel set:

```
atlas-kernels: compiled 97 kernels for target 0 (strix-hip, qwen3.6-27b, nvfp4)
atlas-kernels: dedup+parallel: 97/97 unique nvcc invocations
atlas-kernels: built Windows HIP runtime shim (cuda.dll/nvcuda.dll + import libs, 76 exports)
artifact spark-windows-x86_64-amd-hip, 13,612,158 bytes
```

**97 is the number to check.** main alone builds 91; the six kernels this branch
adds (`w4a16_gemv_dp4a`, `wht_bf16`, `moe_expert_gemv_dp4a`,
`dense_gemv_bf16_batch2`, `inferspark_prefill_paged_turbo{4,8}`) are exactly the
set main was missing, and 97 matches the Linux tree that is known to serve the
model. A lower count means kernels silently failed to resolve.

The row is `experimental: true` + `continue-on-error`, so a Windows-clang kernel
regression does not block a release. **That also means it can go red without
failing the aggregate run — check the job itself, not the run conclusion.**

## How it works

Two independent shims, which is why this is tractable at all:

- **Kernels.** `build.rs` mirrors every `.cu`/`.cuh` into `OUT_DIR/hip_mirror/`
  applying the 64-wide-wavefront mask widen, then compiles with
  `hipcc --offload-arch=gfx1151`. The NVIDIA `cp.async`/`mma.sync` bodies are
  `#if defined(__HIP_PLATFORM_AMD__)`-guarded, so the AMD arms compile — there is
  no tensor-core wall. Note ~86 of the `kernels/strix-hip` `.cu` files are git
  symlinks into `kernels/gb10`; they resolve correctly on the hosted runner.
- **Host.** Atlas reaches the driver through `cudarc`, which `dlopen`s
  `["cuda", "nvcuda"]`. `build_hip_shim_windows` compiles the `cu*`/`cudart` shims
  with hipcc into one `cuda.dll` linked against `amdhip64.lib`, exports via
  `dumpbin /SYMBOLS`, and copies it to `nvcuda.dll`. So `cuInit` → `hipInit`,
  `cuModuleLoadData` → `hipModuleLoadData`, and `spark` never knows.

Two Windows-only source accommodations, both isolated to the HIP tree so gb10 /
strix / metal stay byte-identical:

1. `hip/compat/atlas_hip_win_shims.h` — Windows HIP does not declare the CUDA
   mask-argument warp intrinsics (`__shfl_*_sync`, `__any_sync`, `__all_sync`,
   `__activemask`); Linux ROCm does. Force-included on Windows only.
2. `(__bf16)x` → `(__bf16)(float)x`. Windows' `__hip_bfloat16` has ~14 implicit
   `operator T()`, so the direct cast is ambiguous. Routing through `float` picks
   the unique `operator float()` and is numerically identical (bf16→float is exact).

`__dp4a` is **not** a problem: the AMD arm uses `__builtin_amdgcn_sudot4` /
`__builtin_amdgcn_perm` (plain clang builtins), with `__dp4a` only as the NVIDIA
fallback.

## Getting a binary

Download the `spark-windows-x86_64-amd-hip` artifact from any green run of the
release matrix — it runs on every PR, and on `workflow_dispatch` of `release.yml`
with `dry_run=true` (which is how run `30241443396` above was produced, and needs
no PR). Unzip; **keep the DLLs beside `spark.exe`** — `cudarc` `dlopen`s
`nvcuda.dll` from the exe's directory, and that DLL imports `amdhip64_6.dll`.

To build it yourself you need the Windows HIP SDK and MSVC on PATH:

```powershell
$env:ATLAS_TARGET_HW    = "strix-hip"
$env:ATLAS_TARGET_MODEL = "qwen3.6-27b"    # MANDATORY
$env:ATLAS_TARGET_QUANT = "nvfp4"          # MANDATORY
$env:ATLAS_HIPCC        = "C:\Program Files\AMD\ROCm\6.4\bin\hipcc.bin.exe"
$env:HIP_PATH           = "C:\Program Files\AMD\ROCm\6.4"
$env:CUDARC_CUDA_VERSION = "12080"
cargo build --release -p spark-server --target x86_64-pc-windows-msvc --no-default-features --features cuda
```

`ATLAS_TARGET_MODEL`/`QUANT` are not optional — the default target is a
`qwen3-next-80b` kernel dir that does not exist under `strix-hip`, and `build.rs`
panics resolving it. Build from **PowerShell, not Git Bash**: under bash on
Windows, Git's `/usr/bin` precedes MSVC on PATH and rustc invokes the coreutils
`link.exe` instead of the MSVC linker.

## Running it — UNVERIFIED

Hardware: a Strix Halo (Radeon 8060S, gfx1151) box on Windows 11 with a current
AMD Adrenalin driver. Model: `nvidia/Qwen3.6-27B-NVFP4`.

Start from the Linux serve line in `STRIX_NVFP4_MLPERF.md` and drop the Linux-only
parts (`LD_LIBRARY_PATH` is unnecessary — the DLLs sit beside the exe):

```powershell
$env:ATLAS_W4A16_DP4A = "1"; $env:ATLAS_FORCE_GLOBAL_GDN = "1"; $env:ATLAS_W4A16_VARIANT = "v1"
$env:ATLAS_KV_EXTERNAL_RESERVE_GB = "6"
$env:ATLAS_SSM_TAIL_MIDCHUNK = "1"; $env:ATLAS_SSM_TAIL_PROTECT = "1"
$env:ATLAS_MTP_GATE_REPROBE = "64"
.\spark.exe serve <snapshot> --model-name nvidia/Qwen3.6-27B-NVFP4 --port 8081 `
  --max-seq-len 65536 --gpu-memory-utilization 0.35 --kv-cache-dtype bf16 `
  --max-batch-size 1 --speculative --num-drafts 2 --mtp-quantization bf16 `
  --mtp-vocab 100000 --disable-tool-grammar true --enable-prefix-caching `
  --ssm-cache-slots 64 --ssm-checkpoint-interval 16 --disable-thinking
```

The Linux doc's `ATLAS_MTP_DRAFTER_PREFILL=1` and `ATLAS_MTP_CARRY_DRAFTER=1` are
both **dropped on purpose**: `drafter_context` now ships prefill and carry ON by
default, and unset is the shipped configuration. The kill switch, if you need to
A/B against it, is `ATLAS_NO_MTP_DRAFTER_CONTEXT=1` (turns off both).

### The memory model is the thing most likely to bite

On Linux the GPU pool is APU GTT (~60 GB of the 128 GB) and it grows on demand.
On Windows it is WDDM: `hipMalloc` sees the **Variable Graphics Memory** carve-out
you set in Adrenalin / BIOS, and it does **not** grow. Set VGM high *before*
serving; `--gpu-memory-utilization` is a fraction of what the driver exposes, so
the Linux value of 0.35 means something different here.

There is also **no safety net**. When a weight allocation fails, Atlas falls back
to managed/UVM memory (`load_fns.rs:130`, whose log line literally says "paged via
Linux swap"). That path calls `hipMallocManaged`, which AMD does not support on
Windows — so on Windows a too-small carve-out is a **hard error**, not a slow
degrade. If you see `cuMemAllocManaged failed`, the fix is more VGM, not more swap.

### Why the Linux numbers won't reproduce

Do not expect the 7108 s / 25 tok/s Linux figures.

- Windows HIP SDK is ROCm 6.4-era (LLVM 19); the golden Linux run used ROCm 7.13.
  Different codegen for the same kernels.
- WDDM puts every kernel launch through the OS scheduler with no Linux-style
  persistent queues. Decode is small-kernel-heavy (TPOT ~39 ms at K=3, dozens of
  launches per token), which is the workload shape most exposed to per-launch
  overhead.

**Measure launch overhead first.** If it is bad, no amount of kernel tuning
recovers it, and that result is worth knowing before anyone invests further.

## What to report

Most useful first:

1. Does `spark.exe` start and load all kernels? Expect **97** (see above).
   Fewer means kernels silently failed to resolve at load time.
2. Does it produce coherent output for a plain prompt? (Garbage output with a
   clean startup is the NVFP4-corruption signature, not a crash.)
3. `nvidia-smi`-equivalent: what does Adrenalin report for VGM, and what
   `--gpu-memory-utilization` actually fits?
4. TTFT and TPOT for a short prompt, so we can size the WDDM launch-overhead gap.
5. Any `cuMemAllocManaged failed` / missing-DLL / missing-symbol error verbatim —
   the shim exports 76 symbols and a missing one is a one-line fix.
