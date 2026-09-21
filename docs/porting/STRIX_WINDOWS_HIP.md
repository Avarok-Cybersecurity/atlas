# Strix Halo on Windows (native HIP) — build status and test guide

Windows AMD is the **native-HIP** path (`ATLAS_TARGET_HW=strix-hip`, hipcc, gfx1151).
SCALE — Atlas's other AMD toolchain — is Linux-only, so HIP is the only conceivable
Windows AMD target.

> **Status: RUNTIME-VERIFIED** on a Framework Desktop (Ryzen AI MAX+ 395 /
> Radeon 8060S, gfx1151, Windows 11) serving `nvidia/Qwen3.6-27B-NVFP4`,
> 2026-08-13. Clean build to a served token in ~93 s; BFCL v3 seeded 196-entry
> subset **165/196 (84.2%)**.
> The Linux numbers in [`STRIX_NVFP4_MLPERF.md`](STRIX_NVFP4_MLPERF.md) still do
> **not** transfer; see "Why the Linux numbers won't reproduce" below.

### Qwen3.8-27B: builds and serves, but the runtime is NOT yet dependable

`unsloth/Qwen3.8-27B-NVFP4` was taken end to end on the same box on 2026-08-19:
95 kernels built (vision included), kernel target resolved as
`(gfx1151, qwen3.8-27b, nvfp4)`, server ready, coherent text, and 8/8 exact on a
vision OCR probe. **The port is sound. The ROCm runtime under it is not**, and
neither available SDK is a configuration to benchmark or recommend:

| | ROCm 6.4 (`25.Q3`) | ROCm 7.2 (`26.Q3`) |
|---|---|---|
| stability | 77 min, no fault | **hard GPU fault, 3/3 runs, within the hour** |
| degenerate outputs | ~23.5% | 0.5% |
| TTFT (median) | 8251 ms | 4463 ms |
| decode | 16.51 tok/s | 17.04 tok/s |

6.4 stays up but returns two junk tokens for roughly a quarter of requests, and
breaks determinism at temperature 0. 7.2 fixes almost all of that and halves
TTFT, then kills the HIP context under sustained load:

```
ERROR copy_logits_to_host: cuMemcpyDtoHAsync_v2 failed: status 719
ERROR hipErrorLaunchFailure (719) - the CUDA context is destroyed
ERROR free_sequence: ssm_pool.zero_slot(0): cuMemsetD8Async failed: status 719
```

Fault point drifts (7 / 42 / 54 min), so it is load-dependent, not input-dependent.
`ATLAS_NO_MTP_DRAFTER_CONTEXT=1` does **not** avoid it. Recovery needs only a
process restart, not a reboot — the GPU itself stays healthy.

> **If you benchmark this, read the result JSON before believing the score.**
> After the fault every request returns an instant 503 whose body is stored as a
> normal string result, so a dead GPU produces a plausible-looking accuracy table
> rather than an error. One run "finished" 196 samples in 11 minutes and scored
> 0.0 on nine of ten categories. Grep for `Error during inference` first.

No Windows accuracy number for 3.8 is publishable yet; a clean fault-free full
run does not exist on either ROCm. Accuracy measured so far also sits below the
Qwen3.6 baseline, most sharply on the `live_*` families.

## OK, how do I test it?

**You do not need to build anything.** Grab the prebuilt zip CI already publishes,
point one variable at it, run one script:

```powershell
# 1. Download `spark-windows-x86_64-amd-hip` from any green run of the release
#    matrix and unzip it. KEEP THE DLLs BESIDE spark.exe -- cudarc dlopens
#    nvcuda.dll from the exe's own directory, and that DLL imports the
#    versioned HIP runtime (amdhip64_6.dll on 6.x, amdhip64_7.dll on 7.x).

# 2. Weights (~21 GB):
hf download nvidia/Qwen3.6-27B-NVFP4 --local-dir "$env:USERPROFILE\models\Qwen3.6-27B-NVFP4"

# 3. Serve + smoke test:
$env:ATLAS_BIN = "C:\path\to\unzipped\spark.exe"
powershell -ExecutionPolicy Bypass -File scripts\strix-windows\first_run.ps1
```

That is the whole thing. You need the AMD driver and the weights — no MSVC, no HIP
SDK, no Rust, no clone. The script checks the box, starts the server with the
verified configuration, fires a completion at it, and writes the answer to
`first_run_smoke.log` beside the exe:

```
SMOKE OK  finish=length  tokens=63
TEXT: The three primary colors are **Red**, **Yellow**, and **Blue**. ...
```

If you got that, the port works on your hardware. `-Phase check` audits the box
and changes nothing, if you want to look before you leap.

**Building from source instead?** Leave `ATLAS_BIN` unset and the same script
checks the toolchain, repairs the kernel symlinks a Windows clone breaks, builds,
and serves. Everything it does is spelled out below.

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

To build it yourself you need the Windows HIP SDK and MSVC on PATH.
`scripts/strix-windows/first_run.ps1` does all of the below plus the symlink
repair; the raw commands are here for reference:

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

## Running it — VERIFIED

Hardware: a Strix Halo (Radeon 8060S, gfx1151) box on Windows 11 with a current
AMD Adrenalin driver. Model: `nvidia/Qwen3.6-27B-NVFP4`.

`scripts/strix-windows/first_run.ps1` runs exactly this; the values are reproduced
here so the reasoning has somewhere to live. **Three of them changed once this was
run on real hardware** — the pre-runtime guesses are called out inline, because
each one produces a failure that points somewhere else.

```powershell
$env:ATLAS_W4A16_DP4A = "1"; $env:ATLAS_FORCE_GLOBAL_GDN = "1"; $env:ATLAS_W4A16_VARIANT = "v1"
$env:ATLAS_KV_EXTERNAL_RESERVE_GB = "0"   # was 6 -- see below, 6 oversizes the KV pool
$env:ATLAS_SSM_TAIL_MIDCHUNK = "0"        # was 1 -- see below, 1 corrupts shared prefixes
$env:ATLAS_SSM_TAIL_PROTECT = "1"; $env:ATLAS_SSM_TAIL_LEASE_TTL = "128"
$env:ATLAS_MTP_GATE_REPROBE = "64"
.\spark.exe serve <snapshot> --model-name nvidia/Qwen3.6-27B-NVFP4 --port 8081 `
  --max-seq-len 65536 --gpu-memory-utilization 0.80 --kv-cache-dtype bf16 `
  --max-batch-size 1 --speculative --num-drafts 2 --mtp-quantization bf16 `
  --mtp-vocab 100000 --disable-tool-grammar true --enable-prefix-caching `
  --ssm-cache-slots 64 --ssm-checkpoint-interval 16 --disable-thinking
```

**`ATLAS_KV_EXTERNAL_RESERVE_GB` must be 0, not 6.** `hipMemGetInfo` is broken on
Windows HIP — `hipErrorInvalidValue` standalone, `free == 0` under a live context —
so `cuMemGetInfo_v2` now synthesises a truthful figure from tracked allocations.
That tracker reports Atlas-own bytes only, so `build.rs`'s co-tenant *discount*
double-counts and oversizes the KV pool: with 6 it allocated 11.3 GB of KV and
then died on a later 24 MB alloc. `0` fails `build.rs`'s `.filter(|gb| gb > 0.0)`
and falls through to the AUTO path (`baseline_free - free_now`), which is right
given the fixed shim.

**`ATLAS_SSM_TAIL_MIDCHUNK` must be 0, not 1.** Mid-chunk tail capture corrupts
*cross-request* SSM prefix reuse — the regression `scripts/mlperf-edge/ab_midchunk.sh`
already documents. Requests sharing a system-prompt prefix reuse each other's tail
snapshot; observed here as empty 1-token completions on 12/12 `live_multiple`
entries. It is a strict-`"0"` opt-out, **not** a presence flag: absent, or any
other value, leaves it ON.

**`--gpu-memory-utilization 0.80`, not 0.35.** It is a fraction of the total the
driver *reports* (76.9 GB), but the measured allocatable ceiling is ~63 GB, so it
cannot go much past 0.83. 0.80 gives a 61.5 GB budget: 40.3 GB pre-KV + 15.7 GB
reserve → 5.4 GB of KV = **89,232 tokens**.

Two more things a first run needs, neither of them obvious:

- **`--no-fast-load` used to be mandatory.** The fast weight loader needs
  `O_DIRECT`/`posix_fadvise`, is ON by default, and hard-errored on a non-Unix host
  before a single weight loaded. It now warns and falls back, so the flag is
  optional — passing it just silences the warning.
- **A Windows clone breaks `kernels/`.** 219 entries there are git symlinks that
  *chain* (`strix-hip → strix → gb10`), and Git for Windows defaults to
  `core.symlinks=false`, checking each out as a ~32 byte text file containing the
  link target path. `build.rs` now stops the build and names `core.symlinks`
  instead of letting hipcc complain about source it can't parse; the script
  repairs them from the git object store. Cloning with `-c core.symlinks=true`
  needs Developer Mode or an elevated shell and fails *silently* when it doesn't
  take. Once repaired, those 219 files show as modified in `git status` forever —
  real content where git expects a link blob. **Never commit them.**

The Linux doc's `ATLAS_MTP_DRAFTER_PREFILL=1` and `ATLAS_MTP_CARRY_DRAFTER=1` are
both **dropped on purpose**: `drafter_context` now ships prefill and carry ON by
default, and unset is the shipped configuration. The kill switch, if you need to
A/B against it, is `ATLAS_NO_MTP_DRAFTER_CONTEXT=1` (turns off both).

### The memory model is the thing most likely to bite

**The VGM warning this section used to carry did not reproduce.** On the Framework
Desktop, ~64 GB was allocatable with **no BIOS or Adrenalin tuning at all** —
default settings, 128 GB of system RAM. Anyone reading the old advice would have
gone hunting through firmware for a carve-out that was never the problem. What
*is* true:

- `hipInfo` reports **76.87 GB** exposed, but a `hipMalloc` ladder finds the real
  ceiling at **~63 GB**. `--gpu-memory-utilization` is a fraction of the number the
  driver *reports*, not of the number it will honour — hence 0.80, not 0.35, and
  not much above 0.83.
- Measured: at 11.3 GB of KV the process reached 63.3 GB tracked and `hipMalloc`
  started failing. ~5 GB of KV keeps peak near 57 GB with real margin.

There is still **no safety net**. When a weight allocation fails, Atlas falls back
to managed/UVM memory (`load_fns.rs:130`, whose log line literally says "paged via
Linux swap"). That path calls `hipMallocManaged`, which AMD does not support on
Windows — so a too-small budget is a **hard error**, not a slow degrade. If you
see `cuMemAllocManaged failed`, lower `--gpu-memory-utilization`.

### The DLL trap that costs the most time

`amdhip64_6.dll` loads its code-object manager **by plain name**. On a host whose
driver left an older `amd_comgr_2.dll` in System32, HIP binds the stale one and
*every* `hipModuleLoadData` fails as **`CUDA_ERROR_OUT_OF_MEMORY`** — which reads
as a KV-sizing bug and sends you tuning `--gpu-memory-utilization` for hours.
Measured here: System32's copy 109 MB with no version resource, against ROCm 6.4's
115.55 MB.

The build now stages `amd_comgr*`, `hiprtc*` and `hiprtc-builtins*` alongside
`amdhip64`, **and copies them next to the built binary** — `OUT_DIR` staging only
served the packaging step, so `cargo build` followed by running the exe still
picked up whatever was on the system path. If you unzip the CI artifact, keep the
whole folder together for the same reason.

### Why the Linux numbers won't reproduce

Do not expect the 7108 s / 25 tok/s Linux figures.

- Windows HIP SDK is ROCm 6.4-era (LLVM 19); the golden Linux run used ROCm 7.13.
  Different codegen for the same kernels.
- WDDM puts every kernel launch through the OS scheduler with no Linux-style
  persistent queues. Decode is small-kernel-heavy (TPOT ~39 ms at K=3, dozens of
  launches per token), which is the workload shape most exposed to per-launch
  overhead.

**Still the right thing to measure next.** The 2026-08-13 run establishes that the
port is correct and usable (84.2% accuracy, ~25 s/entry at batch 1) but does not
isolate per-launch overhead from everything else in that number. Nobody should
invest in kernel tuning on this target before that split exists.

## What the first run answered

The five questions this section used to ask, answered on 2026-08-13:

1. **Does it start and load all kernels?** Yes — **97**, matching CI and the Linux
   tree.
2. **Coherent output?** Yes. No NVFP4-corruption signature. BFCL v3 seeded
   196-entry subset scores **165/196 (84.2%)** — `simple` and `multiple` 0.967,
   `live_parallel` 0.625 lowest. Not comparable to the porting doc's 89.22, which
   is v4 on a different 167-sample subset; `bfcl-eval` ships v3 data.
3. **VGM / utilization?** No VGM tuning was needed; see the memory section above.
   `0.80` fits, `0.35` needlessly starves the KV pool.
4. **TTFT / TPOT?** ~25 s per BFCL entry end to end at `--max-batch-size 1`. The
   WDDM launch-overhead gap is real but did not stop the port from being usable —
   worth a dedicated measurement, still open.
5. **Errors?** No missing symbols; the 76-export shim was complete. The failures
   worth knowing are the ones above, plus the three filed separately.

## If you are running a benchmark

`bfcl-eval`'s local/OSS path hardcodes `ThreadPoolExecutor(max_workers=100)`
(`base_oss_handler.py:222`) and **ignores `--num-threads`**, which is honoured only
on the hosted-API path. Against `--max-batch-size 1` that is 100-way concurrency,
and the queue passes the 300 s request deadline about five minutes in. Because
Atlas stamps `timeout_at` at *arrival* but `request_start` at *prefill*, those
requests expire **while queued**, die on their first decode step, and come back as
**1 token with `finish_reason: "length"`** — indistinguishable from a legitimate
`max_tokens` stop. Every one scores as a wrong answer and nothing retries.

That invalidated a full 5-hour run at 75.5%. Pinning concurrency to 1 took request
timeouts from 1126/1294 to **0** and the score to 84.2%. Check the serve log for
`Request timeout` before trusting any number off this box; the behavioural half is
filed as #482.
