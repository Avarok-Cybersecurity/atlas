# Porting Atlas to AMD Strix Halo (gfx1151) via SCALE

Status: **working end-to-end** (branch `port/amd-strix-halo`). First target
model `qwen3.6-27b` served `Qwen/Qwen3.6-27B-FP8` generates **coherent output**
on an AMD Radeon 8060S (gfx1151 / Strix Halo, RDNA 3.5) at ~8.8 tok/s decode.
This guide is reproducible from a clean checkout on native Ubuntu.

> **Requires SCALE ≥ 1.7.1.** 1.7.0 SIGSEGVs in the HSA queue-create path on
> gfx1151 (wrong CWSR size for RDNA 3.5); 1.7.1 bundles ROCm 7.2.3 which reads
> `cwsr_size` from sysfs and fixes it. See `atlas-issues-found.md`.

SCALE (scale-lang.com, Spectral Compute) recompiles **unmodified CUDA** for
AMD GPUs. It is a drop-in `nvcc` shim (clang-19 based) that provides the CUDA
runtime/driver/math APIs for AMD. The Atlas philosophy here mirrors Spectral's
advice: start from "it compiles", add a `#if defined(__SCALE__)` shim only
where the compiler genuinely cannot hide a hardware difference, and send
Spectral clean repros for compiler defects.

> **Guard macro:** SCALE defines **`__SCALE__`** (and clang's `__AMDGCN__`)
> in the device pass — it does **NOT** define `__HIP_PLATFORM_AMD__`
> (verified empirically; Spectral's email claim of HIP arch macros is
> imprecise for `.cu` compilation). Use `#if defined(__SCALE__)` — it is
> never defined under nvcc, so the NVIDIA path stays byte-identical.

---

## 0. Quick start (verified working config)

For the impatient, on a native-Ubuntu Strix Halo box with the model already in
the HF cache and SCALE 1.7.1 unpacked at `$SCALE_HOME`:

```bash
# Build (SCALE compiles the unmodified CUDA kernels for gfx1151)
export SCALE_HOME=~/scale171/scale-1.7.1-Linux
export ATLAS_TARGET_HW=strix ATLAS_TARGET_MODEL=qwen3.6-27b ATLAS_TARGET_QUANT=nvfp4
export CUDA_PATH="$SCALE_HOME/targets/gfx1151" CUDA_HOME="$CUDA_PATH"
export PATH="$SCALE_HOME/targets/gfx1151/bin:/opt/rocm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib:$SCALE_HOME/targets/gfx1151/lib:$LD_LIBRARY_PATH"
export CUDARC_CUDA_VERSION=12080
cargo build --release -p spark-server --no-default-features --features cuda

# Serve — three runtime shims are required on gfx1151 (see §4):
export ATLAS_FORCE_GLOBAL_GDN=1     # route GDN prefill to the global-mem kernel (RDNA3.5 64KB LDS cap)
export ATLAS_W4A16_VARIANT=v1       # use the BF16-MMA NVFP4 GEMM (SCALE FP8-MMA encode is broken on gfx1151)
export ATLAS_NO_FP8_PREDEQUANT=1    # skip NVFP4->FP8 predequant (same broken-encode reason)
# SCALE libs FIRST so /opt/rocm cannot shadow the fixed libhsa-runtime64:
export LD_LIBRARY_PATH="$SCALE_HOME/targets/gfx1151/lib:$SCALE_HOME/lib"
export PATH="$SCALE_HOME/targets/gfx1151/bin:$PATH"
target/release/spark serve Qwen/Qwen3.6-27B-FP8 \
  --port 8081 --max-seq-len 4096 --gpu-memory-utilization 0.70 \
  --kv-cache-dtype bf16 --kv-high-precision-layers max --max-batch-size 4
```

A ready-made script lives at `serve-amd.sh` in the repo root. Sections 1–6
below explain each step, the SCALE mechanics, and why each shim is needed.

---

## 1. Toolchain

### 1.1 Get the right SCALE build

`pkgs.scale-lang.com/tar/` ships two lines:

| Tarball | Notes |
|---|---|
| `scale-free-1.4.2-amd64.tar.xz` | Free edition, **stale (Oct 2025)** — `targets/` has **no gfx1151**. Do not use for Strix. |
| `scale-1.7.0-amd64.tar.xz` | Has gfx1151 codegen but **SIGSEGVs at runtime** in HSA queue-create on gfx1151 (wrong CWSR size). Do not use. |
| **`scale-1.7.1-amd64.tar.xz`** | Current (2026), ~1.43 GB. Bundles ROCm 7.2.3 → fixes the gfx1151 queue-create crash. `targets/` includes gfx1151 (+ gfx1150/1152/1153, RDNA4 gfx1200/1201, CDNA gfx942/950). **Use this.** |

```bash
cd ~ && mkdir -p scale171 && cd scale171
curl -L --fail -o s171.tar.xz https://pkgs.scale-lang.com/tar/scale-1.7.1-amd64.tar.xz
tar -xf s171.tar.xz                      # → ~/scale171/scale-1.7.1-Linux
export SCALE_HOME=~/scale171/scale-1.7.1-Linux
```

`SCALE_HOME` is honored by the Atlas build (`find_scale_dir`). A SCALE root
contains `bin/scaleenv` and `targets/<arch>/bin/nvcc`.

### 1.2 Host toolchain prerequisites (bare Ubuntu / WSL)

SCALE compiles host code with the system C++ stdlib and its bundled HSA
runtime needs libnuma:

```bash
sudo apt-get update
sudo apt-get install -y build-essential libnuma1 libnuma-dev
```

Without `build-essential` → `cuda.h: fatal error: 'cstddef' file not found`.
Without `libnuma` → host link fails (`libhsakmt.so.1: undefined reference to
numa_*`). Device-only kernel compilation (`--cuda-device-only -c`) does **not**
need libnuma; full host executables do.

### 1.3 clangd

Spectral recommends their bundled `clangd` for LLM-assisted work (it actually
understands CUDA). Point your editor LSP at
`$SCALE_HOME/llvm/bin/clangd`.

---

## 2. SCALE mechanics (verified facts)

- **No `--ptx`.** SCALE rejects `--ptx`, `--genco`, `-fatbin`, `--emit-llvm`.
  It emits an **AMD GPU code object** (ELF relocatable), not PTX text. Atlas's
  device-compile flag is **`--cuda-device-only -c`**.
- **Target selection** = the per-arch toolchain dir
  `targets/gfx1151/bin/nvcc` (equivalent to `source bin/scaleenv gfx1151`
  without needing a sourced shell). `targets/gfxany` is *not* a generic JIT
  target — it still requires `-arch`.
- SCALE **bundles its own ROCm/HSA/HIP runtime** (`libhsa-runtime64`,
  `libamdhip64`, `libamd_comgr`, `librocblas`) — no system ROCm needed to
  *run*; still needs a kernel-driver path to the GPU.

### 2.1 Per-construct support on gfx1151 (probed)

| CUDA construct | SCALE 1.7.0 / gfx1151 | Action |
|---|---|---|
| BF16 `mma.sync m16n8k16` | ✅ compiles (MMA→MFMA lowered) | none |
| `__shfl_xor_sync` (32-lane, mask 0xffffffff) | ✅ compiles | none |
| `cp.async.cg` + commit/wait_group | ✅ compiles (treated as sync, batched) | none |
| `__launch_bounds__` | interpreted, scaled to hw | re-tune later |
| **FP8 `cvt.rn.satfinite.e4m3x2.f32`** | ❌ `__nv_cvt_floatraw_to_fp8` undefined | `#if __SCALE__` → `__nv_cvt_float_to_fp8` (exact) |
| **FP8 `mma.sync m16n8k32 .e4m3`** | ❌ "does not know how to codegen the PTX type: e4m3" | `#if __SCALE__` dequant→BF16 + BF16 MMA (GPU-verify) |
| **`__shared__` > 64 KB** | ❌ "local memory (N) exceeds limit (65536)" | hard RDNA3.5 LDS cap — AMD-only smem reduction |

### 2.2 Full kernel-set result (qwen3.6-27b, 92 .cu)

**82/92 compile clean** for gfx1151 with **zero source changes** — the entire
BF16 / SSM-GDN / MoE-routing / norm / rope / attention-decode / cp.async /
shfl bulk. Two failure classes only:

- **e4m3 (2 files):** `w4a16_gemm.cu`, `moe_w4a16_grouped_gemm.cu` — the
  FP8/NVFP4 tensor-core path (cvt + `m16n8k32 .e4m3` MMA). Exactly the
  tensor-core/quant landmine Spectral flagged ("matrix layouts differ; the
  permutation optimiser is not yet released; quant dtypes must be rewritten
  for AMD"). BF16 MMA *does* lower — only the **e4m3 PTX type** is missing.
- **LDS (8 `inferspark_prefill*` files):** `__shared__` footprint
  (e.g. `inferspark_prefill` = 70,400 B: Q 16896 + K×2 33792 + V 16896 +
  P 2560 + ml 256) exceeds RDNA3.5's hard **64 KB per-workgroup LDS cap**
  (NVIDIA Blackwell allows ~228 KB). **Not a compiler flag**
  (`-amdgpu-scratch-limit` is not a valid SCALE/LLVM option).

`reshape_and_cache_turbo.cu` (Class-A cvt, no MMA) is **fixed &
compile-verified** → 82/92, via the exact recipe in §4.

---

## 3. Build-system integration (done)

`crates/atlas-kernels/build_target.rs` already abstracts the compiler behind
the `ComputeTarget` trait. The AMD path is purely additive (NVIDIA untouched):

- `ScaleTarget` (`build_target.rs`): invokes
  `$SCALE_HOME/targets/<arch>/bin/nvcc --cuda-device-only -c -O3 <flags> src
  -o out.o`. `output_extension="o"`, `output_is_text=false`.
- `find_scale_dir()` (`build_codegen.rs`): `$SCALE_HOME`/`$SCALE_ROOT`, then
  conventional paths, then a shallow `scale*-Linux` scan. Fails fast (PCND).
- `resolve_compute_target()`: `"amd" | "rocm" | "scale"` → `ScaleTarget`.

Kernel tree (`kernels/strix/`, SSOT via **relative symlinks** to `gb10/` —
identical CUDA source, only `HARDWARE.toml` + `KERNEL.toml` are real files):

```
kernels/strix/
  HARDWARE.toml                 # vendor=amd, arch=gfx1151 (real)
  common/                       # → symlinks to ../../gb10/common/*
  qwen3.6-27b/
    MODEL.toml                  # → symlink to gb10
    nvfp4/
      *.cu                      # → symlinks to gb10
      KERNEL.toml               # real: extra_nvcc_flags=["-ffp-contract=off"]
```

`-ffp-contract=off` is the SCALE/clang spelling of nvcc `--fmad=false` (same
no-FMA-contraction intent, matters for NVFP4/FP8 precision parity).

### 3.1 Pending: binary codegen 3rd mode + runtime load (task #8)

`build_codegen.rs::generate_target_ptx_rs` today has only two modes (driven
by `output_is_text: bool`): text-PTX (NVIDIA — `include_str!` + full
`all_ptx_sets()`) and binary-Metal (Apple — `include_bytes!` +
`metallib_modules()` + **empty** `ptx_modules()`/`all_ptx_sets()` stubs).
SCALE is binary **but** CUDA-runtime-compatible (its driver API exposes
`cuModuleLoad`/`cuModuleLoadData`/`cuModuleLoadDataEx`/`cuModuleLoadFatBinary`
— verified in `targets/gfx1151/include`). So neither existing mode fits.

**Design (3rd mode — "binary, CUDA-runtime"):**
1. `generate_target_ptx_rs`: take an enum `{TextPtx, BinaryCudaRt, BinaryMetal}`
   instead of `output_is_text: bool`. `BinaryCudaRt` emits `include_bytes!`
   `&[u8]` consts **and the full `all_ptx_sets()` metadata block** (sampling/
   dflash/behavior/model_type — identical to the text branch; only the
   module-blob type differs).
2. `TargetPtxSet.modules`: today `Vec<(&'static str,&'static str)>`. Add a
   parallel `module_blobs: Vec<(&'static str,&'static [u8])>` (or make it an
   enum) so the text path is untouched (NVIDIA zero-risk) and the SCALE path
   carries bytes. Keep `KernelTarget`/metadata identical.
3. Runtime: add `KernelModule::from_binary(ctx,&[u8])` beside
   `from_ptx_src`. `atlas-core/src/registry.rs` already has the raw
   `cuModuleLoadData(module, image)` FFI — feed it the blob bytes (skip
   cudarc's `Ptx::from_src`, which is text-only). `cuda_backend.rs` selects
   blob-vs-text by build cfg / a generated flag.

**RESOLVED (probed 2026-05-17) — this is a runtime-MODEL fork, not just a
file-format question:**
- `--cuda-device-only`, `--cuda-device-only -c`, and `-fgpu-rdc` ALL emit
  `ELF 64-bit LSB **relocatable**, AMD GPU` — SCALE never produces a
  directly-loadable code object via the device-only path.
- A normal full build (`nvcc x.cu -o exe`) embeds the device code into the
  host binary via clang **offload bundling** (`clang-offload-bundler`,
  `clang-linker-wrapper` are present) and SCALE's runtime auto-registers it
  (the `cudaMalloc "no device"` test proved the device code was bundled &
  the runtime engaged — it failed only at GPU discovery, not at module load).
- SCALE's CUDA driver API *does* expose `cuModuleLoadData`/`Ex`
  (`cudaTypedefs.h`), but **what artifact its `cuModuleLoadData` accepts is
  unproven** without the AMD runtime live.

Atlas's model = embed PTX text, `cuModuleLoadData` at runtime (driver JIT),
launch by name via the registry. SCALE's native model = offload-bundle device
code into the binary, auto-registered, launched by C++ symbol. **Two paths:**
  1. **SCALE-native (lower risk):** AMD build compiles kernels into the
     binary via SCALE's normal flow; the registry resolves kernels by symbol
     instead of `cuModuleLoadData`(blob). Bigger atlas-core/spark-runtime
     change but uses SCALE exactly as designed.
  2. **Atlas-style:** device-link relocatables → a loadable code object,
     embed bytes, `cuModuleLoadData` it. Needs SCALE to support loading a
     hand-produced code object — unproven.
**Decision deferred until the AMD runtime is live (needs the Windows AMD
ROCm-on-WSL driver, §5) AND Spectral confirms the intended model for a
CUDA-driver-API engine that loads modules dynamically.** Not coded
speculatively (would risk an unvalidatable architectural mis-build).

---

## 4. Remaining work (precisely scoped)

### 4.1 e4m3 — `w4a16_gemm.cu`, `moe_w4a16_grouped_gemm.cu`

**(a) cvt — exact recipe, proven on `reshape_and_cache_turbo.cu`:** SCALE
provides NVIDIA's own `__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3)`
in `cuda_fp8.h` (and `__nv_cvt_fp8_to_halfraw` for decode). Wrap each
`cvt.rn.satfinite.e4m3x2.f32` site:

```cpp
#if defined(__SCALE__)
    // NVIDIA's documented intrinsic — numerically exact, not an approximation.
    unsigned char lo = __nv_cvt_float_to_fp8(b_lo, __NV_SATFINITE, __NV_E4M3);
    unsigned char hi = __nv_cvt_float_to_fp8(a_hi, __NV_SATFINITE, __NV_E4M3);
    d = (unsigned short)(((unsigned short)hi << 8) | lo);
#else
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0,%1,%2;" : "=h"(d) : "f"(a_hi), "f"(b_lo));
#endif
```

~13/16 cvt sites per file, many inside macros — mechanical but must keep the
`#else` byte-identical (shared NVIDIA kernels; `feedback_fp8_backward_compat`).

**(b) e4m3 MMA — `mma.sync m16n8k32 .e4m3` has no gfx1151 codegen.** Replace
with dequant-e4m3→BF16 + BF16 `m16n8k16` MMA (K=32 = 2×k16 accumulated; BF16
MMA is proven to compile). **This rewrite changes numerics-bearing code and
MUST be verified on a GPU before trust** (a wrong thread/fragment mapping =
silently wrong matmul — the exact "faulty implementation" to avoid). Deferred
to a GPU-equipped session; not hand-ported blind. Building blocks: §4.1(a)
encode + `__nv_cvt_fp8_to_halfraw` decode.

### 4.2 LDS > 64 KB — 8 `inferspark_prefill*` files

RDNA3.5 hard 64 KB/workgroup LDS cap. Real fix = reduce `__shared__` under
`#if defined(__SCALE__)`: single-buffer `smem_K` (saves 16,896 B →
`inferspark_prefill` 70,400 → 53,504 ≤ 65,536) or smaller `BR`/`BC` tile.
Algorithmic + perf-affecting → **GPU numeric verification required**;
deferred, not faked.

Parallel: Spectral repro `scripts/scale-probe/e4m3_mma_only_probe.cu` for
native e4m3 MMA codegen on gfx1151 — if SCALE ships it, §4.1(b) becomes a
no-op. (Draft the email; do not auto-send.)

---

## 5. Build, deploy, run

Verified on native Ubuntu (kernel 6.17.0-oem), gfx1151, SCALE 1.7.1. The repo
ships `build-amd.sh` and `serve-amd.sh` that wrap exactly the commands below.

```bash
# Build — SCALE_HOME set, kernels compiled for gfx1151:
export SCALE_HOME=~/scale171/scale-1.7.1-Linux
export ATLAS_TARGET_HW=strix ATLAS_TARGET_MODEL=qwen3.6-27b ATLAS_TARGET_QUANT=nvfp4
export CUDA_PATH="$SCALE_HOME/targets/gfx1151" CUDA_HOME="$CUDA_PATH"
export PATH="$SCALE_HOME/targets/gfx1151/bin:/opt/rocm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib:$SCALE_HOME/targets/gfx1151/lib:$LD_LIBRARY_PATH"
export CUDARC_CUDA_VERSION=12080
rm -rf target/release/build/atlas-kernels-*      # stale-cache guard on .cu change
cargo build --release -p spark-server --no-default-features --features cuda

# Serve — gfx1151 shims (§4) + SCALE libs first so /opt/rocm can't shadow libhsa:
export ATLAS_FORCE_GLOBAL_GDN=1 ATLAS_W4A16_VARIANT=v1 ATLAS_NO_FP8_PREDEQUANT=1
export LD_LIBRARY_PATH="$SCALE_HOME/targets/gfx1151/lib:$SCALE_HOME/lib"
target/release/spark serve Qwen/Qwen3.6-27B-FP8 \
  --port 8081 --max-seq-len 4096 --gpu-memory-utilization 0.70 \
  --kv-cache-dtype bf16 --kv-high-precision-layers max --max-batch-size 4
```

**Memory note (61 GB unified).** Strix shares one LPDDR5X pool between host and
GPU (the `rocm-smi` VRAM carveout is only ~512 MB; GPU buffers live in GTT =
system RAM). The KV sizer fills greedily to `--gpu-memory-utilization` and
CUDA-graph capture allocates on top during warmup, so keep util at **0.70** for
the 27B model; raising it triggers the OOM watchdog mid-warmup. `--max-seq-len`
barely affects KV (it grabs all free memory regardless); util is the real lever.
Restoring 16384 ctx needs the arena/graph accounting work noted in §4.

### 5.1 MTP / speculative decoding

`MODEL.toml mtp_layers` is **dead config** (read by no Rust code). Native MTP
loads automatically when the HF checkpoint has the head
(`config.mtp_num_hidden_layers>0`, true for Qwen3.6-27B-FP8) but only
**activates with the `--speculative` serve flag** (`--num-drafts 1` = K=2).
Keep `--speculative` **off on Strix until MTP is validated on a known-good
CUDA box** (the MTP head itself uses FP8 e4m3 projections → also needs the §4
`#ifdef`).

---

## 6. Verification

1. Phase-0 probes (`scripts/scale-probe/*.cu`) — done; results in §2.1.
2. SCALE compile sweep green for the full qwen3.6-27b kernel set on gfx1151.
3. `cargo build` (strix target) exits 0.
4. Non-spec correctness vs the GB10 baseline (greedy/temp 0) —
   `bench/qwen36_correctness.py`, `tests/single_gpu_suite.py`.
5. MTP (`--speculative`) only after a GB10 validation confirms parity.
6. TTFT/decode tok/s vs GB10 — `bench/qwen36_ttft.py`.

---

## 7. Spectral feedback bundle

Clean repros for compiler defects (send via the existing email thread —
do not auto-send):
- `scripts/scale-probe/e4m3_mma_only_probe.cu` — native e4m3 `m16n8k32` MMA
  codegen on gfx1151 ("does not know how to codegen the PTX type: e4m3").
- `scripts/scale-probe/e4m3_mma_cpasync_probe.cu` — `cvt.rn.satfinite.
  e4m3x2.f32` (`__nv_cvt_floatraw_to_fp8` undefined) on gfx1151.
- Positive controls (compile cleanly, include to show the BF16 path works):
  `bf16_mma_shfl_probe.cu`, `cpasync_only_probe.cu`.
