# DRAFT — Spectral Compute update (DO NOT auto-send; for Azeez to send)

Thread: Atlas ↔ Spectral (Michael Søndergaard / Chris Kitching / Jon).
Context: first SCALE bring-up of Atlas (pure-Rust CUDA LLM engine) on
AMD Strix Halo (gfx1151), model qwen3.6-27b.

---

Subject: SCALE 1.7.0 on gfx1151 — strong first result + two FP8 codegen repros

Hi all,

First data point from the port, and it's a good one for both of us: with
**SCALE 1.7.0** targeting **gfx1151**, **82 of 92** of Atlas's hand-written
CUDA kernels for a production LLM (Qwen3.6-27B) compile to AMD GPU code
objects **with zero source changes** (`--cuda-device-only -c`). That's the
entire BF16 / SSM-GDN / MoE-routing / RMSNorm / RoPE / paged-attention-decode
path — including `mma.sync.m16n8k16` BF16 tensor-core PTX, 32-lane
`__shfl_xor_sync`, and `cp.async.cg` double-buffering, all lowering cleanly.
Exactly the "CPU-like" experience you described. Happy to make this a
co-marketing data point once we have end-to-end numbers.

Two concrete codegen gaps remain, both in the FP8 path, with minimal repros
attached (`scripts/scale-probe/`):

**1. `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` — no gfx1151
codegen.** Repro: `e4m3_mma_only_probe.cu` (pre-packed e4m3 operands, no
conversion). Error: *"this implementation does not know how to codegen the
PTX type: e4m3"* (+ `fragment<...accumulator,16,8,32,float>` /
`fragment<...matrix_a,16,8,32,int,row_major>` decl errors). BF16 m16n8k16
MMA lowers fine, so this looks like the e4m3 tensor-core type specifically.
This is our biggest lever — Atlas's quantized GEMM/MoE hot path has ~200
of these. Is e4m3 `m16n8k32` MMA codegen on the roadmap for gfx1151, even
loosely? (We saw your note that the MMA permutation optimiser isn't
released yet.)

**2. `cvt.rn.satfinite.e4m3x2.f32` inline PTX — no codegen.** Repro:
`e4m3_mma_cpasync_probe.cu`. Error: *"does not provide a suitable
definition for `__nv_cvt_floatraw_to_fp8`, which is needed to codegen this
PTX instruction"*. Note the **C++ intrinsic path is fine**:
`__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3)` from your `cuda_fp8.h`
works perfectly (we've already used it to bridge one kernel). So this is
specifically the inline-PTX `cvt.*.e4m3x2.f32` form lacking its lowering
helper — likely a smaller fix than (1). Several Atlas kernels use the PTX
form directly.

Minor: your `cuda.h` host include needs `build-essential`, and the bundled
`libhsakmt` needs `libnuma` — only flagging since a bare WSL/Ubuntu image
hits `'cstddef' file not found` / `numa_* undefined` before anything else,
which could trip up first-time users.

One confirmation question: for `.cu` compiled by SCALE we found the device
pass defines **`__SCALE__`** (and `__AMDGCN__`) but **not**
`__HIP_PLATFORM_AMD__` — is `#if defined(__SCALE__)` the recommended guard
for SCALE-specific shims, or do you prefer `__AMDGCN__`?

**Bigger architectural question (the one that gates our runtime):** Atlas
isn't a normal CUDA app — it has no host-side `<<<>>>` launches. It compiles
every kernel to PTX at build time, embeds the PTX text, and at runtime does
`cuModuleLoadData(ptx)` + `cuModuleGetFunction(name)` + `cuLaunchKernel`
through the driver API (a JIT/registry model, like an inference engine
loading modules dynamically). On SCALE we see device-only compiles only ever
produce `ELF relocatable` (no PTX, no loadable code object), and the native
path is offload-bundling into the host binary. **What is the intended SCALE
workflow for a driver-API engine that loads modules via `cuModuleLoadData`
at runtime?** Specifically: (a) is there a way to emit a single loadable AMD
code object/fatbin per kernel-set that your `cuModuleLoadData`/
`cuModuleLoadFatBinary` accepts (and the right flags)? or (b) do you
recommend we switch to offload-bundled device code + resolve kernels by
symbol? This is the last thing between us and running Atlas on Strix Halo —
all 92 of our qwen3.6-27b kernels now compile clean for gfx1151 (incl. the
e4m3 tensor-core path via a bit-exact bf16 decomposition).

We're bridging (1)/(2) with `#if defined(__SCALE__)` dequant→BF16 fallbacks
on our side so the port can proceed in parallel — but native e4m3 codegen
would let us delete all of that and is the difference between a
proof-of-concept and a "near-native hand-tuned CUDA on AMD" headline.

Repros are standalone single-file `.cu` (compile with
`targets/gfx1151/bin/nvcc --cuda-device-only -c`); README in
`scripts/scale-probe/` has the exact commands and the pass/fail matrix.

Thanks — this is genuinely promising.

— Azeez
