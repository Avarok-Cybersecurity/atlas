# Native ROCm/HIP build path (no SCALE) — gfx1151 / Strix Halo

Goal: run Atlas on AMD GPUs **without SCALE**, by compiling the existing CUDA
kernels with `hipcc` and replacing SCALE's `libcuda.so` with a thin HIP shim.
Status: **build path validated; full-model serve pending build-system wiring +
a GPU window.** See `../../../.claude` memory `project_hip_port_strix` for the
blow-by-blow.

## Two layers, both validated on silicon (2026-06-03)

### 1. Kernels → hipcc
`hipcc` compiles the unmodified `.cu` directly using the CUDA→HIP **compat
headers** in `compat/` (forward `cuda_*.h` → `hip/*`, alias `__nv_*` types) +
force-included `hip/hip_runtime.h` + a tree-wide warp-mask widen sed (HIP
`__shfl_*_sync` need a 64-bit mask) + `__activemask()` captured as 64-bit.

Result on the full kernel set: **72/92 compile clean, 0 misc failures.** The
remaining 20 are tensor-core kernels using NVIDIA `mma.sync` PTX — hand-ported
to AMD WMMA (`__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32`). `HipTarget` in
`build_target.rs` runs this recipe (`hipcc -x hip --genco`).

WMMA C-fragment layout (empirically mapped on gfx1151): for lane `l`, vgpr
element `e` → output `row = 2*e + (l>>4)`, `col = l & 15`. A-load `a[i]=A[l&15][i]`,
B-load `b[k]=B[k][l&15]`. The ported `w4a16_gemm` (see `ported/`) matches a CPU
dequant-GEMM reference to within 1 bf16 ULP (0/4096 cells worse).

### 2. Runtime → libcuda→HIP shim
The runtime is unchanged (cudarc, `-lcuda`). `libcuda_hip_shim.cpp` re-exports
the **33** CUDA driver symbols the binary imports and maps each to HIP
(`cuModuleLoadData`→`hipModuleLoadData`, `cuLaunchKernel`→`hipModuleLaunchKernel`,
`cuMemAlloc_v2`→`hipMalloc`, graphs/streams/events 1:1). Build it as `libcuda.so`,
put it first on the loader path:
```
hipcc -shared -fPIC libcuda_hip_shim.cpp -o libcuda.so -I/opt/rocm/<ver>/include
```

## Turnkey remaining checklist (every layer below is already proven; this is labor)

**A. MMA kernel ports (build blocker — a full `cargo build` compiles ALL kernels,
so the 20 `mma.sync` ones must compile first).** Recipe: replace the
`mma.sync.m16n8k16.row.col.f32.bf16.bf16.f32` asm with
`__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32` using the validated fragment map
(C: row=2*e+(lane>>4), col=lane&15; A: a[i]=A[lane&15][i]; B: b[k]=B[k][lane&15];
acc = 4× v8f for n64). Also replace `cp.async.cg.shared.global` with synchronous
smem loads (correctness-first; lose pipelining). For the NVFP4 **dense** model:
  - PORT: `w4a16_gemm` (✅ done, in `ported/`), `dense_gemm_tc`,
    `inferspark_prefill` (attention: same MMA shape + 4 cp.async sites).
  - STUB (not used by dense NVFP4): `w8a16_gemm(_t)` (FP8-weight), all `moe_*`
    (MoE), the unused `inferspark_prefill_paged_*` variants, `reshape_and_cache_turbo`
    (CLEAN non-turbo fallback exists). A stub = compile body that traps if called.

**B. Wire `HipTarget` into `build.rs`** (target/compile flow mapped 2026-06-03):
  - `build.rs:159` `resolve_compute_target(vendor)` reads `kernels/<hw>/HARDWARE.toml`
    `vendor`. Add a `strix-hip` HW dir (or env override) with `vendor="hip"`.
  - Before the compile loop (`build.rs:192`): copy the kernel sources to `OUT_DIR`,
    run the mask-widen sed on the copies, stage `hip/compat/` and export
    `ATLAS_HIP_COMPAT_INCLUDE`. `HipTarget.compile` (already added) does the rest.
  - `output_extension()="co"`; the registry codegen (build_codegen.rs) embeds the
    `.co` objects — verify it treats them like the SCALE `.o` path.

**C. Build + link:** `cargo build --release -p spark-server --no-default-features
--features cuda` with the HIP env; link/LD against the shim `libcuda.so`
(`-L.../hip` first) + `libamdhip64`.

**D. Serve + measure (GPU window — bounce the SCALE demo server):** `spark serve
Qwen/Qwen3.6-27B-FP8 …`; coherence check (temp=0 prompts); decode tok/s via
`~/bench-atlas.sh` (target 20). Decode needs **zero** MMA ports, so even with
stubbed prefill-extras the decode number is measurable once prefill (w4a16_gemm +
inferspark_prefill) works.
