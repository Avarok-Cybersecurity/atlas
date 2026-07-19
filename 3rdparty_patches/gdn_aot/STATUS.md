# GDN FlashInfer AOT integration — artifacts + status (2026-06-30)

Route A (integrate FlashInfer GDN into Atlas). The 11-13× scan is proven; this dir holds the AOT bridge.

## Done
- `gdn_holo_0.{o,h}` — AOT-exported GDN kernel (ARM aarch64 ELF + C ABI header). Via `compiled_fn.export_to_c`.
- `gdn_holo.so` — linked shared lib (`g++ -shared` + `aot_config --ldflags --libs`). Runtime dep: libcute_dsl_runtime.so.
- `delta_rule_sm120_aot_export.patch` — the 3-hunk vendored-kernel patch enabling export (grid_x→Int32, stream→CUstream annotation).
- `gdn_export.py` / `gdn_dump_meta.py` — export + arg-metadata/reference-IO dump.
- `gdn_harness.cpp` — native C++ harness. **BUILDS + RUNS: wrapper ret=0, no CUDA error → kernel loads & launches from the AOT artifact.** Mechanically proves the C-ABI path.

## RESOLVED 2026-06-30 — native C++ call is BIT-EXACT ✅
`gdn_harness.cpp` calls the C-ABI wrapper and matches the JIT reference: **max_abs_err=0.000000, cos=1.000000**, state+output fully written. Root cause of the earlier zero-output: `cu_seqlens` must be **int64** (kernel validates dtype==int64, cu_cute assumed_align=8) — the harness now builds it as int64 [0,T]. Descriptor packing (shapes[]/strides[] = the dynamic-mask dims) was correct all along.
**=> Route A C-ABI integration path fully proven: export -> link -> native call -> bit-exact GDN. Remaining is mechanical: Rust FFI + convention adapter + wire-in.**

## (historical) earlier open item
Harness output is zero (cos=0): the `{int32 shapes[3]; int64 strides[2]}` tensor structs aren't mapping to the kernel's memref descriptor convention yet, so no correct write.
Exact captured arg metadata (the target values):
- g_q: shape(2048,128,16) stride(2048,1,128)   [leading_dim=1, unit-stride dim=1]
- g_k: shape(128,2048,16)  stride(1,2048,128)   [leading_dim=0]
- g_v: shape(128,2048,32)  stride(1,4096,128)   [leading_dim=0]
- g_o: shape(128,2048,32)  stride(1,4096,128)   [leading_dim=0]
- alpha/beta [65536], state/init_state [524288], tensormaps [6144], cu_seqlens [2]
- scale=0.08838835, num_q=16 num_k=16 num_v=32 num_sab=32 num_seqs=1 total_ckpt=1 ckpt_every=0 grid_x=32
Candidate fixes to try: which 2 of 3 strides go in strides[2] + their order (current guess: the two non-unit-stride dims); verify shapes[3] dim order; confirm o readback layout (kernel writes via the (128,2048,32) strided view).
Reference IO saved on gx10 /tmp/gdn_ref/*.bin (q,k,v,g,beta,cu,o_ref) for numeric compare.

## Run recipe (gx10, quiet GPU)
g++ -O2 gdn_harness.cpp -o h -I. -I/usr/local/cuda/include ./gdn_holo.so -lcudart -L<cute_lib> -lcute_dsl_runtime -Wl,-rpath,<cute_lib>
LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat:<cute_lib>:/usr/local/cuda/lib64 CUTE_DSL_ARCH=sm_121a ./h

## STEP 3 DONE 2026-06-30 — Rust FFI -> shim -> AOT kernel is BIT-EXACT ✅
`gdn_shim.cpp` wraps the header's static-inline funcs into extern "C" `atlas_gdn_load` + `atlas_gdn_prefill`
(shape-generic, head_dim D=128 fixed). Built into `libatlasgdn.so` (bundles gdn_holo_0.o + cute runtime).
`gdn_rs.rs` is a pure-Rust harness (raw cudart + atlasgdn FFI, no cudarc) that loads ref IO, calls the kernel,
compares: **atlas_gdn_prefill ret=0, max_abs_err=0.000000, cos=1.000000.** Full chain Rust->C shim->AOT GDN proven.
Build/run (gx10):
  g++ -O2 -fPIC -shared gdn_shim.cpp gdn_holo_0.o -o libatlasgdn.so -I. -I/usr/local/cuda/include -lcudart -L<cute> -lcute_dsl_runtime -Wl,-rpath,<cute>
  rustc -O gdn_rs.rs -o gdn_rs -L. -L/usr/local/cuda/lib64 -L<cute>
  LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat:.:<cute>:/usr/local/cuda/lib64 CUTE_DSL_ARCH=sm_121a ./gdn_rs

## NEXT (step 4 — wire into Atlas proper)
- Move the shim into a real Atlas crate (build.rs links gdn_holo_0.o + cute runtime; or dlopen libatlasgdn.so).
- Convention adapter: Atlas log-space cumulative gate gc -> FI linear per-token alpha=exp(gc per-token); qk-l2norm; state layout.
- Replace the 3 FLA scan kernels in the prefill GDN path behind a flag (scalar fallback retained).
- Ship libcute_dsl_runtime.so + compat driver in the cuda13.2 container.
- e2e numerics (full Holo prefill) + the ~11x speedup measurement.

## STEP 4a DONE 2026-06-30 — Atlas-NATIVE layout adapter is BIT-EXACT ✅
KEY FINDING: Atlas `gate` is ALREADY linear α (kernel gated_delta_rule_fla.cu:16 "gate[] LINEAR decay
(NO exp)"; recompute_wu applies logf itself) == FlashInfer's alpha. NO gate-space conversion needed.
So the adapter is pure layout:
- q/k/v: pass Atlas packed QKV ([Q(key_dim)|K|V(value_dim)] bf16, row stride conv_dim) DIRECTLY via
  conv_dim strides (q/k strides{conv_dim,kd}, v{conv_dim,vd}) — NO copy.
- gate/beta: deinterleave Atlas [gate(nv)|beta(nv)] fp32 (stride 2nv) -> contiguous alpha,beta[T,nv]
  via cudaMemcpy2DAsync (in-shim).
- output: Atlas contiguous [T,value_dim] -> o strides{nv*vd, vd}.
New shim entry `atlas_gdn_prefill_packed(qkv,gate_beta,output,h_state,init_state,tensormaps,cu,
  scale,total,nk,nv,kd,vd,conv_dim,gb_stride,num_seqs,stream)` takes Atlas's EXACT native pointers.
gdn_harness_packed.cpp packs the ref IO into Atlas layout -> **bit-exact (max_abs_err=0, cos=1.0).**
=> Atlas call site becomes trivial: hand over the pointers prefill_gdn_full_inner already has
(q_ptr=gdn_bufs.qkv, gate_ptr=gdn_bufs.gate_beta, gdn_bufs.output, ssm_state.h_state, dims).

## STEP 4 remaining
- Atlas Rust binding: dlopen libatlasgdn.so (no build.rs link-time dep) OR build.rs link; call
  atlas_gdn_prefill_packed from prefill_gdn_full_inner behind ATLAS_GDN_FLASHINFER=1 (FLA fallback).
- STATE-CARRY layout: validated single-call full-sequence (init_state=0). Multi-chunk prefill carries
  h_state across outer chunks -> verify FI state layout == Atlas h_state ([nv,kd,vd]) for the carry
  (FI test transposes state; check before enabling chunked).
- Ship libatlasgdn.so + libcute_dsl_runtime.so + cuda-13.2/compat in the container.
- e2e full-Holo prefill numerics + ~11x speedup measurement (the PR-packaging gate).

## E2E WORKING 2026-06-30 — prefill correct + ~1.3× e2e speedup; decode pending state-transpose
DIAGNOSTIC CHAIN (all confirmed):
1. cross-impl A/B (gdn_fla_vs_fi example): Atlas FLA vs FlashInfer on identical input -> cos=0.999993,
   norm_ratio=1.0 => GDN MATH IS EQUIVALENT (garbage was NOT a math/convention bug).
2. The garbage was a DTYPE mismatch: export/harness used fp16, but Atlas GDN q/k/v/o are BF16. The fp16
   kernel read bf16 bits as fp16 -> garbage. FIX: re-export with torch.bfloat16 (gdn_export.py), relink.
3. Also fixed an async use-after-free (binding freed tensormaps/init/cu before the async kernel ran) ->
   managed shim entry atlas_gdn_prefill_packed_managed caches scratch internally (no per-call alloc/free/sync).
RESULT (holo35b, same binary, A/B via ATLAS_GDN_FLASHINFER 1 vs 0):
- PREFILL CORRECT: first token matches FLA ("The first 6 planets...").
- PREFILL SPEEDUP: 2K 3806->4945 (1.30x), 4K 3938->5291 (1.34x), C=8 up to 1.46x. (11x is the
  chunk_delta_h sub-kernel; e2e is Amdahl-bound by GDN's ~24-38% prefill share -> grows with context.)
- DECODE DRIFTS: 'The' then garbage = the known state k<->v transpose (FI writes S[v][k], Atlas decode
  reads S[k][v]). THE one remaining fix for full generation coherence.
NEXT: add k<->v transpose of h_state after the FI call (small per-head 128x128 transpose) -> decode
coherent -> needle/quality test; then larger-context prefill speedup; then perf-tune.
