# Atlas issues found during the Strix Halo (gfx1151 / SCALE) port

Running log of bugs, waste, and design smells uncovered while bringing Atlas up
on AMD Strix Halo. "Atlas-side" = our code; "SCALE-side" = Spectral's compiler/
runtime. Issues that also affect the GB10/CUDA build are flagged **(also GB10)**.

Status legend: 🔧 fixed · 🩹 worked around · 📋 open · 💭 design question

---

## #A1 — Duplicate full-precision weight copies retained on GPU 🔧 **(also GB10)**
**Where:** `crates/spark-model/src/weight_loader/qwen35_dense.rs`
**Severity:** high (caused OOM on 61 GB Strix; silent ~2-3× weight bloat on GB10)

The dense Qwen3.6 loader builds NVFP4 weights from a BF16 intermediate, but then
**keeps the BF16 intermediate too** — and for SSM layers, a *third* transposed
NVFP4 copy:
- Attention (16 layers): stored both `q/k/v_proj` (BF16) **and** `q/k/v_nvfp4`.
  The BF16 `o_proj` source (`_o_dense`) was allocated then dropped **without
  freeing** → permanent leak.
- SSM/linear-attn (48 layers): stored `in_proj_qkvz` (BF16) **and** `qkvz_nvfp4`
  **and** `qkvz_nvfp4_t`. The transient `qkv_dense`, `z_dense`, `out_proj_dense`
  BF16 buffers were never freed after being consumed → permanent leaks.

The runtime forward (prefill, decode, multi_seq, ssm_forward) **always**
dispatches the NVFP4 weights; the BF16 fields are an unreachable fallback. So
the BF16 copies are dead weight whenever NVFP4 is present.

Measured: loading the 27B model consumed ~60 GB by layer 50/64 (≈77 GB
projected) — far more than even an all-BF16 model (~54 GB), which is the
signature of multiple retained copies. On GB10's 119 GB it fit, so it was never
noticed; it's pure waste there too.

**Fix:** free the BF16 intermediates as soon as the NVFP4 weight is built, and
store `DenseWeight::null()` in the (unreachable) BF16 fields. Unconditional — the
weights are provably dead, so GB10 benefits as well.

**Follow-ups still open (📋):**
- `qwen35_dense.rs` MTP loader (2nd fn, ~line 341) likely has the same pattern.
- `qwen3.rs` (other model family) has its own `in_proj_qkvz` — check separately.

---

## #D1 — Why a BF16 hop / dead fallback at all? 💭 design question (raised by Azeez)
Today: FP8-on-disk → dequant to **BF16** (GPU) → quantize to **NVFP4** → keep
NVFP4 for compute. The BF16 hop exists because `quantize_to_nvfp4` takes BF16
input, and historically prefill used a BF16 GEMM while decode used the NVFP4
GEMV — so both were kept. Prefill now also uses NVFP4 (`*_nvfp4_t`), leaving the
BF16 path vestigial.

Open design question: pick **one** weight precision and only materialize BF16
transiently when a kernel genuinely needs it.
- All-NVFP4: smallest (~14 GB for 27B); prefill+decode already have NVFP4 kernels.
- All-FP8 (native, `w8a16`): ~27 GB, avoids the NVFP4 re-quant entirely and may
  be higher quality (relates to the FP8-ceiling work in memory).
Either removes the dual-storage by construction rather than freeing after the
fact. Deferred — bigger refactor than the port needs right now.

---

## #S1 — `ld.lld -shared` produced a module SCALE's loader rejects 🔧 Atlas-side
**Where:** `crates/atlas-kernels/build_target.rs` (`ScaleTarget::compile`)
The AMD path compiled each `.cu` to a relocatable, then ran `ld.lld -shared` to
make an ELF DYN — on the (untested, no live runtime) assumption that
`cuModuleLoadData` needs a linked code object. Empirically on SCALE 1.7.1 the
opposite is true: `cuModuleLoadData` **accepts the bare `--cuda-device-only -c`
relocatable** (load+getFunction+launch all succeed) and **rejects the ld.lld DYN
with `CUDA_ERROR_INVALID_IMAGE`**. SCALE links the relocatable itself at load.
**Fix:** drop the `ld.lld -shared` step; emit the relocatable directly.

## #S2 — Stale `common/` kernel silently shadowed the model override 🔧 Atlas-side
**Where:** build kernel-set assembly + `run-build.sh`
`run-build.sh` set `ATLAS_TARGET_QUANT=fp8`, but `model_kernel_dir =
model_dir.join(quant)` and **no `qwen3.6-27b/fp8/` dir exists** (only `nvfp4/`;
GB10 is the same). So the 4 model-specific overrides were skipped and the stale
`common/w4a16_gemm.cu` (3 kernels, **no** `predequant_nvfp4_to_fp8`) shadowed the
real 10-kernel `qwen3.6-27b/nvfp4/w4a16_gemm.cu` — same stem → same
`t0__w4a16_gemm.o`. Result: `predequant_nvfp4_to_fp8` missing at runtime.
**Fix:** `ATLAS_TARGET_QUANT=nvfp4` (matches GB10 + doc §5).
**Open (📋):** `collect_cu_files` dedups common-vs-model by stem, but the build
gives **no warning** when a requested quant dir is absent or when a `common/`
file shadows nothing/everything. A missing quant dir should warn or fail fast
(PCND), not silently fall back to whatever `common/` has.

## #S3 — `cuModuleGetFunction` succeeds for a missing kernel 📋 SCALE-side (+ Atlas hardening)
When a module is missing a kernel symbol, SCALE 1.7.1's `cuModuleGetFunction`
returns **success** with a bogus handle; the failure only surfaces later as
`CUDA_ERROR_INVALID_IMAGE` at `cuLaunchKernel`, with no kernel name in the error.
This made #S2 hard to localize. Worth (a) reporting to Spectral, and (b) Atlas
hardening: include the `module::func` name in the launch-failure error so a bad
image is immediately attributable. (NVIDIA returns `CUDA_ERROR_NOT_FOUND` here.)

## #4 — memlock default too low for large-model pinning 🩹 environment
Bare-metal Strix default `memlock` = 8 GB; weight-load host pinning needs more.
The DGX container used `--ulimit memlock=-1`. Worked around with a
`limits.d` drop-in (`azeez - memlock unlimited`). Not an Atlas bug, but Atlas
docs should call out the requirement for non-container hosts. (Was not the real
OOM cause — see #A1 — but would have bitten regardless.)

---

_Last updated: 2026-06-01 (Strix bring-up session)._

---

## #A2 — `inferspark_prefill` LDS footprint exceeds RDNA3.5 64 KB cap 📋 Atlas-side **(gfx1151-specific)**
**Where:** `kernels/.../inferspark_prefill*.cu`; surfaces at runtime as
`cuFuncSetAttribute(MAX_DYNAMIC_SHARED=69688) failed: CUDA_ERROR_INVALID_VALUE`
at prefill layer 0.
The prefill attention kernel requests ~68 KB dynamic shared memory (Q/K/V/P
tiles). Qwen3.6-27B has `head_dim=256`, which makes the tiles large. gfx1151
(RDNA3.5) has a hard **65,536 B** per-workgroup LDS cap; GB10/Blackwell allows
~228 KB, so it only bites on AMD. This is the §4.2 item from the porting doc.
**Fix (deferred, numerics-bearing → needs GPU verification):** reduce the
`__shared__` footprint under `#if defined(__SCALE__)` — single-buffer the K
tile and/or shrink the BR/BC tile so the kernel fits in 64 KB. The runtime
reaches this cleanly now (server loads + listens + accepts requests), so it can
be iterated on-device.

## #M1 — Memory-budget tuning for 61 GB unified (Strix) 📋 config/tuning
After #A1, the 27B model loads (weights 28.75 GB) and the server reaches
`Listening`, but the OOM watchdog (2 GB threshold) tripped at default
`--gpu-memory-utilization`. The KV-cache sizer greedily fills to the util budget
and CUDA-graph capture allocates *on top* of that during warmup, overshooting.
Lowering to `--gpu-memory-utilization 0.70` (with `--max-seq-len 4096
--max-batch-size 4`) leaves enough headroom and the server stays up.
**Notes / follow-ups:**
- Restoring `--max-seq-len 16384` (needed for Claude Code tool use) needs more
  headroom: the activation buffer arena is sized from max_seq_len. Either shrink
  the arena, account for graph-capture memory inside the util budget, or expose
  a KV-cache cap so KV doesnt greedily consume all free memory.

---

## #A2 — inferspark_prefill LDS footprint exceeds RDNA3.5 64 KB cap (open, Atlas-side, gfx1151-specific)
Where: kernels/.../inferspark_prefill*.cu; surfaces at runtime as
"cuFuncSetAttribute(MAX_DYNAMIC_SHARED=69688) failed: CUDA_ERROR_INVALID_VALUE"
at prefill layer 0.
The prefill attention kernel requests ~68 KB dynamic shared memory (Q/K/V/P
tiles). Qwen3.6-27B has head_dim=256, which makes the tiles large. gfx1151
(RDNA3.5) has a hard 65,536 B per-workgroup LDS cap; GB10/Blackwell allows
~228 KB, so it only bites on AMD. This is the section 4.2 item from the porting doc.
Fix (deferred, numerics-bearing -> needs GPU verification): reduce the
__shared__ footprint under #if defined(__SCALE__) — single-buffer the K tile
and/or shrink the BR/BC tile so the kernel fits in 64 KB. The runtime reaches
this cleanly now (server loads + listens + accepts requests), so it can be
iterated on-device.

## #M1 — Memory-budget tuning for 61 GB unified (Strix) (open, config/tuning)
After #A1, the 27B model loads (weights 28.75 GB) and the server reaches
Listening, but the OOM watchdog (2 GB threshold) tripped at default
gpu-memory-utilization. The KV-cache sizer greedily fills to the util budget and
CUDA-graph capture allocates on top of that during warmup, overshooting.
Lowering to --gpu-memory-utilization 0.70 (with --max-seq-len 4096
--max-batch-size 4) leaves enough headroom and the server stays up.
Notes / follow-ups:
- Restoring --max-seq-len 16384 (needed for Claude Code tool use) needs more
  headroom: the activation buffer arena is sized from max_seq_len. Either shrink
  the arena, account for graph-capture memory inside the util budget, or expose
  a KV-cache cap so KV does not greedily consume all free memory.
- The OOM watchdog reads cuMemGetInfo free, which on a unified-memory APU does
  not count reclaimable page cache. Not the blocker here (the fast loader uses
  O_DIRECT and bypasses page cache), but worth knowing for APU targets.

---

## #A3 — Generation runs but output is gibberish (open, Atlas/SCALE numerics, gfx1151)
After #A1/#A2/#M1, the full pipeline runs end-to-end on Strix (prefill via the
global-memory split4 GDN path -> decode -> tokens), 40 tok at 9.7 tok/s,
TTFT 2.7 s, no crashes. But output is incoherent from token 1
(" vela-pills Stateless ..."), so logits are already wrong after prefill —
a compute-kernel numerics bug, not the model.
Ruled out: FP8 KV cache (gibberish persists with --kv-cache-dtype bf16).
Prime suspects (in order):
1. The e4m3 m16n8k32 -> BF16 MMA rewrite (atlas_mma_e4m3 in w4a16_gemm.cu,
   commit fde2620). The porting doc explicitly says this was NEVER GPU-verified
   ("a wrong thread/fragment mapping = silently wrong matmul"). It feeds every
   NVFP4 w4a16 GEMM (FFN + attention projections) -> pervasive gibberish.
2. NVFP4 w4a16 GEMM kernel codegen differences on SCALE/gfx1151.
3. split4 GDN prefill kernel correctness on SCALE (less-trodden fallback path).
Next: per-layer / per-kernel cosine A/B vs the GB10 BF16 reference (the method
used for the GB10 FP8-drift work) to localize the first diverging kernel. Start
by dumping L0 hidden after each sub-op (embed, qkvz GEMM, GDN core, FFN) and
comparing to a known-good run. Consider an env to force the dense BF16 GEMM
path (bypass the e4m3 MMA) to confirm/deny suspect #1 cheaply.

### #A3 update (2026-06-01) — localized: GDN (LinearAttention) blows up the residual from L0

Per-layer hidden[0] norm dump (widened the existing profile-gated diagnostic in
`prefill_b/forward_layers.rs` to all layers under ATLAS_DUMP_LAYER_NORM):
- L0 (LinearAttention) hidden norm = **166,564** — already ~1000x too large after
  the first layer (first 64 BF16 elements average ~20,000 each). Healthy would be
  O(10-100).
- Norm grows monotonically L0->L63 (166k -> 1.04M).
- Every FullAttention layer is a **no-op**: norm byte-identical before/after
  (L2==L3, L6==L7, ...). Its O(1) output is lost against the GDN-inflated residual
  (BF16 rounding) — i.e. a *consequence* of the blow-up, not necessarily a 2nd bug.

Root cause is the GDN/LinearAttention prefill path inflating its output starting
at L0. Prime suspect: the **global-memory `gated_delta_rule_prefill_split4`
kernel** that #A2's fix forces on gfx1151 — it is the rarely-used fallback (GB10
uses the persistent kernels), so it is the least-tested path and the most likely
to be wrong or miscompiled by SCALE. Secondary suspects inside the GDN layer:
the gated RMSNorm / output normalization, the qkvz NVFP4 GEMM scale handling, or
conv1d. (The e4m3->BF16 MMA itself is EXONERATED: bit-exact on gfx1151.)

Next steps to pin it:
1. Run the same model on GB10 with ATLAS_FORCE_GLOBAL_GDN=1 (forces split4 there
   too). If GB10+split4 is coherent -> SCALE miscompiles split4 on gfx1151. If
   GB10+split4 is also garbage -> split4 kernel itself is buggy (fix benefits both).
2. Dump the embedding norm (pre-L0) to confirm the input to L0 is sane (~O(10)),
   isolating the inflation to the L0 GDN layer rather than embedding scaling.
3. Unit-test split4 GDN output vs a CPU delta-rule reference on gfx1151 (probe).
4. Check whether the gated RMSNorm after the GDN core is actually applied on the
   SCALE path (a skipped/wrong post-norm would leave the large pre-norm GDN state).

### #A3 ROOT CAUSE (2026-06-01) — m128 pipelined NVFP4 GEMM drops scale2 on SCALE/gfx1151

Localized the gibberish to ONE kernel: `w4a16_gemm_t_m128` (the cp.async-pipelined
NVFP4 W4A16 GEMM, used for SSM qkvz + attention proj + FFN). It produces output
~1/scale2 (~6000x) too large — i.e. the global NVFP4 scale (scale2 = global_max/
(6*448)) is effectively NOT applied. q/k get L2-normalized downstream so the
inflation is hidden; the raw z-gate exposed it (gate_z=179k -> silu -> post_norm
16769 -> out_proj 166k -> residual blows up from L0).

PROOF: forcing the qkvz GEMM to the NON-pipelined base `w4a16_gemm` kernel
(ATLAS_W4A16_NOPIPE=1, nulls qkvz_nvfp4_t) drops qkvz_q from 167422 -> 3.743
(correct). So m128 is the culprit; the base kernel scales correctly. (Base is
NOT a drop-in: its non-transposed [N,K] output layout mismatches the SSM
deinterleave -> gdn_out=0, so the model still isn't coherent on that path.)

EVERY primitive used by m128 was tested bit-exact on gfx1151 and PASSES:
- atlas_mma_e4m3 (e4m3->bf16 MMA decomposition): max|cand-cpu|=0.0
- (float)__nv_fp8_e4m3 cast, __nv_cvt_fp8_to_halfraw, __nv_cvt_float_to_fp8
- cvt.f32.bf16 inline PTX (bf16x4_to_e4m3x4 A-conversion)
- rsqrtf, __expf, warp_reduce_sum, __shared__/__syncthreads block reduction
- __constant__ float[] initializer (E2M1_LUT)
- cp.async.cg.shared.global + commit/wait_group + predication (single-shot)
- kernel arg ABI (ptr/f32/u32 mix), arg_f32 passing, transpose_for_gemm preserves scale2
- base w4a16_gemm (same dequant expr `LUT*(float)fp8*scale2`) works correctly

So the defect emerges only in the ASSEMBLED m128 kernel — the double-buffered
cp.async pipeline + unrolled DEQUANT_T macro. Difference vs the working base
kernel: base reads block-scale from GLOBAL and dequants to BF16 (BF16 MMA);
m128 reads block-scale from SMEM (cp.async-loaded smem_Bs), dequants
`sv0=(float)f0*scale2; lo=LUT*sv0`, converts lo->FP8 (atlas_cvt_e4m3x2_f32),
FP8 MMA. The scale2 multiply in the pipelined/unrolled DEQUANT_T is what gets
effectively lost (likely a SCALE -O3 miscompile of the macro, or the
cp.async-loaded smem_Bs block-scale read). Single-shot cp.async tested fine;
the double-buffered pipeline loop interaction is the remaining untested combo.

NEXT (next session): round-trip probe of w4a16_gemm_t_m128 in isolation
(known A + CPU-quantized B, compare to CPU A@dequant(B)) to pin the exact line;
then fix (candidates: apply scale2 to the FP32 accumulator at output instead of
in the pipelined dequant; or read block-scale from global like the base kernel;
or -O2 the kernel). Verify qkvz_q ~ O(10) and coherent generation. The fix must
cover all w4a16_gemm_t_m128 callers (qkvz, attn q/k/v/o, FFN gate/up/down).
Debug env flags left in tree (gated, harmless): ATLAS_DUMP_GDN, ATLAS_W4A16_NOPIPE,
ATLAS_DUMP_LAYER_NORM. Probes in /tmp/modprobe/ (mma_gfx, gnorm, cvtbf16, argabi,
consttest, cpasync, blockred, mathtest, fp8cast — all PASS).

### #A3 update-2 (2026-06-01) — m128 bug is the FP8-MMA path, NOT scale2-form

Tried inlining scale2 in m128 DEQUANT_T (LUT*bs*scale2, matching the working base
kernel's expression form). NO change: qkvz_q still exactly 167422. So scale2 IS
applied correctly; the m128 defect is elsewhere.

Confirmed contrast:
- base w4a16_gemm (BF16 dequant -> BF16 m16n8k16 MMA): qkvz_q=3.74 CORRECT.
- m128 w4a16_gemm_t_m128 (FP8 dequant via atlas_cvt_e4m3x2_f32 -> FP8 MMA via
  atlas_mma_e4m3 decomposition): qkvz_q=167422 WRONG (~45000x).
Both apply scale2 identically; both use exonerated primitives in isolation.

=> The bug is in the m128 FP8-MMA path as ASSEMBLED: either the in-kernel A->FP8
(bf16x4_to_e4m3x4 on smem_A with its stride), the dequant->FP8 (atlas_cvt_e4m3x2_f32
on small ~0.001-0.45 values), the FP8 fragment feeding of atlas_mma_e4m3, or the
transposed weight tiling/accumulation. atlas_mma_e4m3 was only verified on inputs
~[-3,3], NOT on the tiny dequanted-weight magnitudes the real kernel produces.

HIGHEST-CONFIDENCE FIX (next session): convert w4a16_gemm_t_m128 to dequant->BF16
+ BF16 m16n8k16 MMA (exactly the base kernel's proven-correct approach) while
keeping m128's transposed tiling/layout — so the SSM deinterleave still gets the
right layout (base kernel alone gives gdn_out=0 due to non-transposed layout).
Then it covers all m128 callers (qkvz, attn q/k/v/o, FFN). Alternative: round-trip
probe of w4a16_gemm_t_m128 (known A + CPU-quantized B vs CPU ref) to pin the exact
FP8-path line. Also worth trying: the ATLAS_FP8_DEQUANT_*_TO_BF16 levers (route to
BF16 dense_gemm) if 61GB memory allows at reduced ctx/batch — fastest coherence proof.
Inline-scale2 attempt reverted; tree clean except debug env flags (gated).

### #A3 *** DEFINITIVE ROOT CAUSE *** (2026-06-01) — device float->E4M3 encode is broken on SCALE/gfx1151

THE BUG: every device-side float->FP8-E4M3 ENCODE path is broken on SCALE 1.7.1 /
gfx1151 — produces a byte that decodes to ~16.0 for ANY input:
  - __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3)  -> 1.0->16, 2.0->16, 0.45->16
  - __nv_cvt_float_to_fp8(x, __NV_NOSAT, ...)            -> same
  - __nv_fp8_e4m3(x) class constructor                  -> same
FP8 DECODE works fine (__nv_cvt_fp8_to_halfraw, (float)__nv_fp8_e4m3). A software
bit-manipulation encoder works (2.0->2.0). (Earlier MMA probe used the encode on
the HOST = correct; only DEVICE codegen is broken — that's why it passed.)

Impact: the m128 NVFP4 GEMM (w4a16_gemm_t_m128) ENCODES both the dequanted weight
(atlas_cvt_e4m3x2_f32, w4a16_gemm.cu ~line 24, #if __SCALE__) and the A input
(bf16x4_to_e4m3x4) to FP8 before the FP8 MMA -> all garbage (~16) -> qkvz_q
167422 vs correct 3.74 -> gibberish. The base w4a16_gemm dequants straight to
BF16 (no float->fp8 encode) -> correct. Used by ALL m128 callers (qkvz/attn/FFN).

FIX (next session, two options):
  (A) Replace the broken __nv_cvt_float_to_fp8 in the #if __SCALE__ branch of
      atlas_cvt_e4m3x2_f32 (and any float->fp8 encode) with a CORRECT software
      e4m3 encoder. MUST match SCALE's __nv_cvt_fp8_to_halfraw decoder convention
      (decode 0x30->1.0, 0x38->1.5, 0x40->2.0 — derive exact format by decoding
      bytes 0x00-0xFF first, then write the inverting RNE encoder). Smallest blast
      radius; keeps the FP8-MMA path.
  (B) Rewrite w4a16_gemm_t_m128 compute to dequant->BF16 + BF16 m16n8k16 MMA (the
      proven base-kernel path), keeping m128's transposed tiling. Larger but
      avoids FP8 encode entirely.
ALSO: quantize_bf16_to_nvfp4 (block-scale fp8 encode at load) likely uses the same
broken intrinsic — verify/fix or it stores garbage block scales (base path may be
self-consistent garbage). And this is a clean Spectral compiler-bug repro:
/tmp/modprobe/fp8enc.cu (intrinsic vs software encoder round-trip).

Probes (all in /tmp/modprobe/, green except the encode ones which expose the bug):
fp8enc/fp8small/fp8ctor = ENCODE BROKEN; fp8cast/consttest = decode OK; mma_gfx/
cpasync/blockred/argabi/cvtbf16/mathtest/gnorm = OK.

### #A3 COMPLETE DIAGNOSIS (2026-06-01) — FP8-MMA path is unusable on SCALE; fix = BF16 MMA

Two compounding SCALE/gfx1151 facts make the m128 NVFP4 GEMM's FP8 path unusable:
1. Device float->E4M3 ENCODE is broken (all variants -> ~16). [crash-level bug]
2. SCALE's __NV_E4M3 decode is a NON-STANDARD narrow format: measured
   value = 2^(((b>>4)&7)-3) * (1 + (b&0xF)/16), i.e. sign/exp3(bias3)/mant4,
   range ~[0.125, 31], NO zero/subnormals below 0.125. (Standard e4m3 is
   [~0.002, 448].) So even with a correct software encoder (verified: matches
   for >=0.125), the dequanted NVFP4 weights (~0.001-0.45, many <0.125) AND the
   un-scaled LUT*blockscale (~60-2688) BOTH fall outside [0.125,31] -> clamp/lose.
   FP8 MMA simply can't carry this GEMM's dynamic range on SCALE.

=> FIX (decided): rewrite w4a16_gemm_t_m128 to dequant->BF16 + BF16 m16n8k16 MMA
(exactly the base w4a16_gemm path, which is proven correct on SCALE: qkvz_q=3.74),
keeping m128's transposed tiling + cp.async pipeline (those are fine). Replace the
DEQUANT_T FP8 store (atlas_cvt_e4m3x2_f32 -> smem_B_fp8) with a BF16 store
(__float2bfloat16(LUT*bs*scale2) -> smem_B_bf16), and COMPUTE_MMA's
bf16x4_to_e4m3x4(A)+atlas_mma_e4m3 with a direct BF16 m16n8k16 MMA (A already
BF16 in smem_A). Apply to all w4a16_gemm_t* variants (qkvz/attn/FFN). This drops
the broken float->fp8 encode entirely. Verify qkvz_q ~ O(10) + coherent gen.
(GB10 keeps native FP8 e4m3 — guard the BF16 path under #if defined(__SCALE__).)

Bonus: clean Spectral compiler-bug repro = /tmp/modprobe/fp8enc.cu (device
__nv_cvt_float_to_fp8 round-trips everything to 16) + /tmp/modprobe/decmap.cu
(non-standard __NV_E4M3 decode table).

### #A4 (2026-06-01) — m128 GEMM FIXED via BF16 rewrite; next bug = split4 GDN outputs ~0

FIX SHIPPED (uncommitted): rewrote w4a16_gemm_t_m128 DEQUANT_T + COMPUTE_MMA to
dequant->BF16 + BF16 m16n8k16 MMA (2x over the 32-K step), dropping the broken
float->E4M3 encode. smem_B_fp8 -> smem_B_bf16. RESULT: qkvz_q 167422->3.743,
gate_z 179293->4.380 (both CORRECT). Confirms the #A3 root cause and fix.
(NOTE: attn q/k/v/o + FFN use w4a16_gemm_t_m128_v2/v3 variants — same FP8-encode
bug, NOT yet rewritten. Must apply the same BF16 conversion there for full coherence.)

NEXT BUG (#A4): with correct GDN inputs (qkvz_q=3.74, gate_z=4.38, conv_out=0.091,
post_l2norm_q=0.399, v_in=0.090, decay[0..4]=[0.24,1.0,0.99,0.98],
beta[0..4]=[0.07,0.94,0.26,0.99] — ALL valid), gdn_out=0.000. So the
gated_delta_rule_prefill_split4 kernel (forced on gfx1151 by #A2 ATLAS_FORCE_GLOBAL_GDN
because the persistent GDN kernels exceed 64KB LDS) produces ~0 for normal-magnitude
inputs on SCALE (it gave 10.17 earlier only because the inputs were the huge broken-qkvz
garbage). split4 keeps H in a per-thread float H_reg[K_DIM] register array (likely
spills to scratch at K_DIM=128) + reads H_global; suspect SCALE miscompiles the large
register-array / scratch path, OR an underflow. Verify gdn_out is exactly 0 vs tiny
(re-dump %.6f). FIX OPTIONS: (a) debug/fix split4 on SCALE; (b) make a TESTED persistent
GDN kernel fit 64KB by storing H in BF16 smem (H[128x128]x2=32KB fits) and re-enable it
instead of split4. Then attn/FFN m128_v2/v3 BF16 rewrite -> coherent.

### #A5 (2026-06-01) — ALL NVFP4 GEMMs fixed (BF16); forward HEALTHY; residual content-flow bug remains

PROGRESS: applied the BF16-MMA rewrite to BOTH transposed NVFP4 GEMMs that the
forward uses, and disabled the broken FP8 predequant:
1. w4a16_gemm_t (line 277): DEQUANT_T->BF16 store, COMPUTE_MMA->2x m16n8k16 BF16.
   Used by SSM qkvz + SSM out_proj (via w4a16_gemm_n128). VERIFIED correct:
   qkvz_q=3.743251 == base w4a16_gemm (NOPIPE) 3.743 exactly.
2. w4a16_gemm_t_m128 (line 960): same BF16 conversion, 2 M-chunks. (attention path)
3. ATLAS_NO_FP8_PREDEQUANT=1 (init.rs predequant_for_prefill early-return): out_proj
   was using out_proj_fp8 -> fp8_gemm_t (broken encode) -> 168651; now uses BF16
   w4a16_gemm_t -> out_proj=0.274 (correct). FFN already used base w4a16_gemm (BF16, fine).

RESULT: forward pass is now NUMERICALLY HEALTHY. Per-layer residual norm L0=0.276,
L15=0.82, L31=1.06, L47=1.20, L63=2.83 (smooth monotonic — was 168k->1M). Logits
sane: max~7.8 min~-3.3 nan=0 (was inflated). out_proj/qkvz/gate all correct.

REMAINING BUG: output is GIBBERISH and ~INPUT-INDEPENDENT — top logit is ALWAYS
token 11 ("," ) for every prompt (chat AND raw completion), top5 always punctuation
(11,198,25,13,220). Logits vary only slightly by input (7.84 vs 7.5). So content is
barely flowing -> final hidden ~dominated by a constant/bias, content signal too weak
to change the argmax. VERIFIED-CLEAN on SCALE/gfx1151 (probes /tmp/modprobe/, all green):
GEMMs (qkvz==base), sw_exp softmax (ldexpf, <0.2% err), attention QK^T/AV (BF16 MMA),
all fp8 decode / cvt.f32.bf16 / rsqrtf / block-reduce / cp.async / __constant__.
NOT yet verified: split4 GDN *values* (forced LDS-fallback, 48/64 layers, per-thread
H_reg[K_DIM] register array — suspect SCALE scratch-spill mishandling breaks the
recurrence so SSM carries no content), RoPE (cos/sin/powf), embedding lookup.

NEXT: per-layer cosine A/B vs a GB10 BF16 reference (the established method) to find
the first diverging layer/op — single-kernel probing has hit diminishing returns.
Quick checks first: (a) dump embedding for 2 different tokens (constant? -> embed bug);
(b) standalone split4 recurrence vs CPU delta-rule on multi-token input; (c) RoPE
cos/sin probe. Likely split4 (content not accumulating) given input-independence.
Debug flags in run-serve.sh: ATLAS_W4A16_VARIANT=v1, ATLAS_NO_FP8_PREDEQUANT=1,
ATLAS_FORCE_GLOBAL_GDN=1 (all needed); dumps gated off.

### #A6 (2026-06-01) — remaining bug LOCALIZED: context not propagating to last token

After all GEMM/encode fixes (forward numerically healthy, residual 0.28->2.83),
output is still gibberish and ~INPUT-INDEPENDENT. Localized via last-token hidden
dump across 2 prompts (modified forward_layers dump to read token proc_count-1):
- "apple" (diff last token) -> L0 hidden DIFFERS from chat prompts => embedding/
  per-token projection WORKS.
- "Banana elephant volcano" vs "The quick brown fox jumps" (same chat-template last
  token, DIFFERENT earlier content) -> last-token hidden NEARLY IDENTICAL at EVERY
  layer L0..L63 (differ only ~1-2%: L63 [-0.2158,-0.2539,0.3359] vs [-0.2129,-0.2578,
  0.3379]). => the last token is NOT aggregating prior-token context. Context
  propagation is ~50-100x attenuated, from L0 onward, affecting BOTH SSM (L0 GDN) and
  attention layers. That's why logits are input-independent (top token always 11 ",").

So the bug is CROSS-TOKEN PROPAGATION in the prefill, NOT the per-token math (GEMMs,
GDN recurrence, softmax, MMA all verified correct in isolation). Shared cause across
attention + SSM => prefill sequence handling. Candidates: (a) attention kv_len/causal
so the last token only weakly attends to prior; (b) SSM full-kernel not accumulating H
across the sequence (single-thread recurrence probe passed, but full split4 multi-thread
/ smem-k-q-per-token loading untested for propagation); (c) batched-prefill path
(prefill_b_forward_layers) treating tokens with wrong seq vs batch; (d) attention/SSM
output scaled ~100x too small so content barely enters the residual.

NEXT: (1) 2-sequence full-split4 propagation probe (does last output depend on earlier
tokens?); (2) dump attention output magnitude + check kv_len/softmax-scale in the
batched prefill; (3) GB10 per-layer cosine reference. The core SCALE compiler bug
(float->E4M3 encode) IS fixed and the forward is numerically sound — this is a separate
prefill-propagation issue. run-serve.sh flags: FORCE_GLOBAL_GDN, W4A16_VARIANT=v1,
NO_FP8_PREDEQUANT, DUMP_LAYER_NORM (reads last token now).

### #A6 CORRECTED + #A7 (2026-06-01) — ROOT CAUSE FOUND & FIXED: SCALE __NV_E4M3 is non-standard

My earlier #A6 "context not propagating" was WRONG (artifact of testing two
semantically-similar noun-list prompts). The real bug:

**SCALE's `__nv_fp8_e4m3`→float and `__nv_cvt_fp8_to_halfraw(b,__NV_E4M3)` decode
(and `__nv_cvt_float_to_fp8(...,__NV_E4M3)` encode) use a NON-STANDARD narrow E4M3
format on gfx1151.** Probe (verified, /tmp/modprobe/probe_scale): encode a value with
standard E4M3 then decode via SCALE → 1.0→1.5, 0.5→1.0, 3.5→2.75, 0.75→1.25 (only 2.0
correct). On real NVIDIA __NV_E4M3 IS standard, so GB10 is fine — this is SCALE-specific.

`quantize_bf16_to_nvfp4.cu` ENCODES per-16-block scales as STANDARD E4M3 (software,
lines 26-85). Every GEMM/GEMV/attention kernel DECODED those scale bytes with SCALE's
non-standard `(float)__nv_fp8_e4m3` → ALL NVFP4 block-scales decoded wrong → all weights
garbage. It "matched base" in A/B because base used the same broken decode.

FIX: standard pure-bit-math `scl_fp8(b)` (decode, byte-identical to (float) on NVIDIA)
and `scl_enc_fp8(f)` (encode), injected into every dense-relevant kernel, replacing all
`__NV_E4M3` intrinsic uses. ~20 files: w4a16_gemm (27b + common), w4a16_gemv,
w4a16_gemv_fused, dense_gemv_fp8w, all paged_decode_attn_* + inferspark_prefill_* +
reshape_and_cache. Progression as fixes landed: pure gibberish → "Paris"+gibberish
(prefill GEMM fixed) → first-token-correct+decode-garbage (GEMV fixed) → FULLY COHERENT
(comprehensive decode+encode fix).

VERIFIED COHERENT on Strix Halo (Qwen3.6-27B-FP8, --kv-cache-dtype bf16):
- "capital of France" → "Paris"
- "17 plus 25" → "17 plus 25 equals **42**."
- "sentence about a cat" → "The cat curled up on the windowsill, basking in the warm
  afternoon sun."
- reasoning (train mph, step-by-step LaTeX), code (Fibonacci w/ docstring), knowledge
  (Jupiter/Saturn/Mars facts) — all coherent and correct.

REMAINING (non-blocking): (1) reshape_and_cache.cu:128 float2_to_fp8x2 KV-write x2 encode
still uses __NV_E4M3 — only hit on FP8 KV (we use bf16 KV); fix with paired scl_enc_fp8
if FP8 KV needed. (2) Same SCALE decode bug exists in other models' kernels
(qwen3.6-35b-a3b, nemotron-super-120b, all moe_*) — apply the same scl_fp8/scl_enc_fp8
fix when porting those. (3) Debug dumps (ATLAS_DUMP_LAYER_NORM/GDN, qkvz_qLAST) still
present, gated by env vars — remove for clean build.

LESSON: when A/B-comparing two kernels on the SAME platform, both can share an
upstream-decode bug and "match" while both being wrong. Verify against a KNOWN-GOOD
reference (real NVIDIA) or against first-principles (the probe), not just sibling kernels.

---

## #A6 (HIP merge Phase 2) — coherence merge ~8 GB post-weights over-alloc → OOM watchdog (FIXED 2026-06-03)

BRANCH: merge-coherence (159f80a merge + 6ca4665 HIP-fix). The dense-Qwen3.6-27B
coherence branch merged into the HIP port builds + serves, but the merged binary
consumed ~8 GB more during model construction than the port baseline
(port-safety-pre-merge), dropping post-construction free below the 2048 MB OOM
watchdog → the watchdog terminated the server before first request. Held across
util/ctx/max-prefill sweeps and with MTP off.

ROOT CAUSE: the coherence merge ACTIVATED the native-FP8 SSM in_proj/out_proj
prefill path (qwen35_dense.rs: builds a single-scale FP8 copy of qkvz [16384×5120]
and out_proj [5120×6144] per SSM layer via bf16_to_fp8). That is 80 MB + 30 MB =
110 MB × 48 SSM layers = ~5.3 GB of EXTRA weights the baseline never allocated.
gfx1151 cannot even dispatch this path (no cp.async fp8_gemm_n128), so 6ca4665
added HW flag `disable_fp8_ssm_prefill` (HARDWARE.toml) → build.rs emits
`cargo:rustc-env=ATLAS_HW_DISABLE_FP8_SSM_PREFILL`, gated in qwen35_dense.rs via
`option_env!(...).is_none()`.

THE GATE WAS DEAD. `cargo:rustc-env` from atlas-kernels/build.rs is CRATE-LOCAL —
it does not reach spark-model (which has no build.rs). So `option_env!` in
spark-model was always None → fp8_ssm_prefill stayed true → the ~5.3 GB FP8 SSM
prefill weights were allocated anyway. Serve log "SSM in_proj_qkv + out_proj via
native FP8 prefill GEMM" fired (baseline-premerge log never prints it). This is
the SAME dead-flag trap the SCALE-port notes warned about for ATLAS_HW_FORCE_BR32.

FIX (additive, SSOT/PCND): added crates/spark-model/build.rs — resolves the SAME
HARDWARE.toml (workspace/kernels/<ATLAS_TARGET_HW|gb10>/HARDWARE.toml, mirroring
atlas-kernels resolution) and re-emits the `[hardware]` gating flags as
cargo:rustc-env (disable_fp8_ssm_prefill → ATLAS_HW_DISABLE_FP8_SSM_PREFILL,
force_br32_prefill → ATLAS_HW_FORCE_BR32). Emitted ONLY when the key is present
and true; gb10 omits both keys → no env → option_env! None → NVIDIA/GB10 unchanged.
Added `toml = "0.8"` build-dependency (already in workspace lock). The single
authoritative value still lives only in HARDWARE.toml.

RESULT: serve log no longer prints the native-FP8 message; KV-cache
post-construction free rose from 7.1 GB (pre-fix, tiny 158 MB arena) to 11.8 GB
(2049-tok arena) / 10.1 GB (full 8193-tok arena at ctx 16384). Server reaches
"Listening" with MTP K2 at ctx 16384, no OOM kill. Coherent: primes
"2, 3, 5, 7, 11, 13, 17, 19" + "Paris". MTP K2 ~18 tok/s, K2 accept 90-96%.

This fix ALSO un-deads ATLAS_HW_FORCE_BR32 → gfx1151 prefill now routes to the
LDS-safe BR32 kernel (was dispatching the BR64 kernel that overflows the 64 KB
LDS cap per the HARDWARE.toml rationale) — strictly safer on this hardware.
