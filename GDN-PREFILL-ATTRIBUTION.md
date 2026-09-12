# GDN chunked-prefill attribution — 1×H100, Qwen3.8-27B-FP8 (#928)

Source: nsys round 9 (`nsys-r9-prefill`, cell XY, 2026-09-11); prefill A = 1193
tok / 368.263 ms busy union, prefill B = 4593 tok / 1163.475 ms. Both windows
show **96 launches** of each GDN kernel across the model's **48 linear-attention
layers** — 2 per layer, unexplained here and not load-bearing for any ratio.

## Geometry (derived, then confirmed)
`nk=16`, `nv=48`, `kd=vd=128`, `CHUNK=64`, `qk_stride = conv_dim = 10240`,
`head_repeat = 3`, state `h` FP32 `[nv][128][128]` (`ssm_h_dtype=f32` in the r9
serve flags). Derived from round 9's own projection shapes — ssm `in_proj_qkvz`
N=16384 = 2·nk·128 + 2·nv·128, `out_proj` K=6144 = nv·128 — and matched by
`crates/atlas-core/src/config/parsers/qwen4_exp_tests.rs`. `num_chunks` = **19**
at T=1193, **72** at T=4593.

## Per-launch table (µs = nsys total ÷ 96)
| kernel | grid | thr | T | µs/launch | GFLOP | **TFLOP/s** | MB | **GB/s** | bound |
|---|---|---|---|---|---|---|---|---|---|
| `chunk_delta_h_vfused` | `[48,1,1]` | 256 | 1193 | 1019.1 | 3.83 | **3.75** | 96.2 | 94 | latency |
| | | | 4593 | 3917.2 | 14.49 | **3.70** | 346.9 | 89 | latency |
| `chunk_fwd_o` | `[nt,48,1]` | 512 | 1193 | 209.8 | 3.35 | **16.0** | 89.7 | 427 | compute/TC |
| | | | 4593 | 749.0 | 12.71 | **17.0** | 340.0 | 454 | compute/TC |
| `recompute_wu` | `[nt,48,1]` | 256 | 1193 | 155.1 | 1.90 | **12.2** | 60.0 | 387 | solve-serial |
| | | | 4593 | 494.9 | 7.19 | **14.5** | 227.4 | 459 | solve-serial |

FLOPs/(chunk, head): delta-h `W·S` + `Kᵀ·duc` = 4.194 M; fwd_o `q·kᵀ` + `q·Sᵀ` +
`tril(kq)·uc` = 3.678 M; wu `k·kᵀ` + two forward substitutions = 2.081 M. Bytes:
delta-h 49 408 R + 49 152 W plus 131 072 B of h per head per launch; fwd_o
81 920 R + 16 384 W; wu 33 280 R + 32 896 W. H100 SXM5 roofline: 3.35 TB/s HBM,
67 TFLOP/s FP32 FMA, 989 TFLOP/s dense BF16 tensor core.

## The verdict: `chunk_delta_h_vfused` is latency-bound

* **2.7 % of HBM**, **5.6 % of FP32 peak**, **0.38 % of BF16 tensor-core peak** —
  and it issues **zero** MMAs: both per-chunk matmuls are scalar FP32 loops
  (`kernels/gb10/common/gated_delta_rule_fla.cu`, `cdh_vtile_core`).
* **Occupancy**: grid `[nv,batch] = [48,1]` = **48 CTAs on 132 SMs** (36 %),
  `__launch_bounds__(256,1)` = 8 warps of 64 slots = **4.5 % machine-wide warp
  residency** — and those 48 CTAs are each a serial chain of `nchunks` steps.
* **Per-chunk cost is flat in T** — 1019.1/19 = **53.6 µs**, 3917.2/72 =
  **54.4 µs** ≈ 95 000 cycles at 1.755 GHz, against a one-SM FP32 floor of
  16 384 cycles (9.3 µs) for the same 2.097 M MACs: **5.8× above its own one-SM
  floor**. That is the whole finding, and why linearity in T is not a memory wall.
* **Where the cycles go**: with `SPLIT=2, VT=1`, `KH=64`, per token `i` the
  `wsp` reduction is a **64-deep dependent FP32 FMA chain**, then a `__shfl_xor`
  butterfly, then 64 independent FMAs into `Snew` — 4096 dependent FMAs per
  thread per chunk with 8 warps to interleave, and live state `Sold[64] +
  Snew[64]` = **128 FP32 registers**, leaving no room to software-pipeline
  across `i`. Smem operands are read 2 bytes at a time.

`chunk_fwd_o` and `recompute_wu` are the control: same file, same dtypes, same
per-chunk data, but grid `[nchunks, nv, 1]` (912 / 3456 CTAs) and their big
matmuls already on `mma.sync` via `mma_gram` — **16–17** and **12–14.5 TFLOP/s**,
**4.4×** the spine's rate. What is left in them is scalar: fwd_o's triangular
`tril(kq)·uc` on 128 of its 512 threads (0.53 of its 3.68 MFLOP), and wu's two
forward substitutions (79–85 % of it, per the in-file 2026-08-22 measurement).

## FLA reference structure (for comparison; no code taken)

FLA's `chunk_gated_delta_rule` uses the same three-pass WY decomposition Atlas
mirrors: (1) chunk-parallel, build `T = (I + tril(diag(β)·K·Kᵀ, -1))⁻¹` and form
`W = T·(β·e^{g}·K)`, `U = T·(β·V)`; (2) `chunk_fwd_h`, serial over chunks,
`h_{c+1} = e^{g_last}·h_c + K̃_cᵀ·(U_c − W_c·h_c)`; (3) `chunk_fwd_o`,
chunk-parallel again. The difference is entirely in pass (2): FLA runs both
per-chunk `[64×128]×[128×128]` products as **BF16 tensor-core matmuls with FP32
accumulation** (`tl.dot`), keeping the state in the FP32 accumulator. Atlas does
the identical algebra in scalar FP32; (1) and (3) are already equivalent.

## The lever, and what it measured

`[defaults] gdn_prefill_tc` — **true on `kernels/hopper` since round 13**,
false on every other target, with `ATLAS_GDN_PREFILL_TC` overriding either
way — routes the spine to
`gated_delta_rule_chunk_delta_h_tcfuse_x2`: both per-chunk products on
`mma.sync.m16n8k16`, bf16 operands, f32 accumulator — and that accumulator IS
the recurrent state (64 registers per thread against the scalar spine's 128 of
live state). `h` stays f32 in memory; the decay math stays exact f32. Per CTA per
chunk: 512 MMAs for `W·S` (4 m-tiles × 16 n-tiles × 8 k-steps) + 512 for `Kᵀ·duc`
(8 × 16 × 4), against 8192 scalar FMAs per thread. Grid `[nv, batch]` and block
256 are unchanged.

⚠️ **The variable was PRESENCE-gated and is now grammar-gated**, so
`ATLAS_GDN_PREFILL_TC=0` means OFF where it used to mean ON. Every A/B in this
document ran it as `=1` and is unaffected. On Hopper `=0` is now the arm that
CHANGES anything: the family is the declared default there, and `=0` is the
whole-family kill switch — spine and both remnant twins, because the twins read
the same resolved bit. The kernel stays where it is —
`kernels/gb10/common/gated_delta_rule_chunk_tc.cu`, shared and validated on
GB10; the declaration gates the PROBE that loads it as well as the launch, so a
target with the lever off does not ask the kernel audit about a module nothing
can reach.

**Numerics contract, measured** (`native_gdn_chunk_prefill_microtest`, GB10,
nv=48, f64 CPU reference, 2026-09-11). Two operands are newly rounded to bf16:
`S_c` (Phase A's B operand) and `duc` (Phase B's). One limb costs **2.0–2.7e-3**
rel_rms on the f32 state — over budget — and splitting `S_c` alone moved it only
2.72e-3 → 2.44e-3: chunk 0 is exact, so the error is *injected* in Phase B,
because with gates ≈0.9 the chunk decay `exp(Σ₆₄ log g)` is ~1e-3 and `S_{c+1}`
is therefore almost entirely the `Kᵀ·duc` correction. The shipped arm carries a
second bf16 limb of **both**, at zero shared-memory cost (`S_c`'s residual
overwrites `St` in place; `duc`'s aliases the dead `Wp`):

| T | chunks | vfused (scalar) | tcfuse (1 limb) | **tcfuse_x2 (shipped)** |
|---|---|---|---|---|
| 256 | 4 | 0.196 ms / 4.11 TF/s | 0.113 / 7.13 (1.74×) | **0.124 / 6.49 (1.58×)** |
| 1193 | 19 | 1.050 ms / 3.64 | 0.489 / 7.83 (2.15×) | **0.486 / 7.87 (2.16×)** |
| 4593 | 72 | 3.969 ms / 3.65 | 1.764 / 8.22 (2.25×) | **1.730 / 8.38 (2.29×)** |
| | `h` rel_rms | 1.1e-7 | 2.0–2.7e-3 | **3.0–3.9e-6** |
| | `uc` / `S_c` rel_rms | 1.65e-3 | 2.1–3.1e-3 | **1.660e-3 = the spine** |
| | per-chunk drift | 1.02× | 1.07–1.18× | **1.02× = the spine** |

`uc` and `S_c` are **bf16 tensors**: the scalar spine itself measures 1.65e-3
on both, so that is the storage floor and a 1e-3 gate on them is unsatisfiable
by construction. The shipped arm lands on that floor to four digits — at the
output dtype it is indistinguishable from the scalar spine — and its f32-state
deviation is **250–700× inside the 1e-3 budget**, with drift flat over 72 serial
chunks. `ptxas -v`: 243 regs / 0 spills at `sm_90a`, 255 / 0 at `sm_121a`,
255 / 20 B spill at `sm_100a`.

**What this does NOT establish.** GB10 is not H100 (~48 SMs against 132): the
48-CTA grid that starves Hopper nearly fills it, so the H100 speedup should be
*larger* — a prediction, not a receipt. And the standing lesson on this kernel is
that a spine change can read cos=1.0000 and still cost 1.4 BFCL points (the
SPLIT=4 note in `gated_delta_rule_fla.cu`): promotion to default needs the
ssm-poisoning tripwire. Hence opt-in — until round 13 ran it on the hardware;
see **The H100 answer** below, which is why `kernels/hopper` now declares the
row true and `kernels/gb10` still does not.

Not taken: (b) fusing `recompute_wu` into the delta-h pass — rejected, different
grids, fusing would drag `wu` to 48 CTAs; (c) an H100 DV-split to lift 48 CTAs
toward 132 — the in-file GB10 verdict (2026-06-25, `gdn_cdh_vblock_microtest`:
0.71×/0.65×/0.34× at VTILES=2/4/8, bit-parity 18/18) says this *loses* while the
kernel is latency-bound, worth re-testing now the MMA rewrite changed that bound;
(d) `tril(kq)·uc` in `chunk_fwd_o` on tensor cores, at most 0.53/3.68 of 6.2 %.

---

# The two remnants, attributed — and the Hopper twins (#928, 2026-09-11)

Same trace, same geometry (`nk=16`, `nv=48`, `kd=vd=128`, `CHUNK=64`; 912 CTAs
at T=1193, 3456 at T=4593; 96 launches each). The section above named
`chunk_fwd_o`'s `tril(kq)·uc` and `recompute_wu`'s forward substitutions as
what is left once the spine moved to tensor cores. Here is what they cost.

## `chunk_fwd_o` — 3.678 MFLOP per (chunk, head), 14.5% of it scalar

| term | shape | MAC | unit | threads |
|---|---|---:|---|---|
| `q·kᵀ` | M=64 N=64 K=128 | 524 288 | `mma.sync` | 128 of 512 |
| `q·S_cᵀ` | M=64 N=128 K=128 | 1 048 576 | `mma.sync` | 128 of 512 |
| `tril(kq)·uc` | Σ_i (i+1) × 128 = 2080×128 | 266 240 | **scalar f32** | 128 of 512 |

`mma_gram` hardwires M=64 across four warps and is fenced to `tid < 128`; the
triangular loop is fenced to `tid < v_dim`, also 128. **Twelve of sixteen warps
issue no arithmetic at all** — they stage 96.5 KB of shared memory and idle.
Bytes per CTA: 82 176 R (`q` 16 384, `k` 16 384, `uc` 16 384, `S_c` 32 768,
`gc` 256) + 16 384 W = 98 560, i.e. 89.9 MB per launch and **427 GB/s of
H100's 3.35 TB/s (12.7%)** at 16.0 TFLOP/s (1.6% of bf16 tensor-core peak,
24% of FP32 FMA peak — the ratio that says the scalar term sets the rate).

What is serial: the inner `for l <= i` is a DEPENDENT f32 FMA chain, 2080 FMAs
per thread with two shared-memory operands each (`kq` broadcast, `ucb` 2-way
conflicted at a 64-element bf16 stride). At a 4-cycle FMA latency that is a
~8 300-cycle floor per CTA against 209.8 µs / 6.9 waves ≈ 53 300 cycles — an
ESTIMATE, not a measurement; the only measured decomposition of this kernel is
the nsys total.

Occupancy is the second half of the finding and it IS measured: `ptxas
-arch=sm_90a --fmad=false` gives the parent **104 registers at 512 threads**,
so it runs **one CTA per SM** — 16 warps of 64 slots, 25% — because two would
need 64 registers or fewer.

## `recompute_wu` — 2.081 MFLOP per (chunk, head), 49.6% of it scalar

| term | shape | MAC | unit | threads |
|---|---|---:|---|---|
| `K·Kᵀ` | M=N=64 K=128 | 524 288 | `mma.sync` | 128 of 256 |
| `(I+L)U = βV` | 2016 × 128 | 258 048 | **scalar f32** | 128 of 256 |
| `(I+L)W = βe^{gc}K` | 2016 × 128 | 258 048 | **scalar f32** | 128 of 256 |

Bytes per CTA: 33 280 R (`k` 16 384, `v` 16 384, `gate` 256, `beta` 256) +
33 024 W (`W` 16 384, `U` 16 384, `gc` 256) = 66 304 → 60.5 MB per launch,
**390 GB/s (11.6% of HBM)** at 12.2 TFLOP/s. (The table above records 32 896 W;
the sum of the three writes is 33 024 and the 0.4% difference changes no ratio.)

Half the arithmetic, 79–85% of the time — that split is MEASURED, by the
in-file solve-removed probe of 2026-08-22 (prologue 14.3/16.4/20.5 µs against
68.9/96.6/140.8 µs total at nt=1/16/64). What is serial: one thread per
right-hand-side column walking 64 rows, and `acc[64]` indexed by a runtime row,
which ptxas puts in LOCAL memory — **512 bytes of stack frame at sm_90a**. The
right-looking block of 16 already cut that traffic ~6-8x (1.95–2.28x measured);
what remains is the shape, not the blocking. 87 registers at 256 threads =
2 CTAs/SM = 512 threads/SM, the same 25% warp residency as `fwd_o`.

## The twins

`kernels/hopper/common/gdn_fwd_o_hopper.cu` and `..._recompute_wu_hopper.cu`,
new stems (not same-stem overrides: the parents share a 2105-line file with
twelve other entry points), declared with their shared `gdn_prefill_hopper.cuh`
in `kernels/hopper/HARDWARE.toml`'s `[kernels] overrides` — the SSOT for which
kernels this target owns rather than inherits, and what
`crates/atlas-kernels/tests/inherited_overrides.rs` checks them against as
ADDITIONS (a new stem must bring entry points gb10 does not declare).

**One lever for the family.** The twins are selected by the SAME bit as the
tensor-core state spine: `[defaults] gdn_prefill_tc`, with
`ATLAS_GDN_PREFILL_TC` overriding under the 2026-09-11 grammar. It is resolved
ONCE per prefill, in `ops::gdn_prefill_fla`, and handed to the twins' launcher
as a value — not re-read from the environment there. That matters in exactly
one direction: `ATLAS_GDN_PREFILL_TC=0` is an explicit OFF, and a presence
check would have turned the twins ON for it while the spine stayed off, which
is a prefill that is neither leg of an A/B.

`ATLAS_NO_GDN_PREFILL_TC_REMNANTS=1` is the ONE-VARIABLE A/B that separates
them: it keeps the spine and pins `wu`/`fwd_o` to their parents. It is
presence-gated, like the other `ATLAS_NO_*` kill switches, and it is documented
beside the `gdn_prefill_tc` row in `kernels/hopper/HARDWARE.toml` because that
row is where an operator reading the target's defaults will look for it.

* `fwd_o`: every product re-tiled 4 m-tiles × 4 n-quarters so all 16 warps
  compute; `kq` masked, decayed and split to two bf16 limbs in the C fragment
  where it is produced; the triangular term run as a masked
  [64×64]×[64×128] MMA (0.524 M MAC against the triangle's 0.266 M — 2× the
  arithmetic, no dependent chain); `q·S_cᵀ` kept in the f32 accumulator instead
  of a bf16 round-trip. 97 536 B of smem (under the parent's 98 816, because the
  `kq` lo limb aliases the dead `sk`) and **64 registers, 0 spill**, so
  `__launch_bounds__(512, 2)` is free and resident CTAs double.
* `wu`: blocked triangular solve — `X_j ← T_jj·B_j`, then `B_i ← B_i − L_ij·X_j`
  for i > j — both on `mma.sync`, 16 columns per warp so each holds its [64×16]
  panel in one C fragment and the solve is warp-local (`__syncwarp`, never
  `__syncthreads`). 328 k MAC per solve against the triangle's 258 k (1.27×).
  `T_jj = (I+L_jj)^{-1}` is built once per (chunk, head) by the parent's own
  exact f32 forward substitution — 4 × 680 MAC, 0.4% of the kernel — and shared
  by both solves. `K·Kᵀ` and the `L` build fuse (the Gram is symmetric, so the
  element the parent re-read from a 16 KB f32 buffer is the fragment's own).
  79 104 B of smem, **114 registers and 0 bytes of stack frame** against the
  parent's 512. `(512, 2)` was tried and rejected: it fits 64 registers only by
  spilling 84 bytes, which is the same defect under a different name.

Both also replace the parents' 128/64-element operand strides with padded
136/72/24, because the parents' put all eight `grp` rows of every fragment read
on one bank group.

**What was NOT established when the twins landed.** No H100 had run either of
them: everything above was a compile-time receipt (`ptxas -v` at sm_90a, CUDA
13.0, `--fmad=false`, cross-compiled on gx10-a309 2026-09-11) plus host
simulation of the index maps and of the limb arithmetic against an f64
reference (`crates/spark-model/src/layers/ops/ssm_gdn_remnants*_tests.rs`), and
no speedup was claimed, predicted or implied. Round 13 supplied the hardware
receipt; it is below.

---

# The H100 answer (round 13, 2026-09-11) — the family is Hopper's default

1xH100 80GB HBM3, Qwen/Qwen3.8-27B-FP8 @ `3c0379030` (196 sm_90a kernels),
`h100-round13-report.md`. Three serve cells on ONE binary, one variable apart:
**A** the control (family off), **T2** `ATLAS_GDN_PREFILL_TC=1
ATLAS_NO_GDN_PREFILL_TC_REMNANTS=1` (spine only), **T1** `ATLAS_GDN_PREFILL_TC=1`
(spine + both twins). Frozen ladder, temp 0 / seed 42, 1 warmup + 3 reps,
`MAX_BATCH_SIZE=16`, client-side streaming TTFT.

| metric | A (off) | T2 (spine only) | **T1 (whole family)** | T1 vs A | T1 vs T2 = the twins |
|---|---|---|---|---|---|
| 1193/256 C=1 TTFT | 269.1 ms | 184.7 | **162.4** | **-39.6%** | **-12.1%** |
| 1193/256 C=16 agg | 429.05 tok/s | 498.71 | **521.19** | **+21.5%** | +4.5% |
| 1193/256 C=16 TTFT | 2 279.9 ms | 1 571.2 | **1 372.8** | **-39.8%** | -12.6% |
| 4593/512 C=1 TTFT | 889.3 ms | 565.0 | **491.5** | **-44.7%** | **-13.0%** |
| 4593/512 C=16 agg | 308.86 tok/s | 383.52 | **405.91** | **+31.4%** | +5.8% |
| 4593/512 C=16 TTFT | 7 524.7 ms | 4 786.1 | **4 145.5** | **-44.9%** | -13.4% |

T1's short-prompt C=1 TTFT of **162.4 ms beats vLLM 0.28.0's 179 ms on the same
box** — the first metric in this campaign where Atlas leads.

**Quality.** Coherency 4/4 on T1 (`'391'`, `'Tokyo'`, `'rotaregirfer'` all OK);
determinism **8/8 md5-identical across 3 runs**; zero content-loop, fuzzy or
SimHash watchdog fires. Against the control, 5 of 7 coherency outputs are
md5-identical and the two that differ differ cosmetically ("we can break the
multiplication down" -> "you can…", "Thus" -> "Therefore"). T2 — the spine
alone — is md5-identical to A on all 7, so the divergence belongs entirely to
the twins and is at the bf16 storage floor.

**Per-kernel attribution, nsys, T=4593 C=1, 96 launches each.** Two captures,
identical recipe, `ATLAS_NO_GDN_PREFILL_TC_REMNANTS` the only difference:

| kernel | T2 = gb10 parent | **T1 = Hopper twin** | speedup |
|---|---|---|---|
| `chunk_fwd_o` -> `chunk_fwd_o_hopper` | 71 548.4 µs (13.46% of busy) | **16 715.5 µs (3.63%)** | **4.28x** |
| `recompute_wu` -> `recompute_wu_hopper` | 47 388.6 µs (8.91%) | **29 558.3 µs (6.43%)** | **1.60x** |
| `chunk_delta_h_tcfuse_x2` (spine, same kernel both cells) | 51 775.3 µs | 52 060.3 µs | 0.99x *(control)* |
| **GDN family total** | **170 712.3 µs (32.11%)** | **98 334.1 µs (21.38%)** | **1.74x** |
| prefill GPU busy (union) | 531.6 ms | 459.9 ms | **-71.7 ms** |

The shared spine kernel landing within 0.55% is the internal control that
nothing else moved: the -71.7 ms of prefill busy is accounted for by the twins'
-72.4 ms, every non-GDN row is unchanged to under 1% (`nvjet_sm90_…1x2_h`
162 939 vs 162 961 µs; `per_token_group_quant_fp8` 47 659 vs 47 660 µs), and it
shows up one-for-one in client TTFT (565.0 -> 491.5 ms). Against the pre-TC
baseline for this shape — GDN chunk recurrence 376 ms, 32% of a 1 171 ms
prefill — the family is now 98.3 ms, 21% of a 460 ms prefill: a **3.8x cut**.

**Scope.** This is an H100 receipt and it changes ONE file's row:
`kernels/hopper/HARDWARE.toml`. `kernels/gb10` and `kernels/b200` still declare
`gdn_prefill_tc = false` — GB10 has ~48 SMs, the count the 48-CTA grid nearly
fills, so the Hopper margin does not transfer by argument, and B200 has no
serving receipt of any kind.

**The oracle.** `native_gdn_prefill_remnants_microtest` is still where the
twins' numerics verdict comes from; it SKIPS on every image but
`kernels/hopper`. Round 13 was its first hardware run and it panicked before its
first comparison on a harness unit error (`take(full, rows, per)` re-applies
`NV`, and the caller passed `nt * NV`), so round 13's per-kernel numbers above
are nsys, not the microtest — a better measurement of speed (the live engine at
the real T) and no measurement at all of numerics, which live only in that
example. The unit error is fixed and pinned by a host test at the microtest's
own geometry (`examples/common/gdn_remnants.rs`).
