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

`ATLAS_GDN_PREFILL_TC` (presence, default OFF) routes the spine to
`gated_delta_rule_chunk_delta_h_tcfuse_x2`: both per-chunk products on
`mma.sync.m16n8k16`, bf16 operands, f32 accumulator — and that accumulator IS
the recurrent state (64 registers per thread against the scalar spine's 128 of
live state). `h` stays f32 in memory; the decay math stays exact f32. Per CTA per
chunk: 512 MMAs for `W·S` (4 m-tiles × 16 n-tiles × 8 k-steps) + 512 for `Kᵀ·duc`
(8 × 16 × 4), against 8192 scalar FMAs per thread. Grid `[nv, batch]` and block
256 are unchanged.

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
ssm-poisoning tripwire. Hence opt-in.

Not taken: (b) fusing `recompute_wu` into the delta-h pass — rejected, different
grids, fusing would drag `wu` to 48 CTAs; (c) an H100 DV-split to lift 48 CTAs
toward 132 — the in-file GB10 verdict (2026-06-25, `gdn_cdh_vblock_microtest`:
0.71×/0.65×/0.34× at VTILES=2/4/8, bit-parity 18/18) says this *loses* while the
kernel is latency-bound, worth re-testing now the MMA rewrite changed that bound;
(d) `tril(kq)·uc` in `chunk_fwd_o` on tensor cores, at most 0.53/3.68 of 6.2 %.
