# Row-wise FP8 for mixed-precision checkpoints

Branch `feat/fp8-rowwise-mixed-precision`, stacked on `feat/video-support` (PR #516).
Measured on dgx-00, 2026-08-15.

## The problem

`unsloth/Qwen3.8-27B-NVFP4` is `format = mixed-precision`:

| group | format | modules |
|---|---|---|
| group_0 | FP8 E4M3, `strategy = channel` (per-row) | `self_attn.{q,k,v,o}_proj`, `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`, `lm_head`, layers 56-63 MLP |
| group_1 | NVFP4, `tensor_group` | `mlp.{gate,up,down}_proj` |

Atlas serves the FP8 group by **dequantising to BF16 and re-quantising to NVFP4** —
8-bit weights served at 4 bits. Visible as `quantize_to_nvfp4` lines in the serve log.

That fallback is deliberate and correct as far as it goes: the native `w8a16` kernels
index `block_scale[n/128, k/128]`, so a per-row `[N,1]` scale would hand 127 of every
128 rows another row's multiplier — in-bounds, so it would not fault, it would just be
wrong. `weight_loader/qwen35_dense.rs::proj_is_fp8_any_scale` refuses it for exactly
this reason.

**The cost is not hypothetical.** The GDN loader arm's own comment records the last
time this was measured, on a checkpoint whose toolchain deliberately kept the SSM
projections high-precision: BFCL-ST non_live **85.4 → 76.6**, about 7 points. And on
the video benchmark's hardest leg, the double-quantised checkpoint answered
`Red, Blue` where the natively-loaded FP8 build of the same weights managed
`Red, Blue, Yellow`.

## What is already in the tree

* `layers/ops/dispatch_proj.rs::cublas_fp8_rowwise_proj` — a cuBLASLt **row-wise** FP8
  GEMM, GB10-supported, documented at ~1.8× the BF16 path (152 vs 85 TF). Used today
  by the SSM prefill projection.
* `weight_map/quantize_fns.rs::load_fp8_weight` — reads a `[N]` scale (widening BF16 →
  F32) and tags `WeightQuantFormat::Fp8PerRow`. **It has no callers.**
* `ATLAS_GDN_BF16_WEIGHTS=1` — the BF16 mitigation for the GDN half, default-off,
  "gated for a clean A/B + KL-drift gate before flipping the default".

## Done on this branch

**1. Row-wise passthrough** (`10f90f57`). `requant_weight_rowwise_fp8_cached` converted
block-fp8 → BF16 → row-wise fp8 unconditionally. A checkpoint that is *already*
row-wise now passes its own pointers through untouched — converting it would be
fp8 → bf16 → fp8, spending precision to produce what it already is. The conversion
path now asserts `Fp8BlockScaled` (both current callers carry it, so this is a no-op
today). The decision is a pure `rowwise_pair_passthrough` so it is testable on the
CPU-only CI.

**2. A/B harness** (`scripts/fp8_rowwise_ab.py`). `collect` / `compare`, deliberately
sequential — `kl_coherence_gate.py` wants two live serves, which on one unified 121 GB
pool is the co-tenancy that has taken this box down before, and it corrupts the
throughput half of the measurement long before it OOMs.

## Measured: the BF16 lever is not the answer

`ATLAS_GDN_BF16_WEIGHTS=1` vs baseline, unsloth/Qwen3.8-27B-NVFP4, same flags:

| | baseline (NVFP4 requant) | GDN BF16 | |
|---|---|---|---|
| prefill | 507 tok/s | 137 tok/s | **−72.9%** |
| decode | 5.3 tok/s | 5.0 tok/s | −5.0% |
| dark-green probe | `Red, Blue` | `red, blue, yellow` | quality ↑ |
| pre-KV memory | 59.3 GB | 57.3 GB | −2.0 GB |

Drift, KL over the shared prefix: token match 74.1%, mean KL 0.0083, p99 0.115.

So the BF16 route buys the quality back and pays **three quarters of prefill** for it.
That is the argument for the row-wise FP8 route rather than the BF16 one: same ≥FP8
precision, at FP8 memory and FP8 GEMM speed.

## The blocker the reconnaissance found

**There is no per-row FP8 GEMV for decode.** The whole `w8a16` family — `w8a16_gemv`,
`w8a16_gemv_batch4`, `w8a16_gemv_fused`, `w8a16_gemm*` — is block-scaled
(`kernels/gb10/common/w8a16_gemv.cu:143`: `block_scale[n_block * k_blocks + k_block]`).
The row-wise path is cuBLASLt, and cuBLASLt is a prefill instrument.

Note also that `WeightQuantFormat::Fp8PerRow`'s doc comment claims it is "Consumed by
`w8a16_gemv` / `w8a16_gemm`". **That is false** and worth fixing separately: those
kernels index a block grid. Nothing currently produces an `Fp8PerRow` weight, so no
live path misindexes today — but `quant_dispatch` asserts nothing, so the first caller
to produce one would silently get garbage.

## Prior art — this is the FOURTH instance of one family

Swept all 45 open PRs plus the merged history. The defect family is *"the
checkpoint ships a module at higher precision and the loader quantises it down
to NVFP4 anyway"*, and it keeps recurring in different modules:

| | module | checkpoint | status |
|---|---|---|---|
| **#257** | GDN `in_proj_qkv` / `in_proj_z` / `out_proj` | nvidia Qwen3.6-27B-NVFP4 (`Nvfp4Variant::Standard`) | **MERGED** 2026-07-10 |
| **#484** | MoE shared expert | Laguna-XS-2.1 | open |
| **#406** | MLP (native BF16 on disk, `Bf16Raw`) | gemma-4-E2B | open |
| *this* | attention q/k/v/o + GDN + lm_head, **per-channel FP8** | unsloth Qwen3.8-27B-NVFP4 (`CompressedTensors`, `mixed-precision`) | — |

**#257 is the closest, and it is already in this tree.** Its cherry-pick onto
the Strix branch (`38c9dea`, carried by the still-open **#336**) states the
cost in the same terms this branch found: *"fixes the double-quant that
regressed non_live to ~76"*, on a checkpoint that is mixed-precision because
"modelopt keeps the SSM projections high-precision". #336 also names the lever
and the expected evidence — serve with `ATLAS_NO_GDN_FP8=0` and look for
`SSM in_proj_qkv ... native`, "no lossy FP8→BF16→NVFP4 double-quant".

So the native-FP8 GDN path exists and works — for **block-scaled and
per-tensor** scales. It is the **per-channel `[N,1]`** form that is still
refused, which is exactly this checkpoint's form. That is the remaining gap,
and no open PR claims it.

The remedy every sibling chose is the same one this branch is taking: **load
what the checkpoint ships, in its own format, rather than down-converting.**
#484 adds a native packed-NVFP4 loader arm; #406 keeps the BF16 on disk. Given
three one-off arms already exist, a reviewer may reasonably ask for the
general form instead of a fourth — worth raising before building more arms.

### Open PRs that touch the same code (conflict watch)

* **#400** `feat(fp8): load F8_E8M0 block scales` — `weight_map/loaders_fp8.rs`
  + `gemm_quant.rs`. Same file, and architecturally the same shape as this
  work: accept a scale representation the checkpoint ships and adapt it once at
  load. Closest thing to a template.
* **#404** SSM batched NVFP4 decode (`ssm_batched.rs`) — the QKVZ/out_proj
  decode arm any decode-side wiring here would touch.
* **#474** `w8a16_gemv_batch4` accumulation order + a bit-parity microtest —
  the w8a16 family, and the microtest is the pattern a new per-row GEMV would
  need to copy.
* **#475** Nemotron milestone B — flags a `proj_batch_min()` hazard where FP8
  returns 2 without checking `w8a16_gemv_batch4/16` actually resolved.
* **#519** decode-GEMV LUT staging (w4a16) — adjacent decode-GEMV work.

For the accuracy bar, **#514 / #495** show the house standard: BFCL with
ratcheted floors and a measured ±0.40 noise band. The KL harness here sizes a
change; it does not replace that.

## ✔ LANDED — via BF16, because the FP8 GEMM is dead on GB10

`ATLAS_FP8_ROWWISE=1` works and is measured. Getting there killed the
assumption the plan rested on.

### The FP8 GEMM does not work on this hardware

`cublas_fp8_rowwise_proj` ends in `cublaslt::fp8_gemm_act_weight_t_rowwise`,
which declares both scales `SCALE_MODE_OUTER_VEC_32F`. On sm_121
`cublasLtMatmulAlgoGetHeuristic` returns **status 15 (NOT_SUPPORTED)** and
prefill 400s. Padding M to 16 — which that call also needed, and now does —
does not change it.

**Control**: the BLOCK-scaled `Qwen/Qwen3.8-27B-FP8` with `ATLAS_CUBLAS_FP8=1`,
this branch's arm inert, reaches the same call through the requant path and
fails identically. It is the GEMM, not per-row weights.

**Its sibling is worse**: `ATLAS_FP8_W8A8=1` (block-scaled cuBLASLt) passes the
heuristic and returns `"kililililil…"` to a plain prompt.

So the cuBLASLt FP8 prefill family is dead code on GB10 — one arm errors, one
is silently wrong — behind default-off flags nothing in the repo sets
(`ATLAS_CUBLAS_FP8` appears exactly once, its own definition). Its docstring's
"~1.8× the bf16 path (152 vs 85 TF)" does not reproduce here.

### So: dequantise once to BF16, multiply with cuBLASLt BF16

Still no re-quantisation — FP8 E4M3 is exactly representable in BF16 — and no
new kernel. The same `dequant_fp8_blockscaled_bf16` serves both layouts:
`block_n = 1, block_k = K, sk = 1` makes its index `scale[n]`.

Measured on `unsloth/Qwen3.8-27B-NVFP4`, same box and flags, vs the NVFP4
baseline:

| | baseline | `ATLAS_FP8_ROWWISE=1` | `ATLAS_GDN_BF16_WEIGHTS=1` |
|---|---|---|---|
| prefill | 507 tok/s | **585 tok/s (+15.5%)** | 137 tok/s (−72.9%) |
| decode | 5.3 tok/s | 5.3 tok/s | 5.0 tok/s |
| dark-green probe | `red, blue` | `red, blue, yellow` | `red, blue, yellow` |

Drift vs baseline: token match 82.5%, mean KL 0.0054, p99 0.039 (shared
prefix). Gates with the flag on: **vision-fidelity 14/14 + 3/3, video-fidelity
13/13**, both control held.

The BF16 lever buys the *same* precision back and costs three quarters of
prefill; this buys it and gains 15%. The GEMM is the difference, not the
precision — which is also why the earlier "BF16 is not the answer" conclusion
was about `dense_gemm`, not about BF16.

### What is still on the table

FP8 *memory* as well as FP8 precision — this holds a BF16 copy of the GDN
projections, which is why the win is speed and quality rather than footprint.
That needs a per-row FP8 GEMM that works on sm_121, with a bit-parity
microtest in the shape of PR #474's. Upside, not a blocker.

Decode is still NVFP4 (unchanged tok/s above). A decode-side fold needs the
per-row `w8a16_gemv` variant.

## Measured decode, and the dispatch flags nobody sets

**Decode**, 3 reps x 400 forced tokens (`ignore_eos`), spread ±0.02 tok/s:

| config | decode |
|---|---|
| baseline (NVFP4 decode) | 5.26 tok/s |
| `ATLAS_GDN_BF16_WEIGHTS=1` + cuBLASLt | 5.08 tok/s (−3.4%) |
| `ATLAS_FP8_ROWWISE=1` (NVFP4 decode kept) | ~5.3, unchanged |

The −3.4% is not the GEMM change: under the BF16 lever the DECODE weights are
BF16 too, 4× NVFP4's bytes, and decode is bandwidth-bound. The row-wise fold
keeps NVFP4 for decode and so keeps the decode rate — the one axis where it
still wins, bought with +19 GB.

An earlier 5.1-vs-5.3 figure in this doc came from a 49-token sample (the
"count to 200" prompt stopped early); that was too small to separate from
noise and is superseded by the table above.

### Every CUTLASS / cuBLAS dispatch flag is OFF in all of these runs

`GemmDispatch::from_env` turns each on only with `ATLAS_*=1`, and none were
set — so `cublas_gemm`, `cutlass_gemm` and the whole `cutlass_nvfp4_*` family
were false throughout. `fp8_blockscaled_prefill` is the one inverted default
(on unless `ATLAS_FP8_SINGLE_SCALE=1`).

That is not a flaw in the A/B: **the qwen3.6-27b and qwen3.8-27b serving
recipes set no `ATLAS_*` either**, so this is exactly the config the gates and
production run. The numbers are like-for-like.

It does mean the absolute figures are the unflagged floor, and two things are
worth measuring separately:

* **The documented prefill stack for this family is GDN, not CUTLASS.** The
  `qwen3.6/qwen3.6-27b-nvfp4-prefill-record` recipe pins
  `ATLAS_FFN_NVFP4_MMQ=1`, `ATLAS_GDN_REGRESIDENT=1`, `ATLAS_GDN_FLASHINFER=1`
  and `ATLAS_GDN_LIB=…/libatlasgdn.so`. FlashInfer-GDN is worth +17-20% there,
  and it FAILS OPEN — without `ATLAS_GDN_LIB` and the CUTLASS DSL on
  `LD_LIBRARY_PATH` it silently falls back, costing 40-50% of prefill at
  concurrency, which C=1 cannot see.
* **`cutlass_nvfp4_qkvz` is unmeasured on this target.** It would consume the
  NVFP4 copy — i.e. the double-quantised weights — so it trades the precision
  this branch is about for whatever the tuned kernel buys. Worth an A/B before
  anyone assumes either way.

The row-wise arm returns early and therefore SHADOWS those arms. That is
deliberate (they all read the NVFP4 copy), but it now warns once when it
shadows an enabled flag, so an operator whose CUTLASS setting silently did
nothing finds out.

## ★ The dispatch flags dwarf all of it (qwen3.8-27b, 2026-08-15)

Prefill throughput, cold prompts, `scripts/prefill_probe.py`, 3 reps, median.
Same box, same binary, same serve; only the env differs.

| leg | C=1 | C=4 | pre-KV | dark-green probe |
|---|---|---|---|---|
| baseline (no flags) | 507 | 740 | 60.7 GB | `Red, Blue` |
| GDN stack | 561 (+10.7%) | 860 (+16.2%) | 59.1 GB | `red, blue` |
| CUTLASS NVFP4 | 633 (+24.9%) | 1039 (+40.4%) | 60.1 GB | `red, blue, yellow` |
| **GDN + CUTLASS** | **717 (+41.4%)** | **1295 (+75.0%)** | 59.4 GB | `RED, BLUE, YELLOW` |

Legs:

* GDN — `ATLAS_GDN_FLASHINFER=1`, `ATLAS_GDN_REGRESIDENT=1`, `ATLAS_GDN_LIB=…/libatlasgdn.so`,
  and the CUTLASS DSL lib dir on `LD_LIBRARY_PATH`. **Verified engaged** from the
  boot log — `ATLAS_GDN_FLASHINFER: FlashInfer GDN kernel loaded (opt-in)` — not
  the silent fallback, which is the whole hazard with this flag.
* CUTLASS — `ATLAS_CUTLASS_NVFP4_GEMM=1` plus `ATLAS_CUTLASS_NVFP4_SSM_OUT=1`,
  which the umbrella deliberately does NOT imply.

Three things follow.

**These flags are worth several times the weight-precision work.** +75% at C=4
against +11.9% (BF16+cuBLASLt) and +15.5% (row-wise FP8), and they compose
almost additively with each other. Nothing in the repo sets them — not the
serving recipes, not the gates.

**Memory is not the trade here.** CUTLASS repacks the existing NVFP4 weights
into its own layout rather than adding a copy, so pre-KV moves by about a
gigabyte in either direction across all four legs — noise at this scale, and
nothing like the +19 GB the row-wise fold costs.

**The sensitivity probe IMPROVED under CUTLASS**, from `Red, Blue` to
`red, blue, yellow`, even though it consumes the same double-quantised NVFP4
weights. That was not expected and is not explained; the probe is a coarse
oracle and this wants the real gates before anyone reads a quality claim into
it.

Agentic smoke, run from the TUI on the GDN+CUTLASS serve: **PASS** — 1/1
webserver_ok, 1/1 followed_directions, Σwall 195 s, 6/6 steps over 11 turns and
13 tool calls. That is `iterations = 1`, so it is a smoke test and NOT the gate,
which pins exactly 10. Per-iteration wall is in line with the recipe's 2026-08-14
reference (1925 s over 10 = ~192 s), so the flags cost nothing agentically.

### Still unmeasured

* **Decode under CUTLASS.** The run was killed mid-flight to free the box; no
  number, and none is guessed here.
* **vision-fidelity / video-fidelity on the combined stack.** Needed before the
  probe's apparent quality gain means anything.
* **The full agentic gate** at the pinned 10 iterations.
* Everything above is qwen3.8-27b only. qwen3.6-27b is untested with these flags.

## Next steps, in order

1. **Fix the stale `Fp8PerRow` doc comment** and add a `scale_format.expect(...)` at
   the `quant_dispatch` FP8 arms, so the trap cannot be walked into.
2. **Prefill-only wiring first.** Load the mixed-precision GDN projections via
   `load_fp8_weight` (already produces `Fp8PerRow`) and route prefill through
   `cublas_fp8_rowwise_proj`, keeping the NVFP4 copy for decode. Mixing precision
   across phases is already an accepted pattern here — the native-FP8 SSM path logs
   "NVFP4 kept as structural fallback for decode batch paths". Behind
   `ATLAS_FP8_ROWWISE=1`, default-off.
3. **Measure** with `scripts/fp8_rowwise_ab.py` + the vision/video gates. The
   hypothesis to kill: that row-wise recovers the BF16 quality gain without the
   −72.9% prefill.
4. **Decode kernel**, if step 3 justifies it: `w8a16_gemv_rowwise` indexing `scale[n]`.
   Mechanically a small edit to `w8a16_gemv.cu`, but it needs its own numeric test
   against a BF16 reference before anything dispatches to it.
5. **Attention q/k/v/o** last — same treatment, bigger blast radius.

## Harness limitation worth knowing

KL is scored **only up to the first divergence**. These are free-running generations:
once one token differs the two sides are continuing different sentences, and comparing
distributions across different contexts measures nothing about precision. Scoring past
that point reported mean KL ~5 nats with p99 pinned at the `-30` floor — numbers that
look alarming and mean nothing. The sharper instrument is teacher-forced logprobs over
one fixed token sequence (`prompt_logprobs`), where both sides see identical context at
every position; that is the upgrade this harness wants.
