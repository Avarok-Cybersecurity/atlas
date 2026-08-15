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
