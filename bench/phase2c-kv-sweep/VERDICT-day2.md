# Phase 2c Day 2 — Phase 2b kernel bisects all NEGATIVE

## TL;DR

Tested all three Phase 2b kernel changes individually (compile-time
`#ifdef` gates + runtime env-var gate). None individually explains
the 0.5% mean cosine gap vs the May 23 `rne` reference dump. The
bench is deterministic (bit-identical reruns) but the rne reference's
exact image provenance is unclear — its filesystem timestamps predate
any image still on disk, so cross-comparison is partly apples-to-pears.

## Bisect runs

All bisects run on a fresh build per config (cargo cache hot, ~40s
build), same probe, same HF reference. Each replaces ONE Phase 2b
change with the pre-Phase-2b alternative:

| Config | What changed | mean | min @ L35 | Δ vs baseline |
|---|---|---|---|---|
| `fp8-baseline` | current build (RNE + `__expf` + FP16 P×V) | 0.9615 | 0.9179 | — |
| `phase2c-fp8-poly` | `-DATLAS_FAST_SOFTMAX_EXP=1` (polynomial sw_exp) | 0.9617 | 0.9184 | +0.0002 |
| `phase2c-bf16pv` | `-DATLAS_DISABLE_FP16_PV=1` (BF16 P×V revert) | 0.9619 | 0.9172 | +0.0004 |
| `phase2c-rne-off` | `ATLAS_DISABLE_RNE=1` (truncate dequant) | 0.9569 | 0.8990 | **−0.0046** |
| `rne` (May 23 ref) | unknown exact image (filesystem ts predates all current images) | 0.9668 | 0.9290 | +0.0053 |

## What this means

1. **`__expf` is exonerated.** Reverting to the FA4 polynomial gives
   essentially identical cosine (+0.0002 mean). The "polynomial was
   masking deeper drift" memory note doesn't reproduce in this bench.

2. **FP16 P×V is exonerated.** Reverting to BF16 P×V also moves
   essentially nothing (+0.0004 mean). The 8×-finer probability
   precision doesn't surface at the dump granularity.

3. **RNE is helping, not hurting.** Disabling RNE (returning to
   truncation) makes things 0.5% WORSE — confirming the Phase 2b
   memory note that RNE was the right call.

4. **Phase 2b changes individually are essentially neutral or
   positive** relative to the current build. Yet the rne reference
   is 0.5% better at every layer.

## The rne reference is not a clean baseline

The `/workspace/atlas-dumps/numdrift/rne/` directory:
- Filesystem timestamp: `May 23 16:58 local`
- No `response.jsonl` (only the atlas_L*.bin + atlas_final_norm.bin)
- No docker image with matching tag currently exists (rne images now
  are `rnegated`/`bf16pv`/`poly`/`realfix2`/etc, all built May 24)
- The 2026-05-24 09:26 EDT atlas-gb10:fp8-dequant-rne image is 16+
  hours AFTER the dump's filesystem timestamp — so the dump can't be
  from that image either.

Likely provenance: a pre-Phase-2 baseline run on an earlier
atlas-gb10 image (now overwritten). Direct cosine comparison against
current realfix2 is biased by all the OTHER changes between that
era and now (scheduler refactors, weight loader changes, MoE kernel
tweaks, etc. — not just the three Phase 2b kernel changes I
bisected).

## What this rules out

For the "deep-layer drift on Qwen3.6-FP8" problem:

- **NOT** the FP8 KV cache dtype (Day 1, 8-way tie)
- **NOT** the FP8 KV cache calibration (Day 1, 3-way tie)
- **NOT** the `__expf` softmax replacement (Day 2, +0.0002 delta)
- **NOT** the FP16 P×V MMA (Day 2, +0.0004 delta)
- **NOT** the RNE dequant (Day 2, −0.0046 delta — RNE helps)

What remains in the ~4% Atlas-vs-HF compute gap:

- **Per-kernel intermediate precision** within each layer (attention
  output proj, MoE expert FFN, residual addition, RMSNorm) — not yet
  audited at per-step granularity
- **MoE expert-routing divergence** vs HF reference at deep layers
  (the "8/8 → 3/8 overlap" theory from `project_qwen36_drift_moe_smoking_gun`)
- **FP8 weight quantization inherent ceiling** (~0.989 per
  PHASE2A_VERDICT) — at best Atlas can reach this; ~2.2% of the gap
  is unfixable without different weights
- **Some non-kernel infrastructure change** between the rne era and
  now that I haven't identified (the unexplained 0.5%)

## Bisect infrastructure committed

Future bisects on prefill kernels:

- `ATLAS_EXTRA_NVCC_FLAGS` env var passed to nvcc invocations
  (`crates/atlas-kernels/build_target.rs`). Example:
  `ATLAS_EXTRA_NVCC_FLAGS="-DSOME_MACRO=1"` to flip an `#ifdef`-gated
  kernel path.
- `#ifdef ATLAS_FAST_SOFTMAX_EXP` already gated polynomial vs `__expf`
  (Phase 2b).
- `#ifdef ATLAS_DISABLE_FP16_PV` (new this session) gates FP16 vs
  BF16 P×V in both HDIM=256 and HDIM=512 paths of
  `prefill_paged_compute.cuh`.
- `ATLAS_DISABLE_RNE=1` (runtime, no rebuild) reverts the RNE dequant
  to truncation in `atlas-quant::fp8::f32_to_bf16` and the production
  mirror in `spark-model::weight_map::fp8_lut::f32_to_bf16`.

These gates make the next round of bisects (Day 3) a single
container bounce, no rebuild needed (for the runtime gate) or one
40s rebuild (for the compile gates).

## Day 3 plan

The remaining drift is in the model's deep-layer compute. Three
attack vectors, all of which need per-layer intermediate dumps to
localize:

1. **MoE expert-routing audit.** Existing `dump_expert_ids` captures
   Atlas's expert selections. Need a Python script that runs the same
   probe through HF and captures HF's expert selections. Compare
   overlap. If 3/8 overlap at L35-L39 as memory suggests, the fix is
   to increase MoE gate precision (currently BF16 input → f32 in
   shared mem).

2. **Per-sub-step cosine within each layer.** Extend `ATLAS_NEMO_DUMP`
   to capture: pre-attention, attn-output, pre-MoE, MoE-output,
   post-residual, post-RMSNorm. Compute cosine vs HF intermediate at
   each. Find the step where the largest single-layer delta appears.

3. **NVFP4 weight checkpoint comparison.** dgx2 has
   `RedHatAI/Qwen3.6-35B-A3B-NVFP4` cached. Different weight quant
   scheme. Run same probe. If cosine vs HF NVFP4 reference is
   significantly better than FP8 (e.g., 0.99+), the fix is to ship
   NVFP4. If similar (~0.96), Atlas's compute drifts regardless of
   weight quant.
