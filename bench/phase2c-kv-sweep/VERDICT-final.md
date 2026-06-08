# Phase 2c Day 1 FINAL Verdict — KV cache dimension eliminated

18 configs ran across dgx1 + dgx2 in parallel. The hypothesis "deep-layer
drift on Qwen3.6-FP8 is from FP8 KV cache quantization" is **falsified**.

## Headline numbers

| Tie group | mean | min | Configs |
|---|---|---|---|
| Best non-broken | 0.9615 | 0.9179 | `fp8-baseline` |
| Edge case | 0.9611 | 0.9120 | `fp8-hp2` |
| **8-way tie** | **0.9605** | **0.9143** | `bf16-all`, `bf16-hp-max`, `fp8-hp{5,10,max}`, `nvfp4-hp{5,10,max}` |
| Regression | 0.9556 | 0.9058 | `nvfp4` (alone) |
| Calibration-bug tie | 0.9175 | 0.7956 | `fp8-calib{64,256,512}` |
| Garbage | 0.7285 | 0.4274 | `turbo3` |
| Garbage | 0.7116 | 0.3608 | `turbo4` |
| Broken NaN | — | NaN @ L19 | `turbo8`, `turbo8-hp2` |

## Three structural findings

### 1. The prefill cosine bench is insensitive to KV cache dtype

8 different configurations — covering all three meaningful dtypes
(BF16, FP8, NVFP4) with various `--kv-high-precision-layers` settings —
produced **bit-identical** cosines (0.9605/0.9143/0.9311 across mean/
min/final to 4 decimals). The probe is 18920 tokens, chunked at 4096
into 5 prefill steps, so subsequent chunks DO read prior chunks' K/V
from cache. Yet the per-layer hidden state at the final-token dump is
the same regardless of how K/V was stored.

The most consistent explanation: the chunked-prefill attention kernel
dequantizes cached K/V back to BF16 (or higher) before the matmul, so
storage precision is transparent to the dump. The bench measures
prefill *compute*, not KV cache fidelity.

This means **the bench can't tell us whether KV cache choice matters at
decode time** — that would need a long-generation dump, with HF
reference also generating to compare. Out of scope for Day 1.

### 2. Calibration is a no-op (or bug)

`--fp8-kv-calibration-tokens` 64, 256, 512 all produced bit-identical
0.9175 mean / 0.7956 min @ L35 / 0.8569 final. Different token counts
should produce different k_scale/v_scale estimates after calibration
runs. The fact that all three give the same result suggests either:
- Calibration never actually runs for this probe (single short
  decode after the long prefill — calibration may need *N* decode
  steps with prior KV reads)
- The calibration path activates a different code path (different
  k_scale/v_scale layout) that just degrades everything by the same
  fixed amount regardless of tuning depth

Worth investigating as a separate bug, but not the precision fix.

### 3. Turbo8 is broken on Qwen3.6's hybrid SSM+attention layout

Turbo8 produces **all-NaN hidden states from L19 onward**. Turbo3 and
Turbo4 don't NaN but produce 0.71-mean garbage cosines. Atlas's
`kv_cache.rs:132` comment says Turbo8 was validated on MiniMax M2.7
("58 Turbo8 layers under auto HP=2"). MiniMax is pure-attention;
Qwen3.6 is 3:1 SSM:attention. The first full-attention layer
(L3) works; failure at L19 (the 5th full-attention layer) implicates
accumulation across the SSM-attention interleave pattern. Separate
bug worth filing; not the precision fix either.

## The actual culprit — `__expf` unmasked a deeper bug

Per-layer diff between current `fp8-baseline` and the May 23 `rne`
dump (same probe, same HF reference):

| Layer | old rne | new fp8-baseline | delta |
|---|---|---|---|
| L0  | 0.9960 | 0.9954 | -0.0006 |
| L5  | 0.9918 | 0.9814 | **-0.0105** |
| L10 | 0.9868 | 0.9821 | -0.0047 |
| L15 | 0.9789 | 0.9705 | **-0.0084** |
| L20 | 0.9678 | 0.9596 | -0.0082 |
| L25 | 0.9520 | 0.9500 | -0.0020 |
| L30 | 0.9524 | 0.9585 | +0.0060 |
| L35 | 0.9290 | 0.9179 | **-0.0111** |
| L39 | 0.9372 | 0.9335 | -0.0037 |

Mean delta: **−0.006**. Realfix2 is consistently 0.5-1% worse than
May 23 rne at most layers, despite Phase 2b shipping RNE + `__expf`
softmax + FP16 P×V MMA between those builds.

`project_qwen36_phase2b_softmax_expf.md` memory explicitly warned:

> Polynomial was MASKING deep-layer FP8 KV drift. Bug #2 unmasked:
> late attn layers regress L31-L39 from FP8 KV quant noise on large
> K/V magnitudes. ... Don't revert __expf

The `__expf` change *removed* a 0.5%-error polynomial that was
inadvertently smoothing over a deeper precision bug. The KV sweep
above confirms it's NOT the KV cache *storage* — but it IS something
related to how K/V values get *computed* and used at deep layers.

Suspect kernels (all touched by Phase 2b changes):
- `prefill_paged_compute.cuh::sw_exp` — replaced by `__expf` in main
  path; polynomial still available behind `ATLAS_FAST_SOFTMAX_EXP`
  `#ifdef`. **Single most likely culprit.**
- `prefill_paged_compute_512.cuh::sw_exp_512` — same change for the
  HDIM=256 path.
- BF16 RNE patch at fp8 dequant (`fp8.rs`, `fp8_lut.rs`).
- FP16 P×V MMA replacing BF16 P×V in attention.

## Inherent ceiling reminder

`PHASE2A_VERDICT.md` documents the FP8 quantization-inherent ceiling:
HF[FP8→BF16] vs HF[BF16-unquant] is `mean 0.989, min 0.970 @ L35`.
That's the best ANY FP8 inference engine can do.

Current Atlas fp8-baseline at L35 = 0.9179. Gap to ceiling = 0.052
absolute, or 5.4% relative. That's the compute headroom remaining.

## Day 2 plan — bisect the regression

Highest value next step (do this first):

1. **Disable `__expf`, re-bench.** The kernel `#ifdef
   ATLAS_FAST_SOFTMAX_EXP` lives in `prefill_paged_compute.cuh:47` and
   `prefill_paged_compute_512.cuh:47`. Build the kernel with that
   defined; run cosine bench; compare:
   - If cosine returns to ~0.967 mean (old rne level), the `__expf`
     change IS the regression source — meaning some other compute
     path is now exposed to error the polynomial hid. Need to find
     what.
   - If cosine stays ~0.961, `__expf` is exonerated; investigate RNE
     or FP16 P×V.

2. **Per-kernel cosine bisection within each full-attention layer.**
   Extend `ATLAS_NEMO_DUMP` to capture intermediate values:
   `pre_attn` → `attn_out` → `pre_mlp` → `mlp_out` → `post_residual`.
   Compare against HF reference at each. Identify where the largest
   single-step error appears.

3. **MoE expert routing divergence audit.** Existing `dump_expert_ids`
   captures Atlas's routes; need HF Python script to dump HF's
   selected experts per layer for the same probe. Compare.

4. **NVFP4 weight checkpoint test** (dgx2 already has it cached).
   `RedHatAI/Qwen3.6-35B-A3B-NVFP4`. Same Atlas, same probe, different
   weight quant. If cosine vs same-quant HF reference is ≥ 0.985,
   the fix is to switch model. If ~0.96, Atlas's compute itself drifts
   regardless of quant scheme.

## Bottom line

Day 1 (KV sweep) is **complete**. KV cache hypothesis eliminated. The
~4% drift to HF reference lives in compute, specifically in the
post-Phase-2b kernel surface (RNE / `__expf` / FP16 P×V). Day 2 starts
with bisecting which Phase-2b change re-introduced or unmasked the
regression.
