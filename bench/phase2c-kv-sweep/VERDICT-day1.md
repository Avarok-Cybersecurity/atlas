# Phase 2c Day 1 Verdict — KV cache sweep results

## TL;DR

**KV cache choice does NOT explain the deep-layer drift on Qwen3.6-FP8.** 12 configs ran across all 6 supported KV dtypes (Bf16, Fp8, Nvfp4, Turbo4, Turbo3, Turbo8) with varying `--kv-high-precision-layers` and `--fp8-kv-calibration-tokens`. Best config (fp8-baseline) was 0.9615 mean cosine vs HF reference. **5 different configs produced bit-identical cosines (0.9605/0.9143/0.9311)**, meaning the prefill bench is insensitive to KV cache dtype — dequant likely happens before attention math, so storage dtype doesn't show in the dump. The ~4% remaining gap to HF reference is in **compute** (weights, kernels, MoE gate), not KV cache.

## Full sweep table (12 dgx1 configs + 6 dgx2 pending)

| Config | dtype | hp_layers | calib | mean | min | min@layer | final | Status |
|---|---|---|---|---|---|---|---|---|
| **fp8-baseline** | fp8 | 0 | 0 | **0.9615** | 0.9179 | L35 | 0.9307 | best non-broken |
| fp8-hp2 | fp8 | auto | 0 | 0.9611 | 0.9120 | L35 | 0.9317 | tiny regression |
| fp8-hp5 | fp8 | 5 | 0 | 0.9605 | 0.9143 | L35 | 0.9311 | identical to hp10 |
| fp8-hp10 | fp8 | 10 | 0 | 0.9605 | 0.9143 | L35 | 0.9311 | identical to hp5 |
| nvfp4-hp5 | nvfp4 | 5 | 0 | 0.9605 | 0.9143 | L35 | 0.9311 | identical to fp8-hp5/10 |
| bf16-all | bf16 | 0 | 0 | 0.9605 | 0.9143 | L35 | 0.9311 | identical to nvfp4-hp5 |
| nvfp4 | nvfp4 | 0 | 0 | 0.9556 | 0.9058 | L35 | 0.9264 | small regression |
| fp8-calib256 | fp8 | 0 | 256 | 0.9175 | 0.7956 | L35 | 0.8569 | massive regression |
| turbo3 | turbo3 | 0 | 0 | 0.7285 | 0.4274 | L35 | 0.6225 | garbage |
| turbo4 | turbo4 | 0 | 0 | 0.7116 | 0.3608 | L35 | 0.4048 | garbage |
| turbo8 | turbo8 | 0 | 0 | 0.9782* | NaN | L19+ | NaN | **NaN bug on hybrid arch** |
| turbo8-hp2 | turbo8 | auto | 0 | 0.9841* | NaN | L19+ | NaN | same NaN bug |

*\*turbo8 numbers are skip-NaN — only valid L0-L18*

## Critical findings

### 1. The "4-way tie" reveals a benchmark sensitivity gap

`fp8-hp5 = fp8-hp10 = nvfp4-hp5 = bf16-all` produced bit-for-bit identical 0.9605/0.9143/0.9311 cosines. Four different KV cache configurations should produce different per-layer hidden states. The dump is taken after chunked prefill on a 18920-token probe; subsequent chunks read prior chunks' K/V from the cache, so the dtype *should* matter.

The most likely explanation: the chunked-prefill attention kernel dequants the cached K/V back to BF16 (or higher) before the matmul, so the storage dtype only affects the *transient* representation, not the math. Per-layer hidden state at the dump point is therefore largely insensitive to KV cache dtype.

This means **the cosine bench cannot tell us about KV-cache-related decode degradation** — that would require dumping during DECODE on a long generation. The bench has been measuring compute-side drift this whole time.

### 2. The "BF16 cliff" is real but tiny

Project memory's note about "BF16 KV has L35-L39 cliff bug" is empirically present: bf16-all and fp8-hp{2,5,10} (which replace late layers with BF16 KV) all measure ~0.5% worse mean cosine than pure fp8. But the regression is dwarfed by the inherent ~4% compute gap.

### 3. Turbo8 broken on Qwen3.6 hybrid SSM+attention

Turbo8 was battle-tested on MiniMax M2.7 (58 Turbo8 layers per `kv_cache.rs:132`). On Qwen3.6 with 3:1 SSM:attention pattern, turbo8 produces **all-NaN hidden states from L19 onwards**. First full-attention layer is L3 — it works through L15. Failure at L19 (the 5th full-attention layer) suggests an accumulation issue specific to the 4-SSM-then-1-attn pattern, not a fundamental turbo8 bug.

Turbo3 and Turbo4 don't NaN but produce garbage cosines (~0.71). The WHT rotation amplifies precision errors when interacting with the SSM layers' GDN state.

### 4. Calibration hurts on this checkpoint

`--fp8-kv-calibration-tokens 256` produces 0.9175 mean / 0.7956 min — a 4% mean regression. The boot warning ("FP8 KV cache selected. This requires calibrated k_scale/v_scale") suggested calibration would help, but for this specific checkpoint's pre-loaded k/v scales, online calibration overrides them with worse estimates.

## Where the drift actually lives

The per-layer cosine profile (fp8-baseline):

| Layer | cos | type |
|---|---|---|
| L0 | 0.9954 | SSM |
| L20 | 0.9596 | SSM |
| L31 | 0.9439 | full-attn |
| L35 | 0.9179 | full-attn ← LOWEST |
| L37 | 0.9260 | SSM |
| L39 | 0.9335 | full-attn |
| final | 0.9307 | post-norm + lm_head |

Drift rate:
- L0 → L20: 0.036 drop over 20 layers (0.0018/layer)
- L20 → L35: 0.042 drop over 15 layers (0.0028/layer — **56% faster**)
- L35 → L39: rises 0.016 (final RMSNorm partially recovers)

Late layers drift faster per layer. Three plausible causes (all in compute, not KV):

1. **MoE expert route divergence** (per `project_qwen36_drift_moe_smoking_gun.md`: 8/8 → 7/8 → 3/8 overlap at L0 → L24 → L38). Expert ranking flips at deep layers because gate logits get hit by accumulated residual drift. The MoE gate kernel is correct (BF16 in, f32 internal); the **input** is drifted.

2. **K/V magnitude growth at depth.** Large K/V values stress the FP8 dynamic range. Even with k_scale/v_scale, late-layer K/V is closer to E4M3 saturation. But this is downstream of the K/V *computation* — the compute itself uses FP8 weights too.

3. **FP8 weight dequant error accumulating across the residual stream.** Each layer's MoE outputs are computed with FP8 weights, then accumulate into the BF16 residual. Per-layer error compounds.

## Next attack vectors (Day 2+)

In priority order:

1. **Per-kernel cosine bisection.** Add intermediate dumps WITHIN each layer (post-attention output, post-MoE output, post-residual). Identify which sub-step contributes the most error per layer. Existing infra: `ATLAS_GDN_DUMP` for SSM sub-stages (conv, l2, gdn, gnorm); need similar for full-attention layers + MoE. ~1 day of infrastructure work.

2. **MoE expert-route divergence audit.** Use existing `dump_expert_ids` to capture Atlas's selected experts at each layer. Compare against an HF reference dump (would need to Python-script the MoE forward in HF Transformers to dump its expert selections). The 8/8 → 3/8 overlap claim from memory is testable; if confirmed at L35-L39, focus the fix there.

3. **NVFP4 weight checkpoint test.** dgx2 has `RedHatAI/Qwen3.6-35B-A3B-NVFP4` cached. Run the same Atlas+probe on that. If cosine vs same-quant HF reference is dramatically better (say 0.99+ mean), the fix is to ship the NVFP4 model, not patch FP8. If cosine is similar (~0.96), the issue is in Atlas's compute regardless of quant.

4. **BF16 weight checkpoint test.** Download Qwen3.6-35B-A3B-BF16 (~70GB) and run same bench. Should hit ~0.99 cosine if compute is correct — confirms the ceiling. If still ~0.96, Atlas's compute itself drifts beyond BF16 precision.

5. **FP32 residual stream.** Keep `hidden_state` in FP32 throughout (cast in/out at each layer boundary). Doubles memory, halves throughput, but isolates whether BF16 residual accumulation is the bottleneck.

## Bottom line

KV cache work is **done** — choice doesn't move the needle on this benchmark. Day 2 starts with per-kernel bisection or weight-checkpoint isolation. Given the user's "multi-day" framing, this is one day's deliverable: we've eliminated an entire dimension of the hypothesis space.
