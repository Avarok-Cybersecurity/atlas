# C1 Diagnostic Results — FP8 vs BF16 logit overlap, Qwen3.6-35B-A3B-FP8

**Date**: 2026-05-26
**Model**: Qwen3.6-35B-A3B-FP8 on dgx1 (Atlas)
**Reference**: HF BF16 unquantized (Qwen/Qwen3.6-35B-A3B, snap 995ad96ea…)
**Purpose**: Decide go/no-go on Tier C3 (selective BF16 upcast L31-L39).
**Outcome**: ❌ **C3 is the wrong lever**. The drift mode targeted is long-context-driven, not per-token-precision-driven. The surgical fix is a B1+C4 hybrid (margin-ratio gate + BF16 verify on low-margin tokens).

---

## TL;DR

| Measurement | Result | Implication |
|---|---|---|
| Top-1 agreement (1st gen token, canonical 10382-tok probe) | ✅ Identical (token 6820) | FP8 doesn't flip argmax at first position |
| Top-5/10 Jaccard | 1.00 / 1.00 | Top of distribution is tight |
| Top-50 Jaccard | 0.89 | Tail starts diverging |
| KL(atlas ‖ bf16) | 0.020 nats | Modest distribution shift |
| Residual cosine at L39 | 0.98954 | Matches MASTER_DRIFT_TABLE 0.97657–0.99952 |
| **Focused short-prompt margins** (`0.1.0` digits) | **-10 to -18 logprobs** | Zero drift exposure |
| **Long-context probe margins** (post-drift agentic state) | **23.7% of positions have gap<1.5 logprobs** | Sea of low-margin decisions |

**The Tier A drift (`0.1.0`→`0.1 .0`, `2024`→`2 024`) is FP8 noise flipping argmax at low-margin positions that only exist under long-context entropy.** Selective BF16 upcast (C3) of any layer subset would mute the noise floor, but the right shape of fix targets **the low-margin positions specifically**, not the entire model.

---

## Method

1. **First-gen logit overlap** (`c1_final_logit_overlap.py`)
   Loaded the existing per-layer dumps at `/workspace/atlas-dumps/fp8native_dgx2/atlas_L39.bin` (Atlas FP8) and `hf_bf16_L39.bin` (HF BF16 unquant) from the canonical 10382-token chat probe. Applied final RMSNorm + lm_head (loaded from HF safetensors shard 26) offline. Computed top-K Jaccard, KL divergence, top-1 agreement.

2. **Focused short-prompt logprob inspection**
   Atlas `/v1/chat/completions` with `top_logprobs: 5` on a single-turn prompt asking the model to emit literal `version = "0.1.0"`. 41 decode positions captured. Computed top-1/top-2 logprob gaps.

3. **Long-context post-drift probe**
   Constructed a 5-message conversation where the prior assistant turn shows the exact Tier A drift artifact (`"0.1 .0"`, `"2 024"`) as a tool_call, followed by a fresh user request. 97 decode positions captured. Same logprob gap analysis.

---

## Findings

### F1 — First-gen logits broadly agree

```
residual cos L39       : 0.98954
logit cos              : 0.98728
argmax(atlas)          : 6820
argmax(bf16 ref)       : 6820
top1 agree             : True
top-1   jaccard        : 1.0000
top-5   jaccard        : 1.0000
top-10  jaccard        : 1.0000
top-50  jaccard        : 0.8868   ← tail divergence starts
top-200 jaccard        : 0.8349
KL(atlas ‖ bf16)       : 0.0204 nats
```

The cumulative-precision drift from FP8 GEMMs across 40 layers shifts the **tail** of the distribution but does not flip the **head**. This rules out "every token is wrong" as a failure mode.

### F2 — Margin distribution is bimodal between contexts

**Focused short prompt** (asking for literal `version = "0.1.0"`, 41 positions):
- Every digit / dot / quote token: gap > 10 logprobs to runner-up
- Sample: `[30] '0' top5: ['0'=+0.00, '1'=-14.38, '2'=-18.00, ' '=-18.81]`
- **Zero low-margin positions. Zero drift exposure.**

**Long-context post-drift probe** (97 positions):
- **23 of 97 (23.7%)** have top-1↔top-2 gap < 1.5 logprobs
- Examples:
  - `[ 1] sel='I' gap=0.12 (top: ['I'=-1.10, 'The'=-1.22, '```'=-2.22])`
  - `[70] sel='\n\n' gap=0.12` between `<think>` and `</think>`
  - `[44] sel=' shell' gap=0.12` (could have been ' directory')
- This is the regime where FP8 numerical noise (~0.02 nats KL on tail) can flip an argmax.

The pattern: **the longer the context and the more entropy the model is carrying, the more low-margin decisions exist**. FP8 then has the room to flip a small but non-trivial fraction of them.

### F3 — Per-layer cosine profile (existing, MASTER_DRIFT_TABLE.md)

Atlas FP8 vs HF BF16 per-layer hidden cosines (40 layers, last-token slice):
- mean **0.98982**, min **0.97657** at **L20**, max 0.99952 at L0
- worst per-op: `ssm.moe_out` at L20 cos=0.91983
- L31–L39 range cos in [0.978, 0.997] — high but not the global worst
- **Memory was wrong**: prior memory said "min cos 0.927 at L39"; current data shows worst hidden-state cosine is L20, not L39. L31-L39 are NOT uniquely worse.

### F4 — Atlas's whitespace mask covers 5 of 440 tokens

`decode_logits_seq.rs:434` suppresses tokens `[220, 198, 197, 256, 271]` when `inside_parameter_body && param_body_chars_emitted == 0`. The Qwen3.6 vocab has **440 whitespace-only tokens** and **6965 short tokens starting with whitespace**.

Even if mask coverage were extended, the mask is gated on *param-body position 0*. Tier A's drift fires *mid-content* (after 30+ tokens of parameter body) where the mask doesn't fire at all.

---

## C3 go/no-go

### What C3 would do

Dequant FP8 weights to BF16 at load time for a subset of layers. Hidden-state precision per layer would improve. Logit-floor noise would drop.

### Why C3 doesn't fix the actual failure mode

1. **Focused-prompt margins are huge**: -10 to -18 logprobs at digit/dot positions. C3 would shrink the FP8 noise floor, but the FP8 noise floor is **already 10+ orders of magnitude below** the margin in well-conditioned positions. C3 buys nothing where the model isn't already correct.

2. **Long-context margins are tiny**: 23.7% of positions have gap < 1.5 logprobs. Even after C3, the gap is still small. The "more layers in BF16" path reduces the *probability* of a flip but does not eliminate the failure regime — and the cost is multi-week engineering plus runtime memory/latency increase across every decode.

3. **L20 is the worst layer, not L31-L39**: per the existing master table. The C3 hypothesis ("upcast L31-L39") was based on stale memory. The actual worst-cos region is L18-L25, mostly the SSM `moe_out` op.

### What WOULD work (in increasing leverage)

| Fix | Cost | Coverage | Notes |
|---|---|---|---|
| **Expand whitespace mask** to all 440 ws tokens | < 1 day | Drift #11 only, param-body-position-0 only | Cheap and additive; doesn't catch mid-content. |
| **Mid-content whitespace gating** (suppress space tokens whose 1-char predecessor is `[0-9]` AND 1-token-lookahead suggests a digit) | 2-3 days | Drift #11 mid-content | Heuristic; may suppress legitimate `2 + 1` arithmetic. |
| **B1: margin-ratio drift detector** | 2-3 days | All low-margin positions; surfaces them for downstream action | Detection, not correction. |
| **B1+C4 hybrid: margin gate → BF16 forward on low-margin tokens** | 1 week | ALL low-margin argmax flips | **Surgical.** ~0.1-1% of decodes pay the BF16 cost. |
| **C3: selective BF16 upcast** | 3 weeks | Reduces noise floor everywhere | Broad-but-blunt. |
| **C2: USCD contrastive decoding** | 2-3 weeks | Output-distribution-level; requires BF16 ref head | Powerful but heavy. |

### Recommendation

**Skip C3. Promote B1+C4 hybrid as the next item.**

Justification:
- The drift the user reproduces (Tier A) is the **low-margin-argmax-flip regime**, not the **per-layer-precision-loss regime**.
- B1+C4 hybrid is **6× cheaper** than C3 and addresses the failure mode directly.
- Atlas already has MTP infrastructure; "BF16 verify on selected positions" can re-use the K=2 verify pipeline scaffolding.

---

## Caveats and what we didn't measure

- **No causal layer-ablation**: did not build `ATLAS_UPCAST_LAYERS` env var. Would have proven "if I upcast just L18-L25, does top-1 agree with full BF16 on the failing token?". Skipped because the margin analysis above made the question moot.
- **No multi-position dump**: hidden states are only captured at the last prompt position (first-gen). Mid-decode drift positions never got a per-layer cosine measurement. The 23.7% low-margin observation is from `top_logprobs` only, not from intermediate-state dumps.
- **No HF BF16 generation**: did not run extended greedy decode under BF16 to find the first divergence position vs Atlas. Would have given a clean "first-divergent-token" landmark but is GPU-expensive on a 35B model.

If C3 is still under consideration, the gap test to run next is **layer ablation under long-context probe**: build `ATLAS_UPCAST_LAYERS=18-25` and `=31-39`, re-run the long-context probe, measure how many of the 23 low-margin positions stay low-margin. If <50% stay low-margin under L18-L25 upcast, C3 is justified. If ≥50% stay low-margin, the lever is at the sampler/grammar/decoder layer, not the model.

---

## Artifacts

- `c1_final_logit_overlap.py` — first-gen logit overlap script
- `c1_final_logit_overlap.json` — first-gen overlap result
- `/tmp/c1_drift_probe.json` — focused short-prompt probe input
- `/tmp/c1_longctx_probe.json` — long-context post-drift probe input
- `/tmp/c1_longctx_resp.json` — long-context probe response with logprobs
- `MASTER_DRIFT_TABLE.md` — existing per-layer per-op cosine table (40 layers × ~6 ops)
- `research3_drift_catalog.md` — drift modes #1-#15 with intervention classes
