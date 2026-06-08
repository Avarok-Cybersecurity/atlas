# C1 5-Agent Parallel Research Synthesis (2026-05-26)

After C1 diagnostic pointed at B1+C4 hybrid + whitespace mask, 5 parallel
agents (2 online, 3 local) ran in background while WS1/WS2/B1/C4v1 were
implemented. This synthesizes their findings.

## Headline findings cross-cut

| Agent | Topic | Strongest signal |
|---|---|---|
| 1 (online) | FP8 long-ctx drift SOTA | **SageAttention2/vLLM two-level FP8 accumulation: 91%→13% NIAH collapse at 128K from Hopper FP22 accumulator, fixed to 89% with two-level accum. GB10 sm_121 has the same FP22 accumulator.** Possibly the literal bug. |
| 2 (online) | Prod margin-gate patterns | **TRT-LLM `mtp_relaxed_acceptance_op` (mtp.py:823-828)** uses `log p[top1] − log p[draft] ≤ relaxed_delta` over top-K — DeepSeek-R1 default 0.6 logprobs, topk=10. Exact mirror of our B1, just inverted. |
| 3 (local) | Atlas MTP hook map | B1+C4 hook at `verify_k2_step.rs:119+` (post-pipeline, pre-decision). Extending `accepted = (drafts[0] == v0) && (margin > T)`. Top-2 from already-on-host verify logits, no extra D2H. |
| 4 (local) | Qwen3.6 vocab scan | 425 pure-whitespace tokens (G1), 1161 short-leading-ws-with-non-alpha (G2). 0 ` 0..9` atomic tokens — Qwen3.6 always splits to `[220, digit]`, so the digit-collapse drift fires entirely on token 220. Existing OnceLock pattern in helpers.rs is the right home. |
| 5 (local) | decode_logits_seq audit | Top-1/top-2 from `f32_logits` is CHEAP — single O(V) scan. forced-token-fastpath BYPASSES logit_bias (must gate either before fastpath or via post-emit). MTP K=2 calls `run_pipeline` so a margin gate as a 9th pipeline stage auto-covers MTP. |

## Three biggest insights for the active work

### 1. SageAttention2 / FP22 accumulator (agent 1)

The single most surprising finding. Hopper/Blackwell FP8 Tensor Cores
accumulate matmul partial sums in **FP22** (1 sign + 8 exp + 13 mantissa,
not the full FP32 we'd expect). At long context (16K+), accumulated
softmax(QK)·V over many KV tokens overflows the FP22 mantissa, dropping
NIAH benchmark from 91% (BF16) to 13% (FP8).

**vLLM two-level fix**: split the accumulation into a per-tile FP32
intermediate before the FP8 MMA reduction — recovers to 89% NIAH at 128K.
Released as `--fp8-kv-cache` patch in vLLM blog 2026-04-22.

**GB10 implication**: sm_121 has the same FP22 accumulator. Atlas's
`atlas-kernels` FP8 attention kernels MAY have this issue. Worth a quick
audit of the attention reduce step. If we ship two-level accum, we may
not need any of the margin-gate work — drift could collapse to baseline.

**Memory cross-ref**: `project_qwen36_phase2b_softmax_expf.md` already
noted "late attn layers regress L31-L39 from FP8 KV quant noise on large
K/V magnitudes" — this is EXACTLY the FP22 accumulator overflow regime.
The softmax `__expf` patch (2026-05-24) didn't fix it; two-level accum
might.

Next action: read `crates/atlas-kernels/src/cuda/attn_fp8.cu` or equiv,
check the accumulator dtype in the MMA reduce, compare to vLLM patch.

### 2. TRT-LLM relaxed acceptance is the canonical margin gate (agent 2)

Production code:
```python
# tensorrt_llm/_torch/speculative/mtp.py:823-828 (paraphrased)
top_k_logprobs = torch.topk(target_logprobs, k=relaxed_topk).values
margin = top_k_logprobs[0] - draft_logprobs
accepted = margin <= relaxed_delta  # i.e., target NOT much more confident
```

DeepSeek-R1 default: `relaxed_topk=10, relaxed_delta=0.6` logprobs.

Atlas's B1 uses `gap < 1.5 logprobs` — wider than DeepSeek's 0.6. We
should consider tightening to 0.6 for fewer false positives, or staying
loose for more aggressive detection. Empirical tuning post-deploy.

At T=1.0, gap=1.5 ↔ 4.5× prob ratio (~80% top-1 / 18% top-2).
At T=1.0, gap=0.6 ↔ 1.8× prob ratio (~64% / 35%).

C4v1's lift schedule treats gap=0 as "50/50" via `(threshold-gap)*0.5`,
which means at 1.5 threshold + gap=0, we lift top-2 by 0.75 logprobs —
roughly equivalent to converting "80%/18%" into "60%/40%". Soft enough
to not introduce noise on healthy positions; firm enough to escape the
FP8-flip attractor.

### 3. The vocab scan was bigger than expected (agent 4)

Atlas's hardcoded mask covers 5 of **425 whitespace-only tokens**. WS1
fixes that with the OnceLock vocab scan. WS2 (mid-content gate) leverages
the same scan via a new `is_digit_ending` predicate.

Agent 4 also identified the existing `OnceLock<Arc<[bool]>>` pattern in
`helpers.rs:323-385` (NUMERIC_TOKEN_MASK, BOUNDARY_TOKEN_MASK,
MID_WORD_TOKEN_MASK). My `whitespace_mask` module uses the equivalent
`OnceLock<HashSet<u32>>` pattern. Future cleanup: merge into the
helpers.rs convention for consistency.

## What was implemented (this session)

| Item | Where | Cost |
|---|---|---|
| WS1: boot-time whitespace vocab scan | `whitespace_mask.rs` new + `serve.rs:280` + `decode_logits_seq.rs:464` + `emit_step.rs:170` | ~30 LoC |
| WS2: mid-content gate (digit-ending → suppress ws) | `whitespace_mask.rs::is_digit_ending` + `decode_logits_seq.rs:467` | ~25 LoC |
| B1: margin-ratio drift detector | `decode_logits_seq.rs:489` (top-2 scan) + periodic summary helper | ~50 LoC |
| C4v1: low-margin top-2 lift | `decode_logits_seq.rs:545` (lift schedule) | ~10 LoC |
| Total | | ~115 LoC |

## What was NOT done (future work)

| Item | Why | Effort estimate |
|---|---|---|
| MTP K=2 margin gate at `verify_k2_step.rs:119+` | Non-MTP path covers most decode positions today; MTP coverage is follow-up | ~30 LoC, ~1 day |
| Two-level FP8 attention accumulator (SageAttention2) | Possibly the literal long-context root cause; requires kernel-level CUDA work | 1-2 weeks if it works, 0 if vLLM's patch ports cleanly |
| Refactor `whitespace_mask` to match `helpers.rs` OnceLock<Arc<[bool]>> pattern | Cosmetic; current design works model-agnostic | ~20 LoC |
| Real BF16 reverify on flagged positions (full C4) | Genuinely novel; ~1 week to implement once we have BF16 head head-only | ~200 LoC, multi-week |
| Tighten margin threshold from 1.5 → DeepSeek's 0.6 | Empirical — needs measurement post-deploy | trivial |

## Recommended next sequence

1. **Deploy + measure (this session)**: build atlas-gb10:tier-wsc4, opencode probe, compare files-written + B1 firing rate.
2. **If positive**: investigate FP22 accumulator in Atlas attention kernel — could be the bigger fix.
3. **If neutral or negative**: tune B1 threshold (0.6 vs 1.5), check that the C4v1 lift schedule isn't over-correcting in long prose contexts.
4. **Independent track**: read SageAttention2 / vLLM patch for two-level accum; estimate Atlas kernel diff.

## Artifacts

- `research_C1agent1_fp8_drift_sota.md` (113L)
- `research_C1agent2_prod_margin_gates.md` (337L)
- `research_C1agent3_mtp_hook_map.md` (272L)
- `research_C1agent4_ws_mask_scan.md` (222L)
- `research_C1agent5_decode_audit.md` (188L)
- `qwen36_whitespace_tokens.json` (vocab dump)
- `research_C1_results.md` (the C1 writeup that triggered this work)
