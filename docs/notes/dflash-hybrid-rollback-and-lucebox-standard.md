# DFlash on hybrid recurrent targets: rollback semantics, LuceBox comparison, and Atlas evidence

Date: 2026-05-25

This note records the implementation lessons from the Spark-1 Atlas DFlash investigation. It is intentionally written as engineering notes rather than a product claim: the goal is to preserve what we learned about rollback-based speculative decoding on Qwen3.5/Qwen3.6-style hybrid targets, where attention KV cache and recurrent SSM/GDN state have different rollback properties.

## Executive summary

DFlash-style speculative decoding is simple to reason about on a pure-attention transformer: the draft model proposes a block, the target verifies it, rejected suffix tokens are discarded, and the target KV cache can be cropped back to the committed prefix.

Hybrid Qwen-family targets are different. Qwen3.5/Qwen3.6-style models contain both full-attention layers and stateful linear-attention / GDN / SSM layers. The full-attention KV cache can be cropped, but the recurrent state is not just a sequence-indexed KV array. If verification advances recurrent state and a later rejection tries to "crop" back, the recurrent state may already have been mutated past the rollback point. A crop-only implementation can therefore emit tokens from a target state that is not equivalent to autoregressive decoding from the committed prefix.

The practical standard we adopted for Atlas is therefore not "stricter than LuceBox token-id exactness." The target is LuceBox-equivalent semantics:

- no drafter-only token leaks after target rejection;
- emitted tokens are target-approved from the committed target state, or produced by an empirically equivalent safe strategy;
- no z-lab-style pathological loops or semantic collapse;
- comparable speedups on the retained benchmark class.

In short: match the effective correctness class that LuceBox demonstrates, do not use z-lab crop-only behavior as the quality bar, and keep speed and correctness claims separate.

## Terminology

- **AR**: ordinary autoregressive target decoding, one token at a time.
- **DFlash / speculative decoding**: use a drafter to propose multiple tokens, then use the target to verify and commit an accepted prefix plus, typically, one target bonus token.
- **Accepted draft token**: a proposed token whose ID matches the target verifier output at that position.
- **Target bonus token**: the next token sampled/selected from the target verifier state immediately after the accepted prefix.
- **Committed prefix**: the output tokens that have been accepted as the authoritative sequence.
- **Committed-state approval invariant**: every emitted token must either be an accepted draft token verified by the target, or a target bonus token decoded from the target state corresponding to the committed prefix.
- **Crop-only rollback**: after rejection, only crop the target attention KV cache to the accepted length and continue. This is unsafe when recurrent state was advanced and cannot be rewound by KV cropping.
- **Safe rollback / replay family**: snapshot/restore target state, replay from the committed prefix, keep forked/disposable verifier state, or use tree-aware parent-lineage recurrent state so rejection never corrupts the canonical committed state.

## Why hybrid recurrent state changes the algorithm

The classic speculative-decoding invariant assumes the target state can be restored to the accepted prefix after any rejection. In pure attention this is usually represented as:

1. prefill prompt into target KV cache;
2. run target over a proposed draft block;
3. compare draft IDs to target verifier IDs;
4. if only the first `k` draft tokens are accepted, crop the target cache to prefix length `prefix + k`;
5. append the target bonus token and repeat.

For hybrid recurrent targets, step 4 is not sufficient. The recurrent layers have advanced internal state while verifying tokens that may later be rejected. If that state is stored as a mutable rolling state rather than as a restorable per-token KV slab, a `crop()` operation on attention cache does not rewind it.

This was observed directly in the z-lab reference probe on the Qwen3.5 hybrid target:

- target architecture: 48 `linear_attention` layers plus 16 `full_attention` layers;
- z-lab DFlash exactness against AR-like block size 1: `2/6` prompts exact;
- the probe caveat explicitly notes that cache crop is a no-op for linear-attention recurrent state, so block speculative mode and AR-like mode may diverge even at temperature 0 when rejections occur.

The failure mode is not merely cosmetic. Examples from the upstream reference included degenerate outputs such as:

- `Hummingbirds are tiny wings wings.` instead of a normal sentence;
- repeated queue phrases in a stack-vs-queue explanation;
- repeated `2.` lines and leaked thinking markers in a weather-adjective list.

That is the unsafe crop-only class. Atlas should not treat those outputs as an acceptable correctness standard just because they came from a public reference implementation.

## LuceBox strategy class

The relevant LuceBox idea is not "bit-identical token IDs in every ad-hoc probe." The useful target class is the implementation strategy:

- snapshot/restore of target state;
- replay from the committed prefix;
- tree-aware state management for recurrent/GDN/conv state;
- parent-lineage tracking so speculative branches do not overwrite canonical committed state.

The source-level strategy we audited falls into this safe family with machinery such as:

- `snapshot_kv()` / `restore_kv()`;
- `snapshot_ssm_state()` / `restore_ssm_state()`;
- tree-aware SSM/GDN operators such as `ggml_ssm_conv_tree`, `ggml_gated_delta_net_tree`, and persistent tree variants.

The empirical LuceBox artifacts available during this investigation did not prove universal strict token-id exactness. In particular:

- legacy chain probing at `NGEN=8` had a short-prefix pass;
- chain `NGEN=64` failed to emit comparable tokens under that harness configuration;
- DDTree `NGEN=64` completed but diverged from its own AR reference at generated token 45 on the count-to-20 prompt.

That means the operational standard should not be "Atlas must be more bit-exact than LuceBox." The standard should be: Atlas must be in the safe LuceBox strategy family and empirically no worse on matching/pathology, while preserving the required speedup.

## Atlas current strategy class

Atlas is not in the z-lab crop-only family. The Spark-1 Atlas path has explicit recurrent-state checkpoint/commit machinery for the Qwen3.5 hybrid target:

- pre-verify copy of canonical checkpoint state into scratch verifier state;
- commit of the full-accept scratch state or partial-accept intermediate state back into canonical state;
- separate generic and chunked verifier paths;
- trace hooks to compare live verification against sequential committed-state replay.

The important invariant is not whether a final string matches an AR string byte-for-byte in every receipt. The important invariant is whether emitted DFlash tokens were target-approved from the committed target state. We used the following proof shape:

1. capture the committed target state before verification;
2. verify a draft block in the live DFlash path;
3. for each emitted token, classify it as either:
   - an accepted draft token where `draft[j] == target_verified[j]`, or
   - a target bonus token from the verifier state after the accepted prefix;
4. independently replay or probe from the committed snapshot where feasible;
5. assert zero drafter-only leaks.

A Spark-1 committed-state probe over generic and chunked verifier modes found:

- 225 verifier rounds inspected;
- 479 emitted tokens classified;
- 255 target-approved draft tokens;
- 224 target bonus tokens;
- 0 drafter-only leaks.

The direct generic-vs-chunked count-to-20 probe produced identical final text and completion-token counts, and the chunked verifier path did not need to be demoted by the correctness evidence.

## Same-prompt Atlas / z-lab / LuceBox matrix

A same-prompt matrix over the six official probe prompts separated three questions that had been conflated:

1. Is the output strictly token-id exact to AR?
2. Is it non-pathological and prompt-following?
3. Is each emitted token target-approved from the committed target state?

Headline findings for the Spark-1 matrix:

- Atlas DFlash target-approved commit check: `5/6 pass`, `1/6 unknown`, `0/6 fail`.
- Atlas DFlash text exactly equal to Atlas AR on the matched NVFP4 stack: `4/6`.
- Atlas DFlash token IDs exactly equal to Atlas AR: `4/6`.
- z-lab DFlash text exactly equal to z-lab AR: `2/6`.
- Atlas pathological rows: `0/6`.
- z-lab pathological rows: `3/6`.
- Atlas no worse than z-lab per row: `6/6`.

Per-prompt summary:

- `alpha beta gamma delta`: Atlas DFlash stopped cleanly at EOS while Atlas AR overran EOS in that stack. This is not a DFlash regression.
- `Count from 1 to 20`: Atlas DFlash and Atlas AR were token-id identical. z-lab was exact. LuceBox DDTree later diverged on tail tokens outside the meaningful count string.
- `Hummingbirds`: Atlas DFlash and Atlas AR were identical and coherent. z-lab collapsed to `tiny wings wings`.
- `Stack vs queue`: Atlas DFlash and Atlas AR were identical and coherent. z-lab looped phrases.
- `Spanish translation`: Atlas DFlash and Atlas AR were identical. z-lab truncated.
- `Weather adjectives`: Atlas DFlash and AR diverged at token 45/64 on lexical detail, but both outputs were non-pathological. This row was kept as `unknown`, not overclaimed.

This is why the Atlas standard became: target-approved, LuceBox-equivalent, non-pathological semantics with speed, not universal AR token-id exactness.

## Speed accounting: do not confuse prompt peaks with full-run throughput

The retained Spark-1 full72 contract used:

- Qwen3.5 27B NVFP4 target;
- qwen35 DFlash drafter;
- 72 prompts;
- max generated tokens 160;
- temperature 0.0;
- concurrency 1;
- full vocab;
- NVFP4 KV;
- FP32 recurrent decode path enabled where required;
- OpenAI-compatible usage completion-token accounting.

The best retained legal line at the time of this note was approximately:

- matched AR: `13.1447` output tok/s;
- Atlas DFlash: `23.5710` output tok/s;
- speedup vs matched AR: about `1.79x`;
- full72 coverage: `72/72`;
- still below the aspirational `30.0` output tok/s Big D gate.

A later suspicious `29-30 tok/s` claim was traced to per-prompt/category speed rather than aggregate full72 throughput. The artifact covered 72 prompts but was stitched from four 18-prompt batches with server restarts. Its real aggregate per-request throughput was about `23.74` tok/s and wall aggregate about `17.70` tok/s; only one prompt exceeded 30 tok/s. That is useful evidence, but it is not a 30 tok/s full72 aggregate pass.

The recurring reporting rule is:

- report aggregate output TPS;
- separately report wall aggregate TPS for stitched/restarted runs;
- separately report mean/median/max per-prompt TPS;
- never promote a category peak as a full-run pass.

## Practical acceptance gates for future work

A DFlash optimization for hybrid targets should not be accepted on speed alone. A useful gate has at least three layers.

### 1. Coverage and throughput

- same prompt manifest and generation cap as the baseline;
- same model family / target checkpoint unless the goal explicitly pivots;
- explicit AR and DFlash artifacts;
- fresh full-run aggregate throughput;
- clear stitched-vs-uninterrupted accounting.

### 2. Quality and LuceBox-equivalent matching

- no degenerate loops or prompt collapse on the shared probe set;
- no worse than LuceBox/z-lab on common prompts under the agreed standard;
- rows that diverge from AR are classified rather than hidden.

### 3. Committed-state approval

For deterministic probe scope, every emitted token should be explainable as one of:

- accepted target-verified draft token;
- target bonus token from committed verifier state;
- explicit unknown requiring more instrumentation.

Any drafter-only token after rejection is a hard failure. Any implementation that relies on crop-only rollback for recurrent state is not in the safe strategy family.

## Implementation notes for Atlas maintainers

When editing Atlas DFlash code, preserve these invariants:

1. Treat recurrent state as first-class canonical state, not as a side effect of the attention KV cache.
2. Verify from disposable/scratch state or a snapshot that can be discarded safely.
3. Commit only the accepted target-approved path back to canonical state.
4. Make verifier mode explicit in summaries: generic vs chunked, gamma, quantization, force envs, and trace flags.
5. Keep debug hooks that can classify accepted draft tokens vs target bonus tokens.
6. Fail closed in validation mode if hybrid target verification skips the pre-verify copy or commit path.
7. Keep sampler-history and argmax tie behavior separate from committed-state bugs. A numerical tie or sampler-history mismatch is not automatically a drafter leak.

Likely speed levers identified during the Spark-1 run were structural rather than small knobs:

- remove or overlap Atlas-only repropose overhead inside verify;
- reduce drafter forward overhead with fused proposer kernels / fewer host syncs;
- reduce accept/commit overhead by batching recurrent-state copies and synchronizations;
- specialize verifier kernels for the actually used `K` / gamma shape;
- avoid spending more time on already-falsified small knobs unless a new structural reason appears.

## What not to claim

Do not claim:

- Atlas is universally AR-token-exact across the full72 benchmark;
- LuceBox has proven universal token-id exactness in the available artifacts;
- z-lab crop-only DFlash is a safe quality reference for hybrid Qwen3.5;
- per-prompt 29-30 tok/s means full72 aggregate 30 tok/s;
- a speed-only result is acceptable if it worsens matching or leaks drafter-only tokens.

What can be claimed from the recorded evidence:

- crop-only rollback is unsafe for hybrid recurrent targets;
- LuceBox is the right target strategy family because it snapshots/restores/replays or tracks tree state instead of relying on attention-cache crop alone;
- Atlas already has checkpoint/commit machinery that belongs to the safe family;
- Spark-1 probes showed zero drafter-only leaks in the inspected committed-state traces;
- Atlas outputs were non-pathological and no worse than z-lab on the six-prompt matrix;
- the remaining work is to keep improving full-run throughput while preserving or improving LuceBox-equivalent matching.

## Artifact trail

These paths are from the Spark-1 investigation workspace and may not exist in a clean checkout, but they identify the evidence that produced this note:

- `artifacts/audits/spark1_atlas_lucebox_zlab_exactness_matrix_20260525.md`
- `artifacts/audits/spark1_atlas_dflash_committed_state_probe_20260525.md`
- `docs/SPARK1_ATLAS_DFLASH_REMEDIATION_PLAN_LUCEBOX_EQUIV_20260525.md`
- `docs/structural_findings/HYBRID_CACHE_ROLLBACK_INVARIANT_VIOLATION_20260525.md`
- `LUCEBOX_EMPIRICAL_STANDARD_20260525T1920Z.md`
- `USER_DIRECTIVE_LUCEBOX_EQUIVALENT_SUCCESS_20260525.md`

External Spark-1 reference paths used during the audit:

- `/home/dnola/dflash-exactness-audit-20260525/results/official_reference_exactness_v2.json`
- `/home/dnola/dflash-exactness-audit-20260525/results/lucebox/lucebox_exactness_summary.json`
- `/home/dnola/dflash-exactness-audit-20260525/logs/lucebox/test_dflash_ddtree.log`

## Recommended next actions

1. Keep a retained six-prompt exactness matrix in CI or a reproducible script for hybrid DFlash changes.
2. Add a small committed-state approval regression probe that asserts zero drafter-only leaks on generic and chunked verifier paths.
3. Update runtime summaries to always record verifier mode, gamma, quantization, and whether committed-state trace hooks were enabled.
4. Continue speed work only on branches that preserve the LuceBox-equivalent matching bar.
