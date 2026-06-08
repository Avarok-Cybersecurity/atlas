# Noisy-Logits Resilience — 10-agent synthesis

**Date**: 2026-05-26
**Goal**: Produce a ranked, scope-honest plan for making Atlas resilient against the character-level drift Qwen3.6-35B-A3B-FP8 produces (axum→axut, lean://, runaway bash, etc.). Each entry below is sourced from the per-topic agent files in `research3_*.md`.

---

## Convergent findings across the 10 agents

| Insight | Sources | Strength |
|---|---|---|
| **L31-L39 deep-layer cliff is THE hotspot** for FP8 drift compounding | research3_beam_bestofk (Wired-for-Overconfidence arXiv 2604.01457), research3_spec_verify, research3_noise_resilient (INSIDE clipping), Atlas Phase 2b memory | 4 independent agents converge |
| **No published FP8-specific decoder exists** — Atlas is unexplored territory | research3_beam_bestofk, research3_noise_resilient, research3_quantization_sampler | unanimous |
| **Margin-based detection** (top1 vs top2 logit gap) is the cheapest drift signal | research3_beam_bestofk (MARS 2601.15498), research3_self_consistency, research3_quantization_sampler | 3 agents converge |
| **N=3 SC voting in tool-arg regions** under T=0.6-0.7 (higher than current 0.3 because grammar removes diversity) | research3_self_consistency (SOFT-SC, TrACE, USC), research3_beam_bestofk | 2 agents converge |
| **Hidden-state linear probe** on residual stream catches drift at <1ms, ~85% accuracy, validated on Qwen3.5-A3B family | research3_critic_verify (3 independent 2025 papers on same model family) | strong single source |
| **Full beam search RULED OUT** — makes confidently-wrong WORSE | research3_beam_bestofk, research3_constrained_beam (CABS suggests sub-structure rerank, not beam) | 2 agents converge |
| **Self-critique INSUFFICIENT** — same noise corrupts self-check | research3_critic_verify (Self-Verification Dilemma arXiv 2602.03485) | 1 strong source |
| **EAGLE/Medusa-style spec decode WON'T catch our drift** — they verify by distribution match (if both quantized models prefer wrong token, accepted) | research3_spec_verify | 1 strong source |

---

## Master finding: the drift catalog gives an empirical priority order

The `research3_drift_catalog.md` agent assembled all 15 observed drift modes (across 6 epochs of opencode probes + 4 audit waves) and ranked them by **frequency × impact × Atlas-side fixability**:

| Rank | Drift family | Atlas-side fix | Scope |
|---|---|---|---|
| **#1** | Penalty-zeroing + fastpath family (runaway bash, lean://, same-tool-call loop) | **Invert rep-penalty exemption at `decode_logits_seq.rs:405,449-466`** — keep active outside, exempt only the specific structural marker tokens. This is the F8 finding from Wave-2 sampler audit that we never shipped. | **single-file fix, 1-2 hours** |
| **#2** | Reasoning-collapse cascade (233→0 char decay, reasoning↔action disconnect, mid-paragraph `</think>`) | F1 already shipped (reasoning_content round-trip); add **`POST_THINK_MIN_REASONING` floor** | **half-day** |
| **#3** | Path/value drift family (one-byte path drift, line-number leak, timestamp leak, phantom dir) | **Opencode-side tool-result hardening** — canonical-path echo, replace `(no output)`. Closes 4 patterns at once | client-side fix, not Atlas, **upstream PR** |

The catalog finding: **"None of the 15 patterns are pure 'model-quality, unfixable.'"** Every drift mode has a server-side defense with non-trivial leverage. Top-3 delivers ~70% of observable-drift reduction.

---

## Tiered plan

### Tier A — Ship this week (1-3 day items, high confidence)

| Item | Scope | Source | Why |
|---|---|---|---|
| **A1. Invert rep-penalty exemption (F8 fix)** | 2 hrs | drift catalog #1 | Kills runaway bash + lean:// + same-tool-call loops in one file change |
| **A2. SymSpell tier-2 fuzzy repair against prompt vocab, edit-dist ≤2** | 1 day | research3_fuzzy_repair | **~70% drift recovery**, ~300 LoC, gated by `ATLAS_TOOL_ARG_FUZZ_REPAIR=1` |
| **A3. Doom-loop detector ported from opencode** (last-N tool-call hash compare → early stop) | half-day | research3_production_mitigations | Cheap kill-switch for the runaway loops Tier-5c can't catch |
| **A4. POST_THINK_MIN_REASONING floor** | half-day | drift catalog #2 | Stops mid-paragraph `</think>` and reasoning collapse |
| **A5. Min-p floor 0.08 on FP8 path** | 30 min | research3_quantization_sampler #3 | Trivial sampler-side knob, raises distribution tail |
| **A6. MTP acceptance rate as drift gauge** | 1-2 days | research3_production_mitigations #4 | **Reuses existing MTP K=2 infra**; <30% accept for 64 tokens → `finish_reason="drift_detected"` |

**Tier-A total: ~1 week.** Expected impact: doom-loop attractors gone, ~70% of path drift auto-corrected, reasoning collapse bounded, free drift detector running.

### Tier B — Next 2-3 weeks (high-leverage detectors)

| Item | Scope | Source | Why |
|---|---|---|---|
| **B1. Margin-ratio drift detector** (MARS-inverse) | 2-3 days | research3_beam_bestofk #1, research3_quantization_sampler #2 | Detect "suspiciously high margin" (FP8 confidently-wrong signature) cheaply |
| **B2. Top-nσ truncation with FP8-aware n** (Top-nσ paper arXiv 2411.07641) | 1-2 days | research3_quantization_sampler #1 | Theoretically-grounded sampler change; FP8 paths get n=1.3 vs BF16 n=1.0 |
| **B3. Linear probe on residual stream at `<tool_call>` opener** | ~3 days | research3_critic_verify #1 | **<1ms, ~85% catch rate**, bootstrap labels from our existing `op_drift.json` / `MASTER_DRIFT_TABLE.md` |
| **B4. N=3 Soft-SC voting on tool-arg region only**, T=0.7 inside grammar mask | ~1 week | research3_self_consistency #1, research3_beam_bestofk #5 | Region-scoped, doesn't slow non-tool tokens; 3× cost on ~10% of tokens = ~20% slowdown only on agentic turns |

**Tier-B total: ~2-3 weeks.** Adds three independent detection signals (margin, probe, voting) and one principled sampler change.

### Tier C — Multi-week, biggest leverage (architectural)

| Item | Scope | Source | Why |
|---|---|---|---|
| **C1. Diagnostic gate**: measure top-K logit overlap FP8 vs BF16 inside tool-arg spans | 1 day, **gates C2-C4** | research3_constrained_beam #1 | **If the right token isn't in FP8's top-K, all of C2-C4 are dead ends**. Must measure first. |
| **C2. USCD contrastive decoding** with BF16 reference head | 1-2 weeks, requires C1=pass | research3_noise_resilient #1 | Subtract FP8 logits from BF16-reference logits to denoise; only viable if reference is present |
| **C3. Selective BF16 upcast of L31-L39 attention layers** | 2-3 weeks | research3_beam_bestofk #2, research3_spec_verify #2, research3_noise_resilient #5 | **THE structural fix**. Hotspot precisely identified across 4 agents + Atlas Phase 2b memory |
| **C4. QSpec-style FP8-draft + BF16-verify on tool-arg tokens only** | 3-4 weeks | research3_spec_verify #1 | Recovers quality at modest throughput cost; only on ~10% of decode spend |

### Tier D — Out of Atlas's scope (deferred)

- **D1. opencode-side tool-result hardening** (canonical-path echo, `(no output)` replacement). Client-side fix; closes 4 catalog patterns at once. Send PR to sst/opencode if user wants.

### Ruled out (don't try)

- **Full beam search** — makes confidently-wrong worse (research3_beam_bestofk, research3_constrained_beam)
- **Lookahead decoding** — noise is in target, not drafter (research3_beam_bestofk)
- **Contrastive decoding without BF16 reference** — gated on C1+C2 (research3_noise_resilient)
- **Stochastic rounding for inference** — confirmed harmful (research3_noise_resilient, matches Atlas Phase 2b)
- **Self-critique loop** — same noise corrupts the self-check (research3_critic_verify)
- **EAGLE/Medusa naively** — distribution-match accepts confidently-wrong tokens (research3_spec_verify)
- **Separate critic model on GB10** — doesn't fit memory budget alongside 80B-class models (research3_critic_verify)
- **AdaDec with τ=1.0** — Phase-1 measurement showed 30% trigger rate on Qwen3.6-FP8 vs 7% paper baseline; threshold needs to be ~3.0 to be cost-effective, which loses most of the signal

---

## Honest expectations

**Tier A alone** should deliver a meaningful improvement on the opencode multi-turn task. Specifically:
- **A1 (penalty inversion)** stops the runaway bash + lean:// loops we observed in Wave-1 and Wave-3 directly
- **A2 (fuzzy repair)** auto-corrects the `axum→axut` / hyphen-dropped path drift that Tier-5c retry can't reliably break out of
- **A3 (doom loop)** terminates degenerate loops before they exhaust max_tokens
- **A6 (MTP acceptance gauge)** gives early termination on whole-turn drift

**Tier B** adds detectors that improve over time as we tune thresholds against real probe data.

**Tier C** is where we either accept the FP8 precision floor or pay for selective BF16 upcast. Gate on C1 — if the right tokens are in FP8's top-K, we don't need C3.

**Tier A → Tier B → Tier C** is the suggested order. Each tier validates the next.

---

## Recommended starting point

Land **Tier A (A1 through A6) as one image**:
- 1 week of work
- Mostly independent fixes (no entangled refactors)
- Each one can be A/B tested against current epoch4 baseline
- A1 alone (the F8 penalty inversion we never shipped) is likely the biggest single win

Then re-run the opencode axum probe and measure. If file count goes from 0-3 → 8-15+, Tier A is sufficient. If still degenerate, move to Tier B detectors.

---

## Open questions for the user

1. **Approve Tier A bundle (6 fixes, ~1 week)?** This is the recommendation.
2. **For Tier D (opencode-side)**: file PRs to opencode for canonical-path echo + `(no output)` replacement? Or stay strictly Atlas-side?
3. **For Tier C gate (C1 diagnostic)**: run it after Tier A ships, or in parallel?
