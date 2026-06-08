# Combating Agentic Wandering in Atlas + Qwen3.6-35B-A3B-FP8

**Author**: Claude research subagent (Opus 4.7)
**Date**: 2026-05-26
**Target failure**: opencode "create rust axum project" → 5-7 bash/ls/mkdir calls, no `write` call, runs out of tool-call budget.
**Constraint**: Inference-time only. No retraining. Server-side preferred.

---

## TL;DR

The community has a name for this: **overacting** (TACT 2605.05980), the **lazy-agent gap** (ASA 2602.04935),
and **goal drift through inaction** (Goal-Drift 2505.02709 — GD_inaction). Token-amplification of
142x has been measured on Qwen-Code, 609x on MiniMax-M2 — Atlas's existing `--max-num-tools` cap is
load-bearing, but not sufficient. The most Atlas-deployable interventions cluster into three layers:

1. **Sampler-layer** (Atlas-native): logit bias on exploratory tools after N consecutive non-commit calls,
   short-CoT enforcement, tool_choice gating.
2. **Activation-layer** (Atlas-native, needs hidden-state hook): TACT-style steering vectors at the
   `</think>` token.
3. **Orchestration-layer** (opencode-side or Atlas proxy): budget-tracker prompt injection, ReflAct
   goal reminders, OpenFang/OpenClaw n-gram loop detection.

Top-5 ranked below.

---

## Ranked Interventions

### #1 — TACT: Activation-Steering for Overthinking / Overacting
**Paper**: `arxiv 2605.05980` (Mitigating Overthinking and Overacting in Coding Agents via Activation Steering)
**URL**: https://arxiv.org/html/2605.05980
**Headline result**: **+5.8 pp** SWE-resolve on Qwen3.5-27B, **+4.8 pp** on Gemma-4, **up to 26%** fewer
steps to resolve. Single-rollout (no extra inference passes).

**Mechanism**:
1. **Offline (one-time, hours):** Run an LLM-as-judge on a labeled trajectory corpus, tagging each
   step as Overthinking (OT), Overacting (OA, == "tool calls that return cached info or fire without
   integrating recent observations"), or Calibrated (CAL). Collect hidden states at the `</think>`
   token. Compute mean-difference axes: `v_OA = mean(OA_hidden) - mean(CAL_hidden)`. Gram-Schmidt
   orthogonalize against `v_OT`. ~15 LoC.
2. **Online (per token, ~free):** During decoding, project the residual stream at layer ℓ onto these
   axes. If |projection| > k·σ, subtract the excess. ~15 LoC hook (Algorithm 1 in paper).

**Trigger**: Continuous, per-token, projection magnitude vs. calibration band.
**Server-side**: YES — pure Atlas. Needs a `register_residual_hook(layer, vector, threshold)` API,
which is the same shape Atlas already has for KV-cache modification.
**OA definition is a near-perfect match** for opencode's failure: "tool calls retrieving information
already observed" = the 5th `ls` call after 4 prior `ls`/`bash` returned the same empty dir.

**Implementation cost**:
- Axis-extraction pipeline: ~300 LoC Python (uses HF transformers), 1–2 days.
- Atlas hook: ~50 LoC Rust in sampler/decode path, 1–2 days.
- Calibration corpus: 200–500 labeled opencode trajectories (use Sonnet/GPT-OSS as judge), 1 day.
- **Total: ~5 days, ~400 LoC across Python + Rust.**

**Risk**: needs hidden-state access at the right layer (paper used L21 for 8B; L40-ish for 35B
likely). One axis per quant format (FP8 hidden states differ from BF16 — re-calibrate).

---

### #2 — Brief-Is-Better: Cap CoT Budget at 32–64 Tokens
**Paper**: `arxiv 2604.02155` (Brief Is Better: Non-Monotonic Chain-of-Thought Budget Effects in
Function-Calling Language Agents)
**URL**: https://arxiv.org/pdf/2604.02155
**Headline result**: **+45% relative** task accuracy on BFCL-v3 Multiple at 32-token CoT vs.
no-CoT; **256-token CoT regresses BELOW no-CoT baseline**. Tested on Qwen2.5-1.5B.

**Mechanism**: Bound the model's reasoning prefix per turn. Above ~32–64 tokens of pre-tool-call
prose, function-calling accuracy falls non-monotonically — extra thinking is where the model talks
itself into "let me first check…" exploration loops.

**Why it fits opencode**: Qwen3.6 is a thinking model. Open `<think>` block + 800-token ramble
about "I should first verify the environment" is the literal cause of the 5-bash wander. Cap the
think section at ~64 tokens and the model must either commit or stop.

**Trigger**: After `<think>` token detected; counter at 64; force `</think>`.
**Server-side**: YES — pure Atlas. This is a stop-token rewrite in the sampler.

**Implementation cost**:
- ~30 LoC in `sampler.rs`: detect `<think>`, count, inject `</think>` at cap.
- Plus per-model config (HARDWARE.toml-style: `max_think_tokens_per_turn = 64`).
- **Total: ~1 day, ~50 LoC.**

**Risk**: Qwen3.6 was trained with long-think; hard-clamping may hurt single-turn reasoning quality.
Solution: ramp the cap by *tool-call turn index* (first turn 256, then 64, then 32) — degrades
gracefully into commitment.

---

### #3 — Tool-Call Loop Guard (OpenFang / Strands-style DebounceHook)
**Paper / source**: `arxiv 2602.14798` (Overthinking Loops in Agents) +
production frameworks (Strands `DebounceHook`, OpenFang SHA-256 pattern hash, OpenClaw rolling window).
**URLs**:
- https://arxiv.org/html/2602.14798v1
- https://dev.to/aws/how-to-prevent-ai-agent-reasoning-loops-from-wasting-tokens-2652

**Headline result**: Strands `DebounceHook` reduced demo tool calls **from 14 → 2 (7x)**. Paper
itself measured **142x token amplification on Qwen-Code, 609x on MiniMax-M2** under loop attacks —
even modest guard recovers most of that. NoWait (generation-level mitigation) was demonstrated
**insufficient**: 15.37x amplification persisted; the loop must be broken at tool-call structure
level, not token level.

**Mechanism**: Hash `(tool_name, args)`. Maintain rolling window (size 3–5). On 2nd identical hash:
inject system warning. On 3rd: cancel the tool call and force a different action.

**Why it fits opencode**: opencode's "5x bash ls" failure is literally this. Strands data shows a
hard guard at N=2 identical calls reclaims 7x of wasted budget.

**Trigger**: SHA-256 of (tool_name, normalized_args) appears ≥2 times in last 5 calls.
**Server-side**: Atlas can host this as a *response post-processor* in the OpenAI-compat layer
(commit-cbf951c-style wrapper). Doesn't need model internals. opencode doesn't need to change.

**Implementation cost**:
- ~80 LoC Rust in the OpenAI-compat response handler.
- Wire into existing tool_choice machinery. When triggered, force `tool_choice="none"` and inject
  `<system>You've called bash 5 times. Call write or finish.</system>` on next turn.
- **Total: ~1–2 days, ~100 LoC.**

**Risk**: Legitimate retries (e.g. file-not-found → check spelling → retry) get false-positive
blocked. Mitigation: only block when *both* tool_name AND args match exactly twice; warn on
tool_name-only repetition.

---

### #4 — Goal-State Reflection (ReflAct) injected as System Reminder
**Paper**: `arxiv 2505.15182` (ReflAct: World-Grounded Decision Making in LLM Agents via
Goal-State Reflection)
**URL**: https://arxiv.org/html/2505.15182v2
**Headline result**: **+36.4% on ALFWorld** (85.1→93.3% with GPT-4o), **+38.1% on Jericho**, +8.5%
on ScienceWorld. Beats ReAct + Reflexion + WKM combined.

**Mechanism**: Every N turns (paper uses every turn), inject a forced "internal-state vs goal"
reflection: `"Currently I am at [X], holding [Y]. My goal is to [Z]. Does my next action move me
toward Z?"`. Replaces ReAct's "Thought:" with goal-relative reasoning.

**Why it fits**: opencode's failure mode is precisely "lost the goal" — model explores filesystem,
forgets the user asked to *create* an axum project.

**Trigger**: Every K tool-call turns (K=3–5), OR after N exploratory calls without state-changing
calls (`write`/`edit`).
**Server-side**: Atlas can inject the reflection as a synthetic system message in the OpenAI-compat
turn-management layer — opencode never sees it. Cleaner: opencode-side as a system-prompt
augmentation.

**Implementation cost**:
- ~60 LoC in OpenAI-compat handler: track last K turns of tool names; if no state-changing tool,
  prepend `<reminder>You have called [bash, ls, mkdir] 4 times. The user asked for "create rust
  axum project". Your next action should be `write` to actually create a file.</reminder>`
- **Total: ~1 day, ~80 LoC.**

**Risk**: prompt injection — model may treat it as part of user message. Use a model-specific
sentinel (Qwen3.6 supports `<|im_start|>system`).

---

### #5 — Budget Tracker (BATS) Prompt-Level Status Injection
**Paper**: `arxiv 2511.17006v1` (Budget-Aware Tool-Use Enables Effective Agent Scaling)
**URL**: https://arxiv.org/html/2511.17006v1
**Headline result**: **+12.0 pp** absolute on BrowseComp (12.6→24.6%) with Gemini-2.5-Pro.
**31.3% cost reduction** at parity accuracy.

**Mechanism**: Append a tracker block to every tool response: `"Query Budget Used: 4/10, Remaining:
6. Budget regime: HIGH. Policy: tools should now be selected for commitment, not exploration."`
Adds policy guidelines mapped to budget regimes (high/medium/low/critical).

**Why it fits**: Qwen3.6-35B has no native concept of a global budget. By the 5th bash call it
doesn't know it's burning. A tracker makes the budget *first-class context*.

**Trigger**: Append on every tool response. Switch budget-regime hint at thresholds.
**Server-side**: Atlas's OpenAI-compat layer appends to the `tool` message before returning it to
opencode. opencode is unmodified.

**Implementation cost**:
- ~40 LoC Rust to count tool calls per session_id and append to tool-result messages.
- **Total: ~0.5 day, ~50 LoC.**

**Risk**: Inflates context. At 10 calls × 80-token tracker = 800 tokens added. Mitigate by appending
only when regime *changes*.

---

## Honorable Mentions (not top-5 because lower ROI for Atlas)

- **ASA (`2602.04935`)** — fixes "Lazy Agent" miss rate **0.18→0.50 F1**. Requires pre-trained
  steering vectors per domain; high calibration cost. Re-evaluate when opencode trajectory corpus
  is large enough (>1000 traces).
- **MICE for CATs (`2504.20168`)** — model-internal confidence for "should I call this tool?".
  Requires a *learned classifier* (not training-free per Atlas constraint). Defer.
- **Sequence-Level Entropy stop (`2510.08146`)** — 25–50% compute savings on math reasoning. Author
  explicitly says: "Doesn't address pathological repetition." Wrong tool for this job.
- **EET (Experience-Driven Early Termination, `2601.05777`)** — 19–55% cost reduction on SWE-bench
  but signal is *past trajectories*, requires a corpus. Useful as Phase-2 augmentation.
- **Inter-Rollout Action Agreement / TrACE (`2604.08369`)** — needs multiple rollouts per step,
  doubles compute. Wrong economics for Atlas decode-throughput goals.
- **BiasBusters (`2510.00307`)** — 25–38% selection bias measured but mitigation is filter+uniform
  sampling, which is wrong for opencode (we want commitment bias, not uniformity).
- **CAR-bench, SWE-Bench-Pro (`2509.16941`), Goal-Drift report (`2505.02709`)** — benchmarks, not
  interventions. Useful for validating fixes.

---

## Recommended Atlas Roadmap

**Phase 1 — Ship in 1 week (~150 LoC total, no model corpus needed)**:
- #2 Brief-Is-Better: cap `<think>` per turn at 64 tokens after turn-3.
- #3 Loop Guard: SHA-256 hash, ban after 2 identical (tool_name, args).
- #5 Budget Tracker: append regime hint to tool responses.

These are **purely server-side**, pure sampler/handler edits, zero opencode changes, zero retraining.
Expected combined recovery: based on Strands 7x + Brief +45% + BATS +12pp, conservatively **30–50%
of currently-failing "create axum project" runs should now reach the `write` call**.

**Phase 2 — 2 weeks once Phase-1 ships**:
- #4 ReflAct goal-reminder injection (needs prompt-template care per model).
- Trajectory corpus collection (logs from Phase-1 deploy).

**Phase 3 — 1 month, requires hidden-state hook in Atlas**:
- #1 TACT activation steering. Highest expected impact (+5.8pp SWE-resolve) but biggest engineering
  surface. Land after the corpus from Phase-2 exists.

---

## Trigger Condition Matrix

| Intervention | Trigger condition | Side | Cost (days) | LoC |
|---|---|---|---|---|
| #1 TACT | Per-token: `‖proj(h_ℓ, v_OA)‖ > k·σ` | Atlas (residual hook) | 5 | ~400 |
| #2 Brief-CoT | After `<think>`; counter ≥ N_turn-specific | Atlas sampler | 1 | ~50 |
| #3 Loop Guard | SHA-256(tool, args) repeats 2x in window 5 | Atlas OAI-compat | 1.5 | ~100 |
| #4 ReflAct | Every K=3 turns OR N non-state-changing tool calls | Atlas OAI-compat (or opencode) | 1 | ~80 |
| #5 Budget Tracker | Every tool response | Atlas OAI-compat | 0.5 | ~50 |

**Total Phase-1 (2+3+5): ~3 days, ~200 LoC.** All in Rust, in the OpenAI-compat handler / sampler.
No CUDA work. No model retraining. No opencode changes. Should be shippable in next alpha.

---

## Direct quotes anchoring the ranking

- TACT paper: *"a_t returns an observation already in K_t, or fires without adequately processing
  recent observations"* — opencode's bash-without-write defined exactly.
- Overthinking-Loops paper: *"NoWait constrains token generation within individual reasoning
  steps... cycle-inducing tools exploit a different mechanism"* — confirms loop guards must operate
  at tool-call level (Phase-1 #3), not sampler level alone.
- Brief-Is-Better paper: *"32-token CoT improves Qwen2.5-1.5B accuracy by +45% relative; 256-token
  CoT degrades below no-CoT baseline."* — quantifies #2 directly.
- Goal-Drift report (`2505.02709`): introduces `GD_inaction` = "passive abandonment of
  goal-consistent behavior" — exactly the opencode failure: not wrong actions, but *missing* the
  `write` action.
- BATS: *"31.3% cost reduction… 24.6% vs 12.6% ReAct"* — quantifies #5.

Sources:
- [TACT (arxiv 2605.05980)](https://arxiv.org/html/2605.05980)
- [Brief Is Better (arxiv 2604.02155)](https://arxiv.org/pdf/2604.02155)
- [Overthinking Loops in Agents (arxiv 2602.14798)](https://arxiv.org/html/2602.14798v1)
- [ReflAct (arxiv 2505.15182)](https://arxiv.org/html/2505.15182v2)
- [Budget-Aware Tool-Use / BATS (arxiv 2511.17006)](https://arxiv.org/html/2511.17006v1)
- [ASA Training-Free Representation Engineering (arxiv 2602.04935)](https://arxiv.org/html/2602.04935)
- [MICE for CATs (arxiv 2504.20168)](https://arxiv.org/abs/2504.20168)
- [Think Just Enough / Sequence-Level Entropy (arxiv 2510.08146)](https://arxiv.org/html/2510.08146v1)
- [EET Experience-Driven Early Termination (arxiv 2601.05777)](https://arxiv.org/pdf/2601.05777)
- [Inter-Rollout Action Agreement / TrACE (arxiv 2604.08369)](https://arxiv.org/pdf/2604.08369)
- [BiasBusters (arxiv 2510.00307)](https://arxiv.org/html/2510.00307v1)
- [Goal Drift Technical Report (arxiv 2505.02709)](https://arxiv.org/pdf/2505.02709)
- [Canonical Path Deviation (arxiv 2602.19008)](https://arxiv.org/pdf/2602.19008)
- [Strands DebounceHook pattern (AWS Dev blog)](https://dev.to/aws/how-to-prevent-ai-agent-reasoning-loops-from-wasting-tokens-2652)
