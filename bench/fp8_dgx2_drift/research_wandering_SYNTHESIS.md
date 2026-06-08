# Agentic Wandering — Synthesis of 5-agent research sweep (2026-05-26)

Five parallel research agents returned with 5 reports. Synthesizing here for the
2026-05-27 morning decision.

## The big picture

Wandering has **two distinct root causes** that need separate attacks:

### Cause 1 — FP8 MoE routing drift (W5)

**Documented identical failure**: vLLM Issue #36872 reports `Qwen3.5-35B-A3B-FP8`
(sibling architecture) collapses across consecutive tool-calling turns:
spec-decode acceptance goes 61% → 0.9% → 0%.

**Mechanism**: FlashInfer Issue #2822 — on the FP8 monolithic MoE kernel,
`Qwen3.5-35B-A3B`'s gate produces all-negative router logits across **every one
of its 40 MoE layers**. The FP8 routing path picks *different* top-k experts
than the fp32 reference. GSM8K drops to 0%.

Our own evidence: Atlas drift catalog shows MoE top-k overlap collapsing 8/8 →
3/8 by L38 — exactly the depth where "write a file" vs "explore with bash" gets
committed.

**Why sampler-level fixes can't help**: top-k expert flips happen inside
transformer blocks, 40 layers upstream of the sampler. By the time the sampler
sees "wandering" probability mass, the wrong experts already produced it. This
explains our 30% sampler ceiling cleanly.

**Confidence: ~75% that FP8 is causing or strongly exacerbating wandering.**

### Cause 2 — Generic agentic "overacting" (W1, W2, W3, W4)

Independent of quantization, all LLM agents exhibit this. The community has
multiple names:
- **"overacting"** (TACT, arxiv 2605.05980) — tool calls returning info already observed
- **"lazy-agent gap"** (ASA 2602.04935)
- **"GD_inaction"** (Goal-Drift 2505.02709)

Token amplification under loops: 142× on Qwen-Code, 609× on MiniMax-M2.

## Ranked interventions (deployable today)

### P0 — Zero Atlas changes (5 minutes)

**opencode `steps: 10` config for Qwen3.6 agent** (W3 BIG ANSWER)

opencode has a per-agent `steps` config (legacy `maxSteps`) that caps tool-call
rounds and **auto-injects a "summarize and recommend remaining tasks" prompt**
when hit. Setting `steps: 10` in our agent profile is the single most direct
fix.

Also critical: opencode's existing doom-loop check (`DOOM_LOOP_THRESHOLD = 3` at
processor.ts:32) only triggers on **byte-identical** tool+input. That's why our
5-different-bash-call wandering slips past it. Upstream issue #12716 confirms
the community is aware.

### P1 — Atlas server-side (1-3 days each, additive)

| # | Intervention | LoC | Expected gain | Source |
|---|---|---|---|---|
| P1a | **Tool-name DRY n-gram penalty** (penalize repeated `<tool_call>{"name":"bash"` token sequences using existing DRY infra) | ~50, 1d | breaks consecutive-bash sequences | W4 #1, vLLM PR #11368 |
| P1b | **Loop-Guard via tool-call hash** (SHA-256 of (tool_name, args), ban after 2 repeats) | ~100, 1.5d | 142×→7× token recovery; class-based not byte-identical | W1 #3, W2 B, W3 Roo Code |
| P1c | **BATS Budget Tracker** (append budget regime to every tool response: "{remaining_steps} steps left") | ~50, 0.5d | +12pp BrowseComp, -31% cost | W1 #5, smolagents |
| P1d | **Brief-is-Better think-cap** (cap `<think>` to 32-64 tokens/turn) | ~50, 1d | +45% BFCL-v3 at 32 tok | W1 #2 |
| P1e | **smolagents-style "you have N steps remaining" injection** at user-message position | ~150, 2d | clean force-finalize | W2 C, W3 Claude Code |

**Note about Brief-is-Better**: aggressive (cap thinking 768 → 32). Risk: might
hurt legitimate reasoning. Test carefully.

### P2 — Higher-leverage but more invasive (3-5 days)

| # | Intervention | LoC | Source |
|---|---|---|---|
| P2a | **TACT activation steering at `</think>`** | ~400, 5d | W1 #1, +5.8pp SWE-resolve |
| P2b | **ReflAct goal-state reminder injection** | ~80, 1d | W1 #4, +36.4% ALFWorld |
| P2c | **Per-tool-name budget** (bash=5, grep=10) | ~110, 2d | W2 A |
| P2d | **`tool_choice="none"` flip after K calls** | ~70, 1d | W2 D — most invasive |

### P3 — Architectural (multi-week)

| # | Intervention | Notes |
|---|---|---|
| P3a | **BF16 inference (test FP8 hypothesis)** | W5 Experiment A — DEFINITIVE test. If pass rate climbs 30%→90%, FP8 is the issue. We have `hf_forward_bf16_unquant.py`. Memory: 70GB BF16 weights + activations, tight on 119GB GB10. |
| P3b | **Selective L31-L39 BF16 upcast** | W5 Experiment B. ~9GB memory cost. Less invasive than full BF16. Phase-2b already identified L31-L39 as the FP8 regression zone. |
| P3c | **Force fp32 routing under FP8 expert weights** | W5 Experiment C. FlashInfer #2822's resolution. Cheapest "fix" if it works. Need to instrument gate sign at L0/L20/L38 first. |
| P3d | **EAST entropic activation steering** | W4 #4, ~1 week kernel work |

## Recommended sequence

1. **P0 immediately**: try opencode `steps: 10` config + N=10 harness. 5 minutes
   of setup, ~50 min to measure. If pass rate jumps significantly, our problem
   is half-solved without any Atlas changes.

2. **P3 Experiment A in parallel**: kick off the BF16 inference experiment.
   This is the decisive test of the FP8 hypothesis. Multi-hour setup; can run
   while the harness work proceeds.

3. **P1a + P1b + P1c stack** (~3 days total): tool-name DRY + loop-guard hash
   + budget tracker. These are additive, low-risk, address the documented
   "overacting" pattern with strong evidence. Expected combined: 30-50%
   recovery of currently-failing runs per academic numbers.

4. **Conditional on P3a result**:
   - If BF16 hits 90%+, ship selective L31-L39 upcast (P3b) as the production fix.
   - If BF16 also wanders, the problem is fundamental to the model + agent setup; deeper work needed.

## Open questions for user direction

1. **Authorization to modify opencode config** (~/.config/opencode/agents/atlas.md to add `steps: 10`)? Or stand up a per-harness agent definition? The former is one line; the latter requires harness script changes.

2. **BF16 experiment scope**: full Qwen3.6-35B-A3B BF16 (70GB weights, tight memory) vs selective L31-L39 upcast (~9GB extra)?

3. **Order**: P0 first (free) and then P1 stack while BF16 setup happens? Or skip straight to BF16 since it's the decisive test?

## Files

Full per-agent reports:
- `research_wandering_arxiv.md` (W1 — academic)
- `research_wandering_prod.md` (W2 — production serving)
- `research_wandering_frontends.md` (W3 — agentic frontends)
- `research_wandering_replan.md` (W4 — replanning/reflection)
- `research_wandering_fp8.md` (W5 — FP8 linkage — the big one)

Total: ~9000 words of research across 5 angles.

## My one-line take

**It's likely FP8.** The W5 evidence is too aligned with our observations to
ignore. The smartest next move is the cheapest decisive test: opencode
`steps:10` to bound the agentic-wandering half, then a quick BF16 sanity check
to test the FP8-is-the-cause hypothesis. Both can happen in the remaining ~8h
of the /loop window if the user authorizes.
