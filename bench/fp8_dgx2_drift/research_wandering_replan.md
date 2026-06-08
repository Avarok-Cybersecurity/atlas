# Mid-Trajectory Replanning & Self-Reflection as Anti-Wandering Interventions

**Target failure:** Qwen3.6-35B-A3B-FP8 on opencode emits 5–7 bash exploration calls without
ever writing files for 30–40% of "create rust axum project" runs. The model commits to
"exploring" instead of "acting." This document ranks interventions by **server-side
deployability** (Atlas is a stateless OpenAI-compatible inference server; opencode is
the client).

---

## TL;DR — Ranked Deployability

| # | Intervention                                  | Where it lives          | Atlas-only? | Cost      |
|---|-----------------------------------------------|-------------------------|-------------|-----------|
| 1 | Tool-name n-gram penalty (DRY-style)          | sampler / logits proc   | YES         | ~1 day    |
| 2 | Per-request "action-budget" anchor injection  | tokenizer template hook | YES (partial) | ~2 days |
| 3 | Focused-ReAct early-stop on duplicate tool    | request post-processor  | YES         | ~3 days   |
| 4 | Entropic Activation Steering (EAST)           | model forward hook      | YES         | ~1 week   |
| 5 | AdaPlanner / ReWOO / Task-Shield rewrites     | client orchestrator     | NO          | weeks+    |

**Bottom line: #1, #2, #3, #4 are all stateless server-side. Atlas does not need
opencode changes for any of the top four.**

---

## 1. Tool-Name N-gram Repetition Penalty (DRY-style sampling)

**Paper / source:** DRY ("Don't Repeat Yourself") sampler, originally Oobabooga
text-gen-webui PR #5677, ported to vLLM in PR #11368 by AlpinDale (2024). General
n-gram blocking is standard HuggingFace `no_repeat_ngram_size`. The Focused-ReAct
paper (Li et al., WiNLP 2024, arXiv 2410.10779) reports **18%–530% accuracy gain
and up to 34% runtime reduction** by detecting duplicate actions and early-stopping
— Atlas can apply the same signal *during* sampling instead of after.

**Why it fits the failure mode:** When the model is "wandering" it re-emits
`<tool_call>{"name":"bash"` over and over. That literal token sequence repeats.
DRY penalizes any token whose continuation would extend an n-gram already in the
generated prefix, with a multiplier `α · base^(match_length − allowed)`.

**Trigger condition (concrete):** After the second `bash` tool call in a turn,
exponentially penalize the logits for the token sequence
`<tool_call>{"name":"bash"`. Atlas already has the assistant turn boundary
(tool_call IDs in the prompt). Count occurrences in *prefix tokens* (no client
state needed — it's all in the request).

**Implementation cost:** Atlas's sampler lives in `src/sampler/` (Rust). DRY is
~80 LoC of token-scan + logit-shift; the n-gram scan reuses existing prefix
buffers. Sampling parameter `dry_multiplier` is OpenAI-compatible (vLLM already
ships it). **~1 day, additive, low risk.**

**Atlas-only?** YES. Sampling parameters are server-side. Opencode does not need
to know.

**Caveat:** Penalizes legitimate repeat tool calls too (e.g., `bash ls /a` then
`bash ls /b`). Mitigate by penalizing only on the *tool-name token boundary*, not
on arguments. Atlas's tool-call grammar (XGrammar) makes this exact boundary
trivially detectable.

---

## 2. Per-Request "Action-Budget" Anchor Injection

**Paper / source:** Plan-and-Solve (PS) prompting, Wang et al., ACL 2023
(arXiv 2305.04091): the phrase *"devise a plan, then carry it out step by step"*
is a single-line system-prompt hack that reduces missing-step errors. Focused-ReAct
(arXiv 2410.10779) re-iterates the user's original goal at every reasoning step.
**Plan-and-Solve achieves zero-shot CoT-level gains with a one-sentence prefix.**

**Why it fits:** Qwen3.6's wandering is partly a missing-explicit-plan failure —
the model emits exploratory bash because no plan token has been committed. A
forced anchor like *"Before any tool call, state in ≤30 tokens which file you will
write first"* makes the act of *not* writing a file structurally awkward.

**Trigger condition:** Atlas detects (a) a coding-agent chat template signature
(opencode's system prompt is identifiable — it contains `## Tools` and bash/edit
descriptions), and (b) a user message containing creation verbs ("create",
"build", "make", "implement"). When both fire, append a short anchor sentence at
the end of the system prompt:

```
ANCHOR: Before your first tool call, write one line:
"PLAN: I will create <file> first." Then make that tool call.
```

**Implementation cost:** ~2 days. Atlas already has a request preprocessor
(`src/server/handlers/chat.rs` style). Anchor is gated by env var so it can be
A/B'd. Risk: anchor leaks into output — fix with a stop-sequence on `"PLAN: "`
in the streaming layer, or strip the line server-side before returning.

**Atlas-only?** **PARTIAL.** Injecting into the system prompt is server-side
and stateless, but it is *modifying the user's request*. We should expose it as
an opt-in header (e.g. `X-Atlas-Action-Anchor: 1`) or env-gated experiment.
This is the same pattern as vLLM's `--enable-reasoning` / `--reasoning-parser`
— a server-side prompt mutation that's opt-in.

---

## 3. Focused-ReAct Early-Stop on Duplicate Tool Call

**Paper:** Li, Xu, Chen — *Focused ReAct: Improving ReAct through Reiterate and
Early Stop*, WiNLP 2024 (arXiv 2410.10779). Verified mechanism: when the
*current generation* repeats an action seen in the prior turns, the harness
"triggers a termination request instructing the model to generate a final answer
based on existing information." Reported gains: 18%–530% accuracy, –34% runtime
on Gemma-2 2B / Phi-3.5-mini / Llama-3.1 8B.

**Why it fits:** This is the closest published analog to the opencode failure
mode. Wandering = duplicate `bash` calls. Stopping the duplicate and forcing the
model to "make a decision now" with what it already knows is exactly the
needed intervention.

**Trigger condition:** During streaming, when the sampler emits the start of a
`<tool_call>` block whose decoded JSON has `{"name":"bash"}` and that name
appears in the request's tool_call history ≥ N times (N=3 is the Focused-ReAct
default), Atlas:

1. Suppresses the tool_call open token (logit→-inf on `<tool_call>` start).
2. Forces a `assistant` content prefix: `"I have enough information. I will now write the file. "`
3. Lets the model continue normally.

This is a sampler-level intervention, no client coordination needed.

**Implementation cost:** ~3 days. Atlas's tool-call grammar already knows where
the `<tool_call>` boundary is (XGrammar). We add a per-request counter keyed on
tool name in the prompt-parsing pass (counts *prompt-included* prior tool calls,
which is what opencode replays on every turn). When the threshold trips, force-
emit a fixed content prefix via the sampler's "biased prefix" path (already
exists for the reasoning parser).

**Atlas-only?** YES. Atlas sees the entire chat history on every request
because opencode sends it. The repetition signal is in the prompt; the
intervention is in the sampler. No opencode change required.

**Risk:** False positives on legitimately repeated bash. Mitigation: condition
on *identical arguments* via a Jaccard-on-tokens check, or on consecutive same-
tool calls (not just frequency).

---

## 4. Entropic Activation Steering (EAST)

**Paper:** Schmied et al., ICML 2024 (arXiv 2406.00244), *Controlling Large
Language Model Agents with Entropic Activation Steering*. EAST computes an
entropy-weighted steering vector from logged trajectories and adds it to the
residual stream at one layer to **directly raise/lower action entropy beyond
what sampling temperature can do**. Reported on language-bandits: shifts from
overconfident exploitation to information-seeking.

**Why it relevant (inverted):** Opencode's wandering is *over-exploration*, the
opposite of bandit collapse. EAST works in both directions — compute the
*negative* of the exploration vector and inject it to push the model toward
*committal/exploitative* behavior. The paper explicitly demonstrates entropy
steering in both directions.

**Trigger condition:** Same detector as #2 (coding-agent template + creation
verb). Inject a small (~0.5σ) negative-exploration steering vector at one
mid-layer for the duration of the response. No per-token gating needed —
the vector is just a forward-hook addition.

**Implementation cost:** ~1 week. Three pieces:
1. Collect ~200 logged Atlas trajectories tagged "wandered" vs "wrote files".
   We already have these in `/workspace/atlas-mtp/bench/fp8_dgx2_drift/logs/`.
2. Compute steering vector (entropy-weighted hidden-state mean diff).
3. Add a Rust hook in `src/models/qwen36/forward.rs` that adds `α · v` to the
   residual at layer L. Atlas already supports per-request hidden-state hooks
   for the MTP path — this re-uses that infrastructure.

**Atlas-only?** YES. Entirely a forward-pass intervention, opaque to the client.

**Risk:** Steering vectors are model-specific and quant-sensitive — the FP8
weights mean the vector must be computed on the FP8 model, not the BF16
reference. Also, EAST was validated on bandit toy tasks; transfer to coding
agents is unproven.

---

## 5. Client-Side-Only Approaches (NOT Atlas-deployable)

The following are well-cited but require the **client** (opencode) to run a
secondary planner / verifier loop, so Atlas alone cannot deploy them:

- **AdaPlanner** (Sun et al., NeurIPS 2023, arXiv 2305.16653) — adaptive
  in-plan / out-of-plan refinement requires a closed-loop planner module
  outside the inference call. The planner re-prompts after each environment
  observation. Atlas serves the prompts but cannot orchestrate the loop.
- **ReWOO** (Xu et al., arXiv 2305.18323) — three-module architecture
  (Planner / Worker / Solver) with a separate planning LLM call. Reports
  5× token efficiency, +4% HotpotQA accuracy. **Architecturally a client.**
- **Reflexion** (Shinn et al., NeurIPS 2023, arXiv 2303.11366) — verbal RL
  loop with cross-episode memory. Requires the client to detect failure,
  generate a reflection, and re-run. Atlas is stateless across requests.
- **Task Shield** (Jia et al., ACL 2025, arXiv 2412.16682) — every tool call
  is verified by a *second* LLM call against the user's goal. 2.07% attack
  success on AgentDojo. The second-LLM verifier is a client orchestration
  pattern.
- **MemGPT / Letta** (arXiv 2310.08560) — hierarchical RAM/disk memory with
  function-calling agent. Maintains a separate goal-state slot. **Stateful
  by definition; not implementable in a stateless server.**
- **Voyager** (Wang et al., arXiv 2305.16291) — skill-graph + curriculum.
  Requires environment introspection and a learned skill library; not an
  inference-server concern.
- **Self-RAG** (Asai et al., ICLR 2024) — retrieval interleaved with
  generation; control loop owned by the client.

These should still inform opencode's design, but Atlas cannot ship them alone.

---

## Recommended Atlas Roadmap

| Phase | Ship                                             | Risk | ETA  |
|-------|--------------------------------------------------|------|------|
| P1    | Tool-name DRY n-gram penalty (#1)                | low  | 1d   |
| P2    | Focused-ReAct duplicate-tool stop (#3)           | med  | 3d   |
| P3    | Action-anchor system-prompt injection (#2)       | med  | 2d   |
| P4    | EAST steering vector (#4) — pending logged data  | high | 1wk  |

Phases P1+P2+P3 are **fully server-side, no opencode change, no client config
change**. They can be A/B'd via env vars on dgx2 against the existing
`bench/fp8_dgx2_drift/logs/` "create rust axum project" prompt.

**Acceptance gate:** N≥20 runs per arm, t-test on first-tool-call-is-`edit`
rate (current baseline ≈60% from feedback_no_n1_stochastic_ab.md protocol).
Target: >85% first-tool-is-write rate post-P2.

---

## Specific Answer to the Critical Question

> *Can a stateless LLM inference server inject these interventions, or do they
> require client-side orchestration?*

**Server-side, no client change required:**
1. DRY n-gram penalty on tool-name tokens.
2. Focused-ReAct duplicate-action early-stop (executed in the sampler).
3. Action-anchor system-prompt injection (server-side template hook).
4. EAST entropic activation steering (forward-pass hook).

**Client-side only:** Reflexion, AdaPlanner, ReWOO, Task-Shield, MemGPT,
Voyager, Self-RAG. These are excellent designs and we should advocate for
them in opencode, but they cannot live in Atlas alone.

---

## Sources

- DRY sampler PR: github.com/vllm-project/vllm/pull/11368
- Focused ReAct: arxiv.org/abs/2410.10779
- Plan-and-Solve: arxiv.org/abs/2305.04091
- EAST: arxiv.org/abs/2406.00244
- AdaPlanner: arxiv.org/abs/2305.16653
- ReWOO: arxiv.org/abs/2305.18323
- Reflexion: arxiv.org/abs/2303.11366
- Task Shield: arxiv.org/abs/2412.16682
- MemGPT: arxiv.org/abs/2310.08560
- Voyager: arxiv.org/abs/2305.16291
- Tool-call hacking / mode collapse: arxiv.org/abs/2510.10931
- WESE (weak explore → strong exploit): arxiv.org/abs/2404.07456
- vLLM logits processors: docs.vllm.ai/en/latest/design/logits_processors/
