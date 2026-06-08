# Research 3 — Critic-Verified Tool Calling for Atlas Qwen3.6 FP8

**Question.** Use a small/fast SECONDARY model (or cheaper signal) to verify tool-call arguments emitted by the noisy primary (Qwen3.6-35B-A3B-FP8) before execution.
**Date.** 2026-05-26. **Author.** Claude (research3, no code changes).

---

## 1. Constitutional AI / RLAIF at inference (not training)

Constitutional AI and RLAIF are predominantly **training-time**: a critic LLM produces preference labels for a reward model (Bai 2022, arxiv/2212.08073). The critic does not normally sit in the inference path.

Inference-time exceptions:
- **Reflect** (arxiv/2601.18730) — plug-and-play critic re-reads a draft answer against a written constitution, emits rewrite/pass.
- **Constitution or Collapse?** (arxiv/2504.04918) — small self-critics suffer **model collapse** under recursion. A critic that **re-derives** from the *original context* (not from the primary's emission) sidesteps this.

Production deployments are rare and wrap **safety** not **correctness**: e.g. Singapore's public-service chatbot uses `gemini-2.0-flash-lite` as a centralised guardrail critic (Galileo). Reported cost: +1000 ms per call; hot paths only.

**Verdict.** Constitutional AI doesn't solve our problem, but Reflect-style "critic re-reads context + draft, emits verdict" is the cleanest template.

---

## 2. arXiv 2024–2026 — critic-verified tool calling

Relevant primary literature:

| Paper | arxiv | Method | Result |
|---|---|---|---|
| **CRITIC** (Gou et al.) | 2305.11738 | Primary model self-critiques via tool feedback; iterative. | Improves QA but adds 2–4× tokens. |
| **CRITICTOOL** | 2506.13977 | Benchmark for tool-call self-critique. Categorises 5 error modes: tool-selection, tool-hallucination, param-key, param-value, env. | Shows even GPT-4 misses ~25% of param-value errors. |
| **VerifyLLM** (Grigorev 2025) | survey | Critic uses Linear Temporal Logic to detect missing-prerequisite / redundancy errors in action sequences. | Symbolic, no LLM critic in the loop. |
| **Try, Check and Retry** | 2603.11495 | Divide-and-conquer: a checker LLM verifies each sub-call before retry. | +9 pts on long-context tool benches. |
| **Tool Hallucination Reliability Alignment** | rhythmcao 2025 | Finetune for self-abstention rather than emit a wrong call. | Eliminates 40% of param-value hallucinations without an external critic. |
| **AgentV-RL** | 2604.16004 | Two-stage trains a *verifier* with multi-turn tool-integrated reasoning. | Verifier signal beats outcome reward by ~6 pts. |
| **TraceSafe / AgentDoG / ToolSafe** | 2604.07223, 2601.18491, 2601.10156 | Mid-trajectory safety/correctness guardrails using small critic LLMs in a two-stage cascade (e.g. Qwen3.5-9B → Qwen2.5-14B). | 94–96% interception, 6–10% FP on safety violations. |

**Most directly relevant pattern.** **Re-derivation critic.** Feed the critic the *same tool schema and conversation context*, ask it to emit args independently, and accept iff arguments match (by schema-aware equality or JSON normalisation). Closest paper: **AgentV-RL** uses exactly this "verifier re-derives" pattern; ~6 pt outcome gain at the cost of one full forward pass.

---

## 3. MoE + critic via the primary's own hidden states (no rerun)

This is the most promising line for Atlas because it avoids a second model entirely.

Key 2025 findings:

- **Reasoning Models Know When They're Right** (arxiv/2504.05419, ICLR review). A linear probe on the primary's hidden states predicts answer correctness with up to **90% accuracy** and high calibration. Probe is ~10k params, runs in <0.1 ms. Used at inference to early-exit reasoning (24% token saving) without rerunning the model.
- **Internal Representations as Indicators of Hallucinations in Agent Tool Selection** (arxiv/2601.05214). Linear probe on hidden states catches tool-call hallucinations (wrong tool, malformed params, tool-bypass) at **86.4% accuracy** in the *same forward pass* as generation. Explicit zero-rerun design.
- **ICR Probe** (arxiv/2507.16488, ACL 2025). Aggregates per-layer residual-stream contribution (ICR Score) into a small classifier; SOTA hallucination detector with sub-millisecond overhead and a few-million-param probe.
- **LatentAudit** (arxiv/2604.05358). White-box faithfulness monitor; **0.77 ms** overhead and AUROC 0.942 on Llama-3-8B PubMedQA — i.e. a sub-millisecond linear probe.
- **CLUE** (arxiv/2510.01591). Non-parametric verification: cluster hidden states from past correct/incorrect runs and classify the new one by nearest cluster. Useful when labelled probe data is scarce.
- **Behavioral Steering in 35B MoE** (arxiv/2603.16335). Demonstrates SAE-based probes on **Qwen3.5-35B-A3B** residual stream — the exact same family as Atlas's Qwen3.6-35B-A3B-FP8. Probes generalise across tasks.

**Atlas implication.** We already capture the residual stream during every forward. A trained linear probe on Qwen3.6-A3B's hidden state at the moment of `<tool_call>` opener would give a per-token correctness score *for free* (≤1 ms, no extra model, no extra GPU memory). This is the **cheapest verifier by an order of magnitude**.

Limitation: probes need a labelled dataset of "good vs drifted tool-call hidden states". We have the drift telemetry (`MASTER_DRIFT_TABLE.md`, `op_drift.json`) to bootstrap one. Bootstrap: replay every prompt in `fp8_dgx2_drift/`, label by argument-equality vs HF BF16 reference, train a 1-layer MLP on residuals at the open-tool-call token.

---

## 4. Self-critique loops (primary verifies own output mid-stream)

- **Reflexion** (Shinn 2023). External evaluator prompts primary to grade its trajectory; doubles latency.
- **Self-RAG** (arxiv/2310.11511). Inline reflection tokens (`[Retrieve]`, `[ISREL]`, `[ISSUP]`). One extra token per critique.
- **SRGen** (arxiv/2510.02919). Lightweight test-time reflection that steers away from impending mistakes. No retrain.
- **Double-Checker** (arxiv/2506.21285). Fine-tune for inserted "checking" passes; modest gain, large overhead.
- **Self-Verification Dilemma** (arxiv/2602.03485). Over-trained self-verification *suppresses* first-draft accuracy. Risk for Atlas.

**Pros vs external critic.** Reuses primary weights, no extra VRAM, inline with KV cache.
**Cons.** The same FP8 quant drift that produced the bad tool call also corrupts the self-check (Constitution-or-Collapse). Atlas's drift is *structural* (FP8 MoE-routing dequant — see `project_qwen36_drift_moe_smoking_gun.md`), so the primary cannot detect what its own corrupted state is hiding.

**Verdict.** Self-critique alone is insufficient against FP8 drift. Can be a cheap first stage in a cascade, but a probe or external model must back-stop it.

---

## 5. Latency budget — can we afford a 1.5B critic?

Rough numbers on GB10:

- Qwen2.5-1.5B FP16: ~150 tok/s decode = **6.7 ms/tok**. Prefill at 4k context ≈ 60–100 ms.
- Phi-3.5-mini (3.8B) FP16: ~80 tok/s decode = **12 ms/tok**.
- Tool-call payload: typically 50–200 args tokens.

So a re-derivation critic running 200 output tokens ≈ **1.3–2.4 s overhead** per tool call. The user-quoted 2 s is correct.

**Is 2 s tolerable?**

- **Interactive (Claude Code, opencode):** ~10 tool calls per turn × 2 s = 20 s. Painful.
- **Batched/back-end:** fine.
- **Selective trigger:** if a probe (Section 3) flags only ~10% of calls for re-derivation, average overhead drops to 200 ms — clearly tolerable.

Production data points:

- **DeepEval / TruLens LLM-as-judge**: +1000 ms (galileo.ai). Used in CI/CD, not hot path.
- **Singapore guardrail (gemini-2.0-flash-lite)**: live, but a small flash model on TPU, not 1.5B local.
- **Sherlock** (arxiv/2511.00330). Reliable agentic workflows accept ~10% wall-clock overhead from verification when correctness gain is >5%.
- **SuffixDecoding** (NeurIPS 2025 Spotlight, suffix-decoding.github.io) and **Optimizing Agentic LM Inference via Speculative Tool Calls** (arxiv/2512.15834): speculative tool-call drafts from a cache validated by the primary in a single forward pass. Up to **5.3× speedup** on AgenticSQL. The critic isn't a model — it's the primary's own logits over a drafted suffix. Verification overhead ~zero when drafts hit.

**Verdict.** A *gated* 1.5B critic (probe triggers re-derivation only when suspicious) is tolerable. An *always-on* 1.5B critic is not.

---

## 6. opencode / Cursor / Aider / Claude Code — built-in critic loops?

- **Claude Code.** `PreToolUse`/`PostToolUse` **hooks** with three handler types: shell command, prompt (one-shot LLM eval), agent (subagent with tools). The "prompt" handler is an inline critic. **Task** tool enables orchestrator/sub-agent cascade. **No built-in critic on tool args by default** — user installs hooks.
- **Cursor (v1.7, Oct 2025).** Hooks for `beforeShellExecution`, `beforeMCPExecution`, `beforeReadFile`, `afterFileEdit`, `stop`. Commands only, no native LLM critic. Verification is community-built.
- **Aider.** No critic loop. Relies on unified-diff edit; user reviews. Retry-on-edit-fail but no semantic verifier.
- **opencode.** Routes to 75+ providers; single agent loop. No native critic.

**Common pattern.** Verification, if present, is a user-configured hook running a shell linter or a prompt-handler LLM call. **None ship a default tool-arg critic model.** Open gap in the agent harness market.

---

## Top 5 implementation patterns for Atlas (ranked)

### 1. Linear probe on Qwen3.6 residual stream at the `<tool_call>` opener token

- **Why first.** Zero extra model, ≤1 ms overhead, leverages existing drift telemetry to train. Direct precedent on the same Qwen3.5-A3B family (arxiv/2603.16335, 2601.05214).
- **What.** Train a 1-layer MLP (~10k params) on residual-stream activations at the moment the primary emits `<tool_call>`. Label by "did the args match HF BF16 reference". Run probe inline; flag suspicious calls.
- **Expected.** 80–90% catch rate on drifted calls per the ICR/Internal-Reps numbers. Fail-open or fail-to-critic on flag.
- **Effort.** ~3 days: residual extraction (Atlas already has this), labelling pipeline (replay `fp8_dgx2_drift/`), probe training, runtime hook.
- **Atlas memory.** `op_drift.json` + `MASTER_DRIFT_TABLE.md` already provide a labelled corpus.

### 2. Gated re-derivation critic (Qwen2.5-1.5B FP16) triggered by the probe

- **Why second.** Combines #1's cheapness with a definitive cross-check. Mirrors AgentV-RL pattern, with the probe as a router.
- **What.** When probe fires, run a 1.5B critic with the **same conversation + tool schema** (not the primary's emission). Compare JSON-normalised args. Reject mismatch.
- **Cost.** ~2 s per flagged call × ~10% flag rate ≈ 200 ms average overhead.
- **Risk.** Critic may itself be wrong; need a "two-of-three" tie-break if the critic disagrees with the primary.

### 3. SuffixDecoding-style speculative tool-call cache

- **Why third.** Orthogonal to correctness — primarily a speedup — but the cache is also a *signal*: a tool call whose args don't match any past suffix is suspect. NeurIPS 2025 Spotlight; 5.3× speedup on AgenticSQL; arxiv/2512.15834 for the agentic variant.
- **What.** Cache prior `(tool_name, args)` tuples. Draft from cache; verify against primary logits in one forward pass. Calls that *can't* be drafted from cache (novel suffix) get routed to pattern #1 + #2.
- **Effort.** ~1–2 weeks (suffix tree + sampler integration). Higher than #1 but compounding wins.

### 4. Self-RAG-style inline `[VERIFY]` reflection token

- **Why fourth.** Cheap, reuses the primary, no second model. But blocked by the FP8 drift root cause (the primary cannot reliably self-detect MoE expert mis-routing).
- **What.** Fine-tune primary to emit a `[VERIFY]` token before tool execution; downstream consumer treats `[VERIFY]` as a gate. Works only after the FP8 dequant fix lands.
- **Defer until.** Phase 2b RNE / softmax patches stabilise; reassess once cos drift < 0.99 across layers.

### 5. Schema-constrained re-emission via XGrammar

- **Why fifth.** Not a critic, but a structural guard. Atlas already uses XGrammar (`project_xgrammar.md`). Tighten the tool-call grammar to fail-fast on impossible args (wrong types, out-of-enum values, missing required fields). Combine with CRITICTOOL's 5-error-mode taxonomy to design a fast deterministic filter.
- **Caveat.** Catches *syntactic* but not *semantic* errors. Param-value hallucinations slip through. Pair with #1 to cover semantics.

---

## Synthesis

The cheapest, highest-leverage move is **#1: a linear probe on the residual stream at the tool-call opener**. It is supported by three independent 2025 papers (arxiv/2504.05419, 2601.05214, 2507.16488), one of which used the same Qwen3.5-A3B model family (2603.16335). Latency is sub-millisecond; we already capture the data. **#2 (gated re-derivation)** layers on top as a definitive second opinion for the ~10% of calls the probe flags. **#3 (SuffixDecoding)** is a parallel speedup that doubles as a novelty detector. None of opencode / Cursor / Aider / Claude Code ship this by default — Atlas can lead.

Cost summary:

| Pattern | Latency | Cost | Catches |
|---|---|---|---|
| #1 probe | <1 ms | 1-time train (~3 days) | ~85% drift |
| #2 gated 1.5B critic | 200 ms avg, 2 s on flagged | 1.5B FP16 (~3 GB VRAM) | ~95% combined |
| #3 SuffixDecoding | ~0 (speedup) | suffix tree | novelty signal |
| #4 self-RAG verify | ~1 token | retrain | only after FP8 fix |
| #5 XGrammar tighter | ~0 | grammar edit | syntactic only |

Start with #1+#5 in week 1, add #2 in week 2, layer #3 when bandwidth.
