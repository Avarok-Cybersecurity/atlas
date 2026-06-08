# Multi-Turn Agentic Coherence Loss: arXiv 2024-2026 Literature Survey

**Target**: Atlas (Rust LLM inference) serving Qwen3.6-35B-A3B-FP8 to opencode. **Symptom**: single-pass cosine vs HF BF16 = 0.994, yet after ~5-10 opencode turns the model emits empty tool parameters, corrupted argument JSON, repetition loops, and hallucinated file paths. **Scope**: inference-time (not training) mitigations grounded in the 2024-2026 literature.

---

## 1. What Recent Papers Diagnose As "Multi-Turn Coherence Loss"

### 1.1 Drift as a bounded equilibrium, not a runaway

Boyko et al., **"Drift No More? Context Equilibria in Multi-Turn LLM Interactions"** (arXiv:2510.07777, 2025). Formalises drift as the turn-wise KL divergence between the deployed model and a goal-consistent reference. Empirically the divergence settles into a **noise-limited equilibrium** rather than diverging unboundedly. The intervention that works in their experiments is the simplest one: **periodically re-inject the system prompt / task reminder**. This shifts the equilibrium point lower without retraining.

**Implication for Atlas**: a slow re-statement of the original opencode tool schema every K turns should measurably reduce drift. This is a serving-side concatenation, not a model change.

### 1.2 Behavioural drift and survey of multi-turn failures

**"Agent Drift"** (arXiv:2601.04170) decomposes drift into semantic, coordination, and behavioral drift. The mechanism: the model's prior outputs become future inputs, so tiny stylistic/format biases compound. For coding agents this manifests as "the model copies its own slightly wrong tool-call format from earlier in the conversation." Sun et al., **"Beyond Single-Turn"** survey (arXiv:2504.04717): "Most LLMs suffer significant performance degradation in multi-turn scenarios, with errors compounding over successive exchanges." The system prompt is at the *beginning* of an opencode conversation - by turn 10 it is far away, and recency-biased attention preferentially looks at the recent (often broken) tool calls.

### 1.4 Lost-in-the-middle for tool calling specifically

Kate, Pedapati et al., **"LongFuncEval: Measuring the effectiveness of long context models for function calling"** (arXiv:2505.10570, 2025). The most directly Atlas-relevant numbers in the literature:

| Pressure axis | Reported degradation |
|---|---|
| Tool-catalog size grows | 7-85% drop in correct function call |
| Tool-response length grows | 7-91% drop in answer retrieval |
| Multi-turn dialogue length grows | 13-40% drop |

Position-bias variation across models was 5% (GPT-4o) to 75% (Mistral-large). Recency bias dominates: information later in the context is over-weighted, including the model's own earlier malformed tool calls.

Modarressi et al., **"NoLiMa: Long-Context Evaluation Beyond Literal Matching"** (arXiv:2502.05167, ICML 2025) generalises the "needle in haystack" benchmark to require *latent associative* retrieval. 10 of 12 frontier models fall below 50% of their short-context baseline at 32K tokens. Liu et al.'s **"Found in the Middle"** (arXiv:2406.16008) and **"Lost in the Middle, and In-Between"** (arXiv:2412.10079) confirm the U-shape is positional-attention-bias driven and is partially recoverable with calibration.

### 1.5 Repetition / mode collapse / attractor cycles

**"Unveiling Attractor Cycles in LLMs"** (arXiv:2502.15208, ACL 2025) - successive paraphrasing converges to **2-period attractor cycles**. The repetition loop Atlas observes is the model being drawn toward a low-dimensional fixed point, not random failure. **"Solving LLM Repetition Problem in Production"** (arXiv:2512.04419) gives a Markov-model analysis: greedy decoding cannot escape repetitive loops because each repeat raises the conditional probability of the next. Beam search with early stopping is the cleanest fix; in a streaming sampler the closest analogue is strong rep penalty + low min-p. **Engorgio** (arXiv:2412.19394, ICLR 2025): adversarial prompts can suppress EOS and force 2-13x longer outputs; the same mechanism happens accidentally inside long opencode trajectories where the model has seen many turns ending in tool calls rather than EOS. **LZ Penalty** (arXiv:2504.20131) - Lempel-Ziv-style residual penalty that breaks loops without harming non-degenerate generation; drop-in sampler change.

### 1.6 Self-correction makes things worse

Huang, Chen et al., **"Large Language Models Cannot Self-Correct Reasoning Yet"** (arXiv:2310.01798); **"Self-Correction Bench"** (arXiv:2507.02778) finds a 64.5% self-correction blind-spot rate across 14 open models. Implication: a multi-turn agent that *sees its own bad output and tries to fix it* often degrades further. Combined with sycophancy work (**SycEval**, arXiv:2502.08177; arXiv:2509.21305) this means user pushback ("that didn't work") frequently makes the model abandon a correct path.

### 1.7 Coding-model-specific failure analysis

**Qwen3-Coder-Next Technical Report** (arXiv:2603.00729). Identifies "context hallucination" as the primary multi-turn failure mode for coding agents and introduces **Best-Fit-Packing** at training time to keep trajectories intact. The fact that the *Qwen team itself* names this as the dominant failure mode is significant. They claim coherence over up to 300 turns post-fix.

**Kimi K2** (arXiv:2507.20534) and **DeepSeek-V3.2** (arXiv:2512.02556) frame the trade-off as "ephemeral reasoning" vs "autonomous endurance"; both report error recovery over long sessions is now a primary training objective. **SWE-EVO** (arXiv:2512.18470) taxonomy: Kimi-K2 ~70% incorrect-implementation; Qwen3-Coder & gpt-oss-120b similar; DeepSeek-R1 dominated by **stuck-in-loop** and tool-use failures with early exits. Atlas's symptom (empty params, corrupted args, repetition) sits squarely in the stuck-in-loop + instruction-following cluster.

---

## 2. The FP8 / Quantization Angle - Why 0.994 Cosine Is Not Enough

This is the most important section for Atlas. The literature is now clear that **single-pass cosine similarity is a fundamentally inadequate metric** for predicting multi-turn quality of a quantized model.

### 2.1 Long-context quantization is worse than short-context

Reddy et al., **"Does quantization affect models' performance on long-context tasks?"** (arXiv:2505.20276, EMNLP 2025). FP8 averages ~0.8% drop on short tasks but degradation grows monotonically with context length. 4-bit can fall by up to 59% at 128K. The key insight: **per-token quantization error is small, but it accumulates and interacts with positional bias.** A 0.994 cosine at turn 1 is consistent with sub-1% per-pass error - and after 10 turns of error compounding through KV cache and self-attention, the effective error is well above the threshold for stable tool-call grammar.

### 2.2 Activation outliers carry agent-critical signal

**"Mitigating the Impact of Outlier Channels"** (arXiv:2404.03605); **"Activation Outliers in Transformer Quantization"** (arXiv:2603.04308); **"TWEO: Transformers Without Extreme Outliers"** (arXiv:2511.23225). Zeroing only 0.1% of outliers causes 600-1000% perplexity spike. The outliers are **not noise** - they encode fine-grained contextual cues used for retrieval and instruction-following. FP8's clip-vs-round trade-off attacks exactly these channels. Cosine similarity averaged over all hidden dimensions is dominated by the bulk; the agent-critical signal lives in the tail.

### 2.3 MoE routing is uniquely fragile to FP8

Yang, Liu et al., **"EAQuant: Enhancing Post-Training Quantization for MoE Models via Expert-Aware Optimization"** (arXiv:2506.13329, 2025). Two failure modes specific to quantized MoE:

1. **Structural instability**: tiny logit perturbations flip top-k expert selection. Once a wrong expert is chosen, an entirely different sub-network processes the token.
2. **Dynamic workload imbalance**: power-law expert usage means a handful of "core" experts handle most traffic; quantization noise pushes borderline tokens to underutilized "niche" experts that were never calibrated as a primary router target.

This matches the Atlas memory finding (`project_qwen36_drift_moe_smoking_gun.md`): expert routing diverges 8/8 -> 7/8 -> 3/8 from layer 0 to 38. FP8 routed-expert dequant is the prime suspect, and EAQuant gives the mechanistic story.

### 2.4 FP8 KV-cache on Qwen3 MoE specifically

vLLM blog "The State of FP8 KV-Cache and Attention Quantization in vLLM" (2026-04-22): on Qwen3-30B-A3B-Instruct-2507 with FP8 KV + FP8 attention at 256K context, recovery is 94-98% of BF16 baseline. The longest context buckets show the widest gap. For MLA models (GLM-Flash), FP8 KV is reported as functionally broken in multi-turn because per-tensor scaling compounds.

### 2.5 Consensus root cause for coding MoE models

Synthesising 1 and 2: three compounding sources that each look small in isolation. (a) Positional / recency attention bias (LongFuncEval, NoLiMa) - system prompt and tool schema receive less attention each turn. (b) Autoregressive feedback (Agent Drift, Attractor Cycles) - earlier malformed tool calls become templates. (c) Quantization-induced routing instability (EAQuant) - each forward pass is a slightly different network than BF16, with the difference biased toward outlier dimensions that carry agent-relevant signal. Single-pass cosine measures none of these - they are all *temporal compounding* effects.

---

## 3. Inference-Time Mitigations With The Strongest Evidence

Ranked by empirical evidence × Atlas implementability.

| Rank | Intervention | Paper | Implementable in Atlas? |
|---|---|---|---|
| 1 | Periodic system-prompt / tool-schema re-injection | arXiv:2510.07777 | Yes - serving-side context surgery |
| 2 | LZ Penalty replacing standard rep penalty | arXiv:2504.20131 | Yes - sampler change, drop-in |
| 3 | Selective CFG at high-entropy positions (tool-call openers, arg names) | arXiv:2510.13940 | Yes - already have XGrammar mask hook |
| 4 | Constrained decoding extended to argument-level schema (not just opener) | arXiv:2411.15100, arXiv:2601.04426 (XGrammar-2) | Yes - upgrade existing XGrammar |
| 5 | EAGLE-3 style speculative decode with tool-call validation | arXiv:2401.15077, arXiv:2512.15834 | Yes - Atlas already has MTP infra |
| 6 | Min-p / top-nσ replacing top-p at the sampler | arXiv:2407.01082, arXiv:2411.07641 | Yes - sampler change |
| 7 | Prefix attention sink retention across turns | arXiv:2309.17453, arXiv:2410.10781 | Partial - KV cache management |
| 8 | PRM-based candidate re-ranking at tool-call boundaries | arXiv:2511.08325 AgentPRM | Heavy - needs PRM model |

### 3.1 Why these in particular

- **#1 (Drift No More)** is the cheapest experiment with the most direct evidence and zero new kernels.
- **#2 (LZ Penalty)** directly attacks the attractor-cycle / stuck-in-loop failure SWE-EVO identifies for Atlas's symptom class. Replaces an existing sampler component.
- **#3 (Selective CFG)** is critical because the unconstrained Atlas tail behaviour is what produces empty params / wrong arg names. CFG only at uncertain positions avoids the 2x decode cost.
- **#4 (XGrammar-2)** extends the existing grammar enforcement Atlas already runs (per `project_xgrammar.md` and `project_grammar_bytelevel_vocab.md`) from "open the tool call correctly" to "all required arguments are emitted with valid values."
- **#5 (Speculative tool calls)** matches Atlas's existing MTP infra (`project_mtp_verify_status.md`, `project_pass16_wy_investigation.md`). Tool-call validation in a single forward pass means rejected drafts never enter conversation history.

### 3.2 What is *not* a good fit

- **Beam search** (arXiv:2512.04419) is effective but breaks the streaming token contract opencode expects.
- **Self-correction loops** (arXiv:2310.01798, arXiv:2507.02778) actively make things worse on borderline-broken outputs.
- **Re-quantizing weights at a different bit width** at inference time is not a serving-time intervention.
- **Classifier-free guidance at every position** doubles decode cost; only the *selective* form (arXiv:2510.13940) is viable.

---

## 4. Direct Answers To The Posed Questions

**Q: Consensus root cause for code-tuned models?**
Three compounding error sources: positional/recency attention bias, autoregressive feedback of the model's own earlier malformed outputs, and quantization-induced MoE routing instability. The Qwen3-Coder-Next, Kimi K2, DeepSeek-V3.2, and SWE-EVO reports all converge on "context hallucination" / "stuck-in-loop" as the dominant multi-turn failure mode for code-tuned MoE models in 2025-2026.

**Q: Best inference-time intervention with strongest evidence?**
Periodic system-prompt/tool-schema re-injection (arXiv:2510.07777) plus selective grammar enforcement extended to argument-level schemas (XGrammar-2, arXiv:2601.04426). The reminder intervention has the cleanest empirical signal in the literature; selective grammar enforcement has direct precedent in production inference systems.

**Q: Does FP8 interact with multi-turn coherence in non-obvious ways?**
Yes. The literature is explicit: single-pass cosine ≥ 0.99 is consistent with multi-turn failure because (a) per-token error compounds across the autoregressive feedback loop (arXiv:2505.20276), (b) outlier channels carrying agent-critical signal are exactly the channels FP8 clips (arXiv:2404.03605, arXiv:2603.04308), and (c) MoE routing top-k selection is bistable under tiny logit perturbations (arXiv:2506.13329 EAQuant). Atlas's measured 8/8 -> 3/8 expert agreement at deep layers (memory: `project_qwen36_drift_moe_smoking_gun.md`) is the exact mechanism EAQuant describes.

---

## 5. Bibliography

Multi-turn / drift: 2307.03172, 2406.16008, 2412.10070, 2502.05167, 2504.04717, 2505.10570, 2510.07777, 2601.04170. Repetition / mode collapse: 2412.19394, 2502.15208 (ACL 2025), 2504.20131, 2512.04419. Self-correction & sycophancy: 2310.01798, 2502.08177, 2507.02778, 2509.21305. Quantization & FP8: 2404.03605, 2505.20276 (EMNLP 2025), 2506.13329 (EAQuant), 2511.23225, 2603.04308; plus vLLM blog (2026-04-22). Coding-model tech reports: 2507.20534 (Kimi K2), 2512.02556 (DeepSeek-V3.2), 2603.00729 (Qwen3-Coder-Next), 2512.18470 (SWE-EVO). Inference-time interventions: 2309.17453, 2401.15077, 2407.01082, 2411.07641, 2411.15100 (XGrammar), 2510.13940 (selective CFG), 2511.08325 (AgentPRM), 2512.15834 (speculative tool calls), 2601.04426 (XGrammar-2).
