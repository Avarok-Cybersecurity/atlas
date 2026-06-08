# Inference Bible — Multi-Turn Agentic Coherence Research for Qwen3.6-35B-A3B-FP8 on Atlas

**Source**: *Inference Engineering* by Philip Kiely (Baseten, 2026), accessed via local RAG (`/workspace/inference_bible/`, Qdrant collection `inference_bible`).
**Index health**: 4 collections live on `10.10.10.2:6333` (`inference_bible` present and serving). Hybrid search (dense `nv-embedqa-1b-v2` 4096d + BM25 + Jina reranker) confirmed via 13 successful queries.
**Cross-ref**: `/workspace/.claude/projects/-workspace/memory/project_qwen36_*.md` (drift root cause, phase 2b, post-think EOS).

---

## Query log and key citations

| # | Query | Best chapter hit | Page(s) |
|---|---|---|---|
| 1 | tool calling / guided decoding | Ch. 4 (vLLM, profiling) | 102, 108, 116 |
| 2 | constrained decoding grammar JSON | Ch. 2 / Ch. 5 / App. A | 51, 137, 227 |
| 3 | multi-turn / chat template / stateful | Ch. 5 (prefix cache) / App. A | 50, 139, 214 |
| 4 | long-context degradation | Ch. 5.3.4 / Ch. 5.1.3 | 141, 128, 130 |
| 5 | FP8 KV compounding | Ch. 5.1.2 / 5.1.3 | 127, 129, 130 |
| 6 | MoE FP8 dequant routing | Ch. 5.4 | 145–148 |
| 7 | spec decode rollback | Ch. 5.2.1, 5.2.4 | 131–137 |
| 8 | production observability | Ch. 7.4.3, 7.5 | 203–206 |
| 9 | prefix cache chat-template corruption | Ch. 5.3.1, Ch. 2.2 | 50, 139, 140 |
| 10 | quality eval / regression gates | Ch. 1.3, Ch. 4.5 | 32–35, 114 |
| 11 | structured output / FSM | App. A, Ch. 2.2.1 | 49, 227 |
| 12 | cold-start / scale-to-zero | Ch. 7.2.2, 7.2.4 | 189–192 |
| 13 | FP8 microscaling MXFP8 | Ch. 5.1.1 | 123–126 |

---

## Findings mapped to Atlas's known drift signature

### F1 — FP8 KV "compounds from token to token" (Ch. 5.1.2, p. 127–129)

> *"The KV cache for each token is used by each subsequent token. This means precision errors introduced by quantization can compound from token to token… Compounding errors is exactly the reason why attention layers are the riskiest to quantize."*

This is the textbook description of Atlas's Phase-2b iter-2 finding (`project_qwen36_phase2b_softmax_expf.md`): once the softmax polynomial was replaced with `__expf`, deep-layer attention layers (L31–L39) **regressed** from 0.946 → 0.929 cos because the polynomial had been masking FP8 KV drift on large K/V magnitudes. The bible explicitly warns that this compounding is the dominant failure mode for FP8 KV at long context, and that **attention is the riskiest quantization target precisely because each step depends on every prior step**. Multi-turn agentic transcripts are the worst case: each tool round adds 1–5k tokens to the cache, and by turn 5–8 the drift accumulated over ~20k cached KV positions hits the same regime that already destroys L31–L39 single-shot.

### F2 — "Use FP8 with high dynamic range; *microscaling* MXFP8 if possible" (Ch. 5.1.3, p. 128 + Ch. 5.1.1, p. 124–126)

> *"A moderate approach to low-precision inference uses a format like FP8 with high dynamic range — if possible, a microscaling format like MXFP8 — to carefully quantize select linear layers, activations, and often KV cache values."*

The book lists MXFP8 (Blackwell, 2024) and NVFP4 (Blackwell proprietary, block=16 + global FP32 scale) as the production-grade formats. Atlas's current FP8 KV is **tensor-scale**, not micro-block. Memory `project_qwen36_phase2b_softmax_expf.md` already noted that NVFP4 KV is *best at deep layers, worst early* and FP8 KV is *best early, worst deep*. The bible's prescription is the missing third option: **per-block FP8 (MXFP8-style) KV** would combine FP8's early-layer headroom with NVFP4's late-layer locality. This is a concrete Atlas roadmap item, not a research detour.

### F3 — MoE expert routing is *not* the place to push FP8 (Ch. 5.1.2, p. 127 + Ch. 5.4, p. 145–148)

The bible's quantization hierarchy is **"low risk for weights and activations, moderate for KV cache, high for attention"** — and crucially singles out compounding-sensitive paths. Atlas's smoking-gun memo (`project_qwen36_drift_moe_smoking_gun.md`) shows gate logits going indecisive at L38 (8/8 → 7/8 → 3/8 expert overlap with HF BF16) **because** small FP8 dequant drift flips routing decisions. The bible doesn't single out gate projections by name, but its rule "attention layers are riskiest because of compounding" applies verbatim to MoE routers: the router's output is *consumed by the routing decision*, which then determines **which experts run at all**. A 0.01-magnitude routing-logit drift that would be invisible in a dense MLP becomes an *expert-set divergence* in MoE. **Action**: keep gate proj in BF16 (already SOTA in vLLM/SGLang for FP8 MoE).

### F4 — Chat-template implementation is "step zero" and "differs subtly from model to model" (Ch. 2.2, p. 48–50)

> *"This is handled by the chat template, which differs subtly from model to model and must be implemented correctly in the inference engine."*

Atlas just shipped (2026-05-25) the `POST_THINK_MIN_CONTENT=16` EOS-suppression fix, which is a textbook example of chat-template/template-state bugs that the bible flags as **step zero** for inference. The fix scoped a watchdog guard to `require_tool_call`, exempting genuine short answers. Multi-turn agentic transcripts amplify any chat-template bug because each turn re-tokenizes the entire conversation, including all `<|im_start|>`/`</think>`/`<tool_call>` markers; one off-by-one token causes the model to emit `assistantassistant` artefacts (matches the symptom table in `project_qwen36_fp8_post_think_eos.md`).

### F5 — Prefix caching across turns demands *exact* token equality (Ch. 5.3.1, p. 137–140)

> *"Prefixes end at the first unique token… your context engineering determines your TTFT savings."*

For multi-turn agentic flows the cached KV is the **single largest correctness risk** if the new turn's tokenization or chat-template rendering diverges by even one token from the cached prefix. Any silent re-tokenization (BPE merges differing between turns, special tokens injected by a watchdog, tool-result envelopes added in one turn but not cached) yields a **cache hit on a corrupted prefix** — exactly the failure mode the bible warns about. Atlas's existing Wave-5 vision-prefix-cache contamination fix (`project_bug_sweep_wave5_2026_04_22.md`) is the analogue for the agentic path: tool-call boundaries and `</think>` markers must be guaranteed byte-identical between cached prefill and the next turn's prompt.

### F6 — Speculative decode draft acceptance falls off with depth (Ch. 5.2.1, p. 131–133)

> *"Token acceptance rate is high early in the draft sequence, but draft tokens get less reliable deeper in the sequence."*

Atlas memory `project_qwen36_drift_moe_smoking_gun.md` already noted MTP K2 verify can resurrect drift when the verifier accepts a sequence that the dense forward would not have generated. The bible's prescription — **"aim for short, high-percentage sequences"** — is directly applicable: under multi-turn agentic load, capping MTP draft length to K=1 (or disabling spec-decode entirely after `</think>` boundary tokens) is safer than running K=2 with low-acceptance drafts that pollute KV cache.

### F7 — Quality evals must be the gate, not throughput (Ch. 1.3.1, p. 32–34 + Ch. 4.5.2, p. 114)

> *"Eval datasets serve two purposes: acting as a set of varied and realistic inputs, and spot checking that performance optimizations haven't impacted model output quality."* + *"Establish a baseline: Some performance optimization techniques risk model quality."*

Atlas's `feedback_no_n1_stochastic_ab.md` rule (N≥10 statistical gate harness in `bench/reasoning_eval.py`) is precisely the bible-mandated practice. The multi-turn agentic coherence failure is **not visible** in single-turn coherence tests (memory: 13/14 prompts clean at Phase-1, but the failure mode requires 5–8 tool rounds). The eval set must include **multi-turn agentic transcripts** — not just N≥10 single-shot prompts. HumanEval-style code-completion runs are insufficient; a tool-using opencode session replay is the right shape.

### F8 — Observability surfaces causality, not just symptoms (Ch. 7.5, p. 204–206)

> *"These metrics are interdependent. A spike in latency could come from request volume, but it could also come from long input sequences. Seeing these metrics together lets inference engineers understand not only what is happening but also why."*

Atlas's `feedback_useful_logs.md` already encodes "log request summaries, tool names, timing; not per-token DECODE spam." The bible's Ch. 7.5 prescription extends this: at minimum, log **per-turn token count, KV cache hit rate, MTP acceptance rate, sampler temperature, post-think guard state, and watchdog firings**. A multi-turn coherence regression in production should be reconstructable from logs alone without re-running.

### F9 — Long-context handling needs cache-aware routing, not just bigger KV (Ch. 5.3.4, p. 141–143)

> *"Cache-aware routing allocates traffic based on KV cache rather than simply dividing requests evenly across replicas."*

For Atlas's two-DGX deployment, multi-turn agentic conversations should be pinned to the same node to maximize KV hit rate. Without this, every tool round potentially re-prefills 5–20k tokens with FP8 KV compounding drift starting from scratch — making each subsequent turn *worse* than the prior one. This is a deployment-pattern fix, not a kernel fix.

### F10 — Cold-start affects stateful inference quality (Ch. 7.2.2, p. 189–191)

Cold-start spin-up of a new replica drops all prefix caches; under autoscale-up during a long agentic session, requests can be silently routed to a cold replica that *re-prefills the entire history* with potentially different rounding behavior (e.g., MoE expert ID floor() vs. round-half-even on a freshly-warmed CUDA graph). The bible recommends `scale-to-zero` only for stateless workloads; **stateful agentic inference should disable scale-to-zero by default** and use warm-pool/active-active (Ch. 7.3.4, p. 198–200).

---

## Atlas-specific implications

The bible's framework is consistent with Atlas's evidence: the multi-turn coherence failure is **not one bug** but the *compounding* of three already-identified single-shot drifts (FP8 KV at L31–L39, MoE routing at L38, post-think watchdog) over 5–8 tool rounds. The bible's guidance moves Atlas toward:

1. **MXFP8 (block-scaled FP8) KV cache** as a successor to tensor-scale FP8 — partially mitigates F1.
2. **BF16-resident MoE gate projections** even when experts stay FP8 — addresses F3.
3. **Multi-turn coherence in the N≥10 eval gate harness** — addresses F7.
4. **Cache-aware routing across DGX-1/DGX-2** for opencode sessions — addresses F9.
5. **Conservative MTP (K=1 across turn boundaries)** — addresses F6.

None of these contradict Atlas memory or current alpha-2.35 SOTA; they extend the existing roadmap with book-grounded justification.

---

**Word count**: ~1450
**File**: `/workspace/atlas-mtp/bench/fp8_dgx2_drift/research_inference_bible.md`
