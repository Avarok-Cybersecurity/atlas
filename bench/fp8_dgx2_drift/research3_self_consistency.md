# Self-Consistency / N-Sample Voting Decoders for Code-Gen and Tool Calling
**Date**: 2026-05-26
**Context**: Qwen3.6-35B-A3B-FP8 from opencode at T=0.3 exhibits per-token "character drift" (one-byte path drift, phantom tool args, paraphrase loops, premature `<|im_end|>`). Hypothesis: a noisy T=0.3 distribution can be coaxed into the correct mode by drawing K samples and consensus-voting.

---

## 1. arXiv landscape (2023–2026)

### 1.1 Classical SC (Wang 2022) and Optimal-SC (Blend-ASC, arXiv:2511.12309, Nov 2025)
Sample K paths at T>0; majority-vote. Brittle for free-form (exact match fails). Optimal-SC validates **power-law scaling in K** with a saturating tail; dynamic-allocation uses **6.8× fewer samples** than fixed-K. K=3–5 captures most SC gain on reasoning; K>20 is flat. ([2511.12309](https://arxiv.org/abs/2511.12309))

### 1.2 Universal Self-Consistency (USC) — arXiv:2311.17311
LLM judges its own K outputs, picks "most consistent." Extends SC to free-form (code, summarization, QA). **Matches execution-based voting on code-gen without running the code**, K=8–20. ≈+10% inference time. ([2311.17311](https://arxiv.org/abs/2311.17311))

### 1.3 Soft Self-Consistency (SOFT-SC) — arXiv:2402.13212 — **directly relevant**
Replaces discrete majority with a **continuous likelihood-weighted score** so it works when no two candidates match exactly (large action spaces — i.e. agents and tool calls). Tested on Bash writing (+1.3pp), WebShop (+6.6pp), ALFWorld (+4.7pp). Reaches SC parity at **~half the samples**. ([2402.13212](https://arxiv.org/abs/2402.13212))

### 1.4 Functional Majority Voting (FMV) — arXiv:2604.15618 (2026)
Execution-signature voting: run each candidate on test inputs, score by output-agreement. Pure string match — no AST canonicalization. Qwen3-4B-Thinking: 37.7%→52.7% on LiveCodeBench-v6 at K=64; >40% at K=32; knee at K≈16. ([2604.15618](https://arxiv.org/html/2604.15618))

### 1.5 Inter-Rollout Action Agreement / TrACE — arXiv:2604.08369 (2026)
**Adaptive K**: sample K candidate actions, commit if threshold agree; otherwise sample more. No learned verifier. TrACE-4 matches SC-4 with **33% fewer calls** on GSM8K, 39% fewer on MiniHouse. TrACE-8 matches SC-8 with 55/65% fewer. **The agentic blueprint opencode wants.** ([2604.08369](https://arxiv.org/abs/2604.08369))

### 1.6 Confidence- and semantic-based voting (2025)
Self-Certainty ([2502.18581](https://arxiv.org/pdf/2502.18581)) and Confidence-SC ([2502.06233](https://arxiv.org/pdf/2502.06233)) replace counts with per-token log-prob aggregation — server-side ready because we already have logprobs. Best-of-Majority ([2510.03199](https://arxiv.org/pdf/2510.03199)) restricts to high-frequency answers then top-k — minimax-optimal for pass@k. Semantic Voting ([2509.23067](https://arxiv.org/pdf/2509.23067)) embeds candidates and votes by cosine — escapes the string-match cliff without an LLM judge.

### 1.7 CodeT — arXiv:2207.10397
Generate K solutions + K' tests; pick solutions whose execution agrees with peers on most tests. Codex pass@1 +18.8pp on HumanEval. Benchmark for verifier-grounded voting. No direct tool-call analog (no sandbox).

### 1.8 Quantization × self-consistency
**Gap**: no paper claims SC as remediation for quantization noise. Marlin/AWQ work ([2411.02355](https://arxiv.org/pdf/2411.02355)) attacks noise at the source. The user's intuition is right and unexplored: a slightly-noisy FP8 distribution is exactly the regime where draws around a still-correct mode help.

---

## 2. Production inference engines — what's actually server-side

| Engine | `n>1` support | Server-side voting? | Notes |
|---|---|---|---|
| **vLLM** | `SamplingParams.n`, `SamplingParams.best_of` (deprecated for some paths) | **No consensus vote**. `best_of` picks by cumulative log-prob; that is *not* SC. The caller reconciles. | Copy-on-write KV in PagedAttention makes n>1 cheap for shared prefix — the parent KV blocks are shared until the candidates diverge. This is the key cost lever. ([docs.vllm.ai](https://docs.vllm.ai/en/stable/api/vllm/sampling_params.html)) |
| **SGLang** | `n>1`, parallel sampling via `fork()` in SGL programs | Has rerank model integration but no built-in majority/SC. RadixAttention shares prefix KV. | Frontend exposes a `select()` primitive; the user supplies the rerank/vote logic. ([docs.sglang.ai](https://docs.sglang.ai/supported_models/rerank_models.html)) |
| **TensorRT-LLM** | `beam_width`, `num_return_sequences` | No SC. Beam search is the only built-in selection. | Worst fit for our use-case. |
| **TGI** | `best_of` | Same — log-prob best-of, not SC. | |
| **MLC-LLM** | not exposed in serve path | None. | |
| **Atlas** | currently n=1 in the request path | None. We would be building this from scratch. | |

**Bottom line**: **no production engine ships server-side majority/USC/SOFT-SC out of the box.** They all give you the K samples cheaply (shared prefix KV) and expect the application to vote. This is good news — implementing the vote at the Atlas-server layer is the standard approach, not a hack.

---

## 3. Tool-call voting — specific patterns

Tool-call action space is structured but small per call: tool-name (~10–50 choices), arg-keys (fixed by schema), arg-values (free-form). Friendlier to voting than prose, harsher than multiple-choice.

Schemes:
1. **Tool-name plurality first, then per-arg field-by-field** (Soft-SC for WebShop/ALFWorld): tool-name = hard mode-vote; on tie, escalate. After tool agreement, vote each arg slot independently post-canonicalization.
2. **Field-level normalization before counting** (DSPy pattern): lower-case, trim, sort keys, normalize numbers, resolve relative paths. Directly addresses F16(c) one-byte hyphen drift — a path-normalizer collapses `foo-bar` and `foobar` into the modal answer.
3. **Semantic voting on free-form arg values** ([2509.23067](https://arxiv.org/pdf/2509.23067)): embed, cluster, pick centroid. Overkill for paths; right for commit messages or search queries.
4. **AST/JSON-AST voting for code snippets**: no production tool-call AST vote in the literature. FMV's execution-signature is the closest — irrelevant here.

NLT ([2510.14453](https://arxiv.org/abs/2510.14453), Oct 2025) is **complementary**: converts tool-calling into natural-language sub-task. SC on top is additive.

---

## 4. Cost / quality knee for K

Synthesized from CodeT, FMV, Optimal-SC, Soft-SC, TrACE:

| K | pass@1 lift over greedy | Cost (shared-prefix KV) | When |
|---|---|---|---|
| 2 | +1–3pp | ≈1.4–1.7× | Cheap insurance for high-stakes calls |
| **3** | +3–6pp | **≈2.0–2.4×** | **Sweet spot, latency-bounded interactive** |
| **5** | +6–10pp | ≈3.0–3.5× | **Sweet spot, code-gen / tool selection** |
| 8 | +8–12pp | ≈4.5× | Code-gen; reasoning saturates |
| 16 | +10–14pp | ≈8× | Offline eval only |
| 32–64 | flat | ≥15× | Research/benchmark |

Atlas math at ~25 tok/s decode, batch=1, shared-prefix KV: a 200-token answer at K=3 ≈ 8s (vs 8s greedy) — almost free under our batch capacity. K=5 ≈ 13s (+60% wall-clock) — reasonable for `Edit`/`Bash` calls. K=8 hits the per-step batch ceiling. The first 3–5 samples are effectively free GPU on today's single-stream decode.

---

## 5. Voting with grammar constraints

Two empirical findings:
1. **Constrained decoding does NOT remove the need for sampling diversity.** Grammar guarantees syntactic validity, not semantic correctness. The XGrammar paper (arXiv:2411.15100) and Draft-Conditioned CD (arXiv:2603.03305) both note this: DCCD specifically reports +24pp strict-structured accuracy by **best-of-K over the grammar-constrained sampler** — direct evidence that K-sampling on top of constraints is additive.
2. **K samples under the same grammar are more correlated** than unconstrained samples (the constraint kills many divergent paths). Practical implication: bump T slightly higher (T=0.5–0.7) when sampling K under XGrammar than you would unconstrained (T=0.3). This re-injects the diversity the grammar removes. Soft-SC and TrACE both note that the temperature optimum for SC sits above the greedy optimum.
3. **Constraints make voting easier**: the candidate set is already in a canonical form (well-formed JSON, valid tool-name from the union grammar), so field-level counting works without much normalization.

---

## Top-5 ranked CONCRETE interventions

Ranked by `expected_impact / engineering_cost`, given the F1–F21 baseline and the Atlas server architecture (single-stream decode today, no n>1 wiring).

### #1 — Server-side K=3 Soft-SC vote on tool-name + per-arg field-vote, gated to tool-call turns only (ship)
- Sample K=3 at T=0.7 (raised from 0.3 to recover diversity under XGrammar grammar), shared-prefix KV, score by Soft-SC: tool-name plurality wins; if tied, fall back to highest cumulative log-prob.
- Per-arg: normalize paths (resolve `..`, drop trailing `/`, lower-case file extensions), then plurality-vote each JSON field independently.
- Why #1: directly addresses F16(c) one-byte hyphen drift and F4 MTP verify corruption, **without** needing any precision work (F12 stays open). Reuses the existing XGrammar pipeline.
- Cost: ~2–3 days. Wall-clock per tool-call turn: 1.6×–2.0× (acceptable in opencode flows).
- Failure mode: if all K samples agree on the same wrong answer (correlated noise from FP8 KV calibration F5), SC adds nothing. So this stacks with F5 fix — do them together.

### #2 — Adaptive K via inter-rollout agreement (TrACE-style), default K=2 escalating to K=5 on disagreement
- Always draw 2; if they agree on tool-name AND on every arg slot (after normalization), commit immediately. If they disagree, draw 3 more, vote across all 5.
- Why #2: TrACE-style adaptive compute gives most of the SC gain at ~half the cost. Critically, the easy turns (≥80% of opencode interactions) pay only the K=2 tax.
- Cost: ~3–4 days (the controller + per-arg disagreement detector + escalation path).
- Pairs naturally with #1 — implement #1 first as the "K=5 voter", then add #2 as the gating front-end.

### #3 — Logprob-weighted (Soft-SC) tie-break for the MTP K=2 verify path (F4 follow-on)
- The MTP verify path already produces 2 candidate next-tokens. Today it returns post-mask argmax. Replace with Soft-SC scoring across the K=2 verify hypotheses: weight by token log-prob × draft acceptance prob, return argmax of the weighted score. Adds **zero** new sampling cost (MTP already drafts K).
- Why #3: cheapest win, ~2 hr code change, plugs directly into F4. No new request path, no template change.
- Bounded by MTP K — won't help beyond K=2, but is essentially free.

### #4 — USC-style "self-judge" final pass for ambiguous tool calls (escalation only)
- After #1/#2 vote, if the leading tool-name has <60% of K votes, run a single judge call: "Given these K candidate tool calls, which is most consistent with the user's request?" using the same Qwen3.6 model.
- Why #4: USC matches execution-based voting on code-gen without an execution sandbox. For tool calls this is the analog of CodeT but without needing a sandbox. Adds ~1 extra LLM call per ambiguous turn (most turns won't trigger).
- Cost: ~2–3 days, including a careful judge prompt. Risk: judge introduces its own bias; gate to <10% of turns via the agreement threshold.

### #5 — Temperature recalibration + diversity injection under grammar
- Today opencode runs T=0.3 unconstrained-style. With XGrammar **and** K-sampling enabled, switch the per-token sampling to T=0.6–0.7 inside the grammar mask only. Outside the mask (free-form prose between tool calls) keep T=0.3.
- Why #5: grammar collapses the candidate set; T=0.3 + grammar is over-greedy and produces correlated K-samples (defeats the vote). DCCD and Soft-SC both explicitly show this. Independent of #1–#4: do this even if voting isn't shipped.
- Cost: <1 day. Pure config + sampler hook. Lowest risk.

---

## Risks and non-goals

- **Don't ship K>5 in interactive opencode flows.** The throughput math says we go GPU-bound at K=8 and pass the user's interactivity threshold (>15s per tool call).
- **Don't ship voting before F2, F3, F5 land.** Voting around a broken grammar (`[^<]*`) or a recalibrating-during-decode FP8 KV (F5) will just give K consistent wrong answers — correlated noise is the failure mode SC cannot fix. Soft-SC's "needs diversity" requirement is real.
- **Don't replace MTP with vanilla K-sampling.** MTP gives draft acceptance >50% on greedy paths; the verify-path Soft-SC tie-break (#3) keeps MTP and adds robustness. A standalone K=5 sampler would discard MTP's speedup.
- **No production engine has this.** That's the opportunity, not the blocker — server-side SC for tool calls is a legitimately novel Atlas surface area, especially combined with NVFP4 / FP8 economics on GB10.

---

## Sources

- Wang & Prasad et al., **Soft Self-Consistency Improves Language Model Agents**, [arXiv:2402.13212](https://arxiv.org/abs/2402.13212)
- Chen et al., **Universal Self-Consistency for Large Language Model Generation**, [arXiv:2311.17311](https://arxiv.org/abs/2311.17311)
- **Optimal Self-Consistency (Blend-ASC)**, [arXiv:2511.12309](https://arxiv.org/abs/2511.12309)
- **Functional Majority Voting for Code Generation**, [arXiv:2604.15618](https://arxiv.org/html/2604.15618)
- Sethi, **Don't Overthink It: Inter-Rollout Action Agreement (TrACE)**, [arXiv:2604.08369](https://arxiv.org/abs/2604.08369)
- **Self-Certainty for Best-of-N**, [arXiv:2502.18581](https://arxiv.org/pdf/2502.18581)
- **Confidence Improves Self-Consistency**, [arXiv:2502.06233](https://arxiv.org/pdf/2502.06233)
- **Best-of-Majority (pass@k)**, [arXiv:2510.03199](https://arxiv.org/pdf/2510.03199)
- **Semantic Voting**, [arXiv:2509.23067](https://arxiv.org/pdf/2509.23067)
- **CodeT: Code Generation with Generated Tests**, [arXiv:2207.10397](https://arxiv.org/pdf/2207.10397)
- **Natural Language Tools (NLT)**, [arXiv:2510.14453](https://arxiv.org/abs/2510.14453)
- **XGrammar**, [arXiv:2411.15100](https://arxiv.org/pdf/2411.15100)
- **Draft-Conditioned Constrained Decoding**, [arXiv:2603.03305](https://arxiv.org/pdf/2603.03305)
- vLLM SamplingParams docs, [docs.vllm.ai](https://docs.vllm.ai/en/stable/api/vllm/sampling_params.html)
- SGLang Rerank docs, [docs.sglang.ai](https://docs.sglang.ai/supported_models/rerank_models.html)
