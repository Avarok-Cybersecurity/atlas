# Research 3: Constrained Beam Search for Atlas FP8 Tool-Arg Drift

**Date:** 2026-05-26
**Author:** Claude (research agent)
**Premise:** Qwen3.6-35B-A3B FP8 emits **syntactically valid** tool-call JSON (XGrammar mask enforces shape), but **semantically wrong** values inside the constrained span (e.g. plausible `filePath` syntax, wrong value). Greedy + grammar mask explores **one** candidate trajectory. Constrained beam search explores **K** trajectories, all valid per grammar, and selects the highest-likelihood one at termination. Could this close the FP8 vs BF16 drift gap on tool-call argument fields?

---

## 1. XGrammar — native beam search support

**Status: no native beam search loop, but the primitives exist.** The XGrammar README, blog, and Catalyst page document only single-stream `accept_token` / `fill_next_token_bitmask`. There are **no open issues or PRs mentioning beam search** (manual check of `mlc-ai/xgrammar/issues`).

However, **the matcher state is fork-able and rollback-able**:

- `GrammarMatcher.fork()` returns a deep-copy of the parsing state (stack, token history), sharing the compiled grammar (CompiledGrammar is immutable).
- `rollback(n)` rewinds n tokens; XGrammar's persistent execution stack makes this O(1).
- `BatchGrammarMatcher` provides thread-pool-backed `batch_fill_next_token_bitmask` over a slice of matchers.

These three APIs — fork, rollback, batched mask fill — are exactly the primitives a constrained-beam-search outer loop needs. XGrammar-2 (May 2026) extends this with `traverse_draft_tree` for speculative-decoding tree masks, the same shape of computation a beam loop would issue. The XGrammar team built fork/rollback specifically because **multiple parallel parsing stacks were needed for the persistent-stack design** — beam search is just another consumer.

**Atlas implication:** the work is in the outer sampler/scheduler loop, not in XGrammar itself.

## 2. Outlines (dottxt) + beam

Outlines v1 documents three constraint mechanisms — grammar compilation, **prefix trees that prune invalid paths during beam search**, and FSM-based sampling. The "prunes invalid paths during beam search" wording in dottxt's own materials is the **strongest signal that beam search is a first-class target** for Outlines.

Concrete production data on constrained-beam vs greedy quality on code generation is sparse. The HuggingFace `transformers` integration that uses Outlines/CFG (Saibo's PR #27557) exposes `num_beams` directly via `model.generate(..., num_beams=K, force_words_ids=...)`. The HF blog "Guiding Text Generation with Constrained Beam Search" demonstrates forced lexical constraints (e.g. translate-with-required-word). No published code-gen / tool-call accuracy numbers — this is empirical territory.

## 3. llama.cpp GBNF + beam

**Status: not implemented.** GitHub issue `ggml-org/llama.cpp#2923` "combined beam search + grammar sampling strategy" has been **open since 2023-08-31, no PR, no assignee.** The maintainers' note: implementation is complex because the grammar code must remain compatible with "beam search, sampling, all tokenizers, Unicode, etc." Beam search itself is supported in llama.cpp, GBNF is supported, but they have not been composed.

Practical reading: nobody has shipped this in llama.cpp because the combinatorial state-management work is real (clone grammar state per beam, prune dead beams, rerank). Atlas would face the same engineering cost.

## 4. arXiv — modern constrained-beam-search lineage

The classical lineage starts with **Hokamp & Liu 2017** (grid beam search) and **Post & Vilar 2018** (dynamic beam allocation), both for NMT lexical constraints. Modern LLM-era work that is directly relevant to Atlas:

- **CABS — Confidence-Aware Sub-Structure Beam Search** (arXiv 2406.00069, USC/Amazon, NAACL 2024). Beams branch at **sub-structure boundaries** (one beam per JSON field, not per token). A confidence head over the hidden state — not the conditional token probability — ranks sub-structures. On Amazon product-attribute generation: **+16.7 % recall @ 90 % precision** vs token-level beam. This is the closest published analog to "wrong filePath value inside a valid JSON" — same failure mode, same model class.

- **CRANE** (arXiv 2502.09061, Feb 2025). Alternates unconstrained reasoning with constrained generation on delimiter symbols. Uses **greedy decoding**, not beam. +10 pp on GSM-Symbolic / FOLIO. Relevant as a competing intervention: it suggests the win may come from giving the model a free reasoning span before the constrained span, not from K-way exploration.

- **Lost in Space** (arXiv 2502.14969). Argues the GCD performance gap is dominated by **tokenization** (leading-space tokens, format choices) and not by the search algorithm. Worth heeding: tokenization sanity-checks are cheaper than beam.

- **Trie-Based Beam for LLMs** (arXiv 2502.00085). Memory-efficient beam via prefix-trie KV-cache sharing. **No quality improvement on HumanEval** across beam widths — i.e. for code generation, beam search on a strong base model does not move quality. This is a yellow flag for Atlas's hypothesis.

- **CABS-LLM site (cabsllm.github.io)** confirms the sub-structure formulation generalizes; their evaluation is constrained to product-catalog generation.

- **DOMINO** (Beurer-Kellner 2024) — minimally invasive token-mask pre-computation; orthogonal to beam.

## 5. vLLM / SGLang / TRT-LLM production support

- **vLLM** soft-deprecated `use_beam_search` in v0.6.2 (Sep 2024), eliminated it from `SamplingParams` in v0.6.3. Beam search still lives behind `LLM.beam_search()` / `BeamSearchParams`, but is **"used rarely, hinders other optimizations"** (RFC #6226). vLLM 0.7.3 beam is **slower than 0.5.4** (issue #14426). Composing beam with XGrammar / Outlines logits processors in vLLM is **not a tested path**.

- **SGLang.** Open issues #903 (2024) and #3032 (Jan 2025) propose beam search; not yet shipped as of v0.5.11 (May 2026). XGrammar-2 integration is excellent (3× faster than vLLM), but **n=K is not exposed**.

- **TensorRT-LLM** supports beam search natively (C++ runtime). Structured output uses Outlines logit masks. Documentation states beam + structured-output are composable, but `diversity_penalty` (group beam) is not supported. **No FP8-specific notes** — TRT-LLM's FP8 path is orthogonal to the sampler.

**FP8 interaction.** No paper or release note that I located addresses "FP8-quantized model + constrained beam." Quantization noise affects per-token logits; beam search consumes ranked logits and is in principle **more robust to small per-token miscalibration** than greedy because it preserves the runner-up at every step. There is no published evidence either way.

## 6. Performance budget for tool-arg emission

Tool-call arguments at Atlas are typically **≤500 decoded tokens** per call, often <100.

| K  | Decode cost (vs greedy) | KV-cache cost (vs greedy) | XGrammar mask cost | Wall-clock budget @ 35 tok/s baseline |
|----|-------------------------|---------------------------|--------------------|---------------------------------------|
| 1  | 1×                      | 1×                        | 1× (≈40 µs/tok)    | 100 tok = 2.8 s                       |
| 2  | ~1.7× (KV share helps)  | ~1.7× (prefix shared)     | 2× (batched)       | 100 tok ≈ 4.8 s                       |
| 4  | ~3.0×                   | ~3.0×                     | 4× (≤160 µs/tok)  | 100 tok ≈ 8.4 s                       |
| 5  | ~3.7×                   | ~3.7×                     | 5× (~200 µs/tok)  | 100 tok ≈ 10 s                        |

Notes: (a) prefix-trie KV sharing recovers ~30 % vs naive K-way per the trie-beam paper; (b) XGrammar mask cost is CPU-side, parallelizable, and dwarfed by the GPU forward pass on GB10 (44 GB FP8 weights, ~3 B active per token, memory-bound); (c) Atlas's MTP path *cannot* be straightforwardly composed with beam — MTP draft-verify assumes K=1, and per memory `project_minimax_mtp_k_cap.md` the MTP head naturally caps drafts.

**Verdict on budget.** For a tool-arg span of ≤100 tokens, K=2 adds ~2 s wall-clock — **tolerable for a tool call** where correctness >> latency. K=5 adds ~7 s — borderline. The argument for beam over best-of-N rerank is **shared prefix KV** and **shared XGrammar state up to the divergence point**, which best-of-N cannot exploit.

## 7. Atlas-specific feasibility

- Atlas has **no existing beam search code path** (`grep -rni beam_search /workspace/atlas-mtp/*.rs` returns nothing). This is a from-scratch sampler addition.
- Atlas uses XGrammar via the Rust crate (`docs.rs/xgrammar`). The Rust crate exposes `BatchGrammarMatcher` and the underlying matcher; fork/rollback have to be plumbed if not yet bound — needs source check.
- KV-cache: Atlas paged-attention already supports shared prefixes (radix prefix cache, Marconi). Beam K candidates sharing a prefix is a special case of what the cache already does.
- MTP composition is **not free**: MTP commits to a single decoded sequence per step. K-way beam disables MTP for the constrained span. Net: lose 59.9→36.4 tok/s (per `project_models.md` MTP gain) in exchange for K-way search. At K=2 this still nets ~1.0× greedy throughput because beam × (no-MTP) ≈ 1.7 × 0.6.
- The drift problem (`project_qwen36_fp8_post_think_eos.md`, `MASTER_DRIFT_TABLE.md`) is **inside** the constrained span, and `MASTER_DRIFT_TABLE.md` cosine analysis shows MoE expert-routing flips on 8-bit FP8 dequant — meaning the corruption is on the **conditioning** path (hidden states feeding logits), not on a single bad sampling decision. Beam search re-runs sampling K times but cannot re-route an expert that was incorrectly selected upstream. **Beam will help if and only if the top-1 token under FP8 is sometimes the wrong-value token while top-2 is right-value.** This is empirically unknown for Atlas; needs measurement.

## 8. Ranked top-5 concrete interventions

1. **Measure first: top-K logit overlap between FP8 and BF16 inside tool-arg spans.** Run the existing dual-forward harness (`hf_dual_forward.py`, `cosine_run.py`) and dump top-5 logits at every decode step inside a constrained tool-call span for both quantizations. If the correct token is in FP8's top-2 ≥80 % of the time, beam K=2 is justified. If not, beam is theatre — invest the engineering elsewhere. **Effort: 1 day. No code in Atlas core. Highest information value.**

2. **Sub-structure best-of-N rerank (CABS-lite), no beam.** Decode the tool-call body N=4 times with temperature ~0.3 (independent samples through the existing path, full MTP intact), then rerank by **mean log-prob over argument-value spans only** (mask out the JSON-shape tokens whose probability is dominated by the grammar). Reuses existing infra, zero sampler changes, no KV-share gains but no MTP loss. Captures the bulk of CABS's published gain (+16.7 % recall) without grammar-state forking. **Effort: 2–3 days. Wraps the existing chat endpoint.**

3. **Plumb XGrammar `fork()` + `rollback()` into Atlas's Rust sampler and implement K=2 beam search restricted to the constrained span only.** Greedy outside the span, K-way inside. Reuses Atlas's existing prefix-cache for KV sharing across the 2 beams. Disables MTP only inside the span. Target gain: closes the FP8-vs-BF16 drift on tool-arg values per intervention #1's measurement. **Effort: 1–2 weeks. Live experiment after intervention #1 validates premise.**

4. **CRANE-style two-mode decoding: free reasoning, then constrained tool call.** Already Atlas's default path via `<think>` — but per memory `project_qwen36_phase2b_softmax_expf.md`, the FP8 drift is largest at deep layers L31-L39 during the constrained span. Add an explicit "scratchpad" between the model's reasoning closer and the tool-call opener: prompt-level intervention that lets the model emit one free-form sentence stating the intended argument values before the constrained JSON. Costs ~50 tokens, gives BF16-style hidden states a chance to assert the right value before the grammar mask kicks in. **Effort: 1 day. Pure prompt-template change. Orthogonal to beam.**

5. **Tokenization audit per Lost in Space.** Atlas's tool-call openers (`<tool_call>` token, JSON brace handling) and the `filePath` field type — verify that paths are emitted as character-level digit-equivalent tokens, not as one mega-token, and that leading-space variants are handled. Cheap, addresses a published-but-uninvestigated drift source on Atlas. **Effort: 0.5 day. Pure measurement.**

---

**Bottom line.** Constrained beam search is *technically feasible* on Atlas — XGrammar gives you fork/rollback, Atlas's prefix cache gives you KV sharing, and K=2 is wall-clock tolerable on tool-arg spans. But the **dominant FP8 drift mechanism on Qwen3.6** (MoE expert-routing flips, deep-layer hidden-state corruption) is *upstream of the sampler* and beam search cannot fix it. Intervention #1 (measure top-K logit overlap) is the gate that decides whether to spend the 1–2 week sampler-rewrite effort on #3, or to settle for #2's cheap sub-structure rerank.

---

## Sources

- XGrammar paper: arXiv 2411.15100, MLC blog 2024-11-22, Catalyst CMU
- XGrammar-2 blog (2026-05-04): blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation
- XGrammar-2 paper: arXiv 2601.04426
- llama.cpp issue #2923 (beam + grammar, still open)
- CABS: arXiv 2406.00069, cabsllm.github.io
- CRANE: arXiv 2502.09061
- Lost in Space: arXiv 2502.14969
- Trie-based beam: arXiv 2502.00085
- HuggingFace constrained beam blog: huggingface.co/blog/constrained-beam-search
- vLLM beam deprecation: vllm-project/vllm RFC #6226, PR #11297
- SGLang beam: sgl-project/sglang issues #903, #3032
- TRT-LLM C++ runtime docs (nvidia.github.io/TensorRT-LLM)
- Atlas memory: project_qwen36_drift_moe_smoking_gun.md, project_qwen36_fp8_post_think_eos.md, MASTER_DRIFT_TABLE.md
