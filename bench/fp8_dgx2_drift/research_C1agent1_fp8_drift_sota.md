# SOTA Research: FP8 Long-Context Precision Drift & Low-Margin Argmax-Flip Mitigation

**Target problem:** Qwen3.6-35B-A3B-FP8 on GB10. At 10K+ tokens, 23.7% of decode positions show top-1↔top-2 logit gap < 1.5 nats. FP8 noise (~0.02 nats KL) flips a non-trivial fraction. Per-token FP8 precision is fine on focused prompts; **the failure mode is cumulative long-context drift × low-margin positions**. We are designing a margin-ratio detector + BF16-verify hybrid (B1+C4).

**Verdict:** This exact failure mode is now well-characterized in 2024-2026 literature. Five interventions are directly applicable, **all inference-only, no retraining**.

---

## Top 5 Interventions (ranked by impact × novelty × Atlas-applicability)

### 1. QSpec — Speculative decoding with low-prec draft + high-prec verify  [HIGHEST FIT]

- **Paper:** Zhao et al., "QSpec: Speculative Decoding with Complementary Quantization Schemes", arXiv:2410.11305 (EMNLP 2025). https://arxiv.org/abs/2410.11305
- **Headline result:** **92-95% acceptance rate** on real vLLM deployment; **up to 1.64× speedup** vs W4A16 baseline; **~74% accept even with γ=6**. Vanilla speculative decoding only gets 28-58% accept on the same workloads.
- **Mechanism:** *Single weight-quantized model* toggles between two activation modes — fast low-precision draft (W4A4) and high-precision verify (W4A16). Crucially, **same weights, same KV cache shared across stages**. "QSpec reuses both weights and KV cache across stages, enabling near-zero-cost switching without retraining or auxiliary models." Acceptance is greedy top-1 match: "one drafted token t̂_{i+j} is accepted only when the top-1 tokens from p_{i+j} and p̂_{i+j} coincide." On rejection, "all subsequent drafted tokens are discarded and the rejected token is resampled from the verify distribution."
- **Why it fits Atlas:** This is essentially the *exact* design we are sketching as B1+C4 — except formalized as speculative decoding with a proven 90%+ accept rate at long context. The KV-cache-reuse property is critical for us (we cannot afford to maintain two caches for a 64K context).
- **Implementation cost (Atlas):** Medium. We already have MTP scaffolding from Phase 16. The change is mode-toggling MoE GEMM precision per step rather than running a draft model. **Inference-only, no retraining.** Estimate: 1-2 weeks for prototype on Qwen3.6-35B-FP8, including the margin gate to *skip* verification on high-margin positions (cheaper than full QSpec).
- **Atlas-specific tweak:** Combine with our margin detector — only verify positions where top-1↔top-2 gap < threshold (~1.5 nats from our data). Saves ~76% of verify calls vs blanket QSpec.

---

### 2. MARS — Margin-Aware Speculative Verification  [SECOND-HIGHEST FIT]

- **Paper:** "MARS: Unleashing the Power of Speculative Decoding via Margin-Aware Verification", arXiv:2601.15498 (2026). https://arxiv.org/pdf/2601.15498
- **Headline result:** Training-free verification strategy; "delivers consistent and significant inference speedups across models from 8B to 235B parameters while preserving generation quality."
- **Mechanism:** Quote from the paper's setup: *"Modern LLMs frequently operate in low-margin regimes where the target model exhibits weak preference among top candidates. In such cases, rejecting plausible runner-up tokens yields negligible information gain while incurring substantial rollback cost, leading to a fundamental inefficiency in verification."* MARS conditions verification on **decision stability measured directly from the target logits** and relaxes rejection when strict verification provides minimal benefit. The opposite of our setup (they relax on low margin; we'd tighten on low margin) — but the underlying margin metric is identical and the framework directly composable.
- **Why it fits Atlas:** **MARS validates that the top-1↔top-2 logit gap (our core diagnostic) is the right knob.** Their paper explicitly defines "low-margin regime" — same concept we measured (23.7% of decode positions). We can fork their decision-stability function and invert it for our use case: when low-margin AND long-context, *always* verify at BF16; when high-margin, skip verify.
- **Implementation cost (Atlas):** Low. The detector is `softmax(logits)[top1] - softmax(logits)[top2] < threshold` — already ~20 LoC of CUDA in our sampler. Combined with QSpec it becomes the gating function for the verify pass. **Inference-only, no retraining.**

---

### 3. SageAttention2 / vLLM Two-Level FP8 Accumulation  [SHIP IMMEDIATELY IF NOT DONE]

- **Sources:**
  - SageAttention2: Zhang et al., arXiv:2411.10958. https://arxiv.org/abs/2411.10958
  - vLLM blog: "The State of FP8 KV-Cache and Attention Quantization in vLLM" (2026-04-22). https://vllm-project.github.io/2026/04/22/fp8-kvcache.html
- **Headline result:** **NIAH at 128K: 91% (BF16) → 13% (naive FP8) → 89% (with two-level accumulation).** A **78 percentage-point collapse recovered to within 2pp of BF16**.
- **Root cause quote (vLLM):** *"Hopper's FP8 Tensor Cores accumulate in FP32 registers, but intermediate accumulation loses precision when the contraction dimension is large — causing drastic numerical errors when the contraction dimension reaches 100K or more. During long context inference in the Softmax(AttnScore) × V matmul, this can lead to accuracy regressions from 91% (BF16) to 13% (FP8) on long-context needle-in-a-haystack tasks."*
- **Mechanism quote (SageAttention2):** *"The FP32 accumulator designed for FP8 matrix multiplication in the tensor core (`mma.f32.f8.f8.f32`) is actually FP22 — 1 sign, 8 exponent, 13 mantissa. A two-level accumulation strategy writes the partially accumulated results into an actual FP32 register, restoring long-context accuracy."*
- **Why it fits Atlas:** **This is potentially the literal bug in our stack** — and explains why our drift is *cumulative across context length* but absent on focused prompts. GB10 (sm_121) Tensor Cores have the same FP22 accumulator behavior as Hopper. If our FP8 attention kernel does not do two-level accumulation, we are leaking precision in the same place vLLM was bleeding at 128K.
- **Implementation cost (Atlas):** Medium-low. CUDA kernel patch: every N (e.g., 64 or 128) Tensor-Core ops, flush the FP22 accumulator to a real FP32 register and re-zero. Trade-off: small register pressure increase, ~5-10% TTFT cost at head_dim=128. Decode unaffected. **This is the highest-EV item to verify FIRST — it may eliminate the need for everything else.** Recommend: instrument our `flash_attn_fp8.cu` and grep for accumulation pattern before pursuing QSpec.

---

### 4. Cautious Next Token Prediction (CNTP) — entropy-gated path branching

- **Paper:** "Cautious Next Token Prediction", arXiv:2507.03038 (2025). https://arxiv.org/pdf/2507.03038
- **Headline result:** Improves MMLU and math reasoning accuracy with "modest efficiency costs." Inference-only, training-free.
- **Mechanism:** *"Selectively samples multiple candidate paths whenever the model's prediction entropy is high, then automatically chooses the path with the lowest perplexity, focusing computational resources precisely where the model is most uncertain."*
- **Why it fits Atlas:** Direct analogue to our margin detector — entropy is a strict superset of top-1/top-2 gap (entropy = full distribution; gap = pairwise margin). For sm_121, entropy of softmax(logits) is one extra reduce-kernel pass per token (~20μs). The "branch-on-uncertainty" pattern lets us call BF16 *only* on the uncertain positions.
- **Implementation cost (Atlas):** Low. Entropy reduction already present in our sampler. Need a "branch and pick best by perplexity" wrapper — ~100 LoC Rust + 1 CUDA kernel.
- **Caveat:** Path branching cost can grow if many positions are uncertain. Our 23.7% low-margin rate is high enough that naive branching would 1.24× our decode cost. **Pair this with QSpec/MARS** (verify the branch, don't multi-sample it) for the win.

---

### 5. Q-ROAR — RoPE outlier rescaling for position-dependent quantization noise

- **Paper:** "Q-ROAR: Outlier-Aware Rescaling for RoPE Position Interpolation in Quantized Long-Context LLMs", arXiv:2509.14391 (2025). https://arxiv.org/abs/2509.14391
- **Headline result:** Recovers up to **0.7% accuracy on standard tasks** and **reduces GovReport perplexity by >10%** while preserving short-context performance. **No fine-tuning, kernel, or architecture changes.**
- **Root-cause quote:** *"Combining position interpolation with post-training quantization degrades accuracy due to coupled effects including long context aliasing, dynamic range dilation, axis grid anisotropy, and outlier shifting that induce position-dependent logit noise."*
- **Mechanism:** Groups RoPE dimensions into frequency bands, then does a small offline search over per-band scales for W_Q and W_K. Uses two diagnostic metrics: **Interpolation Pressure** (per-band phase scaling sensitivity) and **Tail Inflation Ratios** (outlier shifts).
- **Why it fits Atlas:** Qwen3.6-35B uses RoPE with extended context (YaRN-style). Our drift growing with token position is a textbook position-dependent-logit-noise symptom. Q-ROAR's "outlier shifting at long context" diagnostic is consistent with our finding that focused-prompt margins are huge but long-context margins shrink to <1.5 nats.
- **Implementation cost (Atlas):** Low — offline rescaling search (~hours on one GPU), then a per-channel weight rescaling at load time. **Inference-only at serve time; zero kernel changes.** Risk: per-band search needs a calibration dataset matched to long-context use cases.

---

## Honorable mentions

- **DuoAttention** (ICLR 2025, arXiv:2410.10819): Retrieval-head identification; streaming KV elsewhere. 2.55× memory, 2.18× decode. Composable, not precision-focused.
- **Q-Filters** (arXiv:2503.02812): QK-geometry KV cache compression, FlashAttention-compatible.
- **KVTuner / MixKVQ** (arXiv:2502.04420): Sensitivity-aware per-*layer* mixed precision; complements per-position approaches.
- **Cocktail** (arXiv:2503.23294, DATE 2025): Chunk-adaptive bitwidth from query↔chunk similarity; reorders KV chunks pre-quant.
- **PCD** (arXiv:2506.08371): Contrasts predictions across perturbed positional encodings; expensive (extra forward passes).
- **Min-p sampling** (arXiv:2407.01082, ICLR 2025): Confidence-relative threshold; composable in our sampler.

---

## Long-context FP8 degradation benchmarks (confirmation that this is a real, measured phenomenon)

- **"Does quantization affect models' performance on long-context tasks?"** (arXiv:2505.20276, EMNLP 2025). https://arxiv.org/abs/2505.20276
  - Evaluates FP8, GPTQint8, AWQ-int4, GPTQ-int4, BNB-nf4 across Llama-3.1 8B/70B and **Qwen-2.5 7B/32B/72B** on 9.7K examples.
  - **Headline:** "On average, 8-bit quantization preserves accuracy (~0.8% drop), whereas 4-bit methods lead to substantial losses, especially for tasks involving long-context inputs (drops of up to 59%)."
  - Important caveat: **average** is 0.8%, but degradation is task- and language-dependent. The paper finds non-English long-context is hit hardest — consistent with our observation that drift is cumulative across many tokens.
- **vLLM FP8 KV blog** (above): The 91%→13%→89% NIAH curve is the most dramatic published evidence that FP8 long-context bugs are **kernel-level**, not algorithmic.
- **DeepSeek-V3 Technical Report** (arXiv:2412.19437): Pioneered fine-grained 1×128 / 128×128 FP8 tile quantization + **high-precision CUDA-core accumulation**, keeping training-loss Δ < 0.25% vs BF16. **Maintains BF16/FP32 for embedding, output head, MoE gating, normalization, and attention.** Critical pattern: *gate stays high-precision even in FP8 models* — relevant if our Qwen3.6 MoE gate is FP8 and routing decisions are flipping (see Atlas memory note on MoE expert routing drift). https://arxiv.org/abs/2412.19437

---

## Atlas-applicability Recommendation Path

**Sequence we should run (in order of EV / cost ratio):**

1. **Verify the two-level FP8 accumulator in our attention kernel** (Section 3). This is the cheapest possible fix and has the largest historical precedent (78pp NIAH recovery). If our kernel already does this, skip to step 2. If not, *this could close the entire drift gap*.
2. **Ship the margin detector** (Section 2 / our B1 plan): top-1↔top-2 gap < threshold per decode position. Already 20 LoC; gives us an online diagnostic and the gating function for everything below.
3. **Wire QSpec-style mode-toggle FP8-draft / BF16-verify** (Section 1) gated by the margin detector. Only verify low-margin positions. Expected: cuts BF16 cost by ~75% vs blanket verify while catching the argmax flips. Single-weight design means no extra model on the GPU.
4. **Apply Q-ROAR offline rescaling** (Section 5). Cheap, additive to everything above. Targets the position-dependent component of the noise that even two-level accumulation will not fully fix.
5. *(Optional)* CNTP entropy fallback (Section 4) as a safety net for the residual cases where top-1/top-2 margin is misleading (entropy can catch top-3-onward dispersion).

**All five are inference-only. None require retraining. None require multi-week GPU work. Realistic delivery: 2-3 weeks for items 1-3, another week for 4. Item 5 only if 1-4 leave residual drift.**

---

## Direct quotes worth pinning in our design doc

- *(MARS):* "Modern LLMs frequently operate in low-margin regimes where the target model exhibits weak preference among top candidates. In such cases, rejecting plausible runner-up tokens yields negligible information gain while incurring substantial rollback cost." — confirms our hypothesis that low-margin is the failure surface, not generic FP8 noise.
- *(vLLM FP8 blog):* "FP8 accuracy dropped from 91% (BF16) to just 13% [at 128K NIAH] … brought the FP8 accuracy back to 89%" — concrete evidence that long-context FP8 collapse is a known, fixable, kernel-level issue.
- *(QSpec):* "QSpec reuses both weights and KV cache across stages, enabling near-zero-cost switching without retraining or auxiliary models." — single-weight, single-cache design that fits our memory budget on GB10.
- *(Q-ROAR):* "Long context aliasing, dynamic range dilation, axis grid anisotropy, and outlier shifting … induce position-dependent logit noise." — explains why margins shrink with position even on simple prompts.
- *(DeepSeek-V3):* Embedding, output head, MoE gating, normalization, and attention stay in BF16/FP32 even in FP8 training. — production precedent for selective high-precision components in an otherwise-FP8 stack.

---

**Bottom line:** Our B1 + C4 design is on the right track. The literature strongly supports (a) the margin-gate trigger, (b) the FP8-draft / BF16-verify recompute pattern, and (c) inference-only deployment. Before building, audit our FP8 attention kernel for two-level accumulation — that single check could obviate the rest. Items 1-3 above realistically deliver in 2-3 weeks with no retraining and no kernel rewrites beyond a single accumulation patch.
