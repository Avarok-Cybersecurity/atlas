# Long-Context Multi-Turn Drift: arXiv SOTA Survey (2024-2026)

**Target**: Atlas Qwen3.6-35B-A3B-FP8 multi-turn degradation in opencode after ~5K-15K tokens. Single-pass cosine vs HF BF16 is 0.994; multi-turn quality collapses. Architecture: 40-layer hybrid (30 GDN + 10 full-attention), MoE 256 experts top-8, `rope_theta=10M`, `partial_rotary_factor=0.25`, YaRN context-extension, FP8 weights + FP8 KV cache.

Cross-references: `project_qwen36_drift_moe_smoking_gun.md` (MoE expert flip from 8/8 → 3/8 across L0→L38), `project_qwen36_phase2b_softmax_expf.md` (FP8 KV cache deep-layer drift unmasked after softmax fix), `project_qwen36_phase2b_results.md` (MMA precision is the residual ceiling, not rounding).

---

## 1. Why models degrade well below their training-time max context

### 1.1 "Context rot" is now an empirically established phenomenon
Chroma's 2025 study (https://www.morphllm.com/context-rot) tested 18 frontier models including GPT-4.1, Claude Opus 4, Gemini 2.5 — **every model degrades at every input-length increment tested**, with an average 39% performance drop when instructions are split across turns. The three measured mechanisms are: (a) lost-in-the-middle (attention attends well to ends, poorly to middle), (b) attention dilution from softmax-over-N tokens (effective focus collapses), and (c) distractor interference (similar-but-irrelevant content actively misleads). None of these are training-cutoff issues — they appear at 10K-30K tokens routinely.

### 1.2 The "Drift No More?" formalism (Dongre et al., arXiv 2510.07777, Oct 2025)
Multi-turn drift is formalised as turn-wise KL divergence between the test model and a goal-consistent reference. Critically, the authors prove **drift converges to a noise-limited equilibrium, not runaway decay**, and that **simple "reminder" interventions empirically reduce KL divergence in line with theoretical predictions**. This implies Atlas's multi-turn collapse is partially recoverable by sampling-/prompting-side interventions, even without fixing the underlying FP8 noise floor.

### 1.3 Error-accumulation is not exponential
"Beyond Exponential Decay" (arXiv 2505.24187, May 2025) shows error accumulates sub-exponentially in well-trained LLMs and is dominated by **strategy choices and tokenwise feedback**, not memory exhaustion. This matches Atlas's observed pattern: single-pass 0.994 cosine but multi-turn collapse implies the per-token error is amplified by the autoregressive loop, not by a single bad layer.

---

## 2. Runtime-only KV-cache fixes (no fine-tuning required)

### 2.1 StreamingLLM + attention sinks (Xiao et al., ICLR 2024, 2309.17453)
Foundational. Demonstrates that **keeping the KV of the first 1-4 tokens** ("sink tokens") plus a recent window restores quality at infinite length, and that LLMs allocate disproportionate attention to position-0 regardless of content. NVIDIA shipped it in TensorRT-LLM January 2024. **Caveat for Atlas**: at 5K-15K tokens we are not at the StreamingLLM regime — full KV still fits — but the *attention-sink dynamics* still operate. See §3.

### 2.2 DuoAttention (Han Lab, arXiv 2410.10819, ICLR 2025)
Splits attention heads into **retrieval heads** (need full KV) and **streaming heads** (need only sink + recent window). Uses 64 sink tokens + 256 recent. Achieves 2.55× MHA / 1.67× GQA memory reduction and 2.18× decode speedup. **Identification of streaming-vs-retrieval heads is offline** (one-shot calibration on synthetic needle tasks), no fine-tuning.

### 2.3 KIVI / KVLinC / Kitty (KV cache quantization)
- **KIVI** (ICML 2024, 2402.02750): 2-bit KV cache, per-channel keys + per-token values. Degrades long-context quality measurably at 32K+ tokens.
- **KVLinC** (arXiv 2510.05373) and **Kitty** (arXiv 2511.18643) explicitly target KIVI's long-context degradation with Hadamard rotation + linear correction. Kitty maintains accuracy at 32K.
- **"More for Keys, Less for Values"** (arXiv 2502.15075): asymmetric K vs V bit-allocation — keys are more sensitive than values, so spend bits there. Direct relevance to Atlas's FP8 KV path: if Atlas treats K/V symmetrically, switching to 8-bit K / 6-bit V (or vice-versa for FP8) may improve deep-layer drift.

### 2.4 Q-Filters (arXiv 2503.02812, 2024)
Compatible with FlashAttention (no attention-matrix materialization). Uses QK geometry to score KV importance offline. Achieves 99% needle-in-haystack accuracy at 32× compression. Reduces generation-perplexity drop **by 65% vs StreamingLLM**.

### 2.5 MInference / RetrievalAttention / SnapKV / PyramidKV
- **MInference 1.0** (arXiv 2407.02490, NeurIPS 2024 spotlight): dynamic sparse attention masks for prefill. 10× prefill speedup on RULER/Needle/PG-19. Pure prefill — does not address multi-turn decode.
- **RetrievalAttention** (arXiv 2409.10516): vector-search-based KV retrieval. 1-3% of KV accessed per query. **Out-of-distribution-aware** — handles query-vs-key distribution shift, which is the analog of Atlas's deep-layer noise issue.
- **SnapKV / PyramidKV / H2O**: attention-score-based eviction. All suffer from **monotonic position bias** (early tokens retained even when stale) and are **incompatible with FlashAttention** at >100K tokens. For Atlas at 15K, this is acceptable but provides no near-term win.

### 2.6 FlowKV (arXiv 2505.15347, NeurIPS 2025) — directly relevant
**Multi-turn isolation mechanism**: preserves accumulated compressed KV from past turns, applies compression *only* to the newly generated KV of the latest turn. Prevents recompression of older context (catastrophic forgetting). **Improves instruction-following retention from 10.9% → 75.4% in later turns vs baseline eviction.** Wraps any KV compression method, no training needed.

---

## 3. Attention-sink injection at runtime — for SHORT context (15K)

### 3.1 ZeroTuning (Wang et al., arXiv 2505.11739, May 2025) — most relevant
**Tunes only the initial-token attention bias at inference time**, no parameter updates, no KV-cache changes, kernel-agnostic (works with FlashAttention/SDPA). Supervised mode calibrates on validation; unsupervised minimizes output entropy. **Improves Llama-3.1-8B multi-turn score 7.804 → 7.966** (+0.16) and +11.71% on classification. Direct Atlas mitigation candidate.

### 3.2 ACT — Attention Calibration (yuzz1020.github.io/ACT/)
"Unveiling and Harnessing Hidden Attention Sinks" (ICML 2024). Identifies *hidden* sinks (positions other than 0 that absorb attention mass) and rescales their attention scores at inference. **Training-free**.

### 3.3 PASTA / AutoPASTA — Post-hoc Attention Steering
Runtime attention-mass redirection toward user-marked spans. Demonstrably improves task accuracy over prompting alone. No fine-tuning. Useful for tool-calling where the function-signature span should dominate attention.

### 3.4 Qwen-specific caveat (critical)
Per "When Attention Sink Emerges" (ICLR 2025) and follow-ups: **Qwen models do NOT consistently sink to position 0**. LLaMA shows 77-89% sink concentration on token 0 across all depths; Qwen shows 39-48% with distributed sinks across common linguistic elements. Qwen2.5-7B sinks to position **1**, not 0. **Implication**: naïve StreamingLLM-style "keep first 4 tokens" is not safely transferable to Qwen3.6 — the actual sink positions must be measured empirically, layer-by-layer and head-by-head, before injection.

---

## 4. FP8 KV cache × long context — quality interaction

### 4.1 ShadowKV (arXiv 2410.21465) demonstrates FP8 viability
Reports FP8 maintains accuracy on RULER at 128K and LongBench >4K. **But**: ShadowKV uses **per-layer per-head dynamic scales**, not Atlas's likely-static block scales. Static scaling under hidden-state-norm growth (Atlas measures 0.57 → 10.63 from L0 → L38) is exactly the failure mode flagged in `project_qwen36_phase2b_softmax_expf.md`.

### 4.2 "The Pitfalls of KV Cache Compression" (arXiv 2510.00231)
Demonstrates degradation curves are **non-monotonic in compression ratio** — moderate compression can mask precision bugs by adding regularization-like noise, while heavy compression exposes them. Matches Atlas's observation that the polynomial softmax was *masking* deep-layer FP8 KV noise.

### 4.3 Asymmetric K/V precision
Adaptive KV quantization papers (More-for-Keys, ThinKV) consistently recommend **more bits on K, fewer on V** for long-context coherence. Atlas's `--kv-cache-dtype fp8` treats K and V identically — splitting precision (e.g., NVFP4 K + FP8 V, or FP8 K + INT4 V) may bias quantization budget where it matters.

### 4.4 NVFP4 KV — empirically better at deep layers (Atlas internal data)
`project_qwen36_phase2b_softmax_expf.md` Table: `__expf + NVFP4 KV = 0.968 mean, 0.931 deep` beats `__expf + FP8 KV = 0.967 mean, 0.927 deep` *specifically at L35-L39*, which is where multi-turn collapse manifests. NVFP4's per-block calibration handles the L38 ||h||=10.63 magnitudes better than FP8's larger blocks.

---

## 5. Hybrid SSM/attention — layer-specific interventions

### 5.1 LongMamba (arXiv 2504.16053, 2025)
**Training-free** receptive-field enlargement for Mamba/GDN. Demonstrates that pure linear-attention recurrent state has finite capacity and degrades on retrieval beyond training length. Solution: dilated/strided state updates at inference. Relevant for Atlas's 30 GDN layers.

### 5.2 "Understanding and Enhancing Mamba-Transformer Hybrids" (arXiv 2510.26912)
For Qwen3.6's 3:1 GDN:attention ratio, the **softmax attention layers carry retrieval**, the GDN layers carry compression. Memory-recall degradation at long context concentrates in the **softmax-attention layers** (which is consistent with Atlas L35-L39 being the worst — those layers are mostly full-attention given the 30/10 distribution clusters).

### 5.3 Qwen3-Next architecture confirmation
The Qwen3-Next 80B (1:3 attention:GDN in repeating blocks of 4) is documented to have **needle-in-haystack quality reliant on the 25% softmax layers** (developer.nvidia.com/blog/new-open-source-qwen3-next-models). Atlas Qwen3.6-35B has a similar 10/40 ratio. The implication: **precision degradation in the 10 attention layers** (especially deep ones) bottlenecks multi-turn retrieval.

### 5.4 SSM state preservation during speculative-decode rollback
"Mamba Drafters for Speculative Decoding" (arXiv 2506.01206) and the vLLM bug tracker (sgl-project/sglang #18590, vllm-project/vllm #39273) confirm: **SSM/GDN recurrent state is NOT rewindable** in a verify-reject cycle without explicit per-token checkpointing. Atlas's MTP K=1 path may corrupt GDN state on rejected drafts unless it checkpoints all 30 GDN layers (the existing memory `project_qwen36_drift_moe_smoking_gun.md` and Atlas's prior NGram experiments confirm this is a known landmine — 73% reject rate × 30 layers of corrupted state = exactly the multi-turn collapse pattern).

---

## 6. Position encoding (YaRN/NTK) at runtime
`rope_theta=10M` + `partial_rotary_factor=0.25` + YaRN scaling factor=4.0 + `original_max_position_embeddings=262144` means **only 25% of head dims rotate**, and YaRN is already applied at training time. There is no runtime YaRN tweak that helps at 15K — we are well within the trained range. Position encoding is exonerated as a cause of <30K-token degradation.

---

## 7. Direct answers to user questions

1. **Why degrade <30K context?** Three additive mechanisms: (a) precision-driven per-layer drift (Atlas's MoE 8/8 → 3/8 cascade), (b) FP8 KV noise on deep-layer attention reads (L35-L39 ||h||=10.63), (c) multi-turn drift compounding (KL-divergence equilibrium per Dongre 2025). All three operate in Atlas right now.

2. **Runtime-only multi-turn fixes that demonstrably work**: ZeroTuning (initial-token bias), FlowKV (per-turn KV isolation), DuoAttention (head-split sink+window), Q-Filters (FlashAttention-compatible compression), reminder-injection per Dongre. All training-free.

3. **Layer-specific interventions for Qwen3.6's 30+10 hybrid**: The 10 attention layers dominate long-context retrieval; precision interventions should be concentrated there. Atlas already has `--kv-high-precision-layers` infra (per memory) — repoint it to NVFP4 (not BF16, which has the latent L35-L39 bug) for the deep attention layers.

4. **Post-hoc sink injection on Qwen**: Possible (ZeroTuning, ACT) but **must measure Qwen's actual sink positions first** — Qwen2.5-7B sinks to position 1, not 0. Atlas needs a sink-discovery pass before injection. Naive "add token 0" will not work as it does on LLaMA.

