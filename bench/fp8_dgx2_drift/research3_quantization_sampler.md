# Research 3: Quantization-Aware Sampling

**Premise tested.** FP8 E4M3 has a 3-bit mantissa (~2^-3 ≈ 12.5% relative
precision). When `lm_head` is in FP8 — or upstream activations feeding a BF16
`lm_head` are FP8 — logits inherit a noise band on the order of a few percent
of magnitude. Question: is a BF16-tuned sampler wrong for those logits?

Bottom line: **no production framework (vLLM, TRT-LLM, SGLang) ships
quantization-aware sampler logic.** All keep `lm_head` in BF16/FP16 and treat
the sampler as quant-agnostic. Academia has, in the last 18 months, converged
on two directly applicable ideas: **margin-aware decisions** (MARS, CM-ASD,
Cautious-NTP, Calibrated-TopK) and **uncertainty-triggered logit steering**
for quantized models (Tethered Reasoning). These are the levers Atlas can pull.

---

## 1. Confirmed at framework level: `lm_head` stays high-precision

vLLM, TRT-LLM, SGLang all keep the LM head in BF16/FP16 regardless of body
precision. [vLLM FP8 docs](https://docs.vllm.ai/en/v0.5.4/quantization/fp8.html):
*"all Linear modules (except for the final `lm_head`) have their weights
quantized down to FP8_E4M3."* This isn't choice — it's empirically forced:
[arXiv:2411.02355](https://arxiv.org/pdf/2411.02355) shows including `lm_head`
in FP8 measurably hurts quality, more than any single body Linear.

So the noise floor we observe is **upstream-FP8 noise projected into BF16
logits**, not native FP8 logits — but still real: 2^-3 hidden-state errors
dotted with the embedding matrix yield per-vocab logit σ ≈
sqrt(hidden_dim) × hidden_state_noise_rms. By the time hidden states reach
`lm_head`, they've absorbed ~36 layers × 7 FP8 GEMMs of mantissa loss, even
with DeepGEMM-style two-level FP32 accumulation inside each GEMM.

---

## 2. The directly relevant arXiv papers (2024–2026)

### 2.1 Top-nσ — Gaussian noise model for logits (arXiv:2411.07641)

[Top-nσ](https://arxiv.org/pdf/2411.07641) is the single most-applicable paper:

- Logits decompose into a **Gaussian noise region** (μ, σ²) plus a small
  **outlier informative region**.
- Cutoff: `threshold = M − n·σ`, M = max logit, n ≈ 1.0 empirically.
- **Temperature-invariant** (T rescales M and σ identically).
- Cost: one O(vocab) pass for (μ, σ, M).

Paper makes no quantization claim. But the structure — Gaussian noise floor
below informative outliers — is **exactly what FP8 produces**: uncorrelated
mantissa errors over hidden_dim, projected onto vocab rows, is approximately
Gaussian by CLT. Top-nσ is the closest existing "FP8-aware truncation."

### 2.2 MARS — margin-aware speculative verification (arXiv:2601.15498)

[MARS](https://arxiv.org/pdf/2601.15498) is canonical: *"Modern LLMs frequently
operate in low-margin regimes where the target model exhibits weak preference
among top candidates."* Strict verification is wasteful when top1−top2 is
negligible. They condition verification relaxation on **logit margin measured
directly from target logits**. Spec-decode mirror of what Atlas needs in
straight sampling: when FP8 logit gap < FP8 noise floor, the greedy decision is
ill-conditioned.

### 2.3 Confidence-Modulated Adaptive Speculative Decoding (arXiv:2508.15371)

[CM-ASD](https://arxiv.org/pdf/2508.15371) uses **entropy + logit margin** as
twin uncertainty signals to modulate draft length and verification strictness.
Closest existing production-shape framework for treating low-margin tokens
differently.

### 2.4 Cautious Next-Token Prediction — CNTP (arXiv:2507.03038, ACL 2025)

[CNTP](https://arxiv.org/pdf/2507.03038) detects high-entropy steps and fires
parallel lookahead trials, picking lowest-perplexity. *"The less confident the
model is, the more trials it should sample."* Lookahead fallback triggered by
uncertainty — direct answer to "what to do when FP8 noise > top1−top2 margin."

### 2.5 Tethered Reasoning / Helix manifold steering (arXiv:2602.17691)

[Tethered Reasoning](https://arxiv.org/pdf/2602.17691) is the only paper that
**explicitly models quantization noise as effective-temperature increase**
(*"ε_q creates an effective temperature increase, causing earlier collapse than
full-precision."*). Fix: logit-level intervention triggered by uncertainty —
when a Mahalanobis-distance score crosses τ, *"penalize the top logit
proportionally to confidence deficit."* Only 0.2–2.5% of tokens get steered,
but 4-bit Granite at T=3.0 retains 88.84% GSM8K (vs collapse without steering).
Strongly suggests light-touch uncertainty-triggered intervention helps FP8.

### 2.6 Sample Smart, Not Hard / Calibrated-TopK (arXiv:2510.05987)

[Calibrated-TopK](https://arxiv.org/pdf/2510.05987) calibrates truncation by
**rank-wise correctness**, not confidence alone. Operational insight:
*"Greedy-Threshold makes sampling greedy at very low confidence steps"* —
**invert** the FP8 intuition. Greedy when confident (margin ≫ noise floor),
wider distribution when margin ≤ noise floor. Negligible overhead (single
vector-op).

### 2.7 Min-p (arXiv:2407.01082) and p-less (arXiv:2509.23234)

[Min-p](https://arxiv.org/pdf/2407.01082) at threshold 0.05 is reported to help
[1.58-bit models](https://www.thoughtworks.com/insights/blog/generative-ai/Min-p-sampling-for-LLMs)
counteract "rare prediction noise" — **already empirically tuned for quantized
models**, just not advertised that way.

### 2.8 Adaptive Temperature Scaling (arXiv:2409.19817, EMNLP 2024)

[ATS](https://arxiv.org/pdf/2409.19817) predicts per-token temperature from
token-level features. Directly composable with FP8 noise: tiny head predicts
"this position is FP8-noisy → raise T" vs "clean → leave T."

### 2.9 "Give Me BF16 or Give Me Death" (arXiv:2411.02355)

W8A8-FP is essentially lossless **provided `lm_head` is BF16**. Including
`lm_head` in quantization is the single biggest accuracy hit.

### 2.10 Layer-wise FP4 sensitivity (arXiv:2603.08747)

Confirms MLP up/down projections are most FP4-sensitive; does **not** address
`lm_head` or sampling — measurement gap Atlas could fill on GB10.

---

## 3. TRT-LLM / DeepGEMM / community: nothing on sampler side

[TRT-LLM Sampling](https://nvidia.github.io/TensorRT-LLM/latest/features/sampling.html)
lists temperature, top-k, top-p, min-p, beam, spec, guided — none
quant-aware. [Numerical Precision docs](https://nvidia.github.io/TensorRT-LLM/reference/precision.html)
have **zero discussion** of FP8 effect on LM head, logits, or sampling.

[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) is a pure GEMM library —
no sampler. Two-level accumulation (FP8 in, FP32 promote, BF16 store) is the
only noise mitigation, strictly inside the GEMM.

vLLM's [FP8 KV-cache blog](https://vllm.ai/blog/2026-04-22-fp8-kvcache) ack's
long-context precision regressions but only proposes attention-side fixes.

**Pertinent failure mode**: vLLM
[#39407](https://github.com/vllm-project/vllm/issues/39407) — Gemma 4 FP8_BLOCK
double-scaling saturates logits at soft-cap (23.625), greedy locks repetitively.
Inverse case: when FP8 noise destroys the gap entirely, every greedy step is
degenerate.

---

## 4. Local de-noising / Gaussian smoothing — answered

**No production framework does this; no published paper recommends it for FP8.**
[SmoothQuant+](https://arxiv.org/pdf/2312.03788) operates on **activations
before GEMM**, not logits after. Theoretical reason it shouldn't help: a
Gaussian filter over the vocab dim blurs informative outliers together with
noise — exactly wrong. Top-nσ (cut below M − n·σ) strictly dominates because
it preserves informative-token ranking. Where smoothing implicitly helps is
spec-decode verification (CM-ASD, MARS) via margin-based relaxation.

---

## 5. FP4 / NVFP4 sampler implications

NVFP4: FP8 E4M3 per-16-element scale + FP32 per-tensor scale. Effective
mantissa precision comparable to FP8 for dominant in-block values. Early
systems ([NVIDIA](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/),
[ZeroShot](https://zeroshot.it.com/nvfp4-on-blackwell-practical-guide-theory-and-benchmarks-for-4%E2%80%91bit-llms/))
converge on:

- Keep `lm_head` in FP8 or BF16, never NVFP4.
- Keep attention in FP8 or BF16; mixed-precision beats full-NVFP4 on
  quality-sensitive evals.

No NVFP4 system documents sampler-side changes vs FP8/BF16. Gap: NVFP4 likely
needs slightly larger n in Top-nσ (or larger min-p) because the noise floor
is taller.

---

## 6. Recommended top-5 sampler-side interventions for Atlas

Ranked by (ROI × low regression risk). All additive, gateable by env-var.

### 1. Top-nσ truncation with FP8-aware n (HIGHEST ROI)

Implement Top-nσ ([arXiv:2411.07641](https://arxiv.org/pdf/2411.07641)):
compute (μ, σ, M) in one O(vocab) pass, cut below `M − n·σ`. Set n adaptively
from quant metadata:

- BF16: n = 1.0
- FP8 E4M3: n ≈ 1.3
- NVFP4: n ≈ 1.5

Composable with temperature (paper proves invariance). Eliminates the long tail
of "FP8 noise-floor tokens slip into top-p" cases BF16 never sees.

### 2. Margin-gated greedy fallback (Calibrated-TopK style)

Define `gap = logit[0] − logit[1]`. Estimate
`noise_floor ≈ k · σ_hidden_state · sqrt(hidden_dim)` (precomputable,
~2^-3 × hidden_state_rms). If `gap < noise_floor`, fall through to top-p/min-p
with T_fallback = 1.2 × T_normal. Inverse of
[Sample Smart Not Hard](https://arxiv.org/pdf/2510.05987): greedy when
confident, broader when noisy. Cost: 2 comparisons. Risk: needs careful
noise_floor calibration.

### 3. Min-p with FP8-aware floor

Lift min-p from 0.05 to ~0.08 on FP8 paths. Min-p 0.05 already
[helps 1.58-bit models](https://www.thoughtworks.com/insights/blog/generative-ai/Min-p-sampling-for-LLMs);
raising slightly accounts for the bigger noise floor. Trivial config change.

### 4. Uncertainty-triggered lookahead (CNTP-lite)

When margin < noise_floor AND position > ~16, fire 2 parallel lookahead drafts
of length 4–6, pick lowest cumulative perplexity to next punctuation/EOS.
Direct [CNTP](https://arxiv.org/pdf/2507.03038) adoption. ~1–3% of tokens
trigger (per Tethered Reasoning steering rate). Best for tool-call args where
small errors cascade.

### 5. Top-logit clamping (Tethered Reasoning, lite)

When margin < noise_floor AND entropy moderate, subtract ~0.5 × noise_floor
from the top logit before sampling. Lightweight
[Tethered Reasoning](https://arxiv.org/pdf/2602.17691) without manifold
construction. Rationale: if gap is within noise, "top" isn't reliably max —
pulling it down by half the noise band gives the runner-up a fair shot.
Start conservative (0.25× noise_floor).

---

## 7. NOT recommended

- **Gaussian logit smoothing**: blurs informative tokens; Top-nσ dominates.
- **Quantize `lm_head` to FP8**: empirically catastrophic (arXiv:2411.02355).
- **Beam as primary fallback**: too expensive at 60 tok/s; CNTP lookahead is
  cheaper and similar quality.
- **Hard-coded T-bump for all FP8**: blunt; noise floor varies by position.
  Per-position margin-gated (interventions 2, 4, 5) is the right granularity.

---

## 8. Atlas MTP note

Atlas MTP draft logits are FP8 like target; the **draft-vs-target margin
comparison** amplifies low-margin errors. CM-ASD (arXiv:2508.15371) and MARS
(arXiv:2601.15498) are the right references: relax verification when target
top1−top2 < FP8 noise floor (MARS-style); shrink draft length when draft
entropy is high (CM-ASD-style). Independent of main-sampler interventions.

---

## Sources

- [arXiv:2411.07641 — Top-nσ](https://arxiv.org/pdf/2411.07641)
- [arXiv:2601.15498 — MARS](https://arxiv.org/pdf/2601.15498)
- [arXiv:2508.15371 — CM-ASD](https://arxiv.org/pdf/2508.15371)
- [arXiv:2507.03038 — Cautious NTP](https://arxiv.org/pdf/2507.03038)
- [arXiv:2602.17691 — Tethered Reasoning](https://arxiv.org/pdf/2602.17691)
- [arXiv:2510.05987 — Sample Smart, Not Hard](https://arxiv.org/pdf/2510.05987)
- [arXiv:2407.01082 — Min-p](https://arxiv.org/pdf/2407.01082)
- [arXiv:2509.23234 — p-less](https://arxiv.org/pdf/2509.23234)
- [arXiv:2409.19817 — Adaptive Temperature Scaling](https://arxiv.org/pdf/2409.19817)
- [arXiv:2411.02355 — Give Me BF16 or Give Me Death](https://arxiv.org/pdf/2411.02355)
- [arXiv:2603.08747 — Diagnosing FP4 Inference](https://arxiv.org/pdf/2603.08747)
- [arXiv:2505.22179 — Speculative Decoding Meets Quantization](https://arxiv.org/html/2505.22179v1)
- [arXiv:2405.18710 — To FP8 and Back Again](https://arxiv.org/pdf/2405.18710)
- [TRT-LLM Sampling docs](https://nvidia.github.io/TensorRT-LLM/latest/features/sampling.html)
- [TRT-LLM Numerical Precision docs](https://nvidia.github.io/TensorRT-LLM/reference/precision.html)
- [vLLM FP8 docs](https://docs.vllm.ai/en/v0.5.4/quantization/fp8.html)
- [vLLM FP8 KV-cache blog](https://vllm.ai/blog/2026-04-22-fp8-kvcache)
- [vLLM bug #39407 — Gemma 4 31B FP8 logit saturation](https://github.com/vllm-project/vllm/issues/39407)
- [DeepGEMM repo](https://github.com/deepseek-ai/DeepGEMM)
- [Thoughtworks — Min-p sampling for LLMs](https://www.thoughtworks.com/insights/blog/generative-ai/Min-p-sampling-for-LLMs)
- [NVIDIA — Introducing NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [ZeroShot — NVFP4 on Blackwell](https://zeroshot.it.com/nvfp4-on-blackwell-practical-guide-theory-and-benchmarks-for-4%E2%80%91bit-llms/)
