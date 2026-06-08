# Research 3 — Noise-Resilient Sampling for Code Generation under FP8

Survey of arXiv (Jan 2024 – May 2026) on inference-time sampling techniques that
compensate for quantization noise, detect token-level errors before emission,
recalibrate over-confident wrong logits, and recover via retries / restarts.
Cross-mapped to Atlas drift modes on Qwen3.6-35B-A3B-FP8 hybrid (10 attn + 30
GDN): confident-wrong character substitution (`axum` → `axut`), runaway bash
concatenation, leaked `<|control|>` bytes inside tool-call args.

---

## 1. Decoding methods specifically for quantized / FP8 / FP4 LLMs

The *decoding-time* literature on quantization compensation is much thinner
than the weight-side literature (GPTQ, FOEM, low-rank correction).

- **DecDEC** (arXiv 2412.20185, ICLR 2025) — closest hit: dynamically detects
  salient channels *at each decoding step* and keeps them in FP16 while the
  rest stay in 3-bit. Llama-3-8B PPL 10.15 → 9.12. Per-step adaptive, not a
  one-shot post-training fix.
- **Progressive Mixed-Precision Decoding** (arXiv 2410.13461) — prefill vs.
  decode have different quantization sensitivity; raise bit width only on the
  last decode steps. Useful intuition that not every token is equally
  noise-sensitive.
- **SnapMLA** (arXiv 2602.10718) — RoPE-aware per-token FP8; confirms
  RoPE-bearing channels are the prime quantization victims.
- **Stochastic rounding for FP8 inference** (arXiv 2502.01070, 2503.09975) —
  consistently finds SR does *not* help and is sometimes harmful for small
  models. RTE/RNE is the right dequant choice (matches Phase 2b
  `fp8-dequant-rne`).
- **InfiR2 / FP8-RL** (arXiv 2509.22536, 2601.18150) — FP8 training, not decode.

**Headline**: there is no published inference-time decoder explicitly designed
to compensate for FP8 sampling noise. Frontier is DecDEC-style dynamic channel
rescue, a weight-side fix. Any Atlas decoder-side mitigation is novel.

---

## 2. Lookahead-free uncertainty estimation (beyond AdaDec's entropy)

AdaDec (arXiv 2506.08980, Jun–Sep 2025) uses Shannon entropy + a learned
threshold + lookahead re-rank, gaining up to +20.9% pass@1. The lookahead is
what we want to avoid. Cheaper alternatives:

- **TokUR** (arXiv 2505.11737) — injects **low-rank random weight perturbation**
  at decode time and measures predictive-distribution variance for one token.
  Single-forward-pass, not multi-sample. Most directly applicable to Atlas.
- **INSIDE / EigenScore** (arXiv 2402.03744) — eigenvalues of the response
  covariance in dense embedding space; companion result: *test-time feature
  clipping of extreme activations reduces over-confident generations*. Maps
  cleanly to FP8 outliers.
- **ICR Probe** (arXiv 2507.16488) — MLP probe on Information-Contribution-to-
  Residual-stream across layers; one extra matmul per token.
- **Spectral attention features** (arXiv 2502.17598) — top-k Laplacian
  eigenvalues of attention; adds a per-step eigendecomp.
- **CLAP** (arXiv 2509.09700) — joint residual-stream activation probe.
- **Intra-layer local information scores** (arXiv 2603.22299) — cheaper than
  MC dropout.
- **Logit-only estimator** (arXiv 2502.00290) — keeps signal at logit layer
  but contrasts over multiple sampling runs.
- **First-Hallucination-Tokens-Are-Different** (arXiv 2507.20836) — the *first*
  hallucinated token has a stronger signal than conditional ones. Directly
  relevant to Atlas's `axum→axut`, where the first wrong byte is load-bearing.

**Atlas takeaway**: entropy alone is insufficient; cheapest no-lookahead
add-ons are (a) TokUR low-rank perturbation, (b) INSIDE activation clipping,
(c) ICR-style residual-stream probe.

---

## 3. Token-level error detection at decode

Beyond entropy:

- **Variance signals across stochastic generations** (arXiv 2507.04137,
  Dec 2025) — reference-free token-level detection via variance in token
  log-probs across K samples. Requires N>1 — compatible with sample-and-vote,
  not single-sample streaming.
- **HaMI** (arXiv 2504.07863, NeurIPS 2025) — Multiple Instance Learning over
  token representations; trained on labeled corpora, inference is one head.
- **Token-Guard / Self-Checking Decoding** (arXiv 2601.21969) — latent-space
  per-step verification with explicit risk score; designed to catch
  hallucinated tokens *before they propagate*. Closest spirit to Atlas's
  tool-call-arg position.
- **HALP** (arXiv 2603.05465) — VLM but the "predict risk from pre-generative
  internal states" principle ports.
- **Internal Reps for Tool-Selection** (arXiv 2601.05214) — agent-specific:
  lightweight classifier on contextualized embeddings flags wrong tool calls
  during generation. Highly applicable to opencode + Atlas.
- **Real-Time Hallucinated Entities** (arXiv 2509.03531) — linear/MLP probes
  during generation, AUC > 0.85.

Dominant pattern: pre-trained probe on residual stream → fires flag → caller
decides.

---

## 4. Calibration: detecting when confidence is unreliable

Convergent finding: LLMs are systematically over-confident with identifiable
internal circuits.

- **Multicalibration for Code Generation** (arXiv 2512.08810, Dec 2025) —
  IGLB and LINR multicalibration; uncalibrated scores are over-confident on
  code specifically.
- **Wired for Overconfidence** (arXiv 2604.01457) — pinpoints specific MLP
  blocks and attention heads in mid-to-late layers that write the
  confidence-inflation signal. Targeted inference-time intervention on those
  circuits improves calibration. Directly relevant: Atlas drift concentrates
  in deep attention layers (L31–L39 per Phase 2b probe).
- **LLMs Overconfident / FermiEval** (arXiv 2510.26995) — verbalized 80% CI
  is wrong ~50% of the time.
- **Dunning-Kruger LLM** (arXiv 2603.09985) — overconfidence is worse for
  small/quantized models.
- **Know When You're Wrong** (arXiv 2603.06604) — SFT/PPO/GRPO/DPO all inflate
  confidence; not inference-time but explains bias origin.
- **Ensemble Semantic Entropy** (arXiv 2603.27098) — LiveCodeBench: ensemble
  semantic entropy is the only signal that strongly correlates with functional
  correctness, +53.4% selective-generation accuracy at strict FPR. Strongest
  direct evidence that *for code*, naive confidence fails and semantic-level
  signals work.

---

## 5. Sampling with retries / restarts (research-side, not engineering-side)

- **Speculative Rejection / Fast Best-of-N** (arXiv 2410.20290) — score
  partials early, kill unlikely ones. ~16-32× speedup over naive Best-of-N.
- **Self-Estimating Best-of-N** (arXiv 2503.01422) — early-decoding sample
  close to other samples is more likely to win; propagate consistency to prune.
- **LATTS** (arXiv 2509.20368) — step-level verifier; if rejected, sample from
  a stronger model.
- **Reward-Shifted Spec Sampling** (arXiv 2508.15044) — training-free alignment
  via weak-aligner draft + strong-aligner verify.
- **EARS** (arXiv 2512.13194) — acceptance threshold *adapts to target model's
  uncertainty*. Rejection-sampling counterpart to AdaDec's threshold; near-
  perfect mechanism match for Atlas's drift profile.
- **CARS** (arXiv 2510.01902) — constraint-aware rejection sampling validated
  on program fuzzing; meshes with XGrammar.
- **AdverMCTS** (arXiv 2604.10449) — MCTS against pseudo-correct generations;
  behavioral-level rejection.
- **ROCODE / IterGen / SemGuard** (arXiv 2411.07112, 2410.07295, 2509.24507) —
  backtracking decoders. ROCODE = incremental program analysis + rewind;
  IterGen = grammar-symbol KV rewind; SemGuard = learned semantic evaluator.
  Research version of the rewind-and-retry pattern Atlas has stubbed for
  spec-decode.
- **Inverse-Entropy Voting** (arXiv 2511.02309) — sequential rather than
  parallel self-consistency; entropy-weighted vote.

---

## 6. Stochastic decoding for FP8 specifically

Honest answer: no paper directly claims "T>0 + voting helps FP8 specifically".
Closest signals:

- **Temperature in Test-Time Scaling** (arXiv 2510.02611) — different T's
  solve different problems; some items unsolvable below a certain T. Argues
  against single-T inference. FP8 noise indirectly enlarges this effect.
- **Ensemble Semantic Entropy** (arXiv 2603.27098) — multi-sample voting works
  on code.
- **USCD** (arXiv 2409.05923, ACL Findings 2025) — Uncertainty-aware Selective
  Contrastive Decoding for code. Uses an intentionally-degraded "lame prompt"
  as noise prior, contrasts logits against it, JS divergence between token
  uncertainty and output noise ~0.25. +16.59% pass@1. Closest match to "treat
  FP8 as the lame model and contrast against a BF16 reference" — Atlas's
  ATLAS_GDN_DUMP comparison run is exactly this prior.
- **Adaptive Temperature for Code** (arXiv 2309.02772) — adaptive per-token T;
  pre-dates FP8 wave but precision-agnostic.

Cleanest supported hypothesis: FP8 logits + periodic re-sample at higher T
when uncertainty trips + majority-vote across small N. TokUR and Variance-
Signals validate this for non-quantized models; no FP8 ablation exists yet.

---

## Top-5 inference-time interventions for Atlas (ranked)

All five are zero-training, no-weight-changes, decoder-side, and can be staged
without disrupting EP=2 / NVFP4 paths.

1. **USCD-style contrastive decoding against the lame prior**
   (arXiv 2409.05923). Atlas already has a degraded prior available — the FP8
   noise floor itself can serve as the "lame model" by running the same prompt
   with maximum entropy / no system instructions, OR by contrasting FP8 logits
   against a periodic BF16 reference shard. Largest published code gains
   (+16.59% pass@1), drop-in at logits.
2. **Variance-across-stochastic-samples token-level error gate**
   (arXiv 2507.04137). Run N=3–5 fast forward passes per high-stakes position
   (tool-call args, identifier characters), measure log-prob variance, and
   reject-and-resample positions where σ exceeds a learned threshold. Cheaper
   than full Best-of-N because it triggers only at gated positions, and
   directly attacks the `axum→axut` confident-wrong substitution.
3. **EARS uncertainty-adaptive rejection sampling** (arXiv 2512.13194). Couple
   the existing spec-decode reject path to a *model-uncertainty-modulated*
   acceptance threshold rather than a fixed one. Re-uses the MTP rewind
   machinery Atlas already has and matches the empirical fact that the model
   drifts when uncertain.
4. **TokUR low-rank weight-perturbation uncertainty estimate**
   (arXiv 2505.11737). One extra forward pass per gated position with a small
   random low-rank perturbation; gives a noise-aware uncertainty score that
   is more sensitive than naive entropy. Combine with #3 as the trigger
   signal.
5. **INSIDE feature-clipping at inference** (arXiv 2402.03744). Clip extreme
   activations in late attention layers — the same L31–L39 the Phase 2b probe
   identified as the FP8-KV cliff — to reduce over-confident generations.
   Zero new compute, pure activation-level intervention; orthogonal to the
   sampling stack.

**Defer for now (good papers, wrong shape for current Atlas)**:

- ROCODE / IterGen / SemGuard backtracking — needs an incremental program
  analyzer (heavy build-out).
- ICR / CLAP / FFT probes — require labeled training data per model.
- Wired-for-Overconfidence circuit-level interventions — needs head/MLP
  attribution work specific to Qwen3.6 architecture.
- Speculative Rejection / Best-of-N — too expensive at 35B FP8 + 16k context
  for interactive code-agent use.

---

## Sources

- AdaDec — https://arxiv.org/abs/2506.08980
- DecDEC — https://arxiv.org/abs/2412.20185
- Progressive Mixed-Precision Decoding — https://arxiv.org/abs/2410.13461
- SnapMLA — https://arxiv.org/abs/2602.10718
- FP8 Across Accelerators (datacenter) — https://arxiv.org/abs/2502.01070
- FP8 on Intel Gaudi — https://arxiv.org/abs/2503.09975
- TokUR — https://arxiv.org/abs/2505.11737
- INSIDE / EigenScore — https://arxiv.org/abs/2402.03744
- ICR Probe — https://arxiv.org/abs/2507.16488
- Spectral Attention Maps — https://arxiv.org/abs/2502.17598
- CLAP — https://arxiv.org/abs/2509.09700
- Intra-Layer Local Information Scores — https://arxiv.org/abs/2603.22299
- Logit-Only Uncertainty — https://arxiv.org/abs/2502.00290
- First-Hallucination-Tokens — https://arxiv.org/abs/2507.20836
- Variance Signals Token-Level — https://arxiv.org/abs/2507.04137
- HaMI Adaptive Token Selection — https://arxiv.org/abs/2504.07863
- Token-Guard / Self-Checking — https://arxiv.org/abs/2601.21969
- HALP (VLM) — https://arxiv.org/abs/2603.05465
- Internal Reps for Tool-Selection — https://arxiv.org/abs/2601.05214
- Real-Time Hallucinated Entities — https://arxiv.org/abs/2509.03531
- Multicalibration for Code — https://arxiv.org/abs/2512.08810
- Wired for Overconfidence — https://arxiv.org/abs/2604.01457
- LLMs Overconfident / FermiEval — https://arxiv.org/abs/2510.26995
- Dunning-Kruger LLM — https://arxiv.org/abs/2603.09985
- Know When You're Wrong — https://arxiv.org/abs/2603.06604
- Ensemble Semantic Entropy — https://arxiv.org/abs/2603.27098
- Speculative Rejection (Best-of-N) — https://arxiv.org/abs/2410.20290
- Self-Estimating Best-of-N — https://arxiv.org/abs/2503.01422
- LATTS — https://arxiv.org/abs/2509.20368
- Reward-Shifted Spec Sampling — https://arxiv.org/abs/2508.15044
- EARS — https://arxiv.org/abs/2512.13194
- CARS — https://arxiv.org/abs/2510.01902
- AdverMCTS — https://arxiv.org/abs/2604.10449
- ROCODE — https://arxiv.org/abs/2411.07112
- IterGen — https://arxiv.org/abs/2410.07295
- SemGuard — https://arxiv.org/abs/2509.24507
- Inverse-Entropy Voting — https://arxiv.org/abs/2511.02309
- Temperature in Test-Time Scaling — https://arxiv.org/abs/2510.02611
- USCD — https://arxiv.org/abs/2409.05923
- Adaptive Temperature for Code — https://arxiv.org/abs/2309.02772
