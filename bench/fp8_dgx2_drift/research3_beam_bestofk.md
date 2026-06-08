# Research 3: Beam Search, Best-of-K, and Multi-Candidate Decoding for FP8 Drift

**Target problem:** Qwen3.6-35B-A3B-FP8 on Atlas exhibits character-level drift at *low-entropy* positions
(`axum`→`axut`, `axum-v3`→`axuma-aadac`, hyphens dropped, runaway parens, leaked control chars in tool args).
Model is *confidently wrong*. AdaDec / entropy-gated interventions do not trigger.
**Question:** what do production inference engines and post-Oct-2025 research do for multi-candidate
decoding tuned to quantization noise?

---

## 1. Production engines: beam search status

### vLLM
- **Beam search is soft-deprecated** in vLLM. Issue #6226 explicitly lists "implementing beam search
  introduces significant system complexity that hinders potential optimizations". Beam-search is now
  a separate offline path (`vllm.LLM.use_beam_search`) gated by `VLLM_ALLOW_DEPRECATED_BEAM_SEARCH=1`.
  V1 engine sampler does not support it natively.
- **No FP8-aware sampling adjustments.** The V1 sampler operates on whatever logits arrive; it has
  no knowledge of upstream weight/activation precision. Logits-processor API is the only extension
  point and runs *after* the noisy GEMM has already happened.
- **Custom logits processors** (`vllm.v1.sample.logits_processor.LogitsProcessor`) are the supported
  surface for intervention but apply a (num_requests x vocab_size) tensor transform — they cannot
  re-run the forward pass.

### SGLang
- Supports offline-quantized models (FP8, GPTQ, AWQ, AutoRound) and ModelOpt integration. Their docs
  explicitly state that **"quantized models must be validated via benchmarks post-quantization to
  guard against abnormal quantization loss regressions"** — i.e. the engine itself takes no runtime
  precaution against drift.
- No documented FP8-aware sampler. Speculative-decoding pipeline (EAGLE-3) is the dominant
  multi-candidate path.

### TensorRT-LLM
- Supports beam search + Variable-Beam-Width-Search (VBWS) in the new TRTLLM Sampler. Logits
  processors are exposed via `SamplingParams.logits_processor`.
- Has no documented FP8-specific compensation in sampling. The only FP8-aware compensation in the
  TRT-LLM stack is *pre-sampler*: ModelOpt's per-tensor / per-block scaling calibration done at
  build time.

### DeepGEMM / Marlin (kernel layer)
- DeepGEMM's FP8 MQA logits kernel is documented to have "noticeable per-element error" on
  Blackwell/B200, with the recommended validation being **cosine similarity over the whole tensor,
  not per-element**. This is exactly the failure mode Atlas observes: aggregate cosine looks fine
  (0.967 mean) while individual confidently-wrong tokens slip through.
- No sampling-side compensation. Marlin and DeepGEMM are GEMM kernels; the sampler runs after.

**Bottom line:** none of the production engines have an FP8-aware sampler or beam variant. The
industry has converged on "calibrate well at build time + post-quant benchmark + speculative decode
for throughput, not quality." Atlas's drift is an unaddressed failure mode in the SOTA stack.

---

## 2. Post-Oct 2025 arXiv: margin-aware and self-certainty work

### MARS (2601.15498, Jan 2026) — *most directly applicable*
Margin-Aware Speculative Verification uses the **logit ratio r_t = z_(2) / z_(1)** (top-2 over top-1)
as the verification signal. When r_t > θ (default 0.9) the model is in a "low-margin" regime and MARS
relaxes verification (accepts plausible runner-ups). The contrapositive is the lever for Atlas:
**when the FP8 model claims an extremely high margin but the choice is wrong, the ratio is anomalous
relative to a BF16 reference distribution.** Per the MARS authors, "the logit ratio's invariance to
global magnitude scaling makes it potentially useful for detecting over-confident-wrong tokens in
low-precision inference. Quantization artifacts often compress margins while inflating raw logit
magnitudes." Not empirically validated by them for that inverse use, but the formula is clean and
cheap (1 div per token).

### Self-Certainty (2502.18581)
Self-certainty = mean KL(predicted_token_dist || uniform). Explicitly designed to outperform
entropy-based metrics that are biased by superficial confidence. Used in best-of-N selection and
shown to beat self-consistency on code generation. **Per-token application** is feasible (one KL
per token) and could be a drift flag uncorrelated with raw entropy — which is exactly what Atlas
needs given AdaDec doesn't trigger.

### DecDEC (2412.20185)
Dynamic Error Compensation. Identifies salient channels offline; at inference, keeps them in higher
precision and compensates accumulated quantization error in real time. Reduces 3-bit Llama-3-8B
perplexity 10.15 → 9.12. Conceptually closest to Atlas's bug but the implementation is *pre* sampling
(changes the GEMM, not the decoder). Adapting it means Atlas-side quant-error compensation in the FP8
dequant path — bigger surgery than a sampler change.

### CDSL / CoLD (2412.10418, NAACL 2025)
Constrained Decoding with Speculative Lookaheads. Algorithm:
1. Draft model generates L lookahead tokens.
2. Target validates via argmax-mismatch → acceptance fraction `a = first_mismatch / L`.
3. Reward function scores the validated portion → `r`.
4. Four-state decision matrix:
   - High a, high r → accept all
   - Low a → discard, regenerate from target with lookahead scoring
   - High a, low r → discard, single-token constrained generation
5. **Scoring is rollout-level, not per-token** (limitation for char-level drift).

CDSL's idea of "draft → target verifies → reward-scores the survivors" maps directly onto:
*MTP drafts → Atlas verifies → grammar/structural reward catches drift*. Atlas already has the
infrastructure (MTP heads, XGrammar) — the missing piece is the reward function and the rollback
on low reward.

### TECP (token-entropy conformal prediction) + EARS (2512.13194)
TECP uses token entropy + conformal quantiles for provably-bounded per-token rejection thresholds —
Atlas-specific calibration set could yield model-specific thresholds. EARS uses
threshold = 1 - max(P_target) as a cheap (one-max-per-token) adaptive rejection signal.

---

## 3. Best-of-K / N-sample voting patterns

| Method | Selector | Overhead | Notes |
|---|---|---|---|
| **Vanilla self-consistency** | Majority vote | N forward passes | Standard; expensive |
| **Functional Majority Voting** (2604.15618) | Execute candidates on self-generated inputs; cluster by runtime signature | N + executor sandbox | Code-specific; "without large compute overhead" per authors |
| **Speculative Rejection** (2410.20290) | Reward model halts low-scoring branches early | 16-32x cheaper than Best-of-N | Needs a reward model |
| **Self-Certainty voting** (2502.18581) | Weighted by KL-from-uniform rank | Free (uses existing logits) | Beats self-consistency on code |
| **Soft Best-of-N** (2505.03156) | Distribution-aware sampling | N forward passes | Tunable temperature on the selector |

For Atlas's drift profile, FMV is most attractive: if the bug is `axum`→`axut` in a bash command,
the bash command *won't execute* and the variant with `axum` will. But FMV needs sandboxed
execution — out of scope for the inference engine; this is a client-side wrap.

---

## 4. Speculative decoding as drift mitigator

This is the most important finding for Atlas:

- **DFlash (vLLM, May 2026)**: Qwen3.6 35B-A3B FP8 207→484 tok/s with spec decode on Blackwell. Author
  notes "FP8 target weights don't appear to hurt token acceptance materially" — but Atlas's own data
  shows MoE expert routing diverges at deep layers from FP8 dequant; whether MTP catches Atlas's drift
  depends on whether drafter and verifier see the same noisy logits.
- **ML-SpecQD (2503.13565)**: critical data point — "Testing with unquantized BF16 multi-token
  prediction produced 0% acceptance ... re-quantizing to match the backbone achieves ~46% acceptance."
  **Smoking gun for the BF16-verify pattern**: a BF16 drafter against an FP8 target is hostile, not
  helpful. The drafter must match the target precision to *speed up*; but flipping it (BF16 verifier,
  FP8 drafter) gives *quality* recovery at a throughput cost. Not done in production but it is the
  natural cure for confidently-wrong FP8 tokens.
- **SubSpec** (vLLM RFC #39427, 2026): 4-bit substitute drafts, BF16 verify, 6.2× kernel speedup —
  exists in the wild. Verification is correctness-preserving: any token the BF16 model wouldn't have
  emitted at the same position is rejected.

---

## 5. Lookahead decoding with noisy lookahead

Lookahead Decoding (Microsoft / vLLM, 2402.02057) is **exact, parallel, training-free** — it doesn't
care if the lookahead is noisy because the target model verifies every accepted token. The
verification mechanism is greedy argmax match. If FP8 noise causes the lookahead to drift, the
target rejects and falls back to single-token decode — *for that step*. But Atlas's failure mode is
that the target itself is confidently wrong, so the verifier accepts the wrong token. **Lookahead
decoding does not help when the noise is in the target, not the draft.**

---

## Ranked top-5 concrete interventions for Atlas

### #1: Margin-ratio drift detector (MARS-inverse), 2-3 days
Per-token compute `r_t = logit[top-2] / logit[top-1]` plus the entropy that AdaDec already
computes. Build a calibration table on a known-good BF16 reference: for each (token, context-class)
pair, record the BF16 margin distribution. At Atlas FP8 inference time, flag tokens where the FP8
margin is in the top decile (suspiciously decisive) but the predicted token lies in the BF16
distribution's tail. **Doesn't require re-running BF16 at inference** — only at calibration. When
flagged, fall back to a second strategy (resample with temperature, or re-decode with BF16-equiv
small head). Scope: per-token KL is ~1 div + 1 compare. Should plug into Atlas's existing sampler.

### #2: MTP self-verify with BF16 head, 1-2 weeks
Atlas already runs MTP. Add a configuration where the MTP draft is FP8 but the final argmax
verification of each MTP-accepted token re-runs *only the lm_head* in BF16 against the same hidden
state. lm_head is ~120M params for Qwen3.6 — cheap. Reject tokens where BF16-lm_head and FP8-lm_head
disagree on top-1. This catches the case where MMA precision corrupted the projection. **Does not
require re-running attention or MoE in BF16.** Per ML-SpecQD's data, the lm_head is where the
projection-precision impact is most localized. Scope: ~1 week loader + sampler change, ~1 week
integration with MTP rollback path.

### #3: XGrammar-driven reward gate (CDSL pattern), 1 week
Atlas has XGrammar for tool calls. Extend its role: when generating bash/code/tool-arg content,
the grammar already knows valid character classes at each position. Add a *reward* signal: for each
MTP-accepted token, score against grammar plausibility (1 if in the allowed-byte-set at that
position, lower otherwise — including bracketing balance / quote balance / hyphen-vs-letter
discrimination for identifier-like contexts). On low reward, rollback to single-token greedy with
strict grammar mask. Cleaner than per-token character checks because it composes with existing
XGrammar infrastructure. Scope: ~1 week. Targets exactly the failure modes listed
(runaway parens, dropped hyphens, leaked control chars).

### #4: Self-certainty per-token threshold, 3-4 days
Compute mean-KL-from-uniform on every token; calibrate a per-model threshold via conformal
prediction (TECP-style). When self-certainty < threshold AND entropy is also low (the contradiction
that AdaDec misses), flag as drift candidate. Combine with #1 or #3 as the recovery mechanism.
Cheap (one log-sum over vocab). Lower coverage than #1 but stacks with it.

### #5: Server-side N=4 self-consistency for tool-arg content only, 1 week
Most general-text drift is tolerable; the user pain is in tool args / bash. Detect (via prompt-side
or grammar state) that the current output region is a tool argument or code block. For that
sub-region only, generate K=4 candidates at low temperature and run a self-certainty + structural
validator (URL-parse, bash-parse, JSON-parse) selector. Reject candidates that fail to parse; pick
the highest self-certainty among parsers. Per FMV's results this is "without large compute
overhead" because the sub-regions are short (tens of tokens, not full responses). Scope: ~1 week
for region detection + N-sample plumbing + validator wiring.

---

## What we ruled out

- **Full beam search**: vLLM's deprecation reasoning applies. Beam search picks highest cumulative
  log-prob — but FP8 noise means the wrong path often has the higher log-prob. Makes it *worse*.
- **BF16 full-model verify**: 2x memory, ~2x latency; Atlas's reason for FP8 is the memory budget.
- **DecDEC port**: high engineering cost (kernel-level); prior memory notes show GDN exonerated and
  drift is in MoE / FP8 dequant / gated RMSNorm. Multi-week kernel rewrite.
- **Lookahead decoding**: doesn't address noisy *target*, only noisy *drafter*.
- **Contrastive decoding (amateur vs expert)**: needs a second forward pass; framing doesn't map
  onto a single FP8 model.

---

## Sources

- vLLM beam-search deprecation: https://github.com/vllm-project/vllm/issues/6226
- vLLM FP8 KV-cache state (Apr 2026): https://vllm-project.github.io/2026/04/22/fp8-kvcache.html
- vLLM custom logits processors: https://docs.vllm.ai/en/latest/features/custom_logitsprocs/
- SGLang quantization docs: https://docs.sglang.io/advanced_features/quantization.html
- TRT-LLM sampling: https://nvidia.github.io/TensorRT-LLM/latest/features/sampling.html
- DeepGEMM FP8 per-element error note: https://pyshine.com/DeepGEMM-Efficient-FP8-GEMM-Kernels/
- MARS (margin-aware verification): https://arxiv.org/abs/2601.15498
- Self-Certainty Best-of-N (2502.18581): https://arxiv.org/pdf/2502.18581
- DecDEC (2412.20185): https://arxiv.org/abs/2412.20185
- CDSL / CoLD (2412.10418): https://aclanthology.org/2025.naacl-long.239/
- Functional Majority Voting (2604.15618): https://arxiv.org/abs/2604.15618
- Speculative Rejection (2410.20290): https://arxiv.org/abs/2410.20290
- ML-SpecQD (2503.13565): https://arxiv.org/abs/2503.13565
- SubSpec RFC: https://github.com/vllm-project/vllm/issues/39427
- DFlash on Qwen3.6 35B FP8: https://allenkuo.medium.com/when-speculative-decoding-helps-local-llms-and-when-it-doesnt-5c41dd804e4b
- Lookahead Decoding (2402.02057): https://arxiv.org/pdf/2402.02057
- XGrammar: https://arxiv.org/pdf/2411.15100
- Quantization confidence dilemma (2111.08163): https://arxiv.org/pdf/2111.08163
- EARS (2512.13194): https://arxiv.org/pdf/2512.13194
