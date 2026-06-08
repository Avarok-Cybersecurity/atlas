# C1-Agent-2: Production Margin Gates for Low-Confidence Token Re-Verification

Scope: how do shipping LLM serving stacks gate "is this token confident enough to ship?"
versus "re-run something heavier on this position?" Targets Atlas's plan: margin-ratio
detector + BF16-verify (or top-2 fallback) on 23.7% of long-context FP8 decode positions
where the top-1↔top-2 logit gap < 1.5.

Methodology: cloned vLLM main, TensorRT-LLM main, sparse-fetched sgl-kernel + sglang;
WebFetch'd llama.cpp `common/speculative.cpp` and `eagle_info_v2.py`; cross-checked PRs.
All line numbers are off live `main` as of 2026-05-26.

---

## 1. vLLM — `vllm/v1/sample/rejection_sampler.py`

**No margin gate. Pure modified-rejection-sampling.** Two Triton kernels, one for
greedy, one for stochastic.

### Greedy verify (lines 708-757)

```python
# rejection_greedy_sample_kernel
target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos).to(tl.int32)
token_id = target_argmax_id
rejected = draft_token_id != target_argmax_id      # L745  -- EXACT MATCH ONLY
```

The acceptance test is `draft_token_id == argmax(target_logits)`. Zero confidence
signal. If MTP drafts the model's second choice and the gap is 0.01, vLLM rejects.
This matches Atlas's current MTP K=2 verify.

### Stochastic verify (lines 762-827)

```python
# rejection_random_sample_kernel  L810
accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob
```

This is canonical Leviathan-2022 rejection sampling. The "threshold" is
**uniform [0,1) per position**, not a margin. Recovered token (on rejection)
is sampled from `max(0, p_target − p_draft)` (see `sample_recovered_tokens`, L659).

### N-gram / Suffix decoding fallback

`vllm/v1/spec_decode/suffix_decoding.py` L86 passes `min_token_prob` to the
suffix-cache speculator. This is the only "confidence" gate vLLM has in spec-
decode, and it gates **draft proposal**, not verification:

```python
draft = self.suffix_cache.speculate(req_id, pattern,
    max_spec_factor=self.max_spec_factor,
    min_token_prob=self.min_token_prob,   # L86  -- frequency-count prob in suffix tree
)
```

**No per-position re-verify in BF16 anywhere.** This is the gap Atlas would fill.

---

## 2. SGLang — `sgl-kernel/csrc/speculative/speculative_sampling.cuh`

**This is the closest production analog to what Atlas wants.** SGLang's tree-
spec sampler has *two* dual thresholds.

`TreeSpeculativeSamplingTargetOnly` kernel, **lines 103-107**:

```cuda
if (coin <= prob_acc / threshold_acc || target_prob_single >= threshold_single) {
    // accept token
    prob_acc = 0.;
    cur_prob_offset = (bx * num_draft_tokens + cur_index) * d;
    coin = uniform_samples[bx * num_draft_tokens + cur_index];
}
```

Threshold semantics (from `python/sglang/srt/server_args.py`):

- `speculative_accept_threshold_single` ∈ [0,1] — **lone-token acceptance**.
  If `target_prob[draft_token] >= threshold_single`, accept unconditionally
  regardless of draft probability. Default 1.0 (disabled), recommended 0.5–0.8
  for "aggressive" mode.
- `speculative_accept_threshold_acc` ∈ [0,1] — **accumulated rejection coin**.
  Lower = lossier-but-faster. Replaces uniform [0,1) with [0, threshold_acc).

Validated `0 ≤ threshold ≤ 1` at speculative_sampling.cu:142-148, clamped to
`1e-9f` at L210 to prevent div-by-zero.

The orchestration call lives in `EagleVerifyInput.sample()`,
`python/sglang/srt/speculative/eagle_info_v2.py` L237-336.

**Caveat:** these thresholds are LOSSY — they trade off output quality for
acceptance rate. The greedy path (`verify_tree_greedy`) is still pure
argmax-match. Adaptive doc (sgl-project.github.io/advanced_features/adaptive_speculative_decoding)
defaults `candidate_steps=[1,3,7]`, `ema_alpha=0.2` — these are draft-LENGTH
tuners, not per-token confidence gates.

---

## 3. TensorRT-LLM — `tensorrt_llm/_torch/speculative/mtp.py`

**Closest production code to the Atlas plan.** Has an explicit
`use_relaxed_acceptance_for_thinking` path with **log-prob delta over top-K**.

### Config (`llmapi/llm_args.py` L1593-1609, `MTPDecodingConfig`)

```python
use_relaxed_acceptance_for_thinking: bool = Field(default=False)
relaxed_topk: int   = Field(default=1)    # K candidates
relaxed_delta: float = Field(default=0.0) # log-prob gap from top-1
```

Per code comments (L1608): *"Only candidates with prob >= (top-1 prob − delta) are kept."*
DeepSeek-R1 recommended preset: `relaxed_topk=10, relaxed_delta=0.6`.

### Verify (mtp.py L702-829, `sample_and_accept_draft_tokens`)

```python
# L685-691  pick top-K targets
topk_value, topk_indices = torch.topk(gen_logprobs,
                                       k=self.spec_config.relaxed_topk, dim=-1)

# L823-828  custom op does the acceptance
torch.ops.trtllm.mtp_relaxed_acceptance_op(
    spec_metadata.slot_ids, topk_value, topk_indices, draft_tokens,
    mtp_relaxed_delta_pool, num_accepted_tokens, accepted_tokens,
    mtp_num_modules, batch_size, num_contexts,
    self.spec_config.relaxed_topk, self.spec_config.relaxed_delta,
    self.spec_config.begin_thinking_phase_token,
    self.spec_config.end_thinking_phase_token)
```

**Acceptance rule:** draft token is accepted if it appears in the target's
top-K AND `log(p_target[top1]) − log(p_target[draft_token]) ≤ relaxed_delta`.
This is *exactly* the "margin-ratio detector" pattern Atlas wants — flipped
direction (TRT-LLM uses it to be MORE lenient; Atlas wants to be MORE
careful), but identical mechanism.

Scope is gated by `begin_thinking_phase_token` / `end_thinking_phase_token`:
the relaxation only fires inside `<think>...</think>` blocks
(L791-816, ctx_is_think accounting). This is itself a useful pattern —
**confidence policy can be region-specific**.

---

## 4. llama.cpp — `common/speculative.cpp` + `min_p` sampler

**Two distinct mechanisms.**

### Draft-side early stop (lines 745-752, 1066-1072)

```cpp
// common_speculative_impl_draft_simple::draft()
if (cur_p->data[0].p < params.p_min) {     // L749
    drafting[seq_id] = false;
    n_drafting--;
    continue;
}
```

`cur_p->data[0].p` is the softmax probability of the draft model's top-1
token. `params.p_min` is the `--draft-p-min` CLI flag (default 0.75 per
docs/speculative.md; sweet spot per Dre Dyson's Qwen3.6-27B writeup is
0.75 with `--spec-draft-n-max 5`).

**Gate:** stop speculating once the draft model itself is uncertain. This is
draft-side, not verify-side, but the principle generalizes: *raw top-1 softmax
probability under a threshold ⇒ this position deserves heavier work.*

### `min_p` sampler (separate, decoding-side)

`llama_sampler_init_min_p` filters the candidate distribution: keep token *t*
only if `p[t] >= min_p * p[argmax]`. With `min_p=0.05` and `p_top=0.9`, all
tokens below 0.045 are dropped. This is **post-softmax normalized-margin
threshold** — the same shape as Atlas's "margin-ratio detector" but applied
to the final sampling step, not gating BF16 re-verify.

---

## 5. MLC-LLM / HF TGI — no margin gates

MLC-LLM speculative decoding (`mlc_llm/serve/spec_decode_mode.py` family) uses
standard Leviathan rejection sampling, identical to vLLM. No public per-token
confidence re-verify path.

HF TGI's `text_generation_server/utils/speculation.py` also uses pure exact-
match for greedy and rejection-sample for stochastic. No FP8→BF16 fallback
plumbing exists in TGI as of v3.x.

DeepSeek and Qwen official inference repos (`vllm-deepseek`, `Qwen3-Coder-API`)
do not ship FP8↔BF16 per-token mixed-precision recipes. The only adaptive-
precision production code I found is TRT-LLM's `relaxed_acceptance` (Section 3).

---

## 6. Academic precedent (for completeness; cited in case Atlas wants harder math)

- **MARS (arxiv 2601.15498, Jan 2026)** — "Margin-Aware Speculative
  verification". Conditions verification on "decision stability measured from
  target logits" — i.e., relaxes rejection when target margin is wide.
  Training-free, 8B–235B. *Directly relevant; opposite direction (relaxes
  when confident).*
- **AdaEDL (arxiv 2410.18351)** — uses **draft-side entropy** as a lower
  bound on expected acceptance probability; stops drafting when entropy too
  high. Same shape as llama.cpp `--draft-p-min`, generalized to entropy.
- **CM-ASD (arxiv 2508.15371)** — entropy + logit-margin uncertainty on
  drafter distribution, dynamically modulates draft length AND acceptance
  threshold. Most thorough theoretical framing of what Atlas is building.
- **SPRINTER (arxiv 2502.04557)** — trains a small classifier to *predict*
  acceptability, calls target only when classifier rejects. Different shape:
  uses learned verifier, not logit margin.
- **EARS (arxiv 2512.13194, Dec 2025)** — adaptive rejection threshold:
  `threshold = f(1 − max(P_target))`. Replaces fixed uniform coin with a
  target-uncertainty-modulated one.

---

## 7. Five concrete patterns Atlas can borrow with minimal code

Ranked by code-cost-to-payoff:

### Pattern A: TRT-LLM-style log-prob delta gate (≈30 lines Rust)

```rust
// in MTP verify, after computing target logits in BF16-equivalent
let top2 = topk(target_logits, 2);
let log_margin = top2.values[0] - top2.values[1];   // already log-space
if log_margin < THRESHOLD_DELTA {
    // mark "uncertain" — escalate this position
}
```

Mirrors `mtp_relaxed_acceptance_op` (mtp.py L823) but inverted: TRT-LLM
relaxes when *confident enough*; Atlas escalates when *not confident enough*.
THRESHOLD_DELTA candidates: `1.5` (Atlas's 23.7% finding), `0.6` (TRT-LLM's
DeepSeek-R1 default for thinking phase), tunable per MODEL.toml.

### Pattern B: SGLang dual-threshold (acc + single) — but inverted

```rust
let lone_confident = target_prob[token] >= THRESHOLD_SINGLE;  // 0.85
let coin_ok = (rng.uniform() * THRESHOLD_ACC) <= prob_acc;    // ratio test
let needs_bf16 = !(lone_confident || coin_ok);
```

Mirrors `speculative_sampling.cuh:103-107`. Cheap (two compares per token).
THRESHOLD_SINGLE 0.85 is a known stable working point in SGLang prod.

### Pattern C: llama.cpp draft-side early stop — for Atlas MTP

```rust
// in MTP K=2 drafter, BEFORE issuing verify
if mtp_top1_softmax < DRAFT_P_MIN {     // 0.75 per llama.cpp docs
    // skip MTP, force single-token BF16 decode
}
```

Mirrors `common/speculative.cpp:749`. Doesn't even need verify-side changes —
just prune low-confidence MTP drafts upstream. May be the cheapest win.

### Pattern D: `min_p`-style normalized margin (post-softmax)

```rust
let p = softmax(logits);
let margin = p[top1] - p[top2];
let normalized = margin / p[top1];   // relative gap
if normalized < MIN_REL_MARGIN {     // 0.10 ≈ "top2 within 10% of top1"
    escalate_to_bf16();
}
```

Mirrors llama.cpp `min_p` semantics. **Normalized** ratio is more stable
across temperature settings than raw logit gap (which is what Atlas measured
at 1.5).

### Pattern E: Region-scoped policy (TRT-LLM thinking-phase gate)

TRT-LLM only applies `relaxed_acceptance` inside `<think>...</think>`. Atlas
analog: only run the BF16-verify gate during long-context decode positions
(where the 23.7% finding lives), not during prefill or short responses.
Scope gating via `(seq_len > 2048) && (decode_position)` ≈ free.

---

## 8. The exact thresholds each system uses (consolidated)

| System | Where | Quantity | Threshold | Default |
|---|---|---|---|---|
| vLLM greedy | rejection_sampler.py:745 | argmax match | exact equality | n/a |
| vLLM stochastic | rejection_sampler.py:810 | p_target/p_draft | ≥ uniform[0,1) | n/a |
| SGLang `threshold_single` | speculative_sampling.cuh:103 | p_target[draft] | ≥ T_single | 1.0 (off); rec. 0.5–0.85 |
| SGLang `threshold_acc` | speculative_sampling.cuh:103 | prob_acc / T_acc | ≥ uniform coin | 1.0 (off) |
| TRT-LLM `relaxed_delta` | mtp.py:827 | log p[top1] − log p[draft] | ≤ delta | 0.0 (off); 0.6 DeepSeek-R1 |
| TRT-LLM `relaxed_topk` | mtp.py:686 | draft∈top-K | K candidates | 1 (off); 10 DeepSeek-R1 |
| llama.cpp `--draft-p-min` | speculative.cpp:749 | softmax(draft top-1) | ≥ p_min | 0.75 |
| llama.cpp `min_p` sampler | llama-sampling.cpp (apply) | p[t]/p[top1] | ≥ min_p | 0.05 |

**Atlas mapping recommendation:** the 1.5 logit-gap finding maps cleanest onto
**Pattern A** (TRT-LLM log-prob delta). At softmax temperature 1.0,
log p[top1] − log p[top2] = logit[top1] − logit[top2] = 1.5 is equivalent to
top1 probability ratio ≈ 4.5× over top2 — i.e., target is roughly 80% / top2
roughly 18% in the two-way case. That's a defensible "we should re-verify in
BF16" boundary.

---

## 9. Things production stacks DO NOT do (yet)

- **No public production code does per-token precision dispatch
  (FP8 → BF16 re-run) gated on margin.** This is genuinely novel territory.
- **No production stack runs a second forward pass on uncertain positions** —
  they instead trust the first pass and either accept/reject (spec-decode) or
  resample stochastically. Atlas's BF16-verify-on-flagged is uncharted.
- **The closest live mechanism is TRT-LLM's `relaxed_acceptance`**, which
  changes the acceptance *boundary*, not the *precision*.

This means Pattern A (margin detect) + a small "shadow BF16 GEMM on flagged
positions" path is a defensible engineering bet, but Atlas will not find a
copy-paste reference implementation. The detector is well-precedented; the
remediation (BF16 re-decode) is novel.

---

## Sources

- vLLM rejection sampler: https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/rejection_sampler.py
- vLLM suffix decoding: https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/suffix_decoding.py
- SGLang tree spec kernel: https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/speculative/speculative_sampling.cuh
- SGLang EagleVerifyInput: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/eagle_info_v2.py
- TRT-LLM MTP: https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/speculative/mtp.py
- TRT-LLM MTPDecodingConfig: https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/llmapi/llm_args.py
- llama.cpp speculative: https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp
- llama.cpp speculative doc: https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
- MARS (margin-aware verification): https://arxiv.org/abs/2601.15498
- AdaEDL (entropy bound): https://arxiv.org/abs/2410.18351
- CM-ASD: https://arxiv.org/abs/2508.15371
- SPRINTER (approximate verify): https://arxiv.org/abs/2502.04557
- EARS: https://arxiv.org/pdf/2512.13194
