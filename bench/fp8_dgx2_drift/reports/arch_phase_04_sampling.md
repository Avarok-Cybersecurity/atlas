# Phase 04 — Logit Post-Processing & Sampling Pipeline: vLLM vs Atlas

**Scope:** every transformation between raw model-output logits (FP32 cast)
and the chosen next-token id. Focus on Qwen3.6-A3B-FP8 path served via
Atlas vs vLLM-V1 (`vllm/v1/sample/sampler.py`).

## File anchors

### vLLM (Python, GPU tensor ops, batched)
- Driver: `vllm/v1/sample/sampler.py` (`Sampler.forward` / `sample`)
- Top-k/top-p/min-p: `vllm/v1/sample/ops/topk_topp_sampler.py`,
  `vllm/v1/sample/logits_processor/builtin.py`
- Penalties: `vllm/v1/sample/ops/penalties.py` →
  `vllm/model_executor/layers/utils.py::apply_penalties`
- Bad-words / allowed-mask: `vllm/v1/sample/sampler.py::apply_logits_processors`
- Grammar bitmask: applied **outside** sampler in
  `vllm/v1/worker/gpu_model_runner.py:2675` immediately before `_sample`,
  via `vllm/v1/structured_output/utils.apply_grammar_bitmask`
- Defaults: `vllm/sampling_params.py:151-173`

### Atlas (Rust, CPU on host f32 buffer, per-sequence)
- Pre-sample composable pipeline: `crates/spark-server/src/scheduler/logit_processors/mod.rs`
  (9 stages, see `run_pipeline`)
- Big monolithic pre-sample block: `crates/spark-server/src/scheduler/decode_logits_seq.rs::process_seq_logits`
- Core sampling pipeline: `crates/spark-runtime/src/sampler/sample_impl.rs::sample_with_params_seeded`
- `SamplingParams` shape + defaults: `crates/spark-runtime/src/sampler.rs:44-130`
- Adaptive sampler: `crates/spark-server/src/adaptive_sampler.rs`
- Whitespace / attractor masks: `crates/spark-server/src/whitespace_mask.rs`,
  `crates/spark-server/src/attractor_mask.rs`

---

## Master pipeline comparison

The ordering below tracks "logically what each engine does, in order, to a
single sequence's logit row". For Atlas the pre-sample pipeline runs on
host f32 (after BF16→FP32 expansion or direct FP32 logits); for vLLM it
runs on a `[B, V]` GPU tensor.

| Order | vLLM transform | Atlas transform | Notes / divergence |
|---|---|---|---|
| 0 | Raw model logits (BF16/FP32) on GPU | BF16/FP32 logits D2H copied to host then expanded to f32 (`decode_logits_seq.rs:53-70`) | vLLM keeps everything on GPU. Atlas pays D2H + f32 expansion per step; gives Atlas freedom for arbitrary Rust transforms. |
| 1 | `logits.to(float32)` (`sampler.py:87`) | f32 already (CPU buffer) | Same precision domain entering the pipeline. |
| 2 | (optional) `logits.clone()` for `raw_logprobs` snapshot (`sampler.py:84`) | None (logprob extraction reads f32_logits directly at the end, `decode_logits_seq.rs:683`) | vLLM uses pre-mutation logits for `raw_logprobs`; Atlas uses post-grammar-mask logits. Logprobs reported to the user are NOT comparable. |
| 3 | `allowed_token_ids_mask` → `-inf` (`sampler.py:285`) | — | vLLM-only. Atlas has no equivalent caller-side allowlist mask. |
| 4 | `bad_words` exclusion (`ops/bad_words.py`) | — | vLLM-only. Atlas has `logit_bias` (general) but no first-class bad-words. |
| 5 | **non_argmax_invariant** logits processors:<br>• `MinTokensLogitsProcessor` (mask stop tokens until min_tokens reached)<br>• `LogitBiasLogitsProcessor` (`logits[slice] += bias`) | **F2-confidence early-stop arm** (`decode_logits_seq.rs:91-116`) — sets `force_end_thinking` after 30 consecutive p>=0.95 tokens (read-only on logits) | Both engines run a "non-argmax-invariant" group here, but Atlas's contents are heavily reasoning-mode aware (no analog in vLLM). |
| 6 | — | **Mid-word `</think>` mask** (`:136-147`) — `-inf` on `think_end` if previous token decoded mid-word | Atlas-only. Direct drift mitigation: FP8 noise was biasing `</think>` up at low-margin word boundaries. |
| 7 | — | **Post-close `</think>` + `<think>` mask** (`:151-178`, F9) — once thinking ended, hard-mask both think tokens for the rest of the response | Atlas-only. Production-repetition mitigation. |
| 8 | — | **Tool-call-during-thinking mask** (`:188-202`) — `-inf` on tool-call-start while inside `<think>`, `-12.0` bias when tool-loop detected | Atlas-only. KV-contamination preventer. |
| 9 | — | **Forced `</think>` blanket-mask injector** (`:241-265`) — when budget exhausted + sentence boundary, set all logits to `-inf` except `</think>=0.0` | Atlas-only. Hard injection (post-policy). |
| 10 | — | **One-shot pin-to-tool-call-start** (`:267-290`, change 3b) — same blanket-mask trick to force `<tool_call>` after `</think>` when `require_tool_call` | Atlas-only. |
| 11 | — | **Forced-token fast path** (`:344-356`) — when grammar admits a single legal token, early-return that token (skips sampling entirely) | Atlas-only optimization. xgrammar Tier 3b. |
| 12 | **Grammar bitmask** applied OUT-OF-SAMPLER in `gpu_model_runner.py:2675` BEFORE `_sample` (xgrammar/outlines) | **Grammar bitmask** applied IN pipeline at `:365-370`; skipped while `inside_thinking` | Different placement: vLLM applies BEFORE the sampler entry; Atlas applies AFTER its detector/bias stack. **Order matters: Atlas's logit_bias and rep_penalty run AFTER grammar mask, which can re-rank inside the still-allowed set, but cannot un-mask. vLLM's penalty/min-tokens/logit-bias run BEFORE grammar — same outcome on which tokens survive, but the bias values that "win" can differ at low margin.** |
| 13 | — | **AdaDec diagnostic** entropy log (`:377-381`) | Read-only diagnostic. Atlas-only. |
| 14 | — | **Adaptive sampler** zone/entropy/LZ tracking + effective_temperature computation (`:392-412`) | Atlas-only. Behind `--adaptive-sampling`; can OVERRIDE temperature and force greedy via `should_use_greedy`. |
| 15 | — | **WS1/AM1/Tier-1 param-body pos-0 mask** (`:483-515`) — push `(token_id, -8.0)` for `</`, all ~440 whitespace tokens, and `lean://`-prefix attractor tokens into `logit_bias_local` | Atlas-only. Direct FP8-drift attractor suppression. |
| 16 | — | **WS2 mid-content whitespace gate** (`:516-543`) — `-3.0` bias on whitespace tokens when previous token ends in a digit | Atlas-only. Drift mitigation for `0.1.0`→`0.1 .0`. |
| 17 | — | **A4 POST_THINK_MIN_REASONING floor** (`:557-564`) — `-8.0` bias on `</think>` until 16 thinking tokens emitted | Atlas-only. |
| 18 | — | **B1 margin-ratio scan** (`:586-610`) — O(V) scan for top-1/top-2 + margin, periodic WARN when low-margin rate > threshold | Read-only detector. Atlas-only. |
| 19 | — | **C4v1 top-2 lift** (`:632-642`) — currently DISABLED (`false &&`) | Atlas-only, intentionally inert. |
| 20 | — | (CALL INTO `sample_with_params_history` — Atlas pipeline below starts now) | The pre-sample work above is in `process_seq_logits`; the sampler proper starts here. |
| 21 | (penalty step, inside sampler) **Repetition penalty** via `apply_repetition_penalties` C++ op (`utils.py:88`): for tokens in prompt_mask ∪ output_mask, `if logit>0: /=rep_pen else: *=rep_pen` | **Windowed repetition penalty** (`sample_impl.rs:51-68`) — same formula, but with caller-configurable `repetition_penalty_window` (vLLM has no window) | **Diff #1 — windowing.** Atlas can cap penalty history to N most recent tokens; vLLM uses full prompt + output. **Diff #2 — guard:** Atlas short-circuits when `rep_pen == 1.0` AND when `rep_pen <= 0.0` (treats 0 as no-op); vLLM applies regardless (rejected at param-validation, `sampling_params.py:440`). **Diff #3 — exemption:** F8 stop-token exemption in MEMORY is NOT in current code; both engines penalize EOS by default. |
| 22 | **Frequency penalty** `logits -= freq_pen * output_bin_counts` (`utils.py:92`) | **Frequency penalty** `raw_logits[tid] -= freq_pen * count` (`sample_impl.rs:73-92`) | Atlas window-aware; vLLM all-history. |
| 23 | **Presence penalty** `logits -= pres_pen * output_mask` (`utils.py:93`) | **Presence penalty** combined into the same loop as freq (`sample_impl.rs:89`) | Equivalent formula. Atlas window-aware. |
| 24 | — | **LZ penalty** (`sample_impl.rs:94-97`, `sampler.rs::apply_lz_penalty`) — penalize tokens that would extend repeated n-gram patterns; arXiv:2504.20131 | Atlas-only. |
| 25 | — | **DRY penalty** (`:99-109`, `sampler.rs::apply_dry_penalty`) — exponential penalty for extending repeated sequences (llama.cpp PR #9702) | Atlas-only. |
| 26 | — | **Logit bias** additive per-token (`:111-116`) | vLLM has this too but as a SEPARATE non-argmax-invariant processor (step 5); Atlas applies it AFTER penalties. Order divergence: vLLM bias → penalties; Atlas penalties → bias. **At low margin under FP8 drift this re-orders what survives.** |
| 27 | — | **Greedy bypass: temperature <= 0.0** → `argmax(raw_logits)` and RETURN (`sample_impl.rs:125-132`) | Atlas-only fast path. vLLM uses `greedy_sample` in `sampler.sample` but only AFTER applying `argmax_invariant` processors and STILL pays the full transform for the `random_sampled` parallel path (it computes both then `torch.where`s). |
| 28 | — | **Top-n-sigma filter** (`sample_impl.rs:135-150`) — keep tokens with logit >= mean − n·sigma (NVFP4 noise filter) | Atlas-only. Temperature-invariant pre-temperature filter. |
| 29 | **Temperature scale** (`sampler.py:138`): `logits.div_(temp)` (skipped on greedy) | **Temperature scale** (`sample_impl.rs:152-158`): per-finite-token `v / temperature` into Vec<(idx, scaled)> | Same arithmetic. Atlas eagerly drops `-inf` from top-n-sigma here; vLLM has no top-n-sigma to drop. |
| 30 | **argmax_invariant processors** (`sampler.py:180-181`): just `MinPLogitsProcessor` by default | Sort descending + top-k truncate (`sample_impl.rs:170-174`) | vLLM puts min-p HERE (before top-k/top-p) and computes probs internally; Atlas defers min-p to step 32. |
| 31 | **Min-p filter** (`builtin.py:100-114`) — `probs = softmax(logits); thresh = min_p * max(probs); logits[probs<thresh] = -inf` | (no min-p yet — happens at step 32) | **Diff #4 — placement & semantics.** vLLM computes softmax JUST for the min-p mask, then applies top-k/top-p on the masked LOGITS (re-softmax inside top-p). Atlas applies min-p AFTER softmax on the already-truncated set. Numerically these can differ when (top-k AND min-p) both active. |
| 32 | **Top-k** (`topk_topp_sampler.py:152-217`) — sort or topk-only path, set non-top-k logits to `-inf` | (Top-k already applied in step 30 by truncating the sorted Vec) | Same effect, different code path. vLLM's `apply_top_k_only` avoids full sort; Atlas always sorts. |
| 33 | **Top-p** — `probs = softmax(logits_sort); cumsum; mask where cumsum<=1−p; logits[mask]=-inf` (`topk_topp_sampler.py:181-188`) | **Softmax** then **Min-p** then **Top-p** (`sample_impl.rs:176-219`) — softmax on sorted truncated set with `(l−max).exp()` normalization, then `probs.retain(p>=min_p*max_prob)`, then cumulative-prob nucleus | **Order differs.** vLLM: min-p → top-k → top-p (all on logits, with softmax done twice). Atlas: top-k → softmax → min-p → top-p (all on probs, softmax done once). **Probabilities can differ when min-p AND top-p are both set: the renormalization basis is different.** |
| 34 | **Entropy record** — vLLM does not record entropy on the post-filter distribution by default (logprobs path exists but is sampler-output, not metric) | **Entropy record** (`sample_impl.rs:183-197`) — `H = -Σ p·ln(p)` recorded to `LAST_ENTROPY`/`LOW_ENTROPY_TOKENS` atomics | Atlas-only telemetry. |
| 35 | **Random sample** — `probs.div_(q).argmax(dim=-1)` where `q.exponential_()` (Gumbel-max trick, `random_sample`) OR FlashInfer `top_k_top_p_sampling_from_logits` rejection sampler | **Multinomial sample** (`sample_impl.rs:221-239`) — `threshold = U[0,1)*sum; cumsum until cumsum>=threshold; return idx` with optional seeded `StdRng` | Both unbiased random. vLLM's Gumbel-max is GPU-batched and matches FlashInfer's rejection sampler statistically. Atlas's CPU multinomial is per-sequence. **Determinism:** Atlas advances seed per-token (`step_seed = a.seed + len(output_tokens)`); vLLM uses per-request `torch.Generator`. |

---

## Atlas-only patches (not present in vLLM, all observed in the Phase-04 trace)

| Atlas patch | Site | Effect at logit time | Drift hypothesis |
|---|---|---|---|
| F2 confidence early-stop arm | `decode_logits_seq.rs:91-116` | Sets a flag (no logit change) | Thinking-loop watchdog |
| Mid-word `</think>` mask | `:136-147` | `-inf` on `</think>` | FP8 biases `</think>` up at low-margin word boundaries → masks it |
| Post-close `</think>`/`<think>` mask | `:151-178` | `-inf` on both | Production-repetition mitigation |
| Tool-call-during-thinking mask | `:188-202` | `-inf` or `-12.0` on `<tool_call>` | KV contamination prevention |
| Forced `</think>` blanket-mask | `:241-265` | All other logits `-inf`, `</think>=0` | Forces close at policy-defined boundary |
| One-shot pin-to-tool-call-start | `:267-290` | All other logits `-inf`, tool-call-start = 0 | MiniMax M2 wander-prevention |
| Forced-token fast path | `:344-356` | Returns immediately, skips sampling | xgrammar Tier 3b throughput |
| AdaDec diagnostic | `:377-381` | Read-only entropy log | Diagnostic |
| Adaptive sampler temp/greedy override | `:396-412` | Overrides temperature OR forces greedy | Zone-aware temperature scheduling |
| WS1 / AM1 / close-tag mask | `:483-515` | `logit_bias[id] = -8.0` on `</`, ~440 WS tokens, `lean://` attractors | Direct FP8-drift attractor suppression at param-body pos 0 |
| WS2 mid-content whitespace gate | `:516-543` | `-3.0` bias on WS tokens after digit-ending token | FP8 `0.1.0`→`0.1 .0` mitigation |
| A4 POST_THINK_MIN_REASONING floor | `:557-564` | `-8.0` bias on `</think>` for first 16 thinking tokens | Reasoning-collapse cascade |
| B1 margin scan | `:586-610` | Read-only top-1/top-2 + margin metric | Diagnostic |
| C4v1 top-2 lift | `:632-642` (disabled) | Would push top-2 logit up at low margin | DISABLED — introduced own drift mode |
| Top-n-sigma | `sample_impl.rs:135-150` | `-inf` on tokens below mean − n·sigma | NVFP4 quantization noise filter |
| LZ penalty | `:94-97` | Extends rep penalty to n-gram extensions | Loop prevention |
| DRY penalty | `:99-109` | Exponential penalty on repeated subsequences | Loop prevention |
| Greedy bypass after penalties | `:125-132` | argmax(raw_logits) and return | Throughput on temperature=0 |
| Entropy metric record | `:183-197` | Records H to global atomics | Telemetry |
| Per-token seed advance | `:434` | `seed + len(output_tokens)` | Atlas-specific determinism scheme |
| POST_THINK_MIN_CONTENT EOS guard | `decode_logits_step.rs:372-376` | Adds EOS to `suppress_eos` until 16 post-think tokens | EOS-suppression (acts on emission, not logits, but in the same flow) |

## vLLM-only transforms (not present in Atlas)

| vLLM transform | Site | Notes |
|---|---|---|
| `allowed_token_ids_mask` | `sampler.py:285` | Per-request token allowlist (`-inf` for everything else). |
| `bad_words` exclusion | `ops/bad_words.py` | First-class bad-words token sequence support. |
| `MinTokensLogitsProcessor` | `builtin.py:166+` | Masks STOP tokens (including EOS) until `min_tokens` reached. Atlas has `min_tokens` but applies it as `min_tokens_suppresses` in the post-sample emission gate (`decode_logits_step.rs:344`), not as a logit mask. **Functionally equivalent outcome but at a different stage.** |
| FlashInfer rejection sampler | `topk_topp_sampler.py:244-290` | Optional GPU-resident rejection sampling for top-k/top-p (env `VLLM_USE_FLASHINFER_SAMPLER`). |
| `LogitBiasLogitsProcessor` as separate stage | `builtin.py:117-163` | Same arithmetic as Atlas's logit_bias, but placed BEFORE penalties in vLLM vs AFTER in Atlas. |

---

## Critical-question answers

### 1. Order of operations

**vLLM order:**
1. `allowed_token_ids_mask` (`-inf`)
2. `bad_words` (`-inf`)
3. `MinTokensLogitsProcessor` (mask STOP tokens to `-inf`)
4. `LogitBiasLogitsProcessor` (additive bias)
5. Repetition penalty → frequency penalty → presence penalty
6. *(out-of-sampler, before `_sample`)* Grammar bitmask (`-inf`)
7. Greedy split: `argmax` for temp≈0 path is computed
8. Temperature scale
9. `MinPLogitsProcessor` (`-inf`)
10. Top-k (`-inf`) → Top-p (`-inf`)
11. softmax + Gumbel-max sample

**Atlas order:**
1. F2 confidence arm (no-op on logits)
2. Mid-word/`</think>`/think-end masks (`-inf`)
3. Tool-call-during-think masks (`-inf` or `-12`)
4. Forced-`</think>` blanket-mask (`-inf` + 0.0 on target)
5. Pin-to-tool-call-start blanket-mask
6. Forced-token fast-path (early return)
7. Grammar bitmask (`-inf`)
8. AdaDec diagnostic (no-op)
9. Adaptive temperature/greedy override (modifies `sampling_temp`)
10. WS1/AM1/Tier-1/WS2/POST_THINK_MIN_REASONING populate `logit_bias_local` (-8.0 / -3.0)
11. B1 margin scan (no-op)
12. **Enter sampler:**
13. Repetition penalty (windowed) → freq → presence → LZ → DRY
14. **Logit bias additive** (AFTER penalties — order divergence)
15. Greedy bypass (temp<=0): argmax → return
16. Top-n-sigma (`-inf`)
17. Temperature scale
18. Sort descending + top-k truncate
19. Softmax (single-shot, with `−max` shift)
20. Entropy record
21. Min-p (on PROBS)
22. Top-p (cumsum on probs)
23. Multinomial sample

**Material divergences:**
- **Grammar bitmask placement:** vLLM applies it OUTSIDE the sampler, AFTER penalties+bias. Atlas applies it BEFORE penalties+bias. Survival set is the same; the bias `-8.0` value applied AFTER `−inf` is a no-op on Atlas, but the rep_penalty multiplier applied to an `−inf` logit produces `−inf` (correct) in both engines.
- **Logit bias vs penalty:** vLLM does `bias → penalty`. Atlas does `penalty → bias`. For surviving tokens this re-orders.
- **Min-p basis:** vLLM computes min-p on probs derived from full-vocab logits (then masks logits and re-softmaxes inside top-p). Atlas computes min-p on probs from the already-top-k-truncated set. **At Qwen3.6 vocab=151k, this is a substantive difference** when top-k is set.

### 2. Default values

| Parameter | vLLM default (`sampling_params.py`) | Atlas default (`SamplingParams::greedy`/server `ActiveSeq`) | Match? |
|---|---|---|---|
| `temperature` | 1.0 | 0.0 in `SamplingParams::greedy`; per-request via `ActiveSeq::temperature` (seeded from request, falls back to MODEL.toml) | Diverges. **When the caller omits `temperature`, vLLM gives 1.0 (stochastic), Atlas gives whatever MODEL.toml says, usually 0.7-1.0 for prose, or 0.0 in pure-greedy harnesses.** |
| `top_p` | 1.0 | 1.0 in `greedy()`; per-request | Match. |
| `top_k` | 0 (disabled) | 0 (disabled) | Match. |
| `min_p` | 0.0 | 0.0 in `greedy()`; some MODEL.toml entries set 0.05-0.1 | Match at the parameter level. MEMORY's "min-p floor 0.08 on FP8" is MODEL.toml-driven, not a sampler default. |
| `repetition_penalty` | 1.0 | 1.0 in `greedy()` | Match at default. MODEL.toml drops it to 1.0 for prose categories per memory; Coder uses 1.1. |
| `presence_penalty` | 0.0 | 0.0 | Match. |
| `frequency_penalty` | 0.0 | 0.0 | Match. |
| `top_n_sigma` | (not present) | 0.0 | Atlas-only param. |
| `lz_penalty` / `dry_*` | (not present) | 0.0 / 1.75 / 2 | Atlas-only. |

### 3. `rep_penalty == 1.0` short-circuit

- **Atlas:** YES — `if rep_penalty != 1.0 && rep_penalty > 0.0 && !token_history.is_empty()` (`sample_impl.rs:51`).
- **vLLM:** NO short-circuit on `1.0`; the `apply_repetition_penalties` C++ op runs unconditionally when penalties are not all-default. There IS a global skip via `sampling_metadata.no_penalties` (`sampler.py:305`) when ALL penalty values are at their defaults across the batch. So vLLM batches with at least one non-default sequence run the kernel for everyone. Functionally `rep_penalty == 1.0` is a no-op (multiplying/dividing by 1.0); the difference is CPU/GPU work, not numerics.

### 4. Min-p mechanism

Both engines implement the SAME min-p formula `keep if p_i >= min_p * max(p)`. Differences:
- **vLLM:** computed on softmax of FULL-vocab (post-penalty) logits. Mask is applied to LOGITS (`-inf`). Top-p then RE-softmaxes inside the surviving set.
- **Atlas:** computed on the softmax of the ALREADY-TOP-K-TRUNCATED set. The `max_prob` basis is therefore the top-1 of the truncated set (same token, but the normalization differs if there were `-inf` tokens dropped earlier by top-n-sigma).

So the FLOOR is the same shape but applied at different basis points. **At Qwen3.6 with top-k unset and top-n-sigma=0, behavior matches.** With top-k set or top-n-sigma>0, vLLM and Atlas produce different surviving sets at the same `min_p`.

### 5. XGrammar mask order

- **vLLM:** grammar bitmask is the LAST mask before `_sample`. Logit bias and penalties have already run on full vocab; the grammar then zeroes (`-inf`) anything not legal. Top-k/top-p run on the post-mask logits, so they pick from the grammar-legal set only.
- **Atlas:** grammar bitmask is applied in the PRE-SAMPLE block (`decode_logits_seq.rs:365-370`), BEFORE the sampler is even called. The sampler then sees `-inf` on illegal tokens and processes the rest. `top_n_sigma` and `temperature` running over a vector with `-inf` cells is fine (the `is_finite` filter in `sample_impl.rs:155` drops them). **One subtle hazard: Atlas's softmax computes `(logit − max).exp()` so `-inf − max = -inf → exp = 0`, correct. The entropy record then ignores zero-prob tokens via the `if p > 1e-10` guard. Numerically safe.**

### 6. Are Atlas-specific detectors MASKING the drift?

This is the key question for the FP8 drift mission. My read of each:

| Atlas patch | Masking or Fixing? | Evidence |
|---|---|---|
| Mid-word `</think>` mask | **MASKING** | The patch literally exists because FP8 biases `</think>` upward at low-margin word boundaries (per the in-code comment at `decode_logits_seq.rs:118-135`). It does not address the kernel-level cause; it suppresses the symptom at one token id. |
| Post-close `</think>`/`<think>` mask | Defensible (production-repetition is a real failure mode independent of drift) | This is a vLLM-conventional production safety; not strictly drift-related. |
| WS1 / AM1 close-tag + 440-WS-token bias | **MASKING** | "FP8 noise routinely flips the first-content token to `lean` (id 2588) or ` lean` (id 15192)" — the comment in-code is an explicit acknowledgment that we're papering over a kernel bug. -8.0 bias is a sledgehammer; the underlying drift remains. |
| WS2 digit-ending whitespace gate | **MASKING** | Same diagnosis: -3.0 bias to flip a low-margin FP8 selection. |
| A4 POST_THINK_MIN_REASONING floor | Mixed: a length-floor IS a real production policy (DeepSeek-R1 ships one), but the trigger to add it here was "reasoning collapse cascade", which IS the drift. So it's a defensible policy that ALSO masks drift. |
| B1 margin-ratio detector | Diagnostic only — does not mask anything. Useful for the investigation. |
| C4v1 top-2 lift | DISABLED. Was an attempted MASK; backed out. |
| Top-n-sigma | **MASKING (by design)** — explicitly described in `sampler.rs:55-58` as an NVFP4 quantization-noise filter. |
| Adaptive sampler greedy gate | **POTENTIALLY AMPLIFYING** — it forces greedy when top-1 prob clears 0.9, but FP8 drift CAUSES top-1 to be wrong. By going greedy at the moment the kernel is least trustworthy, the gate locks in the bad selection. |
| LZ / DRY penalties | Loop-prevention, not drift. Defensible. |
| Post-think EOS guard | Production policy (R1 ships equivalent) — defensible. |

**Conclusion:** at least six of the Atlas patches (mid-word mask, WS1, AM1, WS2, top-n-sigma, and arguably A4) are explicitly drift-suppressors that hide the kernel-level FP8 issue at the sampler. The adaptive greedy gate (when `--adaptive-sampling` is on) likely AMPLIFIES drift in long-context decode by going greedy at the lowest-margin positions.

---

## Assessment: which Atlas patches mask drift vs fix it

### Masking the drift (sampler is papering over a kernel bug)

1. **Mid-word `</think>` mask** — pure symptom suppression. Confirmed by in-code comment.
2. **WS1 (440 whitespace tokens at param-body pos 0, -8.0)** — pure symptom suppression. The mask exists specifically because FP8 biases the wrong token up.
3. **AM1 (`lean://`-prefix attractor mask)** — pure symptom suppression. Same root cause.
4. **WS2 (digit-ending mid-content WS gate, -3.0)** — pure symptom suppression. Targets the Tier A drift mode (`0.1.0`→`0.1 .0`) directly.
5. **Top-n-sigma filter (`top_n_sigma > 0`)** — explicitly an NVFP4 quantization noise filter per the parameter doc. Doesn't fix the noise, drops tokens that fall outside its expected distribution.
6. **C4v1 top-2 lift (disabled)** — was an explicit attempt to undo FP8 flips by lifting top-2.

### Amplifying or risking amplification

7. **Adaptive-sampler greedy gate** (`should_use_greedy` when top-1 prob >= 0.9) — at FP8 long-context, low-margin positions account for 23.7% of decode positions (per `research_C1_results.md`); greedy at exactly those moments **commits** to whichever side FP8 noise pushed. With `--adaptive-sampling` unset, this is harmless; with it on, this is a drift amplifier.
8. **Logit-bias-after-penalty ordering** — when the WS1 -8.0 bias is large vs the rep_pen multiplicative effect, ordering them in the opposite of vLLM means the rep_pen sees pre-bias logits and the bias sees post-rep_pen logits. At extreme bias values the effect is dominated by the bias; at typical values both effects compose. Not a "amplifier" so much as a "deterministically-different" behavior.

### Defensible (production policy, not drift-related)

9. **F2 confidence early-stop**, **post-close think mask**, **tool-during-think mask**, **forced `</think>`**, **pin-to-tool-call-start** — these are standard reasoning-mode production policies (similar logic in DeepSeek-R1, s1, Qwen3 reports).
10. **POST_THINK_MIN_REASONING / POST_THINK_MIN_CONTENT** — length floors are production policy. The fact that drift triggered the need is incidental.
11. **LZ / DRY** — loop prevention, orthogonal to drift.
12. **B1 margin-ratio** — read-only diagnostic.

### Fixing nothing (vestigial)

13. **F8 stop-token rep-penalty exemption** mentioned in MEMORY does not appear in the current sampler. If it ever shipped, it is now gone. The current implementation applies rep_penalty to every token in the history window including EOS/stop tokens.

---

## Implications for the FP8 drift investigation

1. **The sampler is doing the job the kernels SHOULD be doing.** WS1+AM1+WS2+top-n-sigma together patch ~440 + a handful + ~80 tokens (WS+attractor+top-n-sigma quantile), which is a meaningful chunk of Qwen3.6's vocab. Removing these patches and observing the rate at which the sampler-suppressed tokens win the argmax would give a direct measurement of FP8 drift impact at the sampler level.
2. **vLLM has none of this and still works on BF16/AWQ-INT4.** The fact that Atlas needs 7+ drift-specific sampler patches that vLLM doesn't is itself evidence the kernel-level numerics are different.
3. **The adaptive greedy gate should be disabled for the drift investigation.** It deterministically commits to whichever side FP8 noise pushed at low-margin positions — the exact regime the bench harness is trying to characterize.
4. **The ordering divergences (grammar-before-penalty, bias-after-penalty, min-p-after-top-k) are unlikely to be the drift root cause** but they DO make Atlas's output distribution non-identical to vLLM's even on BF16. Any A/B comparison must control for this.
5. **An equivalent-to-vLLM `--no-drift-patches` mode would be useful** — disable WS1, AM1, WS2, top-n-sigma, mid-word mask, A4 floor, adaptive greedy gate, and the post-penalty bias re-ordering, to isolate what the kernels actually emit.
