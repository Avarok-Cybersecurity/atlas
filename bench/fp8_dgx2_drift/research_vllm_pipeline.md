# vLLM v1 Post-Sample → SSE Pipeline (Map + Atlas Gap Analysis)

**Scope.** Trace one token's journey from sampler output to the SSE
chunk hitting the client in vLLM v1 `main` (2026-05-25), then identify
the gaps in Atlas that map onto the Qwen3.6-FP8 opencode drift mode
(empty `<parameter=KEY></parameter>` bodies, Cargo.toml losing `=`
signs, `axum-v51` → `axums_v51`, `which cargo` loops, schema type
errors).

Primary vLLM files inspected:

- `vllm/v1/sample/sampler.py`, `vllm/v1/sample/logits_processor/state.py`
- `vllm/v1/structured_output/utils.py` (`apply_grammar_bitmask`)
- `vllm/v1/spec_decode/{eagle.py,ngram_proposer.py,...}`
- `vllm/v1/engine/detokenizer.py` (`FastIncrementalDetokenizer`,
  `SlowIncrementalDetokenizer`)
- `vllm/tool_parsers/hermes_tool_parser.py`
- `vllm/tool_parsers/utils.py` (`partial_tag_overlap`,
  `extract_intermediate_diff`, `compute_tool_delta`,
  `find_common_suffix`)
- PRs **40785** (Qwen3-Coder fragmented-tag + speculative-decoding fix,
  2026-04), **31778** (stream_interval>1 bug, 2026-05), **32517 /
  37098** (parameterless `{}` args, 2026-05), **32518** (lost closing
  brace), **34101** (stop-string truncation), **33965** (duplicate
  names + `</parameter>` leak), **38973** ($ref / oneOf typing,
  2026-05).

---

## Step 1 — Logits processors (`v1/sample/logits_processor/state.py`)

`LogitsProcessors` partitions registered processors into
`argmax_invariant` and `non_argmax_invariant` lists; per-step the
sampler applies the non-argmax-invariant block before the temperature
gate and the argmax-invariant block (e.g. `min_p`) after. State
transitions across decode steps are aggregated through
`BatchUpdateBuilder` (`removed_append`, `added`, `moved`) and a single
`get_and_reset()` produces a frozen `BatchUpdate` per step — this is
how a request can join/leave the persistent batch without
re-instantiating processors.

**Atlas gap.** `decode_logits_seq.rs` runs an *ad-hoc* sequence
of masks/biases (mid-word `</think>` mask, force-think-end,
require_tool_call pin-to-start, `tool_call_start` suppression,
`adaptive`) followed by `sample_with_params_history`. No
argmax-invariant separation; therefore the forced-token fast path
correctly bypasses sampling, but greedy / temp-0 paths still pay the
full DRY / repetition / presence cost. More importantly, processors
are not composable with a `BatchUpdate`-style add/remove cycle, so
turning *one* request into a "tool-only" pinned state requires the
mutable `a.require_tool_call` + `a.suppress_tool_call` ad-hoc fields
that already exist; new processors require editing the central
function (cost-of-change ≈ vLLM's pipeline order).

## Step 2 — Grammar bitmask (`v1/structured_output/utils.py`)

vLLM keeps the bitmask GPU-resident:
`xgr.apply_token_bitmask_inplace(logits, grammar_bitmask, indices=index_tensor)`
operates **in place on the device logits tensor**. The CPU path is
only a fallback for CPU devices and includes the float32 round-trip
for old xgrammar kernels. Bitmasks for speculative tokens are
reordered to match GPU batch layout via
`scheduler_output.scheduled_spec_decode_tokens` + cumulative offsets.

**Atlas gap.** `decode_logits_seq.rs:330` calls
`gs.apply_bitmask_to_logits(&mut f32_logits)` after **every active
sequence** has already had its full-vocab logits expanded BF16→FP32 on
the CPU at lines 24–41. For a single grammar-active request that's a
~262k × 4-byte CPU pass per token; with `--tp 1` and one
grammar-active stream this is fine for latency but it imposes a
~300 µs floor that *re-introduces drift* by forcing FP8 dequant to go
through an intermediate FP32 round, then back to FP32 for sampling.
Atlas has no GPU-side `apply_token_bitmask_inplace` equivalent —
fallback is the only path. Direct consequence for the FP8 drift mode:
when xgrammar is *off* (current opencode runs), the entire content
phase has no structural constraint *and* the model is sampled from
CPU-rounded logits — small magnitude bias from FP8 weight dequant
flips low-margin token choices that drive the `axum-v51` →
`axums_v51`, missing-`=` Cargo.toml lines.

## Step 3 — Sampler (`v1/sample/sampler.py`)

`Sampler.forward()`:
1. (optional) extract raw logprobs
2. cast to fp32
3. `apply_logits_processors()` — allowed_token_ids mask,
   `apply_bad_words()`, non-argmax-invariant processors,
   `apply_all_penalties()`
4. `sample()` — greedy/random mix via `torch.where()` on
   `(temperature < _SAMPLING_EPS)`, with greedy/argmax computed first
   so the greedy tensor is reused as the output buffer for the
   `where()` to avoid an extra allocation.

No `.item()` in the hot loop; only `pin_memory=True` + `non_blocking`
transfers for logprob CPU staging.

**Atlas gap.** `process_seq_logits` is per-sequence and entirely on
host. Each sequence pays an O(V) CPU expansion + an O(V) CPU bitmask
fill + an O(V) sampling pass. For Qwen3.6 (151,936 vocab) this is the
single biggest non-MoE CPU cost per token. Worth noting: this also
makes Atlas's `forced_token_fastpath_enabled` *load-bearing* — that's
the only place an O(V) pass is avoided.

## Step 4 — MTP / EAGLE / NGram rollback (`v1/spec_decode/`)

`SpecDecodeBaseProposer` owns the propose/verify boundary. Files like
`eagle.py`, `gemma4.py`, `medusa.py`, `ngram_proposer.py`,
`dflash.py`, `suffix_decoding.py` each implement `propose`; the runner
maintains the canonical `output_token_ids` and re-issues a verify
pass. The detokenizer is *not* informed token-by-token of proposed
tokens; it only sees the **accepted** prefix after rejection sampling
finalises. When a draft is rejected and tokens are rewound, the
detokenizer's `_protected_step()` catches
`"Invalid prefix encountered"` and reinitialises
`self.stream = tokenizers.decoders.DecodeStream(skip_special_tokens=...)`
**without** the prompt tokens — i.e. the stream gracefully recovers
from a corrupted UTF-8 prefix that comes from a rewind.

**Atlas gap.** Atlas's `decode_logits_seq.rs` and its
`scheduler/{mtp_step,verify_k2_step,verify_k3_step,verify_k4_step,
rollback}.rs` chain do NOT have an equivalent reset on the byte
detokenizer. Atlas's `handle_token.rs` does its OWN incremental
decode via `ctx.state.tokenizer.decode(&state.all_toks)` — a *full
re-decode* per token, which is robust against rewinds (the byte
stream is recomputed from the full canonical token list each time) at
the cost of O(N) decode work per emitted token. This is *better* than
vLLM's `DecodeStream` for the FP8 drift scenario *because* Atlas
already mirrors the reasoning path (handle_token.rs lines 313–328) —
the previous incremental decoder dropped metaspace bytes at
byte-level BPE boundaries which is exactly the missing-`=` and
glued-`name = test-rust-axum-v32version` symptom (handle_token.rs
lines 308–321 already document this). **Confirm Atlas is on the
"full re-decode" path everywhere.** PR 40785's "fragmented tag
detection" lesson — *use cumulative `current_text`, not
`delta_text`* — is already adopted in `streaming_impl.rs` (the
detector accumulates into `self.buffer`).

## Step 5 — Detokenizer holdback (`v1/engine/detokenizer.py`)

`get_next_output_text(finished, delta)` returns
`output_text[:-buffer_length]` (or the delta equivalent) where
`stop_buffer_length = max(len(s) for s in stop) - 1`. The remaining
≤(stop_len-1) bytes stay withheld until either: (a) a stop is
matched, or (b) the request finishes and the full tail flushes.
`check_stop_strings()` searches at offset
`1 - new_char_count - stop_string_len`.

**Atlas gap.** Atlas implements this verbatim in
`handle_token.rs::apply_stop_string_holdback` (lines 611–651) with
the same algorithm and even the same lineage comment. **No gap
here.**

## Step 6 — Hermes tool-call streaming (`tool_parsers/hermes_tool_parser.py`)

vLLM uses **two operations** that Atlas does not:

1. **`partial_tag_overlap(text, tag) -> int`** — returns the length of
   the longest prefix of `tag` that matches a suffix of `text`. Used in
   *two* places: (a) `_extract_content()` to **hold back** the suffix
   that could be the start of `<tool_call>`, (b)
   `_extract_tool_call_jsons()` to strip a partial `</tool_call>`
   suffix before the body is parsed.

2. **`extract_intermediate_diff(curr, old)`** + **`compute_tool_delta`**
   — compute the **incremental** delta for streaming JSON arguments
   by stripping a `find_common_suffix` (closing brackets/quotes that
   the partial-JSON parser optimistically supplied but the model
   hasn't actually emitted yet) and a `find_common_prefix` (already
   sent). The withheld suffix is required to **end with** the
   previously-pending close, otherwise vLLM **raises** — this guards
   against the model walking backward.

The state machine: `prev_tool_call_arr[i]` (full parsed dict),
`streamed_args_for_tool[i]` (string already sent), `_sent_content_idx`
(boundary in plain content). `make_tool_call_id()` is invoked when
the name first arrives; `compute_tool_delta` emits **only the new
fragment** of `function.arguments`.

PR 40785 (2026-04-25) **switched from `delta_text` to `current_text`
analysis** specifically to handle Qwen3-Coder under speculative
decoding — when 3+ tokens land in a single delta, the per-delta
scanner fragments tags. The same PR ensures all accumulated args
emit before the closing `</function>` even when value + close arrive
in the same delta.

**Atlas gaps — direct hit on the drift symptom.**

- `streaming_impl.rs::safe_emit_len` only holds back partial openers
  (`<tool_call>`, `<|tool_call>`, `<minimax:tool_call>`, `<function`,
  `[TOOL_CALLS]`). It **does NOT** hold back partial *body* suffixes
  like `</parameter`, `</function`, or `</tool_call`. When the
  detector is `inside_tag=true` and `</tool_call>` straddles a
  boundary, the parser waits for the close to land (correct), but
  there is no symmetric **content-side** holdback for these closers
  appearing inside Qwen3.6's `<parameter=KEY>VALUE</parameter>`
  pattern. If the model emits `</parameter` as a leading suffix
  fragment, then `>` next token, the body parser sees the value end
  too early on partial parses.

- Atlas's `streaming_impl.rs` emits the **full canonicalised JSON in
  one `ToolCallDelta`** at `</tool_call>` (see line 67 +
  tool_handlers.rs line 211–214) — by design, because XML
  `<parameter=KEY>VALUE</parameter>` needs the close before it can
  become JSON. vLLM's per-key argument streaming via
  `compute_tool_delta` is not feasible in the same way, but Atlas
  has the **information** to do per-`<parameter>` mini-deltas:
  when a `</parameter>` arrives, emit a JSON-fragment delta with
  just `{"KEY":VALUE,` (or `,KEY":VALUE`). The current
  one-shot-at-close means a 30-line Cargo.toml content takes one
  big delta — and if the model then *modifies* a value mid-stream
  on a re-decode pass (FP8 drift produces logit-flip and the model
  re-emits a different value), Atlas has no mechanism to detect
  that the value *changed* from the prefix — vLLM's
  `extract_intermediate_diff` would catch this.

- Atlas has **no `find_common_suffix` guard** on emitted args. The
  drift mode "string sneaks into a numeric field" can be caught
  cheaply by validating that the emitted-so-far args remain a
  *prefix* of the canonical args after each `<parameter>` close.

## Step 7 — Final SSE wire chunk

vLLM serializes `DeltaMessage` into JSON SSE frames; nothing
Atlas-relevant beyond the per-chunk shape.

---

## Atlas-specific PRs to mirror

- **PR 40785** — fragmented-tag detection via cumulative buffer (done
  in Atlas streaming_impl) + ensure args emit *before* close even
  when both land in one delta (**Atlas gap**:
  `streaming_impl.rs:57–80` emits Delta then End in the SAME match
  — fine, but only because Atlas re-parses the whole inner; the
  problem hits Atlas when XML body has multiple
  `<parameter=KEY>VAL</parameter>` blocks: only the final canonical
  JSON ever streams).
- **PR 31778** — handle multi-token deltas where
  `<tool_call_start> + body + <tool_call_end>` arrive together
  (Atlas safe via cumulative re-decode but should add an explicit
  test analogous to vLLM's
  `test_extract_tool_calls_streaming_speculative_decode_loss`).
- **PR 32518** — lost closing brace from partial-JSON parser; Atlas
  uses `find_balanced_json_end` (correct) but the qwen3_coder
  XML→JSON canonicaliser does not run an explicit balanced-bracket
  check, so a `<parameter=` body that itself contains `}` corrupts
  the synthesized JSON; mitigated today by canonical
  `serde_json::from_str` round-trip in `streaming_impl.rs:182`.
- **PR 34101** — stop-string truncation; Atlas already mirrors via
  `apply_stop_string_holdback`.
- **PR 33965** — duplicate tool names + `</parameter>` leak; Atlas
  has `tool_arg_dedup_within` + `name_run` (better than vLLM's
  in-PR fix), and `</parameter>` stripping in the reasoning sanitizer
  (handle_token.rs:200–206).

---

## Five ranked actionable Atlas fixes

(in executive summary, with file:line)
