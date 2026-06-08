# SGLang post-processing pipeline — deep dive

Companion to `research_vllm_sglang_moe.md` (B1 vLLM agent). Scope: detokenizer
cursor algebra, tool parsers, sampling/grammar/spec-decode interaction. Target:
explain Atlas's multi-turn coherence failures on Qwen3.6-35B-A3B-FP8 in opencode
(empty `<parameter=` bodies, path drift, repetition loops) by cross-referencing
SGLang's `main` (commits visible 2026-05).

---

## 1. Incremental detokenizer — the three-cursor algorithm

### 1.1 SGLang's `DecodeStatus` (detokenizer_manager.py L56-65)

```python
@dataclasses.dataclass
class DecodeStatus:
    decoded_text: str
    decode_ids: List[int]
    surr_offset: int
    read_offset: int
    sent_offset: int = 0
```

Three cursors over the same token list:

| cursor         | meaning                                                          |
| -------------- | ---------------------------------------------------------------- |
| `surr_offset`  | start of the "surrounding-context" decode (anchors BPE for the   |
|                | partial decode so the leading metaspace is preserved)            |
| `read_offset`  | end of the *committed* prefix; tokens `[surr..read]` are decoded |
|                | as the stable surrogate                                          |
| `sent_offset`  | byte offset in `decoded_text` that has already been sent to the  |
|                | client                                                           |

### 1.2 Per-step update (detokenizer_manager.py L180-194)

```python
read_ids.append(self.trim_matched_stop(s.decode_ids[s.surr_offset:], ...))
surr_ids.append(s.decode_ids[s.surr_offset : s.read_offset])
# … batch tokenizer.decode of both …
new_text = read_texts[i][len(surr_texts[i]):]
if recv_obj.finished_reasons[i] is None:
    if new_text and not new_text.endswith("�"):
        s.decoded_text += new_text
        s.surr_offset = s.read_offset
        s.read_offset = len(s.decode_ids)
        new_text = ""
    else:
        new_text = find_printable_text(new_text)
incremental_output = output_str[s.sent_offset:]
s.sent_offset = len(output_str)
```

Key properties:

1. **Two decodes per step**: the parent `tokenizer.decode(decode_ids[surr:])`
   (the "read" decode) and a shorter `tokenizer.decode(decode_ids[surr:read])`
   (the "surr" decode). The delta `read - surr` is the candidate emission.
2. **U+FFFD gate**: when the new tail ends with `�` (Python's replacement-char
   for incomplete UTF-8), `surr_offset` and `read_offset` are NOT advanced.
   The unsafe tail is fed through `find_printable_text` (sglang/utils.py),
   which keeps the prefix up to the last whitespace / CJK boundary and drops
   the rest. The held-back portion will be re-decoded next iteration with the
   newly-arrived token.
3. **sent_offset is a strict prefix counter**: every emission slices
   `output_str[sent_offset:]` and bumps `sent_offset` to `len(output_str)`.
   It only ever increases, so partial UTF-8 can never leak.

### 1.3 `find_printable_text` (sglang/utils.py)

```python
def find_printable_text(text: str):
    if text.endswith("\n"):       return text
    if _is_chinese_char(ord(text[-1])):     return text
    if _is_chinese_char(ord(text[-2])):     return text[:-1]
    return text[: text.rfind(" ") + 1]
```

Heuristic borrowed from HF `TextStreamer`. For Latin/ASCII content this means
**the printable region ends at the last space**; everything after is buffered
until the next token confirms a word boundary. This is materially different
from Atlas's "stable region ends at the last non-FFFD byte" rule.

### 1.4 vLLM vs SGLang vs Atlas — comparison

| Engine | Decode pattern | Hold-back rule | Where state lives |
| ------ | -------------- | -------------- | ----------------- |
| vLLM v1 (`detokenizer.py:IncrementalDetokenizer.update`) | Two `tokenizer.decode` calls per step: a `prefix` (last N tokens before this step) and a `current` (prefix + new token). Emit `current[len(prefix):]`. | Last 4 tokens kept as the "prefix" reroll buffer; stop-string holdback uses a separate `output_buffer` sized to `max(len(s)) - 1`. | Per-request `IncrementalDetokenizer` (Python). |
| SGLang | Two decode calls (`surr` + `read`); offsets advance only when new text does NOT end with `�`; FFFD branch falls back to `find_printable_text`. | Word-boundary holdback via `find_printable_text` (last space). | `DecodeStatus` map keyed by `rid`. |
| **Atlas** | **Single** `tokenizer.decode(&state.all_toks)` of the entire cumulative content sequence every token; emit `full[state.emitted .. stable_end]` where `stable_end = full.trim_end_matches('\u{FFFD}').len()`. | FFFD trim only. **No word-boundary holdback.** | `StreamState.all_toks` + `state.emitted` in `chat_stream/handle_token.rs`. |

Atlas's approach is structurally similar to SGLang's but **lacks the
word-boundary heuristic**. The cost is O(n²) over the response length (the
full decode is re-run every token), and — more critically for the present
bug — Atlas emits as soon as UTF-8 is valid, even mid-word. That's normally
fine because the model emits whole BPE tokens, but it interacts badly with
the per-delta sanitiser: a sub-word fragment that *contains* a partial
opener like `<too` will not match `<tool_call>` in the bare-string scan in
`safe_emit_len`. SGLang's word-boundary holdback would have buffered that
fragment until after the space.

**Observation 1 (atlas-side risk).** `handle_token.rs:329` (`stable_end =
full.trim_end_matches('\u{FFFD}').len()`) emits at any byte boundary that
isn't an incomplete codepoint. The sanitiser at L446
(`sanitize_content_chunk`) does keep a tag-scan tail buffer, but the
**detector** (`StreamingToolDetector::safe_emit_len`,
`tool_parser/streaming_impl.rs:368-397`) only holds back *known tag
prefixes*. Any *content* fragment between tags is emitted immediately —
including a tool-call body byte that may turn out to be the leading byte of
a multi-byte logical fragment.

---

## 2. Tool parsers — re-parse-from-buffer pattern

### 2.1 Base class (`base_format_detector.py`)

```python
def parse_streaming_increment(new_text, tools):
    self._buffer += new_text
    if not self._ends_with_partial_token(self._buffer, self.bot_token):
        # not a tool call OR cannot yet decide → return as normal_text
        ...
    # consume buffer; advance current_tool_id, current_tool_name_sent,
    # streamed_args_for_tool[i]; emit `argument_diff` (the delta to the
    # previous tool-call args).
```

State per detector instance: `current_tool_id` (-1 until first tool seen),
`current_tool_name_sent: bool`, `streamed_args_for_tool: List[str]`,
`prev_tool_call_arr: List[Dict]` (last parsed snapshot).

The model-specific detector overrides `parse_streaming_increment`. There is
**no** general "min length" or "non-empty enforcement". The base class will
happily emit `parameters=""` for the first chunk (name only) followed by
JSON deltas.

### 2.2 Qwen3-Coder detector — the smoking-gun precedent

From `qwen3_coder_detector.py:282-477`:

```python
# Parameter: <parameter=name>value...
if current_slice.startswith(self.parameter_prefix):
    name_end = current_slice.find(">")
    if name_end != -1:
        value_start_idx = name_end + 1
        rest_of_slice = current_slice[value_start_idx:]

        cand_end_param = rest_of_slice.find(self.parameter_end_token)
        cand_next_param = rest_of_slice.find(self.parameter_prefix)
        cand_end_func   = rest_of_slice.find(self.function_end_token)

        candidates = []
        if cand_end_param != -1:
            candidates.append((cand_end_param, len(self.parameter_end_token)))
        if cand_next_param != -1:
            candidates.append((cand_next_param, 0))
        if cand_end_func != -1:
            candidates.append((cand_end_func, 0))

        if candidates:
            best_cand = min(candidates, key=lambda x: x[0])
            end_pos = best_cand[0]
            param_name = current_slice[len(self.parameter_prefix):name_end]
            raw_value  = rest_of_slice[:end_pos]

            if raw_value.startswith("\n"): raw_value = raw_value[1:]
            if raw_value.endswith("\n"):   raw_value = raw_value[:-1]
            # → emit JSON fragment, NO empty-value guard
```

Two things matter:

1. **Empty values are accepted.** `raw_value = ""` happily flows through
   `_convert_param_value()` (which for string-typed params returns `""`
   verbatim) and is serialised as `"key": ""`. SGLang does not enforce a
   minimum content length.
2. **The terminator is "smallest of three".** If a parameter never closes
   with `</parameter>` but the *next* `<parameter=` or `</function>` shows
   up first, the parser will *truncate at that point and emit whatever was
   between the two markers* — even if the value was empty. This is robust
   against malformed output but **propagates empty bodies to the client**
   rather than triggering recovery.

Atlas's `qwen3_coder.rs` parser (called from `parse_one_call` once the
detector has the *complete* buffer) is structurally similar and similarly
permissive — but Atlas has the extra problem that the detector hands
incomplete buffers to `parse_one_call` only on close (the "all-or-nothing"
streaming model), whereas SGLang streams arg-deltas. That delta model means
SGLang emits the *name* immediately and the client UI can show a tool-call
header before any args; clients tolerate empty-args-so-far. Atlas waits for
`</tool_call>` to land before extracting args (`streaming_impl.rs:57-83`),
which means the *whole envelope* must be syntactically clean.

**Observation 2.** SGLang's permissive parse + delta-emit hides empty-value
bugs from the model side. Atlas's all-or-nothing parse means a single empty
`<parameter=path>` body either (a) gets emitted as `path: ""` and confuses
the client, or (b) trips one of the `dropping_empty_tool_call` paths in
`pipeline_helpers.rs` and produces no tool call at all. **Both** failure
modes match the bugs reported live. SGLang's design is not a fix — it
papers over the same root cause.

### 2.3 Per-step buffer re-scan, no incremental optimisation

Both `base_format_detector` and `qwen3_coder_detector` re-scan their
`_buffer` from `parsed_pos` on every call. There's no incremental tokenizer
state to corrupt: the parser is pure-string over Python `str`. This is the
same shape as Atlas's `StreamingToolDetector::process` (loop until no
forward progress). The Atlas implementation is slightly more defensive
(explicit `safe_emit_len` so partial tag prefixes never reach the client)
but loses on Mistral `[TOOL_CALLS]` because it doesn't track
`bot_token`-style sentinels at the base level.

---

## 3. Sampling — penalisers and grammar

### 3.1 `sampling_batch_info.py`

The orchestrator is `BatchedPenalizerOrchestrator`, which holds:

- `BatchedRepetitionPenalizer`
- `BatchedFrequencyPenalizer`
- `BatchedPresencePenalizer`
- `BatchedMinNewTokensPenalizer` ← **this is the only "non-empty" enforcement**

`BatchedMinNewTokensPenalizer` sets the EOS-token logit to `-inf` until the
sequence has produced `min_new_tokens` new tokens. There is **no
content-vs-tool-call distinction** — it just forces N tokens of *something*.
Atlas has no equivalent gating, which means a degenerate FP8 path where the
model immediately emits a (broken) `<tool_call><function=write><parameter=
content></parameter></function></tool_call>` envelope can finalise in ~30
tokens before any guard fires.

### 3.2 Grammar / structured output

`update_regex_vocab_mask()` allocates a mask per batch row; for each row
with an active grammar, `fill_vocab_mask()` writes the allowed-token bitmap.
The mask is applied before sampling via `apply_logits_bias()` (add `-inf` to
forbidden positions).

SGLang uses **XGrammar** as the default backend, with an `outlines` /
`llguidance` fallback. The grammar instance lives in the request object and
is *stepped* in the scheduler after each accepted token. This is precisely
the integration shape Atlas already has via `xgrammar` — but with one
critical difference: SGLang's grammar instance is **per-request**, and the
grammar is rolled back when spec-decode drafts are rejected (see §4).

### 3.3 Custom logit processors

`CustomLogitProcessor.from_str()` instantiates user-provided processors and
applies them after the structural mask. Importantly, **logit processors run
BEFORE sampling but AFTER grammar mask** — meaning a user processor cannot
re-enable a grammar-banned token, but it can ban additional tokens. There
is no Atlas equivalent of pluggable logit-processors as a public API.

---

## 4. Speculative decoding — rollback and grammar interaction

The `speculative/` directory contains `eagle_worker.py`,
`eagle_worker_v2.py`, `spec_utils.py`, `draft_extend_attention_backend.py`,
etc. The relevant invariants (gleaned from `eagle_worker.py` and
`spec_utils.py`):

1. **Draft k tokens** with the draft model (EAGLE head).
2. **Verify in parallel** with the target model: one forward pass that
   produces logits for positions `[t, t+1, ..., t+k]`.
3. **Accept the longest matching prefix** (greedy or rejection sampling
   based on `sampling_params.temperature`).
4. **Rollback**: discard the rejected suffix from `req.output_ids`, rewind
   the KV cache (`kv_cache_pool.rewind(req, n_rejected)`), and — critically
   — **rewind the grammar** (`req.grammar.rollback(n_rejected)`).
   `XGrammarGrammar.rollback()` is a single FFI call into XGrammar that
   pops the matcher stack by N positions.

Atlas already implements grammar rollback (`grammar/state.rs:180`
`self.matcher.rollback(n as i32)`, called from `scheduler/rollback.rs:387`).
The relevant questions for parity are (a) does the rollback fire on
**every** rejection path (verify, MTP, ngram self-spec), and (b) is the
*count* always `n_drafted - n_accepted` rather than `n_drafted` (off-by-one
would systematically advance the matcher one position too far per
verification step, which under FP8's higher rejection rate would
accumulate to mid-parameter drift within ~50 tool calls).

**Observation 3.** Audit `scheduler/rollback.rs:387` for the rollback count
relative to spec-decode acceptance. With FP8 rejection rates near 73%
(memory: `project_qwen36_fp8_post_think_eos.md` — though that's NGram, not
MTP), even a 1-token off-by-one over many verifications drifts the grammar
matcher into the wrong byte position inside a `<parameter=...>` body.

---

## 5. Issue cross-reference

| SGLang issue | Symptom | Relevance |
| ------------ | ------- | --------- |
| #9654        | "Streaming multiple tool calls fail with `tool_choice=auto` in Qwen3-Thinking — only first tool emitted; later parsing raises" | Same shape as Atlas multi-tool MiniMax/Qwen drift. SGLang fix: parser-level multi-`<invoke>` handling. Atlas already handles this in `streaming_impl.rs:96-102` (F75). |
| #8331        | "qwen3 function call parser is too eager" | Detector trips on bare `<tool_call>` in reasoning content. Atlas has the Layer-A in-think scanner for this (handle_token.rs:232-272). |
| #26172 (PR)  | "fix: correct tool parser for Qwen 3.5" | Open. Confirms upstream parser instability for the Qwen3.x family. |
| #15135 (closed) | "Add schema-based type conversion to Qwen3CoderDetector" | Closed without merge. Suggests upstream knows the type-coercion gap (string vs int values for the same param) — Atlas has its own `type_coerce.rs`. |
| #23687       | "Qwen3.6-27B-FP8 (dense): FP8 weight_scale_inv silently dropped" | **Direct evidence of FP8 quant pipeline bugs causing Qwen3.6 output degradation upstream too.** Not Atlas-specific. |

---

## 6. Conclusions

SGLang's pipeline is materially more careful than Atlas's in three places:
(a) word-boundary hold-back via `find_printable_text` in the detokenizer,
(b) explicit `streamed_args_for_tool` delta-emission in the parser, and
(c) grammar rollback on spec-decode rejection. SGLang is *less* careful
than Atlas in detecting partial tag prefixes (no `safe_emit_len` analogue)
and in suppressing in-think tool-call leaks. The empty-`<parameter=>`
failure mode reproduces on SGLang's Qwen3-Coder detector for the same root
cause (permissive empty-value parsing), but SGLang papers over it via
delta-streaming the name early and accepting empty args as a non-fatal
state. The path-drift and repetition-loop failures are not addressed in
SGLang either — they are model-side symptoms of FP8 quant drift
(memory: `project_qwen36_phase2b_softmax_expf.md`) amplified by
grammar-rollback gaps during spec-decode.
