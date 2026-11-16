<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# llama.cpp server compatibility

A client written for llama.cpp's `llama-server` can talk to Atlas without
changes. The contract followed is `tools/server/README.md` and the handlers in
`tools/server/` at llama.cpp tag `b11583`. The code is in
`crates/spark-server/src/api/llama_compat/`.

There is no second inference path. `/completion` builds the same scheduler
request the OpenAI `/v1/completions` streaming route builds and sends it to the
same scheduler; the layer translates request and response shapes.

**The rule for fields Atlas cannot honor:** a llama.cpp field that changes what
the model produces, and that Atlas has no equivalent for, is refused with HTTP
400 naming the field. It is never ignored. A field at the value llama.cpp
documents as its default (a request that does not use the feature) is
accepted. In particular a request with `json_schema` or `grammar` never gets
unconstrained text with HTTP 200, and a request with `n_probs` gets
probabilities or a 400.

Errors use llama.cpp's envelope: `{"error": {"code": 400, "message": "...",
"type": "invalid_request_error"}}`.

## Routes

| Route | Status | Notes |
|---|---|---|
| `POST /completion`, `/completions` | supported | Fields below. Blocking and `stream`. |
| `POST /tokenize` | supported | `content` (string, or array of strings and token ids), `add_special`, `with_pieces`. `parse_special=false` is refused. A body without `content` is Atlas's own `prompt` / `messages` request and keeps its `{tokens, count}` answer. |
| `POST /detokenize` | supported | Answers `{"content", "text"}`: `content` is llama.cpp's (special tokens written out), `text` is the field Atlas has always returned (special tokens dropped). |
| `POST /apply-template` | supported | `messages`, optional `tools`, `chat_template_kwargs.enable_thinking` / `reasoning_effort`. Answers `{"prompt"}`, rendered by the same renderer as chat requests. Image parts, other template kwargs and `add_generation_prompt=false` are refused. Thinking follows the model's default unless the kwargs say otherwise. |
| `GET /props` | supported | `default_generation_settings` (`params`, `n_ctx` = `--max-seq-len`), `total_slots` (= `--max-num-seqs`), `model_alias`, `model_path` (the model id Atlas was started with), `chat_template`, `bos_token`, `eos_token`, `modalities`, `build_info` (`atlas-<version>`). `POST /props` is not provided. |
| `GET /health` | supported | `200 {"status":"ok","model":...}` when ready (was `"ready"`; llama-server says `ok`). `503` while loading or after a GPU fault, with llama.cpp's `error` object next to Atlas's `status` and `reason`. `/v1/health` is not provided. |
| `POST /embedding`, `/embeddings` | 501 | llama.cpp's own answer when embeddings are off. Atlas serves generative models only. |
| `/infill`, `/rerank`, `/slots`, `/lora-adapters`, `/models` | not provided | 404. |

With `--require-auth`, `/completion`, `/completions`, `/apply-template`,
`/props` and `/embedding(s)` need the bearer token (`/health` stays public, as
in llama.cpp). `/completion` also takes a rate-limit slot and is refused with
503 after a GPU fault, like `/v1/*`.

## `/completion` request fields

Aliases and "JSON `null` means absent" follow llama.cpp.

### Honored

| Field | Mapping |
|---|---|
| `prompt` | String, token ids, mixed array, or `{"prompt_string": ...}`. A string prompt gets the model's start-of-text token when its tokenizer adds one (llama.cpp's `add_special`). Several prompts in one request: 400. |
| `n_predict` (`max_tokens`, `max_completion_tokens`) | `-1` = until the context is full. `0` and values below `-1`: 400. Capped at the room left in `--max-seq-len`. |
| `temperature` | Negative = 0 (greedy). Above 2: 400. |
| `top_k`, `top_p`, `min_p` | Clamped as llama.cpp clamps them. |
| `top_n_sigma` | Negative disables. Exactly `0`: 400 (llama.cpp means "top token only"). |
| `repeat_penalty`, `presence_penalty`, `frequency_penalty` | Presence and frequency must be within -2 to 2. |
| `dry_multiplier`, `dry_base`, `dry_allowed_length` | Mapped. With DRY on, `dry_penalty_last_n` other than 64 and custom `dry_sequence_breakers`: 400. |
| `seed` | Negative = random. |
| `stop` | String or array. Matched on the decoded text, excluded from `content`, generation is cancelled when one completes; `stop_type` is `word` and `stopping_word` names it. |
| `stream` | Bare SSE `data: {json}` frames, one per token, closing frame with `stop: true`. No `[DONE]`, as in llama-server. |
| `logit_bias` | `[[token, bias], ...]` or `{"token": bias}`; a token is an id or a string; `false` bans it. Out-of-range ids and malformed entries: 400 (llama.cpp skips them). |
| `n_probs` (`logprobs`) | 0 to 20. Returned as `completion_probabilities`, one entry per generated token: `{id, token, bytes, logprob, top_logprobs: [{id, token, bytes, logprob}]}`. Log-probabilities are pre-sampling, so `post_sampling_probs=true` is refused. A token the scheduler returned no logprobs for keeps its slot with `logprob: null` and an empty `top_logprobs`. In stream mode each frame carries its token's entry. |
| `return_tokens` | `tokens` holds the generated ids (always set per frame when streaming). |
| `json_schema` | Converted by XGrammar. An empty object means "any object". |
| `grammar` | GBNF/EBNF with a `root` rule. |

`json_schema` and `grammar` are validated with XGrammar's parser before the
request is queued. 400 when: both are sent, the text does not parse, or the
grammar engine did not initialize at startup. `grammar_lazy` and
`grammar_triggers` are refused.

### Accepted, no effect in Atlas

`cache_prompt` (prefix caching is a server setting in Atlas,
`--enable-prefix-caching`, off by default; it changes speed, not output), `n_keep`, `n_discard` (Atlas never shifts context; a full context ends
the request with `stop_type: "limit"`), `n_cache_reuse`, `id_slot`, `echo`
(llama.cpp validates it and does nothing with it on this route), `verbose`,
`stream_options`, `sse_ping_interval`, `reasoning_format`,
`continue_final_message`, and the fields below at their off value.

### Refused unless at the off value

`dynatemp_range` (0), `typical_p` (1), `xtc_probability` (0), `mirostat` (0),
`repeat_last_n` (64: the penalty window is set by the server), `min_keep` (0),
`n_indent` (0), `adaptive_target` (negative), `t_max_predict_ms` (0),
`ignore_eos` (false), `post_sampling_probs`, `timings_per_token`,
`return_progress`, `backend_sampling`, `reasoning_control` (false),
`reasoning_budget_tokens` (-1), `lora` (empty), `response_fields` (empty),
`image_data` (empty), `samplers` (only the documented default order),
`n_cmpl` / `n` (1), `prompt` as several prompts or with `multimodal_data`.

### Defaults differ from llama.cpp's

A request that omits a sampling field gets the server's defaults (the model's
`generation_config.json` and CLI), which `/props` reports, not llama.cpp's
`temperature 0.8`, `top_k 40`, and so on.

## Response differences

- `tokens_evaluated` is the prompt length; `tokens_cached` and `timings.cache_n`
  are the prefix-cache hit; `timings.prompt_n` excludes them.
- `timings.prompt_ms` is time to first token, `timings.predicted_ms` is the
  decode time. `truncated` is always `false`.
- A server-side `timeout` finish is reported as `stop_type: "limit"`.
- `generation_settings` lists the settings the request ran with, not every
  llama.cpp field.
- `piece` and `token` strings of a half-character token follow llama.cpp
  (byte array in `/tokenize`, valid-UTF-8 prefix in probabilities, with the
  full `bytes`). Single-token decoding of SentencePiece models can drop a
  leading space.

## Known limits

- If the grammar engine accepts the text but fails to compile it against the
  tokenizer, the scheduler decodes unconstrained (logged as a warning). The
  parse checks above catch syntax and schema errors; this tokenizer-dependent
  failure is not caught before the request is queued.
- Several prompts per request, `n_cmpl > 1` and per-token timing are not
  implemented; they are refused, not approximated.
