# arch_phase_05_tool_grammar.md — Tool-call + structured-output grammar pipeline: vLLM vs Atlas

Sources
- vLLM (v1): `/home/nologik/vllm/vllm/vllm/entrypoints/openai/{serving_chat,protocol,chat_utils}.py`, `vllm/entrypoints/openai/tool_parsers/*.py`, `vllm/v1/structured_output/{__init__,backend_xgrammar,backend_types,utils}.py`, `vllm/v1/sample/{sampler,ops/penalties}.py`, `vllm/v1/worker/gpu_model_runner.py`, `vllm/model_executor/layers/utils.py`.
- Atlas: `/workspace/atlas-mtp/crates/spark-server/src/grammar/{mod,engine,state,schema,compile_tools,compile_misc}.rs`, `crates/spark-server/src/tool_parser/{hermes,qwen3_coder,qwen3_xml,minimax_xml,bare_json,gemma4,mistral,parse_single_a,parse_single_b,fuzzy_repair,validation}.rs`, `crates/spark-server/src/api/chat/sampling_setup.rs`, `crates/spark-server/src/scheduler/{emit_step,decode_logits_seq,decode_logits_step}.rs`, `crates/spark-runtime/src/sampler/sample_impl.rs`, `crates/spark-server/src/tokenizer/{chat_impl,jinja_helpers}.rs`, `crates/spark-server/src/toml_repair.rs`, `crates/xgrammar/src/{tokenizer/{vocab_type,info,hf_metadata},structural_tag/{mod,converter_tags}}.rs`.

Workload context: opencode harness uses `tool_choice="auto"` by default; tools include Bash / Write / Edit / Read. Atlas hits 30% cargo_valid, vLLM hits 100%. Both serve `cpatonn/Qwen3.6-Coder-A3B-FP8` (or DevQuasar Qwen3-Next FP8 build).

## 1. Tool-definition injection (tools array → model context)

| Concern | vLLM | Atlas |
|---|---|---|
| Entry | `serving_chat.py:218-220` `tool_dicts = [t.model_dump() for t in request.tools]` | `tokenizer/chat_impl.rs:115-117` `tools: Option<&[serde_json::Value]>` |
| Renderer | HF `tokenizers.apply_chat_template(..., tools=tools)` → Python `jinja2` (`chat_utils.py:480-482`) | `minijinja` with custom helpers / `unknown_method_callback` (`jinja_helpers.rs:60-120`) |
| `tools` payload | OpenAI dict shape, passed verbatim into Jinja `tools` context var (`chat_utils.py:521`) | OpenAI dict shape, passed verbatim into Jinja `tools` context var (`chat_impl.rs:136, 151, 197, 205`) |
| Extra Jinja vars | `chat_template_kwargs` only (model-defined) | `add_generation_prompt=true`, `enable_thinking`, `reasoning_effort` (`high`/`none`), `disable_tool_steering`, `add_vision_id=false` (`chat_impl.rs:149-157`) |
| `arguments` pre-parse | none (`tojson` filter in template handles dict-vs-str) | F76 (2026-04-29): tool-message `arguments` strings parsed to dicts before render (`chat_impl.rs:124-134`, `tokenizer.rs::normalize_tool_call_arguments`) — needed for MiniMax template's `_args.items()` |
| Schema mutation | none — tools are passed as-is | TAFC: optionally inserts an `_think` string property at head of every tool's `properties` (`grammar/schema.rs:29-58`) **iff** `[behavior].enable_tafc=true` in MODEL.toml. Stripped post-parse (`parse_single_b.rs:157`) |
| Implicit `minLength` | none | `enforce_min_length_on_required_strings` (`grammar/schema.rs:60-…`) adds `"minLength": 1` to every required string property of every tool's JSON schema **before** grammar compilation (`compile_tools.rs:60, 132, 203, 423`) |

Notes
- Both engines feed the model the same `tools` array via Jinja, so the **string the model sees** at prefill is bit-identical iff (a) Atlas's minijinja produces the same bytes as HF's Python `jinja2`, and (b) Atlas does not inject `_think`. Atlas's minijinja is a re-impl; small differences in `tojson` whitespace, attribute ordering of Maps, or filter semantics could shift the prompt by a token or two (no direct evidence of this here — flag for prefill bytewise comparison).
- TAFC `_think` injection (when enabled) **changes the schema the model trains-saw-vs-runtime-sees**, which can shift attention distribution at the `<tool_call>` boundary. Off by default — confirm not flipped on in the FP8 drift workload.

## 2. Grammar compilation (JSON Schema → token-mask FSM)

| Concern | vLLM | Atlas |
|---|---|---|
| When grammar is **built at all** | only when `response_format` is `json_object` / `json_schema` / `structural_tag`, OR `structured_outputs=*` is set, OR `tool_choice="required"`/named-tool (which goes through an inline tool-call extractor, NOT through xgrammar). Tool_choice="auto" → **NO grammar** (`protocol.py:826-884`, `serving_chat.py:560-603`). | Every request with `tools_active` (non-empty `tools` + `tool_choice != "none"`) compiles a structural_tag grammar (`api/chat/sampling_setup.rs:172-210`). `tool_choice="auto"` → grammar still active with **triggered_tags** (`compile_tools.rs:81-87`). |
| Backend | `xgrammar` (default), `guidance`, `outlines`, `lm-format-enforcer` (`v1/structured_output/__init__.py:104-139`) | xgrammar-rs (Atlas's own port; `crates/xgrammar/`) — no alternative backend |
| `TokenizerInfo` build | `xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=...)` → vocab type **auto-detected** from HF tokenizer (`backend_xgrammar.py:60-63`) | F68 fix: `from_tokenizer` serializes the `tokenizers::Tokenizer` to JSON, runs `detect_metadata_from_hf` to recover the vocab type (`engine.rs:82-129`). The older `new(vocab, VocabType::RAW, …)` path is still present for callers that pass a pre-decoded vocab — must NOT be used on ByteLevel-BPE tokenizers (Qwen, MiniMax, Mistral) or grammar silently rejects every `\n`/space token (F68 root cause). |
| Tool-grammar shape | n/a (vLLM does not build tool grammars for auto/required; `structural_tag` is only built when the user explicitly sends one as `response_format`) | `compile_structural_tag_raw` → `xgrammar::Grammar::from_structural_tag` (`compile_misc.rs:13-37`). Triggered_tags JSON: `{"triggers":[…],"tags":[…],"at_least_one":…,"stop_after_first":…}` |
| Per-parser body content | n/a | hermes / bare_json: `content.type="json_schema"` (with `minLength≥1` injected) — `compile_tools.rs:63-68, 135-140`. qwen3_coder / qwen3_xml: `content.type="grammar"` with hand-written EBNF (`compile_tools.rs:257-263, 342-348`) **NOT** `json_schema` — sole grammar in tree that emits an EBNF body. gemma4: `content.type="json_schema"`. minimax_xml: `content.type="any_text"` (intentionally loose; outer frame enforced) |
| Triggers | n/a | auto (use_triggers=true): per-tool LATE trigger like `<tool_call>\n<function={name}` (qwen3_coder, `compile_tools.rs:303-307`) or `<tool_call>{"name":""` (hermes). required: SHORT shared trigger (`<tool_call>` / `{"name":""` / `<minimax:tool_call>`). F66/F67 documented the failure mode of LATE triggers under required-mode. |
| at_least_one / stop_after_first | n/a | both = `!use_triggers`. required → matcher refuses EOS until one tool tag completes; auto → matcher accepts EOS at any point (`compile_tools.rs:83-86, 153-156, 312-313`) |
| Schema sanitization | xgrammar rejects unsupported features (multipleOf, uniqueItems, format != allowlist, etc.) and `validate_xgrammar_grammar` raises (`backend_xgrammar.py:221-271`); request 400's | `sanitize_schema_for_grammar` SILENTLY drops the offending tool from the grammar (`compile_tools.rs:47-59, 119-131, 190-208, 410-422`) and logs `warn!`. The tool is still **advertised to the model** via the chat template but **cannot be emitted** under grammar. |
| Compilation cache | `XgrammarBackend(.cache_enabled=True, cache_limit_bytes=VLLM_XGRAMMAR_CACHE_MB·1MB)` (`backend_xgrammar.py:64-69`) | `GrammarCompiler::new(.cache_enabled=true, .cache_limit_bytes=-1)` (`engine.rs:135-141`) |

Critical Atlas-only behaviors
- **`minLength: 1` baked into every required string** (`schema.rs::enforce_min_length_on_required_strings`). XGrammar lowers this to a body rule that physically requires the model to emit at least one non-stop content byte. Two consequences:
  1. If the model has any FP8-driven preference for `\"\"` (empty string) at low-margin positions, grammar deflects it to *something else* — but the FSM has no notion of "right content," only "any non-empty content." The model then has to pick a continuation; under FP8 noise the continuation can be a near-uniform draw from the allowed-token cloud, which is much wider for first-byte-of-content than for the empty-string token (single-id).
  2. For qwen3_coder / qwen3_xml, the EBNF goes further (`value ::= first_char rest`, `first_char ::= [^ \t\r\n<]`, `rest ::= [^<]*`) — the **very first byte** of every parameter value is masked away from whitespace AND `<`. vLLM imposes neither constraint.

- **EBNF body for qwen3_coder/qwen3_xml** (`compile_tools.rs:257-263`): Atlas hand-writes
  ```
  root ::= param ("\n" param)*
  param ::= "<parameter=" paramname ">" value "</parameter>"
  paramname ::= [a-zA-Z_] [a-zA-Z_0-9]*
  value ::= first_char rest
  first_char ::= [^ \t\r\n<]
  rest ::= [^<]*
  ```
  This rejects parameter values that contain ANY `<` byte mid-content. The F2-revert comment (compile_tools.rs:244-256) documents that the looser `[^<] | "<" [^/]` form let the model fall into XML-attribute syntax, so the strict form is restored and `<` mid-value is "handled via parser-side recovery." But: Cargo.toml dependency strings like `serde = { version = "1", features = [...] }`, Rust generics `Vec<T>`, shell redirects (`echo X > file`), and HTML/JSX all contain `<`. **Any tool argument value with a `<` mid-content is grammar-rejected** for qwen3_coder, regardless of what the prompt actually needs. The model must then route around the `<` (emit a different character, restart the value, or close the tag) — every divergence is an opportunity for FP8 drift to derail the response.

## 3. Grammar enforcement during sampling (when/how the mask is applied)

| Concern | vLLM | Atlas |
|---|---|---|
| Mask fill location | CPU thread pool (`structured_output/__init__.py:160-178`); per-request `matcher.fill_next_token_bitmask(bitmask, idx)` | CPU per-request `GrammarState::fill_bitmask` → `xgrammar::GrammarMatcher::fill_next_token_bitmask` (`state.rs:89-100`) |
| Mask apply location | GPU, in `gpu_model_runner.py:2675` `apply_grammar_bitmask(scheduler_output, grammar_output, batch, logits)`. Internally calls `xgr.apply_token_bitmask_inplace(logits, grammar_bitmask, indices=index_tensor)` (`v1/structured_output/utils.py:126`) — disallowed → `-inf` on-device | CPU, in `decode_logits_seq.rs:369` `gs.apply_bitmask_to_logits(&mut f32_logits)`. Sets disallowed → `f32::NEG_INFINITY` in a CPU O(V) loop (`state.rs:193-202`). Logits already on host (Atlas runs sampler on CPU) |
| Position in sampling pipeline | (1) bitmask → -inf — (2) Sampler.forward: allowed_token_ids mask → bad_words → non-argmax-invariant logitsprocs (logit_bias incl.) → **penalties (repetition / freq / presence)** → (3) sample: greedy OR temp scale → top-k/top-p → multinomial (`sampler.py:67-126, 263-316`) | (1) forced-token fast-path (`decode_logits_seq.rs:344-356`) — (2) bitmask → -inf (`:365-370`) — (3) `sample_with_params_history`: rep_penalty → pres/freq → LZ → DRY → logit_bias (+) → **greedy bypass at temp≤0** OR top_n_sigma → temp scale → top-k → softmax → min-p → top-p → multinomial (`sample_impl.rs:43-239`) |
| Order of grammar mask vs rep_penalty | grammar FIRST → rep_penalty SECOND. -inf survives `/` or `*` by rep_penalty. Allowed tokens get penalised normally. | grammar FIRST → rep_penalty SECOND. **Same order as vLLM.** No double-mask issue. |
| Penalty target set | output tokens **AND** prompt tokens (`apply_penalties` uses both `prompt_mask` and `output_mask`, `model_executor/layers/utils.py:78-94`) | **output tokens only** (`sample_impl.rs:51-67` reads `token_history` which is `a.output_tokens` — `decode_logits_seq.rs:678`). Prompt tokens never penalised. |
| Stop-token exemption from rep penalty | none — stop tokens are penalised exactly like any other token | none — stop tokens (`</tool_call>`, `<\|im_end\|>`, EOS) are penalised exactly like any other token. The `stop_token_ids` field on `SamplingParams` exists (`sampler.rs:96`) but is **never read** by `sample_with_params_seeded` — dead field. The MEMORY.md note about a HF-convention stop-token exemption in `atlas-gb10:fp8-final` is **either not in this branch or not implemented in the runtime sampler**. |
| Forced-token fast-path | not present | `forced_token()` (`state.rs:163-168`) returns the sole grammar-legal next token; Atlas emits it directly without running the sampler (`decode_logits_seq.rs:344-356`). Gated by `inside_thinking`, `top_logprobs`, kill-switch, and Tier-1 (in-param-body, zero content) suppression. **Bypasses logit_bias / rep_penalty / sampler entirely** for these positions — design intent is "byte-identical to all-but-one-masked sample," but the WS1/AM1/Tier-1 logit_bias entries that would have applied are skipped. |
| Spec-decode rollback | `XgrammarGrammar.rollback(num_tokens)` (`backend_xgrammar.py:186-189`) | `GrammarState::rollback(n)` (`state.rs:179-181`) — matcher max_rollback = unlimited |
| Thinking-mode handling | `should_fill_bitmask(request)` returns False during reasoning unless `enable_in_reasoning=true` (`structured_output/__init__.py:290-304`). Mask is full-allow during thinking | `if !a.inside_thinking && let Some(ref mut gs) = a.grammar_state && gs.fill_bitmask()` (`decode_logits_seq.rs:365-370`). Bitmask filled, but applied only outside `<think>`. Same effective behavior. |

Critical Atlas-only ops at the sampler boundary
- **WS1 mask** (`decode_logits_seq.rs:483-515`): at parameter-body position 0, push `(510, -8.0)` for `</` + all whitespace-only token IDs (vocab-scanned, ~440 on Qwen3.6) + attractor tokens (e.g. `lean`, ` lean`) at -8.0. **vLLM has nothing analogous.**
- **WS2 mid-content mask** (`:516-543`): if previous output token is digit-ending, push -3.0 for every whitespace token. Targets the Tier-A drift mode `0.1.0`→`0.1 .0` documented in `research_C1_results.md`. **vLLM has nothing analogous.**
- **A4 POST_THINK_MIN_REASONING floor** (`:544-564`): -8.0 on `</think>` token until `thinking_tokens >= 16`. **vLLM has nothing analogous.**
- **B1 margin-ratio detector** (`:566-610`): O(V) scan for top1/top2 margin, logs low-margin positions inside parameter bodies. Diagnostic only — does not affect logits.
- **C4v1 lift** (`:612-642`): DISABLED in source. Would lift top-2 by `(LOW_MARGIN_THRESHOLD - margin) * 0.5` at low-margin positions. Reverted because it introduced reasoning-text-in-param-body drift.
- **Order check**: WS1/WS2/A4 push to `logit_bias_local` (`:482`), which `sample_with_params_history` applies *after* rep_penalty/pres/freq/LZ/DRY (`sample_impl.rs:111-116`). Grammar mask was applied *before* sample call. So pipeline is: **grammar -inf → rep_penalty → pres/freq → LZ → DRY → WS1/WS2/A4/AM1 bias → greedy/sample**. WS1 etc. cannot rescue a token already masked to -inf by grammar (additive bias on -inf is still -inf), so they only affect tokens the grammar already allowed.

## 4. Tool-call parsing (raw output → structured calls)

| Concern | vLLM | Atlas |
|---|---|---|
| Parser registry | plugin loader; one parser per model selected via `--tool-call-parser` (e.g. `hermes`, `qwen3_xml`, `qwen3_coder`, `minimax_m2`, `mistral`, `gemma4`/`pythonic`, etc.) — `tool_parsers/*.py` | trait `ToolCallParser` (`tool_parser.rs`) with impls `Hermes`, `Qwen3Coder`, `Qwen3Xml`, `Mistral`, `Gemma4`, `MinimaxXml`, `BareJson`. Wired in `app_state.rs:36`; selected from `MODEL.toml` |
| Hermes | regex `<tool_call>(.*?)</tool_call>` (`hermes_tool_parser.py:48-50`) + `partial_json_parser` for truncated JSON streaming | `parse_one_call` (`parse_single_a.rs:8-86`): plain `serde_json::from_str` for complete JSON; truncated-JSON recovery walks args from last `}`/`]` (`:41-66`); fallback through MiniMax XML, qwen3_coder, tag-style |
| qwen3_coder | regex-driven (`qwen3coder_tool_parser.py:54-66`); `tool_call_parameter_regex` recovers missing `</parameter>` via lookahead `(?:</parameter>|(?=<parameter=)|(?=</function>)|$)` | hand-rolled byte-walker (`parse_single_b.rs::parse_qwen3_coder_call`). Recovery branches: NVFP4 `<\|function=` / `<function name=` (Wave-7 generalised), embedded `name=` attribute (`:32-39`), `name=name` duplicate sanitiser (`:43-47`), sibling-`</function>` boundary (`:74-78`), missing `</parameter>` walks to next `<parameter=` or `</function>` (`:90-104`), empty-args JSON fallback (`:137-148`), TAFC `_think` strip (`:157`) |
| qwen3_xml | XML expat parser (`qwen3xml_tool_parser.py::StreamingXMLToolCallParser`) returning schema-typed args | Wraps `Qwen3CoderParser` for parse + grammar; runs `coerce_all` for schema typing (`qwen3_xml.rs:17-54`, `type_coerce.rs`) |
| MiniMax | `minimax_m2_tool_parser.py` regex-driven; F75-class double-`<invoke>` handling is plain regex `findall` | `parse_minimax_xml_calls_all` (`parse_single_a.rs:228-247`) bounded per-invoke; F80b drops empty-path writes for `Write`/`Edit`/`MultiEdit` (`:175-206`); F76 `_args.items()` Jinja shim |
| Tag-style fallback | n/a per parser | `parse_tag_style_call` (`parse_single_b.rs:174-210`) handles `<function>NAME</function><parameters>…` if all else fails |
| Type coercion | `_convert_param_value` in qwen3_coder parser (`qwen3coder_tool_parser.py:134-238`): string / int / float / bool / object via `json.loads` + `ast.literal_eval` fallback | `wants_typed_arguments`-gated `coerce_all` (`type_coerce.rs`); qwen3_coder leaves values as strings (`parse_single_b.rs:123`), qwen3_xml typifies via schema |
| Streaming delta parser | per-parser `extract_tool_call_required_streaming` / `extract_tool_calls_streaming` (e.g. `qwen3coder_tool_parser.py`, `hermes_tool_parser.py`) | `streaming_impl.rs` central dispatcher; leak markers via `LeakMarkers` (envelope_open / envelope_close strings); F71 + F73 + F75 streaming sanitisers normalize `<minimax:_call>` → `<tool_call>` |
| `<tool_call>` literal recovery (Wave 7) | partial_json_parser handles trailing fragments; no Atlas-style literal-`</tool_call>` body-injection | `parse_qwen3_coder_call` accepts literal `</tool_call>` mid-body and the surrounding parser drops it; Wave-7 also added empty-`{}` repair (model emits `<tool_call>{}</tool_call>` → drop, no tool emitted) |

## 5. Repair / fallback (post-hoc fixes after parse)

| Concern | vLLM | Atlas |
|---|---|---|
| Post-parse JSON repair | per-parser ad-hoc (e.g. `partial_json_parser` for streaming-truncated args; Hermes parser may swallow malformed `<tool_call>` and return content) | F2 + Wave-7: per-parser recovery branches (above). For free-form content, **SC1 TOML auto-repair** (`toml_repair.rs:41-77`): if the `content` parameter of a `Write` to a Cargo.toml fails `toml::from_str`, try (a) preamble-strip to first `[` at line start (`:84-98`), (b) insert newlines at common drift boundaries (`:100-…`), (c) (a)+(b) combined. Accept only if `toml::from_str` succeeds on a candidate. |
| Fuzzy argument repair | none | A2 fuzzy_repair (`tool_parser/fuzzy_repair.rs`): Levenshtein-distance match against prompt-vocabulary for identifier-like tokens (paths, file names, version strings). Catches FP8 single-byte drift like `axum-v3`→`axut-v3`, `wave1`→`wave/`. Caller-driven from validator failure path. Default `LEV_DEFAULT_MAX=2`. |
| Tool-arg validation | request 400 with `partial_json_parser.MalformedJSON` only; no schema validation of args | F78 tier-2 validator (`tool_parser/validation.rs`): schema-validates parsed args. Rejection routes to (a) fuzzy repair, (b) Tier-5c re-roll (re-inference with augmented prompt), (c) finish_reason=stop with no tool emitted. F80b additionally drops empty-path writes from MiniMax XML pre-validator (`parse_single_a.rs:175-206`). |
| TOML grammar fallback | none | qwen3_coder grammar tries `qwen_xml_parameter` content type first; on EBNF parser error falls back to plain EBNF body (`compile_tools.rs:321-371`) |
| Streaming envelope sanitiser | n/a | F73 `<minimax:_call>` → `<tool_call>` normaliser in streaming layer; F71 leak-marker buffering |

## 6. Critical-question answers

1. **Atlas chat template exactly matches Qwen tokenizer expectations?** Both engines pass the OpenAI-shape `tools` array through the **same model-shipped Jinja template**. Atlas's renderer is minijinja (Rust port), vLLM's is `transformers`'s reference `jinja2`. F76 patches MiniMax's `.items()` call; the `unknown_method_callback` (`jinja_helpers.rs:60-120`) is a compatibility shim. **No evidence of bytewise divergence at the prompt level in this scan**, but a literal-byte comparison of the rendered prompts (Atlas vs vLLM, same `tools` JSON, same messages) is the only way to rule this out — flag for a follow-up `arch_phase_06_prompt_bytes` diff. Atlas adds `disable_tool_steering`, `add_vision_id=false`, and **drives `reasoning_effort` to `"high"`/`"none"`** based on `enable_thinking` (`chat_impl.rs:144-148, 198-202`) — these are Jinja variables; if a Qwen-family template ignores them, no drift; if it branches on them, prompt differs.

2. **Does Atlas mask tokens during prefill (forced tokens)?** Atlas's grammar matcher state is created during prefill (`scheduler/prefill_{a,b}_step.rs:62/60` → `compile_grammar_state`), and a top-k mask warm-up runs against the cache (`state.rs:62-68` `compile_top_k_masks(8)`). But the **bitmask is NOT applied to any prefill logit** — prefill tokens are not sampled, they are teacher-forced from the prompt. The matcher's first `fill_bitmask` happens at the first decode step (`decode_logits_seq.rs:367`). **Both engines apply masks only at sample time.** No prefill divergence on this axis.

3. **Atlas's `tool_call_grammar` — when active / bypassed?**
   - **Active** for every request with non-empty `tools` and `tool_choice != "none"` (`sampling_setup.rs:190-210`), regardless of auto/required. Triggered_tags semantics let the model emit free text in auto mode UNTIL the trigger prefix appears; then the tag body is hard-constrained.
   - **Bypassed** only when (a) `[behavior].disable_tool_grammar=true` in MODEL.toml (`:184-189`), (b) `response_format` set with `tool_choice="none"` (grammar swapped to JsonSchema/JsonObject — `:174-183`), (c) `tools` empty, (d) the selected `ToolCallParser` returns `None` from `compile_tool_grammar` (Mistral default; opt-out — `tool_parser.rs:238`), (e) `gs.fill_bitmask()` returns false (xgrammar reports the matcher imposes no constraint at this state).
   - **Penalty interaction**: A1 (2026-05-26) inverted Phase-3.1 zero-penalty-in-body — penalties now apply inside the tool body at full strength (`decode_logits_seq.rs:438-457`). The change closed the runaway-attractor failure mode (mismatched parens, lean://) at the cost of soft pressure (~9% logit divide for rep_pen=1.1) on legitimate JSON-structure repetition (`":"`, `","`).

4. **Does Atlas's grammar mask BEFORE rep-penalty cause it to mask high-rep-penalty tokens twice?** No. Pipeline is **bitmask (-inf disallowed) → rep_penalty → pres/freq → LZ → DRY → logit_bias add → greedy/sample**. Disallowed tokens are -inf before rep_penalty runs; `(-inf) / rep_penalty = -inf` and `(-inf) * rep_penalty = -inf`, so no double-mask harm. **Allowed tokens that are also in the history get rep_penalty exactly once.** vLLM has the same ordering (grammar mask → sampler.forward → apply_logits_processors → penalties → sample) and the same property. **Not a divergence.**

5. **Stop-token rep-penalty exemption?** Neither engine exempts stop tokens. Atlas's `SamplingParams.stop_token_ids` field exists but is dead (no reads in `sample_impl.rs`). vLLM's `apply_penalties` (`model_executor/layers/utils.py:78-94`) penalises every token in `output_mask` ∪ `prompt_mask` with no stop-token bypass. **Not a divergence with vLLM, but a divergence with HF Transformers' `RepetitionPenaltyLogitsProcessor` which also doesn't exempt stops — so the "HF convention" claim in MEMORY.md (`project_qwen36_fp8_post_think_eos.md`) is incorrect or refers to a different branch.**

## 7. Atlas-only grammar ops (no vLLM analogue under tool_choice="auto")

1. **Tool-call grammar compiled and enforced in tool_choice="auto"** (`sampling_setup.rs:172-210`). The single biggest divergence — vLLM does nothing here.
2. **Per-tool EBNF body** for qwen3_coder/qwen3_xml (`compile_tools.rs:257-263`): hand-written grammar that rejects whitespace-leading and `<`-containing parameter values.
3. **Implicit `minLength: 1`** on every required string property (`schema.rs::enforce_min_length_on_required_strings`, called from every `compile_*_tool_grammar`).
4. **TAFC `_think` schema augmentation** (`schema.rs:29-58`, behind `[behavior].enable_tafc`).
5. **Forced-token fast-path** (`state.rs:163-168` + `decode_logits_seq.rs:344-356`) — emit a sole-legal token without sampling. Bypasses logit_bias, penalties, sampler.
6. **WS1 / WS2 / AM1 / A4 logit-bias modulations** at parameter-body position 0 and digit-ending mid-content (`decode_logits_seq.rs:482-564`). All additive biases on whitespace / attractor / close / `</think>` tokens.
7. **B1 margin detector** + the disabled C4v1 lift (`:566-642`).
8. **SC1 TOML repair** (`toml_repair.rs`) on `write`-tool content args.
9. **A2 fuzzy repair** (`tool_parser/fuzzy_repair.rs`) — Lev-distance prompt-vocab match.
10. **F78 tier-2 schema validator** + Tier-5c re-roll (`tool_parser/validation.rs`).
11. **F80b empty-path-write drop** for MiniMax XML on Write/Edit/MultiEdit (`parse_single_a.rs:175-206`).
12. **Stop-after-first / at_least_one** (xgrammar structural-tag extensions used via raw JSON to access the underlying flags — `compile_misc.rs:13-37`).
13. **F68 vocab auto-detection** — required for ByteLevel BPE tokenizers (Qwen, MiniMax, Mistral); vLLM gets this for free via `xgr.TokenizerInfo.from_huggingface`.
14. **Schema sanitization that silently drops a tool** (vs vLLM's 400) — tool advertised in prompt but unreachable under grammar.
15. **Wave-7 literal-`</tool_call>` recovery, empty-`{}` repair** in parsers.

## 8. vLLM-only grammar ops (no Atlas analogue)

1. **Tool extraction without grammar in auto mode** — entire path is parser-only (`serving_chat.py:1458-1480`).
2. **Grammar backends** other than xgrammar (guidance / outlines / lm-format-enforcer) selectable per server.
3. **Penalty target = prompt ∪ output** (`model_executor/layers/utils.py:78-94`). Atlas penalises output only.
4. **GPU-side `apply_token_bitmask_inplace`** (`v1/structured_output/utils.py:126`) — runs on-device, no host transfer of the bitmask back through the sampler.
5. **`structural_tag` request-shape passthrough** — caller can hand vLLM a fully-formed structural-tag JSON via `response_format.structural_tag` (`protocol.py:874-884`).
6. **Spec-decode bitmask interleave** — vLLM fills (k+1) bitmasks per spec-decode step (one per draft + bonus), accept-and-rollback across the whole speculative window (`v1/structured_output/__init__.py:184-279`). Atlas's MTP path goes through `GrammarState::rollback(n)` per-token (`state.rs:179-181`).

## 9. Assessment — does Atlas's grammar pipeline degrade vs vLLM's?

**Probability rating: 8 / 10.**

The dominant lever is **#1**: Atlas runs a structural-tag grammar for `tool_choice="auto"` while vLLM runs none. The grammar's effect is "free text allowed until the trigger; the trigger commits the model to a specific shape." For a clean, well-trained model the shape matches training and there is no cost. For an FP8-drifted model the shape forces token choices the model would have otherwise rejected: anywhere training-distribution mass sits at a token that the grammar masks out, the matcher hands the sampler a much sharper-than-training distribution, and FP8 logit noise on the surviving tokens (~23.7% of long-context positions have top1↔top2 gap < 1.5 — `research_C1_results.md`) determines the output. **This is the FP8 × grammar interaction memory has been chasing.**

Three secondary amplifiers:

- **#2 EBNF strictness** rejects values starting with whitespace or containing `<` mid-content. Cargo.toml, Rust code, and shell commands all contain `<` (generics, redirects). When the model wants to emit `serde = { version = "1", features = ["derive"] }` (no `<`) the grammar is silent; when it wants to emit `Vec<u8>` or `cargo run > out.txt` the grammar diverts. Every diversion is a chance for FP8 drift to follow.
- **#3 implicit minLength** changes the EBNF for required strings such that the very first byte is mass-forced to a non-empty content token. The first-byte distribution after `<parameter=name>` is exactly the kind of branchy decision FP8 weakens.
- **#5 forced-token fast-path** bypasses penalties and logit_bias entirely on grammar-deterministic positions. Most of the time this is harmless (structural punctuation), but the WS1/AM1 masks were designed assuming they fire — and they don't, on the very positions xgrammar already locks. Probably small effect.

What would knock the rating to ~5: a head-to-head comparison where Atlas runs `[behavior].disable_tool_grammar=true` on a Cargo.toml prompt and the cargo_valid rate climbs back to vLLM's 100%. The infrastructure for this A/B exists (`sampling_setup.rs:184-189`). Until that's run, the grammar pipeline remains the most plausible deterministic explanation for the 30%↔100% gap; the other arch_diff_* reports (MoE expert FP8 dequant, etc.) explain the *numerical* drift but not why vLLM's same-drift sampling still cargo-builds.

What would knock the rating to ~10: bytewise prompt diff confirming `tools` Jinja rendering differs between minijinja and Python jinja2 on Qwen3-Coder's template (e.g. attribute ordering of nested objects in `properties`), proving the model sees a different prompt and therefore generates a different token distribution before grammar ever runs.

Recommended next probes
1. **A/B `disable_tool_grammar=true` vs default** under the C1 harness on the same FP8 build. If cargo_valid → 100%, Item #1 is the smoking gun.
2. **Bytewise prompt diff**: render the same `(messages, tools)` payload through Atlas (`apply_chat_template_jinja`) and vLLM (`tokenizer.apply_chat_template(..., tools=tools)`) and `diff -u` the rendered strings. Targets Item Q1.
3. **Pre-grammar logit dump** at parameter-body position 0 under FP8: confirm top-1 is a whitespace token (which WS1 will demote) and that the second choice is *not* a content token but a `</` close. If true, the grammar+WS1+rep_penalty pile-up still leaves no high-probability content choice — the model is being asked to commit to an arbitrary draw from FP8 noise.
4. **Test EBNF relaxation**: rebuild the qwen3_coder grammar with `value ::= [^<\x00]+` (no leading-byte constraint, no `<` ban) and rerun C1. If cargo_valid lifts substantially, Item #2 is a contributor.
