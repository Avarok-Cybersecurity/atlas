# Internal-Server Audit — Anti-Patterns and Atlas-Specific Departures on Qwen3.6-35B-A3B-FP8

**Date**: 2026-05-26
**Scope**: chat orchestrator, jinja-templates, sampler/MTP,
tool_parser, grammar/compile_tools, openai response shape, MODEL.toml.
**Method**: every file in the task list, cross-referenced with
research_synthesis.md, the vLLM/SGLang pipeline notes, and the
MISSION_PROGRESS epoch log.

The opencode failure mode (drift to semantically-wrong tool args by
turn 5+) is consistent with a layered profile: (a) chat-template
inputs that diverge from training distribution after each turn, (b)
sampler/grammar interlocks that constrain shape but not semantics, (c)
late workarounds that quietly bypass earlier fixes.

---

## A. Chat-template inputs depart from upstream Qwen3.5/3.6 expectation

**A1. `reasoning_content` round-trip is broken (multi-turn root cause).**
`openai/chat_message.rs:6-19` defines `IncomingMessage` with `role`,
`content`, `tool_calls`, `tool_call_id`, `name` — and no
`reasoning_content` field. Atlas emits `reasoning_content` on
responses (`chat_response.rs:218`, `annotations.rs:12`) but cannot
read it back. The Qwen3.5/3.6 jinja template
(`jinja-templates/qwen3_5_moe.jinja:90-104` and the openai variant)
explicitly handles `message.reasoning_content` — if set, it renders
`<think>\n{reasoning}\n</think>` for prior assistant turns. Atlas's
`chat/msg_entry.rs:86-110` only forwards `tool_calls`. The template
falls into the `'</think>' in content` branch which also fails
because Atlas strips think tags from streamed content
(`chat_stream/handle_token.rs:344-355`, `chat_blocking.rs:227-269`).
Net: every multi-turn replay loses prior reasoning. With
`thinking_in_tools=true` (MODEL.toml:176), the template still renders
an EMPTY think wrapper for prior turns, training the model — token by
token — to compress its reasoning toward empty across turns. vLLM and
SGLang both preserve `reasoning_content` on incoming messages.

**A2. Doubled empty `<think>\n\n</think>\n\n` injection.** With
`thinking_in_tools=true`, `chat/msg_entry.rs:73-80` prepends
`<think>\n\n</think>\n\n` to every assistant message whose
`msg_idx > last_query_index`. The Jinja template ALSO emits
`<|im_start|>assistant\n<think>\n + '' + \n</think>\n\n + {content}`
at `qwen3_5_moe.jinja:101` for the same turns. Result: nested empty
think markers in the rendered prompt on every post-last-user
assistant turn. This is exactly the "chat-template artefacts" Phase-1
retest observed in MISSION_PROGRESS. No SFT data contains nested
empty thinks; attention has nothing to ground on at these positions.

**A3. `enable_thinking: false` from the client is silently ignored.**
`openai/chat_request.rs:404-406`: only truthy `enable_thinking`
counts. `thinking_explicitly_requested()` returns false for an
explicit false; the request falls through to MODEL.toml
`thinking_default=true` and the model thinks anyway. vLLM and SGLang
honor explicit false.

**A4. `tool_choice="none"` drops the entire `# Tools` block from
system.** `api/chat/mod.rs:97-99` and `chat/template.rs:71-80` set
`jinja_tools = None` when `tools_active=false`, so the Qwen template
skips lines 46-53 (the tool definition + format instructions). A
client emitting `tool_choice: "none"` mid-conversation has the model
suddenly receive a system message that never mentions tools it
already half-used. vLLM passes the list through regardless.

**A5. Vacuous-system heuristic silently mutates input**
(`chat/msg_entry.rs:186-196`). The strip is empirical (Open WebUI
2026-05-17). With Qwen's tool-format guidance living in the system
slot, dropping the FIRST message shifts indices and the
`loop.first` branch in the template misfires.

**A6. `disable_tool_steering` is plumbed but never consumed**
(`tokenizer/chat_impl.rs:155`): dead variable in the minijinja
context.

**A7. The openai-variant template is the ONLY thing gating the
historical `<think>` wrapper on `enable_thinking`**
(`jinja-templates/openai/qwen3_5_moe.jinja:100` vs the base at line
100). If `openai/qwen3_5_moe.jinja` ever fails to ship in a Docker
image, `tokenizer/chat_impl.rs:181-219` falls back silently to the
base template, regressing every multi-turn flow.

---

## B. Sampler / scheduler departures

**B1. `forced_token_fastpath` bypasses every logit_bias and penalty.**
`scheduler/decode_logits_seq.rs:280-327` returns the grammar's sole
legal token directly when grammar narrows to one option. Only Tier-1
(empty-parameter) is gated at line 315. The exponential `<tool_call>`
bias decay (`sampling_setup.rs:99-108`), the `suppress_tool_call -12.0`
bias (`decode_logits_seq.rs:166-173`), the client's `logit_bias`,
DRY, repetition/presence/frequency penalty, adaptive sampling — all
no-ops when the grammar pins one token. The grammar pins one token
often inside `<parameter=KEY>` openers, `</tool_call>` close, JSON
punctuation. Under FP8 drift the grammar is what locks the model in
and every corrective bias is bypassed.

**B2. `repetition_penalty_window=256` hardcoded at 4 sites**
(`prefill_a_step.rs:234,310,390` + `lifecycle.rs:180,265`). MODEL.toml
has no knob for it. With Qwen3.6 plus rep_penalty=1.1 a 256-token
window penalizes legitimate recurrences of file paths, language
keywords, tool names. SSOT/PCND violation.

**B3. Suppress-tool-call bias is -12.0, not -inf**
(`decode_logits_seq.rs:166-173`). Comment: "so the model can still
escape if its evidence for a tool call is overwhelming." Under FP8
drift the model's evidence is exactly what is unreliable. vLLM hard-
masks. Soft-bias is the wrong primitive when the thing you don't
trust is the model's certainty.

**B4. Whitespace mask set is 5 tokens** (`emit_step.rs:163-166` and
`decode_logits_seq.rs:431-440`) — 220, 198, 197, 256, 271. Qwen
byte-level BPE has dozens of whitespace-only tokens. The comment
acknowledges "not bulletproof"; the proper boot-time vocab scan
recommended by research_synthesis A7 was never built.

**B5. MTP verify pipeline replays the full pipeline per position**
(`scheduler/verify_pipeline_helper.rs:131-206`), speculatively
advancing the grammar per pick. When forced-token fastpath (B1)
returns terminal-grammar tokens inside verify, the speculative
advance can desync — line 182 logs "stale bitmask in the pipeline."
Mostly self-recovering at K=2 but a SSOT violation: the verify
pipeline must remain byte-identical with `process_seq_logits` and
diverges in implementation.

**B6. `process_seq_logits` and `verify_pick_with_pipeline` are two
implementations of the same algorithm.** Comment at
`verify_pipeline_helper.rs:21` admits "mirroring." Any new mask risks
a one-sided landing.

---

## C. Grammar / tool-parser anti-patterns

**C1. qwen3_coder EBNF rejects ALL `<` bytes in parameter values.**
`grammar/compile_tools.rs:248-254`:
```
value ::= first_char rest
first_char ::= [^ \t\r\n<]
rest ::= [^<]*
```
The comment acknowledges Rust generics, shell redirection and HTML
will be refused. When the model samples a `<` mid-value, `accept_token`
returns false (`emit_step.rs:220-227`) and the sequence terminates.
Lines 333-339 duplicate the same EBNF in the fallback path (SSOT).

**C2. `wants_typed_arguments=true` only fires in the BLOCKING path**
(`tool_parser/qwen3_coder.rs:28-30`, `chat_blocking.rs:310-314`).
Streaming emits one `ToolCallDelta` carrying full canonical JSON at
`</tool_call>` close (`streaming_impl.rs:74-80`); coercion would have
to happen in `chat_stream/tool_handlers.rs`. opencode runs streaming
— SchemaError-class failures hit only that path.

**C3. `validate_tool_calls` strictness varies by tool family.**
`validation.rs:394-422` enforces "absolute or relative, ≥3 chars"
for `WRITE_FAMILY`. `FILE_TOOLS` (Read/Glob/LS) at line 452 does NOT
enforce non-empty for Theia compatibility. Two validators for similar
tools, keyed on a name string rather than declared schema property.

**C4. `backfill_required_params` silently rewrites empty descriptions**
(`validation.rs:138-172`) — "Run: {command}" overwrites the model's
own decision to omit a description. Multi-turn, the model learns
descriptions don't have to match commands.

**C5. `safe_emit_len` holds opener prefixes but not closer prefixes**
(`tool_parser/streaming_impl.rs:368-397`). Documented in
research_synthesis 2a as the axum-v51 → axums_v51 mutation class.
For the active detector this is moot inside a tool envelope, but the
no-detector branch (Anthropic adapter, Mistral) inherits the same
gap.

**C6. `disable_tool_grammar` MODEL.toml flag** turns OFF tool-call
grammar entirely (`sampling_setup.rs:184-189`). Off for Qwen3.6 today;
documented escape hatch without any quality cross-link.

---

## D. Tool-call streaming, reasoning, MTP interactions

**D1. Reasoning emitted as both `reasoning_content` and `reasoning`**
(`chat_response.rs:218-222`, `:267-289`). Clients that read both
double the thinking text in their display.

**D2. SimHash semantic-loop guard sees MTP bonus tokens.**
`chat_stream/handle_token.rs:494-510` runs SimHash on every emitted
content token. MTP K=2 commits two tokens per verify step; the bonus
token's generation context is the PROPOSE distribution. SimHash's
classifier was tuned on non-MTP output. Cancel races between SimHash
and MTP propose can truncate mid-sentence.

**D3. Stop tokens for tool turns include `</tool_call>` as a SAMPLER
stop** (`sampling_setup.rs:128-133`). This is in addition to the
grammar's stop-after-first behavior. Two enforcement layers for the
same boundary, both load-bearing — when grammar disagrees with the
sampler at `</tool_call>` (e.g. after a forced-token fastpath has
advanced grammar to terminal but the sampler still has the token
listed as stop), the sampler wins and the request ends with a
partial envelope.

**D4. The cooperative `cancel_flag` is shared between stream and
scheduler** (`chat_stream/handle_token.rs:74-75`, `emit_step.rs:23-28`),
but `chat_stream/handle_token.rs:42` sets a 256-token suppress
streak before flipping the flag. Inside that 256-token window the
scheduler keeps generating tokens that are then suppressed by the
stream — wasted decode for the user, FP8 drift accumulates on a path
no one will read.

---

## E. Sampling presets and MODEL.toml

**E1. `[sampling.tools]`** (MODEL.toml:113-120): `temperature=0.6`,
`presence_penalty=0.0`, deviating from the Qwen team's recommended
`presence_penalty=1.5` outside tool body. The comment lines 76-83
acknowledges the deviation but never solved the underlying loop
attractor.

**E2. Inside-body penalty zeroing is total** (`decode_logits_seq.rs:405,
449-466`): `in_tool` zeros `repetition_penalty`, `presence_penalty`,
`frequency_penalty`, `lz_penalty`, `dry_multiplier`. Inside a tool
body the model has NO protection against pathological attractors —
the `Cargo.toml.new.tmp.bak1.tmp.newnewnew.tmp.oldoldold`
filename-garbage observed 2026-05-25 lives entirely inside one
`<parameter=file_path>` body.

**E3. `max_thinking_budget=768`** (MODEL.toml:206) with sentence-defer
ceilings (`decode_logits_seq.rs:212-235`) can fire `</think>` mid-
paragraph on opencode-style long-reasoning requests, leaving a
truncated content tail.

**E4. `default_num_drafts=1` was benchmarked 2026-04-10**
(MODEL.toml:228), pre-MoE-watchdog and pre-grammar changes. No
re-benchmark documented.

---

## F. Cross-cutting comparison with upstream engines

| Concern | Atlas | vLLM / SGLang |
|---|---|---|
| `reasoning_content` on incoming | dropped | preserved |
| `enable_thinking: false` explicit | ignored | honored |
| Multi-turn `<think>` template | empty + nested | client-controlled |
| Forced-token fastpath bypasses bias | yes | no |
| Tool grammar value | `[^<]*` | full JSON / utf8 |
| Repetition window | 256 hardcoded | per-request |
| Suppress-tool-call bias | -12.0 soft | -inf hard |
| Tool args streaming | one chunk at close | per-key delta |
| `reasoning` + `reasoning_content` | both emitted | one canonical |

---

## Ranked Top-5 CONCRETE Atlas-internal anti-patterns

### #1 — `reasoning_content` round-trip is structurally absent
**Severity**: CRITICAL — primary contributor to turn-5+ semantic drift.
- `crates/spark-server/src/openai/chat_message.rs:6-19` —
  `IncomingMessage` has no `reasoning_content` field.
- `crates/spark-server/src/api/chat/msg_entry.rs:86-110` — message
  rebuild drops it.
- `jinja-templates/qwen3_5_moe.jinja:90-104` (and openai variant) —
  template expects it; receives undefined; emits empty wrapper.
**Effect**: every multi-turn replay loses prior reasoning; model
learns to compress thinking toward empty across turns.

### #2 — Doubled empty `<think>\n\n</think>\n\n` injection
**Severity**: CRITICAL — direct training-distribution corruption.
- `crates/spark-server/src/api/chat/msg_entry.rs:73-80` — prepends
  empty think wrapper to historic assistant content.
- `jinja-templates/openai/qwen3_5_moe.jinja:100-104` — template ALSO
  emits a think wrapper for the same turns.
- `kernels/gb10/qwen3.6-35b-a3b/MODEL.toml:176` — `thinking_in_tools=true`
  triggers both paths simultaneously.
**Effect**: nested empty think markers in every post-last-user
historic assistant turn — pure off-distribution noise.

### #3 — `forced_token_fastpath` bypasses all logit_bias and penalties
**Severity**: HIGH — defeats every server-side correction.
- `crates/spark-server/src/scheduler/decode_logits_seq.rs:280-327` —
  fast-path returns the grammar-forced token without running the
  sampling pipeline.
- Only the Tier-1 empty-parameter gate at line 315 protects one case.
**Effect**: under FP8 drift the grammar is what locks the model in;
all corrective biases (suppress_tool_call -12.0, client logit_bias,
DRY, repetition penalty) are no-ops at every grammar-forced position.

### #4 — qwen3_coder EBNF tool body rejects all `<` bytes
**Severity**: HIGH — silent grammar-desync on legitimate code.
- `crates/spark-server/src/grammar/compile_tools.rs:248-254`
  (`first_char ::= [^ \t\r\n<]`, `rest ::= [^<]*`) and `:333-339`
  (duplicated fallback).
- `crates/spark-server/src/scheduler/emit_step.rs:220-227` —
  `accept_token` returns false on `<`; sequence ends.
**Effect**: opencode coding turns containing Rust generics, shell
redirect, comparison operators, or HTML truncate at the first `<`
byte. Retries inherit the constraint and emit 1-char garbage instead
(Epoch 3 finding).

### #5 — `repetition_penalty=1.1` + zero-penalty-inside-body + DRY-off
**Severity**: HIGH — doom-loop attractor class.
- `kernels/gb10/qwen3.6-35b-a3b/MODEL.toml:113-120` — global
  rep_penalty=1.1, dry_multiplier=0.5.
- `crates/spark-server/src/scheduler/decode_logits_seq.rs:405,449-466`
  — `in_tool = inside_tool_body && !inside_thinking` zeros every
  penalty inside `<tool_call>`.
- `crates/spark-server/src/scheduler/prefill_a_step.rs:234,310,390`
  and `lifecycle.rs:180,265` — `repetition_penalty_window=256`
  hardcoded.
**Effect**: inside a tool body the model has NO protection against
attractors (`Cargo.toml.new.tmp.bak1.tmp.newnewnew.tmp.oldoldold`);
outside, the 256-token window penalizes natural recurrences of paths
and keywords. Both directions wrong.

---

## Concluding observation

The 0.04 BF16-LUT-to-BF16-unquant cosine gap is real but small.
Multi-turn failure is not numeric — it is the compounding of nine
independent template/sampler/grammar departures from upstream Qwen
guidance and from the vLLM/SGLang norm. MISSION_PROGRESS Epochs 1-6
tightened structural enforcement at grammar+sampler+validator layers;
the unresolved layers are PROMPT FIDELITY (A1, A2, A3) and
LOGIT-BIAS RELIABILITY (B1). Fixing those two classes restores the
training distribution the FP8 model was tuned for and should close
the bulk of the observable drift without further numeric work.
