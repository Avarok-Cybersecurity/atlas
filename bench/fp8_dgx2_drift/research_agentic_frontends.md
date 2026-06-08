# Agentic Frontends — Configurations & Known Mitigations for Open-Weight Coders

**Date:** 2026-05-25
**Context:** Atlas user runs Qwen3.6-35B-A3B-FP8 through **opencode**. Atlas already implements `wants_typed_arguments=true` for the qwen3_coder parser (per `tool_parser/qwen3_coder.rs:28`). This survey identifies what *other* frontends do and the highest-leverage server-side mitigations.

---

## 1. opencode (sst/opencode) — Primary Target

### 1.1 Bash tool schema (zod)

The bash tool schema exposed to the model is (verified from issue #13146):

```ts
parameters: zod.object({
    command:     zod.string().describe("The command to execute"),
    timeout:     zod.number().describe("Optional timeout in milliseconds").optional(),
    workdir:     zod.string().describe("...").optional(),
    description: zod.string().describe("Clear, concise description...")  // REQUIRED — no .optional()
})
```

**Critical numeric field: `timeout` is `z.number()` in *milliseconds*.** If the model emits `"timeout": "30"` (string) it 400s. If it emits `"timeout": 30` (seconds, integer) it works but kills the command at 30ms. Atlas's typed-args coercion correctly converts `"30"` → `30` but Atlas cannot know the semantic unit (ms vs s) — that's a model-prompt problem.

**`description` is *required* but undocumented in the schema shown to the model** (the issue's whole point). Every well-tuned model occasionally forgets it. Atlas can't fix this server-side, but the qwen3_coder system prompt should be reviewed.

Other built-in opencode tools (write, edit, read, grep, glob, apply_patch, todowrite, webfetch, websearch): all use zod with mostly string parameters. The numerically-typed ones are: `read.offset`, `read.limit`, `read.pages`-ish, and `bash.timeout`. (See `opencode.ai/docs/tools/`.)

### 1.2 Retry / SchemaError behavior

Today (per issues #1388, #15906, #29142, #24604, #17169):

- **A SchemaError aborts the turn.** opencode does *not* feed the validation error back to the model. Open feature request #1388 ("Auto Tool Failure Retry") has been open since 2025-07.
- Subagents can enter **infinite retry loops** on edit/write SchemaError (#17169).
- Issue #15906: explicit request to "retry invalid tool-call diff / malformed tool input instead of aborting."
- Issue #29142: write/edit get called with the wrong key (`fileContent` instead of `content`) — opencode just dies.

**Implication for Atlas:** because opencode does not echo the schema error back, the *only* defense is to ship a wire-protocol-correct tool call on the first try. This is exactly what `wants_typed_arguments` was added for.

### 1.3 Known Qwen3 issues in opencode

- #1809 (qwen3-coder-30B): "not able to call any tool" — root cause was missing `--enable-auto-tool-choice`/`--tool-call-parser qwen3_coder` on the server side.
- #4255 (LM Studio + Qwen): **hangs indefinitely on empty `tool_calls: []` arrays**. Atlas should never emit an empty `tool_calls` array — omit the field entirely when none are present.
- #8184 (SGLang): strict JSON schema validation fails on **empty tool parameter objects**. Atlas must avoid emitting `"parameters": {}` for no-arg tools — use the schema as declared.
- #17750: `function.arguments` may contain invalid JSON (e.g., trailing XML close-tag bleed). Atlas's qwen3_coder parser already strips these in `parse_single_b.rs` but reviewers should grep for `</parameter>` bleed.

### 1.4 No XML → JSON history conversion

opencode does **not** convert qwen3_coder XML in the assistant's `content` into a structured `tool_calls` array for the chat history. If the model leaks a bare `<tool_call>` into content (Atlas's `leak_markers` already suppresses this), opencode treats it as text and the next turn sees `<tool_call>` in history — which conditions the model to keep doing it. Atlas's leak detector is the right defense.

---

## 2. Cline / Roo Code / Goose / Continue / Aider / qwen-code — How others do it

| Frontend | Tool format | Retry on schema error | Notes for Qwen3-Coder |
|---|---|---|---|
| **Cline** | OpenAI-compatible (JSON tool_calls) over `chat/completions`. Configured by setting OpenAI-Compatible provider + base URL. | Best-effort: surfaces error to model in next turn ("the previous tool call failed") — does *not* abort the agent. | Works against `--tool-call-parser qwen3_coder` if the server returns OpenAI JSON. |
| **Roo Code** (Cline fork) | Same as Cline; can also drive Qwen Code CLI directly. | Same surface-and-retry. Recommended sampler in docs: `temperature=0.25, top_p=0.9, num_ctx ≥ 65536`. | Issue #11219: explicit support for Qwen3-Coder-Next added. #6630: "raw text for thinking and tool calls" when OpenRouter strips the XML — same XML-in-content failure mode. |
| **Goose** (Block) | Native JSON tool calling. | Retries with model feedback. | Issue #6883: "Tool calling fails with many tools via Ollama" — Qwen3-Coder switches from JSON tool_calls to *XML in the content field* when >5 tools are presented. **Highly relevant: this is the canonical failure mode opencode hits with Atlas.** |
| **Continue.dev** | Converts tools to XML in the *system* message and parses XML back out — exactly the qwen3_coder native protocol. Set `capabilities: [tool_use]` per-model. | Streams XML, parses, then surfaces parse errors. | Issue #5419 / #8744: tool-loading 404 against vLLM endpoints that don't advertise the tool-use capability flag. |
| **Aider** | Uses litellm; not "agentic" in the same way (it's edit-format-driven, not free-form tools). | N/A. | `model-settings.yml` per-model: `extra_params: { temperature: 0.7, top_p: 0.8, top_k: 20, repetition_penalty: 1.05 }`, `edit_format: diff`, `use_repo_map: true`. |
| **qwen-code** (official CLI) | Native qwen3_coder XML. | Built-in retry; sends explicit "rethink your tool call" follow-up. | Reference implementation of correct behavior. |
| **Claude Code** | Anthropic `/v1/messages` only — does not target open models. | Anthropic's API does its own schema validation server-side and 400s. | Reference for tool schema shape (e.g. Anthropic's Bash tool `timeout` is `integer` in seconds). |

### 2.1 Community consensus on Qwen3-Coder samplers

All sources (Unsloth, Qwen vLLM recipe page, the muxup parameter reference, Roo Code docs) converge on the **vendor recommendation**:

```
temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05
```

Some Roo Code users prefer `temperature=0.25, top_p=0.9` for stricter tool adherence. Aider's groq Qwen entries use `temperature=0.6, top_p=0.95, top_k=20`.

`presence_penalty=0..2` is mentioned as a remedy for "endless repetitions" but with the warning that high values cause language mixing.

### 2.2 Community-found prompt patterns for long multi-turn

- **Trim system prompts aggressively.** llama.cpp discussion #20000 (Qwen3-Coder-Next, ~80k context repetition/hallucination): user dropped system+tools from 8k → 4k tokens and the repetition stopped.
- **Update the Jinja template.** Unsloth's GGUF discussion #10 requires guards on `tool.parameters.properties is defined and is mapping` — without these the template crashes on no-arg tools. Atlas should verify its Jinja templates in `crates/spark-server/src/api/chat/template.rs`.
- **Higher quant = better tool calling.** Q5_K_XL / Q6_K_XL meaningfully outperform Q4_K_XL on tool calling. Atlas's FP8 path is roughly comparable to Q6+ in fidelity — Atlas should not be at a disadvantage here.

### 2.3 Tool-count threshold

Goose issue #6883 documents that **Qwen3-Coder switches from native JSON to XML-in-content when >5 tools are provided**. This is *the same mode collapse* you see when opencode shows ~12 tools to the model. The model "regresses" to its training distribution (XML output) under tool pressure. Atlas's qwen3_coder parser is already prepared to handle XML output — the design is correct.

---

## 3. Tool format conversion — what others do that Atlas doesn't (yet)

| Frontend | What it converts | Where |
|---|---|---|
| Continue | XML → JSON before recording in chat history. Model sees its own past tool calls as XML in the next system message render. | Custom XML parser |
| Goose | Falls back to parsing XML from `content` when tool_calls is empty. | Plugin layer |
| vLLM `qwen3_coder` parser | Parses XML → emits OpenAI `tool_calls` JSON on the wire. | Server side |
| **Atlas qwen3_coder parser** | Already does the above (`format_tool_calls` re-renders as XML on prompt build; `parse_*` decodes XML → OpenAI JSON on response). | `tool_parser/qwen3_coder.rs:131-150` |

Atlas is structurally aligned with the SOTA. The only gaps are **typed-argument coercion** (now landed) and **schema-aware coercion** (Atlas currently uses heuristic coercion; using the tool's declared JSON-schema would be tighter).

---

## 4. Server-side configurations Atlas should support

Listed in priority order for opencode compatibility:

1. **Schema-aware typed-argument coercion** (status: shipped 2026-05-25). When the tool schema declares `timeout: number`, coerce string `"30"` → `30`. Already present via `wants_typed_arguments`. **Extend** to use the *declared* JSON schema from the tool definition rather than heuristic detection — this catches the bash `timeout` field by name.
2. **Never emit empty `tool_calls: []` or empty `parameters: {}`.** Both break opencode (#4255) and SGLang strict-schema (#8184). Audit Atlas's `openai/chat_response.rs` to omit the field instead of sending an empty array/object.
3. **Strip `</tool_call>` / `</parameter>` bleed from `function.arguments`.** (#17750.) Atlas's `parse_single_b.rs` already trims these, but add an explicit assertion in tests.
4. **Hermes parser** and **qwen3_xml parser** dispatch from a single content-classifier. Atlas already does this in `parse_dispatch.rs` — keep it that way. The vLLM community currently disagrees on whether `qwen3_coder` or `qwen3_xml` parser is correct; Atlas should support both.
5. **Server-side retry hook for schema validation.** If the tool's declared schema rejects the assembled args, Atlas could attempt a single in-server re-roll (re-sample with the schema as a constraint via XGrammar) before returning the response. This would *invisibly* fix the opencode no-retry gap for ~5-10% of failing calls (the rate quoted in opencode #1388).
6. **`description` field hinting in the qwen3_coder system prompt.** Atlas's prompt should explicitly say "for `bash`/`Bash`, ALWAYS include both `command` AND `description`" — the same pattern Atlas already uses for Write's `file_path`+`content` rule (`qwen3_coder.rs:118`).
7. **Sampler defaults.** When `MODEL.toml` declares the model family is Qwen3-Coder-like, default to `temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05` (matching vendor + Unsloth recommendation) unless the request overrides. Atlas already exposes per-model sampler defaults; just confirm Qwen3.6 inherits these.
8. **Anthropic `/v1/messages` parity** for clients that prefer it (Cline can speak either; opencode is OpenAI-format only today, but the changelog hints at Anthropic-format support being added for Claude Code compatibility).

---

## 5. Atlas cross-reference

- `crates/spark-server/src/tool_parser/qwen3_coder.rs:28` — `wants_typed_arguments=true` (shipped today).
- `crates/spark-server/src/tool_parser/qwen3_coder.rs:45-127` — system prompt; F33 bash retry rule; IMMEDIATE_TOOL_USE block; IMPORTANT block already covers write/edit traps.
- `crates/spark-server/src/tool_parser/qwen3_coder.rs:131-150` — `format_tool_calls` (Atlas → wire XML).
- `crates/spark-server/src/tool_parser/qwen3_coder.rs:152-200` — leak_markers; suppresses bare `<tool_call>`, `<function=`, `<parameter=`, `<tool_response>`, `<function_results>`.
- `crates/spark-server/src/tool_parser/type_coerce.rs` — `coerce_all` (shared with qwen3_xml).
- `crates/spark-server/src/openai/chat_response.rs` — audit for empty `tool_calls`/`parameters`.

---

## 6. Sources

- opencode: issues #1197, #1388, #1721, #1809, #3011, #4255, #4788, #8184, #11313, #13146, #13900, #14519, #15080, #15906, #16993, #17169, #17750, #20902, #24604, #25430, #29142; `opencode.ai/docs/tools/`, `/docs/config/`, `/docs/troubleshooting/`.
- Cline: docs.cline.bot/provider-config/qwen-code.
- Roo Code: docs.roocode.com/providers/qwen-code; issues #6630, #11219.
- Goose: issues #3748, #6883.
- Continue: docs.continue.dev/customize/deep-dives/model-capabilities; issues #5419, #6860, #8744.
- Aider: aider.chat/docs/config/adv-model-settings.html; resources/model-settings.yml.
- Unsloth GGUF discussion #10 (Qwen3-Coder-30B chat template & sampler).
- vLLM recipe pages for Qwen3-Coder and Qwen3.5/3.6.
- llama.cpp discussion #20000 (long-context Qwen3-Coder-Next repetition).
- muxup.com "Vendor-recommended LLM parameter quick reference" (2025q2).
