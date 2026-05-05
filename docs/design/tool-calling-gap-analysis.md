# Tool Calling Gap Analysis: Atlas vs vLLM

**Date**: 2026-03-22
**Models**: Qwen3.5 (35B/122B), Nemotron-H (30B/120B)

---

## Executive Summary

vLLM uses Python Jinja2 (via HuggingFace transformers) for 100% template compatibility, passes full structured messages (including `tool_calls` as parsed dicts) to the template, and has model-specific tool parsers with streaming support. Atlas uses Rust minijinja (~95% compatible), pre-formats tool_calls as text before template rendering (bypassing the template's own tool_call logic), and has robust fallback parsing but lacks some vLLM features.

**Key gaps** (ordered by impact):
1. Multi-turn tool_calls not passed as structured data to Jinja template
2. No grammar enforcement for `tool_choice=required` (prompt-only)
3. Thinking suppression not coordinated with tool calling
4. No type coercion for XML parameter values (all treated as strings)
5. Minor minijinja vs Jinja2 compatibility edge cases

---

## 1. Chat Template Rendering

### vLLM Approach
- Uses **Python Jinja2** (`jinja2.sandbox.ImmutableSandboxedEnvironment`) via HuggingFace `tokenizer.apply_chat_template()`
- 100% Python Jinja2 compatibility — every filter, test, and syntax feature works
- Passes `tools` as structured Python dicts to the template context
- Passes full `ConversationMessage` objects including `tool_calls`, `reasoning_content`, `tool_call_id`
- Pre-processes `tool_calls[].function.arguments` from JSON strings to Python dicts before passing to template (critical for `|items` iteration)
- Template resolution: tokenizer_config.json → processor → fallback templates

### Atlas Approach
- Uses **Rust minijinja v2** with custom filters (`rtrim`, `ltrim`, `split_first`, `split_last`)
- ~95% Jinja2 compatible — most features work, but some Python-specific constructs need conversion
- `convert_python_jinja_to_minijinja()` translates: `[::-1]→|reverse`, `.startswith()→is startingwith`, `.rstrip()→|rtrim`, `.split()[0]→|split_first`
- Passes `tools` as structured JSON values to template context (correct)
- Template resolution: override dir → tokenizer_config.json → standalone .jinja → default ChatML

### Gap Analysis

| Feature | vLLM | Atlas | Gap |
|---------|------|-------|-----|
| Jinja engine | Python Jinja2 | Rust minijinja v2 | Minor — minijinja covers 95%+ |
| `tools` context | Structured dicts | Structured JSON values | **No gap** — both work |
| `tool_calls` in messages | Parsed dict objects | **Not passed** — pre-formatted as text | **MAJOR GAP** |
| `reasoning_content` | Structured field | Not passed to template | Medium gap |
| `|items` filter | Native Python | minijinja `|items` filter | **No gap** |
| `|tojson` filter | Native Python | minijinja with `json` feature | **No gap** |
| `.split()` method | Native Python | Custom filter workarounds | Minor gap |
| `.startswith()` | Native Python | `is startingwith` test | Minor gap |
| `namespace()` | Native Jinja2 | Supported in minijinja v2 | **No gap** |
| `loop.previtem/nextitem` | Native Jinja2 | `adjacent_loop_items` feature | **No gap** |

### Actionable Items

**P0 — Pass tool_calls as structured data to Jinja template:**
Currently, Atlas converts multi-turn assistant messages with tool_calls like this:
```
content = "I'll search for that." + "\n<tool_call>\n<function=search>..."
messages = [{"role": "assistant", "content": content}]
```
But the Jinja templates expect:
```
messages = [{"role": "assistant", "content": "I'll search for that.", "tool_calls": [{"function": {"name": "search", "arguments": {"query": "..."}}}]}]
```
This means the template's tool_call rendering code (lines 105-130 in qwen3_5.jinja, lines 121-159 in nemotron_h.jinja) is never exercised. The tool_calls are embedded in content text instead.

**Fix:** Build `json_messages` with `tool_calls` and `reasoning_content` fields. Parse `arguments` from JSON string to object before passing (matching vLLM's `_postprocess_messages`).

**P1 — Remove duplicate tool_call formatting:**
Currently Atlas has TWO formatting paths:
1. `parser.format_tool_calls()` — pre-formats tool_calls as text into content (api.rs line 83)
2. Jinja template tool_call rendering — handles `message.tool_calls` structured data

Only #2 should be used when the Jinja template handles tools. The parser's `format_tool_calls()` should only be used for legacy templates that don't support structured tool_calls.

---

## 2. Tool Call Parsing (Model Output → API Response)

### vLLM Approach
- **35 model-specific parsers** in `vllm/tool_parsers/`
- `Qwen3CoderToolParser`: 712-line Python class with regex-based XML parsing
- `Hermes2ProToolParser`: JSON parsing with `partial_json_parser` library
- Type coercion: converts XML parameter values based on JSON schema types (int, float, bool, object, array)
- Security fix: removed `eval()` call (CVE GHSA-79j6-g2m3-jgfw)
- Streaming: token-by-token state machine with delta accumulation
- Streaming for Qwen3Coder: tracks current_tool_index, header_sent, json_started, json_closed states
- PR #35347 added `Qwen35CoderToolParser` for Qwen3.5-specific streaming fixes

### Atlas Approach
- 2 parser implementations: `HermesParser`, `Qwen3CoderParser`
- Format-agnostic output parsing via `parse_tool_calls()` with 4-level fallback:
  1. `<tool_call>` tags (primary)
  2. `<tools>` tags (NVFP4 variant fallback)
  3. Bare `<function>` tags (no wrapper fallback)
  4. JSON in code blocks (catch-all fallback)
- `StreamingToolDetector`: buffers text, detects `<tool_call>` tags, emits events
- All parameter values treated as strings (no type coercion)
- Robust error handling for NVFP4 artifacts (`<|function=` pipe prefix)

### Gap Analysis

| Feature | vLLM | Atlas | Gap |
|---------|------|-------|-----|
| Parser count | 35 model-specific | 2 + fallbacks | By design — Atlas uses universal parsing |
| Streaming detection | Token-level state machine | Text-level buffer + tag detection | Different approach, both work |
| Type coercion | Schema-aware (int, float, bool, dict) | All strings | **Gap** — clients may need typed values |
| `partial_json_parser` | Yes (hermes) | Manual truncated JSON recovery | Minor gap |
| NVFP4 artifact tolerance | None (not needed on H100) | Extensive (`<\|function=`, pipe prefix) | Atlas advantage |
| Multi-format fallback | Parser-specific | 4-level universal fallback | Atlas advantage |
| Spec-decode streaming fix | PR #35615 (param loss) | N/A (different streaming approach) | No gap |
| Security | Fixed eval() CVE | No eval(), safe by design | Atlas advantage |

### Actionable Items

**P2 — Add schema-aware type coercion:**
vLLM's `Qwen3CoderToolParser._convert_param_value()` converts parameter values based on the tool's JSON schema:
- `"type": "integer"` → `int(value)`
- `"type": "number"` → `float(value)`
- `"type": "boolean"` → `value == "true"`
- `"type": "object"/"array"` → `json.loads(value)`

Atlas currently returns all values as strings, which works for most clients (OpenCode, Cline) but may break strict schema validation in some clients.

**Fix:** Add optional type coercion in `parse_qwen3_coder_call()` that checks the tool schema and converts values. Keep string as default for safety. Use `serde_json::from_str` for object/array, never `eval()`.

---

## 3. Grammar-Constrained Decoding for Tool Calls

### vLLM Approach
- `tool_choice=required` triggers structured output enforcement via JSON grammar
- Uses xgrammar (default) or outlines/lm-format-enforcer backends
- **Known issue**: enforces JSON grammar even for XML-format models (Qwen3-Coder), degrading performance
- Feature request #27766: read format constraints from tool parsers (not yet implemented)
- No grammar enforcement specific to tool_choice — all grammar enforcement is via `adjust_request()` setting `structured_outputs.json`

### Atlas Approach
- xgrammar-rs integration with structural tags
- `compile_hermes_tool_grammar()`: creates grammar for `<tool_call>{"name":"fn","arguments":...}</tool_call>`
- `compile_qwen3_coder_tool_grammar()`: creates grammar for `<tool_call><function=fn>...</function></tool_call>`
- `use_triggers=true` (auto): free text before trigger prefix
- `use_triggers=false` (required): enforces tool format from token 1
- Per-token bitmask application with MTP rollback support

### Gap Analysis

| Feature | vLLM | Atlas | Gap |
|---------|------|-------|-----|
| Grammar enforcement | JSON schema (wrong for XML models) | Format-specific structural tags | **Atlas advantage** |
| tool_choice=required | Triggers JSON grammar | Triggers format-specific grammar | **Atlas advantage** |
| tool_choice=auto | No grammar (pure model decision) | Free text + trigger prefix | **Atlas advantage** |
| XGrammar integration | Server-wide, all models | Server-wide, per-request compilation | Comparable |
| MTP spec-decode rollback | N/A for grammar | `GrammarState::rollback(n)` | Atlas advantage |
| Grammar caching | Yes | Yes (via xgrammar compiler cache) | Comparable |

Atlas is ahead here. vLLM's grammar enforcement actually harms Qwen3-Coder models by forcing JSON format on an XML-expecting model.

### Actionable Items

**No major gaps** — Atlas's grammar implementation is more sophisticated than vLLM's for tool calling. However:

**P3 — Verify grammar works end-to-end with Qwen3.5:**
The grammar infrastructure exists but needs integration testing. Ensure `compile_qwen3_coder_tool_grammar()` is actually invoked when `tool_choice=required` is set for Qwen3.5 models with `--tool-call-parser qwen3_coder`.

---

## 4. Thinking and Tool Calling Interaction

### vLLM Approach
- `enable_thinking` passed to Jinja template context via `chat_template_kwargs`
- Template controls `<think>` generation prompt
- Reasoning parser (e.g., `--reasoning-parser qwen3`) extracts `<think>` content into `reasoning_content` response field
- **Known issue** (Mar 2026): `enable_thinking=false` doesn't work for Qwen3.5 (issue #35574)
- No thinking budget enforcement at the decoding level — purely template-driven
- No coordination between thinking and tool calling

### Atlas Approach
- `enable_thinking` passed to Jinja template context
- Thinking budget: 512 token hard cap (forced `</think>` injection via `think_end_token_id`)
- When model supports thinking, always enables thinking template even if client didn't request it (for tool call reliability)
- Comment: "model needs the <think> template to generate tool calls reliably (10/10 vs 2/10 without)"
- Reasoning content extracted from `<think>...</think>` and streamed as `reasoning_content`

### Gap Analysis

| Feature | vLLM | Atlas | Gap |
|---------|------|-------|-----|
| enable_thinking template control | Yes | Yes | No gap |
| Thinking budget enforcement | No (template only) | Yes (512 tok hard cap) | Atlas advantage |
| Force thinking for tool reliability | No | Yes (always enables for thinking models) | Atlas advantage |
| Thinking + tool call interaction | Independent | Coordinated (thinking warm-up) | Atlas advantage |
| Thinking suppression for Qwen3.5 | Broken (#35574) | Works (empty think block) | Atlas advantage |

### Actionable Items

**No major gaps** — Atlas handles thinking + tool calling better than vLLM.

**P3 — Consider per-request thinking budget:**
Atlas currently hard-caps at 512 tokens. Some requests (complex multi-tool planning) may benefit from more thinking. Consider allowing the client to override via `thinking.budget_tokens`.

---

## 5. Nemotron-H Support

### vLLM Approach
- Full Nemotron-H model support in `vllm/model_executor/models/nemotron_h.py`
- LatentMoE: `fc1_latent_proj` and `fc2_latent_proj` for expert routing projection
- MTP speculative decoding support via `nemotron_h_mtp.py`
- Tool calling: `--tool-call-parser qwen3_coder` (same as Qwen3-Coder)
- Reasoning: `--reasoning-parser nemotron_v3`
- Nemotron 3 Super 120B: supported with TP=4 on H100, FP8/NVFP4 quantization
- No reported buffer aliasing issues (H100 has larger memory)

### Atlas Approach
- Nemotron-H support via custom model implementation
- LatentMoE with `fc1_latent_proj` / `fc2_latent_proj`
- Custom Nemotron-H Jinja template with full tool calling support
- Same `qwen3_coder` parser format
- Running on GB10 (DGX Spark) — different memory constraints than H100

### Gap Analysis

| Feature | vLLM | Atlas | Gap |
|---------|------|-------|-----|
| LatentMoE | Supported | Supported | No gap |
| Tool calling format | qwen3_coder | qwen3_coder | No gap |
| Chat template | Model's native Jinja | Custom override (nemotron_h.jinja) | Atlas uses tuned template |
| MTP support | Yes | Yes | No gap |
| Buffer aliasing | No issues (H100) | Addressed (GB10-specific) | Different hardware |

### Actionable Items

**P2 — Validate Nemotron-H tool calling with structured template path:**
Once P0 (structured tool_calls in Jinja) is implemented, verify the nemotron_h.jinja template correctly renders multi-turn tool conversations. The template expects `message.tool_calls` with `function.arguments` as mappings (line 151: `for args_name, args_value in tool_call.arguments|items`).

---

## 6. Community Reports and Known Issues

### vLLM Issues (as of March 2026)
- **#35574**: Qwen3.5 cannot close thinking by `enable_thinking: false`
- **#35266**: Missing opening brace for Qwen3.5 streaming tool calls
- **#35347**: PR introducing Qwen35CoderToolParser for streaming fixes
- **#35615**: Qwen3Coder streaming parameter loss with speculative decode
- **#30439**: Qwen3 Coder parser does not stream tool call arguments
- **#29192**: Tool calling parsers fail to populate tool_calls array for Qwen2.5-Coder
- **#27766**: Feature request: read format constraints from tool parsers for guided decoding
- **#22132**: Qwen3 tool call format clobbered by guided decoding
- **GHSA-79j6-g2m3-jgfw**: RCE via eval() in Qwen3Coder parser (fixed in 0.10.1.1)

### Atlas Advantages Over vLLM
1. No `eval()` vulnerability — all parsing is safe string manipulation
2. Multi-format fallback parsing (4 levels) handles NVFP4 quantization artifacts
3. Grammar enforcement uses correct XML format for Qwen3-Coder (vLLM uses wrong JSON format)
4. Thinking budget enforcement prevents infinite thinking loops
5. NVFP4-specific tolerance for common quantization artifacts

### Atlas Disadvantages vs vLLM
1. Multi-turn tool_calls not passed as structured data (P0)
2. No schema-aware type coercion (P2)
3. Smaller parser ecosystem (2 vs 35 models)
4. minijinja edge cases may diverge from Python Jinja2

---

## 7. Priority-Ordered Action Items

### P0 — Critical (Correctness Impact)

**1. Pass structured tool_calls to Jinja template context**
- **File**: `crates/spark-server/src/api.rs` (lines 65-112, 244-246)
- **What**: Build JSON messages with `tool_calls` and `reasoning_content` fields, not just `role` + `content`
- **Why**: The Qwen3.5 and Nemotron-H Jinja templates have specific logic for rendering `message.tool_calls` that is never triggered because Atlas pre-formats them as text. This may cause subtle formatting differences in multi-turn conversations, especially around thinking block placement relative to tool calls.
- **How**:
  1. In the message loop, instead of appending `parser.format_tool_calls()` to content text, store tool_calls as a separate structured field
  2. Build `json_messages` as `{"role": ..., "content": ..., "tool_calls": [...], "reasoning_content": ...}`
  3. Parse `arguments` from JSON strings to serde_json objects before passing to template (match vLLM's `_postprocess_messages`)
  4. Remove the manual `format_tool_calls()` injection when the Jinja template handles tool_calls natively

### P1 — High (Quality Impact)

**2. Ensure grammar enforcement is wired for tool_choice=required**
- **File**: `crates/spark-server/src/scheduler.rs`, `crates/spark-server/src/api.rs`
- **What**: Verify that `compile_qwen3_coder_tool_grammar()` or `compile_hermes_tool_grammar()` is invoked when tool_choice=required, creating a `GrammarState` that constrains decoding from token 1
- **Why**: Atlas has the grammar infrastructure but it needs to be confirmed it's wired end-to-end

### P2 — Medium (Polish)

**3. Add schema-aware type coercion for Qwen3-Coder XML parameters**
- **File**: `crates/spark-server/src/tool_parser.rs` (in `parse_qwen3_coder_call`)
- **What**: Optionally convert parameter values based on tool JSON schema types
- **Why**: Some clients expect typed values (int not "5"), though most OpenAI clients handle string→type conversion themselves

**4. Validate Nemotron-H multi-turn tool conversations**
- After P0 is implemented, test that the nemotron_h.jinja template correctly renders conversations with tool_calls, tool responses, and thinking blocks

### P3 — Low (Future Enhancement)

**5. Per-request thinking budget override**
- Allow clients to set thinking budget via `thinking.budget_tokens` instead of fixed 512 cap

**6. Add partial_json_parser equivalent for Hermes format**
- Currently Atlas manually recovers truncated JSON; a proper incremental JSON parser would be more robust

**7. Track minijinja Jinja2 compatibility improvements**
- minijinja v2 covers most needs but monitor for new Python Jinja2 features used in model templates

---

## Architecture Comparison Summary

```
vLLM Architecture:
  Client → OpenAI API → parse_chat_messages() → [ConversationMessage with tool_calls as dicts]
                       → tokenizer.apply_chat_template() (Python Jinja2)
                       → tool_parser.extract_tool_calls() (model-specific parser)
                       → structured_output/xgrammar (JSON grammar, broken for XML models)

Atlas Architecture:
  Client → OpenAI API → manual message processing (tool_calls → text in content)
                       → Jinja template with minijinja (tools passed correctly, tool_calls not)
                       → tool_parser::parse_tool_calls() (universal fallback parsing)
                       → xgrammar structural tags (correct XML grammar for Qwen3-Coder)
```

Atlas's architecture is sound but has a gap in the message→template pipeline where tool_calls are pre-formatted instead of passed as structured data. Fixing this (P0) would align Atlas with vLLM's approach while maintaining Atlas's advantages in grammar enforcement and NVFP4 tolerance.
