# `qwen3_xml` Tool Parser — Refined Implementation Strategy

## Context

Qwen3.6-35B-A3B-FP8 produces empty string values for required tool-call parameters when thinking mode is enabled (`--tool-call-parser qwen3_coder`). The `qwen3_xml` parser is the Qwen-team's fix: same `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>` wire format, same grammar, but with schema-driven type coercion (string→int/bool/array/object) applied as a post-processing pass.

Reference design: `qwen3_xml_atlas_design.md`.

---

## Open questions resolved

**§7.1 — where the coercion hook installs:**
- **Non-streaming:** `chat_blocking.rs:275` `build_choice_message` already has `_state: &AppState`; drop the `_`, call `coerce_all` after `backfill_required_params` (line 307) and before `validate_tool_calls` (line 311).
- **Streaming:** Add `wants_typed_arguments: bool` to `StreamCtx` (ctx.rs), initialize at construction in `chat_stream/mod.rs:178`. Hook goes in both `handle_complete_tool_call` (after backfill, line 41) and `handle_tool_call_delta` (after backfill, line 192).
- **Anthropic path:** Confirmed Anthropic handlers do not call `parse_tool_calls` directly — they route through the same OpenAI blocking/streaming infrastructure, so the two hooks above cover it automatically.

**§7.2 — MODEL.toml plumbing:**
- Fully exists. `atlas-kernels/build_parse.rs:191-195` → `ModelBehavior.tool_call_parser` → Tier-2 in `serve_phases/runtime.rs:220-230`. Nemotron-Super already uses it for `bare_json`. Just add the field to Qwen3.6's MODEL.toml.

---

## Files to CREATE

### `crates/spark-server/src/tool_parser/qwen3_xml.rs` (~80 LoC)

```rust
// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

pub struct Qwen3XmlParser;

impl ToolCallParser for Qwen3XmlParser {
    fn name(&self) -> &str { "qwen3_xml" }
    fn wants_typed_arguments(&self) -> bool { true }

    // Identical wire format — delegate everything else to Qwen3CoderParser.
    fn system_prompt(&self, tools: &[ToolDefinition], tc: &ToolChoice) -> String {
        Qwen3CoderParser.system_prompt(tools, tc)
    }
    fn format_tool_calls(&self, calls: &[IncomingToolCall]) -> String {
        Qwen3CoderParser.format_tool_calls(calls)
    }
    fn format_tool_response(&self, content: &str) -> String {
        Qwen3CoderParser.format_tool_response(content)
    }
    fn leak_markers(&self) -> LeakMarkers { Qwen3CoderParser.leak_markers() }
    fn compile_tool_grammar(
        &self, engine: &mut GrammarEngine, tools: &[ToolDefinition], use_triggers: bool,
    ) -> Option<Result<CompiledGrammar, GrammarError>> {
        Qwen3CoderParser.compile_tool_grammar(engine, tools, use_triggers)
    }
    fn has_tool_grammar(&self) -> bool { true }
}
```

### `crates/spark-server/src/tool_parser/type_coerce.rs` (~200 LoC)

```rust
// SPDX-License-Identifier: AGPL-3.0-only
use super::{ToolCall, ToolDefinition};

/// Run schema-driven type coercion on all calls in `calls`.
/// Finds each call's matching ToolDefinition by name; no-ops if not found.
pub fn coerce_all(calls: &mut [ToolCall], tools: &[ToolDefinition]) {
    for call in calls.iter_mut() {
        let def = tools.iter().find(|t| t.function.name == call.function.name);
        coerce_call_args(call, def);
    }
}

fn coerce_call_args(call: &mut ToolCall, tool_def: Option<&ToolDefinition>) {
    let Some(schema) = tool_def.and_then(|t| t.function.parameters.as_ref()) else { return };
    let Some(props) = schema.get("properties").and_then(|p| p.as_object()) else { return };

    let Ok(mut args) = serde_json::from_str::<serde_json::Value>(&call.function.arguments) else { return };
    let Some(obj) = args.as_object_mut() else { return };

    let mut changed = false;
    for (key, prop) in props {
        let Some(ty) = prop.get("type").and_then(|t| t.as_str()) else { continue };
        let Some(val) = obj.get_mut(key) else { continue };
        // No-op if already the right JSON type.
        match ty {
            "integer" | "number" => {
                if let serde_json::Value::String(s) = val {
                    if let Ok(n) = s.parse::<f64>() {
                        if let Some(num) = serde_json::Number::from_f64(n) {
                            *val = serde_json::Value::Number(num);
                            changed = true;
                        }
                    }
                }
            }
            "boolean" => {
                if let serde_json::Value::String(s) = val {
                    match s.as_str() {
                        "true" | "True" => { *val = serde_json::Value::Bool(true); changed = true; }
                        "false" | "False" => { *val = serde_json::Value::Bool(false); changed = true; }
                        _ => {}
                    }
                }
            }
            "array" | "object" => {
                if let serde_json::Value::String(s) = val {
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(s) {
                        *val = parsed;
                        changed = true;
                    }
                }
            }
            "null" => {
                if val == &serde_json::Value::String("null".to_string()) {
                    *val = serde_json::Value::Null;
                    changed = true;
                }
            }
            _ => {} // "string" and unknowns: no-op
        }
    }
    if changed {
        if let Ok(s) = serde_json::to_string(&args) {
            call.function.arguments = s;
        }
    }
}
```

### `crates/spark-server/src/tool_parser/tests/group_e.rs` (~280 LoC)

SPDX header, `use super::super::*;`, then:

| Test | What it pins |
|------|-------------|
| `coerce_integer_string` | `"10"` → `10` (number) |
| `coerce_number_float` | `"3.14"` → `3.14` (number) |
| `coerce_boolean_true` | `"true"` → `true` |
| `coerce_boolean_True` | `"True"` → `true` |
| `coerce_boolean_false` | `"false"` → `false` |
| `coerce_array_string` | `"[1,2,3]"` → array |
| `coerce_object_string` | `"{\"a\":1}"` → object |
| `coerce_null_string` | `"null"` → JSON null |
| `no_coerce_already_number` | `42` (already Number) → untouched |
| `no_coerce_bad_parse` | `"notanumber"` for integer type → left as string |
| `empty_arg_preserved` | `""` for integer schema → left as `""` (can't parse, no-op) |
| `coerce_all_multi_call` | two calls, two different schemas coerced independently |
| `coerce_ignores_missing_tool` | call name not in tool list → no panic, args unchanged |
| `qwen3_xml_name` | `Qwen3XmlParser.name() == "qwen3_xml"` |
| `qwen3_xml_wants_typed` | `Qwen3XmlParser.wants_typed_arguments() == true` |
| `qwen3_coder_not_typed` | `Qwen3CoderParser.wants_typed_arguments() == false` |
| `qwen3_xml_has_grammar` | `Qwen3XmlParser.has_tool_grammar() == true` |
| `qwen3_xml_system_prompt_markers` | prompt contains `<tool_call>`, `<function=`, `<parameter=` |

---

## Files to MODIFY

### `crates/spark-server/src/tool_parser.rs`

1. **Trait** — add after `broken_opener_stop_strings`:
   ```rust
   fn wants_typed_arguments(&self) -> bool { false }
   ```

2. **`ToolCallFormat` enum** — add:
   ```rust
   Qwen3Xml,
   ```

3. **`FromStr`** — add arm and update error string:
   ```rust
   "qwen3_xml" => Ok(Self::Qwen3Xml),
   // error: "Supported: hermes, qwen3_coder, qwen3_xml, gemma4, mistral, minimax_xml, bare_json"
   ```

4. **`into_parser()`** — add arm:
   ```rust
   Self::Qwen3Xml => Box::new(Qwen3XmlParser),
   ```

5. **`name()`** — add arm:
   ```rust
   Self::Qwen3Xml => "qwen3_xml",
   ```

6. **Sub-modules** — after `mod qwen3_coder;`:
   ```rust
   mod qwen3_xml;
   mod type_coerce;
   ```

7. **Re-exports** — after `pub use qwen3_coder::*;`:
   ```rust
   pub use qwen3_xml::*;
   pub use type_coerce::coerce_all;
   ```

---

### `crates/spark-server/src/tool_parser/tests/mod.rs`

Add `mod group_e;`.

---

### `crates/spark-server/src/api/chat_blocking.rs`

`build_choice_message` (line 275): drop `_` from `_state: &AppState`.

After `backfill_required_params` (line 307), before `normalize_paths`:
```rust
if state.tool_call_parser.as_ref().map_or(false, |p| p.wants_typed_arguments()) {
    tool_parser::coerce_all(&mut tool_calls_i, &tools_ref);
}
```

---

### `crates/spark-server/src/api/chat_stream/ctx.rs`

Add field:
```rust
pub(super) wants_typed_arguments: bool,
```

---

### `crates/spark-server/src/api/chat_stream/mod.rs`

In `StreamCtx { ... }` literal (line 178), add:
```rust
wants_typed_arguments: state
    .tool_call_parser
    .as_ref()
    .map_or(false, |p| p.wants_typed_arguments()),
```

---

### `crates/spark-server/src/api/chat_stream/tool_handlers.rs`

**`handle_complete_tool_call`** — after backfill (line 41), before validate (line 45):
```rust
if ctx.wants_typed_arguments {
    tool_parser::coerce_all(std::slice::from_mut(tc), &ctx.tool_defs_for_backfill);
}
```

**`handle_tool_call_delta`** — after backfill (line 192–195), before validate (line 199):
```rust
if ctx.wants_typed_arguments {
    tool_parser::coerce_all(std::slice::from_mut(&mut tc), &ctx.tool_defs_for_backfill);
}
```

---

### `crates/spark-server/src/cli.rs`

Update `--tool-call-parser` doc comment to include `qwen3_xml` in the supported list.

---

### `kernels/gb10/qwen3.6-35b-a3b/MODEL.toml`

In `[behavior]` section, add:
```toml
tool_call_parser = "qwen3_xml"
```

---

## What does NOT change

- `parse_qwen3_coder_call` — untouched; existing tests stay green.
- `parse_dispatch.rs` — untouched.
- `streaming.rs` / `streaming_impl.rs` — untouched.
- `grammar/compile_tools.rs` — untouched; grammar reused via delegation.
- No new Cargo.toml dependencies.
- No xgrammar pin bump.

---

## CI checklist

```bash
cargo fmt --check
ATLAS_SKIP_BUILD=1 cargo clippy -Dwarnings --tests --all-features
cargo test --workspace -- tool_parser::tests
bash scripts/check-license-headers.sh
# LoC: qwen3_xml.rs ~80 | type_coerce.rs ~200 | group_e.rs ~280 — all under 500 cap
```

Commit message: `spark-server: add qwen3_xml tool parser with schema-driven type coercion`
