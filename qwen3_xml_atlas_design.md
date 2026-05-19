# qwen3_xml Tool Parser for Atlas — Design Doc & State of Investigation

**Status:** Pre-implementation. All architectural questions answered; one workflow question open (where to install the type-coercion call site).

**Author trail:** Conversation with Claude, May 2026.

---

## 1. The bug we're solving

Qwen3.6-35B-A3B-FP8 served by Atlas with `--tool-call-parser qwen3_coder` and thinking mode enabled produces tool calls with **empty string values** for required parameters:

```
[tools] exec failed: Provide a command to start. raw_params={"command":""}
[tools] read failed: Missing required parameter: path raw_params={"path":""}
[tools] write failed: Missing required parameter: content raw_params={"content":"","path":"BOOTSTRAP.md"}
```

Field names extract correctly; values are empty. Same symptom is well-documented in the vLLM/Qwen ecosystem (HF Qwen3.6 #40, NVIDIA dev forum, vLLM #29192, #36769, #39056).

## 2. What `qwen3_xml` actually is (corrected from the original brief)

The original brief assumed `qwen3_xml` is a different *format* parser. **It is not.** Reading vLLM PR #25028 directly:

- Both `qwen3_coder` and `qwen3_xml` parse the **identical wire format**: `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`.
- `qwen3_xml` is a **better parser** for the same format. From the PR description (Zhikaiiii, Qwen team):
  1. "make sure the params of corresponding type are returned" — **schema-driven type coercion** (integer/boolean/array/object) instead of always emitting JSON strings.
  2. "handle function format error such as missing `}` for params" — **robustness to malformed/truncated XML**.
- The Qwen team's intent is for `qwen3_xml` to **replace** `qwen3_coder` ("we intent to use qwen3_coder_xml replace qwen3_coder ... it can completely replace qwen3_coder" — comment thread on PR #25028).
- vLLM's own docs now recommend `qwen3_xml` for Qwen3-Coder-class models.

## 3. Atlas architecture (what we mapped)

### 3.1 Repo layout

- Pure Rust monorepo, Cargo workspace.
- Tool-call parsing: `crates/spark-server/src/tool_parser/` (a directory of submodules, all wired through `tool_parser.rs` root).
- Two HTTP surfaces: `api/chat/` (OpenAI) and `anthropic/` (Anthropic). Per AGENTS.md, protocol-drift between them has burned days; any fix needs to land on both paths.

### 3.2 The parser trait (`tool_parser.rs`)

```rust
pub trait ToolCallParser: Send + Sync {
    fn name(&self) -> &str;
    fn system_prompt(&self, tools: &[ToolDefinition], tool_choice: &ToolChoice) -> String;
    fn format_tool_calls(&self, calls: &[IncomingToolCall]) -> String;
    fn format_tool_response(&self, content: &str) -> String { ... }
    fn leak_markers(&self) -> LeakMarkers { LeakMarkers::EMPTY }
    fn compile_tool_grammar(&self, ...) -> Option<Result<CompiledGrammar, GrammarError>> { None }
    fn has_tool_grammar(&self) -> bool { false }
    fn broken_opener_stop_strings(&self) -> &'static [&'static str] { &[] }
}
```

**Crucial observation:** the trait controls **what the model is told** (system prompt), **how server-injected tool calls are rendered back into the prompt** (`format_tool_calls`), **what counts as a content leak** (`leak_markers`), and **what grammar constrains decoding**. The trait does **NOT** own the extraction code. Extraction is shared.

### 3.3 The shared extractor (`parse_dispatch.rs`)

`parse_tool_calls(text: &str)` is **format-agnostic**:

1. Strips `</think>` (line 13–19) — so thinking-mode artifacts are gone before parsing.
2. Normalizes MiniMax outer tags (`<minimax:tool_call>` → `<tool_call>`) including the BPE-broken `<minimax:_call>` variant.
3. Scans for `<tool_call>...</tool_call>` envelopes; depth-aware close finder so literal `</tool_call>` strings inside parameter values don't terminate early.
4. For each envelope, calls `parse_one_call(inner, idx)` (in `parse_single_a.rs`), which auto-detects:
   - Gemma-4 native (`call:fn{...}`)
   - JSON / Hermes (`{"name":"...","arguments":{...}}`)
   - Truncated JSON recovery
   - MiniMax XML inner (`<invoke name="X"><parameter name="K">V</parameter>`)
   - Qwen3-Coder XML (`<function=NAME><parameter=KEY>V</parameter>`) — via `parse_qwen3_coder_call`
   - Tag-style fallback (`<function>NAME</function>`)
5. A long tail of fallbacks (Gemma-4 wrapper, Mistral native, bare-invoke, bare-function, JSON-in-codeblock, etc.).

### 3.4 The Qwen3-coder extractor (`parse_single_b.rs::parse_qwen3_coder_call`)

Lives 294 LoC. Already implements most of what `qwen3_xml` would need:

- Tolerates `<function=NAME>`, `<function NAME>`, `<|function=NAME>`, `<|function NAME>`.
- Stops parameter scan at `</function>` boundary (prevents bleed between consecutive function blocks — see opencode session bug fix).
- Recovery for missing `</parameter>` close: scans for least of `</parameter>`, next `<parameter=`, `</function>`.
- JSON-body fallback when no `<parameter>` tags found (grammar-constrained emission of `<function=Bash>{"command":"..."}</function>`).
- Strips a `_think` scratchpad field (TAFC mode).
- **All values are emitted as JSON strings** — this is line 117–118 explicit comment: "Always treat values as strings — the OpenAI tool calling spec requires arguments to match the tool's JSON schema, and the model emits values as raw text inside XML tags."

**This last point is THE gap vs. vLLM's `qwen3_xml`.** Atlas's parser produces `{"limit":"10"}` where the schema declares `limit: integer`; clients that validate types reject it.

### 3.5 The streaming detector (`streaming.rs` + `streaming_impl.rs`)

- Buffers the *entire* `<tool_call>...</tool_call>` body (no incremental arg streaming for XML — explicit comment line 117 of `streaming_impl.rs`).
- Emits `ToolCallStart` early when the function name is extractable (so clients see "tool fired" instantly).
- At close-tag time, calls `parse_one_call(inner)` and emits `ToolCallDelta + ToolCallEnd` (or full `ToolCall` if no early `Start` was emitted).
- `safe_emit_len()` holds back partial open-tag prefixes to prevent splitting across stream chunks.

### 3.6 Grammar layer (`grammar/compile_tools.rs`)

`compile_qwen3_coder_tool_grammar` uses xgrammar's `qwen_xml_parameter` content type — a native XML-parameter content-type that enforces the JSON schema during decoding. **The schema enforcement is already there at the grammar level.** When grammar is active and the model honors it, type-correct values are sampled. When grammar is *not* active (thinking mode bypasses it, or `--reasoning-parser qwen3` flow), the parser falls back to whatever the model emitted as raw text — and that's when we see strings instead of integers.

### 3.7 CLI plumbing (`cli.rs`)

```rust
#[arg(long, value_name = "FORMAT")]
pub tool_call_parser: Option<String>,
```

The raw string converts to `ToolCallFormat` via `FromStr`, which dispatches to a `Box<dyn ToolCallParser>` via `ToolCallFormat::into_parser()`.

### 3.8 MODEL.toml

Qwen3.6's `MODEL.toml` has a `[behavior]` table with `thinking_in_tools`, `max_thinking_budget`, `default_num_drafts`, `enable_loop_watchdog`. **No `default_tool_call_parser` field exists today.** Whether to add one is the open plumbing question.

## 4. House rules from AGENTS.md / CONTRIBUTING.md

- **SPDX header on line 1 of every `.rs`:** `// SPDX-License-Identifier: AGPL-3.0-only`. Enforced by CI.
- **500 LoC cap per `.rs` file under `crates/`.** vLLM's `qwen3xml_tool_parser.py` is ~1100 lines — we must split.
- AI-generated PRs are the explicit default. No flag in the PR description is needed beyond marking it AI-authored.
- Commits: `<area>: <imperative summary>` (e.g. `spark-server: add qwen3_xml tool parser`).
- One logical change per commit.
- CI gates (all GPU-free): `fmt`, `clippy -Dwarnings --tests --all-features`, `cargo test --workspace`, license headers, typos, cargo-deny, file-size cap.
- AGENTS.md flag: **"When you hit a regression, never assume the model is at fault — always look for the Atlas bug first."**

## 5. Test conventions

- Tests under `crates/spark-server/src/tool_parser/tests/`.
- Naming: `group_a.rs`, `group_b.rs`, `group_c.rs`, `group_d.rs` (no semantic grouping I can detect — just LoC partitioning to stay under the 500-LoC cap). New test file should be `group_e.rs`.
- Pattern: `use super::super::*;`, `#[test] fn descriptive_name()`, asserts via `parse_tool_calls(input)` or direct `parse_qwen3_coder_call(input, 0)`.
- Comments document the original session/issue that the test pins (e.g. "Reference failure: opencode-session.md 2026-04-25").

## 6. The contribution shape

### 6.1 What changes (high-level)

1. **New parser variant `qwen3_xml`** registered in `ToolCallFormat`. Same envelope, same grammar, same `format_tool_calls`, same leak markers as `qwen3_coder` — but declares intent to type-coerce its arguments.
2. **New post-processing pass** that walks the parsed `Vec<ToolCall>`, looks up each call's matching `ToolDefinition`, and rewrites argument JSON to honor the declared schema types (string→int, string→bool, string→array, string→object).
3. **New trait method** `wants_typed_arguments() -> bool` (default `false`; `Qwen3XmlParser` returns `true`). Cleaner than threading a `ToolCallFormat` parameter into the dispatcher.
4. **Default-change for Qwen3.6:** add `default_tool_call_parser = "qwen3_xml"` to `kernels/gb10/qwen3.6-35b-a3b/MODEL.toml`. **Requires verifying or adding the plumbing that reads this key.**
5. **Tests** as `group_e.rs`.

### 6.2 Files (with LoC estimates)

```
crates/spark-server/src/tool_parser/qwen3_xml.rs          NEW  ~150 LoC
  - struct Qwen3XmlParser;
  - impl ToolCallParser:
    - name() = "qwen3_xml"
    - compile_tool_grammar -> reuse compile_qwen3_coder_tool_grammar
    - has_tool_grammar -> true
    - leak_markers -> same as Qwen3CoderParser (re-export)
    - system_prompt -> use the Qwen-team-recommended text from PR #25028
    - format_tool_calls -> identical to Qwen3CoderParser

crates/spark-server/src/tool_parser/type_coerce.rs        NEW  ~250 LoC
  - pub fn coerce_call_args_to_schema(
        call: &mut ToolCall,
        tool_def: Option<&ToolDefinition>,
    )
  - pub fn coerce_all(calls: &mut [ToolCall], tools: &[ToolDefinition])
  - For each declared property:
    - integer/number -> try parse, leave as string on fail
    - boolean -> "true"/"false"/"True"/"False" -> bool, else leave string
    - array/object -> serde_json parse if it looks like JSON, else leave string
    - null -> "null" literal -> JSON null
    - string -> no-op
  - Robust: never panics, never strips a field that already has a non-string JSON type
  - Handles _think field correctly (already-stripped at extractor)
  - Pure function; fully unit-testable

crates/spark-server/src/tool_parser/tests/group_e.rs      NEW  ~280 LoC
  - Empty-arg regression: parse_qwen3_coder_call returns "" for missing
    value -> assert qwen3_xml-with-coercion preserves "" (the schema doesn't
    auto-fix nulls; the test pins the contract, not the outcome).
  - Type coercion: integer "10" -> 10 (number), boolean "true" -> true, etc.
  - Nested: array of objects with mixed types.
  - Round-trip: format_tool_calls -> parse_tool_calls preserves semantics.
  - Pin that qwen3_xml shares qwen3_coder's grammar (compile produces same
    output for same tools).
  - Smoke: system_prompt mentions <tool_call>, <function=, <parameter=.

crates/spark-server/src/tool_parser.rs                    MODIFIED
  - mod qwen3_xml; mod type_coerce;
  - pub use qwen3_xml::*; pub use type_coerce::*;
  - ToolCallFormat::Qwen3Xml variant
  - FromStr "qwen3_xml" => Ok(Self::Qwen3Xml)
  - into_parser arm
  - name() arm = "qwen3_xml"
  - Add trait method `fn wants_typed_arguments(&self) -> bool { false }`

crates/spark-server/src/tool_parser/tests/mod.rs          MODIFIED
  - mod group_e;

crates/spark-server/src/cli.rs                            MODIFIED (doc only)
  - Update the doc comment for tool_call_parser to list qwen3_xml.

crates/spark-server/src/api/chat/mod.rs                   MODIFIED ~5 LoC
crates/spark-server/src/api/chat_stream/tool_handlers.rs  MODIFIED ~5 LoC
crates/spark-server/src/anthropic/handlers.rs             MODIFIED ~5 LoC
  - After parse_tool_calls() (or the streaming detector emits ToolCall):
        if parser.wants_typed_arguments() {
            type_coerce::coerce_all(&mut calls, &request.tools);
        }
  - LANDS ON ALL THREE SURFACES (OpenAI chat, OpenAI chat-stream, Anthropic)
    — AGENTS.md is explicit about this drift class.

kernels/gb10/qwen3.6-35b-a3b/MODEL.toml                   MODIFIED
  - [behavior]
    + default_tool_call_parser = "qwen3_xml"
  - REQUIRES PLUMBING CHECK (see §7).
```

### 6.3 Why a post-processing pass and not a sibling extractor

Three alternatives were considered:

- **Option A:** Thread `ToolCallFormat` into `parse_tool_calls`. Rejected: invasive, every call site changes, dispatcher's auto-detection (Gemma/Mistral/MiniMax/Qwen3-coder/JSON fallback at the same layer) breaks under a per-format parameter.
- **Option B:** Thread the tool list into `parse_tool_calls`, do coercion inside the extractor, gate on a parser flag. Rejected: same invasiveness, plus the extractor doesn't currently know about tools at all (it's pure text-shape parsing).
- **Option C (chosen):** Post-processing pass at the handler boundary where both `parser` and `tools` are already in scope. Single-responsibility extractor stays; new pure function for coercion is independently testable; no signature changes to existing surface.

### 6.4 What does NOT change

- `parse_qwen3_coder_call` is untouched. All existing Qwen3-Coder users keep the exact behavior. Test `parse_qwen3_coder_multiple_params` (which asserts `args["limit"] == "10"`) stays green.
- The shared `parse_tool_calls` dispatcher is untouched.
- The streaming detector is untouched.
- The grammar layer is untouched (we reuse `compile_qwen3_coder_tool_grammar`).
- No new dependencies in `Cargo.toml`.

## 7. Open questions before writing code

### 7.1 Where exactly does type coercion install?

Need to see one or both of:
- `crates/spark-server/src/api/chat/mod.rs` — to locate the line right after `parse_tool_calls` returns
- `crates/spark-server/src/api/chat_stream/tool_handlers.rs` — same for streaming
- `crates/spark-server/src/anthropic/handlers.rs` — the Anthropic-API equivalent

Without these, the hook is sketched but not surgically placed.

### 7.2 Does MODEL.toml have `default_tool_call_parser` today?

If **yes**: the .toml change is a one-line edit.
If **no**: we need to either
- (a) defer the default-change to a follow-up PR (recommended for smallest first PR), or
- (b) add the plumbing in this PR (read TOML in `build.rs` -> emit into `ModelBehavior` -> `serve.rs` uses it as the default when `--tool-call-parser` is unset).

Path (a) keeps this PR focused; path (b) ships the full "Qwen3.6 just works" experience but doubles the surface.

A grep to settle it:
```bash
grep -rn "tool_call_parser\|default_tool_parser\|tool_call_format" crates/atlas-kernels/ crates/atlas-core/ crates/spark-server/src/main_modules/ 2>/dev/null
```

### 7.3 Should we also fix the underlying empty-arg root cause?

Strict reading of the symptom in the bug report: "the model knows the correct field names but fails to populate the values". With grammar (`compile_qwen3_coder_tool_grammar` using `qwen_xml_parameter`) active and the **upstream xgrammar fix** for required-field enforcement landed (see `qwen3_coder_required.rs` test comment: "These tests fail against xgrammar v0.1.32 (the current pin) and pass once the fork-with-fix is wired up via xgrammar-pins.toml"), the model should not produce empty required values in the first place.

**Hypothesis:** the empty-arg symptom is downstream of *grammar not being active in thinking mode*. Qwen3.6 with `enable_thinking=true` may emit the tool call inside `</think>` and the grammar machinery may not engage there. The dispatcher strips `</think>` (good), but by then the empty value is already in the text.

**Recommendation:** do NOT chase this in the qwen3_xml PR. The xgrammar pin bump is a separate, more invasive change (and already has its own plan per the test comment). The qwen3_xml PR adds *defense in depth* — better parser behavior on the post-strip text — and unlocks proper type semantics. Both changes are complementary; neither blocks the other.

### 7.4 Should `qwen3_xml` also be more permissive on malformed values?

vLLM PR #25028 motivation #2: "handle function format error such as missing `}` for params". Atlas's `parse_qwen3_coder_call` already has a missing-close-tag recovery branch (lines 88–104 of `parse_single_b.rs`). The question is whether we want `qwen3_xml`'s extractor to be even more aggressive, e.g.:
- accept `<parameter=key>value` without `</parameter>` at end-of-stream
- accept partial JSON in `<parameter=key>{"partial":` for in-flight streaming

**Recommendation:** out of scope for this PR. Atlas's current recovery is already comparable to vLLM's, and adding new tolerance risks regressing legitimate cases. Type coercion alone is the right first delivery.

## 8. Open workflow item — the actual ask

The conversation paused on **question 7.1 (where the coercion hook installs)**. Three viable answers:

1. **Upload `api/chat/mod.rs` + `api/chat_stream/tool_handlers.rs` + `anthropic/handlers.rs`** → Claude writes the hook with exact line placement.
2. **Skip handler edit, redesign coercion as a method on `ToolCallFormat`** that callers invoke explicitly. Less invasive but exposes a "did you remember to call this?" footgun.
3. **Write parser + coerce + tests now, leave the handler hook as a stub `// TODO: call coerce_all here` for the human to place.** Smallest forward step that doesn't block on more file uploads.

The choice between (1) and (3) is a velocity/precision trade-off; (2) is structurally worse and probably shouldn't be picked.

## 9. Reference materials staged in sandbox

All under `/home/claude/atlas-refs/`:

- `tool_parser.rs` — module root, `ToolCallParser` trait, `ToolCallFormat` enum
- `qwen3_coder.rs` — closest analog parser (system prompt, leak markers, grammar)
- `minimax_xml.rs` — second-closest analog (envelope-aware leak markers, XML inner)
- `parse_dispatch.rs` — the shared `parse_tool_calls` dispatcher
- `parse_single_a.rs` — `parse_one_call`, MiniMax XML extractor, JSON helpers
- `parse_single_b.rs` — `parse_qwen3_coder_call`, tag-style fallback, `bare_function_end`
- `pipeline.rs` + `pipeline_helpers.rs` — bare-function pass infrastructure (out of scope for us)
- `streaming.rs` + `streaming_impl.rs` — the `StreamingToolDetector`
- `cli.rs` — `--tool-call-parser` flag plumbing
- `compile_tools.rs` — grammar compilation, `compile_qwen3_coder_tool_grammar`
- `group_a.rs`–`group_d.rs` — existing tests; pattern to follow for `group_e.rs`
- `qwen3_coder_required.rs` — grammar regression test that pins the required-param contract
- `model_tomls/MODEL.toml` — Qwen3.5 MODEL.toml (Qwen3.6 inline above)
- `model_tomls/qwen3_6_MODEL.toml` — Qwen3.6 MODEL.toml (reconstructed from inline upload)
- `qwen3_xml_tool_parser.md` — the original brief

## 10. References (external)

- vLLM PR #25028 — the canonical `qwen3_xml` introduction by Qwen team (Zhikaiiii)
- vLLM #29192 — Qwen2.5-Coder parser failure (same family of bugs)
- vLLM #36769 — `qwen3coder_tool_parser.py` substring-not-found crash on edge case
- vLLM #39056 — XML tool calls inside `<think>` lost on Qwen3.5-35B-A3B-FP8 with `qwen3_coder` + `qwen3` reasoning parser (exactly our bug's class)
- vLLM tool calling docs — recommend `qwen3_xml` for Qwen3-Coder models
- HF `Qwen/Qwen3.6-35B-A3B` discussion #40 — community confirmation
- NVIDIA Dev Forum: Qwen3.5 Tool Calling finally fixed (chat template + `qwen3_xml`)
- `vllm-qwen35-tool-parser` (Jetson Thor pkg) — confirms type-aware param conversion via `_convert_param_value` as the key delta

## 11. Decision log

- ✅ Format identical to qwen3_coder; type-coercion is the substantive delta.
- ✅ Reuse `compile_qwen3_coder_tool_grammar` rather than fork.
- ✅ Reuse `parse_qwen3_coder_call` extractor; add coercion as post-pass.
- ✅ New trait method `wants_typed_arguments()` (default `false`) to gate coercion.
- ✅ New test file `group_e.rs`; new submodule files `qwen3_xml.rs`, `type_coerce.rs`.
- ✅ Hook lands on all three handler surfaces (OpenAI blocking, OpenAI streaming, Anthropic).
- ✅ MODEL.toml default-change is in-scope per user direction; plumbing existence to be verified.
- ❓ Whether to ship MODEL.toml plumbing in this PR (§7.2) — pending grep.
- ❓ Where exactly to install the coercion hook (§7.1) — pending file upload OR conscious decision to stub.
- ✅ Out of scope: streaming-incremental arg deltas, malformed-XML tolerance beyond what `parse_qwen3_coder_call` already does, xgrammar pin bump.

---

**Next step when resuming:** answer §7.1 and §7.2, then start writing `qwen3_xml.rs` + `type_coerce.rs` (the two new files), then `group_e.rs`, then the registration touches, then the handler hooks, then the .toml change.
