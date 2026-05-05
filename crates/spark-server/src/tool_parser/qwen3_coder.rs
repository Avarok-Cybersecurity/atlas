// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::*;

/// Qwen3-Coder format: XML `<function=name><parameter=key>value</parameter></function>`
/// inside `<tool_call>` tags.
///
/// Used by Qwen3.5-27B, Qwen3.5-35B, Qwen3.5-122B, and Nemotron-H-30B.
pub struct Qwen3CoderParser;

impl ToolCallParser for Qwen3CoderParser {
    fn name(&self) -> &str {
        "qwen3_coder"
    }

    fn compile_tool_grammar(
        &self,
        engine: &mut GrammarEngine,
        tools: &[ToolDefinition],
        use_triggers: bool,
    ) -> Option<Result<CompiledGrammar, GrammarError>> {
        Some(engine.compile_qwen3_coder_tool_grammar(tools, use_triggers))
    }

    fn has_tool_grammar(&self) -> bool {
        true
    }

    fn system_prompt(&self, tools: &[ToolDefinition], tool_choice: &ToolChoice) -> String {
        // Match the model's native Jinja chat_template exactly:
        // Tool definitions as JSON (not XML), response format as XML.
        let mut prompt =
            String::from("# Tools\n\nYou have access to the following functions:\n\n<tools>\n");
        // F33 (2026-04-26): inject a one-line retry rule into the
        // Bash tool's `description`. Tool-description text is part
        // of the tool schema render and is attended on every Bash
        // call. Per Anthropic's leaked Claude Code prompt + plan
        // F33 design.
        for tool in tools {
            let json = if matches!(tool.function.name.as_str(), "Bash" | "bash") {
                let mut t = tool.clone();
                let suffix = " | After ONE failure with \"command not found\" or exit code 127, do NOT retry the same command — the binary is permanently unavailable in this environment. Choose a different approach or tell the user the dependency is missing.";
                t.function.description = Some(match t.function.description {
                    Some(d) if !d.contains("[atlas-f33]") => format!("{d}\n[atlas-f33]{suffix}"),
                    Some(d) => d,
                    None => format!("[atlas-f33]{suffix}"),
                });
                serde_json::to_string(&t).unwrap_or_default()
            } else {
                serde_json::to_string(tool).unwrap_or_default()
            };
            prompt.push_str(&json);
            prompt.push('\n');
        }
        // F34 (2026-04-26): negative-pattern walls have been trimmed
        // (per Wei et al. on contrastive in-context learning:
        // showing "WRONG: emit code in markdown then call Write" is
        // an imitation trap for pattern-completion models). One
        // positive-shot example + a single declarative-logic
        // sentence replace the prior 50+ lines of WRONG examples.
        prompt.push_str("\
</tools>\n\n\
If you choose to call a function ONLY reply in the following format with NO suffix:\n\n\
<tool_call>\n\
<function=example_function_name>\n\
<parameter=example_parameter_1>\n\
value_1\n\
</parameter>\n\
<parameter=example_parameter_2>\n\
This is the value for the second parameter\n\
that can span\n\
multiple lines\n\
</parameter>\n\
</function>\n\
</tool_call>\n\n\
<IMMEDIATE_TOOL_USE>\n\
Tools are IMMEDIATELY EXECUTABLE. When you decide to use a tool, emit the <tool_call> directly. The tool_call's parameter values are the ONLY place file content, commands, or other tool inputs should appear — do NOT pre-render that content as prose or in markdown fences before the call.\n\
For 'bash'/'Bash' tools specifically: do NOT stage the command in a ```bash``` fence and do NOT prefix with phrases like \"Let me run this command:\" or \"Let me execute:\". The next emission after deciding to run a shell command must be the <tool_call> — the command goes inside <parameter=command>, not in markdown.\n\
For 'Write'/'Edit' tools specifically: do NOT pre-render the file content as a markdown ```toml/```rust/```python fence before the tool call. Do NOT write phrases like \"Let me create the Cargo.toml:\" or \"Now the source file:\" followed by a code fence containing the file body — the file body goes inside <parameter=content>, not in markdown. Pre-rendering the same content twice (once in markdown, once in the parameter) wastes tokens, can cause the tool call to be dropped if the markdown is long enough, and is the documented \"narrate-then-tool\" loop pattern. The next emission after deciding to write a file must be the <tool_call>.\n\
Example:\n\
    <tool_call>\n\
    <function=Write>\n\
    <parameter=file_path>/path/to/Cargo.toml</parameter>\n\
    <parameter=content>[package]\nname = \"x\"</parameter>\n\
    </function>\n\
    </tool_call>\n\
</IMMEDIATE_TOOL_USE>\n\n\
<IMPORTANT>\n\
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags.\n\
- EVERY required parameter MUST have a non-empty value.\n\
- For 'write'/'Write': ALWAYS provide both file_path AND content. PREFER 'write' over 'edit' — write the complete updated file content instead of find-and-replace. Only use 'edit' for very small single-line changes.\n\
- For 'edit'/'Edit': file_path MUST be non-empty. oldString MUST be copied verbatim from the file.\n\
- To make MULTIPLE tool calls, use SEPARATE <tool_call> blocks — NEVER embed <tool_call> tags inside bash commands or heredocs.\n\
- NEVER simulate a tool response in your content. Do NOT emit <tool_response>, <file>, \"1: ...\\n2: ...\" line-numbered previews, or any other text pretending to show what a tool returned. The real response is provided by the system AFTER you emit the <tool_call>.\n\
- Tool names like 'write', 'read', 'edit', 'bash' are ONLY valid inside the `<function=NAME>` line of a `<tool_call>` block. NEVER emit bare tags like `<write>`, `<filePath>`, or `<command>` at the top level of your content.\n\
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after.\n\
- If there is no function call available, answer the question with your current knowledge and do not tell the user about function calls.\n\
</IMPORTANT>",
        );
        append_tool_choice_instruction(&mut prompt, tool_choice);
        prompt
    }

    fn format_tool_calls(&self, calls: &[IncomingToolCall]) -> String {
        let mut out = String::new();
        for tc in calls {
            let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                .unwrap_or(serde_json::Value::Object(Default::default()));
            out.push_str("\n<tool_call>\n");
            out.push_str(&format!("<function={}>\n", tc.function.name));
            if let Some(obj) = args.as_object() {
                for (key, val) in obj {
                    let val_str = match val {
                        serde_json::Value::String(s) => s.clone(),
                        other => serde_json::to_string(other).unwrap_or_default(),
                    };
                    out.push_str(&format!("<parameter={key}>\n{val_str}\n</parameter>\n"));
                }
            }
            out.push_str("</function>\n</tool_call>");
        }
        out
    }

    fn leak_markers(&self) -> LeakMarkers {
        // Qwen3-coder XML tags. When the model emits a malformed bare
        // `<parameter=KEY>...` fragment outside the `<tool_call>` wrapper
        // (the streaming detector can't recognise it as a tool call), the
        // sanitizer enters suppression until any of the three close tags
        // closes the leaked block. Outer `<tool_call>` / `<function=` are
        // NOT listed here: the streaming detector reassembles the real
        // tool call from those tags across BPE boundaries and depends on
        // the raw fragments reaching it.
        //
        // `<tool_response>` is a SERVER-internal wrapper that Atlas renders
        // around role=tool messages when they enter the prompt (see the
        // qwen3_5_moe.jinja chat template). The model should NEVER emit
        // it in content — when it does, it's hallucinating a simulated
        // tool exchange (observed in opencode sessions where the model
        // tries to "preview" a read/write before actually calling the
        // tool). Treat it as a leak open and suppress until close.
        const MARKERS: LeakMarkers = LeakMarkers {
            // `<function_results>` (Anthropic-style tool-result wrapper)
            // and `<result>` (its inner element) are NEVER emitted by
            // the qwen3_coder protocol — when they appear in content,
            // the model is hallucinating a tool exchange (observed in
            // ses_23b4781f7ffebc7UgkKWedTmjd 2026-04-25 turn 3 where
            // the model wrote a fake `<function_results><result>1:
            // …pub fn add(a: f64, f64) -> f64…` block showing a
            // bug that didn't exist in the actual file). Treat as
            // leak openers and suppress until close.
            //
            // F8 (2026-04-26): added `<function=`, `<tool_call>`,
            // and `<tool_use>` as orphan_open. Live evidence from
            // dump-fix28 axum-test-18 turn-by-turn: model emitted
            // bare `<function=Bash>` and `<tool_call><function=Write>`
            // in content during multi-turn loops — these are
            // partial tool-call structures the model started but
            // never finished, leaking into the streamed text. The
            // streaming detector classifies them as Content (since
            // they don't form a proper envelope), which then routes
            // through this sanitizer. Adding them here suppresses
            // until any close tag arrives. The detector still
            // handles legitimate `<tool_call>...</tool_call>` blocks
            // BEFORE this sanitizer sees them — only orphans land
            // in Content.
            orphan_open: &[
                "<parameter=",
                "<tool_response>",
                "<function_results>",
                "<result>",
                "<function=",
                "<tool_call>",
                "<tool_use>",
                // F18 (2026-04-26): extended marker list found by
                // post-fix29 sanitiser-gap audit. fix29 cc-session
                // line 29 emitted bare `<>` (empty tag); model
                // also emitted `<param=` typo of `<parameter=`.
                //
                // F20 (2026-04-26): the original `<func` marker
                // (without trailing punctuation) was REVERTED in
                // fix31 because it matches the prefix of legitimate
                // `<function=Write>` tool calls when they stream
                // across BPE-token chunk boundaries. The Content
                // sanitiser then enters suppression and silently
                // drops the entire tool-call argument body —
                // observed as "Error writing file" in fix30
                // cc-session-20. The detector's `safe_emit_len`
                // protects detector-routed chunks but not Content
                // sanitiser path. The truncated-orphan case `<func`
                // without proper completion is so rare it isn't
                // worth the legitimate-call regression risk.
                "<>",
                "<param=",
            ],
            close: &[
                "</parameter>",
                "</function>",
                "</tool_call>",
                "</tool_response>",
                "</function_results>",
                "</result>",
                "</tool_use>",
            ],
            envelope_open: &[],
            envelope_close: &[],
        };
        MARKERS
    }
}

// ── Gemma-4 parser (native format) ──
