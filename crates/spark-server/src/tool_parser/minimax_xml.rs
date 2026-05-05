// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::*;

/// MiniMax XML format.
///
/// Outer tag: `<minimax:tool_call>...</minimax:tool_call>` (rewritten to
/// plain `<tool_call>` during the scanning pre-pass so the existing
/// outer-tag loop handles it).
///
/// Inner format:
/// ```xml
/// <invoke name="tool_name">
/// <parameter name="key1">value1</parameter>
/// <parameter name="key2">value2</parameter>
/// </invoke>
/// ```
///
/// Notes vs qwen3_coder:
/// - `<invoke name="X">` (standard XML attribute) instead of
///   `<function=X>` (Qwen's custom `=` syntax).
/// - `<parameter name="K">` instead of `<parameter=K>`.
/// - Closing tag is `</invoke>`, not `</function>`.
///
/// Values are emitted as raw text — same as qwen3_coder, we keep them
/// as JSON strings on the way out rather than trying to infer types.
pub struct MinimaxXmlParser;

impl ToolCallParser for MinimaxXmlParser {
    fn name(&self) -> &str {
        "minimax_xml"
    }

    fn compile_tool_grammar(
        &self,
        engine: &mut GrammarEngine,
        tools: &[ToolDefinition],
        use_triggers: bool,
    ) -> Option<Result<CompiledGrammar, GrammarError>> {
        Some(engine.compile_minimax_xml_tool_grammar(tools, use_triggers))
    }

    fn has_tool_grammar(&self) -> bool {
        true
    }

    fn system_prompt(&self, tools: &[ToolDefinition], tool_choice: &ToolChoice) -> String {
        // Match MiniMax's native chat_template.jinja output exactly
        // (see `docs/tool_calling_guide.md` in MiniMaxAI/MiniMax-M2
        // on HF). The template emits `<tools>` … `</tools>` with a
        // JSON tool per `<tool>` child, then describes the XML
        // invocation format.
        let mut prompt =
            String::from("# Tools\n\nYou have access to the following functions:\n\n<tools>\n");
        for tool in tools {
            let json = serde_json::to_string(tool).unwrap_or_default();
            prompt.push_str(&format!("<tool>{json}</tool>\n"));
        }
        prompt.push_str(
            "</tools>\n\n\
             When making tool calls, use XML format to invoke tools and pass parameters:\n\
             \n<minimax:tool_call>\n\
             <invoke name=\"tool-name-1\">\n\
             <parameter name=\"param-key-1\">param-value-1</parameter>\n\
             <parameter name=\"param-key-2\">param-value-2</parameter>\n\
             ...\n\
             </invoke>\n\
             </minimax:tool_call>\n",
        );
        append_tool_choice_instruction(&mut prompt, tool_choice);
        prompt
    }

    fn format_tool_calls(&self, calls: &[IncomingToolCall]) -> String {
        let mut out = String::new();
        for tc in calls {
            let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                .unwrap_or(serde_json::Value::Object(Default::default()));
            out.push_str("\n<minimax:tool_call>\n");
            out.push_str(&format!("<invoke name=\"{}\">\n", tc.function.name));
            if let Some(obj) = args.as_object() {
                for (key, val) in obj {
                    let val_str = match val {
                        serde_json::Value::String(s) => s.clone(),
                        other => serde_json::to_string(other).unwrap_or_default(),
                    };
                    out.push_str(&format!(
                        "<parameter name=\"{key}\">{val_str}</parameter>\n"
                    ));
                }
            }
            out.push_str("</invoke>\n</minimax:tool_call>");
        }
        out
    }

    fn leak_markers(&self) -> LeakMarkers {
        // MiniMax uses standard-XML attribute syntax inside its
        // namespaced outer tag. Two layers of detection:
        //
        //   - `envelope_open`/`envelope_close` recognise the outer
        //     `<minimax:tool_call>...</minimax:tool_call>` envelope
        //     in any of its three observed forms: the canonical
        //     special-token (id 200052), the BPE-broken form
        //     `<minimax:_call>` (token id 91125 = `:_` straddles the
        //     trigger boundary), and the legacy `<tool_call>` shape
        //     that `parse_tool_calls` already normalises to. While
        //     inside an envelope the sanitizer treats `<invoke>` /
        //     `<parameter>` as expected content, not orphan leaks.
        //
        //   - `orphan_open`/`close` keep the original protection for
        //     stray inner tags that appear without any envelope
        //     wrapping at all (true model hallucination — drop them).
        //
        // F73 (2026-04-29): the envelope_* layer is the fix for
        // opencode's broken-envelope failure mode. xgrammar's
        // TagDispatch is non-anchored across BPE merge boundaries
        // and lets the model emit `<minimax:_call>...<invoke ...>
        // </invoke>...</minimax:_call>`; before F73 the sanitizer
        // treated the inner `<invoke>` block as orphan and dropped
        // it, leaving `parse_tool_calls` with nothing to extract.
        const MARKERS: LeakMarkers = LeakMarkers {
            orphan_open: &["<parameter name=\"", "<invoke name=\""],
            close: &[
                "</parameter>",
                "</invoke>",
                "</minimax:tool_call>",
                "</tool_call>",
            ],
            envelope_open: &["<minimax:tool_call>", "<minimax:_call>", "<tool_call>"],
            envelope_close: &["</minimax:tool_call>", "</minimax:_call>", "</tool_call>"],
        };
        MARKERS
    }
}

// ── Bare JSON parser (Nemotron-Super-120B) ──
