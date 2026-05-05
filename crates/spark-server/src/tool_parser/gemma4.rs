// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::*;

/// Gemma-4 format: `<|tool_call>call:fn_name{key:val,...}<tool_call|>`
///
/// Tool definitions are injected by the Jinja template using native `<|tool>` blocks.
/// The system_prompt here is empty since the template handles everything.
pub struct Gemma4Parser;

impl ToolCallParser for Gemma4Parser {
    fn name(&self) -> &str {
        "gemma4"
    }

    fn compile_tool_grammar(
        &self,
        engine: &mut GrammarEngine,
        tools: &[ToolDefinition],
        use_triggers: bool,
    ) -> Option<Result<CompiledGrammar, GrammarError>> {
        Some(engine.compile_gemma4_tool_grammar(tools, use_triggers))
    }

    fn has_tool_grammar(&self) -> bool {
        true
    }

    fn system_prompt(&self, _tools: &[ToolDefinition], _tool_choice: &ToolChoice) -> String {
        // Tool definitions are handled by the gemma4.jinja template natively.
        String::new()
    }

    fn format_tool_calls(&self, calls: &[IncomingToolCall]) -> String {
        let mut out = String::new();
        for tc in calls {
            out.push_str("<|tool_call>call:");
            out.push_str(&tc.function.name);
            out.push('{');
            // Convert JSON arguments to native format
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
                && let Some(obj) = v.as_object()
            {
                let mut first = true;
                for (key, val) in obj {
                    if !first {
                        out.push(',');
                    }
                    first = false;
                    out.push_str(key);
                    out.push(':');
                    format_gemma4_value(&mut out, val);
                }
            }
            out.push_str("}<tool_call|>");
        }
        out
    }

    fn format_tool_response(&self, content: &str) -> String {
        format!("<|tool_response>response:result{{value:<|\"|>{content}<|\"|>}}<tool_response|>")
    }
}

// ── Mistral native parser (Mistral Small 4) ──
