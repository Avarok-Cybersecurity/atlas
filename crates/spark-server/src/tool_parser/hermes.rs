// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::*;

/// Hermes-2 format: JSON `{"name":"fn","arguments":{...}}` inside `<tool_call>` tags.
///
/// Used by Qwen3-VL-30B and Qwen3-Next-80B.
pub struct HermesParser;

impl ToolCallParser for HermesParser {
    fn name(&self) -> &str {
        "hermes"
    }

    fn compile_tool_grammar(
        &self,
        engine: &mut GrammarEngine,
        tools: &[ToolDefinition],
        use_triggers: bool,
    ) -> Option<Result<CompiledGrammar, GrammarError>> {
        Some(engine.compile_hermes_tool_grammar(tools, use_triggers))
    }

    fn has_tool_grammar(&self) -> bool {
        true
    }

    fn system_prompt(&self, tools: &[ToolDefinition], tool_choice: &ToolChoice) -> String {
        let tools_json = serde_json::to_string(tools).unwrap_or_else(|_| "[]".into());
        let mut prompt = format!(
            "You are a function calling AI model. You are provided with function signatures within \
             <tools></tools> XML tags. You may call one or more functions to assist with the user \
             query. Don't make assumptions about what values to plug into functions. Here are the \
             available tools:\n<tools>\n{tools_json}\n</tools>\n\n\
             For each function call, return a JSON object with function name and arguments within \
             <tool_call></tool_call> XML tags:\n<tool_call>\n\
             {{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call>"
        );
        append_tool_choice_instruction(&mut prompt, tool_choice);
        prompt
    }

    fn format_tool_calls(&self, calls: &[IncomingToolCall]) -> String {
        let mut out = String::new();
        for tc in calls {
            let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                .unwrap_or(serde_json::Value::Object(Default::default()));
            out.push_str(&format!(
                "\n<tool_call>\n{}\n</tool_call>",
                serde_json::json!({"name": tc.function.name, "arguments": args})
            ));
        }
        out
    }
}

// ── Qwen3-Coder parser (Qwen3.5, Nemotron-H) ──
