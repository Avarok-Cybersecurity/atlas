// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::*;

/// Mistral native format: `[TOOL_CALLS]name[ARGS]{"key":"val",...}`
///
/// Delimiters are Mistral special tokens (`[TOOL_CALLS]` and `[ARGS]`).
/// Arguments are JSON like Hermes, but there is no outer `<tool_call>` wrapper.
/// Multiple calls chain: `[TOOL_CALLS]f1[ARGS]{...}[TOOL_CALLS]f2[ARGS]{...}`.
/// The closing delimiter is either the next `[TOOL_CALLS]` or end-of-text.
pub struct MistralNativeParser;

pub(crate) const MISTRAL_TOOL_CALLS_TAG: &str = "[TOOL_CALLS]";
pub(crate) const MISTRAL_ARGS_TAG: &str = "[ARGS]";

impl ToolCallParser for MistralNativeParser {
    fn name(&self) -> &str {
        "mistral"
    }

    fn system_prompt(&self, tools: &[ToolDefinition], tool_choice: &ToolChoice) -> String {
        // Mistral's native Jinja template handles tool injection via its own
        // [AVAILABLE_TOOLS] block. We still provide a system prompt so the
        // model knows the expected output format when the template pathway
        // is not active (e.g. OpenAI-compatible clients bypass the template).
        let tools_json = serde_json::to_string(tools).unwrap_or_else(|_| "[]".into());
        let mut prompt = format!(
            "You have access to the following tools:\n<tools>\n{tools_json}\n</tools>\n\n\
             When you need to call a tool, respond in Mistral native format:\n\
             [TOOL_CALLS]function_name[ARGS]{{\"arg1\": \"val1\", \"arg2\": \"val2\"}}\n\
             Use JSON for arguments. To call multiple tools, chain them:\n\
             [TOOL_CALLS]f1[ARGS]{{...}}[TOOL_CALLS]f2[ARGS]{{...}}"
        );
        append_tool_choice_instruction(&mut prompt, tool_choice);
        prompt
    }

    fn format_tool_calls(&self, calls: &[IncomingToolCall]) -> String {
        let mut out = String::new();
        for tc in calls {
            let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                .unwrap_or(serde_json::Value::Object(Default::default()));
            out.push_str(MISTRAL_TOOL_CALLS_TAG);
            out.push_str(&tc.function.name);
            out.push_str(MISTRAL_ARGS_TAG);
            out.push_str(&serde_json::to_string(&args).unwrap_or_else(|_| "{}".into()));
        }
        out
    }

    fn format_tool_response(&self, content: &str) -> String {
        // Mistral uses [TOOL_RESULTS] blocks in its native template. Provide
        // a wrapper that round-trips through the prompt without confusing the
        // model's own template post-processing.
        format!("[TOOL_RESULTS]{content}[/TOOL_RESULTS]")
    }
}
