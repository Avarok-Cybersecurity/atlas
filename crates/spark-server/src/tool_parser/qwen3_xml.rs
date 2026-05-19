// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::*;

/// Qwen3-XML parser: same wire format as [`Qwen3CoderParser`], but returns
/// schema-typed argument values (integer, boolean, array, object) instead of
/// always returning strings.
///
/// The name mirrors vLLM's `qwen3_xml` tool-call parser, which the Qwen team
/// intends to replace `qwen3_coder` for Qwen3-Coder-class models. Type
/// coercion is applied as a post-processing pass via [`coerce_all`]; the
/// extractor ([`parse_qwen3_coder_call`]) and XGrammar grammar are shared
/// unchanged.
pub struct Qwen3XmlParser;

impl ToolCallParser for Qwen3XmlParser {
    fn name(&self) -> &str {
        "qwen3_xml"
    }

    fn wants_typed_arguments(&self) -> bool {
        true
    }

    fn system_prompt(&self, tools: &[ToolDefinition], tool_choice: &ToolChoice) -> String {
        Qwen3CoderParser.system_prompt(tools, tool_choice)
    }

    fn format_tool_calls(&self, calls: &[IncomingToolCall]) -> String {
        Qwen3CoderParser.format_tool_calls(calls)
    }

    fn format_tool_response(&self, content: &str) -> String {
        Qwen3CoderParser.format_tool_response(content)
    }

    fn leak_markers(&self) -> LeakMarkers {
        Qwen3CoderParser.leak_markers()
    }

    fn compile_tool_grammar(
        &self,
        engine: &mut GrammarEngine,
        tools: &[ToolDefinition],
        use_triggers: bool,
    ) -> Option<Result<CompiledGrammar, GrammarError>> {
        Qwen3CoderParser.compile_tool_grammar(engine, tools, use_triggers)
    }

    fn has_tool_grammar(&self) -> bool {
        true
    }
}
