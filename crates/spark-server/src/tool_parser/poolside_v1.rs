// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// Poolside v1 format:
/// `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>`.
pub struct PoolsideV1Parser;

impl ToolCallParser for PoolsideV1Parser {
    fn name(&self) -> &str {
        "poolside_v1"
    }

    fn system_prompt(
        &self,
        _tools: &[ToolDefinition],
        _tool_choice: &ToolChoice,
        _levers: &PromptLevers,
    ) -> String {
        String::new()
    }

    fn compile_tool_grammar(
        &self,
        engine: &mut GrammarEngine,
        tools: &[ToolDefinition],
        use_triggers: bool,
    ) -> Option<Result<CompiledGrammar, GrammarError>> {
        let value_close = self
            .param_value_close_delim()
            .expect("poolside_v1 declares a parameter-value close delimiter");
        Some(engine.compile_poolside_v1_tool_grammar(tools, use_triggers, value_close))
    }

    fn has_tool_grammar(&self) -> bool {
        true
    }

    fn param_value_close_delim(&self) -> Option<&'static str> {
        Some("</arg_value>")
    }

    fn format_tool_calls(&self, calls: &[IncomingToolCall]) -> String {
        let mut output = String::new();
        for call in calls {
            output.push_str("<tool_call>");
            output.push_str(&call.function.name);
            let args = serde_json::from_str::<serde_json::Value>(&call.function.arguments)
                .unwrap_or_else(|_| serde_json::json!({}));
            if let Some(args) = args.as_object() {
                for (key, value) in args {
                    output.push_str("<arg_key>");
                    output.push_str(key);
                    output.push_str("</arg_key><arg_value>");
                    match value {
                        serde_json::Value::String(value) => output.push_str(value),
                        value => output.push_str(&serde_json::to_string(value).unwrap_or_default()),
                    }
                    output.push_str("</arg_value>");
                }
            }
            output.push_str("</tool_call>");
        }
        output
    }

    fn wants_typed_arguments(&self) -> bool {
        true
    }

    fn leak_markers(&self) -> LeakMarkers {
        LeakMarkers {
            orphan_open: &["<arg_key>", "<arg_value>"],
            close: &["</arg_key>", "</arg_value>", "</tool_call>"],
            envelope_open: &["<tool_call>"],
            envelope_close: &["</tool_call>"],
        }
    }
}

pub(super) fn parse_poolside_v1_call(text: &str) -> Option<ToolCall> {
    let name_end = text.find("<arg_key>").unwrap_or(text.len());
    let name = normalize_tool_name(text[..name_end].trim());
    if !is_tool_name_component(&name) {
        return None;
    }

    let mut args = serde_json::Map::new();
    let mut rest = &text[name_end..];
    while !rest.is_empty() {
        rest = rest.strip_prefix("<arg_key>")?;
        let key_end = rest.find("</arg_key>")?;
        let key = rest[..key_end].trim();
        if !is_tool_name_component(key) || args.contains_key(key) {
            return None;
        }
        rest = &rest[key_end + "</arg_key>".len()..];
        rest = rest.strip_prefix("<arg_value>")?;
        let value_end = rest.find("</arg_value>")?;
        let raw_value = &rest[..value_end];
        let value = serde_json::from_str(raw_value)
            .unwrap_or_else(|_| serde_json::Value::String(raw_value.to_string()));
        args.insert(key.to_string(), value);
        rest = &rest[value_end + "</arg_value>".len()..];
    }

    Some(ToolCall {
        id: next_tool_call_id(),
        call_type: "function".into(),
        function: FunctionCall {
            name,
            arguments: serde_json::to_string(&args).unwrap_or_else(|_| "{}".into()),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_native_poolside_call() {
        let call = parse_poolside_v1_call(
            "Bash<arg_key>command</arg_key><arg_value>pwd</arg_value>\
             <arg_key>timeout</arg_key><arg_value>120</arg_value>",
        )
        .expect("poolside call");
        assert_eq!(call.function.name, "Bash");
        assert_eq!(
            call.function.arguments,
            r#"{"command":"pwd","timeout":120}"#
        );
    }

    #[test]
    fn preserves_unquoted_string_whitespace_exactly() {
        let call = parse_poolside_v1_call(
            "write_file<arg_key>content</arg_key><arg_value>  first\nlast  </arg_value>",
        )
        .expect("poolside call");
        let args: serde_json::Value = serde_json::from_str(&call.function.arguments).unwrap();

        assert_eq!(args["content"], "  first\nlast  ");
    }

    #[test]
    fn rejects_duplicate_argument_keys() {
        let call = parse_poolside_v1_call(
            "write_file<arg_key>path</arg_key><arg_value>/tmp/a</arg_value>\
             <arg_key>path</arg_key><arg_value>/tmp/b</arg_value>",
        );

        assert!(call.is_none());
    }

    #[test]
    fn rejects_incomplete_argument_pair() {
        assert!(
            parse_poolside_v1_call("write_file<arg_key>path</arg_key><arg_value>/tmp/a").is_none()
        );
    }

    #[test]
    fn blocking_parser_accepts_complete_zero_argument_call() {
        let (_, calls) = parse_tool_calls("<tool_call>get_status</tool_call>");

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_status");
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn blocking_parser_does_not_salvage_incomplete_poolside_envelope() {
        let (_, calls) = parse_tool_calls(
            "<tool_call>write_file<arg_key>path</arg_key>\
             <arg_value>/tmp/a</arg_value>",
        );

        assert!(calls.is_empty());
    }

    #[test]
    fn streaming_parser_does_not_emit_incomplete_poolside_envelope() {
        let mut detector = StreamingToolDetector::new();
        let mut outputs = detector.process(
            "<tool_call>write_file<arg_key>path</arg_key>\
             <arg_value>/tmp/a</arg_value>",
        );
        outputs.extend(detector.flush());

        assert!(
            outputs
                .iter()
                .all(|output| matches!(output, DetectorOutput::Content(_)))
        );
    }
}
