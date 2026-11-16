// SPDX-License-Identifier: AGPL-3.0-only
// Wire format of LFM3's tool calls, from the LiquidAI/LFM3-1.7B model card ("Tool calling") and the `response_template` in its tokenizer_config.json (open `<tool_call>\s*<function=NAME>`, parameters `<parameter=KEY>\n?VALUE\n?</parameter>`), cross-checked against vLLM's tool_parsers (Apache-2.0, Copyright contributors to the vLLM project). No upstream code is copied; extraction is Atlas's shared qwen3_coder path.
#![allow(unused_imports, dead_code)]

use super::*;

/// LFM3's tool-call parser.
///
/// Liquid's XML is exactly the qwen3_coder block form:
///
/// ```text
/// <tool_call>
/// <function=get_weather>
/// <parameter=city>
/// Paris
/// </parameter>
/// </function>
/// </tool_call>
/// ```
///
/// so extraction, streaming detection, typed-argument coercion and the
/// XGrammar structural tag are the shared qwen3_coder machinery (the output
/// parse functions are format-agnostic). What differs is the PROMPT: the
/// checkpoint's own chat template renders the tool schemas (and the deferred
/// `<additional_tools>` block) and re-renders earlier assistant `tool_calls`,
/// so this parser contributes no system prompt beyond the `tool_choice`
/// constraint. Adding Atlas's qwen3_coder instruction block on top would put a
/// second, different copy of the schema in front of a model trained on one.
pub struct Lfm3Parser;

impl ToolCallParser for Lfm3Parser {
    fn name(&self) -> &str {
        "lfm3"
    }

    fn wants_typed_arguments(&self) -> bool {
        true
    }

    fn param_value_close_delim(&self) -> Option<&'static str> {
        Qwen3CoderParser.param_value_close_delim()
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

    fn system_prompt(
        &self,
        _tools: &[ToolDefinition],
        tool_choice: &ToolChoice,
        _levers: &super::PromptLevers,
    ) -> String {
        // Only the `tool_choice` constraint; the schemas are the template's.
        let mut prompt = String::new();
        append_tool_choice_instruction(&mut prompt, tool_choice);
        prompt
    }

    fn format_tool_calls(&self, calls: &[IncomingToolCall]) -> String {
        Qwen3CoderParser.format_tool_calls(calls)
    }

    fn leak_markers(&self) -> LeakMarkers {
        Qwen3CoderParser.leak_markers()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(call: &ToolCall) -> serde_json::Value {
        serde_json::from_str(&call.function.arguments).expect("arguments are JSON")
    }

    /// The model card's example, byte for byte.
    const README_CALL: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";

    #[test]
    fn the_model_cards_example_parses() {
        let (content, calls) = parse_tool_calls(README_CALL);
        assert!(content.is_none(), "{content:?}");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(args(&calls[0])["city"], "Paris");
    }

    #[test]
    fn prose_before_the_call_is_kept_as_content() {
        let text = format!("I will look that up.\n{README_CALL}");
        let (content, calls) = parse_tool_calls(&text);
        assert_eq!(content.as_deref(), Some("I will look that up."));
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn parallel_calls_keep_their_order_and_distinct_ids() {
        let text = format!(
            "{README_CALL}\n<tool_call>\n<function=get_time>\n<parameter=zone>\nUTC\n\
             </parameter>\n<parameter=format>\n24h\n</parameter>\n</function>\n</tool_call>"
        );
        let (_, calls) = parse_tool_calls(&text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
        let second = args(&calls[1]);
        assert_eq!(second["zone"], "UTC");
        assert_eq!(second["format"], "24h");
        assert_ne!(calls[0].id, calls[1].id);
    }

    #[test]
    fn a_multi_line_value_survives_whole() {
        let text = "<tool_call>\n<function=write_file>\n<parameter=path>\n/tmp/a.txt\n</parameter>\n\
                    <parameter=content>\nline one\nline two <b>bold</b> a < b\n</parameter>\n\
                    </function>\n</tool_call>";
        let (_, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        let a = args(&calls[0]);
        assert_eq!(a["path"], "/tmp/a.txt");
        assert_eq!(a["content"], "line one\nline two <b>bold</b> a < b");
    }

    #[test]
    fn a_zero_argument_call_parses() {
        let (_, calls) =
            parse_tool_calls("<tool_call>\n<function=list_tools>\n</function>\n</tool_call>");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "list_tools");
        assert_eq!(args(&calls[0]), serde_json::json!({}));
    }

    #[test]
    fn text_without_a_call_is_plain_content() {
        let (content, calls) = parse_tool_calls("The capital of France is Paris.");
        assert_eq!(content.as_deref(), Some("The capital of France is Paris."));
        assert!(calls.is_empty());
    }

    #[test]
    fn format_tool_calls_round_trips_through_the_shared_parser() {
        let call = IncomingToolCall {
            id: None,
            function: IncomingFunction {
                name: "get_weather".into(),
                arguments: "{\"city\":\"Paris\",\"days\":3}".into(),
            },
        };
        let text = Lfm3Parser.format_tool_calls(&[call]);
        let (_, calls) = parse_tool_calls(&text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let a = args(&calls[0]);
        assert_eq!(a["city"], "Paris");
        assert_eq!(a["days"], "3");
    }

    #[test]
    fn the_registry_resolves_lfm3_and_it_is_a_grammar_parser() {
        let format: ToolCallFormat = "lfm3".parse().expect("lfm3 is registered");
        assert_eq!(format.name(), "lfm3");
        let parser = format.into_parser();
        assert_eq!(parser.name(), "lfm3");
        assert!(parser.wants_typed_arguments());
        assert!(parser.has_tool_grammar());
        assert_eq!(parser.param_value_close_delim(), Some("</parameter>"));
        assert!(format.has_grammar());
    }

    fn tool(name: &str) -> ToolDefinition {
        ToolDefinition {
            tool_type: "function".into(),
            function: FunctionDefinition {
                name: name.into(),
                description: Some("d".into()),
                parameters: None,
            },
        }
    }

    #[test]
    fn the_prompt_leaves_the_schema_to_the_checkpoint_template() {
        let tools = vec![tool("get_weather")];
        let levers = super::super::PromptLevers::OFF;
        let auto = Lfm3Parser.system_prompt(&tools, &ToolChoice::Mode("auto".into()), &levers);
        assert!(auto.is_empty(), "no schema, no instruction block: {auto:?}");
        let required =
            Lfm3Parser.system_prompt(&tools, &ToolChoice::Mode("required".into()), &levers);
        assert!(
            required.contains("MUST call at least one function"),
            "{required}"
        );
        assert!(!required.contains("get_weather"));
        let named = Lfm3Parser.system_prompt(
            &tools,
            &ToolChoice::Specific {
                function: ToolChoiceFunction {
                    name: "get_weather".into(),
                },
            },
            &levers,
        );
        assert!(named.contains("'get_weather'"), "{named}");
        assert!(!named.contains("<tools>"));
    }

    #[test]
    fn leak_markers_and_close_delimiter_are_qwen3_coders() {
        let m = Lfm3Parser.leak_markers();
        let q = Qwen3CoderParser.leak_markers();
        assert_eq!(m.orphan_open, q.orphan_open);
        assert_eq!(m.close, q.close);
        assert!(m.close.contains(&"</tool_call>"));
    }
}
