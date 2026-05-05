// SPDX-License-Identifier: AGPL-3.0-only

//! Tool-call grammar compilation (Hermes, BareJson, Qwen3Coder, Gemma4, MiniMaxXml).

use std::collections::HashMap;

use xgrammar::CompiledGrammar;

use crate::tool_parser::ToolDefinition;

use super::engine::{GrammarEngine, GrammarError};
use super::schema::{enforce_min_length_on_required_strings, sanitize_schema_for_grammar};

impl GrammarEngine {
    // ── Tool call grammars ──

    /// Compile a grammar for Hermes-format tool calls.
    ///
    /// Hermes format: `<tool_call>{"name":"fn","arguments":{...}}</tool_call>`
    ///
    /// Builds raw structural tag JSON with `at_least_one` / `stop_after_first`
    /// (bypasses xgrammar-rs wrapper which doesn't expose these fields).
    ///
    /// - `use_triggers=true` (tool_choice="auto"): triggers active, model chooses freely
    /// - `use_triggers=false` (tool_choice="required"): at_least_one + stop_after_first
    pub fn compile_hermes_tool_grammar(
        &mut self,
        tools: &[ToolDefinition],
        use_triggers: bool,
    ) -> Result<CompiledGrammar, GrammarError> {
        if tools.is_empty() {
            return Err(GrammarError::NoTools);
        }

        let mut tag_entries = Vec::with_capacity(tools.len());
        let mut triggers = Vec::new();
        let mut seen_triggers = HashMap::<String, bool>::new();

        for tool in tools {
            let name = &tool.function.name;
            let raw_schema = tool
                .function
                .parameters
                .as_ref()
                .cloned()
                .unwrap_or_else(|| serde_json::json!({"type":"object","properties":{}}));
            let raw_schema = match sanitize_schema_for_grammar(&raw_schema) {
                Some(s) => s,
                None => {
                    tracing::warn!("Skipping tool '{name}' in grammar — schema unsanitizable");
                    continue;
                }
            };
            if raw_schema.get("properties").is_none() && raw_schema.get("type").is_none() {
                tracing::warn!(
                    "Skipping tool '{name}' in grammar — schema has no properties or type"
                );
                continue;
            }
            let schema = enforce_min_length_on_required_strings(&raw_schema);

            let begin = format!(r#"<tool_call>{{"name":"{name}","arguments":"#);
            tag_entries.push(serde_json::json!({
                "type": "tag",
                "begin": begin,
                "content": {"type": "json_schema", "json_schema": schema},
                "end": "}</tool_call>",
            }));

            let trigger = r#"<tool_call>{"name":""#.to_string();
            if !seen_triggers.contains_key(&trigger) {
                seen_triggers.insert(trigger.clone(), true);
                triggers.push(trigger);
            }
        }

        if tag_entries.is_empty() {
            return Err(GrammarError::NoTools);
        }

        // auto: at_least_one=false (model freely chooses text or tool)
        // required: at_least_one=true + stop_after_first=true (EOS suppressed until one tool call)
        let at_least_one = !use_triggers;
        let stop_after_first = !use_triggers;

        self.compile_structural_tag_raw(&triggers, &tag_entries, at_least_one, stop_after_first)
    }

    /// Compile a grammar for bare-JSON tool calls (no `<tool_call>` wrapper).
    ///
    /// Format: `{"name":"<one_of_tools>","arguments":<schema>}` — top-level
    /// JSON object, nothing else. Used by Nemotron-Super-120B which falls
    /// into degenerate token loops when the qwen3_coder `<tool_call>` wrapper
    /// is forced (its training distribution does not cover that prefix).
    ///
    /// - `use_triggers=true` (tool_choice="auto"): triggers active, model chooses freely
    /// - `use_triggers=false` (tool_choice="required"): at_least_one + stop_after_first
    pub fn compile_bare_json_tool_grammar(
        &mut self,
        tools: &[ToolDefinition],
        use_triggers: bool,
    ) -> Result<CompiledGrammar, GrammarError> {
        if tools.is_empty() {
            return Err(GrammarError::NoTools);
        }

        let mut tag_entries = Vec::with_capacity(tools.len());
        let mut triggers = Vec::new();
        let mut seen_triggers = HashMap::<String, bool>::new();

        for tool in tools {
            let name = &tool.function.name;
            let raw_schema = tool
                .function
                .parameters
                .as_ref()
                .cloned()
                .unwrap_or_else(|| serde_json::json!({"type":"object","properties":{}}));
            let raw_schema = match sanitize_schema_for_grammar(&raw_schema) {
                Some(s) => s,
                None => {
                    tracing::warn!("Skipping tool '{name}' in grammar — schema unsanitizable");
                    continue;
                }
            };
            if raw_schema.get("properties").is_none() && raw_schema.get("type").is_none() {
                tracing::warn!(
                    "Skipping tool '{name}' in grammar — schema has no properties or type"
                );
                continue;
            }
            let schema = enforce_min_length_on_required_strings(&raw_schema);

            let begin = format!(r#"{{"name":"{name}","arguments":"#);
            tag_entries.push(serde_json::json!({
                "type": "tag",
                "begin": begin,
                "content": {"type": "json_schema", "json_schema": schema},
                "end": "}",
            }));
        }

        let trigger = r#"{"name":""#.to_string();
        if !seen_triggers.contains_key(&trigger) {
            seen_triggers.insert(trigger.clone(), true);
            triggers.push(trigger);
        }

        if tag_entries.is_empty() {
            return Err(GrammarError::NoTools);
        }

        let at_least_one = !use_triggers;
        let stop_after_first = !use_triggers;

        self.compile_structural_tag_raw(&triggers, &tag_entries, at_least_one, stop_after_first)
    }

    /// Compile a grammar for Qwen3-Coder XML tool calls.
    ///
    /// Format: `<tool_call>\n<function=name>\n<parameter=key>\nvalue\n</parameter>\n</function>\n</tool_call>`
    ///
    /// Uses XGrammar's `qwen_xml_parameter` content type for native XML parameter support.
    /// Falls back to `json_schema` if `qwen_xml_parameter` is not available in this XGrammar build.
    pub fn compile_qwen3_coder_tool_grammar(
        &mut self,
        tools: &[ToolDefinition],
        use_triggers: bool,
    ) -> Result<CompiledGrammar, GrammarError> {
        if tools.is_empty() {
            return Err(GrammarError::NoTools);
        }

        let mut tag_entries = Vec::with_capacity(tools.len());
        let mut triggers = Vec::new();
        let mut seen_triggers = HashMap::<String, bool>::new();

        // Pre-sanitize all schemas so the fallback path can reuse them.
        struct SanitizedTool {
            name: String,
            schema: serde_json::Value,
        }
        let mut sanitized_tools = Vec::with_capacity(tools.len());
        for tool in tools {
            let name = &tool.function.name;
            let raw_schema = tool
                .function
                .parameters
                .as_ref()
                .cloned()
                .unwrap_or_else(|| serde_json::json!({"type":"object","properties":{}}));
            let raw_schema = match sanitize_schema_for_grammar(&raw_schema) {
                Some(s) => s,
                None => {
                    tracing::warn!("Skipping tool '{name}' in grammar — schema unsanitizable");
                    continue;
                }
            };
            if raw_schema.get("properties").is_none() && raw_schema.get("type").is_none() {
                tracing::warn!(
                    "Skipping tool '{name}' in grammar — schema has no properties or type"
                );
                continue;
            }
            let schema = enforce_min_length_on_required_strings(&raw_schema);
            sanitized_tools.push(SanitizedTool {
                name: name.clone(),
                schema,
            });
        }

        for st in &sanitized_tools {
            let begin = format!("<tool_call>\n<function={}>\n", st.name);
            let end = "\n</function>\n</tool_call>";
            tag_entries.push(serde_json::json!({
                "type": "tag",
                "begin": begin,
                "content": {"type": "qwen_xml_parameter", "json_schema": st.schema},
                "end": end,
            }));
            let trigger = format!("<tool_call>\n<function={}", st.name);
            if !seen_triggers.contains_key(&trigger) {
                seen_triggers.insert(trigger.clone(), true);
                triggers.push(trigger);
            }
        }

        if tag_entries.is_empty() {
            return Err(GrammarError::NoTools);
        }

        let at_least_one = !use_triggers;
        let stop_after_first = !use_triggers;

        match self.compile_structural_tag_raw(
            &triggers,
            &tag_entries,
            at_least_one,
            stop_after_first,
        ) {
            Ok(compiled) => Ok(compiled),
            Err(e) => {
                // Fall back to json_schema content type if qwen_xml_parameter not supported.
                tracing::warn!(
                    "qwen_xml_parameter grammar failed ({e:?}), retrying with json_schema"
                );
                let tag_entries_fallback: Vec<serde_json::Value> = sanitized_tools
                    .iter()
                    .map(|st| {
                        serde_json::json!({
                            "type": "tag",
                            "begin": format!("<tool_call>\n<function={}>\n", st.name),
                            "content": {"type": "json_schema", "json_schema": st.schema},
                            "end": "\n</function>\n</tool_call>",
                        })
                    })
                    .collect();
                self.compile_structural_tag_raw(
                    &triggers,
                    &tag_entries_fallback,
                    at_least_one,
                    stop_after_first,
                )
            }
        }
    }

    /// Compile a grammar for Gemma-4 native tool calls.
    ///
    /// Gemma-4's native format uses special sentinel tokens:
    ///   `<|tool_call>call:NAME{"key":"val",...}<tool_call|>`
    ///
    /// We use standard JSON for the argument block (unlike Gemma's original
    /// unquoted-key / `<|"|>` delimiter format). The existing parser
    /// (`parse_gemma4_native_call` in tool_parser.rs) accepts standard JSON
    /// transparently: the `gemma4_to_json` converter is a no-op when the body
    /// is already valid JSON, and unquoted-key quoting only fires when needed.
    ///
    /// Grammar-constraining the output ensures the model ALWAYS emits the
    /// exact `<|tool_call>call:NAME{...}<tool_call|>` framing instead of
    /// narrating tool calls in plain text (the WARN root cause on 26B Search
    /// and 31B Weather+Search).
    pub fn compile_gemma4_tool_grammar(
        &mut self,
        tools: &[ToolDefinition],
        use_triggers: bool,
    ) -> Result<CompiledGrammar, GrammarError> {
        if tools.is_empty() {
            return Err(GrammarError::NoTools);
        }

        let mut tag_entries = Vec::with_capacity(tools.len());
        let mut triggers = Vec::new();
        let mut seen_triggers = HashMap::<String, bool>::new();

        for tool in tools {
            let name = &tool.function.name;
            let raw_schema = tool
                .function
                .parameters
                .as_ref()
                .cloned()
                .unwrap_or_else(|| serde_json::json!({"type":"object","properties":{}}));
            let raw_schema = match sanitize_schema_for_grammar(&raw_schema) {
                Some(s) => s,
                None => {
                    tracing::warn!("Skipping tool '{name}' in grammar — schema unsanitizable");
                    continue;
                }
            };
            if raw_schema.get("properties").is_none() && raw_schema.get("type").is_none() {
                tracing::warn!(
                    "Skipping tool '{name}' in grammar — schema has no properties or type"
                );
                continue;
            }
            let schema = enforce_min_length_on_required_strings(&raw_schema);

            // Gemma-4 sentinel tokens frame the call; JSON body in between.
            let begin = format!("<|tool_call>call:{name}");
            let end = "<tool_call|>";
            tag_entries.push(serde_json::json!({
                "type": "tag",
                "begin": begin,
                "content": {"type": "json_schema", "json_schema": schema},
                "end": end,
            }));

            let trigger = "<|tool_call>call:".to_string();
            if !seen_triggers.contains_key(&trigger) {
                seen_triggers.insert(trigger.clone(), true);
                triggers.push(trigger);
            }
        }

        if tag_entries.is_empty() {
            return Err(GrammarError::NoTools);
        }

        let at_least_one = !use_triggers;
        let stop_after_first = !use_triggers;

        self.compile_structural_tag_raw(&triggers, &tag_entries, at_least_one, stop_after_first)
    }

    /// F66 (2026-04-29): MiniMax M2.7 XML tool-call grammar.
    ///
    /// Native MiniMax format:
    /// ```xml
    /// <minimax:tool_call>
    /// <invoke name="tool_name">
    /// <parameter name="key1">value1</parameter>
    /// <parameter name="key2">value2</parameter>
    /// </invoke>
    /// </minimax:tool_call>
    /// ```
    ///
    /// Without this grammar, fix39 testing showed the model emit doubled
    /// tokens (`<invokeinvoke`, `<parameterparameter`, repeated phrases)
    /// when invoked through `--tool-call-parser minimax_xml` — XGrammar
    /// was warning "unknown parser format 'minimax_xml', skipping
    /// constrained decoding" and the unconstrained model freelanced
    /// into degenerate token loops at the tool-call boundary.
    ///
    /// Strategy: per-tool structural_tag with the OUTER frame fixed
    /// (`<minimax:tool_call>\n<invoke name="X">` and the closing
    /// `</invoke>\n</minimax:tool_call>`) and `any_text` for the body.
    /// This forces the wrapper structure to be exactly right (eliminates
    /// the `<invokeinvoke` corruption) while letting the model emit any
    /// `<parameter name="K">V</parameter>` sequence inside — the
    /// MinimaxXmlParser at parse time extracts those parameters from the
    /// body.
    ///
    /// The looser `any_text` body content was chosen over a strict
    /// per-parameter schema (which would require a custom EBNF or
    /// nested triggered_tags) because:
    ///   1. The OUTER frame doubling is the actual corruption source —
    ///      eliminating it stops the loop class.
    ///   2. MiniMax M2.7 is well-trained on the inner format and emits
    ///      it cleanly when the outer framing is constrained.
    ///   3. The output-side MinimaxXmlParser performs the strict
    ///      structural validation when extracting parameters.
    pub fn compile_minimax_xml_tool_grammar(
        &mut self,
        tools: &[ToolDefinition],
        use_triggers: bool,
    ) -> Result<CompiledGrammar, GrammarError> {
        if tools.is_empty() {
            return Err(GrammarError::NoTools);
        }

        let mut tag_entries = Vec::with_capacity(tools.len());

        for tool in tools {
            let name = &tool.function.name;
            // Schema sanitization (kept consistent with other parsers
            // even though we don't use the schema for body constraint
            // — this still catches malformed schemas at compile time
            // so they're reported uniformly).
            let raw_schema = tool
                .function
                .parameters
                .as_ref()
                .cloned()
                .unwrap_or_else(|| serde_json::json!({"type":"object","properties":{}}));
            if sanitize_schema_for_grammar(&raw_schema).is_none() {
                tracing::warn!("Skipping tool '{name}' in grammar — schema unsanitizable");
                continue;
            }

            let begin = format!("<minimax:tool_call>\n<invoke name=\"{name}\">");
            let end = "</invoke>\n</minimax:tool_call>";
            tag_entries.push(serde_json::json!({
                "type": "tag",
                "begin": begin,
                "content": {"type": "any_text"},
                "end": end,
            }));
        }

        if tag_entries.is_empty() {
            return Err(GrammarError::NoTools);
        }

        // F67 (2026-04-29): SHORT shared trigger. xgrammar's
        // `triggered_tags` matcher is fully unconstrained until a
        // complete trigger string has been emitted; only after that
        // does it lock subsequent tokens to one of the registered
        // `tag.begin` continuations. With per-tool LATE triggers like
        // `<minimax:tool_call>\n<invoke name="bash"`, the model could
        // emit `<minimax:tool_call></minimax:tool_call>` (no `\n<invoke
        // …>` ever appears), the trigger never fired, and `at_least_one`
        // only blocked EOS — producing the
        // `<minimax:tool_call></minimax:tool_call>...` envelope loop
        // observed in fix40 live testing. The SHORT trigger
        // `<minimax:tool_call>` engages the moment the model opens the
        // envelope, after which xgrammar's TagDispatch alternation
        // forces one of `\n<invoke name="<TOOL>">` for each registered
        // tool — making the close-immediate degenerate output
        // unreachable by construction (proved by
        // `test_minimax_xml_grammar_rejects_degenerate`).
        let triggers = vec!["<minimax:tool_call>".to_string()];

        let at_least_one = !use_triggers;
        let stop_after_first = !use_triggers;

        self.compile_structural_tag_raw(&triggers, &tag_entries, at_least_one, stop_after_first)
    }
}
