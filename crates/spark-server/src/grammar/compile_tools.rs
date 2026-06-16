// SPDX-License-Identifier: AGPL-3.0-only

//! Tool-call grammar compilation (Hermes, BareJson, Qwen3Coder, Gemma4, MiniMaxXml).

use std::collections::HashMap;

use xgrammar::CompiledGrammar;

use crate::tool_parser::ToolDefinition;

use super::engine::{GrammarEngine, GrammarError};
use super::schema::{enforce_min_length_on_required_strings, sanitize_schema_for_grammar};

mod ebnf;
mod xml_grammars;

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
}
