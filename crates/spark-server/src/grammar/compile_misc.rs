// SPDX-License-Identifier: AGPL-3.0-only

//! Misc grammar compilation: structural-tag, JSON schema, JSON, EBNF.

use xgrammar::CompiledGrammar;

use super::engine::{GrammarEngine, GrammarError};

impl GrammarEngine {
    /// Build and compile a structural tag grammar from raw JSON components.
    /// This bypasses xgrammar-rs's `compile_structural_tag` wrapper to access
    /// `at_least_one` and `stop_after_first` parameters.
    pub(super) fn compile_structural_tag_raw(
        &mut self,
        triggers: &[String],
        tags: &[serde_json::Value],
        at_least_one: bool,
        stop_after_first: bool,
    ) -> Result<CompiledGrammar, GrammarError> {
        let structural_tag_json = serde_json::json!({
            "type": "structural_tag",
            "format": {
                "type": "triggered_tags",
                "triggers": triggers,
                "tags": tags,
                "at_least_one": at_least_one,
                "stop_after_first": stop_after_first,
            }
        })
        .to_string();

        let grammar = xgrammar::Grammar::from_structural_tag(&structural_tag_json)
            .map_err(GrammarError::Compilation)?;
        self.compiler
            .compile_grammar(&grammar)
            .map_err(GrammarError::Compilation)
    }

    // ── JSON schema grammar ──

    /// Cap on consecutive inter-token whitespace characters in
    /// response_format json_schema grammars. Unlimited whitespace
    /// (`None`) gives a degenerating model a grammar-legal runway:
    /// measured on the FP8 flagship (issue #131, N=8 strict json_schema
    /// at temp 0), 5/8 requests padded hundreds of \n/\t between JSON
    /// tokens until max_tokens, leaving unterminated JSON. 8 permits
    /// normal pretty-printing (newline + indent) while making runaway
    /// whitespace grammar-illegal, so the matcher forces a structural
    /// token instead.
    const MAX_JSON_SCHEMA_WHITESPACE: i32 = 8;

    /// Compile a grammar that enforces a JSON schema.
    pub fn compile_json_schema(&mut self, schema: &str) -> Result<CompiledGrammar, GrammarError> {
        self.compiler
            .compile_json_schema(
                schema,
                true,                 // any_whitespace
                None,                 // indent
                None::<(&str, &str)>, // separators
                true,                 // strict_mode
                Some(Self::MAX_JSON_SCHEMA_WHITESPACE),
            )
            .map_err(GrammarError::Compilation)
    }

    /// Compile the built-in JSON grammar (any valid JSON).
    pub fn compile_json_grammar(&mut self) -> Result<CompiledGrammar, GrammarError> {
        self.compiler
            .compile_builtin_json_grammar()
            .map_err(GrammarError::Compilation)
    }

    /// Compile a grammar from an EBNF string.
    pub fn compile_ebnf(
        &mut self,
        ebnf: &str,
        root_rule: &str,
    ) -> Result<CompiledGrammar, GrammarError> {
        self.compiler
            .compile_grammar_from_ebnf(ebnf, root_rule)
            .map_err(GrammarError::Compilation)
    }
}
