// SPDX-License-Identifier: AGPL-3.0-only
//
// `Grammar` handle — W7 compatibility shim.
//
// The vendored C++ `xgrammar-rs` exposed a `Grammar` handle type
// (`from_ebnf`, `from_json_schema`, `from_regex`, `from_structural_tag`,
// `builtin_json_grammar`) which `GrammarCompiler::compile_grammar`
// consumed by `&Grammar`. The pure-Rust port works directly on the
// `GrammarData` AST and has no separate handle type.
//
// This thin newtype restores the vendored surface: it owns a parsed,
// **un-normalized** `GrammarData` (the compiler runs normalization).
// Atlas's `grammar/compile_misc.rs` uses `Grammar::from_structural_tag`
// followed by `compiler.compile_grammar(&grammar)`.

use crate::grammar::{GrammarData, parse_ebnf};
use crate::schema::{SchemaConverterOptions, builtin_json_grammar_ebnf, json_schema_to_ebnf};
use crate::structural_tag::structural_tag_to_grammar;

/// A parsed grammar, ready to be handed to
/// [`crate::GrammarCompiler::compile_grammar`].
///
/// Port of the vendored `xgrammar::Grammar` handle.
#[derive(Debug, Clone)]
pub struct Grammar {
    data: GrammarData,
}

impl Grammar {
    /// Construct a grammar from an EBNF string.
    ///
    /// Port of the vendored `Grammar::from_ebnf`.
    pub fn from_ebnf(ebnf_string: &str, root_rule_name: &str) -> Result<Self, String> {
        parse_ebnf(ebnf_string, root_rule_name)
            .map(|data| Self { data })
            .map_err(|e| e.to_string())
    }

    /// Construct a grammar from a JSON schema.
    ///
    /// Port of the vendored `Grammar::from_json_schema`. `indent`,
    /// `separators` and `max_whitespace_cnt` follow the same convention
    /// as `json.dumps()`; `print_converted_ebnf` is accepted for parity
    /// and has no effect.
    pub fn from_json_schema(
        schema: &str,
        any_whitespace: bool,
        indent: Option<i32>,
        separators: Option<(impl AsRef<str>, impl AsRef<str>)>,
        strict_mode: bool,
        max_whitespace_cnt: Option<i32>,
        print_converted_ebnf: bool,
    ) -> Result<Self, String> {
        let _ = print_converted_ebnf;
        let options = SchemaConverterOptions {
            any_whitespace,
            indent,
            separators: separators.map(|(c, s)| (c.as_ref().to_string(), s.as_ref().to_string())),
            strict_mode,
            max_whitespace_cnt,
            ..SchemaConverterOptions::default()
        };
        let ebnf = json_schema_to_ebnf(schema, &options).map_err(|e| e.to_string())?;
        Self::from_ebnf(&ebnf, "root")
    }

    /// Construct a grammar from a structural tag JSON document.
    ///
    /// Port of the vendored `Grammar::from_structural_tag`.
    pub fn from_structural_tag(structural_tag_json: &str) -> Result<Self, String> {
        structural_tag_to_grammar(structural_tag_json)
            .map(|data| Self { data })
            .map_err(|e| e.to_string())
    }

    /// Build the standard JSON grammar.
    ///
    /// Port of the vendored `Grammar::builtin_json_grammar`.
    pub fn builtin_json_grammar() -> Result<Self, String> {
        Self::from_ebnf(&builtin_json_grammar_ebnf(), "root")
    }

    /// The grammar in EBNF form.
    ///
    /// Port of the vendored `Grammar::to_string_ebnf`.
    pub fn to_string_ebnf(&self) -> String {
        crate::grammar::print_grammar(&self.data)
    }

    /// Borrow the underlying parsed AST.
    pub fn data(&self) -> &GrammarData {
        &self.data
    }

    /// Consume the handle, yielding the underlying AST.
    pub fn into_data(self) -> GrammarData {
        self.data
    }
}

impl core::fmt::Display for Grammar {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.to_string_ebnf())
    }
}

impl From<GrammarData> for Grammar {
    fn from(data: GrammarData) -> Self {
        Self { data }
    }
}
