// SPDX-License-Identifier: AGPL-3.0-only
//
// `GrammarCompiler` façade — W7 compatibility shim.
//
// Restores the vendored `xgrammar-rs` `GrammarCompiler` signatures on
// top of the pure-Rust `crate::compiler::GrammarCompiler`. The vendored
// crate took `&TokenizerInfo` + `i32`/`isize` integers and returned
// `Result<_, String>`; the pure-Rust core takes the `TokenizerInfo` by
// value + `usize`/`i64` and returns a typed `CompileError`. This
// newtype bridges both so Atlas's `grammar/engine.rs` and
// `grammar/compile_*.rs` compile unchanged.

use crate::compiler::GrammarCompiler as CoreCompiler;

use super::CompiledGrammar;
use super::grammar::Grammar;
use super::tokenizer::TokenizerInfo;

/// Compiler that turns grammars / schemas / structural tags into a
/// [`CompiledGrammar`], bound to a single tokenizer.
///
/// Port of the vendored `xgrammar::GrammarCompiler`.
pub struct GrammarCompiler {
    inner: CoreCompiler,
}

impl GrammarCompiler {
    /// Construct the compiler.
    ///
    /// `max_threads` and `cache_limit_bytes` keep the vendored integer
    /// types (`i32` / `isize`); they are range-checked and forwarded to
    /// the pure-Rust core. Port of the vendored `GrammarCompiler::new`.
    pub fn new(
        tokenizer_info: &TokenizerInfo,
        max_threads: i32,
        cache_enabled: bool,
        cache_limit_bytes: isize,
    ) -> Result<Self, String> {
        if max_threads < 1 {
            return Err(format!("max_threads must be >= 1, got {max_threads}"));
        }
        if (cache_limit_bytes as i64) < -1 {
            return Err(format!(
                "cache_limit_bytes must be -1 (unlimited) or non-negative, got {cache_limit_bytes}"
            ));
        }
        Ok(Self {
            inner: CoreCompiler::new(
                tokenizer_info.core_clone(),
                max_threads as usize,
                cache_enabled,
                cache_limit_bytes as i64,
            ),
        })
    }

    /// Compile a JSON schema. Port of the vendored
    /// `GrammarCompiler::compile_json_schema`.
    #[allow(clippy::too_many_arguments)]
    pub fn compile_json_schema(
        &mut self,
        schema: &str,
        any_whitespace: bool,
        indent: Option<i32>,
        separators: Option<(impl AsRef<str>, impl AsRef<str>)>,
        strict_mode: bool,
        max_whitespace_cnt: Option<i32>,
    ) -> Result<CompiledGrammar, String> {
        let separators = separators.map(|(c, s)| (c.as_ref().to_string(), s.as_ref().to_string()));
        self.inner
            .compile_json_schema(
                schema,
                any_whitespace,
                indent,
                separators,
                strict_mode,
                max_whitespace_cnt,
            )
            .map_err(|e| e.to_string())
    }

    /// Compile the built-in standard-JSON grammar. Port of the vendored
    /// `GrammarCompiler::compile_builtin_json_grammar`.
    pub fn compile_builtin_json_grammar(&mut self) -> Result<CompiledGrammar, String> {
        self.inner
            .compile_builtin_json_grammar()
            .map_err(|e| e.to_string())
    }

    /// Compile a grammar from an EBNF string. Port of the vendored
    /// `GrammarCompiler::compile_grammar_from_ebnf`.
    pub fn compile_grammar_from_ebnf(
        &mut self,
        ebnf_string: &str,
        root_rule_name: &str,
    ) -> Result<CompiledGrammar, String> {
        self.inner
            .compile_grammar_from_ebnf(ebnf_string, root_rule_name)
            .map_err(|e| e.to_string())
    }

    /// Compile a [`Grammar`] handle. Port of the vendored
    /// `GrammarCompiler::compile_grammar`.
    ///
    /// The pure-Rust core consumes a `GrammarData` by value; the handle
    /// is cloned so the caller keeps ownership (vendored took `&Grammar`).
    pub fn compile_grammar(&mut self, grammar: &Grammar) -> Result<CompiledGrammar, String> {
        Ok(self.inner.compile_grammar(grammar.data().clone()))
    }

    /// Compile a grammar from a structural-tag JSON document. Port of
    /// the vendored `GrammarCompiler::compile_structural_tag` (raw-JSON
    /// form).
    pub fn compile_structural_tag_json(
        &mut self,
        structural_tag_json: &str,
    ) -> Result<CompiledGrammar, String> {
        self.inner
            .compile_structural_tag(structural_tag_json)
            .map_err(|e| e.to_string())
    }

    /// Clear the compiled-grammar cache. Port of the vendored
    /// `GrammarCompiler::clear_cache`.
    pub fn clear_cache(&mut self) {
        self.inner.clear_cache();
    }

    /// Approximate cache size in bytes. Port of the vendored
    /// `GrammarCompiler::get_cache_size_bytes`.
    pub fn get_cache_size_bytes(&self) -> i64 {
        self.inner.cache_size_bytes()
    }

    /// The configured cache memory limit (`-1` = unlimited). Port of
    /// the vendored `GrammarCompiler::cache_limit_bytes`.
    pub fn cache_limit_bytes(&self) -> i64 {
        self.inner.cache_limit_bytes()
    }
}
