// SPDX-License-Identifier: AGPL-3.0-only
//
// GrammarCompiler — port of `class GrammarCompiler` / `GrammarCompiler
// ::Impl` from `cpp/grammar_compiler.cc`.
//
// A tokenizer-bound cache that produces `CompiledGrammar`s, avoiding
// redundant preprocessing of grammars / schemas seen before.

use dashmap::DashMap;

use crate::grammar::functor::{GrammarNormalizer, GrammarOptimizer};
use crate::grammar::{parse_ebnf, GrammarData};
use crate::schema::{builtin_json_grammar_ebnf, json_schema_to_ebnf, SchemaConverterOptions};
use crate::structural_tag::structural_tag_to_grammar;
use crate::tokenizer::TokenizerInfo;

use super::compile::compile_optimized_grammar;
use super::compiled_grammar::CompiledGrammar;

/// An error produced while compiling a grammar, schema or tag.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CompileError {
    /// The EBNF source failed to parse.
    #[error("failed to parse grammar: {0}")]
    Grammar(String),
    /// The JSON schema failed to convert or parse.
    #[error("failed to compile JSON schema: {0}")]
    Schema(String),
    /// A structural tag failed to compile.
    #[error("failed to compile structural tag: {0}")]
    StructuralTag(String),
}

/// The cache key — exactly the request parameters that determine the
/// compiled output. Port of the C++ `GrammarCompilerCacheKeys` union.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CacheKey {
    /// `compile_builtin_json_grammar`.
    BuiltinJson,
    /// `compile_grammar_from_ebnf(ebnf, root_rule)`.
    Ebnf { ebnf: String, root_rule: String },
    /// `compile_json_schema(schema, options...)`.
    Schema {
        schema: String,
        any_whitespace: bool,
        indent: Option<i32>,
        separators: Option<(String, String)>,
        strict_mode: bool,
        max_whitespace_cnt: Option<i32>,
    },
    /// `compile_structural_tag(json)`.
    StructuralTag(String),
}

/// A tokenizer-bound grammar compiler with an internal cache.
///
/// Each compiler instance is tied to one [`TokenizerInfo`]; use a
/// separate compiler per vocabulary. Compilation results are cached
/// (when `cache_enabled`) keyed by the request parameters.
pub struct GrammarCompiler {
    tokenizer_info: TokenizerInfo,
    max_threads: usize,
    cache_enabled: bool,
    cache_limit_bytes: i64,
    cache: DashMap<CacheKey, CompiledGrammar>,
}

impl GrammarCompiler {
    /// Construct a compiler bound to `tokenizer_info`.
    ///
    /// * `max_threads` — upper bound on parallelism for per-state mask
    ///   computation (`1` disables the rayon path). Must be >= 1.
    /// * `cache_enabled` — whether to cache compiled grammars.
    /// * `cache_limit_bytes` — recorded memory budget; `-1` means
    ///   unlimited. (Reported by [`Self::cache_limit_bytes`]; this port
    ///   does not perform LRU eviction — see the module docs.)
    ///
    /// Port of the C++ `GrammarCompiler` constructor.
    pub fn new(
        tokenizer_info: TokenizerInfo,
        max_threads: usize,
        cache_enabled: bool,
        cache_limit_bytes: i64,
    ) -> Self {
        assert!(max_threads >= 1, "max_threads must be >= 1");
        assert!(
            cache_limit_bytes >= -1,
            "cache_limit_bytes must be -1 (unlimited) or non-negative"
        );
        Self {
            tokenizer_info,
            max_threads,
            cache_enabled,
            cache_limit_bytes,
            cache: DashMap::new(),
        }
    }

    /// Run the full W3 pipeline (normalize then optimize) on a parsed
    /// grammar, then compile it against this compiler's tokenizer.
    fn compile_normalized(&self, grammar: GrammarData) -> CompiledGrammar {
        let normalized = GrammarNormalizer::apply(grammar);
        let optimized = GrammarOptimizer::apply(normalized);
        compile_optimized_grammar(optimized, &self.tokenizer_info, self.max_threads)
    }

    /// Fetch from cache or compute via `f`, honoring `cache_enabled`.
    fn get_or_compute<F>(&self, key: CacheKey, f: F) -> CompiledGrammar
    where
        F: FnOnce() -> CompiledGrammar,
    {
        if !self.cache_enabled {
            return f();
        }
        if let Some(hit) = self.cache.get(&key) {
            return hit.clone();
        }
        let value = f();
        self.cache.entry(key).or_insert(value).clone()
    }

    /// Compile a grammar from an EBNF string. Port of
    /// `GrammarCompiler::CompileGrammar(ebnf, root_rule_name)`.
    pub fn compile_grammar_from_ebnf(
        &self,
        ebnf: &str,
        root_rule_name: &str,
    ) -> Result<CompiledGrammar, CompileError> {
        // Parse eagerly so a malformed grammar is a typed error rather
        // than a cached panic.
        let grammar = parse_ebnf(ebnf, root_rule_name)
            .map_err(|e| CompileError::Grammar(e.to_string()))?;
        let key = CacheKey::Ebnf {
            ebnf: ebnf.to_string(),
            root_rule: root_rule_name.to_string(),
        };
        Ok(self.get_or_compute(key, || self.compile_normalized(grammar)))
    }

    /// Compile an already-parsed [`GrammarData`]. Port of
    /// `GrammarCompiler::CompileGrammar(const Grammar&)` — keyed by the
    /// grammar's printed EBNF, as the C++ does.
    pub fn compile_grammar(&self, grammar: GrammarData) -> CompiledGrammar {
        let ebnf = crate::grammar::print_grammar(&grammar);
        let root_rule = grammar.root_rule().name.clone();
        let key = CacheKey::Ebnf {
            ebnf,
            root_rule,
        };
        self.get_or_compute(key, || self.compile_normalized(grammar))
    }

    /// Compile the builtin "any JSON value" grammar. Port of
    /// `GrammarCompiler::CompileBuiltinJSONGrammar`.
    pub fn compile_builtin_json_grammar(&self) -> Result<CompiledGrammar, CompileError> {
        let ebnf = builtin_json_grammar_ebnf();
        let grammar = parse_ebnf(&ebnf, "root")
            .map_err(|e| CompileError::Schema(e.to_string()))?;
        Ok(self.get_or_compute(CacheKey::BuiltinJson, || self.compile_normalized(grammar)))
    }

    /// Compile a JSON schema. Port of
    /// `GrammarCompiler::CompileJSONSchema`.
    #[allow(clippy::too_many_arguments)]
    pub fn compile_json_schema(
        &self,
        schema: &str,
        any_whitespace: bool,
        indent: Option<i32>,
        separators: Option<(String, String)>,
        strict_mode: bool,
        max_whitespace_cnt: Option<i32>,
    ) -> Result<CompiledGrammar, CompileError> {
        let options = SchemaConverterOptions {
            any_whitespace,
            indent,
            separators: separators.clone(),
            strict_mode,
            max_whitespace_cnt,
            ..SchemaConverterOptions::default()
        };
        let ebnf =
            json_schema_to_ebnf(schema, &options).map_err(|e| CompileError::Schema(e.to_string()))?;
        let grammar = parse_ebnf(&ebnf, "root")
            .map_err(|e| CompileError::Schema(format!("generated EBNF failed to parse: {e}")))?;
        let key = CacheKey::Schema {
            schema: schema.to_string(),
            any_whitespace,
            indent,
            separators,
            strict_mode,
            max_whitespace_cnt,
        };
        Ok(self.get_or_compute(key, || self.compile_normalized(grammar)))
    }

    /// Compile a structural tag. Port of
    /// `GrammarCompiler::CompileStructuralTag`.
    ///
    /// Delegates the structural-tag-JSON -> `GrammarData` conversion to
    /// the W5 `structural_tag` module (`structural_tag_to_grammar`).
    pub fn compile_structural_tag(
        &self,
        structural_tag_json: &str,
    ) -> Result<CompiledGrammar, CompileError> {
        let grammar = structural_tag_to_grammar(structural_tag_json)
            .map_err(|e| CompileError::StructuralTag(e.to_string()))?;
        let key = CacheKey::StructuralTag(structural_tag_json.to_string());
        Ok(self.get_or_compute(key, || self.compile_normalized(grammar)))
    }

    /// Clear the internal compiled-grammar cache. Port of
    /// `GrammarCompiler::ClearCache`.
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Approximate bytes held by the cache. Port of
    /// `GrammarCompiler::GetCacheSizeBytes`.
    pub fn cache_size_bytes(&self) -> i64 {
        self.cache
            .iter()
            .map(|e| e.value().memory_size_bytes() as i64)
            .sum()
    }

    /// The configured cache memory limit; `-1` means unlimited. Port of
    /// `GrammarCompiler::CacheLimitBytes`.
    pub fn cache_limit_bytes(&self) -> i64 {
        self.cache_limit_bytes
    }

    /// The tokenizer this compiler is bound to.
    pub fn tokenizer_info(&self) -> &TokenizerInfo {
        &self.tokenizer_info
    }
}
