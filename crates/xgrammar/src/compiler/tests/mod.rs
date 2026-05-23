// SPDX-License-Identifier: AGPL-3.0-only
//
// Tests for the grammar compiler — ported / adapted from
// `tests/python/test_grammar_compiler.py` and `tests/cpp/test_*` of
// xgrammar v0.1.32.
//
// The C++/Python tests load real HuggingFace tokenizers; here a small
// hand-built `TokenizerInfo` exercises the same code paths. This file
// holds the shared fixtures; the test cases live in submodules to keep
// every file under the 250-line cap:
//   compile_tests   — compile entry points, cache behaviour, determinism
//   mask_tests      — AdaptiveTokenMask partition correctness, intervals
//   tier2_tests     — cross-grammar RuleLevelCache + compile_top_k_masks
//   decompose_tests — WGRAMMAR static/dynamic decomposition (Tier 3c)

use crate::compiler::GrammarCompiler;
use crate::grammar::functor::{GrammarNormalizer, GrammarOptimizer};
use crate::grammar::{GrammarData, parse_ebnf, parse_ebnf_default};
use crate::tokenizer::{TokenizerInfo, VocabType};

mod compile_tests;
mod decompose_tests;
mod mask_tests;
mod tier2_tests;

/// A tiny RAW-vocab tokenizer covering the bytes used in the test
/// grammars: single ASCII chars plus a few multi-char tokens.
pub(super) fn small_tokenizer() -> TokenizerInfo {
    let vocab: Vec<String> = [
        "a", "b", "c", "d", "e", "y", "n", "o", "s", "{", "}", "\"", ":", " ", "ab", "abc", "yes",
        "no", "12", "true",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    TokenizerInfo::new(&vocab, VocabType::Raw, None, Some(vec![]), false)
}

/// A cache-enabled compiler bound to [`small_tokenizer`].
pub(super) fn compiler(max_threads: usize) -> GrammarCompiler {
    GrammarCompiler::new(small_tokenizer(), max_threads, true, -1)
}

/// Resolve the sorted-vocab index of a given token's bytes.
pub(super) fn idx_of(info: &TokenizerInfo, token: &[u8]) -> i32 {
    info.sorted_decoded_vocab()
        .iter()
        .position(|(_, t)| t.as_slice() == token)
        .map(|p| p as i32)
        .unwrap_or_else(|| panic!("token {token:?} not in vocab"))
}

/// Optimize an EBNF grammar to the FSM-accelerated form the no-cache
/// compiler core expects. Used by the JIT determinism tests so a
/// *single* optimized grammar (with a fixed FSM node numbering) is
/// compiled twice — comparing two independent `GrammarOptimizer` runs
/// would be invalid, since FSM construction numbers nodes
/// non-deterministically.
pub(super) fn optimized(ebnf: &str) -> GrammarData {
    let g = parse_ebnf_default(ebnf).unwrap();
    GrammarOptimizer::apply(GrammarNormalizer::apply(g))
}

/// Optimize the builtin-JSON grammar to the FSM-accelerated form.
pub(super) fn optimized_builtin_json() -> GrammarData {
    let g = parse_ebnf(&crate::schema::builtin_json_grammar_ebnf(), "root").unwrap();
    GrammarOptimizer::apply(GrammarNormalizer::apply(g))
}
