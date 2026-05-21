// SPDX-License-Identifier: AGPL-3.0-only
//
// Compiler entry-point, cache and multi-thread-determinism tests.

use super::{compiler, optimized, optimized_builtin_json, small_tokenizer};
use crate::compiler::{compile::compile_optimized_grammar, CompileError, GrammarCompiler};

// ----- compile an EBNF grammar -------------------------------------

#[test]
fn compile_ebnf_grammar_succeeds() {
    let c = compiler(1);
    let cg = c
        .compile_grammar_from_ebnf("root ::= \"abc\"\n", "root")
        .expect("compile");
    assert!(cg.grammar().optimized);
    assert_eq!(cg.tokenizer_info().vocab_size(), 20);
    assert!(!cg.adaptive_token_mask().is_empty());
}

#[test]
fn compile_invalid_ebnf_is_typed_error() {
    let c = compiler(1);
    let err = c
        .compile_grammar_from_ebnf("root ::= ::: bad", "root")
        .unwrap_err();
    assert!(matches!(err, CompileError::Grammar(_)));
}

#[test]
fn compile_grammar_data_directly() {
    let c = compiler(1);
    let grammar = crate::grammar::parse_ebnf_default("root ::= \"yes\" | \"no\"\n").unwrap();
    let cg = c.compile_grammar(grammar);
    assert!(cg.grammar().optimized);
}

// ----- builtin JSON grammar ----------------------------------------

#[test]
fn compile_builtin_json_grammar_succeeds() {
    let c = compiler(1);
    let cg = c.compile_builtin_json_grammar().expect("builtin json");
    assert!(cg.grammar().optimized);
    assert!(cg.memory_size_bytes() > 0);
}

// ----- JSON schema --------------------------------------------------

#[test]
fn compile_json_schema_succeeds() {
    let c = compiler(1);
    let schema = r#"{"type":"object","properties":{"x":{"type":"integer"}}}"#;
    let cg = c
        .compile_json_schema(schema, true, None, None, true, None)
        .expect("schema");
    assert!(cg.grammar().optimized);
}

#[test]
fn compile_invalid_json_schema_is_typed_error() {
    let c = compiler(1);
    let err = c
        .compile_json_schema("{not json", true, None, None, true, None)
        .unwrap_err();
    assert!(matches!(err, CompileError::Schema(_)));
}

// ----- cache hits / misses -----------------------------------------

#[test]
fn cache_hit_returns_same_compiled_grammar() {
    let c = compiler(1);
    let a = c.compile_builtin_json_grammar().unwrap();
    let b = c.compile_builtin_json_grammar().unwrap();
    assert!(std::sync::Arc::ptr_eq(a.inner(), b.inner()));
}

#[test]
fn cache_miss_on_different_grammar() {
    let c = compiler(1);
    let a = c.compile_grammar_from_ebnf("root ::= \"a\"\n", "root").unwrap();
    let b = c.compile_grammar_from_ebnf("root ::= \"b\"\n", "root").unwrap();
    assert!(!std::sync::Arc::ptr_eq(a.inner(), b.inner()));
}

#[test]
fn clear_cache_forces_recompile() {
    let c = compiler(1);
    let a = c.compile_builtin_json_grammar().unwrap();
    c.clear_cache();
    let b = c.compile_builtin_json_grammar().unwrap();
    assert!(!std::sync::Arc::ptr_eq(a.inner(), b.inner()));
}

#[test]
fn cache_disabled_never_shares() {
    let c = GrammarCompiler::new(small_tokenizer(), 1, false, -1);
    let a = c.compile_builtin_json_grammar().unwrap();
    let b = c.compile_builtin_json_grammar().unwrap();
    assert!(!std::sync::Arc::ptr_eq(a.inner(), b.inner()));
}

#[test]
fn cache_limit_bytes_reported() {
    let unlimited = GrammarCompiler::new(small_tokenizer(), 1, true, -1);
    assert_eq!(unlimited.cache_limit_bytes(), -1);
    let limited = GrammarCompiler::new(small_tokenizer(), 1, true, 1_000_000);
    assert_eq!(limited.cache_limit_bytes(), 1_000_000);
}

#[test]
fn cache_size_grows_after_compile() {
    let c = compiler(1);
    assert_eq!(c.cache_size_bytes(), 0);
    c.compile_builtin_json_grammar().unwrap();
    assert!(c.cache_size_bytes() > 0);
}

// ----- multi-threaded compilation determinism ----------------------

#[test]
fn multithread_compilation_is_deterministic() {
    // One optimized grammar, compiled by both the single-threaded and
    // the rayon mask-computation path — masks must be identical.
    let grammar = optimized("root ::= \"yes\" | \"no\" | \"abc\"\n");
    let info = small_tokenizer();
    let seq = compile_optimized_grammar(grammar.clone(), &info, 1);
    let par = compile_optimized_grammar(grammar, &info, 8);

    assert_eq!(
        seq.adaptive_token_mask().len(),
        par.adaptive_token_mask().len()
    );
    for (state, m1) in seq.adaptive_token_mask() {
        let m2 = par
            .mask_for_state(state)
            .expect("rayon compile missing a state");
        assert_eq!(m1, m2, "mask for {state:?} differs across thread counts");
    }
}

#[test]
fn multithread_builtin_json_matches_single_thread() {
    let grammar = optimized_builtin_json();
    let info = small_tokenizer();
    let seq = compile_optimized_grammar(grammar.clone(), &info, 1);
    let par = compile_optimized_grammar(grammar, &info, 4);
    assert_eq!(
        seq.adaptive_token_mask().len(),
        par.adaptive_token_mask().len()
    );
    for (state, m1) in seq.adaptive_token_mask() {
        assert_eq!(Some(m1), par.mask_for_state(state));
    }
}

// ----- degenerate empty vocabulary ---------------------------------

#[test]
fn empty_vocab_compiles_with_no_masks() {
    let info = crate::tokenizer::TokenizerInfo::new(
        &[],
        crate::tokenizer::VocabType::Raw,
        None,
        Some(vec![]),
        false,
    );
    let c = GrammarCompiler::new(info, 1, false, -1);
    let cg = c.compile_grammar_from_ebnf("root ::= \"a\"\n", "root").unwrap();
    assert!(cg.adaptive_token_mask().is_empty());
}

// ----- structural tag ----------------------------------------------

#[test]
fn compile_structural_tag_succeeds() {
    let c = compiler(1);
    let doc = r#"{"type":"structural_tag","format":{"type":"const_string","value":"abc"}}"#;
    let cg = c.compile_structural_tag(doc).expect("structural tag");
    assert!(cg.grammar().optimized);
    assert!(!cg.adaptive_token_mask().is_empty());
}

#[test]
fn compile_structural_tag_is_cached() {
    let c = compiler(1);
    let doc = r#"{"type":"structural_tag","format":{"type":"const_string","value":"yes"}}"#;
    let a = c.compile_structural_tag(doc).unwrap();
    let b = c.compile_structural_tag(doc).unwrap();
    assert!(std::sync::Arc::ptr_eq(a.inner(), b.inner()));
}

#[test]
fn compile_invalid_structural_tag_is_typed_error() {
    let c = compiler(1);
    let err = c.compile_structural_tag("{not json").unwrap_err();
    assert!(matches!(err, CompileError::StructuralTag(_)));
}
