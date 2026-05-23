// SPDX-License-Identifier: AGPL-3.0-only
//
// Compiler entry-point, cache and multi-thread-determinism tests.

use super::{compiler, optimized, optimized_builtin_json, small_tokenizer};
use crate::compiler::{CompileError, GrammarCompiler, compile::compile_optimized_grammar};

// ----- compile an EBNF grammar -------------------------------------

#[test]
fn compile_ebnf_grammar_succeeds() {
    let c = compiler(1);
    let cg = c
        .compile_grammar_from_ebnf("root ::= \"abc\"\n", "root")
        .expect("compile");
    assert!(cg.grammar().optimized);
    assert_eq!(cg.tokenizer_info().vocab_size(), 20);
    // Lazy path: masks are JIT-compiled on demand — driving the
    // enumeration produces a non-empty set.
    assert!(!cg.all_reachable_masks().is_empty());
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
    let a = c
        .compile_grammar_from_ebnf("root ::= \"a\"\n", "root")
        .unwrap();
    let b = c
        .compile_grammar_from_ebnf("root ::= \"b\"\n", "root")
        .unwrap();
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
fn jit_mask_compilation_is_deterministic() {
    // `max_threads` is now unused (no eager parallel mask loop). Two
    // independent compiles of the same grammar must still produce
    // byte-identical lazily-computed masks for every reachable state.
    let grammar = optimized("root ::= \"yes\" | \"no\" | \"abc\"\n");
    let info = small_tokenizer();
    let a = compile_optimized_grammar(grammar.clone(), &info, 1, None);
    let b = compile_optimized_grammar(grammar, &info, 8, None);

    let a_masks = a.all_reachable_masks();
    let b_masks: std::collections::HashMap<_, _> = b.all_reachable_masks().into_iter().collect();
    assert_eq!(a_masks.len(), b_masks.len());
    for (state, m1) in &a_masks {
        let m2 = b_masks.get(state).expect("second compile missing a state");
        assert_eq!(
            m1.as_ref(),
            m2.as_ref(),
            "JIT mask for {state:?} differs across compiles"
        );
    }
}

#[test]
fn jit_builtin_json_masks_are_stable() {
    let grammar = optimized_builtin_json();
    let info = small_tokenizer();
    let a = compile_optimized_grammar(grammar.clone(), &info, 1, None);
    let b = compile_optimized_grammar(grammar, &info, 4, None);
    let a_masks = a.all_reachable_masks();
    let b_masks: std::collections::HashMap<_, _> = b.all_reachable_masks().into_iter().collect();
    assert_eq!(a_masks.len(), b_masks.len());
    for (state, m1) in &a_masks {
        assert_eq!(m1.as_ref(), b_masks.get(state).unwrap().as_ref());
    }
}

#[test]
fn mask_cache_starts_empty_and_fills_lazily() {
    // The XGrammar-2 JIT: compilation computes NO masks; they are
    // populated only when reached.
    let grammar = optimized("root ::= \"yes\" | \"no\"\n");
    let info = small_tokenizer();
    let cg = compile_optimized_grammar(grammar, &info, 1, None);
    assert!(
        cg.inner().mask_cache.lock().unwrap().is_empty(),
        "mask cache must be empty right after compilation"
    );
    let masks = cg.all_reachable_masks();
    assert!(!masks.is_empty());
    assert_eq!(
        cg.inner().mask_cache.lock().unwrap().len(),
        masks.len(),
        "every reached state must now be cached"
    );
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
    let cg = c
        .compile_grammar_from_ebnf("root ::= \"a\"\n", "root")
        .unwrap();
    assert!(cg.all_reachable_masks().is_empty());
    assert!(cg.inner().mask_cache.lock().unwrap().is_empty());
}

// ----- structural tag ----------------------------------------------

#[test]
fn compile_structural_tag_succeeds() {
    let c = compiler(1);
    let doc = r#"{"type":"structural_tag","format":{"type":"const_string","value":"abc"}}"#;
    let cg = c.compile_structural_tag(doc).expect("structural tag");
    assert!(cg.grammar().optimized);
    assert!(!cg.all_reachable_masks().is_empty());
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
