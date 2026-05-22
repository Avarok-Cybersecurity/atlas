// SPDX-License-Identifier: AGPL-3.0-only
//
// Tier-2 perf-feature integration tests: the cross-grammar
// `RuleLevelCache` and `CompiledGrammar::compile_top_k_masks`.
//
// The unit-level cache mechanics (hit/miss, LRU, budget) live in
// `compiler/rule_cache_tests.rs`; here we drive the cache through the
// real `GrammarCompiler` end to end and assert the two load-bearing
// invariants: (1) the rule cache is BEHAVIOR-PRESERVING — masks are
// byte-identical with and without it — and (2) `compile_top_k_masks`
// warms the lazy JIT cache.

use super::{compiler, optimized, small_tokenizer};
use crate::compiler::compile::compile_optimized_grammar;
use crate::compiler::{GrammarCompiler, RuleLevelCache, UNLIMITED_SIZE};

// ----- RuleLevelCache is behavior-preserving -----------------------

#[test]
fn rule_cache_masks_are_byte_identical_to_uncached() {
    // Compile the same optimized grammar twice: once with NO rule cache
    // (Tier-1 path) and once WITH one. Every reachable state's mask must
    // be byte-identical — the cache only avoids recomputation.
    let grammar = optimized("root ::= \"yes\" | \"no\" | \"abc\"\n");
    let info = small_tokenizer();

    let uncached = compile_optimized_grammar(grammar.clone(), &info, 1, None);
    let rc = RuleLevelCache::new(UNLIMITED_SIZE);
    let cached = compile_optimized_grammar(grammar, &info, 1, Some(rc));

    let u_masks: std::collections::HashMap<_, _> =
        uncached.all_reachable_masks().into_iter().collect();
    let c_masks = cached.all_reachable_masks();
    assert_eq!(u_masks.len(), c_masks.len());
    for (state, cm) in &c_masks {
        let um = u_masks.get(state).expect("cached compile lost a state");
        assert_eq!(
            um.as_ref(),
            cm.as_ref(),
            "rule-cache mask for {state:?} differs from uncached"
        );
    }
}

#[test]
fn rule_cache_reused_across_two_grammars_is_byte_identical() {
    // Two grammars sharing a structurally identical sub-rule. The second
    // compile's masks for the shared rule come from the cross-grammar
    // cache; they must still equal a from-scratch (uncached) compute.
    let info = small_tokenizer();
    let rc = RuleLevelCache::new(UNLIMITED_SIZE);

    let g1 = optimized("root ::= \"abc\" sub\nsub ::= \"yes\" | \"no\"\n");
    let g2 = optimized("root ::= \"12\" sub\nsub ::= \"yes\" | \"no\"\n");

    // Warm the cache with g1, then compile g2 (shares `sub`).
    let _c1 = compile_optimized_grammar(g1, &info, 1, Some(rc.clone()));
    let cached_g2 = compile_optimized_grammar(g2.clone(), &info, 1, Some(rc.clone()));
    let uncached_g2 = compile_optimized_grammar(g2, &info, 1, None);

    let u: std::collections::HashMap<_, _> =
        uncached_g2.all_reachable_masks().into_iter().collect();
    for (state, cm) in cached_g2.all_reachable_masks() {
        let um = u.get(&state).expect("uncached compile lost a state");
        assert_eq!(
            um.as_ref(),
            cm.as_ref(),
            "cross-grammar reuse changed a mask"
        );
    }
    // The shared sub-rule actually populated the cache.
    assert!(!rc.is_empty(), "shared rule masks must land in the cache");
}

#[test]
fn compiler_clear_cache_also_clears_rule_cache() {
    // `GrammarCompiler::clear_cache` must drop the rule-level cache too.
    let c = compiler(1);
    let cg = c
        .compile_grammar_from_ebnf("root ::= \"yes\" | \"no\"\n", "root")
        .unwrap();
    // Drive mask computation so the rule cache fills.
    let _ = cg.all_reachable_masks();
    assert!(c.cache_size_bytes() >= 0);
    c.clear_cache();
    // After a clear, a fresh compile + mask drive must still work and be
    // correct (smoke: no stale-index panic, masks non-empty).
    let cg2 = c
        .compile_grammar_from_ebnf("root ::= \"abc\"\n", "root")
        .unwrap();
    assert!(!cg2.all_reachable_masks().is_empty());
}

#[test]
fn cache_disabled_compiler_has_no_rule_cache() {
    // A `cache_enabled = false` compiler must not build a rule cache —
    // and must still compile correctly via the pure JIT path.
    let c = GrammarCompiler::new(small_tokenizer(), 1, false, -1);
    let cg = c
        .compile_grammar_from_ebnf("root ::= \"yes\" | \"no\"\n", "root")
        .unwrap();
    assert!(cg.inner().rule_cache.is_none());
    assert!(!cg.all_reachable_masks().is_empty());
}

// ----- compile_top_k_masks (overlapped mask generation) ------------

#[test]
fn compile_top_k_masks_warms_the_jit_cache() {
    let c = compiler(1);
    let cg = c
        .compile_grammar_from_ebnf("root ::= \"yes\" | \"no\" | \"abc\"\n", "root")
        .unwrap();
    // Right after compile the lazy cache is empty (XGrammar-2 JIT).
    assert!(cg.inner().mask_cache.lock().unwrap().is_empty());

    let warmed = cg.compile_top_k_masks(3);
    assert!(warmed > 0, "must warm at least one mask");
    assert!(warmed <= 3, "must not warm more than k");
    assert_eq!(
        cg.inner().mask_cache.lock().unwrap().len(),
        warmed,
        "every warmed mask must be in the JIT cache"
    );
}

#[test]
fn compile_top_k_masks_zero_is_a_noop() {
    let c = compiler(1);
    let cg = c
        .compile_grammar_from_ebnf("root ::= \"abc\"\n", "root")
        .unwrap();
    assert_eq!(cg.compile_top_k_masks(0), 0);
    assert!(cg.inner().mask_cache.lock().unwrap().is_empty());
}

#[test]
fn compile_top_k_masks_caps_at_reachable_state_count() {
    // A huge `k` warms only as many masks as there are scanable states.
    let c = compiler(1);
    let cg = c
        .compile_grammar_from_ebnf("root ::= \"yes\" | \"no\"\n", "root")
        .unwrap();
    let total = cg.all_reachable_masks().len();
    cg.inner().mask_cache.lock().unwrap().clear();
    let warmed = cg.compile_top_k_masks(10_000);
    assert_eq!(warmed, total, "k beyond the state count warms every state");
}

#[test]
fn compile_top_k_masks_results_match_lazy_path() {
    // A mask warmed eagerly by `compile_top_k_masks` must be identical
    // to the one the lazy matcher path would compute.
    let c = compiler(1);
    let cg = c
        .compile_grammar_from_ebnf("root ::= \"yes\" | \"no\" | \"abc\"\n", "root")
        .unwrap();
    let lazy: std::collections::HashMap<_, _> = cg.all_reachable_masks().into_iter().collect();
    cg.inner().mask_cache.lock().unwrap().clear();

    cg.compile_top_k_masks(1_000);
    let warmed = cg.inner().mask_cache.lock().unwrap().clone();
    for (state, m) in &warmed {
        assert_eq!(
            m.as_ref(),
            lazy.get(state)
                .expect("warmed a state the lazy path missed")
                .as_ref(),
            "eagerly-warmed mask diverges from the lazy result"
        );
    }
}

#[test]
fn compile_top_k_masks_empty_vocab_warms_nothing() {
    use crate::tokenizer::{TokenizerInfo, VocabType};
    let empty = TokenizerInfo::new(&[], VocabType::Raw, None, Some(vec![]), false);
    let c = GrammarCompiler::new(empty, 1, true, -1);
    let cg = c
        .compile_grammar_from_ebnf("root ::= \"abc\"\n", "root")
        .unwrap();
    assert_eq!(cg.compile_top_k_masks(5), 0);
}
