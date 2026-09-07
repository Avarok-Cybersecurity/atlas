// SPDX-License-Identifier: AGPL-3.0-only

//! Allocation oracle for the production cold-mask path, with no timing limit.

use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
use std::alloc::System;
use xgrammar::compiler::GrammarCompiler;
use xgrammar::matcher::GrammarMatcher;
use xgrammar::tokenizer::{TokenizerInfo, VocabType};

#[global_allocator]
static ALLOCATOR: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

fn cold_mask_allocations(irrelevant_tokens: usize) -> usize {
    let mut vocab: Vec<String> = (0..irrelevant_tokens)
        .map(|i| format!("z-irrelevant-token-{i:012}"))
        .collect();
    vocab.push("yes".into());
    vocab.push("<eos>".into());
    let eos = (vocab.len() - 1) as i32;
    let info = TokenizerInfo::new(&vocab, VocabType::Raw, None, Some(vec![eos]), false);
    let compiler = GrammarCompiler::new(info, 1, false, -1);
    let grammar = compiler
        .compile_grammar_from_ebnf("root ::= \"yes\"\n", "root")
        .unwrap();

    // Tokenizer construction is intentionally outside the measured region.
    // None of the added z-prefixed tokens can match any state of "yes".
    let region = Region::new(ALLOCATOR);
    let states = grammar.compile_top_k_masks(512);
    let stats = region.change();
    assert_eq!(states, 3);
    let allocations = stats.allocations + stats.reallocations;
    println!("irrelevant={irrelevant_tokens} states={states} allocations={allocations}");

    let mut matcher = GrammarMatcher::new(grammar, None, false, -1);
    assert!(!matcher.accept_string("z-irrelevant", false));
    assert!(matcher.accept_string("yes", false));
    assert!(matcher.accept_token(eos, false));
    assert!(matcher.is_terminated());
    allocations
}

#[test]
fn irrelevant_vocabulary_does_not_allocate_per_token_during_mask_preparation() {
    let small = cold_mask_allocations(4096);
    let large = cold_mask_allocations(8192);
    assert!(
        large <= small,
        "cold mask preparation allocated buffers for irrelevant vocabulary: {small} -> {large}"
    );
}
