// SPDX-License-Identifier: AGPL-3.0-only
//
// No-cache compilation core — port of `class GrammarCompilerSub`
// (`MultiThreadCompileGrammar` + `TagDispatchOptimization`) from
// `cpp/grammar_compiler.cc`.

use std::collections::HashMap;
use std::sync::Arc;

use rayon::prelude::*;

use crate::earley::{NO_PREV_INPUT_POS, ParserState};
use crate::grammar::functor::GrammarFsmHasher;
use crate::grammar::{GrammarData, GrammarExprType};
use crate::tokenizer::TokenizerInfo;

use super::compiled_grammar::{CompiledGrammar, CompiledGrammarImpl};
use super::mask::AdaptiveTokenMask;
use super::mask_gen::MaskGenerator;

/// Compile an already-optimized grammar against `tokenizer_info`,
/// precomputing every reachable scanable state's adaptive token mask.
///
/// Port of `GrammarCompilerSub::MultiThreadCompileGrammar`. The grammar
/// passed in MUST already be optimized (FSMs built); the public
/// `GrammarCompiler` entry points guarantee this.
///
/// `max_threads > 1` parallelizes the per-state mask computation via
/// rayon — the idiomatic Rust replacement for the C++ `ThreadPool`.
pub(super) fn compile_optimized_grammar(
    grammar: GrammarData,
    tokenizer_info: &TokenizerInfo,
    max_threads: usize,
) -> CompiledGrammar {
    let mut grammar = grammar;
    debug_assert!(
        grammar.optimized,
        "grammar must be optimized before compile"
    );

    // Degenerate path: an empty vocabulary has no masks to compute.
    if tokenizer_info.vocab_size() == 0 {
        return CompiledGrammar::from_impl(Arc::new(CompiledGrammarImpl {
            grammar,
            tokenizer_info: tokenizer_info.clone(),
            adaptive_token_mask: HashMap::new(),
        }));
    }

    // Step 1. TagDispatch second-slice precomputation.
    let tag_slice = tag_dispatch_optimization(&grammar, tokenizer_info);

    // Step 2. Hash the per-rule FSMs (the C++ does this when the
    // rule-level cache is enabled; we always run it — the hashes are
    // harmless and let the matcher reuse them later).
    GrammarFsmHasher::apply(&mut grammar);

    let grammar_arc = Arc::new(grammar);

    // Step 3. Enumerate every reachable scanable state of every rule.
    let root_rule_id = grammar_arc.root_rule_id();
    let mut tasks: Vec<(ParserState, bool)> = Vec::new();
    for rule_id in 0..grammar_arc.num_rules() {
        let rule = grammar_arc.rule(rule_id);
        let fsm = grammar_arc.per_rule_fsms[rule_id as usize]
            .as_ref()
            .expect("optimized grammar must have a per-rule FSM");
        let mut reachable = ahash::AHashSet::new();
        fsm.reachable_states(&mut reachable);
        let is_root = rule_id == root_rule_id;
        for state_id in reachable {
            // A state is "scanable" iff it has an outgoing char-range
            // edge (port of the FSM `IsScanableState` predicate — the
            // `earley::fsm_view` module is private, so re-derived here).
            let scanable = fsm
                .fsm()
                .edges(state_id as usize)
                .iter()
                .any(|e| e.is_char_range());
            if !scanable {
                continue;
            }
            let state =
                ParserState::new(rule_id, rule.body_expr_id, state_id, NO_PREV_INPUT_POS, 0);
            tasks.push((state, is_root));
        }
    }

    // Step 4. Compute each state's adaptive token mask, in parallel
    // when `max_threads > 1`.
    let compute = |&(state, is_root): &(ParserState, bool)| {
        let mut generator =
            MaskGenerator::new(Arc::clone(&grammar_arc), state, tokenizer_info, &tag_slice);
        (state, generator.get_adaptive_token_mask(is_root))
    };

    let entries: Vec<(ParserState, AdaptiveTokenMask)> = if max_threads > 1 {
        tasks.par_iter().map(compute).collect()
    } else {
        tasks.iter().map(compute).collect()
    };

    let adaptive_token_mask: HashMap<ParserState, AdaptiveTokenMask> =
        entries.into_iter().collect();

    let grammar = Arc::try_unwrap(grammar_arc).unwrap_or_else(|arc| (*arc).clone());

    CompiledGrammar::from_impl(Arc::new(CompiledGrammarImpl {
        grammar,
        tokenizer_info: tokenizer_info.clone(),
        adaptive_token_mask,
    }))
}

/// Precompute, for each TagDispatch rule, the bitset of tokens that are
/// definitely accepted *from their second character on* (i.e. contain
/// no tag / stop / excluded substring after the first byte).
///
/// Port of `GrammarCompilerSub::TagDispatchOptimization`. The returned
/// map is keyed by rule id; each value is a `vocab`-length bool slice
/// indexed by sorted-vocab index.
fn tag_dispatch_optimization(
    grammar: &GrammarData,
    tokenizer_info: &TokenizerInfo,
) -> HashMap<i32, Vec<bool>> {
    let mut result = HashMap::new();
    let sorted = tokenizer_info.sorted_decoded_vocab();

    for rule_id in 0..grammar.num_rules() {
        let body_id = grammar.rule(rule_id).body_expr_id;
        if grammar.expr(body_id).kind != GrammarExprType::TagDispatch {
            continue;
        }
        let td = grammar.tag_dispatch(body_id);
        let mut bitset = vec![false; sorted.len()];
        for (i, (_, token)) in sorted.iter().enumerate() {
            if token.is_empty() {
                bitset[i] = true;
                continue;
            }
            // Look for a forbidden substring starting at index >= 1.
            let forbidden = td
                .tag_rule_pairs
                .iter()
                .map(|(t, _)| t.as_bytes())
                .chain(td.stop_str.iter().map(|s| s.as_bytes()))
                .chain(td.excluded_str.iter().map(|s| s.as_bytes()))
                .any(|needle| contains_from(token, needle, 1));
            bitset[i] = !forbidden;
        }
        result.insert(rule_id, bitset);
    }
    result
}

/// True if `needle` occurs in `haystack` starting at any index >=
/// `from`. Mirrors C++ `std::string::find(needle, from)`.
fn contains_from(haystack: &[u8], needle: &[u8], from: usize) -> bool {
    if needle.is_empty() {
        return from <= haystack.len();
    }
    if from >= haystack.len() || needle.len() > haystack.len() - from {
        return false;
    }
    haystack[from..].windows(needle.len()).any(|w| w == needle)
}
