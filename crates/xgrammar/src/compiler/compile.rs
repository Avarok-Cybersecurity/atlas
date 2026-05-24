// SPDX-License-Identifier: AGPL-3.0-only
//
// No-cache compilation core — port of `class GrammarCompilerSub`
// (`MultiThreadCompileGrammar` + `TagDispatchOptimization`) from
// `cpp/grammar_compiler.cc`, with the XGrammar-2 JIT optimization.
//
// XGrammar-2 JIT (lazy) MASK COMPILATION
// --------------------------------------
// The original port eagerly enumerated every reachable scanable state
// of every rule and computed an `AdaptiveTokenMask` for ALL of them up
// front (rayon-parallel). For tool-call JSON-schema grammars that is
// hundreds of masks per `compile_*` call — most never used by a single
// generation, making compilation ~1.5x slower than the C++ baseline.
//
// This port keeps Steps 1-2 (TagDispatch precomputation + FSM hashing)
// but defers per-state mask computation entirely: the `CompiledGrammar`
// is built with an empty `mask_cache`, and each state's mask is
// computed lazily — on first lookup by the matcher — via
// `CompiledGrammar::get_or_compute_mask`. The result is byte-identical
// to the old eager output (same `MaskGenerator`, same canonical key).

use std::sync::{Arc, Mutex};

use ahash::AHashMap;

use crate::grammar::functor::GrammarFsmHasher;
use crate::grammar::{GrammarData, GrammarExprType};
use crate::tokenizer::TokenizerInfo;

use super::compiled_grammar::{CompiledGrammar, CompiledGrammarImpl};
use super::decompose::decompose_static_regions;
use super::rule_cache::RuleLevelCache;

/// Compile an already-optimized grammar against `tokenizer_info`.
///
/// Port of `GrammarCompilerSub::MultiThreadCompileGrammar`. The grammar
/// passed in MUST already be optimized (FSMs built); the public
/// `GrammarCompiler` entry points guarantee this.
///
/// Adaptive token masks are NOT computed here — they are compiled
/// lazily on first matcher lookup (XGrammar-2 JIT). `_max_threads` is
/// retained for API parity but is now unused: there is no eager
/// per-state mask loop left to parallelize.
///
/// `rule_cache` is the optional cross-grammar [`RuleLevelCache`] (passed
/// from the [`super::GrammarCompiler`] so it is shared across every
/// grammar that compiler builds). When present, the per-rule FSM hasher
/// is run so the lazy mask computation can key into it.
pub(super) fn compile_optimized_grammar(
    mut grammar: GrammarData,
    tokenizer_info: &TokenizerInfo,
    _max_threads: usize,
    rule_cache: Option<RuleLevelCache>,
) -> CompiledGrammar {
    debug_assert!(
        grammar.optimized,
        "grammar must be optimized before compile"
    );

    // Degenerate path: an empty vocabulary has no masks to compute.
    // The WGRAMMAR decomposition is still computed — it is a grammar
    // property, independent of the tokenizer — so the static/dynamic
    // index is available even on the degenerate path.
    if tokenizer_info.vocab_size() == 0 {
        let decomposition = decompose_static_regions(&grammar);
        return CompiledGrammar::from_impl(Arc::new(CompiledGrammarImpl {
            grammar: Arc::new(grammar),
            tokenizer_info: tokenizer_info.clone(),
            mask_cache: Mutex::new(AHashMap::new()),
            tag_slice: Arc::new(AHashMap::new()),
            rule_cache: None,
            decomposition,
        }));
    }

    // Step 1. TagDispatch second-slice precomputation. Retained on the
    // `CompiledGrammarImpl` (`Arc`-wrapped) — the lazy mask computation
    // feeds it to the `MaskGenerator` on demand without copying the map.
    let tag_slice = Arc::new(tag_dispatch_optimization(&grammar, tokenizer_info));

    // Step 2. Per-rule FSM structural hashing (`GrammarFSMHasher::Apply`
    // in the C++). The hashes + canonical state renumbering populate
    // `grammar.per_rule_fsm_hashes` / `per_rule_fsm_new_state_ids`; they
    // are the lookup keys of the cross-grammar `RuleLevelCache`. Tier 1
    // removed this call as dead work because no cache consumed the
    // hashes — Tier 2 re-enables it here, gated on the cache being
    // present so a cache-disabled compiler pays nothing.
    if rule_cache.is_some() {
        GrammarFsmHasher::apply(&mut grammar);
    }

    // Step 2c. WGRAMMAR static/dynamic decomposition (Tier 3c). A
    // single linear walk of the optimized AST classifies every rule as
    // fixed scaffolding (literal bytes precomputed here, once) or a
    // dynamic value slot. The result is the compile-time index of the
    // grammar's static structure — see `decompose.rs`.
    let decomposition = decompose_static_regions(&grammar);

    // Steps 3-4 (enumerate reachable scanable states + compute every
    // state's `AdaptiveTokenMask`) remain DELETED — see the module docs.
    // The mask cache starts empty and is populated lazily; on a miss the
    // lazy path consults `rule_cache` before recomputing.
    CompiledGrammar::from_impl(Arc::new(CompiledGrammarImpl {
        grammar: Arc::new(grammar),
        tokenizer_info: tokenizer_info.clone(),
        mask_cache: Mutex::new(AHashMap::new()),
        tag_slice,
        rule_cache,
        decomposition,
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
) -> AHashMap<i32, Vec<bool>> {
    let mut result = AHashMap::new();
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
