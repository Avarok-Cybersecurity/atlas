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

use std::collections::HashMap;
use std::sync::Arc;

use crate::grammar::functor::GrammarFsmHasher;
use crate::grammar::{GrammarData, GrammarExprType};
use crate::tokenizer::TokenizerInfo;

use super::compiled_grammar::{CompiledGrammar, CompiledGrammarImpl};

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
pub(super) fn compile_optimized_grammar(
    grammar: GrammarData,
    tokenizer_info: &TokenizerInfo,
    _max_threads: usize,
) -> CompiledGrammar {
    let mut grammar = grammar;
    debug_assert!(
        grammar.optimized,
        "grammar must be optimized before compile"
    );

    // Degenerate path: an empty vocabulary has no masks to compute.
    if tokenizer_info.vocab_size() == 0 {
        return CompiledGrammar::from_impl(Arc::new(CompiledGrammarImpl {
            grammar: Arc::new(grammar),
            tokenizer_info: tokenizer_info.clone(),
            mask_cache: dashmap::DashMap::new(),
            tag_slice: HashMap::new(),
        }));
    }

    // Step 1. TagDispatch second-slice precomputation. Retained on the
    // `CompiledGrammarImpl` — the lazy mask computation feeds it to the
    // `MaskGenerator` on demand.
    let tag_slice = tag_dispatch_optimization(&grammar, tokenizer_info);

    // Step 2. Hash the per-rule FSMs (the C++ does this when the
    // rule-level cache is enabled; we always run it — the hashes are
    // harmless and let the matcher reuse them later).
    GrammarFsmHasher::apply(&mut grammar);

    // Steps 3-4 (enumerate reachable scanable states + compute every
    // state's `AdaptiveTokenMask`) are DELETED — see the module docs.
    // The mask cache starts empty and is populated lazily.
    CompiledGrammar::from_impl(Arc::new(CompiledGrammarImpl {
        grammar: Arc::new(grammar),
        tokenizer_info: tokenizer_info.clone(),
        mask_cache: dashmap::DashMap::new(),
        tag_slice,
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
