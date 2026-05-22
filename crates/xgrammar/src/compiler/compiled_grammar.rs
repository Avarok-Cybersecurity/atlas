// SPDX-License-Identifier: AGPL-3.0-only
//
// CompiledGrammar — port of `class CompiledGrammar` / `CompiledGrammar
// ::Impl` from `cpp/compiled_grammar.cc` + `cpp/compiled_grammar_impl.h`.
//
// A grammar compiled against a specific tokenizer: it bundles the
// optimized, FSM-accelerated grammar, the tokenizer info, and a lazy
// per-parser-state adaptive-token-mask cache. The matcher (W6) consults
// `get_or_compute_mask` to fill the logit bitmask fast.
//
// XGrammar-2 JIT (lazy) MASK COMPILATION
// --------------------------------------
// The eager port precomputed an `AdaptiveTokenMask` for *every*
// reachable scanable state up front. For tool-call JSON-schema grammars
// that is hundreds of masks per `compile_*` call — most never used by a
// single generation. This port computes each state's mask LAZILY, on
// first lookup by the matcher, and caches it in `mask_cache`. Only the
// states a generation actually visits get a mask computed, once each.

use std::sync::{Arc, Mutex};

use ahash::AHashMap;

use crate::earley::ParserState;
use crate::grammar::GrammarData;
use crate::tokenizer::TokenizerInfo;

use super::mask::AdaptiveTokenMask;
use super::mask_gen::MaskGenerator;

/// The inner, shared state of a [`CompiledGrammar`]. Port of
/// `CompiledGrammar::Impl`.
#[derive(Debug)]
pub struct CompiledGrammarImpl {
    /// The optimized, FSM-accelerated grammar (shared — the lazy
    /// [`MaskGenerator`] needs an `Arc<GrammarData>`).
    pub grammar: Arc<GrammarData>,
    /// The tokenizer this grammar was compiled against.
    pub tokenizer_info: TokenizerInfo,
    /// Lazy per-parser-state adaptive-token-mask cache. Equivalent to
    /// the C++ `adaptive_token_mask_cache` — a plain hash map, populated
    /// on demand: empty after compilation, filled by [`CompiledGrammar::
    /// get_or_compute_mask`] as the matcher reaches each state.
    ///
    /// The matcher drives one `CompiledGrammar` single-threaded per
    /// request, so the C++ uses a plain `unordered_map` with no locking.
    /// We cannot drop the lock entirely, however: [`super::super::
    /// matcher::BatchGrammarMatcher`]'s rayon `par_iter_mut` fill path
    /// runs many matchers that were all cloned from one `CompiledGrammar`
    /// — they share this `Arc<CompiledGrammarImpl>` and may call
    /// `get_or_compute_mask` concurrently. A single uncontended `Mutex`
    /// lock is far cheaper than `DashMap`'s shard-hash + shard-select
    /// machinery on every per-token lookup, while still keeping
    /// `CompiledGrammarImpl: Sync`.
    pub mask_cache: Mutex<AHashMap<ParserState, Arc<AdaptiveTokenMask>>>,
    /// TagDispatch second-slice precomputation, keyed by rule id. Built
    /// once at compile time and retained here so on-demand mask
    /// computation can feed it to the [`MaskGenerator`]. `Arc`-wrapped
    /// so [`MaskGenerator::new`] clones a pointer, not the map.
    pub tag_slice: Arc<AHashMap<i32, Vec<bool>>>,
}

impl CompiledGrammarImpl {
    /// Approximate heap memory usage in bytes. Port of
    /// `MemorySize(const CompiledGrammar::Impl&)`. The mask term sums
    /// only the masks computed so far (the lazy cache).
    pub fn memory_size(&self) -> usize {
        let grammar_bytes = self.grammar.complete_fsm.memory_size()
            + self.grammar.num_exprs() as usize * 4
            + self.grammar.num_rules() as usize * 32;
        let mask_bytes: usize = self
            .mask_cache
            .lock()
            .expect("mask_cache mutex poisoned")
            .values()
            .map(|m| m.memory_size())
            .sum();
        grammar_bytes + mask_bytes
    }
}

/// A grammar compiled against a tokenizer — the result of
/// preprocessing performed by [`super::GrammarCompiler`].
///
/// Cheap to clone: the inner state is shared via [`Arc`], matching the
/// C++ pimpl `shared_ptr` semantics.
#[derive(Debug, Clone)]
pub struct CompiledGrammar {
    pimpl: Arc<CompiledGrammarImpl>,
}

impl CompiledGrammar {
    /// Wrap an already-built [`CompiledGrammarImpl`].
    pub fn from_impl(pimpl: Arc<CompiledGrammarImpl>) -> Self {
        Self { pimpl }
    }

    /// The associated optimized grammar.
    pub fn grammar(&self) -> &GrammarData {
        &self.pimpl.grammar
    }

    /// Shared-pointer access to the optimized grammar — used to seed a
    /// lazily-constructed [`MaskGenerator`].
    pub fn grammar_arc(&self) -> Arc<GrammarData> {
        Arc::clone(&self.pimpl.grammar)
    }

    /// The associated tokenizer info.
    pub fn tokenizer_info(&self) -> &TokenizerInfo {
        &self.pimpl.tokenizer_info
    }

    /// Get the adaptive token mask for a canonical parser state,
    /// computing and caching it on first access (the XGrammar-2 JIT
    /// lazy path).
    ///
    /// `canonical` MUST be the canonical cache key — `rule_start_pos =
    /// -1`, `sub_element_id`/`repeat_count`/`partial_codepoint = 0`,
    /// `sequence_id = body_expr_id` — exactly the tuple the eager port
    /// used. `is_root` is whether `canonical.rule_id` is the grammar
    /// root (the root rule has no uncertain tokens).
    ///
    /// The returned `Arc` is byte-identical to what the old eager
    /// compile produced: same [`MaskGenerator`], same canonical key.
    pub fn get_or_compute_mask(
        &self,
        canonical: ParserState,
        is_root: bool,
    ) -> Arc<AdaptiveTokenMask> {
        // Fast path: a cache hit takes one uncontended `Mutex` lock and
        // an `Arc` clone — nothing else.
        if let Some(hit) = self
            .pimpl
            .mask_cache
            .lock()
            .expect("mask_cache mutex poisoned")
            .get(&canonical)
        {
            return Arc::clone(hit);
        }
        // Miss: compute the mask WITHOUT holding the lock — the
        // `MaskGenerator` scan is expensive, and serializing it under
        // the lock would defeat the parallel `BatchGrammarMatcher` fill.
        // A concurrent computer of the same state simply does duplicate
        // work; the `entry` double-check below keeps a single canonical
        // `Arc` (compute-on-miss logic is otherwise identical to before).
        // `&Arc<AHashMap>` deref-coerces to the `&AHashMap` argument.
        let mut generator = MaskGenerator::new(
            Arc::clone(&self.pimpl.grammar),
            canonical,
            &self.pimpl.tokenizer_info,
            &self.pimpl.tag_slice,
        );
        let computed = Arc::new(generator.get_adaptive_token_mask(is_root));
        let mut cache = self
            .pimpl
            .mask_cache
            .lock()
            .expect("mask_cache mutex poisoned");
        Arc::clone(cache.entry(canonical).or_insert(computed))
    }

    /// Approximate memory usage in bytes. Port of
    /// `CompiledGrammar::MemorySizeBytes`.
    pub fn memory_size_bytes(&self) -> usize {
        self.pimpl.memory_size()
    }

    /// Shared-pointer access to the inner state — used by the matcher.
    pub fn inner(&self) -> &Arc<CompiledGrammarImpl> {
        &self.pimpl
    }

    /// Enumerate every reachable scanable canonical [`ParserState`] and
    /// its (lazily-computed) [`AdaptiveTokenMask`].
    ///
    /// This is exactly the state set the *eager* compiler precomputed —
    /// reproduced here so tests can verify the JIT result equals the
    /// old eager result and that the partition invariants hold. Each
    /// returned mask is materialized through [`Self::get_or_compute_mask`],
    /// i.e. the lazy path. Test-only.
    #[cfg(test)]
    pub(crate) fn all_reachable_masks(&self) -> Vec<(ParserState, Arc<AdaptiveTokenMask>)> {
        use crate::earley::NO_PREV_INPUT_POS;

        let mut out = Vec::new();
        if self.pimpl.tokenizer_info.vocab_size() == 0 {
            return out;
        }
        let grammar = &self.pimpl.grammar;
        let root_rule_id = grammar.root_rule_id();
        for rule_id in 0..grammar.num_rules() {
            let rule = grammar.rule(rule_id);
            let fsm = grammar.per_rule_fsms[rule_id as usize]
                .as_ref()
                .expect("optimized grammar must have a per-rule FSM");
            let mut reachable = ahash::AHashSet::new();
            fsm.reachable_states(&mut reachable);
            let is_root = rule_id == root_rule_id;
            for state_id in reachable {
                let scanable = fsm
                    .fsm()
                    .edges(state_id as usize)
                    .iter()
                    .any(|e| e.is_char_range());
                if !scanable {
                    continue;
                }
                let canonical =
                    ParserState::new(rule_id, rule.body_expr_id, state_id, NO_PREV_INPUT_POS, 0);
                let mask = self.get_or_compute_mask(canonical, is_root);
                out.push((canonical, mask));
            }
        }
        out
    }
}
