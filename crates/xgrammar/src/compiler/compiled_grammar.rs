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
use crate::grammar::functor::hash_sequence;
use crate::support::hash::hash_combine;
use crate::tokenizer::TokenizerInfo;

use super::decompose::GrammarDecomposition;
use super::mask::AdaptiveTokenMask;
use super::mask_gen::MaskGenerator;
use super::rule_cache::{RuleLevelCache, RuleMaskKey};

/// The inner, shared state of a [`CompiledGrammar`]. Port of
/// `CompiledGrammar::Impl`.
#[derive(Debug)]
pub struct CompiledGrammarImpl {
    /// The optimized, FSM-accelerated grammar (shared — the lazy
    /// `MaskGenerator` needs an `Arc<GrammarData>`).
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
    /// computation can feed it to the `MaskGenerator`. `Arc`-wrapped
    /// so `MaskGenerator::new` clones a pointer, not the map.
    pub tag_slice: Arc<AHashMap<i32, Vec<bool>>>,
    /// The cross-grammar [`RuleLevelCache`], shared from the owning
    /// [`super::GrammarCompiler`] (`None` when the compiler's cache is
    /// disabled, or for the `vocab_size == 0` degenerate path). The lazy
    /// mask path consults it on a per-state cache miss before falling
    /// back to a full `MaskGenerator` recomputation — a rule
    /// structurally identical to one seen in any previous request
    /// reuses its computed masks.
    pub rule_cache: Option<RuleLevelCache>,
    /// WGRAMMAR static/dynamic decomposition (Tier 3c), computed ONCE
    /// at compile time by [`super::decompose::decompose_static_regions`].
    /// Classifies every rule as fixed scaffolding (with its literal
    /// bytes precomputed) or a dynamic value slot — the compile-time
    /// index of the grammar's static structure. See the module docs of
    /// `decompose.rs` for how this composes with Tiers 2 and 3b.
    pub decomposition: GrammarDecomposition,
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

    /// Build the cross-grammar [`RuleMaskKey`] for a canonical parser
    /// state, returning it together with the shared [`RuleLevelCache`].
    ///
    /// `None` — the rule-level cache is unavailable for this state — when
    /// any of the following hold (faithful to the upstream
    /// `rule_level_cache_is_available` guard in `grammar_compiler.cc`):
    ///   * the compiler has no `rule_cache`;
    ///   * the rule has no structural FSM hash (the hasher could not
    ///     hash it — e.g. an unbroken complex cycle);
    ///   * the canonical FSM node id has no entry in the rule's
    ///     `per_rule_fsm_new_state_ids` renumbering.
    ///
    /// KEY DERIVATION. The structural part is `(fsm_hash,
    /// fsm_new_node_id, state_cnt, edge_cnt)` — exactly the upstream
    /// `RuleLevelCache::Impl::NodeKey`. Upstream caches the
    /// *no-lookahead* mask under the bare `fsm_hash` and the
    /// lookahead-aware mask under `HashCombine(fsm_hash, lookahead_hash,
    /// is_exact_lookahead)`. This port's `MaskGenerator` yields the
    /// final lookahead-applied mask in one shot, so the key folds in the
    /// lookahead hash (and `is_exact_lookahead`) whenever the rule has a
    /// lookahead assertion, and folds in `is_root` — guaranteeing a
    /// cache hit is byte-identical to a fresh recompute. A plain
    /// no-lookahead non-root rule therefore keys on the bare structural
    /// hash and reuses freely across grammars.
    fn rule_mask_key(
        &self,
        canonical: &ParserState,
        is_root: bool,
    ) -> Option<(&RuleLevelCache, RuleMaskKey)> {
        let cache = self.rule_cache.as_ref()?;
        let rule_id = canonical.rule_id;
        let base_hash = (*self.grammar.per_rule_fsm_hashes.get(rule_id as usize)?)?;

        // Map the FSM node id (`element_id`) to its canonical renumbered
        // id via `per_rule_fsm_new_state_ids` (a list of original->new
        // pairs, exactly as the C++ `original_to_new_id` scan).
        let renumber = self
            .grammar
            .per_rule_fsm_new_state_ids
            .get(rule_id as usize)?;
        let fsm_new_node_id = renumber
            .iter()
            .find(|(orig, _)| *orig == canonical.element_id)
            .map(|(_, new)| *new)?;

        let fsm = self.grammar.per_rule_fsms[rule_id as usize]
            .as_ref()
            .expect("optimized grammar must have a per-rule FSM");
        let state_cnt = fsm.num_states() as i32;
        let edge_cnt = fsm.num_edges() as i32;

        // Fold lookahead + root into the hash so a hit is byte-exact.
        let rule = self.grammar.rule(rule_id);
        let mut fsm_hash = base_hash;
        if rule.lookahead_assertion_id != -1 {
            // `hash_sequence` may return `None` for a non-hashable
            // lookahead sequence; treat that as "cache unavailable" so
            // we never key two semantically different masks alike.
            let la_hash = hash_sequence(&self.grammar, rule.lookahead_assertion_id)?;
            fsm_hash = hash_combine(&[fsm_hash, la_hash, rule.is_exact_lookahead as u64]);
        }
        if is_root {
            fsm_hash = hash_combine(&[fsm_hash, u64::MAX]);
        }

        Some((
            cache,
            RuleMaskKey {
                fsm_hash,
                fsm_new_node_id,
                state_cnt,
                edge_cnt,
            },
        ))
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
    /// lazily-constructed `MaskGenerator`.
    pub fn grammar_arc(&self) -> Arc<GrammarData> {
        Arc::clone(&self.pimpl.grammar)
    }

    /// The associated tokenizer info.
    pub fn tokenizer_info(&self) -> &TokenizerInfo {
        &self.pimpl.tokenizer_info
    }

    /// The WGRAMMAR static/dynamic decomposition (Tier 3c) — the
    /// compile-time split of every rule body into fixed scaffolding
    /// spans (with precomputed literal bytes) and dynamic value-slot
    /// spans.
    ///
    /// Computed once during compilation. A scheduler / matcher can
    /// consult it — before the first decode step — to learn the
    /// grammar's static/dynamic structure without any per-token work.
    /// Each [`Segment::Static`](super::Segment::Static) span is exactly
    /// a forced-byte run Tier 3b would otherwise rediscover lazily; the
    /// genuine WGRAMMAR delta is doing this at compile time.
    pub fn decomposition(&self) -> &GrammarDecomposition {
        &self.pimpl.decomposition
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
    /// compile produced: same `MaskGenerator`, same canonical key.
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
        // Per-grammar miss: compute the mask WITHOUT holding the lock —
        // the `MaskGenerator` scan is expensive, and serializing it
        // under the lock would defeat the parallel `BatchGrammarMatcher`
        // fill. A concurrent computer of the same state simply does
        // duplicate work; the `entry` double-check below keeps a single
        // canonical `Arc`.
        //
        // Before recomputing, consult the cross-grammar `RuleLevelCache`
        // (Tier 2): a rule structurally identical to one seen in a
        // previous request — even from a different grammar — already has
        // its masks computed. The cross-grammar key is structural
        // (FSM hash + canonical node + node/edge counts), so a hit is
        // byte-identical to a fresh `MaskGenerator` run.
        let computed = match self.pimpl.rule_mask_key(&canonical, is_root) {
            Some((cache, key)) => {
                cache.get_or_compute(key, || Self::compute_mask(&self.pimpl, canonical, is_root))
            }
            None => Arc::new(Self::compute_mask(&self.pimpl, canonical, is_root)),
        };
        let mut cache = self
            .pimpl
            .mask_cache
            .lock()
            .expect("mask_cache mutex poisoned");
        Arc::clone(cache.entry(canonical).or_insert(computed))
    }

    /// Run a full `MaskGenerator` scan for `canonical` — the
    /// authoritative (uncached) mask computation. `&Arc<AHashMap>`
    /// deref-coerces to the `&AHashMap` argument.
    fn compute_mask(
        pimpl: &Arc<CompiledGrammarImpl>,
        canonical: ParserState,
        is_root: bool,
    ) -> AdaptiveTokenMask {
        let mut generator = MaskGenerator::new(
            Arc::clone(&pimpl.grammar),
            canonical,
            &pimpl.tokenizer_info,
            &pimpl.tag_slice,
        );
        generator.get_adaptive_token_mask(is_root)
    }

    /// Approximate memory usage in bytes. Port of
    /// `CompiledGrammar::MemorySizeBytes`.
    pub fn memory_size_bytes(&self) -> usize {
        self.pimpl.memory_size()
    }

    /// OVERLAPPED MASK GENERATION (Tier 2, "configurable JIT").
    ///
    /// Eagerly compute the `k` most-expensive adaptive-token masks,
    /// populating the lazy JIT cache so they are already warm before
    /// the matcher needs them. Atlas's scheduler calls this during
    /// prefill — while the GPU is busy with the prompt — so the first
    /// decode steps never pay a cold mask-computation stall.
    ///
    /// COST RANKING. A state's mask cost is dominated by the breadth of
    /// its first-character scan: every distinct first byte the state can
    /// accept seeds an `EarleyParser` walk over the matching slice of
    /// the sorted vocabulary. We therefore rank reachable scanable
    /// states by their char-range edge count (a cheap, monotonic proxy
    /// for that breadth) and warm the top `k`.
    ///
    /// Returns the number of masks actually computed (`<= k`, and `<=`
    /// the number of reachable scanable states). `k == 0`, or an empty
    /// vocabulary, computes nothing. Idempotent: a state already in the
    /// cache is recomputed cheaply via `get_or_compute_mask`'s fast path
    /// and still counts toward the return value.
    pub fn compile_top_k_masks(&self, k: usize) -> usize {
        use crate::earley::NO_PREV_INPUT_POS;

        if k == 0 || self.pimpl.tokenizer_info.vocab_size() == 0 {
            return 0;
        }
        let grammar = &self.pimpl.grammar;
        let root_rule_id = grammar.root_rule_id();

        // Collect (cost, canonical_state, is_root) for every reachable
        // scanable state, where cost is the char-range edge count.
        let mut ranked: Vec<(usize, ParserState, bool)> = Vec::new();
        for rule_id in 0..grammar.num_rules() {
            let rule = grammar.rule(rule_id);
            let fsm = grammar.per_rule_fsms[rule_id as usize]
                .as_ref()
                .expect("optimized grammar must have a per-rule FSM");
            let mut reachable = ahash::AHashSet::new();
            fsm.reachable_states(&mut reachable);
            let is_root = rule_id == root_rule_id;
            for state_id in reachable {
                let char_edges = fsm
                    .fsm()
                    .edges(state_id as usize)
                    .iter()
                    .filter(|e| e.is_char_range())
                    .count();
                if char_edges == 0 {
                    continue; // not scanable — no mask to compute
                }
                let canonical =
                    ParserState::new(rule_id, rule.body_expr_id, state_id, NO_PREV_INPUT_POS, 0);
                ranked.push((char_edges, canonical, is_root));
            }
        }

        // Most expensive first. `select_nth_unstable` would suffice, but
        // a full sort keeps the warm order deterministic for tests and
        // the list is small (one entry per scanable FSM state).
        ranked.sort_unstable_by_key(|entry| std::cmp::Reverse(entry.0));
        let take = k.min(ranked.len());
        for &(_, canonical, is_root) in &ranked[..take] {
            self.get_or_compute_mask(canonical, is_root);
        }
        take
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
