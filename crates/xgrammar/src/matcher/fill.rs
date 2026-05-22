// SPDX-License-Identifier: AGPL-3.0-only
//
// GrammarMatcher — next-token bitmask fill and jump-forward search.
//
// Port of `FillNextTokenBitmask`, `SetTokenBitmask` and
// `FindJumpForwardString` from `cpp/grammar_matcher.cc`.
//
// FILL ALGORITHM (faithful to the C++)
// ------------------------------------
// Each latest scanable Earley state has a precompiled `AdaptiveTokenMask`
// partitioning the vocabulary into accepted / rejected / uncertain.
//   * Accepted union   — a token accepted by *any* state is accepted.
//   * Rejected intersect — a token is rejected only if *every* state
//     rejects it; tracked as a running `tmp_rejected` interset that
//     starts as the universal set `{-1}`.
//   * Uncertain tokens — resolved by trial: a one-state probe is pushed
//     onto the parser, the token's bytes are advanced, and the parser's
//     verdict decides accept/reject. The probe is popped afterward, so
//     the matcher state is unchanged.
// `set_token_bitmask` then turns the accepted-bitset + rejected-interset
// into the packed `i32` mask, adding stop tokens when the root rule can
// complete and clearing special tokens.

use crate::compiler::StoreType;
use crate::support::int_set::{intset_intersection, intset_union};

use super::bitmask::BitmaskSlice;
use super::matcher::{FillScratch, GrammarMatcher};

impl GrammarMatcher {
    /// Fill `bitmask` with the set of tokens acceptable for the next
    /// decode position.
    ///
    /// `index` selects the matcher's slice when the caller batches
    /// several matchers into one tensor; for a single matcher pass `0`.
    /// `debug` is accepted for API parity (no effect here).
    ///
    /// Returns `Ok(true)` when the mask is *not* all-accepting (so it
    /// must be applied to the logits), `Ok(false)` when every token is
    /// legal. Returns `Err` if the matcher has terminated or the buffer
    /// is too small. Port of `FillNextTokenBitmask`.
    pub fn fill_next_token_bitmask(
        &mut self,
        bitmask: &mut [i32],
        index: usize,
        debug: bool,
    ) -> Result<bool, FillError> {
        let _ = debug;
        if self.stop_token_accepted() {
            return Err(FillError::Terminated);
        }
        let vocab_size = self.tokenizer_info().vocab_size();
        let words = super::bitmask::bitmask_size(vocab_size);
        let start = index.checked_mul(words).ok_or(FillError::BadIndex)?;
        let slice = bitmask
            .get_mut(start..start + words)
            .ok_or(FillError::BufferTooSmall)?;
        let mut view = BitmaskSlice::new(slice, vocab_size).ok_or(FillError::BufferTooSmall)?;

        // Take the reusable scratch buffers for the duration of the
        // fill (a `Default::default()` left in their place), so the hot
        // path owns them outright with no per-token allocation. They
        // are restored unconditionally before returning.
        let mut scratch = std::mem::take(&mut self.scratch);
        let can_reach_end = self.compute_partitions(&mut scratch);
        self.set_token_bitmask(
            &mut view,
            &scratch.accepted,
            &scratch.rejected,
            can_reach_end,
        );
        self.scratch = scratch;
        Ok(!view.all_set())
    }

    /// Populate `scratch.accepted` (accepted-bitset) and
    /// `scratch.rejected` (rejected-interset) over every latest scanable
    /// state, resolving uncertain tokens by trial. Returns
    /// `can_reach_end`.
    ///
    /// All working buffers are reused from `scratch` — `.clear()`ed,
    /// not reallocated — so this runs with zero heap traffic on the hot
    /// path. The vocab-sized `sorted_decoded_vocab` and
    /// `trie_subtree_nodes_range` are borrowed in place from
    /// `self.compiled_grammar` (a field disjoint from `self.parser` and
    /// `self.scratch`), never cloned.
    fn compute_partitions(&mut self, scratch: &mut FillScratch) -> bool {
        let vocab_size = self.tokenizer_info().vocab_size();
        // `accepted[token_id]` — union of every state's accepted set.
        scratch.accepted.clear();
        scratch.accepted.resize(vocab_size, false);
        // `rejected` — running intersection; `{-1}` is the universal set.
        scratch.rejected.clear();
        scratch.rejected.push(-1);

        // Snapshot the parser's latest scanable states (`ParserState`
        // is `Copy`) into the scratch buffer once, so the trial loop
        // below can mutate `self.parser` freely without holding a
        // borrow of `latest_scanable_states()`.
        scratch.live_states.clear();
        scratch
            .live_states
            .extend_from_slice(self.parser.latest_scanable_states());

        // Resolve each live scanable state to the canonical
        // `ParserState` the compiler keys its mask under (live states
        // carry real positions; the mask cache is position-agnostic),
        // then JIT-compile (or fetch from the lazy cache) its
        // `AdaptiveTokenMask`. `get_or_compute_mask` returns an `Arc`.
        let root_rule_id = self.compiled_grammar.grammar().root_rule_id();
        scratch.states.clear();
        for i in 0..scratch.live_states.len() {
            let live = scratch.live_states[i];
            let canon = self.canonical_mask_state(&live);
            let is_root = canon.rule_id == root_rule_id;
            let mask = self.compiled_grammar.get_or_compute_mask(canon, is_root);
            scratch.states.push((live, mask));
        }

        // Vocab-sized tables borrowed in place — `self.compiled_grammar`
        // is disjoint from `self.parser` and `scratch`.
        let sorted = self
            .compiled_grammar
            .tokenizer_info()
            .sorted_decoded_vocab();
        let subtree = self
            .compiled_grammar
            .tokenizer_info()
            .trie_subtree_nodes_range();

        // Pass 1: seed `accepted` from every state's static accepted set.
        for (_, mask) in &scratch.states {
            let mask = mask.as_ref();
            match mask.store_type {
                StoreType::AcceptedBitset => {
                    for (tid, slot) in scratch.accepted.iter_mut().enumerate() {
                        if mask.accepted_bitset[tid] {
                            *slot = true;
                        }
                    }
                }
                StoreType::Accepted => {
                    for &idx in &mask.accepted_indices {
                        scratch.accepted[sorted[idx as usize].0 as usize] = true;
                    }
                }
                StoreType::Rejected => {}
            }
        }

        // Pass 2: resolve uncertain tokens per state via trial advance.
        // Iterate by index so `scratch.states` is not borrowed across
        // the `self.parser` mutation; the `Arc<AdaptiveTokenMask>` is
        // cheap to clone and keeps the mask alive without that borrow.
        for si in 0..scratch.states.len() {
            let live = scratch.states[si].0;
            let mask = std::sync::Arc::clone(&scratch.states[si].1);
            let mask = mask.as_ref();
            scratch.rejected_delta.clear();
            self.parser.push_one_state_to_check(live);

            let mut prev_token: Option<&[u8]> = None;
            let mut prev_matched = 0usize;
            let mut last_rejected_range = 0i32;

            for &cur_idx in &mask.uncertain_indices {
                let (cur_id, ref cur_token) = sorted[cur_idx as usize];
                if scratch.accepted[cur_id as usize] {
                    continue;
                }
                if cur_idx < last_rejected_range {
                    if mask.store_type == StoreType::Rejected {
                        scratch.rejected_delta.push(cur_idx);
                    }
                    continue;
                }

                let mut is_accepted = true;
                // Reuse the longest common prefix already matched.
                if let Some(prev) = prev_token {
                    let lcp = cur_token
                        .iter()
                        .zip(prev.iter())
                        .take_while(|(a, b)| a == b)
                        .count();
                    if lcp > prev_matched {
                        last_rejected_range = subtree[cur_idx as usize];
                        is_accepted = false;
                    } else if lcp < prev_matched {
                        self.parser.pop_last_states(prev_matched - lcp);
                    }
                    prev_matched = prev_matched.min(lcp);
                }

                // Advance the remaining bytes of this token.
                if is_accepted {
                    for (j, &byte) in cur_token.iter().enumerate().skip(prev_matched) {
                        if !self.parser.advance(byte) {
                            last_rejected_range = subtree[cur_idx as usize];
                            is_accepted = false;
                            break;
                        }
                        prev_matched = j + 1;
                    }
                }

                match mask.store_type {
                    StoreType::AcceptedBitset | StoreType::Accepted => {
                        if is_accepted {
                            scratch.accepted[cur_id as usize] = true;
                        }
                    }
                    StoreType::Rejected => {
                        if !is_accepted {
                            scratch.rejected_delta.push(cur_idx);
                        }
                    }
                }
                prev_token = Some(cur_token);
            }

            // Pop the bytes advanced for the last uncertain token plus
            // the one-state probe step itself.
            self.parser.pop_last_states(prev_matched + 1);

            if mask.store_type == StoreType::Rejected {
                // rejected = intersect(rejected, mask.rejected ∪ delta)
                intset_union(&mut scratch.rejected_delta, &mask.rejected_indices);
                intset_intersection(&mut scratch.rejected, &scratch.rejected_delta);
            }
        }

        self.parser.is_completed()
    }

    /// Turn the accepted-bitset + rejected-interset into the packed
    /// mask. Port of `SetTokenBitmask`.
    fn set_token_bitmask(
        &self,
        view: &mut BitmaskSlice<'_>,
        accepted: &[bool],
        rejected: &[i32],
        can_reach_end: bool,
    ) {
        let sorted = self.tokenizer_info().sorted_decoded_vocab();
        if rejected.len() == 1 && rejected[0] == -1 {
            // Universal rejected set: accepted set is exactly `accepted`.
            view.clear();
            for (tid, &ok) in accepted.iter().enumerate() {
                if ok {
                    view.set(tid, true);
                }
            }
            if can_reach_end {
                for &id in self.stop_token_ids() {
                    view.set(id as usize, true);
                }
            }
        } else {
            // Final rejected set is `rejected \ accepted`.
            view.fill_all();
            for &idx in rejected {
                let id = sorted[idx as usize].0 as usize;
                if !accepted[id] {
                    view.set(id, false);
                }
            }
            for &id in self.tokenizer_info().special_token_ids() {
                view.set(id as usize, false);
            }
            if !can_reach_end {
                for &id in self.stop_token_ids() {
                    view.set(id as usize, false);
                }
            }
        }
    }

    /// Find the jump-forward string: the longest prefix that is forced
    /// by the grammar regardless of which token is sampled.
    ///
    /// The matcher state is left unchanged. Port of
    /// `FindJumpForwardString`: at each step every latest scanable
    /// state must agree on exactly one acceptable byte; the union of
    /// acceptable bytes having size 1 is the equivalent condition.
    #[must_use]
    pub fn find_jump_forward_string(&mut self) -> Vec<u8> {
        if self.stop_token_accepted() {
            return Vec::new();
        }
        let mut result = Vec::new();
        let mut advanced = 0usize;
        loop {
            if self.parser.is_completed() {
                break;
            }
            let mask = self.parser.acceptable_byte_mask();
            let mut only: Option<u8> = None;
            let mut unique = true;
            for (b, &ok) in mask.iter().enumerate() {
                if ok {
                    if only.is_some() {
                        unique = false;
                        break;
                    }
                    only = Some(b as u8);
                }
            }
            let next = match (unique, only) {
                (true, Some(b)) => b,
                _ => break,
            };
            if !self.parser.advance(next) {
                break;
            }
            result.push(next);
            advanced += 1;
        }
        if advanced > 0 {
            self.parser.pop_last_states(advanced);
        }
        result
    }

    /// `find_jump_forward_string` as a UTF-8 `String`, lossily decoding
    /// non-UTF-8 bytes. Convenience for callers that want text.
    #[must_use]
    pub fn find_jump_forward_string_lossy(&mut self) -> String {
        String::from_utf8_lossy(&self.find_jump_forward_string()).into_owned()
    }
}

/// Why [`GrammarMatcher::fill_next_token_bitmask`] could not fill.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum FillError {
    /// The matcher already accepted a stop token and terminated.
    #[error("matcher has terminated; cannot fill next-token bitmask")]
    Terminated,
    /// The supplied buffer is shorter than `bitmask_size(vocab_size)`.
    #[error("bitmask buffer too small for the vocabulary")]
    BufferTooSmall,
    /// `index * bitmask_size` overflowed `usize`.
    #[error("bitmask index out of range")]
    BadIndex,
}
