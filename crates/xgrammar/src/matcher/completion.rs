// SPDX-License-Identifier: AGPL-3.0-only
//
// GrammarMatcher — shortest grammar-legal completion to a stop-legal state.
//
// Powers budget-aware graceful close of structured outputs (Atlas #144):
// when a length-limited response would otherwise end with the stop token
// forbidden mid-structure (e.g. inside an open JSON string), this finds the
// shortest byte sequence that drives the grammar to a state where the root
// rule can complete — so a truncated `finish_reason="length"` response is
// still parseable.
//
// There is no C++ upstream for this; it is an Atlas addition built on the
// same byte-level Earley primitives as `find_jump_forward_string`
// (`advance` / `pop_last_states` / `is_completed`).

use std::collections::{HashSet, VecDeque};

use crate::earley::ParserState;

use super::matcher::GrammarMatcher;

/// Hard cap on parser byte-advances explored, independent of `max_bytes`.
/// Bounds worst-case cost when the grammar branches widely (e.g. a long
/// string-content alphabet) before a legal close is reachable. A close
/// sequence is short in practice; if none is found under this budget the
/// caller falls back to a plain finish — strictly no worse than the prior
/// behavior, which always finished there.
const COMPLETION_NODE_BUDGET: usize = 4096;

impl GrammarMatcher {
    /// The shortest byte sequence that advances the grammar from its
    /// CURRENT state to one where the root rule can complete (a stop token
    /// is legal), or `None` if no such sequence exists within `max_bytes`
    /// and the node budget. The matcher state is left UNCHANGED.
    ///
    /// Returns `Some(empty)` when the grammar can already stop here.
    ///
    /// Soundness: every byte of a returned path is applied via
    /// `parser.advance`, and a path is only returned once
    /// `parser.is_completed()` holds — so the result is always a
    /// grammar-legal completion. The visited-set dedup (on the
    /// canonicalized scanable-state multiset) only bounds the search:
    /// over-merging can at worst make this return `None` when some longer
    /// completion existed, never an illegal path. BFS order makes any
    /// returned path the shortest among those explored.
    #[must_use]
    pub fn find_completion_to_accept(&mut self, max_bytes: usize) -> Option<Vec<u8>> {
        if self.is_terminated() {
            return None;
        }
        if self.parser.is_completed() {
            return Some(Vec::new());
        }

        let mut visited: HashSet<Vec<ParserState>> = HashSet::new();
        visited.insert(self.state_key());
        let mut queue: VecDeque<Vec<u8>> = VecDeque::new();
        queue.push_back(Vec::new());
        let mut budget = COMPLETION_NODE_BUDGET;

        while let Some(path) = queue.pop_front() {
            if path.len() >= max_bytes {
                continue;
            }
            // Replay `path` to position the parser at this search node; the
            // matching `pop_last_states` below restores the original state.
            self.replay(&path);

            let mask = self.parser.acceptable_byte_mask();
            let mut found: Option<Vec<u8>> = None;
            for (b, &ok) in mask.iter().enumerate() {
                if !ok {
                    continue;
                }
                if budget == 0 {
                    break;
                }
                budget -= 1;
                let byte = b as u8;
                if !self.parser.advance(byte) {
                    continue;
                }
                if self.parser.is_completed() {
                    let mut full = path.clone();
                    full.push(byte);
                    found = Some(full);
                    self.parser.pop_last_states(1);
                    break;
                }
                if visited.insert(self.state_key()) {
                    let mut next = path.clone();
                    next.push(byte);
                    queue.push_back(next);
                }
                self.parser.pop_last_states(1);
            }

            if !path.is_empty() {
                self.parser.pop_last_states(path.len());
            }
            if let Some(full) = found {
                return Some(full);
            }
            if budget == 0 {
                break;
            }
        }
        None
    }

    /// Like [`Self::find_completion_to_accept`], but returns the close as
    /// content **token ids**, greedily encoded against this matcher's vocab
    /// (`sorted_decoded_vocab`, which excludes stop/special tokens — so a
    /// close never emits a control token). Returns `Some(empty)` when the
    /// grammar can already stop, and `None` when no bounded close exists or
    /// a close byte is not representable as a content token.
    #[must_use]
    pub fn find_completion_token_ids(&mut self, max_bytes: usize) -> Option<Vec<i32>> {
        let bytes = self.find_completion_to_accept(max_bytes)?;
        if bytes.is_empty() {
            return Some(Vec::new());
        }
        self.encode_bytes_greedy(&bytes)
    }

    /// Greedy longest-match encode of `bytes` into content token ids. The
    /// concatenated token bytes equal `bytes` by construction, so detokenizing
    /// the result reproduces the close exactly. `None` if any position has no
    /// covering token.
    fn encode_bytes_greedy(&self, bytes: &[u8]) -> Option<Vec<i32>> {
        let vocab = self.tokenizer_info().sorted_decoded_vocab();
        let mut out = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            let rem = &bytes[i..];
            let mut best: Option<(i32, usize)> = None;
            for (id, tok) in vocab {
                let len = tok.len();
                if len == 0 || len > rem.len() || !rem.starts_with(tok.as_slice()) {
                    continue;
                }
                if best.is_none_or(|(_, bl)| len > bl) {
                    best = Some((*id, len));
                }
            }
            let (id, len) = best?;
            out.push(id);
            i += len;
        }
        Some(out)
    }

    /// Canonical, hashable key for the parser's current scanable-state set.
    /// Sorting makes the key order-independent so two byte paths reaching
    /// the same logical configuration dedup together.
    fn state_key(&self) -> Vec<ParserState> {
        let mut states = self.parser.latest_scanable_states().to_vec();
        states.sort_unstable_by_key(|s| {
            (
                s.rule_id,
                s.sequence_id,
                s.element_id,
                s.rule_start_pos,
                s.sub_element_id,
                s.repeat_count,
                s.partial_codepoint,
            )
        });
        states
    }

    /// Re-advance the parser along `path`. Every byte was validated by
    /// `advance` when the path was enqueued, so each step succeeds.
    fn replay(&mut self, path: &[u8]) {
        for &b in path {
            let advanced = self.parser.advance(b);
            debug_assert!(advanced, "replayed completion byte must re-advance");
        }
    }
}
