// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! Lookup drafts for the wide verify (#974).
//!
//! The draft for the next step is read out of the sequence itself: find the
//! most recent earlier occurrence of the tokens just generated and propose the
//! tokens that followed it. No draft model runs. On generation that repeats
//! its context (a tool result echoed back, a file re-emitted with one edit,
//! restated instructions, tool-call scaffolding) the continuation is the right
//! one nearly every time, and the step costs the verify alone. On fresh
//! reasoning nothing matches and the MTP head drafts as before.
//!
//! The hookup is [`crate::scheduler::lookup_gate`]: when the index fires, the
//! drafts go into `pending_drafts` and the MTP propose is skipped for that
//! step; the verify, accept and commit are unchanged.
//!
//! # The index
//!
//! [`crate::ngram::NgramProposer`] rescans the whole history for every match
//! length on every step, which is quadratic in the context and proposes one
//! token. This proposer keeps a hash index from every `min_match`-gram in the
//! sequence to the positions it starts at, extended incrementally as tokens
//! commit, so a step costs the candidates at one key. At 88K tokens of context
//! that is the difference between a scan and a lookup.
//!
//! Candidates are tried most recent first and the longest backward match wins
//! (ties keep the most recent), so a block that has been repeated several
//! times follows its latest copy. The continuation is returned only when it
//! fills the requested width: the wide verify dispatches on the number of
//! drafts, and a short draft would change the step shape the MTP head was
//! tuned at.

use std::collections::HashMap;

/// Prompt-lookup proposer with an incremental n-gram index over one sequence.
pub struct LookupDrafter {
    /// Match length the index is keyed on; shorter matches never fire.
    min_match: usize,
    /// Longest backward match considered when ranking candidates.
    max_match: usize,
    /// `hash(tokens[p..p + min_match])` to the start positions `p`, ascending.
    index: HashMap<u64, Vec<u32>>,
    /// How many tokens of the current sequence the index covers.
    indexed_len: usize,
    /// The last `min_match` indexed tokens, to notice a different sequence.
    tail: Vec<u32>,
    /// Armed by the MTP step for the run's single-sequence regime; the gate
    /// never fires with two sequences active (one index, one sequence).
    single_sequence: bool,
    /// Steps where the index fired.
    pub fired: u64,
    /// Drafts proposed across those steps.
    pub proposed: u64,
    /// Drafts the verify accepted across those steps.
    pub accepted: u64,
}

impl LookupDrafter {
    pub fn new(min_match: usize, max_match: usize) -> Self {
        let min_match = min_match.max(2);
        Self {
            min_match,
            max_match: max_match.max(min_match),
            index: HashMap::new(),
            indexed_len: 0,
            tail: Vec::new(),
            single_sequence: false,
            fired: 0,
            proposed: 0,
            accepted: 0,
        }
    }

    pub fn set_single_sequence(&mut self, single: bool) {
        self.single_sequence = single;
    }

    pub fn single_sequence(&self) -> bool {
        self.single_sequence
    }

    pub fn min_match(&self) -> usize {
        self.min_match
    }

    /// Positions indexed so far (tests and diagnostics).
    pub fn indexed_len(&self) -> usize {
        self.indexed_len
    }

    /// Forget the sequence; the next `propose` rebuilds from scratch.
    pub fn reset(&mut self) {
        self.index.clear();
        self.indexed_len = 0;
        self.tail.clear();
    }

    /// Count a verify of `proposed` lookup drafts that accepted `accepted`.
    pub fn record(&mut self, proposed: usize, accepted: usize) {
        self.proposed += proposed as u64;
        self.accepted += accepted as u64;
    }

    fn key(gram: &[u32]) -> u64 {
        // FNV-1a over the token ids; the index verifies every hit token by
        // token, so a collision costs a compare, never a wrong draft.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &t in gram {
            h ^= t as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// Bring the index up to `tokens.len()`. A sequence that does not extend
    /// the indexed prefix (a new request, a watchdog rollback) is re-indexed.
    fn extend(&mut self, tokens: &[u32]) {
        let len = tokens.len();
        let n = self.min_match;
        let continues = len >= self.indexed_len
            && self.indexed_len >= self.tail.len()
            && tokens[self.indexed_len - self.tail.len()..self.indexed_len] == self.tail[..];
        if !continues {
            self.reset();
        }
        if len < n {
            return;
        }
        // Grams already indexed start at p <= indexed_len - n.
        let first = if self.indexed_len >= n {
            self.indexed_len - n + 1
        } else {
            0
        };
        for p in first..=len - n {
            self.index
                .entry(Self::key(&tokens[p..p + n]))
                .or_default()
                .push(p as u32);
        }
        self.indexed_len = len;
        self.tail.clear();
        self.tail.extend_from_slice(&tokens[len - n.min(len)..]);
    }

    /// Propose exactly `width` tokens to follow `tokens` + `next`, or nothing.
    ///
    /// `tokens` is the sequence the model has consumed, prompt and generation
    /// together; `next` is the newest token, emitted but not yet fed (the
    /// scheduler's `last_token`). The drafts are what followed the last
    /// occurrence of `tokens[..] ++ [next]`'s tail, so draft 0 is the token
    /// after `next`, never `next` itself.
    pub fn propose(&mut self, tokens: &[u32], next: u32, width: usize) -> Vec<u32> {
        let len = tokens.len();
        let n = self.min_match;
        if width == 0 || len + 1 < n + width {
            return Vec::new();
        }
        self.extend(tokens);
        // The virtual sequence v = tokens ++ [next]; its last n-gram starts
        // at v index len + 1 - n and ends with `next`.
        let mut suffix: Vec<u32> = tokens[len + 1 - n..].to_vec();
        suffix.push(next);
        let suffix_start = len + 1 - n;
        let Some(starts) = self.index.get(&Self::key(&suffix)) else {
            return Vec::new();
        };
        let mut best: Option<(usize, usize)> = None; // (match length, start)
        for &p in starts.iter().rev() {
            let p = p as usize;
            // The hit needs `width` tokens after it inside `tokens`, and it
            // cannot be the suffix itself.
            if p + n + width > len || p == suffix_start {
                continue;
            }
            if tokens[p..p + n] != suffix[..] {
                continue;
            }
            // Extend the match backwards; longer context, better draft.
            let mut m = n;
            while m < self.max_match
                && p >= m + 1 - n
                && suffix_start >= m + 1 - n
                && tokens[p + n - 1 - m] == tokens[len - m]
            {
                m += 1;
            }
            if best.is_none_or(|(bm, _)| m > bm) {
                best = Some((m, p));
                if m == self.max_match {
                    break;
                }
            }
        }
        let Some((_, p)) = best else {
            return Vec::new();
        };
        self.fired += 1;
        tokens[p + n..p + n + width].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeats_the_continuation_of_the_latest_copy() {
        let mut d = LookupDrafter::new(3, 16);
        // [1 2 3] -> 4 5 the first time, [1 2 3] -> 7 8 the second time.
        let t = vec![1, 2, 3, 4, 5, 9, 1, 2, 3, 7, 8, 9, 1, 2];
        assert_eq!(d.propose(&t, 3, 2), vec![7, 8]);
        assert_eq!(d.fired, 1);
    }

    #[test]
    fn longest_backward_match_beats_recency() {
        let mut d = LookupDrafter::new(3, 16);
        // Older copy shares 4 tokens of context, newer copy shares 3.
        let t = vec![0, 1, 2, 3, 4, 5, 9, 9, 1, 2, 3, 6, 6, 9, 0, 1, 2];
        assert_eq!(d.propose(&t, 3, 1), vec![4]);
    }

    #[test]
    fn nothing_on_fresh_generation() {
        let mut d = LookupDrafter::new(3, 16);
        assert!(d.propose(&[1, 2, 3, 4, 5, 6, 7], 8, 2).is_empty());
        assert_eq!(d.fired, 0);
    }

    #[test]
    fn a_short_continuation_does_not_fire() {
        let mut d = LookupDrafter::new(3, 16);
        // The only earlier copy has four tokens after it; a width of five
        // cannot be filled, a width of four runs into the suffix itself.
        let t = vec![1, 2, 3, 4, 1, 2];
        assert!(d.propose(&t, 3, 4).is_empty());
        assert_eq!(d.propose(&t, 3, 3), vec![4, 1, 2]);
    }

    #[test]
    fn the_suffix_is_not_its_own_continuation() {
        let mut d = LookupDrafter::new(2, 16);
        let t = vec![5, 5, 5];
        // Hits at 0 and 1 continue with 5; the suffix start (2) is skipped.
        assert_eq!(d.propose(&t, 5, 1), vec![5]);
    }

    #[test]
    fn index_extends_incrementally_and_rebuilds_on_a_new_sequence() {
        let mut d = LookupDrafter::new(3, 16);
        let mut t = vec![1, 2, 3, 4, 5];
        assert!(d.propose(&t, 6, 1).is_empty());
        assert_eq!(d.indexed_len(), 5);
        t.extend([6, 1, 2]);
        assert_eq!(d.propose(&t, 3, 1), vec![4]);
        assert_eq!(d.indexed_len(), 8);
        // A different sequence of the same length: re-indexed, no stale hit.
        let u = vec![9, 8, 7, 6, 5, 4, 9, 8];
        assert_eq!(d.propose(&u, 7, 1), vec![6]);
        assert_eq!(d.indexed_len(), 8);
        // A rollback shorter than the indexed prefix is re-indexed too.
        let v = vec![9, 8, 7, 6, 5, 9, 8];
        assert_eq!(d.propose(&v, 7, 1), vec![6]);
        assert_eq!(d.indexed_len(), 7);
    }

    #[test]
    fn matches_the_quadratic_proposer_on_its_own_cases() {
        // The cases `crate::ngram` tests, at the same minimum match.
        let mut d = LookupDrafter::new(2, 16);
        assert_eq!(d.propose(&[1, 2, 3, 4, 5, 1, 2], 3, 1), vec![4]);
        assert!(d.propose(&[1, 2, 3, 4, 5, 6, 7], 8, 1).is_empty());
        assert!(d.propose(&[1], 2, 1).is_empty());
        assert_eq!(d.propose(&[10, 20, 30, 10, 20, 30, 10], 20, 1), vec![30]);
    }

    #[test]
    fn a_long_context_is_one_lookup_per_step() {
        // 100K tokens of noise, then a 64-token block repeated: the second
        // copy drafts the first at every step.
        let mut t: Vec<u32> = (0..100_000u32).map(|i| i.wrapping_mul(2_654_435_761) % 50_000 + 1_000).collect();
        let block: Vec<u32> = (0..64u32).map(|i| 60_000 + i).collect();
        t.extend(&block);
        t.extend(&block[..7]);
        let mut d = LookupDrafter::new(4, 16);
        let mut hits = 0;
        // The newest token (block[7 + 2*step]) is `next`, not yet in `t`.
        for step in 0..28 {
            let next = block[7 + step * 2];
            let got = d.propose(&t, next, 2);
            let want = block[8 + step * 2..8 + step * 2 + 2].to_vec();
            if got == want {
                hits += 1;
            }
            t.push(next);
            t.push(want[0]);
        }
        assert_eq!(hits, 28);
    }
}
