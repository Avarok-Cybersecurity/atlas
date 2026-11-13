// SPDX-License-Identifier: AGPL-3.0-only

//! Prompt-lookup proposer for speculative decoding.
//!
//! Finds the longest matching subsequence in the full token history
//! (prompt + generated tokens) and proposes the token that followed
//! that subsequence. Zero GPU cost — runs entirely on CPU.
//!
//! This is equivalent to vLLM's "prompt lookup decoding" and works
//! well for tasks where the model paraphrases, quotes, or produces
//! structured output similar to earlier content.
//!
//! [`NgramProposer::propose_next`] is the scheduler's entry point. With the
//! `copy_draft` row off (every target's default) it IS [`NgramProposer::propose`],
//! the linear scan below, byte for byte. With the row on it answers from an
//! incremental bigram index (`scheduler::copy_draft::legacy`) that returns
//! the same token as the scan for every history (CPU oracle in
//! `scheduler/copy_draft/legacy_tests.rs`), without rescanning the history
//! every step.

use crate::scheduler::copy_draft::legacy::LegacyLookup;

/// Prompt-lookup based draft proposer for speculative decoding.
///
/// On each call to `propose()`, searches the full token history for
/// the longest suffix match and returns the token that followed it.
/// No explicit cache needed — the token history IS the cache.
pub struct NgramProposer {
    /// Minimum match length (shorter matches are too noisy)
    min_match: usize,
    /// Maximum match length to search for
    max_match: usize,
    /// Running accept/reject stats for logging
    pub accepts: u64,
    pub rejects: u64,
    /// The incremental index `propose_next` uses when `copy_draft != off`;
    /// never allocated otherwise.
    indexed: Option<LegacyLookup>,
}

impl NgramProposer {
    pub fn new(_order: usize) -> Self {
        Self {
            min_match: 2,
            max_match: 16,
            accepts: 0,
            rejects: 0,
            indexed: None,
        }
    }

    /// The scheduler's proposal for `all_tokens`: the same token as
    /// [`Self::propose`], from the incremental index when the copy drafter
    /// is on and from the scan otherwise.
    pub fn propose_next(&mut self, all_tokens: &[u32]) -> Option<u32> {
        if self.min_match >= 2 && crate::scheduler::copy_draft::active() {
            let (min, max) = (self.min_match, self.max_match);
            return self
                .indexed
                .get_or_insert_with(LegacyLookup::new)
                .propose(all_tokens, min, max);
        }
        self.propose(all_tokens)
    }

    /// Record an accepted draft (for stats only).
    pub fn record_accept(&mut self) {
        self.accepts += 1;
    }

    /// Record a rejected draft (for stats only).
    pub fn record_reject(&mut self) {
        self.rejects += 1;
    }

    /// Propose a draft token based on prompt lookup.
    ///
    /// Searches `all_tokens` (prompt + generated) for the longest suffix
    /// match with the tail of `all_tokens`. Returns the token that followed
    /// the best match, or `None` if no match is found.
    pub fn propose(&self, all_tokens: &[u32]) -> Option<u32> {
        let len = all_tokens.len();
        if len < self.min_match + 1 {
            return None;
        }

        let max_n = self.max_match.min(len - 1);
        let mut best_next: Option<u32> = None;
        let mut best_match_len: usize = 0;

        // Try each suffix length from min_match to max_n.
        // Suffix = all_tokens[len-n..len]. Search for it at positions 0..=len-n-1
        // (must leave at least 1 token after the match to propose).
        for n in self.min_match..=max_n {
            let suffix = &all_tokens[len - n..len];
            for start in 0..=(len - n - 1) {
                if all_tokens[start..start + n] == *suffix && n > best_match_len {
                    best_match_len = n;
                    best_next = Some(all_tokens[start + n]);
                    break; // take first (earliest) match for this length
                }
            }
        }

        best_next
    }

    /// Observe is a no-op for prompt lookup (the history IS the cache).
    pub fn observe(&mut self, _history: &[u32], _next: u32) {}

    pub fn len(&self) -> usize {
        0 // No explicit cache
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_lookup_basic() {
        let p = NgramProposer::new(4);

        // History: [1, 2, 3, 4, 5, 1, 2, 3]
        // Suffix [1, 2, 3] matches at position 0, followed by 4
        let tokens = vec![1, 2, 3, 4, 5, 1, 2, 3];
        assert_eq!(p.propose(&tokens), Some(4));
    }

    #[test]
    fn test_prompt_lookup_no_match() {
        let p = NgramProposer::new(4);

        // No repeated pattern
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(p.propose(&tokens), None);
    }

    #[test]
    fn test_prompt_lookup_short() {
        let p = NgramProposer::new(4);

        // Too short for min_match
        let tokens = vec![1, 2];
        assert_eq!(p.propose(&tokens), None);
    }

    #[test]
    fn test_prompt_lookup_repetitive() {
        let p = NgramProposer::new(4);

        // Highly repetitive: [A, B, C, A, B, C, A, B]
        // Suffix [A, B] matches at pos 0 (followed by C) and pos 3 (followed by C)
        let tokens = vec![10, 20, 30, 10, 20, 30, 10, 20];
        assert_eq!(p.propose(&tokens), Some(30));
    }
}
