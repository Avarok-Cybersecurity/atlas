// SPDX-License-Identifier: AGPL-3.0-only
// Portions copied and adapted from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce, zig/src/core/copy_index.zig:1-80. Copyright 2026 TensorFold contributors. Licensed under the Apache License, Version 2.0; code written before TensorFold 0.6.0 is also available under the MIT License. See COPY-DRAFT-ATTRIBUTION.md. Changes for Atlas: the per-token position lists became per-BIGRAM lists, the walk runs earliest first, and the result reproduces Atlas's earlier `NgramProposer::propose` scan token for token instead of TensorFold's latest-first rule; truncation of the stream pops the tail entries; translated from Zig to Rust.

//! Responsibility: the incremental index behind the legacy
//! `--ngram-speculative` proposer, giving EXACTLY the token today's quadratic
//! scan (`NgramProposer::propose`) gives, at a fraction of its cost.
//!
//! The scan answers: for the longest suffix length `n` in
//! `min_match..=min(max_match, len - 1)` that occurs earlier (the occurrence
//! must leave one token after it, overlap allowed), the token after the
//! EARLIEST such occurrence. Every occurrence of a suffix of length `n >= 2`
//! ends with the context's last bigram, so the candidates are exactly the
//! earlier positions of that bigram. For each one (earliest first) the
//! backward match length `m` is measured against the context's tail; the
//! longest `m` wins and, among equal `m`, the earliest position (the first one
//! seen, since only a strictly longer `m` replaces it). The scan's downward
//! closure (a match of length `n` implies one of length `n - 1`) makes "the
//! longest `n` with any match" equal to the largest `m`.
//!
//! The index keeps its own copy of the history it was last asked about. The
//! history may shrink (a rejected n-gram draft pops one token) or belong to a
//! new request: [`LegacyLookup::propose`] finds the longest common prefix with
//! the held copy, pops the bigram entries past it (they are the tail of each
//! list, since positions are appended in order) and appends the rest. Only
//! used when `copy_draft != off`; otherwise the scan runs unchanged.

use std::collections::HashMap;

#[inline]
fn bigram(a: u32, b: u32) -> u64 {
    ((a as u64) << 32) | b as u64
}

/// Bigram -> ascending start positions, over the held history.
#[derive(Default)]
pub(crate) struct LegacyLookup {
    toks: Vec<u32>,
    bigrams: HashMap<u64, Vec<u32>>,
}

impl LegacyLookup {
    pub fn new() -> Self {
        Self::default()
    }

    /// Tokens currently held (tests).
    pub fn held(&self) -> usize {
        self.toks.len()
    }

    fn sync(&mut self, h: &[u32]) {
        let n = self.toks.len();
        let p = if n <= h.len() && self.toks[..] == h[..n] {
            n
        } else {
            self.toks
                .iter()
                .zip(h)
                .position(|(a, b)| a != b)
                .unwrap_or(n.min(h.len()))
        };
        if p < n {
            // Bigram at q covers q and q + 1: every q + 1 >= p is stale.
            let first = p.saturating_sub(1);
            for q in (first..n.saturating_sub(1)).rev() {
                let key = bigram(self.toks[q], self.toks[q + 1]);
                if let Some(list) = self.bigrams.get_mut(&key) {
                    let popped = list.pop();
                    debug_assert_eq!(popped, Some(q as u32));
                    if list.is_empty() {
                        self.bigrams.remove(&key);
                    }
                }
            }
            self.toks.truncate(p);
        }
        for i in self.toks.len()..h.len() {
            self.toks.push(h[i]);
            if i >= 1 {
                self.bigrams
                    .entry(bigram(h[i - 1], h[i]))
                    .or_default()
                    .push((i - 1) as u32);
            }
        }
    }

    /// The token `NgramProposer::propose` returns for `h` with these bounds.
    /// `min_match` must be at least 2 (the caller keeps the scan otherwise).
    pub fn propose(&mut self, h: &[u32], min_match: usize, max_match: usize) -> Option<u32> {
        debug_assert!(min_match >= 2, "the bigram index needs min_match >= 2");
        self.sync(h);
        let len = h.len();
        if len < min_match + 1 {
            return None;
        }
        let max_n = max_match.min(len - 1);
        if max_n < min_match {
            return None;
        }
        let list = self.bigrams.get(&bigram(h[len - 2], h[len - 1]))?;
        let mut best_m = 0usize;
        let mut best_q = 0usize;
        for &q in list {
            let q = q as usize;
            // The occurrence must end at or before len - 2, leaving a token.
            if q + 3 > len {
                break;
            }
            let mut m = 2usize;
            while m < max_n && q + 1 >= m && h[q + 1 - m] == h[len - 1 - m] {
                m += 1;
            }
            if m > best_m {
                best_m = m;
                best_q = q;
                if m == max_n {
                    break;
                }
            }
        }
        if best_m < min_match {
            return None;
        }
        Some(h[best_q + 2])
    }
}
