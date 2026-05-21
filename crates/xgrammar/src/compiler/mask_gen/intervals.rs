// SPDX-License-Identifier: AGPL-3.0-only
//
// Possible-token-interval computation + the main vocabulary scan loop.
// Port of `GetPossibleTokenIntervals` and
// `GrammarMatcherForTokenMaskCache::GetTokenMaskWithFirstCharacterCheck`
// from `cpp/grammar_compiler.cc`.

use super::super::mask::USE_BITSET_THRESHOLD;
use super::MaskGenerator;

/// `(left, right)` half-open ranges of `sorted_decoded_vocab` whose
/// tokens start with a byte allowed by `first_char_mask`.
///
/// Port of `GetPossibleTokenIntervals`. Returns `(intervals, count)`.
pub(crate) fn possible_token_intervals(
    sorted: &[(i32, Vec<u8>)],
    first_char_mask: &[bool; 256],
) -> (Vec<(i32, i32)>, i32) {
    let mut intervals = Vec::new();
    let mut count = 0i32;
    let mut matched_size = 0i32;
    let mut last_interval_end: i32 = -1;

    let lower_bound_by_first = |from: i32, first: u8| -> i32 {
        // First index >= `from` whose token's first byte is >= `first`,
        // i.e. lower_bound on the single-byte key `[first]`.
        let slice = &sorted[from as usize..];
        let pos = slice.partition_point(|(_, tok)| tok.as_slice() < [first].as_slice());
        from + pos as i32
    };

    for i in 0..256i32 {
        if first_char_mask[i as usize] {
            if last_interval_end == -1 {
                last_interval_end = i;
            }
        } else if last_interval_end != -1 {
            let left = lower_bound_by_first(matched_size, last_interval_end as u8);
            let right = lower_bound_by_first(left, i as u8);
            intervals.push((left, right));
            count += right - left;
            last_interval_end = -1;
            matched_size = right;
        }
    }
    if last_interval_end != -1 {
        let left = lower_bound_by_first(matched_size, last_interval_end as u8);
        intervals.push((left, sorted.len() as i32));
        count += sorted.len() as i32 - left;
    }
    (intervals, count)
}

impl MaskGenerator<'_> {
    /// Scan the vocabulary, classifying each token. Returns whether the
    /// rejected partition was filled (selects which mask constructor to
    /// use). Port of `GetTokenMaskWithFirstCharacterCheck`.
    pub(super) fn token_mask_with_first_char_check(
        &mut self,
        first_char_mask: &[bool; 256],
        is_root_rule: bool,
    ) -> bool {
        let sorted = self.tokenizer_info.sorted_decoded_vocab().to_vec();
        let subtree = self.tokenizer_info.trie_subtree_nodes_range().to_vec();
        let (intervals, possible) = possible_token_intervals(&sorted, first_char_mask);
        if intervals.is_empty() {
            return true;
        }

        let mut fill_reject = (sorted.len() as i32 - possible) < USE_BITSET_THRESHOLD as i32;

        if intervals[0].0 != 0 && fill_reject {
            for i in 0..intervals[0].0 {
                self.rejected.push(i);
            }
        }

        let (speculative, spec_mask) = self.speculative_calculation();
        let is_tag_dispatch = self.is_tag_dispatch_rule();
        let definite_bitset: Option<Vec<bool>> = if is_tag_dispatch {
            self.tag_dispatch_second_slice
                .get(&self.init_rule_id)
                .cloned()
        } else {
            None
        };

        let mut prev_matched: i32 = 0;
        let mut last_rejected_range: i32 = 0;
        let mut prev_token: Option<Vec<u8>> = None;

        for interval_idx in 0..intervals.len() {
            let (lo, hi) = intervals[interval_idx];
            let mut i = lo;
            while i < hi {
                if i < last_rejected_range {
                    if fill_reject {
                        self.rejected.push(i);
                        if self.rejected.len() >= USE_BITSET_THRESHOLD {
                            fill_reject = false;
                        }
                    } else {
                        i = last_rejected_range - 1;
                    }
                    i += 1;
                    continue;
                }
                let token = &sorted[i as usize].1;

                // Speculative fast-accept path.
                if speculative
                    && self.try_speculative_accept(token, i, &spec_mask, definite_bitset.as_deref())
                {
                    prev_token = Some(token.clone());
                    i += 1;
                    continue;
                }

                let advanced = self.scan_one_token(token, prev_token.as_deref(), &mut prev_matched);
                prev_token = Some(token.clone());

                let can_reach_end = *self.can_reach_end_prefix_or_stack.last().unwrap();
                if advanced {
                    self.accepted.push(i);
                } else if can_reach_end && prev_matched > 0 {
                    let (la_ok, la_done) = self.is_token_pass_lookahead(token);
                    let exact = super::rule_is_exact_lookahead(&self.grammar, self.init_rule_id);
                    if !is_root_rule && la_ok {
                        if la_done || !exact {
                            self.uncertain.push(i);
                        } else {
                            self.accepted.push(i);
                        }
                    } else {
                        let end = subtree[i as usize];
                        for j in i..end {
                            self.rejected.push(j);
                        }
                        i = end - 1;
                    }
                } else {
                    self.rejected.push(i);
                    last_rejected_range = subtree[i as usize];
                    if self.rejected.len() >= USE_BITSET_THRESHOLD {
                        fill_reject = false;
                    }
                }
                i += 1;
            }
            if interval_idx + 1 < intervals.len() && fill_reject {
                let next_lo = intervals[interval_idx + 1].0;
                for j in hi..next_lo {
                    self.rejected.push(j);
                }
                if self.rejected.len() >= USE_BITSET_THRESHOLD {
                    fill_reject = false;
                }
            }
        }

        self.pop(prev_matched as usize);

        let last_hi = intervals.last().unwrap().1;
        if last_hi != sorted.len() as i32 && fill_reject {
            for i in last_hi..sorted.len() as i32 {
                self.rejected.push(i);
            }
        }
        fill_reject
    }

    /// Feed one token byte-by-byte into the parser, reusing the longest
    /// common prefix with `prev_token` to avoid redundant work.
    /// Returns whether the whole token was accepted. Updates
    /// `prev_matched` to the count of bytes currently advanced.
    fn scan_one_token(
        &mut self,
        token: &[u8],
        prev_token: Option<&[u8]>,
        prev_matched: &mut i32,
    ) -> bool {
        let mut accepted = true;
        if let Some(prev) = prev_token {
            let lcp = token
                .iter()
                .zip(prev.iter())
                .take_while(|(a, b)| a == b)
                .count() as i32;
            if lcp > *prev_matched {
                accepted = false;
            } else if lcp < *prev_matched {
                let rollback = (*prev_matched - lcp) as usize;
                self.pop(rollback);
                let new_len = self.can_reach_end_stack.len() - rollback;
                self.can_reach_end_stack.truncate(new_len);
                self.can_reach_end_prefix_or_stack.truncate(new_len);
            }
            *prev_matched = (*prev_matched).min(lcp);
        }

        if accepted {
            let mut j = *prev_matched as usize;
            while j < token.len() {
                if !self.advance(token[j]) {
                    accepted = false;
                    break;
                }
                let completed = self.is_completed();
                self.can_reach_end_stack.push(completed);
                let or = completed || *self.can_reach_end_prefix_or_stack.last().unwrap();
                self.can_reach_end_prefix_or_stack.push(or);
                *prev_matched = j as i32 + 1;
                j += 1;
            }
        }
        accepted
    }
}
