// SPDX-License-Identifier: AGPL-3.0-only
//
// Coalescence — forced-token fast-path analysis.
//
// Implements the "Coalescence" optimization from dottxt.ai
// (<https://blog.dottxt.ai/coalescence.html>): in a constrained
// grammar many parser states admit *exactly one* legal continuation.
// After `{` in a JSON object the next character is forced; while
// spelling a literal object key only one byte sequence is legal. When
// the grammar admits exactly one token, the model sampling step is
// redundant — the token is fully determined by the grammar.
//
// This module is the pure, side-effect-free analysis half. It answers
// one question, authoritatively:
//
//     "Given the next-token accept bitmask the matcher just computed,
//      is the continuation FORCED — i.e. is there exactly one legal
//      token?"
//
// The matcher (`src/matcher/`) owns the stateful half: it computes the
// bitmask via the normal `compute_partitions` path, hands it here, and
// — when this reports a forced token — exposes it through
// `GrammarMatcher::forced_token` / `next_forced_tokens` so Atlas's
// scheduler can skip the model sample for those positions.
//
// CORRECTNESS
// -----------
// A token is reported "forced" ONLY when the *final* next-token
// bitmask — the exact same mask the normal sampling path applies to
// the logits — has exactly one bit set. That bitmask already folds in:
//
//   * the accepted-union across every live Earley state;
//   * the rejected-intersection;
//   * every uncertain token, resolved by trial advance;
//   * stop tokens (added iff the root rule can complete);
//   * special tokens (always cleared).
//
// So "exactly one bit set" is definitionally equivalent to "exactly
// one grammar-legal token". There is no separate, weaker heuristic
// that could disagree with the sampling path: the forced token is read
// off the very mask sampling would have used. If zero or two-or-more
// bits are set the continuation is NOT forced and the caller falls
// through to the normal mask path. This is the conservative contract
// the task demands — when in doubt, "not forced".
//
// Note the deliberate distinction from `find_jump_forward_string`,
// which detects a forced *byte* sequence. A single forced byte path is
// necessary but NOT sufficient for a single forced *token*: several
// tokens may share that byte prefix (tokenizer ambiguity — dottxt's
// "name" example), and conversely one forced token may not correspond
// to a fully forced byte path once its own bytes are consumed. Token
// forcedness is strictly the bitmask-cardinality question handled here.

use crate::matcher::BitmaskSlice;

/// The result of a forced-token analysis over a next-token bitmask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Forced {
    /// The grammar admits exactly one legal token — its id is carried.
    /// The caller may emit it directly and skip the model sample.
    Token(i32),
    /// The grammar admits zero legal tokens. The matcher is stuck
    /// (a dead state) — the caller must fall through; sampling will
    /// also find an all-reject mask.
    Dead,
    /// Two or more tokens are legal — the continuation is genuinely a
    /// choice. The caller must run the normal sample step.
    NotForced,
}

impl Forced {
    /// The forced token id, if (and only if) the continuation is a
    /// single determined token.
    #[must_use]
    pub fn token(self) -> Option<i32> {
        match self {
            Forced::Token(id) => Some(id),
            Forced::Dead | Forced::NotForced => None,
        }
    }
}

/// Number of bits packed per `i32` mask word.
const BITS_PER_WORD: usize = 32;

/// Analyze a fully-computed next-token bitmask for the forced-token
/// condition.
///
/// `mask` MUST be the final mask produced by the matcher's normal
/// partition path (accepted ∪, rejected ∩, uncertain resolved, stop
/// tokens folded, special tokens cleared) — i.e. exactly the mask the
/// sampling step would apply to the logits. Given that, this returns:
///
///   * [`Forced::Token`] when exactly one bit is set — the unique
///     legal token; the caller may emit it without sampling;
///   * [`Forced::Dead`] when no bit is set;
///   * [`Forced::NotForced`] when two or more bits are set.
///
/// The scan works word-by-word over the packed mask using a hardware
/// popcount, and stops as soon as a word pushes the running set-bit
/// count past one. A genuine choice point — the common case — is
/// therefore rejected within the first non-empty word; it never walks
/// the whole vocabulary bit by bit.
#[must_use]
pub fn analyze_bitmask(mask: &BitmaskSlice<'_>) -> Forced {
    let vocab = mask.vocab_size();
    let words = mask.words();
    if words.is_empty() || vocab == 0 {
        return Forced::Dead;
    }
    // The final word may carry padding bits beyond `vocab`; mask them
    // off so a stray set padding bit can never be mis-read as a token.
    let last = words.len() - 1;
    let tail = vocab % BITS_PER_WORD;
    let tail_keep: u32 = if tail == 0 {
        u32::MAX
    } else {
        (1u32 << tail) - 1
    };

    let mut found_word: Option<usize> = None;
    for (wi, &raw) in words.iter().enumerate() {
        let mut bits = raw as u32;
        if wi == last {
            bits &= tail_keep;
        }
        match bits.count_ones() {
            0 => {}
            1 => {
                if found_word.is_some() {
                    // A set bit in an earlier word plus one here — two
                    // distinct legal tokens.
                    return Forced::NotForced;
                }
                found_word = Some(wi);
            }
            // Two or more bits in a single word — a choice point.
            _ => return Forced::NotForced,
        }
    }
    match found_word {
        Some(wi) => {
            let bit = (words[wi] as u32 & if wi == last { tail_keep } else { u32::MAX })
                .trailing_zeros() as usize;
            Forced::Token((wi * BITS_PER_WORD + bit) as i32)
        }
        None => Forced::Dead,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matcher::{BitmaskSlice, bitmask_size};

    /// Build a `BitmaskSlice`-backed buffer with `set_ids` accepted.
    fn mask_with(vocab: usize, set_ids: &[usize]) -> Vec<i32> {
        let mut buf = vec![0i32; bitmask_size(vocab)];
        {
            let mut view = BitmaskSlice::new(&mut buf, vocab).expect("slice");
            view.clear();
            for &id in set_ids {
                view.set(id, true);
            }
        }
        buf
    }

    #[test]
    fn single_set_bit_is_forced() {
        let vocab = 100;
        let mut buf = mask_with(vocab, &[42]);
        let view = BitmaskSlice::new(&mut buf, vocab).expect("slice");
        assert_eq!(analyze_bitmask(&view), Forced::Token(42));
        assert_eq!(analyze_bitmask(&view).token(), Some(42));
    }

    #[test]
    fn two_set_bits_is_not_forced() {
        let vocab = 100;
        let mut buf = mask_with(vocab, &[7, 88]);
        let view = BitmaskSlice::new(&mut buf, vocab).expect("slice");
        assert_eq!(analyze_bitmask(&view), Forced::NotForced);
        assert_eq!(analyze_bitmask(&view).token(), None);
    }

    #[test]
    fn no_set_bit_is_dead() {
        let vocab = 100;
        let mut buf = mask_with(vocab, &[]);
        let view = BitmaskSlice::new(&mut buf, vocab).expect("slice");
        assert_eq!(analyze_bitmask(&view), Forced::Dead);
        assert_eq!(analyze_bitmask(&view).token(), None);
    }

    #[test]
    fn forced_bit_at_vocab_boundary() {
        // The single set bit being the last logical token id exercises
        // the final (partial) word of the packed mask.
        let vocab = 65; // 3 words, 1 used bit in word 2.
        let mut buf = mask_with(vocab, &[64]);
        let view = BitmaskSlice::new(&mut buf, vocab).expect("slice");
        assert_eq!(analyze_bitmask(&view), Forced::Token(64));
    }

    #[test]
    fn all_set_is_not_forced() {
        let vocab = 40;
        let mut buf = vec![0i32; bitmask_size(vocab)];
        {
            let mut view = BitmaskSlice::new(&mut buf, vocab).expect("slice");
            view.fill_all();
        }
        let view = BitmaskSlice::new(&mut buf, vocab).expect("slice");
        assert_eq!(analyze_bitmask(&view), Forced::NotForced);
    }
}
