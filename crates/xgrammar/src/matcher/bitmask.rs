// SPDX-License-Identifier: AGPL-3.0-only
//
// TokenBitmask — the per-token accept/reject mask the matcher fills.
//
// Port of the C++ token-bitmask representation. The C++ side packs the
// mask into a `DLTensor` of `int32`, one bit per vocab token, with the
// buffer size given by `GetBitmaskSize(vocab_size)`. This module
// reproduces the same packed layout in safe Rust:
//   * `bitmask_size(vocab_size)` — number of `i32` words, identical to
//     the C++ `DynamicBitset::GetBufferSize`;
//   * [`TokenBitmask`] — an owned, vocab-sized packed mask;
//   * helpers to view / fill a caller-supplied `&mut [i32]` slice so
//     Atlas's W7 shim can hand the matcher whatever buffer it owns.
//
// Bit `t` of the mask is set iff token id `t` is grammar-legal for the
// next decode position. Word `t / 32`, bit `t % 32` (LSB-first), the
// exact layout the C++ `DynamicBitset` uses so a bitmask produced here
// is byte-compatible with the upstream CUDA `apply_token_bitmask`.

/// Bits packed per `i32` word — fixed by the C++ `DynamicBitset`.
pub const BITS_PER_WORD: usize = 32;

/// Number of `i32` words needed to hold a bit-per-token mask for a
/// vocabulary of `vocab_size` tokens.
///
/// Port of `GetBitmaskSize` / `DynamicBitset::GetBufferSize`:
/// `ceil(vocab_size / 32)`.
#[must_use]
pub fn bitmask_size(vocab_size: usize) -> usize {
    vocab_size.div_ceil(BITS_PER_WORD)
}

/// Set bit `token` in a packed `i32` mask slice.
#[inline]
fn set_bit(words: &mut [i32], token: usize, value: bool) {
    let word = token / BITS_PER_WORD;
    let bit = token % BITS_PER_WORD;
    let mask = 1u32 << bit;
    let cur = words[word] as u32;
    words[word] = if value { cur | mask } else { cur & !mask } as i32;
}

/// Read bit `token` from a packed `i32` mask slice.
#[inline]
#[must_use]
fn get_bit(words: &[i32], token: usize) -> bool {
    let word = token / BITS_PER_WORD;
    let bit = token % BITS_PER_WORD;
    (words[word] as u32 >> bit) & 1 == 1
}

/// Fill every bit `0..vocab_size` of `words` with `value`, leaving the
/// padding bits of the final word cleared (so `count`/`all` over the
/// logical vocabulary are exact).
fn fill(words: &mut [i32], vocab_size: usize, value: bool) {
    for w in words.iter_mut() {
        *w = if value { -1 } else { 0 };
    }
    if value {
        // Clear padding bits beyond `vocab_size` in the last word.
        let used = vocab_size % BITS_PER_WORD;
        if used != 0 {
            let last = bitmask_size(vocab_size) - 1;
            let keep = (1u32 << used) - 1;
            words[last] = (words[last] as u32 & keep) as i32;
        }
    }
}

/// A packed bit-per-token accept mask owned by a [`super::GrammarMatcher`].
///
/// This is the Rust equivalent of the C++ `DLTensor` int32 bitmask. It
/// is `vocab_size` bits wide, stored LSB-first in `i32` words. The
/// matcher fills it in place each decode step; callers read which
/// tokens are legal via [`Self::is_set`] or apply it to logits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenBitmask {
    words: Vec<i32>,
    vocab_size: usize,
}

impl TokenBitmask {
    /// Allocate an all-rejecting (all-zero) mask for `vocab_size` tokens.
    #[must_use]
    pub fn new(vocab_size: usize) -> Self {
        Self {
            words: vec![0; bitmask_size(vocab_size)],
            vocab_size,
        }
    }

    /// The vocabulary size this mask was sized for.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// The packed `i32` words — the layout Atlas hands to the kernel.
    #[must_use]
    pub fn as_words(&self) -> &[i32] {
        &self.words
    }

    /// Mutable access to the packed words (for the W7 shim / kernels).
    pub fn as_words_mut(&mut self) -> &mut [i32] {
        &mut self.words
    }

    /// True if token id `token` is accepted (its bit is set).
    ///
    /// Panics if `token >= vocab_size`.
    #[must_use]
    pub fn is_set(&self, token: usize) -> bool {
        assert!(token < self.vocab_size, "token id out of range");
        get_bit(&self.words, token)
    }

    /// Set / clear token id `token`. Panics if `token >= vocab_size`.
    pub fn set(&mut self, token: usize, value: bool) {
        assert!(token < self.vocab_size, "token id out of range");
        set_bit(&mut self.words, token, value);
    }

    /// Reject every token (clear all bits).
    pub fn clear(&mut self) {
        fill(&mut self.words, self.vocab_size, false);
    }

    /// Accept every token (set every logical bit).
    pub fn fill_all(&mut self) {
        fill(&mut self.words, self.vocab_size, true);
    }

    /// Number of accepted (set) tokens over the logical vocabulary.
    #[must_use]
    pub fn count_set(&self) -> usize {
        (0..self.vocab_size)
            .filter(|&t| get_bit(&self.words, t))
            .count()
    }

    /// True if every logical token bit is set.
    #[must_use]
    pub fn all_set(&self) -> bool {
        (0..self.vocab_size).all(|t| get_bit(&self.words, t))
    }

    /// Collect the rejected (cleared) token ids — the
    /// `_DebugGetMaskedTokensFromBitmask` helper of the C++ matcher.
    #[must_use]
    pub fn rejected_tokens(&self) -> Vec<i32> {
        (0..self.vocab_size)
            .filter(|&t| !get_bit(&self.words, t))
            .map(|t| t as i32)
            .collect()
    }
}

/// A mutable, vocab-sized view over a caller-owned `i32` slice.
///
/// `FillNextTokenBitmask` in the C++ takes a `DLTensor*`; Atlas owns
/// the backing buffer. This newtype lets the matcher fill that buffer
/// directly without copying through a [`TokenBitmask`]. The slice must
/// be at least `bitmask_size(vocab_size)` words long.
pub struct BitmaskSlice<'a> {
    words: &'a mut [i32],
    vocab_size: usize,
}

impl<'a> BitmaskSlice<'a> {
    /// Wrap `words` as a `vocab_size`-bit mask.
    ///
    /// Returns `None` if `words` is too short to hold the mask.
    #[must_use]
    pub fn new(words: &'a mut [i32], vocab_size: usize) -> Option<Self> {
        if words.len() < bitmask_size(vocab_size) {
            return None;
        }
        Some(Self { words, vocab_size })
    }

    /// The vocabulary size this view covers.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Set / clear token id `token`.
    pub fn set(&mut self, token: usize, value: bool) {
        debug_assert!(token < self.vocab_size);
        set_bit(self.words, token, value);
    }

    /// Read token id `token`.
    #[must_use]
    pub fn is_set(&self, token: usize) -> bool {
        debug_assert!(token < self.vocab_size);
        get_bit(self.words, token)
    }

    /// The packed `i32` words backing this view — exactly
    /// `bitmask_size(vocab_size)` of them.
    ///
    /// Padding bits beyond `vocab_size` in the final word are not
    /// guaranteed cleared (only [`Self::fill_all`] / [`Self::clear`]
    /// normalize them); callers that popcount must mask the tail or
    /// only trust [`Self::vocab_size`] logical bits.
    #[must_use]
    pub fn words(&self) -> &[i32] {
        self.words
    }

    /// Reject every token.
    pub fn clear(&mut self) {
        fill(self.words, self.vocab_size, false);
    }

    /// Accept every token.
    pub fn fill_all(&mut self) {
        fill(self.words, self.vocab_size, true);
    }

    /// True if every logical token bit is set.
    #[must_use]
    pub fn all_set(&self) -> bool {
        (0..self.vocab_size).all(|t| get_bit(self.words, t))
    }
}

impl<'a> From<&'a mut TokenBitmask> for BitmaskSlice<'a> {
    fn from(b: &'a mut TokenBitmask) -> Self {
        let vocab_size = b.vocab_size;
        Self {
            words: &mut b.words,
            vocab_size,
        }
    }
}
