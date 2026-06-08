// SPDX-License-Identifier: AGPL-3.0-only
//
// UTF-8 character-range FSM helpers — port of `CodepointToPackedUTF8`,
// `AddSameLengthCharacterRange` and `GrammarFSMBuilderImpl::AddCharacterRange`
// from `cpp/grammar_functor.cc`.
//
// A character class `[lo-hi]` over unicode codepoints is realized as a
// byte-level FSM: each codepoint expands to its UTF-8 byte sequence and
// a range of codepoints becomes a set of byte-range edges.

use crate::fsm::FsmWithStartEnd;

/// Largest 1-byte UTF-8 value.
pub const MAX_1B: u32 = 0x7F;
/// Smallest packed 2-byte UTF-8 value.
pub const MIN_2B: u32 = 0xC080;
/// Largest packed 2-byte UTF-8 value.
pub const MAX_2B: u32 = 0xDFBF;
/// Smallest packed 3-byte UTF-8 value.
pub const MIN_3B: u32 = 0xE08080;
/// Largest packed 3-byte UTF-8 value.
pub const MAX_3B: u32 = 0xEFBFBF;
/// Smallest packed 4-byte UTF-8 value.
pub const MIN_4B: u32 = 0xF0808080;
/// Largest packed 4-byte UTF-8 value.
pub const MAX_4B: u32 = 0xF7BFBFBF;

/// Pack a unicode codepoint into the big-endian UTF-8 representation
/// used by [`add_character_range`].
pub fn codepoint_to_packed_utf8(cp: u32) -> u32 {
    if cp <= 0x7F {
        cp
    } else if cp <= 0x7FF {
        let b0 = 0xC0 | ((cp >> 6) & 0x1F);
        let b1 = 0x80 | (cp & 0x3F);
        (b0 << 8) | b1
    } else if cp <= 0xFFFF {
        let b0 = 0xE0 | ((cp >> 12) & 0x0F);
        let b1 = 0x80 | ((cp >> 6) & 0x3F);
        let b2 = 0x80 | (cp & 0x3F);
        (b0 << 16) | (b1 << 8) | b2
    } else {
        let b0 = 0xF0 | ((cp >> 18) & 0x07);
        let b1 = 0x80 | ((cp >> 12) & 0x3F);
        let b2 = 0x80 | ((cp >> 6) & 0x3F);
        let b3 = 0x80 | (cp & 0x3F);
        (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
    }
}
#[path = "char_range_same_len.rs"]
mod char_range_same_len;
pub use char_range_same_len::add_same_length_range;

/// Add a `[min, max]` range of packed UTF-8 codepoints to `fsm`,
/// splitting the range at byte-length boundaries.
pub fn add_character_range(
    fsm: &mut FsmWithStartEnd,
    from: usize,
    to: usize,
    mut min: u32,
    mut max: u32,
) {
    assert!(min <= max, "invalid character range: min > max");
    // Clamp max/min to valid packed-UTF-8 values.
    if max > MAX_4B {
        max = MAX_4B;
    } else if max > MAX_3B {
        if max < MIN_4B {
            max = MAX_3B;
        }
    } else if max > MAX_2B {
        if max < MIN_3B {
            max = MAX_2B;
        }
    } else if max < MIN_2B && max > MAX_1B {
        max = MAX_1B;
    }
    if min > MAX_4B {
        min = MAX_4B;
    } else if min > MAX_3B {
        if min < MIN_4B {
            min = MIN_4B;
        }
    } else if min > MAX_2B {
        if min < MIN_3B {
            min = MIN_3B;
        }
    } else if min < MIN_2B && min > MAX_1B {
        min = MIN_2B;
    }

    if max <= MAX_1B {
        add_same_length_range(fsm, from, to, min, max);
        return;
    }
    if max <= MAX_2B {
        if min >= MIN_2B {
            add_same_length_range(fsm, from, to, min, max);
        } else {
            add_same_length_range(fsm, from, to, min, MAX_1B);
            add_same_length_range(fsm, from, to, MIN_2B, max);
        }
        return;
    }
    if max <= MAX_3B {
        if min >= MIN_3B {
            add_same_length_range(fsm, from, to, min, max);
        } else if min >= MIN_2B {
            add_same_length_range(fsm, from, to, min, MAX_2B);
            add_same_length_range(fsm, from, to, MIN_3B, max);
        } else {
            add_same_length_range(fsm, from, to, min, MAX_1B);
            add_same_length_range(fsm, from, to, MIN_2B, MAX_2B);
            add_same_length_range(fsm, from, to, MIN_3B, max);
        }
        return;
    }
    assert!(max <= MAX_4B);
    if min >= MIN_4B {
        add_same_length_range(fsm, from, to, min, max);
    } else if min >= MIN_3B {
        add_same_length_range(fsm, from, to, min, MAX_3B);
        add_same_length_range(fsm, from, to, MIN_4B, max);
    } else if min >= MIN_2B {
        add_same_length_range(fsm, from, to, min, MAX_2B);
        add_same_length_range(fsm, from, to, MIN_3B, MAX_3B);
        add_same_length_range(fsm, from, to, MIN_4B, max);
    } else {
        add_same_length_range(fsm, from, to, min, MAX_1B);
        add_same_length_range(fsm, from, to, MIN_2B, MAX_2B);
        add_same_length_range(fsm, from, to, MIN_3B, MAX_3B);
        add_same_length_range(fsm, from, to, MIN_4B, max);
    }
}

#[cfg(test)]
#[path = "char_range_tests.rs"]
mod tests;
