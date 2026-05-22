// SPDX-License-Identifier: AGPL-3.0-only
//
// Character-class scanning — `AdvanceCharacterClass` and
// `AdvanceCharacterClassStar` from `cpp/earley_parser.cc`.
//
// A `CharacterClass` payload is `[is_negative, lo0, hi0, lo1, hi1, …]`.
// Multi-byte UTF-8 characters are matched incrementally: the first byte
// sets `sub_element_id` to the count of remaining continuation bytes
// and seeds `partial_codepoint`; each continuation byte folds 6 bits in.

use super::parser::EarleyParser;
use super::state::ParserState;
use crate::grammar::{GrammarExpr, GrammarExprType};
use crate::support::encoding::handle_utf8_first_byte;

/// True if `codepoint` lies in one of the `[lo, hi]` ranges packed in
/// `data[1..]` (pairs). `data[0]` is the negative flag, skipped here.
fn codepoint_in_ranges(data: &[i32], codepoint: i32) -> bool {
    let mut i = 1;
    while i + 1 < data.len() {
        if codepoint >= data[i] && codepoint <= data[i + 1] {
            return true;
        }
        i += 2;
    }
    false
}

/// True if some range in `data` could still be matched by a codepoint
/// known to lie in `[min, max]` — used to prune partial UTF-8 matches.
fn range_overlaps(data: &[i32], min: i32, max: i32) -> bool {
    let mut i = 1;
    while i + 1 < data.len() {
        if max >= data[i] && min <= data[i + 1] {
            return true;
        }
        i += 2;
    }
    false
}

/// Codepoint window reachable from `partial` with `remaining` UTF-8
/// continuation bytes still to read.
fn partial_window(partial: i32, remaining: i32) -> (i32, i32) {
    let shift = 6 * remaining;
    let min = partial << shift;
    let max = min | ((1 << shift) - 1);
    (min, max)
}

impl EarleyParser {
    /// Advance a `CharacterClass` element by `ch`. On a completed
    /// (single- or multi-byte) match `element_id` advances; partial
    /// UTF-8 progress is recorded in `sub_element_id`/`partial_codepoint`.
    pub(crate) fn advance_character_class(&mut self, state: ParserState, ch: u8, element_id: i32) {
        let sub: GrammarExpr = self.grammar.expr(element_id);
        debug_assert_eq!(sub.kind, GrammarExprType::CharacterClass);
        // `sub.data` is a borrowed slice into the grammar; iterate it in
        // place — no per-byte `to_vec()` clone. `char_class_step` is
        // `&self` and returns an owned result, so its borrow of the
        // grammar ends before the `&mut self` enqueue below.
        let data = sub.data;
        let is_negative = data[0] != 0;

        if let Some((next, completed)) = self.char_class_step(state, ch, data, is_negative, true) {
            if completed {
                self.queue.enqueue(next);
            } else {
                self.to_be_added.push(next);
            }
        }
    }

    /// Advance a `CharacterClassStar` element by `ch`. Identical range
    /// logic to [`Self::advance_character_class`], but a completed
    /// match loops on the same element instead of advancing.
    pub(crate) fn advance_character_class_star(
        &mut self,
        state: ParserState,
        ch: u8,
        element_id: i32,
    ) {
        let sub: GrammarExpr = self.grammar.expr(element_id);
        debug_assert_eq!(sub.kind, GrammarExprType::CharacterClassStar);
        // Borrowed slice into the grammar — no per-byte `to_vec()` clone.
        let data = sub.data;
        let is_negative = data[0] != 0;

        if let Some((next, completed)) = self.char_class_step(state, ch, data, is_negative, false) {
            if completed {
                self.queue.enqueue(next);
            } else {
                self.to_be_added.push(next);
            }
        }
    }

    /// Shared per-byte step for both character-class element types.
    /// Returns `Some((next_state, completed))` when `ch` is accepted —
    /// `completed` distinguishes a finished character (enqueue) from
    /// partial UTF-8 progress (append directly). `advance_element`
    /// selects the `CharacterClass` (advance `element_id`) vs
    /// `CharacterClassStar` (stay) completion behavior.
    fn char_class_step(
        &self,
        state: ParserState,
        ch: u8,
        data: &[i32],
        is_negative: bool,
        advance_element: bool,
    ) -> Option<(ParserState, bool)> {
        // Mid-character: consuming a UTF-8 continuation byte.
        if state.sub_element_id > 0 {
            if ch & 0xC0 != 0x80 {
                return None;
            }
            let mut next = state;
            next.sub_element_id -= 1;
            next.partial_codepoint = (next.partial_codepoint << 6) | (ch & 0x3F) as i32;
            if next.sub_element_id == 0 {
                let in_range = codepoint_in_ranges(data, next.partial_codepoint);
                if in_range == is_negative {
                    return None;
                }
                next.partial_codepoint = 0;
                if advance_element {
                    next.element_id += 1;
                }
                return Some((next, true));
            }
            let (min, max) = partial_window(next.partial_codepoint, next.sub_element_id);
            if is_negative || range_overlaps(data, min, max) {
                return Some((next, false));
            }
            return None;
        }

        // Non-ASCII first byte: start a multi-byte character.
        if ch >= 0x80 {
            let (accepted, num_bytes, partial) = handle_utf8_first_byte(ch);
            if !accepted {
                return None;
            }
            debug_assert!(num_bytes > 1);
            let (min, max) = partial_window(partial, num_bytes - 1);
            if is_negative || range_overlaps(data, min, max) {
                let mut next = state;
                next.sub_element_id = num_bytes - 1;
                next.partial_codepoint = partial;
                return Some((next, false));
            }
            return None;
        }

        // ASCII byte: a complete single-byte character.
        let in_range = codepoint_in_ranges(data, ch as i32);
        if in_range != is_negative {
            let mut next = state;
            next.sub_element_id = 0;
            if advance_element {
                next.element_id += 1;
            }
            return Some((next, true));
        }
        None
    }
}
