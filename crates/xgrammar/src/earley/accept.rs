// SPDX-License-Identifier: AGPL-3.0-only
//
// Acceptable-next-byte computation for the Earley parser.
//
// The C++ matcher derives the set of bytes a parser can consume by
// inspecting the outgoing char-range edges / sequence elements of every
// latest scanable state. This module exposes that as a reusable query
// so the W6 `GrammarMatcher` can build its token bitmask on top.

use super::parser::EarleyParser;
use super::state::ParserState;
use crate::grammar::GrammarExprType;

impl EarleyParser {
    /// Append the byte ranges `state` can accept to `ranges` as
    /// inclusive `(min, max)` pairs. FSM-backed states contribute their
    /// char-range edges; non-FSM states contribute the current
    /// element's byte/character-class ranges.
    fn acceptable_ranges_of(&self, state: &ParserState, ranges: &mut Vec<(u8, u8)>) {
        if state.rule_id != -1 {
            let fsm = match self.grammar.per_rule_fsms[state.rule_id as usize].as_ref() {
                Some(f) => f,
                None => return,
            };
            for edge in fsm.fsm().edges(state.element_id as usize) {
                if edge.is_char_range() {
                    ranges.push((edge.min as u8, edge.max as u8));
                }
            }
            return;
        }
        let seq = self.grammar.expr(state.sequence_id);
        if state.element_id as usize >= seq.len() {
            return;
        }
        let element = self.grammar.expr(seq[state.element_id as usize]);
        match element.kind {
            GrammarExprType::ByteString if (state.sub_element_id as usize) < element.len() => {
                let b = element[state.sub_element_id as usize] as u8;
                ranges.push((b, b));
            }
            GrammarExprType::CharacterClass | GrammarExprType::CharacterClassStar => {
                // Only the ASCII portion of the ranges is a direct byte
                // range; multi-byte handling is deferred to `advance`.
                let mut i = 1;
                while i + 1 < element.len() {
                    let lo = element[i].clamp(0, 255) as u8;
                    let hi = element[i + 1].clamp(0, 255) as u8;
                    if element[i] <= 255 {
                        ranges.push((lo, hi));
                    }
                    i += 2;
                }
            }
            _ => {}
        }
    }

    /// All byte ranges acceptable as the next input, across every
    /// latest scanable state. Ranges may overlap and are unsorted.
    pub fn acceptable_byte_ranges(&self) -> Vec<(u8, u8)> {
        let mut ranges = Vec::new();
        for state in self.latest_scanable_states() {
            self.acceptable_ranges_of(state, &mut ranges);
        }
        ranges
    }

    /// A 256-entry bitmap: `out[b]` is true if byte `b` can be the next
    /// input. This is the exact acceptable-next-byte set.
    pub fn acceptable_byte_mask(&self) -> [bool; 256] {
        let mut mask = [false; 256];
        for (lo, hi) in self.acceptable_byte_ranges() {
            for b in lo..=hi {
                mask[b as usize] = true;
            }
        }
        mask
    }

    /// True if byte `ch` can be accepted as the next input without
    /// mutating the parser. Note: for negative character classes this
    /// is a conservative pre-check; `advance` is authoritative.
    pub fn can_accept(&self, ch: u8) -> bool {
        self.acceptable_byte_mask()[ch as usize]
    }
}
