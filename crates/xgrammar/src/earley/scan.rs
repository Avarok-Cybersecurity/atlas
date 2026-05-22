// SPDX-License-Identifier: AGPL-3.0-only
//
// Scan — the Earley scan operation.
// Port of `EarleyParser::Scan`, `AdvanceFsm` and `AdvanceByteString`
// from `cpp/earley_parser.cc`. Character-class scanning lives in the
// sibling `scan_charclass.rs`.

use super::fsm_view::{is_non_terminal_state, is_scanable_state};
use super::parser::EarleyParser;
use super::state::ParserState;
use crate::grammar::{GrammarExpr, GrammarExprType};

impl EarleyParser {
    /// Scan input byte `ch` from `state`, producing successor states.
    /// FSM-backed states walk the FSM; non-FSM states dispatch on the
    /// current element's expression type.
    pub(crate) fn scan(&mut self, state: ParserState, ch: u8) {
        if state.rule_id == -1 {
            let cur_rule = self.grammar.expr(state.sequence_id);
            let element_id = cur_rule[state.element_id as usize];
            let element_expr = self.grammar.expr(element_id);
            match element_expr.kind {
                GrammarExprType::ByteString => self.advance_byte_string(state, ch, element_id),
                GrammarExprType::CharacterClass => {
                    self.advance_character_class(state, ch, element_id)
                }
                GrammarExprType::CharacterClassStar => {
                    self.advance_character_class_star(state, ch, element_id)
                }
                other => panic!("Scan: unsupported element type {other:?}"),
            }
        } else {
            self.advance_fsm(state, ch);
        }
    }

    /// Advance an FSM-backed state by `ch` along char-range edges.
    /// A successor that is purely scanable (no rule-ref/epsilon and not
    /// an end state) is added directly without re-processing; others go
    /// through the predict/complete queue.
    fn advance_fsm(&mut self, state: ParserState, ch: u8) {
        // Hold an `Arc` clone (refcount bump, no data copy) so the
        // borrowed edge slice stays valid across the `&mut self`
        // queue/`to_be_added` mutations — no per-scan `to_vec()` clone.
        let grammar = self.grammar.clone();
        let fsm = grammar.per_rule_fsms[state.rule_id as usize]
            .as_ref()
            .expect("FSM-backed rule must have a per-rule FSM");
        let edges = fsm.fsm().edges(state.element_id as usize);
        for edge in edges {
            if !edge.is_char_range() || (ch as i16) < edge.min || (ch as i16) > edge.max {
                continue;
            }
            let mut next = state;
            next.element_id = edge.target;
            let t = edge.target;
            let pure_scanable = !is_non_terminal_state(fsm, t)
                && !fsm.is_end_state(t as usize)
                && is_scanable_state(fsm, t);
            if pure_scanable {
                if self.queue.mark_visited(next) {
                    self.to_be_added.push(next);
                }
            } else {
                self.queue.enqueue(next);
            }
        }
    }

    /// Advance a `ByteString` element by one byte. The string cannot be
    /// skipped, so a partial match is appended directly; a completed
    /// string advances `element_id` and is queued for processing.
    /// Port of `AdvanceByteString`.
    fn advance_byte_string(&mut self, state: ParserState, ch: u8, element_id: i32) {
        let sub: GrammarExpr = self.grammar.expr(element_id);
        debug_assert_eq!(sub.kind, GrammarExprType::ByteString);
        debug_assert!(sub.len() > state.sub_element_id as usize);
        if sub[state.sub_element_id as usize] as u8 != ch {
            return;
        }
        let mut next = state;
        next.sub_element_id += 1;
        if next.sub_element_id as usize == sub.len() {
            next.element_id += 1;
            next.sub_element_id = 0;
            self.queue.enqueue(next);
        } else {
            self.to_be_added.push(next);
        }
    }
}
