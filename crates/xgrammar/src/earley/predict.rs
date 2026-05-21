// SPDX-License-Identifier: AGPL-3.0-only
//
// Predict — the Earley prediction operation.
// Port of `EarleyParser::Predict` and `ExpandNextRuleRefElement` from
// `cpp/earley_parser.cc`. The FSM-walk variant lives in `predict_fsm.rs`.

use super::parser::EarleyParser;
use super::state::{NO_PREV_INPUT_POS, ParserState};
use crate::grammar::GrammarExprType;

impl EarleyParser {
    /// Run prediction on `state`. Returns `(scanable, completable)`:
    /// `scanable` is true when the state can consume an input byte and
    /// must be added to the scanable history; `completable` is true
    /// when the state sits at the end of its rule and can complete.
    ///
    /// After grammar optimization every rule is FSM-backed, so a state
    /// with `rule_id != -1` always walks the per-rule FSM. The
    /// `rule_id == -1` branch handles the non-FSM body view kept for
    /// faithfulness with the C++ source.
    pub(crate) fn predict(&mut self, state: ParserState) -> (bool, bool) {
        if state.rule_id != -1 {
            self.expand_next_rule_ref_on_fsm(state);
            let fsm = self.grammar.per_rule_fsms[state.rule_id as usize]
                .as_ref()
                .expect("FSM-backed rule must have a per-rule FSM");
            let elem = state.element_id;
            return (
                super::fsm_view::is_scanable_state(fsm, elem),
                fsm.is_end_state(elem as usize),
            );
        }

        let grammar_expr = self.grammar.expr(state.sequence_id);
        debug_assert!(
            grammar_expr.kind == GrammarExprType::Sequence
                || grammar_expr.kind == GrammarExprType::EmptyStr
        );
        if state.element_id as usize == grammar_expr.len() {
            return (false, true);
        }
        let element_id = grammar_expr[state.element_id as usize];
        let element_expr = self.grammar.expr(element_id);
        match element_expr.kind {
            GrammarExprType::RuleRef => {
                self.expand_next_rule_ref(state, state.sequence_id, element_id);
                (false, false)
            }
            GrammarExprType::CharacterClassStar => {
                if state.sub_element_id == 0 {
                    self.queue.enqueue(ParserState::new(
                        state.rule_id,
                        state.sequence_id,
                        state.element_id + 1,
                        state.rule_start_pos,
                        0,
                    ));
                }
                (true, false)
            }
            GrammarExprType::Repeat => {
                let min_repeat = element_expr[1];
                self.expand_next_rule_ref(state, state.sequence_id, element_id);
                if state.repeat_count >= min_repeat {
                    self.queue.enqueue(ParserState::new(
                        state.rule_id,
                        state.sequence_id,
                        state.element_id + 1,
                        state.rule_start_pos,
                        0,
                    ));
                }
                (false, false)
            }
            GrammarExprType::ByteString | GrammarExprType::CharacterClass => (true, false),
            other => panic!("Predict: unsupported element type {other:?}"),
        }
    }

    /// True if `rule_id` is permitted to match the empty string.
    pub(crate) fn rule_allows_empty(&self, rule_id: i32) -> bool {
        self.grammar
            .allow_empty_rule_ids
            .binary_search(&rule_id)
            .is_ok()
    }

    /// Expand a `RuleRef`/`Repeat` element of a non-FSM sequence body.
    /// Port of `ExpandNextRuleRefElement`. Records the parent state in
    /// the completable table and enqueues the referenced rule's FSM
    /// start node. `seq_id` is the parent sequence expr; `element_id`
    /// the `RuleRef`/`Repeat` expr id.
    pub(crate) fn expand_next_rule_ref(
        &mut self,
        state: ParserState,
        seq_id: i32,
        element_id: i32,
    ) {
        let grammar_expr = self.grammar.expr(seq_id);
        let sub = self.grammar.expr(element_id);
        debug_assert_eq!(grammar_expr.kind, GrammarExprType::Sequence);
        debug_assert!(sub.kind == GrammarExprType::RuleRef || sub.kind == GrammarExprType::Repeat);
        let ref_rule_id = sub[0];
        let is_repeat = sub.kind == GrammarExprType::Repeat;
        let seq_len = grammar_expr.len();
        let cur_pos = self.completable.len() as i32 - 1;

        let mut right_recursion_to_root = false;
        if state.element_id as usize != seq_len - 1 || is_repeat || state.rule_start_pos == cur_pos
        {
            self.completable
                .last_mut()
                .unwrap()
                .push((ref_rule_id, state));
        } else if state.rule_start_pos == NO_PREV_INPUT_POS {
            right_recursion_to_root = true;
        } else {
            // Right recursion: copy the parent's ancestors forward.
            let parents = self.completable[state.rule_start_pos as usize].clone();
            let mut to_add = Vec::new();
            for (first, parent) in &parents {
                if *first != state.rule_id {
                    continue;
                }
                let already = self
                    .completable
                    .last()
                    .unwrap()
                    .iter()
                    .any(|(f, s)| *f == ref_rule_id && s == parent);
                if !already {
                    to_add.push((ref_rule_id, *parent));
                }
            }
            self.completable.last_mut().unwrap().extend(to_add);
        }

        if self.rule_allows_empty(ref_rule_id) {
            self.queue.enqueue(ParserState::new(
                state.rule_id,
                state.sequence_id,
                state.element_id + 1,
                state.rule_start_pos,
                0,
            ));
        }

        let ref_body_id = self.grammar.rule(ref_rule_id).body_expr_id;
        let ref_fsm = self.grammar.per_rule_fsms[ref_rule_id as usize]
            .as_ref()
            .expect("referenced rule must have a per-rule FSM");
        let start = ref_fsm.start() as i32;
        let new_pos = if right_recursion_to_root {
            NO_PREV_INPUT_POS
        } else {
            self.completable.len() as i32 - 1
        };
        self.queue.enqueue(ParserState::new(
            ref_rule_id,
            ref_body_id,
            start,
            new_pos,
            0,
        ));
    }
}
