// SPDX-License-Identifier: AGPL-3.0-only
//
// Complete — the Earley completion operation.
// Port of `EarleyParser::Complete` from `cpp/earley_parser.cc`.
//
// When a rule finishes, every parent state that predicted it (recorded
// in the completable table at the rule's start position) is advanced
// past the rule reference.

use super::parser::EarleyParser;
use super::state::{NO_PREV_INPUT_POS, ParserState};
use crate::grammar::GrammarExprType;

impl EarleyParser {
    /// Complete `state`: a rule has finished, so advance its parents.
    /// If `state` is a root rule, the root has completed and the stop
    /// token becomes acceptable.
    pub(crate) fn complete(&mut self, state: ParserState) {
        if state.rule_start_pos == NO_PREV_INPUT_POS {
            // A root rule reaching completion means the grammar matched.
            self.accept_stop_token = true;
            return;
        }

        // Copy the matching parents out of the CSR row into the reusable
        // scratch: the row borrows `self.completable`, which can't be
        // held across the `&mut self` `complete_*_parent` calls below.
        self.parent_scratch.clear();
        let row = self.completable.row(state.rule_start_pos);
        for &(ref_id, parent) in row {
            if ref_id == state.rule_id {
                self.parent_scratch.push((ref_id, parent));
            }
        }
        for i in 0..self.parent_scratch.len() {
            let (ref_id, parent) = self.parent_scratch[i];
            if parent.rule_id == -1 {
                self.complete_non_fsm_parent(parent);
            } else {
                self.complete_fsm_parent(parent, ref_id);
            }
        }
    }

    /// Advance a non-FSM parent (`rule_id == -1`) past its `RuleRef` or
    /// `Repeat` element.
    fn complete_non_fsm_parent(&mut self, parent: ParserState) {
        let parent_expr = self.grammar.expr(parent.sequence_id);
        let element_id = parent_expr[parent.element_id as usize];
        let element_expr = self.grammar.expr(element_id);
        match element_expr.kind {
            GrammarExprType::RuleRef => {
                self.queue.enqueue(ParserState::new(
                    parent.rule_id,
                    parent.sequence_id,
                    parent.element_id + 1,
                    parent.rule_start_pos,
                    0,
                ));
            }
            GrammarExprType::Repeat => {
                let min_repeat = element_expr[1];
                let max_repeat = element_expr[2];
                let new_count = parent.repeat_count + 1;
                if new_count >= min_repeat {
                    self.queue.enqueue(ParserState::new(
                        parent.rule_id,
                        parent.sequence_id,
                        parent.element_id + 1,
                        parent.rule_start_pos,
                        0,
                    ));
                }
                if new_count < max_repeat {
                    let mut next = parent;
                    next.repeat_count = new_count;
                    self.queue.enqueue(next);
                }
            }
            other => {
                panic!("Complete: non-FSM parent element must be RuleRef/Repeat, got {other:?}")
            }
        }
    }

    /// Advance an FSM-backed parent. If the parent sits on a repeat-ref
    /// edge for `ref_id`, apply the repeat-count bookkeeping; otherwise
    /// re-enqueue the parent so its FSM walk continues from `target`.
    fn complete_fsm_parent(&mut self, parent: ParserState, ref_id: i32) {
        // Hold an `Arc` clone (refcount bump, no data copy) so the
        // borrowed edge slice stays valid across the `&mut self`
        // `queue.enqueue` calls — no per-call `to_vec()` clone.
        let grammar = self.grammar.clone();
        let fsm = grammar.per_rule_fsms[parent.rule_id as usize]
            .as_ref()
            .expect("FSM-backed parent must have a per-rule FSM");
        let edges = fsm.fsm().edges(parent.element_id as usize);

        let mut handled_as_repeat = false;
        for edge in edges {
            if !edge.is_repeat_ref() {
                continue;
            }
            let info = self.grammar.complete_fsm.repeat_edge_info(edge.aux_index());
            if info.rule_id as i32 != ref_id {
                continue;
            }
            handled_as_repeat = true;
            let new_count = parent.repeat_count + 1;
            if new_count >= info.lower {
                self.queue.enqueue(ParserState::new(
                    parent.rule_id,
                    parent.sequence_id,
                    edge.target,
                    parent.rule_start_pos,
                    0,
                ));
            }
            if new_count < info.upper {
                self.queue.enqueue(ParserState::with_repeat(
                    parent.rule_id,
                    parent.sequence_id,
                    parent.element_id,
                    parent.rule_start_pos,
                    0,
                    new_count,
                ));
            }
            break;
        }
        if !handled_as_repeat {
            self.queue.enqueue(parent);
        }
    }
}
