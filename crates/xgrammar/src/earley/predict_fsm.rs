// SPDX-License-Identifier: AGPL-3.0-only
//
// FSM-walk prediction — `ExpandNextRuleRefElementOnFSM`.
// Port of the FSM-accelerated prediction path from `cpp/earley_parser.cc`.
//
// At an FSM node the parser follows: epsilon edges (advance the node),
// rule-ref edges (predict the referenced rule), and repeat-ref edges
// (predict with repeat-count bookkeeping).

use super::parser::EarleyParser;
use super::state::{NO_PREV_INPUT_POS, ParserState};

/// What kind of non-epsilon expansion edge was encountered.
struct RefEdge {
    target: i32,
    ref_rule_id: i32,
    is_repeat: bool,
}

impl EarleyParser {
    /// Expand `state` along its FSM node's outgoing edges, enqueuing
    /// epsilon successors and predicting referenced rules.
    /// Port of `ExpandNextRuleRefElementOnFSM`.
    pub(crate) fn expand_next_rule_ref_on_fsm(&mut self, state: ParserState) {
        // Hold an `Arc` clone (refcount bump, no data copy) so the
        // borrowed edge slice stays valid across the `&mut self`
        // queue / `predict_ref_edge` mutations — no per-call `to_vec()`.
        let grammar = self.grammar.clone();
        let fsm = grammar.per_rule_fsms[state.rule_id as usize]
            .as_ref()
            .expect("FSM-backed rule must have a per-rule FSM");
        let edges = fsm.fsm().edges(state.element_id as usize);

        for edge in edges {
            if edge.is_epsilon() {
                self.queue.enqueue(ParserState::new(
                    state.rule_id,
                    state.sequence_id,
                    edge.target,
                    state.rule_start_pos,
                    0,
                ));
                continue;
            }

            let Some(ref_edge) = self.classify_ref_edge(state, edge) else {
                continue;
            };
            self.predict_ref_edge(state, ref_edge);
        }
    }

    /// Decode a non-epsilon edge into a [`RefEdge`]. For repeat edges it
    /// also enqueues the past-lower-bound exit transition and reports
    /// `None` once the upper bound is reached.
    fn classify_ref_edge(
        &mut self,
        state: ParserState,
        edge: &crate::fsm::FsmEdge,
    ) -> Option<RefEdge> {
        if edge.is_rule_ref() {
            Some(RefEdge {
                target: edge.target,
                ref_rule_id: edge.ref_rule_id(),
                is_repeat: false,
            })
        } else if edge.is_repeat_ref() {
            let info = self.grammar.complete_fsm.repeat_edge_info(edge.aux_index());
            if state.repeat_count >= info.lower {
                self.queue.enqueue(ParserState::new(
                    state.rule_id,
                    state.sequence_id,
                    edge.target,
                    state.rule_start_pos,
                    0,
                ));
            }
            if state.repeat_count >= info.upper {
                return None;
            }
            Some(RefEdge {
                target: edge.target,
                ref_rule_id: info.rule_id as i32,
                is_repeat: true,
            })
        } else {
            None
        }
    }

    /// Predict the rule referenced by `ref_edge`: record the parent
    /// state in the completable table and enqueue the referenced
    /// rule's FSM start node.
    fn predict_ref_edge(&mut self, state: ParserState, ref_edge: RefEdge) {
        let RefEdge {
            target,
            ref_rule_id,
            is_repeat,
        } = ref_edge;
        let cur_pos = self.completable.len() - 1;

        // Right-recursion detection: a non-repeat ref whose target is a
        // terminal end node lets the callee complete straight to the
        // caller's ancestors, skipping a stack frame.
        let target_is_terminal_end = {
            let fsm = self.grammar.per_rule_fsms[state.rule_id as usize]
                .as_ref()
                .unwrap();
            fsm.fsm().edges(target as usize).is_empty() && fsm.is_end_state(target as usize)
        };

        let mut right_recursion_to_root = false;
        if !is_repeat && target_is_terminal_end && state.rule_start_pos != cur_pos {
            if state.rule_start_pos == NO_PREV_INPUT_POS {
                right_recursion_to_root = true;
            } else {
                self.add_right_recursion_parents(state.rule_id, state.rule_start_pos, ref_rule_id);
            }
        } else if is_repeat {
            // Repeat ref: store the source node + preserve repeat_count.
            self.completable.push_in_latest_row((
                ref_rule_id,
                ParserState::with_repeat(
                    state.rule_id,
                    state.sequence_id,
                    state.element_id,
                    state.rule_start_pos,
                    0,
                    state.repeat_count,
                ),
            ));
        } else {
            // Plain rule ref: store the post-transition target node.
            self.completable.push_in_latest_row((
                ref_rule_id,
                ParserState::new(
                    state.rule_id,
                    state.sequence_id,
                    target,
                    state.rule_start_pos,
                    0,
                ),
            ));
        }

        if !is_repeat && self.rule_allows_empty(ref_rule_id) {
            self.queue.enqueue(ParserState::new(
                state.rule_id,
                state.sequence_id,
                target,
                state.rule_start_pos,
                0,
            ));
        }

        let ref_body_id = self.grammar.rule(ref_rule_id).body_expr_id;
        let start = self.grammar.per_rule_fsms[ref_rule_id as usize]
            .as_ref()
            .expect("referenced rule must have a per-rule FSM")
            .start() as i32;
        let new_pos = if right_recursion_to_root {
            NO_PREV_INPUT_POS
        } else {
            self.completable.len() - 1
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
