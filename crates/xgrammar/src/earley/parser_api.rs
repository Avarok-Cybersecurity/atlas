// SPDX-License-Identifier: AGPL-3.0-only
//
// EarleyParser — history, rollback, push/reset API.
// Split out of `parser.rs` to keep each file under the 250-line cap.
// Port of `PopLastStates`, `PushStateAndExpand`, `PushOneStateToCheck`,
// `Reset` and `ExpandAndEnqueueUnexpandedState` from
// `cpp/earley_parser.{h,cc}`.

use super::parser::EarleyParser;
use super::state::{NO_PREV_INPUT_POS, ParserState};

impl EarleyParser {
    /// Remove the last `count` input positions — the rollback used by
    /// the matcher to undo speculatively-consumed tokens. Advancing
    /// `count` bytes then `pop_last_states(count)` restores the exact
    /// prior state.
    ///
    /// Panics if `count` would empty the history (the first step is the
    /// initial state and must remain — matches the C++ `FATAL` check).
    pub fn pop_last_states(&mut self, count: usize) {
        self.stop_token_is_accepted = false;
        assert!(
            (count as i32) < self.completable.len(),
            "cannot pop {count} states: only {} recorded",
            self.completable.len()
        );
        // CSR rollback: dropping the last `count` rows is an O(count)
        // offset truncation — `indptr` shrinks and `data` is cut at the
        // new last offset. Restores the exact prior state.
        self.completable.pop_rows(count as i32);
        self.is_completed.truncate(self.is_completed.len() - count);
        self.scanable_history.pop_rows(count as i32);
    }

    /// Push `state` into the parser and run predict/complete on it,
    /// recording the resulting scanable states as a new input position.
    pub fn push_state_and_expand(&mut self, state: ParserState) {
        self.queue.clear_visited();
        self.accept_stop_token = false;
        self.to_be_added.clear();

        if !self.expand_and_enqueue_unexpanded(state) {
            self.queue.enqueue(state);
        }
        self.completable.push_row(&[]);
        while let Some(state) = self.queue.pop() {
            let (scanable, completable) = self.predict(state);
            if completable {
                self.complete(state);
            }
            if scanable {
                self.to_be_added.push(state);
            }
        }
        self.is_completed.push(self.accept_stop_token);
        // ZapFormat dead-state pruning — see `advance` / `prune.rs`.
        self.productivity.prune(&mut self.to_be_added);
        let added = std::mem::take(&mut self.to_be_added);
        self.scanable_history.push_row_owned(added);
    }

    /// Push one state as an extra history step that only checks whether
    /// it can accept a token — used by the matcher to probe a candidate
    /// without running prediction/completion. Pop it with
    /// `pop_last_states(1)`.
    pub fn push_one_state_to_check(&mut self, state: ParserState) {
        self.completable.push_row(&[]);
        self.is_completed
            .push(*self.is_completed.last().unwrap_or(&false));
        self.scanable_history.push_row(&[state]);
    }

    /// Reset the parser to a fresh parse of the grammar's root rule.
    pub fn reset(&mut self) {
        self.completable = Default::default();
        self.scanable_history = Default::default();
        self.is_completed.clear();
        self.stop_token_is_accepted = false;
        debug_assert!(self.queue.is_empty());
        let root = Self::root_initial_state(&self.grammar);
        self.push_state_and_expand(root);
    }

    /// Handle an unexpanded initial state: enqueue the rule-body FSM
    /// start node. Returns true when `state` was unexpanded.
    /// Port of `ExpandAndEnqueueUnexpandedState`.
    pub(crate) fn expand_and_enqueue_unexpanded(&mut self, state: ParserState) -> bool {
        if !state.is_unexpanded() {
            return false;
        }
        let body_id = self.grammar.rule(state.rule_id).body_expr_id;
        let fsm = self.grammar.per_rule_fsms[state.rule_id as usize]
            .as_ref()
            .expect("unexpanded rule must have a per-rule FSM");
        self.queue.enqueue(ParserState::new(
            state.rule_id,
            body_id,
            fsm.start() as i32,
            NO_PREV_INPUT_POS,
            0,
        ));
        true
    }
}
