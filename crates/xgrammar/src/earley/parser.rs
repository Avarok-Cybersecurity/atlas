// SPDX-License-Identifier: AGPL-3.0-only
//
// EarleyParser — the grammar-matching engine.
// Port of `class EarleyParser` from xgrammar `cpp/earley_parser.{h,cc}`.
//
// This file holds the parser struct, its working state, and the public
// API: construction, `advance`, `pop_last_states` (rollback),
// `is_completed`, `push_state_and_expand`, `reset`. The three Earley
// operations live in sibling files:
//   predict.rs  — Predict + rule-reference expansion
//   complete.rs — Complete
//   scan.rs     — Scan + the byte/char-class/FSM advance helpers

use std::sync::Arc;

use super::queue::ProcessQueue;
use super::state::{ParserState, NO_PREV_INPUT_POS, UNEXPANDED_RULE_START_SEQUENCE_ID};
use crate::grammar::GrammarData;

/// One `(referenced_rule_id, parent_state)` entry of the
/// completable-states table. The C++ stores
/// `Compact2DArray<pair<int32_t, ParserState>>`; a `Vec<Vec<_>>` is the
/// idiomatic equivalent and supports `truncate`-based rollback.
pub type CompletableEntry = (i32, ParserState);

/// The Earley parser. Drives grammar matching: it keeps the set of
/// Earley items, advances them byte-by-byte, and supports
/// push/rollback of parser state for the matcher's token rollback.
#[derive(Debug)]
pub struct EarleyParser {
    /// The optimized, FSM-accelerated grammar being parsed.
    pub(crate) grammar: Arc<GrammarData>,

    /// `is_completed[i]` records, after consuming `i` inputs, whether
    /// the root rule has completed (stop token acceptable).
    pub(crate) is_completed: Vec<bool>,

    /// `completable[i]` is the list of completable parent states
    /// recorded at input position `i`. Earley completion consults it.
    pub(crate) completable: Vec<Vec<CompletableEntry>>,

    /// `scanable_history[i]` holds the scanable states reached after
    /// consuming input `i-1`. The push/rollback unit of the parser.
    pub(crate) scanable_history: Vec<Vec<ParserState>>,

    /// Scratch list of states to append to the scanable history in the
    /// advance currently in progress.
    pub(crate) to_be_added: Vec<ParserState>,

    /// The predict/complete processing queue (with visited set).
    pub(crate) queue: ProcessQueue,

    /// Set true within an advance round when the root rule completes.
    pub(crate) accept_stop_token: bool,

    /// True after a one-off probe state has been pushed (see
    /// `push_one_state_to_check`); cleared on the next rollback.
    pub(crate) stop_token_is_accepted: bool,
}

impl EarleyParser {
    /// The default root initial state for `grammar`: the root rule,
    /// unexpanded, at the no-previous-input root position.
    pub(crate) fn root_initial_state(grammar: &GrammarData) -> ParserState {
        ParserState::new(
            grammar.root_rule_id(),
            UNEXPANDED_RULE_START_SEQUENCE_ID,
            0,
            NO_PREV_INPUT_POS,
            0,
        )
    }

    /// Construct a parser for `grammar`, seeding it with `initial_state`
    /// (or the root state when `initial_state` is the invalid
    /// sentinel). When `need_expand` is false the initial state is only
    /// placed in the scanable history — no predict/complete is run.
    ///
    /// Panics if the grammar is not optimized (matches the C++
    /// `XGRAMMAR_LOG(FATAL)` contract — the FSM-accelerated parser
    /// requires `per_rule_fsms`).
    pub fn new(
        grammar: Arc<GrammarData>,
        initial_state: ParserState,
        need_expand: bool,
    ) -> Self {
        assert!(
            grammar.optimized,
            "EarleyParser requires an optimized grammar (run GrammarOptimizer first)"
        );
        let init = if initial_state.is_invalid() {
            Self::root_initial_state(&grammar)
        } else {
            initial_state
        };

        let mut parser = Self {
            grammar,
            is_completed: Vec::new(),
            completable: Vec::new(),
            scanable_history: Vec::new(),
            to_be_added: Vec::new(),
            queue: ProcessQueue::new(),
            accept_stop_token: false,
            stop_token_is_accepted: false,
        };

        if need_expand {
            parser.push_state_and_expand(init);
        } else {
            parser.completable.push(Vec::new());
            parser.is_completed.push(false);
            parser.scanable_history.push(vec![init]);
        }
        parser
    }

    /// Build a parser seeded with the grammar's root rule.
    pub fn from_grammar(grammar: Arc<GrammarData>) -> Self {
        Self::new(grammar, ParserState::invalid(), true)
    }

    /// True if the root rule has completed at the current input
    /// position — i.e. the stop token may be emitted now.
    pub fn is_completed(&self) -> bool {
        *self.is_completed.last().unwrap_or(&false)
    }

    /// The scanable states reached at the current input position.
    pub fn latest_scanable_states(&self) -> &[ParserState] {
        self.scanable_history
            .last()
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Number of input positions currently recorded (history depth).
    pub fn num_steps(&self) -> usize {
        self.scanable_history.len()
    }

    /// Advance every scanable state by input byte `ch`.
    ///
    /// Returns true if `ch` is accepted by at least one state. When
    /// `ch` is rejected the parser state is left unchanged, so the
    /// caller may safely try another byte.
    pub fn advance(&mut self, ch: u8) -> bool {
        debug_assert!(self.queue.is_empty(), "queue must be empty before advance");
        self.queue.clear_visited();
        self.to_be_added.clear();
        self.accept_stop_token = false;

        // Scan phase: every scanable state of the latest step.
        let latest = self
            .scanable_history
            .last()
            .cloned()
            .unwrap_or_default();
        for state in &latest {
            self.scan(*state, ch);
        }

        // Rejected: nothing was produced — leave state untouched.
        if self.queue.is_empty() && self.to_be_added.is_empty() {
            return false;
        }

        // Predict / Complete until the queue drains.
        self.completable.push(Vec::new());
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
        let added = std::mem::take(&mut self.to_be_added);
        self.scanable_history.push(added);
        true
    }
}
