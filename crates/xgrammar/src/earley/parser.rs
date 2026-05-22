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

use super::prune::ProductivityTable;
use super::queue::ProcessQueue;
use super::state::{NO_PREV_INPUT_POS, ParserState, UNEXPANDED_RULE_START_SEQUENCE_ID};
use crate::grammar::GrammarData;
use crate::support::Compact2DArray;

/// One `(referenced_rule_id, parent_state)` entry of the
/// completable-states table. The C++ stores
/// `Compact2DArray<pair<int32_t, ParserState>>`; the Rust port now uses
/// the ported [`Compact2DArray`] (CSR) layout, matching it exactly.
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
    /// CSR layout (one flat buffer + offsets) — contiguous, alloc-once,
    /// O(1) row views, and rollback is a cheap `pop_rows` truncation.
    pub(crate) completable: Compact2DArray<CompletableEntry>,

    /// `scanable_history[i]` holds the scanable states reached after
    /// consuming input `i-1`. The push/rollback unit of the parser.
    /// CSR layout, same rationale as `completable`.
    pub(crate) scanable_history: Compact2DArray<ParserState>,

    /// Scratch list of states to append to the scanable history in the
    /// advance currently in progress.
    pub(crate) to_be_added: Vec<ParserState>,

    /// Reusable scratch holding a snapshot of the latest scanable row
    /// while it is scanned. Avoids a per-`advance` heap allocation: the
    /// CSR row cannot be borrowed across the `&mut self` `scan` calls,
    /// so its (small, `Copy`) states are copied here once and reused.
    pub(crate) scan_scratch: Vec<ParserState>,

    /// Reusable scratch for the parents read out of a `completable` CSR
    /// row during `Complete` / right-recursion expansion. The row can't
    /// be borrowed across the `&mut self` queue/`completable` mutations,
    /// so the needed entries are copied here once per call and reused.
    pub(crate) parent_scratch: Vec<CompletableEntry>,

    /// The predict/complete processing queue (with visited set).
    pub(crate) queue: ProcessQueue,

    /// Set true within an advance round when the root rule completes.
    pub(crate) accept_stop_token: bool,

    /// True after a one-off probe state has been pushed (see
    /// `push_one_state_to_check`); cleared on the next rollback.
    pub(crate) stop_token_is_accepted: bool,

    /// Per-rule co-accessibility bitsets driving ZapFormat-style dead-
    /// state pruning (Tier 3a). Computed once from the grammar's FSM
    /// topology; consulted with O(1) bitset lookups after every advance.
    /// See `prune.rs`.
    pub(crate) productivity: ProductivityTable,
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
    pub fn new(grammar: Arc<GrammarData>, initial_state: ParserState, need_expand: bool) -> Self {
        assert!(
            grammar.optimized,
            "EarleyParser requires an optimized grammar (run GrammarOptimizer first)"
        );
        let init = if initial_state.is_invalid() {
            Self::root_initial_state(&grammar)
        } else {
            initial_state
        };

        let productivity = ProductivityTable::build(&grammar);
        let mut parser = Self {
            grammar,
            is_completed: Vec::new(),
            completable: Compact2DArray::new(),
            scanable_history: Compact2DArray::new(),
            to_be_added: Vec::new(),
            scan_scratch: Vec::new(),
            parent_scratch: Vec::new(),
            queue: ProcessQueue::new(),
            accept_stop_token: false,
            stop_token_is_accepted: false,
            productivity,
        };

        if need_expand {
            parser.push_state_and_expand(init);
        } else {
            parser.completable.push_row(&[]);
            parser.is_completed.push(false);
            parser.scanable_history.push_row(&[init]);
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
        if self.scanable_history.is_empty() {
            &[]
        } else {
            self.scanable_history.back()
        }
    }

    /// Number of input positions currently recorded (history depth).
    pub fn num_steps(&self) -> usize {
        self.scanable_history.len() as usize
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

        // Scan phase: every scanable state of the latest step. The CSR
        // row can't be held across the `&mut self` `scan` calls, so copy
        // its (small, `Copy`) states into the reusable scratch buffer.
        self.scan_scratch.clear();
        if !self.scanable_history.is_empty() {
            self.scan_scratch
                .extend_from_slice(self.scanable_history.back());
        }
        for i in 0..self.scan_scratch.len() {
            self.scan(self.scan_scratch[i], ch);
        }

        // Rejected: nothing was produced — leave state untouched.
        if self.queue.is_empty() && self.to_be_added.is_empty() {
            return false;
        }

        // Predict / Complete until the queue drains.
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
        // ZapFormat dead-state pruning: drop scanable states whose FSM
        // node can never reach its rule's end node before they enter the
        // history. Language-preserving — see `prune.rs`.
        self.productivity.prune(&mut self.to_be_added);
        let added = std::mem::take(&mut self.to_be_added);
        self.scanable_history.push_row_owned(added);
        true
    }
}
