// SPDX-License-Identifier: AGPL-3.0-only
//
// Earley parser — port wave W4.
//
// Pure-Rust port of xgrammar's grammar-matching engine
// (`cpp/earley_parser.{h,cc}`). The parser maintains the set of Earley
// items, advances them byte-by-byte, expands rule references (using the
// FSM-accelerated per-rule machines), completes finished rules,
// computes the acceptable-next-byte set, supports push/rollback of
// parser state for the matcher's token rollback, and detects when the
// root rule has completed.
//
// No `unsafe`. The C++ index-based state pools and CSR layouts of the
// grammar/FSM are preserved and accessed safely via slices; the
// parser's transient history (`completable`, `scanable_history`) is a
// `support::Compact2DArray` (CSR: one flat data buffer + an offset
// vector), matching the C++ `Compact2DArray` — contiguous, alloc-once,
// with rollback as a cheap `pop_rows` truncation.
//
// Module map:
//   state           — ParserState, the Earley item
//   queue           — ProcessQueue: predict/complete FIFO + visited set
//   fsm_view        — IsScanableState / IsNonTerminalState predicates
//   parser          — EarleyParser struct + construction + advance loop
//   parser_api      — history / rollback / push / reset
//   predict         — Predict + non-FSM rule-ref expansion
//   predict_fsm     — ExpandNextRuleRefElementOnFSM
//   complete        — Complete
//   scan            — Scan + AdvanceFsm + AdvanceByteString
//   scan_charclass  — AdvanceCharacterClass(Star)
//   accept          — acceptable-next-byte computation

mod accept;
mod complete;
mod fsm_view;
mod parser;
mod parser_api;
mod predict;
mod predict_fsm;
mod prune;
mod queue;
mod scan;
mod scan_charclass;
mod state;

pub use parser::{CompletableEntry, EarleyParser};
pub use queue::ProcessQueue;
pub use state::{NO_PREV_INPUT_POS, ParserState, UNEXPANDED_RULE_START_SEQUENCE_ID, cache_key};

#[cfg(test)]
mod tests;
