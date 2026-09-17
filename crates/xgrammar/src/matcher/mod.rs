// SPDX-License-Identifier: AGPL-3.0-only
//
// Grammar matcher — port wave W6, the final algorithmic subsystem.
//
// Pure-Rust port of `cpp/grammar_matcher.cc` + `include/xgrammar/
// matcher.h`. The matcher is the per-request, per-decode-step state
// Atlas drives: it accepts sampled tokens, fills the next-token accept
// bitmask, supports rollback (for speculative decoding) and detects
// termination.
//
// Module map:
//   bitmask  — `TokenBitmask` / `BitmaskSlice`: the packed i32
//              bit-per-token mask, byte-compatible with the upstream
//              CUDA `apply_token_bitmask` kernel.
//   matcher  — `GrammarMatcher`: construction, accept token/string,
//              rollback, termination, reset.
//   fill     — `FillNextTokenBitmask` + `FindJumpForwardString`.
//   coalesce — Coalescence forced-token fast-path (Tier 3b):
//              `forced_token` / `next_forced_tokens`.
//   completion — `find_completion_to_accept`: shortest grammar-legal
//              close to a stop-legal state (Atlas budget-aware close).
//   batch    — `BatchGrammarMatcher`: parallel batched bitmask fill.
//
// PORT NOTES / SIMPLIFICATIONS vs C++
// -----------------------------------
//  * The C++ `Impl` inherits `EarleyParser`; the Rust port composes
//    one. The parser already exposes `advance` / `pop_last_states` /
//    `push_one_state_to_check` / `is_completed`.
//  * The C++ matcher reads `stop_token_is_accepted_` off the parser
//    base. That field is cleared on every `pop_last_states`, which is
//    wrong for a matcher rollback that does not reach the stop step.
//    The Rust matcher owns its own `stop_token_accepted` flag, undone
//    precisely by `rollback` via the zero-length history entry.
//  * The bitmask is a packed `i32` slice / owned `TokenBitmask` rather
//    than a `DLTensor` — the W7 API shim adapts it to Atlas's tensor.
//  * `BatchGrammarMatcher`'s parallel path requires the natural
//    matcher-to-slice mapping (see `batch.rs`); an arbitrary `indices`
//    permutation falls back to the (always-correct) sequential path.
//  * `max_rollback_tokens` is accepted but ignored — the C++ also
//    deprecated and ignores it; rollback is unbounded.
//  * `_DebugPrintInternalState` is not ported (debug-only, the C++
//    notes its representation is unstable).

mod batch;
mod bitmask;
mod coalesce;
mod completion;
mod fill;
// The `GrammarMatcher` struct lives in `matcher.rs`; the file split
// (struct vs. `fill.rs`) is required by the 250-line-per-file cap.
#[allow(clippy::module_inception)]
mod matcher;

pub use batch::BatchGrammarMatcher;
pub use bitmask::{BITS_PER_WORD, BitmaskSlice, TokenBitmask, bitmask_size};
pub use fill::FillError;
pub use matcher::GrammarMatcher;

#[cfg(test)]
mod tests;
