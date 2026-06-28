// SPDX-License-Identifier: AGPL-3.0-only
//
// `GrammarMatcher` façade — W7 compatibility shim.
//
// Restores the vendored `xgrammar-rs` `GrammarMatcher` constructor and
// `accept_token` arity on top of the pure-Rust
// `crate::matcher::GrammarMatcher`.
//
// Signature differences bridged here:
//  * vendored `new(&CompiledGrammar, Option<&[i32]>, bool, i32)
//    -> Result<Self, String>` vs core `new(CompiledGrammar,
//    Option<Vec<i32>>, bool, i32) -> Self`.
//  * vendored `accept_token(i32) -> bool` vs core
//    `accept_token(i32, bool) -> bool` (extra debug flag).
//
// `fill_next_token_bitmask` here takes `&mut [i32]` directly (the
// pure-Rust port has no FFI `DLTensor` boundary). Atlas's
// `grammar/state.rs` was repointed to pass its bitmask buffer slice;
// see the W7 report for that one-file edit.

use crate::matcher::GrammarMatcher as CoreMatcher;

use super::CompiledGrammar;

/// Per-request grammar matcher: accepts tokens, fills the next-token
/// bitmask, supports rollback and termination.
///
/// Port of the vendored `xgrammar::GrammarMatcher`.
pub struct GrammarMatcher {
    inner: CoreMatcher,
}

impl GrammarMatcher {
    /// Construct the grammar matcher.
    ///
    /// Port of the vendored `GrammarMatcher::new`. `compiled_grammar`
    /// is taken by reference and cloned (cheap — `CompiledGrammar` is
    /// `Arc`-backed). `max_rollback_tokens` is accepted for ABI parity
    /// and ignored (rollback is unbounded, as in upstream).
    pub fn new(
        compiled_grammar: &CompiledGrammar,
        override_stop_tokens: Option<&[i32]>,
        terminate_without_stop_token: bool,
        max_rollback_tokens: i32,
    ) -> Result<Self, String> {
        if let Some(ovr) = override_stop_tokens
            && ovr.is_empty()
        {
            return Err("override_stop_tokens must not be empty".to_string());
        }
        Ok(Self {
            inner: CoreMatcher::new(
                compiled_grammar.clone(),
                override_stop_tokens.map(<[i32]>::to_vec),
                terminate_without_stop_token,
                max_rollback_tokens,
            ),
        })
    }

    /// Accept one token, advancing the matcher state. Returns whether
    /// the token was accepted by the grammar.
    ///
    /// Port of the vendored `GrammarMatcher::accept_token` (the
    /// one-argument form; debug printing is off).
    pub fn accept_token(&mut self, token_id: i32) -> bool {
        self.inner.accept_token(token_id, false)
    }

    /// Accept one token with optional debug printing. Port of the
    /// vendored `GrammarMatcher::accept_token_with_debug`.
    pub fn accept_token_with_debug(&mut self, token_id: i32, debug_print: bool) -> bool {
        self.inner.accept_token(token_id, debug_print)
    }

    /// Accept a string in one rollback step. Port of the vendored
    /// `GrammarMatcher::accept_string`.
    pub fn accept_string(&mut self, input: &str, debug_print: bool) -> bool {
        self.inner.accept_string(input, debug_print)
    }

    /// Fill the next-token bitmask into `bitmask` (`index` selects the
    /// matcher's slice when several share one buffer; pass `0` for a
    /// single matcher). Returns whether the mask is non-trivial and
    /// must be applied to the logits.
    ///
    /// Port of the vendored `GrammarMatcher::fill_next_token_bitmask`.
    /// The vendored crate took a `DLTensor`; the pure-Rust port has no
    /// FFI boundary and operates on the packed `i32` slice directly.
    ///
    /// # Panics
    ///
    /// If the matcher has terminated or `bitmask` is too small — these
    /// are caller bugs the vendored crate also panicked on.
    pub fn fill_next_token_bitmask(
        &mut self,
        bitmask: &mut [i32],
        index: i32,
        debug_print: bool,
    ) -> bool {
        self.inner
            .fill_next_token_bitmask(bitmask, index.max(0) as usize, debug_print)
            .expect("fill_next_token_bitmask: terminated matcher or undersized bitmask")
    }

    /// Find the jump-forward string for jump-forward decoding. Port of
    /// the vendored `GrammarMatcher::find_jump_forward_string`.
    pub fn find_jump_forward_string(&mut self) -> String {
        self.inner.find_jump_forward_string_lossy()
    }

    /// The single grammar-forced next token, if the current state
    /// admits exactly one legal token (the dottxt.ai "Coalescence"
    /// fast-path). When `Some`, the caller may skip the model sample
    /// for this position and emit the returned token directly.
    ///
    /// Additive — not present in the vendored crate. See
    /// `crate::matcher::GrammarMatcher::forced_token`.
    pub fn forced_token(&mut self) -> Option<i32> {
        self.inner.forced_token()
    }

    /// Inspect an already-filled next-token bitmask for the
    /// forced-token condition — the zero-extra-work coalescence entry
    /// point for a caller that just ran `fill_next_token_bitmask`.
    ///
    /// Additive. See `crate::matcher::GrammarMatcher::forced_from_bitmask`.
    pub fn forced_from_bitmask(&self, bitmask: &mut [i32], index: i32) -> Option<i32> {
        self.inner
            .forced_from_bitmask(bitmask, index.max(0) as usize)
    }

    /// The shortest grammar-legal close (as content token ids) that drives
    /// the grammar from its current state to one where a stop token is
    /// legal, or `None` if none exists within `max_bytes`. The matcher state
    /// is unchanged. Powers Atlas budget-aware graceful close (#144).
    ///
    /// Additive. See `crate::matcher::GrammarMatcher::find_completion_token_ids`.
    pub fn find_completion_token_ids(&mut self, max_bytes: usize) -> Option<Vec<i32>> {
        self.inner.find_completion_token_ids(max_bytes)
    }

    /// The maximal chain of grammar-forced tokens from the current
    /// state — Coalescence's forced-chain. The caller skips the model
    /// sample for every returned position. `max` caps the chain length
    /// (`usize::MAX` for unbounded). The matcher state is unchanged;
    /// the caller must feed the tokens back through `accept_token`.
    ///
    /// Additive. See `crate::matcher::GrammarMatcher::next_forced_tokens`.
    pub fn next_forced_tokens(&mut self, max: usize) -> Vec<i32> {
        self.inner.next_forced_tokens(max)
    }

    /// Detect the Coalescence forced chain and accept it in place — the
    /// efficient server primitive. Advances the matcher past the whole
    /// forced run (no peek / rollback / re-accept) and returns the
    /// tokens accepted, in order; the caller skips the model sample for
    /// each. `max` caps the run length (`usize::MAX` for unbounded).
    ///
    /// Additive. See `crate::matcher::GrammarMatcher::accept_forced_chain`.
    pub fn accept_forced_chain(&mut self, max: usize) -> Vec<i32> {
        self.inner.accept_forced_chain(max)
    }

    /// Rollback the matcher by `num_tokens` tokens. Port of the
    /// vendored `GrammarMatcher::rollback`.
    pub fn rollback(&mut self, num_tokens: i32) {
        self.inner.rollback(num_tokens);
    }

    /// Number of accepted-token history steps currently held by the matcher
    /// (== how many tokens `rollback` can undo). Authoritative count of actual
    /// matcher advances — use this instead of an `accept_token` bool tally when
    /// some accepted tokens are exempt (stop/EOS) and do NOT advance the matcher.
    pub fn num_history_steps(&self) -> usize {
        self.inner.num_history_steps()
    }

    /// Whether the matcher has terminated. Port of the vendored
    /// `GrammarMatcher::is_terminated`.
    pub fn is_terminated(&self) -> bool {
        self.inner.is_terminated()
    }

    /// Reset the matcher to its initial state. Port of the vendored
    /// `GrammarMatcher::reset`.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// The maximum number of rollback tokens (always `-1`, unbounded).
    /// Port of the vendored `GrammarMatcher::max_rollback_tokens`.
    pub fn max_rollback_tokens(&self) -> i32 {
        self.inner.max_rollback_tokens()
    }

    /// The stop token ids used by the matcher. Port of the vendored
    /// `GrammarMatcher::stop_token_ids`.
    pub fn stop_token_ids(&self) -> Box<[i32]> {
        self.inner.stop_token_ids().to_vec().into_boxed_slice()
    }
}
