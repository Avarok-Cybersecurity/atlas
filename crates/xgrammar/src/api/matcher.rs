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
        if let Some(ovr) = override_stop_tokens {
            if ovr.is_empty() {
                return Err("override_stop_tokens must not be empty".to_string());
            }
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

    /// Rollback the matcher by `num_tokens` tokens. Port of the
    /// vendored `GrammarMatcher::rollback`.
    pub fn rollback(&mut self, num_tokens: i32) {
        self.inner.rollback(num_tokens);
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
