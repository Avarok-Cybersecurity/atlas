// SPDX-License-Identifier: AGPL-3.0-only
//
// GrammarMatcher — Coalescence forced-token fast-path (Tier 3b).
//
// The stateful half of the dottxt.ai "Coalescence" optimization
// (<https://blog.dottxt.ai/coalescence.html>). The pure analysis lives
// in `crate::compiler::coalesce`; this file wires it onto the matcher.
//
// IDEA. In a constrained grammar many states admit exactly one legal
// token — after `{` in a JSON object, while spelling a literal key,
// inside a fixed enum value. When the grammar forces the token the
// model sampling step is redundant: the token is determined. Atlas's
// scheduler can call `forced_token` / `next_forced_tokens`, emit the
// forced token(s) directly, and skip the GPU sample for those
// positions. dottxt reports up to 5x on nested structures.
//
// API (all ADDITIVE — the existing matcher API is untouched):
//   * `forced_token()`        — the single forced token, if any.
//   * `next_forced_tokens()`  — the maximal forced *chain* from here.
//   * `fill_next_token_bitmask` early-out is handled in `fill.rs`.
//
// CORRECTNESS
// -----------
// `forced_token` computes the SAME next-token bitmask the normal
// sampling path computes (`compute_partitions` + `set_token_bitmask`),
// then asks `analyze_bitmask` whether exactly one bit is set. A token
// is reported forced ONLY when it is the sole bit in that authoritative
// mask — so it is, by construction, the only grammar-legal token.
// Feeding it back through `accept_token` therefore lands the matcher in
// exactly the state the normal path would (the normal path could only
// have sampled that same token). When zero / two-or-more tokens are
// legal, `forced_token` returns `None` and the caller falls through.
//
// `forced_token` and `next_forced_tokens` leave the matcher state
// completely unchanged: the chain walk virtually accepts each forced
// token to look ahead, then rolls every virtual accept back before
// returning.

use crate::compiler::analyze_bitmask;

use super::bitmask::{BitmaskSlice, bitmask_size};
use super::matcher::GrammarMatcher;

impl GrammarMatcher {
    /// The single grammar-forced next token, if the current state
    /// admits exactly one legal token.
    ///
    /// Returns `Some(token_id)` when the constrained grammar leaves the
    /// caller no choice — the token is fully determined, so the model
    /// sampling step for this position can be skipped and `token_id`
    /// emitted directly (then fed back through [`Self::accept_token`]).
    /// Returns `None` when the continuation is a genuine choice (two or
    /// more legal tokens), the state is dead (zero legal tokens), or the
    /// matcher has already terminated.
    ///
    /// This does NOT mutate the matcher: it computes the next-token
    /// bitmask exactly as [`Self::fill_next_token_bitmask`] would and
    /// inspects its cardinality. The returned token is the unique set
    /// bit of that authoritative mask, so accepting it is equivalent to
    /// the normal sample-then-accept path.
    ///
    /// The packed-bitmask buffer is reused from the matcher's scratch
    /// (`FillScratch::coalesce_bitmask`), so a repeated `forced_token`
    /// call — e.g. inside the [`Self::next_forced_tokens`] chain walk —
    /// costs no per-call heap allocation.
    #[must_use]
    pub fn forced_token(&mut self) -> Option<i32> {
        if self.stop_token_accepted() {
            return None;
        }
        let vocab = self.tokenizer_info().vocab_size();
        let words = bitmask_size(vocab);
        // Take the reusable buffer out of scratch for the duration of
        // the fill (a `Default` left in its place), resize once, and
        // restore it before returning — the same `mem::take` discipline
        // `fill_next_token_bitmask` uses for its own scratch.
        let mut buf = std::mem::take(&mut self.scratch.coalesce_bitmask);
        if buf.len() < words {
            buf.resize(words, 0);
        }
        // `fill_next_token_bitmask` cannot fail here: not terminated
        // (checked above) and the buffer is exactly `bitmask_size`.
        let forced = match self.fill_next_token_bitmask(&mut buf, 0, false) {
            Ok(_) => {
                BitmaskSlice::new(&mut buf, vocab).and_then(|view| analyze_bitmask(&view).token())
            }
            Err(_) => None,
        };
        self.scratch.coalesce_bitmask = buf;
        forced
    }

    /// Inspect an *already-filled* next-token bitmask for the
    /// forced-token condition — the zero-extra-work entry point.
    ///
    /// A caller that has just run [`Self::fill_next_token_bitmask`] for
    /// the current decode step already holds the authoritative mask;
    /// it can pass that buffer here to learn whether the step is forced
    /// without a second partition computation. `bitmask` / `index` must
    /// be the same buffer and slice index handed to the fill call.
    ///
    /// Returns `Some(token_id)` iff the mask has exactly one bit set
    /// (the forced token), `None` otherwise. Behaves identically to
    /// [`Self::forced_token`] but reuses the caller's mask, so the
    /// per-token server loop pays the coalescence check for free:
    ///
    /// ```ignore
    /// let nontrivial = matcher.fill_next_token_bitmask(&mut buf, 0, false);
    /// if let Some(tok) = matcher.forced_from_bitmask(&buf, 0) {
    ///     // skip the model sample — emit `tok` directly
    /// }
    /// ```
    #[must_use]
    pub fn forced_from_bitmask(&self, bitmask: &mut [i32], index: usize) -> Option<i32> {
        let vocab = self.tokenizer_info().vocab_size();
        let words = bitmask_size(vocab);
        let start = index.checked_mul(words)?;
        let slice = bitmask.get_mut(start..start.checked_add(words)?)?;
        let view = BitmaskSlice::new(slice, vocab)?;
        analyze_bitmask(&view).token()
    }

    /// The maximal *chain* of grammar-forced tokens starting at the
    /// current state.
    ///
    /// Coalescence's forced-chain: after emitting a forced token the
    /// next state may itself be forced. This walks that chain and
    /// returns every forced token in order, so the caller can skip the
    /// model sample for the whole run at once. The returned vector is
    /// empty when the current state is not forced.
    ///
    /// The matcher state is left completely UNCHANGED: each forced
    /// token is virtually accepted to peek at the next state, and every
    /// such virtual accept is rolled back before returning. The caller
    /// is responsible for feeding the returned tokens back through
    /// [`Self::accept_token`] for real.
    ///
    /// The walk stops at the first non-forced state, at a dead state,
    /// or once a forced token terminates the matcher (a forced stop
    /// token is included as the final element — the matcher would
    /// terminate there, so there is nothing further to force).
    ///
    /// A `max` cap bounds the chain length; pass [`usize::MAX`] for an
    /// unbounded walk. A cap of `0` returns an empty vector.
    #[must_use]
    pub fn next_forced_tokens(&mut self, max: usize) -> Vec<i32> {
        let mut chain = Vec::new();
        if max == 0 || self.stop_token_accepted() {
            return chain;
        }
        // Each iteration: detect a forced token, record it, then
        // virtually accept it to advance to the next state. Every
        // virtual accept is undone by the matched `rollback` below, so
        // the matcher is byte-identical to its entry state on return.
        let mut virtual_steps = 0i32;
        while chain.len() < max {
            let Some(tok) = self.forced_token() else {
                break;
            };
            chain.push(tok);
            // Virtually accept to peek at the next state. `accept_token`
            // records one rollback step (a forced stop token records a
            // zero-length step and terminates — handled by `rollback`).
            if !self.accept_token(tok, false) {
                // The forced token failed to apply. This cannot happen
                // for a token read off the authoritative mask, but if
                // it ever did, drop it from the chain rather than
                // returning an unaccepted token to the caller.
                chain.pop();
                break;
            }
            virtual_steps += 1;
            if self.is_terminated() {
                // A forced stop token (or root completion under
                // `terminate_without_stop_token`): nothing can follow.
                break;
            }
        }
        // Undo every virtual accept — restore the entry state exactly.
        if virtual_steps > 0 {
            self.rollback(virtual_steps);
        }
        chain
    }

    /// Detect the forced chain and ACCEPT it in place — the efficient
    /// server primitive for Coalescence.
    ///
    /// Unlike [`Self::next_forced_tokens`], which peeks and rolls back
    /// (leaving the caller to re-accept every token), this advances the
    /// matcher *as it walks*: each forced token is detected and kept.
    /// The matcher's parser is therefore traversed once, not twice — no
    /// rollback, no caller re-accept. The returned vector lists the
    /// tokens accepted, in order, so the caller can append them to its
    /// output and skip the model sample for each.
    ///
    /// On return the matcher state has advanced past the whole forced
    /// run — exactly as if the caller had sampled and accepted those
    /// tokens one by one. The walk stops at the first non-forced state
    /// (the next position is a genuine choice and must be sampled), at
    /// a dead state, or once a forced token terminates the matcher.
    ///
    /// `max` caps the run length (`usize::MAX` for unbounded); `0`
    /// accepts nothing and returns an empty vector.
    pub fn accept_forced_chain(&mut self, max: usize) -> Vec<i32> {
        let mut accepted = Vec::new();
        if max == 0 || self.stop_token_accepted() {
            return accepted;
        }
        while accepted.len() < max {
            let Some(tok) = self.forced_token() else {
                break;
            };
            // `forced_token` read `tok` off the authoritative mask, so
            // the accept must succeed; guard defensively all the same.
            if !self.accept_token(tok, false) {
                break;
            }
            accepted.push(tok);
            if self.is_terminated() {
                break;
            }
        }
        accepted
    }
}
