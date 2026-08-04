// SPDX-License-Identifier: AGPL-3.0-only
//
// GrammarMatcher — grammar-legal completion to a stop-legal state.
//
// Powers budget-aware graceful close of structured outputs (Atlas #144):
// when a length-limited response would otherwise stop with the stop token
// forbidden mid-structure (e.g. inside an open JSON string), this finds a
// byte sequence that drives the grammar to a state where the root rule can
// complete — so a truncated `finish_reason="length"` response is still
// parseable.
//
// The search is a depth-first, CLOSURE-PREFERRING walk: at every choice
// point it tries structural closers (`"` `}` `]` `,` `:` …) before content
// bytes, so it commits to closing rather than exploring the exponential
// space of string/number content continuations. That is what lets it reach
// DEEP closes — e.g. an unemitted required field needing `,"tags":[]}` —
// which a breadth-first search would never reach before exhausting its
// budget on content branches.
//
// There is no C++ upstream for this; it is an Atlas addition built on the
// same byte-level Earley primitives as `find_jump_forward_string`
// (`advance` / `pop_last_states` / `is_completed`).

use super::matcher::GrammarMatcher;

/// Hard cap on parser byte-advances explored. A close sequence is short in
/// practice; if none is found under this budget the caller falls back to a
/// plain finish — strictly no worse than the prior behavior.
const COMPLETION_NODE_BUDGET: usize = 8192;

/// Priority-ordered "closing" bytes tried before any content byte: string /
/// container closers and JSON structural separators. Ordering the search to
/// take these first makes it commit to closing the current structure instead
/// of extending it.
const CLOSE_PRIORITY: &[u8] = b"\"}],: \n\t";

/// Max distinct CONTENT (non-priority) bytes explored per node. Forced
/// positions — e.g. each character of a required key — expose exactly one
/// acceptable byte, so a small fan-out preserves forced-run closes while
/// bounding branching at genuine content choice points.
const CONTENT_FANOUT: usize = 3;

impl GrammarMatcher {
    /// A grammar-legal byte sequence that advances the grammar from its
    /// CURRENT state to one where the root rule can complete (a stop token
    /// is legal), or `None` if none is found within `max_bytes` and the node
    /// budget. The matcher state is left UNCHANGED.
    ///
    /// Returns `Some(empty)` when the grammar can already stop here.
    ///
    /// Soundness: every byte of a returned path is applied via
    /// `parser.advance`, and a path is only returned once
    /// `parser.is_completed()` holds — so the result is always a grammar-legal
    /// completion. Closure-first ordering means it is not guaranteed to be the
    /// globally shortest close, but it is a valid one and is found without
    /// exploring content breadth.
    #[must_use]
    pub fn find_completion_to_accept(&mut self, max_bytes: usize) -> Option<Vec<u8>> {
        if self.is_terminated() {
            return None;
        }
        if self.parser.is_completed() {
            return Some(Vec::new());
        }
        let mut out = Vec::new();
        let mut budget = COMPLETION_NODE_BUDGET;
        if self.close_dfs(max_bytes, &mut out, &mut budget) {
            // `close_dfs` left the parser advanced through `out`; restore it so
            // the search is side-effect-free for the caller.
            if !out.is_empty() {
                self.parser.pop_last_states(out.len());
            }
            Some(out)
        } else {
            None
        }
    }

    /// Depth-first, closure-preferring search. Appends accepted bytes to
    /// `out`; returns `true` with the parser left advanced through `out` once
    /// the grammar can stop. On `false` it has restored the parser and left
    /// `out` unchanged.
    fn close_dfs(&mut self, remaining: usize, out: &mut Vec<u8>, budget: &mut usize) -> bool {
        if self.parser.is_completed() {
            return true;
        }
        if remaining == 0 || *budget == 0 {
            return false;
        }
        let mask = self.parser.acceptable_byte_mask();
        // 1) Structural closers first — commit to closing the open structure.
        for &b in CLOSE_PRIORITY {
            if mask[b as usize] && self.try_byte(b, remaining, out, budget) {
                return true;
            }
            if *budget == 0 {
                return false;
            }
        }
        // 2) Bounded content fall-back — covers forced required-field bytes
        //    (where exactly one content byte is acceptable).
        let mut content = 0;
        for b in 0u16..256 {
            let byte = b as u8;
            if !mask[b as usize] || CLOSE_PRIORITY.contains(&byte) {
                continue;
            }
            if content >= CONTENT_FANOUT || *budget == 0 {
                break;
            }
            content += 1;
            if self.try_byte(byte, remaining, out, budget) {
                return true;
            }
        }
        false
    }

    /// Advance by `byte` and recurse. Keeps the byte in `out` (parser left
    /// advanced) iff the recursion reaches a completion; otherwise rolls the
    /// byte back. Returns whether a completion was found through `byte`.
    fn try_byte(
        &mut self,
        byte: u8,
        remaining: usize,
        out: &mut Vec<u8>,
        budget: &mut usize,
    ) -> bool {
        *budget = budget.saturating_sub(1);
        if !self.parser.advance(byte) {
            return false;
        }
        out.push(byte);
        if self.close_dfs(remaining - 1, out, budget) {
            return true;
        }
        out.pop();
        self.parser.pop_last_states(1);
        false
    }

    /// Like [`Self::find_completion_to_accept`], but returns the close as
    /// content **token ids**, greedily encoded against this matcher's vocab
    /// (`sorted_decoded_vocab`, which excludes stop/special tokens — so a
    /// close never emits a control token). Returns `Some(empty)` when the
    /// grammar can already stop, and `None` when no bounded close exists or
    /// a close byte is not representable as a content token.
    #[must_use]
    pub fn find_completion_token_ids(&mut self, max_bytes: usize) -> Option<Vec<i32>> {
        let bytes = self.find_completion_to_accept(max_bytes)?;
        if bytes.is_empty() {
            return Some(Vec::new());
        }
        self.encode_bytes_greedy(&bytes)
    }

    /// Greedy longest-match encode of `bytes` into content token ids. The
    /// concatenated token bytes equal `bytes` by construction, so detokenizing
    /// the result reproduces the close exactly. `None` if any position has no
    /// covering token.
    fn encode_bytes_greedy(&self, bytes: &[u8]) -> Option<Vec<i32>> {
        let vocab = self.tokenizer_info().sorted_decoded_vocab();
        let mut out = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            let rem = &bytes[i..];
            let mut best: Option<(i32, usize)> = None;
            for (id, tok) in vocab {
                let len = tok.len();
                if len == 0 || len > rem.len() || !rem.starts_with(tok.as_slice()) {
                    continue;
                }
                if best.is_none_or(|(_, bl)| len > bl) {
                    best = Some((*id, len));
                }
            }
            let (id, len) = best?;
            out.push(id);
            i += len;
        }
        Some(out)
    }
}
