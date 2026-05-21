// SPDX-License-Identifier: AGPL-3.0-only
//
// Lookahead-assertion check for the mask generator. Port of
// `GrammarMatcherForTokenMaskCache::IsTokenPassLookaheadAssertion`
// from `cpp/grammar_compiler.cc`.

use crate::earley::ParserState;

use super::{MaskGenerator, NO_PREV};

impl MaskGenerator<'_> {
    /// Check whether `token` can satisfy `init_rule_id`'s lookahead
    /// assertion, given the per-byte "can reach rule end" flags
    /// accumulated in `can_reach_end_stack`.
    ///
    /// Returns `(acceptable, can_reach_end)`:
    /// * `acceptable` — the token passes the assertion;
    /// * `can_reach_end` — the assertion itself completed.
    ///
    /// When the rule has no lookahead assertion both are `true`.
    pub(super) fn is_token_pass_lookahead(&mut self, token: &[u8]) -> (bool, bool) {
        let lookahead_id = super::rule_lookahead_id(&self.grammar, self.init_rule_id);
        if lookahead_id == -1 {
            return (true, true);
        }

        // Push the lookahead-assertion rule body as a probe state.
        let lookahead_state = ParserState::new(-1, lookahead_id, 0, NO_PREV, 0);
        self.parser.push_state_and_expand(lookahead_state);

        let token_len = token.len() as i32;
        if self.is_completed() {
            // Assertion already satisfied with the empty suffix.
            self.pop(1);
            return (true, true);
        }

        let stack = self.can_reach_end_stack.clone();
        // From every position that can reach the rule end, try to feed
        // the token suffix into the lookahead assertion.
        for i in (0..stack.len() as i32).rev() {
            if !stack[i as usize] {
                continue;
            }
            let mut last_accept_pos = i - 1;
            let mut completed_here = false;
            let mut pos = i;
            while pos < token_len {
                if !self.parser.advance(token[pos as usize]) {
                    break;
                }
                last_accept_pos = pos;
                if self.is_completed() {
                    // The whole assertion finished.
                    self.pop((pos - i + 2) as usize);
                    completed_here = true;
                    break;
                }
                pos += 1;
            }
            if completed_here {
                return (true, true);
            }
            if last_accept_pos == token_len - 1 {
                // Whole token consumed; assertion not finished.
                self.pop((last_accept_pos - i + 2) as usize);
                return (true, false);
            }
            // Not accepted from position `i` — roll back, try earlier.
            self.pop((last_accept_pos - i + 1) as usize);
        }

        self.pop(1);
        (false, false)
    }
}
