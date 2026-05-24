// SPDX-License-Identifier: AGPL-3.0-only
//
// Adaptive-token-mask generator — port of
// `class GrammarMatcherForTokenMaskCache` from `cpp/grammar_compiler.cc`.
//
// For one parser state this walks the whole sorted vocabulary, feeding
// each token byte-by-byte into an `EarleyParser` seeded at that state,
// and classifies every token as accepted / rejected / uncertain.
//
// SIMPLIFICATION vs C++
// ---------------------
// The C++ class also consults a cross-grammar `RuleLevelCache` (the W3
// functor agent deferred `RuleLevelCache`). That cache is purely a
// speed optimization — `GetAdaptiveTokenMask` is fully correct without
// it, just recomputes per state. This port omits the rule-level cache;
// the lookahead-assertion handling and first-character speculative
// pruning (the *correctness*-relevant parts) are ported faithfully.

use std::sync::Arc;

use crate::earley::{EarleyParser, NO_PREV_INPUT_POS, ParserState};
use crate::grammar::{GrammarData, GrammarExprType};
use crate::tokenizer::TokenizerInfo;

use super::mask::AdaptiveTokenMask;

mod intervals;
mod lookahead;
mod speculative;

#[cfg(test)]
pub(crate) use intervals::possible_token_intervals;

/// Per-state mask builder. Owns its own `EarleyParser` seeded at the
/// state being analyzed (`need_expand = false`, exactly as the C++
/// `GrammarMatcherForTokenMaskCache` constructs its base parser).
pub(crate) struct MaskGenerator<'a> {
    parser: EarleyParser,
    grammar: Arc<GrammarData>,
    tokenizer_info: &'a TokenizerInfo,
    init_rule_id: i32,
    init_state: ParserState,
    /// Per-tag-dispatch-rule "definitely accepted since 2nd char" bitset.
    tag_dispatch_second_slice: &'a ahash::AHashMap<i32, Vec<bool>>,
    // Scratch partitions.
    accepted: Vec<i32>,
    rejected: Vec<i32>,
    uncertain: Vec<i32>,
    can_reach_end_stack: Vec<bool>,
    can_reach_end_prefix_or_stack: Vec<bool>,
}

impl<'a> MaskGenerator<'a> {
    /// Construct a generator for `init_state`.
    pub(crate) fn new(
        grammar: Arc<GrammarData>,
        init_state: ParserState,
        tokenizer_info: &'a TokenizerInfo,
        tag_dispatch_second_slice: &'a ahash::AHashMap<i32, Vec<bool>>,
    ) -> Self {
        let parser = EarleyParser::new(Arc::clone(&grammar), init_state, false);
        Self {
            parser,
            grammar,
            tokenizer_info,
            init_rule_id: init_state.rule_id,
            init_state,
            tag_dispatch_second_slice,
            accepted: Vec::new(),
            rejected: Vec::new(),
            uncertain: Vec::new(),
            can_reach_end_stack: Vec::new(),
            can_reach_end_prefix_or_stack: Vec::new(),
        }
    }

    /// Compute the adaptive token mask for the seeded state.
    ///
    /// `is_root_rule` — when true the parent rules are not consulted,
    /// so there are no uncertain tokens (used for the grammar root).
    /// Port of `GrammarMatcherForTokenMaskCache::GetAdaptiveTokenMask`.
    pub(crate) fn get_adaptive_token_mask(&mut self, is_root_rule: bool) -> AdaptiveTokenMask {
        self.accepted.clear();
        self.rejected.clear();
        self.uncertain.clear();
        self.can_reach_end_stack.clear();
        self.can_reach_end_prefix_or_stack.clear();
        self.can_reach_end_stack.push(false);
        self.can_reach_end_prefix_or_stack.push(false);

        let first_char_mask = self.first_character_mask();
        let rejected_filled = self.token_mask_with_first_char_check(&first_char_mask, is_root_rule);

        let vocab_size = self.tokenizer_info.vocab_size();
        let sorted = self.tokenizer_info.sorted_decoded_vocab();
        if rejected_filled {
            AdaptiveTokenMask::from_accepted_rejected(
                vocab_size,
                sorted,
                &self.accepted,
                &self.rejected,
                &self.uncertain,
            )
        } else {
            AdaptiveTokenMask::from_accepted(vocab_size, sorted, &self.accepted, &self.uncertain)
        }
    }

    /// The set of bytes that can be the first byte of an accepted
    /// token. Port of `GetFirstCharacterMask`.
    fn first_character_mask(&self) -> [bool; 256] {
        let mut mask = [false; 256];
        let fsm = self.grammar.per_rule_fsms[self.init_rule_id as usize]
            .as_ref()
            .expect("rule must have a per-rule FSM");
        for edge in fsm.fsm().edges(self.init_state.element_id as usize) {
            if edge.is_char_range() {
                for c in edge.min..=edge.max {
                    mask[c as usize] = true;
                }
            }
        }
        mask
    }

    /// True if `init_rule_id`'s body is a `TagDispatch`.
    fn is_tag_dispatch_rule(&self) -> bool {
        let body = self.grammar.rule(self.init_rule_id).body_expr_id;
        self.grammar.expr(body).kind == GrammarExprType::TagDispatch
    }

    /// Advance the parser one byte; returns whether it was accepted.
    fn advance(&mut self, ch: u8) -> bool {
        self.parser.advance(ch)
    }

    /// Pop `count` advanced states (rollback).
    fn pop(&mut self, count: usize) {
        if count > 0 {
            self.parser.pop_last_states(count);
        }
    }

    /// Whether the current parser position completes the rule.
    fn is_completed(&self) -> bool {
        self.parser.is_completed()
    }
}

/// Whether `init_rule_id`'s rule has an exact lookahead assertion.
pub(crate) fn rule_is_exact_lookahead(grammar: &GrammarData, rule_id: i32) -> bool {
    grammar.rule(rule_id).is_exact_lookahead
}

/// The lookahead-assertion expr id of `rule_id`, or `-1`.
pub(crate) fn rule_lookahead_id(grammar: &GrammarData, rule_id: i32) -> i32 {
    grammar.rule(rule_id).lookahead_assertion_id
}

/// A no-prev-input root position constant alias.
pub(crate) const NO_PREV: i32 = NO_PREV_INPUT_POS;
